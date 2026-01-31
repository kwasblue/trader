# ============================================================================
# 2. state_aggregator.py - State Management (Refactored)
# ============================================================================
from PySide6 import QtCore
import numpy as np
import pandas as pd
import time
import logging
from copy import deepcopy
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Deque, Any, Optional, List


class StateAggregator(QtCore.QObject):
    """
    Aggregates async event feed into coherent GUI snapshots.
    Computes rolling performance metrics and manages memory efficiently.
    """

    snapshot_ready = QtCore.Signal(object)

    # Memory limits
    PRICE_MAXLEN = 2000
    OHLC_MAXLEN = 1000
    EQUITY_MAXLEN = 5000
    METRICS_MAXLEN = 1000
    NEWS_MAXLEN = 200
    TRADES_MAXLEN = 1000
    ALERTS_MAX_CACHE = 50
    TRADES_MAX_CACHE = 50
    SYMBOL_STALE_SECS = 300
    SNAPSHOT_MS = 1000

    def __init__(self, feeder, interval_ms: int = None, window: int = 100):
        super().__init__()
        self.feeder = feeder
        self.window = window
        self.interval_ms = interval_ms or self.SNAPSHOT_MS
        self._logger = logging.getLogger("StateAggregator")

        # Latest values cache
        self.cache = self._init_cache()

        # Rolling arrays for performance metrics
        self._returns: Deque[float] = deque(maxlen=window)
        self._equity_hist: Deque[float] = deque(maxlen=window)

        # Time-series buffers for charts
        self.buffers = {
            "price": {},
            "ohlc": {},
            "equity": deque(maxlen=self.EQUITY_MAXLEN),
            "metrics": deque(maxlen=self.METRICS_MAXLEN),
            "news": deque(maxlen=self.NEWS_MAXLEN),
            "trades": deque(maxlen=self.TRADES_MAXLEN),
        }

        # Symbol activity tracking
        self._last_seen: Dict[str, float] = {}

        # Timers
        self._last_emit = time.time()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._emit_snapshot)
        self.timer.start(self.interval_ms)

        self._prune_timer = QtCore.QTimer()
        self._prune_timer.timeout.connect(self._prune_inactive_symbols)
        self._prune_timer.start(5000)

        # Wire signals
        self._connect_signals()

    def _init_cache(self) -> Dict:
        """Initialize empty cache structure"""
        return {
            "positions": {},
            "orders": {},
            "trades": [],
            "pnl": {
                "realized": 0.0,
                "unrealized": 0.0,
                "drawdown": 0.0,
                "portfolio_value": 0.0,
                "timestamp": None,
            },
            "alerts": [],
            "health": {},
            "regime": {},
            "metrics": {},
            "price": {},
        }

    def _connect_signals(self):
        """Wire all feeder signals to handlers"""
        s = self.feeder.s

        # Market data
        if hasattr(s, "ohlc"):
            s.ohlc.connect(self._on_ohlc)
        if hasattr(s, "regime_breakdown"):
            s.regime_breakdown.connect(self._on_regime)
        if hasattr(s, "news"):
            s.news.connect(self._on_news)

        # Execution
        if hasattr(s, "symbols"):
            s.symbols.connect(self._on_positions)
        if hasattr(s, "orders"):
            s.orders.connect(self._on_orders)
        if hasattr(s, "trades"):
            s.trades.connect(self._on_trades)
        if hasattr(s, "pnl_update"):
            s.pnl_update.connect(self._on_pnl)

        # System
        if hasattr(s, "health"):
            s.health.connect(self._on_health)
        if hasattr(s, "alerts"):
            s.alerts.connect(self._on_alerts)

    @staticmethod
    def _iso_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _mark_seen(self, sym: Optional[str]):
        """Track symbol activity for pruning"""
        if sym:
            self._last_seen[sym] = time.time()

    def _append_price_point(self, sym: str, ts: str, price: float,
                           ma20: Optional[float] = None, ma50: Optional[float] = None):
        """Add price tick to rolling buffer"""
        try:
            dq = self.buffers["price"].setdefault(sym, deque(maxlen=self.PRICE_MAXLEN))
            dq.append({
                "t": ts,
                "p": float(price),
                "ma20": float(ma20) if ma20 is not None else None,
                "ma50": float(ma50) if ma50 is not None else None
            })
            self._mark_seen(sym)
        except (ValueError, TypeError) as e:
            self._logger.warning(f"Invalid price data for {sym}: {e}")

    def _append_ohlc_point(self, sym: str, ts: str, o: float, h: float, 
                          l: float, c: float, v: float):
        """Add OHLC bar to rolling buffer"""
        try:
            dq = self.buffers["ohlc"].setdefault(sym, deque(maxlen=self.OHLC_MAXLEN))
            dq.append({
                "t": ts,
                "o": float(o),
                "h": float(h),
                "l": float(l),
                "c": float(c),
                "v": float(v)
            })
            self._mark_seen(sym)
        except (ValueError, TypeError) as e:
            self._logger.warning(f"Invalid OHLC data for {sym}: {e}")

    # ========================================================================
    # Event Handlers
    # ========================================================================

    def _on_positions(self, rows):
        """Update position cache"""
        try:
            if not isinstance(rows, list):
                rows = [rows]
            for r in rows:
                sym = r.get("symbol")
                if sym:
                    self.cache["positions"][sym] = r
                    self._mark_seen(sym)
        except Exception as e:
            self._logger.error(f"Position update error: {e}")

    def _on_orders(self, rows):
        """Update orders cache"""
        try:
            if not isinstance(rows, list):
                rows = [rows]
            for r in rows:
                oid = r.get("order_id") or r.get("id")
                if oid:
                    self.cache["orders"][oid] = r
        except Exception as e:
            self._logger.error(f"Order update error: {e}")

    def _on_trades(self, rows):
        """Update trades cache and buffer"""
        try:
            if not isinstance(rows, list):
                rows = [rows]
            
            # Update cache (small)
            self.cache["trades"].extend(rows)
            self.cache["trades"] = self.cache["trades"][-self.TRADES_MAX_CACHE:]
            
            # Update buffer (larger)
            for t in rows:
                payload = {
                    "t": t.get("timestamp") or self._iso_now(),
                    "symbol": t.get("symbol"),
                    "side": t.get("side"),
                    "qty": t.get("qty"),
                    "price": t.get("price"),
                    "pnl": t.get("pnl"),
                }
                self.buffers["trades"].append(payload)
                self._mark_seen(t.get("symbol"))
        except Exception as e:
            self._logger.error(f"Trade update error: {e}")

    def _on_pnl(self, data):
        """Handle PnL updates and compute returns"""
        try:
            if not isinstance(data, dict):
                return

            prev_val = float(self.cache["pnl"].get("portfolio_value") or 0.0)
            new_val = float(data.get("portfolio_value") or prev_val)
            
            # Update cache
            self.cache["pnl"].update(data)
            
            # Compute returns for metrics
            if prev_val > 0 and new_val > 0 and new_val != prev_val:
                ret = (new_val - prev_val) / max(prev_val, 1e-9)
                self._returns.append(ret)

            # Store equity history
            self._equity_hist.append(new_val)
            ts = data.get("timestamp") or self._iso_now()
            self.cache["pnl"]["timestamp"] = ts
            self.buffers["equity"].append({"t": ts, "v": new_val})
            
            # Update drawdown
            self._update_drawdown()
            
        except Exception as e:
            self._logger.error(f"PnL update error: {e}")

    def _on_alerts(self, alerts):
        """Update alerts cache"""
        try:
            if not isinstance(alerts, list):
                alerts = [alerts]
            self.cache["alerts"].extend(alerts)
            self.cache["alerts"] = self.cache["alerts"][-self.ALERTS_MAX_CACHE:]
        except Exception as e:
            self._logger.error(f"Alert update error: {e}")

    def _on_health(self, payload):
        """Update health status"""
        try:
            if isinstance(payload, dict):
                self.cache["health"] = payload
        except Exception as e:
            self._logger.error(f"Health update error: {e}")

    def _on_regime(self, payload):
        """Update regime classification"""
        try:
            if isinstance(payload, dict):
                self.cache["regime"].update(payload)
        except Exception as e:
            self._logger.error(f"Regime update error: {e}")

    def _on_ohlc(self, sym: str, df):
        """Handle OHLC DataFrame updates"""
        try:
            if not isinstance(df, pd.DataFrame) or df.empty:
                return

            # Normalize column names (case-insensitive)
            cols = {c.lower(): c for c in df.columns}
            required = ["open", "high", "low", "close"]
            
            if not all(k in cols for k in required):
                self._logger.warning(f"Missing OHLC columns for {sym}: {df.columns.tolist()}")
                return

            # Extract last bar
            last_row = df.iloc[-1]
            o = float(last_row[cols["open"]])
            h = float(last_row[cols["high"]])
            l = float(last_row[cols["low"]])
            c = float(last_row[cols["close"]])
            v = float(last_row[cols.get("volume", cols["close"])]) if "volume" in cols else 0.0
            
            # Get timestamp
            if hasattr(df.index, 'name') and df.index.name == 'timestamp':
                ts = str(df.index[-1])
            elif 'timestamp' in cols:
                ts = str(last_row[cols['timestamp']])
            else:
                ts = self._iso_now()

            # Update caches
            self.cache["price"][sym] = {"price": c, "timestamp": ts}
            self._append_ohlc_point(sym, ts, o, h, l, c, v)
            self._append_price_point(sym, ts, c)

        except Exception as e:
            self._logger.error(f"OHLC update error for {sym}: {e}")

    def _on_news(self, news_list):
        """Update news buffer"""
        try:
            if not isinstance(news_list, list):
                news_list = [news_list]

            for n in news_list:
                self.buffers["news"].append({
                    "t": n.get("timestamp") or self._iso_now(),
                    "headline": n.get("headline"),
                    "source": n.get("source"),
                    "sentiment": n.get("sentiment"),
                })
        except Exception as e:
            self._logger.error(f"News update error: {e}")

    # ========================================================================
    # Performance Metrics
    # ========================================================================

    def _update_drawdown(self):
        """Compute current drawdown from equity history"""
        try:
            if len(self._equity_hist) < 2:
                return
            
            peak = max(self._equity_hist)
            current = self._equity_hist[-1]
            dd = (current - peak) / peak if peak > 0 else 0.0
            self.cache["pnl"]["drawdown"] = float(dd)
            
        except Exception as e:
            self._logger.error(f"Drawdown calculation error: {e}")

    def _compute_metrics(self) -> Dict:
        """Compute rolling performance metrics"""
        try:
            if len(self._returns) < 5:
                return {}

            rets = np.array(self._returns, dtype=float)
            mean_ret = float(np.mean(rets))
            std_ret = float(np.std(rets, ddof=1))
            
            # Downside deviation
            downside_rets = rets[rets < 0]
            downside = float(np.std(downside_rets, ddof=1)) if len(downside_rets) > 0 else 0.0

            # Annualized ratios
            sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0.0
            sortino = mean_ret / downside * np.sqrt(252) if downside > 0 else 0.0
            kelly = mean_ret / (std_ret ** 2) if std_ret > 0 else 0.0

            # Risk metrics
            var_5 = float(np.percentile(rets, 5))
            cvar = float(rets[rets <= var_5].mean()) if np.any(rets <= var_5) else 0.0

            # Higher moments
            series = pd.Series(rets)
            skew = float(series.skew())
            kurt = float(series.kurt())

            # Max drawdown
            cum = np.cumsum(rets)
            roll_max = np.maximum.accumulate(cum)
            drawdowns = cum - roll_max
            max_dd = float(drawdowns.min())

            metrics = {
                "sharpe": round(float(sharpe), 3),
                "sortino": round(float(sortino), 3),
                "kelly": round(float(kelly), 3),
                "cvar": round(float(cvar), 4),
                "var_5": round(float(var_5), 4),
                "skew": round(float(skew), 4),
                "kurtosis": round(float(kurt), 4),
                "max_drawdown": round(float(max_dd), 4),
                "avg_return": round(float(mean_ret), 5),
                "volatility": round(float(std_ret), 5),
                "timestamp": self._iso_now(),
            }

            self.cache["metrics"].update(metrics)
            self.buffers["metrics"].append(metrics)
            return metrics

        except Exception as e:
            self._logger.error(f"Metrics computation error: {e}")
            return {}

    # ========================================================================
    # Snapshot Emission & Memory Management
    # ========================================================================

    def _serialize_buffers(self) -> Dict:
        """Convert deques to lists for JSON-friendly snapshots"""
        try:
            return {
                "price": {sym: list(dq) for sym, dq in self.buffers["price"].items()},
                "ohlc": {sym: list(dq) for sym, dq in self.buffers["ohlc"].items()},
                "equity": list(self.buffers["equity"]),
                "metrics": list(self.buffers["metrics"]),
                "news": list(self.buffers["news"]),
                "trades": list(self.buffers["trades"]),
            }
        except Exception as e:
            self._logger.error(f"Buffer serialization error: {e}")
            return {}

    def _emit_snapshot(self):
        """Emit unified state snapshot to GUI"""
        try:
            # Always emit snapshots - don't block on health status
            # Health will be updated separately via _check_heartbeat in DataFeeder

            # Compute metrics
            self._compute_metrics()

            # Deep copy to prevent mutation
            snap = deepcopy(self.cache)
            snap["buffers"] = self._serialize_buffers()
            snap["timestamp"] = self._iso_now()

            self.snapshot_ready.emit(snap)
            self._last_emit = time.time()

        except Exception as e:
            self._logger.error(f"Snapshot emission error: {e}")

    def _prune_inactive_symbols(self):
        """Remove stale symbol data to prevent memory bloat"""
        try:
            if not self._last_seen:
                return

            now = time.time()
            stale = [
                sym for sym, last in self._last_seen.items()
                if (now - last) > self.SYMBOL_STALE_SECS
            ]

            if stale:
                for sym in stale:
                    self.buffers["price"].pop(sym, None)
                    self.buffers["ohlc"].pop(sym, None)
                    self._last_seen.pop(sym, None)
                
                self._logger.info(f"Pruned {len(stale)} stale symbols")

        except Exception as e:
            self._logger.error(f"Symbol pruning error: {e}")