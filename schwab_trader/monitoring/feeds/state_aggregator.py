from PySide6 import QtCore
import numpy as np
import pandas as pd
import time
from copy import deepcopy
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Deque, Any, Optional


class StateAggregator(QtCore.QObject):
    """
    Aggregates async event feed into coherent GUI snapshots and
    computes rolling performance metrics (Sharpe, Sortino, Kelly, etc.).
    Emits periodic unified state snapshots to the GUI.

    In addition to the latest values in `cache`, this class maintains
    rolling time-series buffers in `self.buffers` so the GUI can plot
    directly without keeping its own arrays.
    """

    snapshot_ready = QtCore.Signal(object)  # emits {"positions","pnl","alerts","metrics","buffers",...}

    # ---------- Tuning knobs ----------
    PRICE_MAXLEN = 2000          # rolling ticks per symbol
    OHLC_MAXLEN = 1000           # rolling bars per symbol
    EQUITY_MAXLEN = 5000         # rolling equity points
    METRICS_MAXLEN = 1000        # rolling metrics points
    NEWS_MAXLEN = 200
    TRADES_MAXLEN = 1000
    ALERTS_MAX_CACHE = 50
    TRADES_MAX_CACHE = 50
    SYMBOL_STALE_SECS = 300      # auto-prune if no updates for 5 minutes
    SNAPSHOT_MS = 1000

    def __init__(self, feeder, interval_ms: int = SNAPSHOT_MS, window: int = 100):
        """
        Args:
            feeder: DataFeeder instance connected to EventBus
            interval_ms: snapshot emission interval (ms)
            window: rolling window for performance metrics (returns/equity dd)
        """
        super().__init__()
        self.feeder = feeder
        self.window = window

        # Latest values (single-frame cache for UI labels/tables)
        self.cache = self._init_cache()

        # Short rolling arrays for perf stats (not the same as GUI buffers)
        self._returns: Deque[float] = deque(maxlen=window)
        self._equity_hist: Deque[float] = deque(maxlen=window)

        # Rolling time-series buffers for charts / replay / analytics
        self.buffers = {
            "price": {},     # {sym: deque([{"t": ts, "p": price, "ma20":..., "ma50":...}], maxlen=PRICE_MAXLEN)}
            "ohlc": {},      # {sym: deque([{"t": ts, "o":..,"h":..,"l":..,"c":..,"v":..}], maxlen=OHLC_MAXLEN)}
            "equity": deque(maxlen=self.EQUITY_MAXLEN),   # [{"t": ts, "v": equity}]
            "metrics": deque(maxlen=self.METRICS_MAXLEN), # [{"t": ts, ... ratios ...}]
            "news": deque(maxlen=self.NEWS_MAXLEN),       # [{"t": ts, "headline":..., "sentiment":...}]
            "trades": deque(maxlen=self.TRADES_MAXLEN),   # [{"t": ts, "symbol":.., "side":.., "qty":.., "price":..}]
        }

        # Last-seen timestamps per symbol (for pruning)
        self._last_seen: Dict[str, float] = {}

        # periodic snapshot
        self._last_emit = time.time()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._emit_snapshot)
        self.timer.start(interval_ms)

        # periodic pruning of stale symbols
        self._prune_timer = QtCore.QTimer()
        self._prune_timer.timeout.connect(self._prune_inactive_symbols)
        self._prune_timer.start(5000)  # every 5s

        # wire feeder signals
        self._connect_signals()

    # ============================================================
    # Cache structure
    # ============================================================
    def _init_cache(self):
        return {
            "positions": {},      # {symbol: {...}}
            "orders": {},         # {order_id: {...}}
            "trades": [],         # last N trades (small cache for widgets)
            "pnl": {
                "realized": 0.0,
                "unrealized": 0.0,
                "drawdown": 0.0,
                "portfolio_value": 0.0,
                "timestamp": None,
            },
            "alerts": [],         # last N alerts (small cache)
            "health": {},         # {"broker":..., "status":..., "details":...}
            "regime": {},         # {"trend":..., "volatility":..., "market":...}
            "metrics": {},        # latest Sharpe, Sortino, Kelly, etc. (point)
            "price": {},          # latest per-symbol price tick (point)
            # Optional one-offs the GUI might render from latest snapshot:
            # "heatmap", "distribution", "trade_stats", "regime_perf",
            # "history", "benchmark", "replay", "session", "config_snapshot"
        }

    # ============================================================
    # Signal wiring
    # ============================================================
    def _connect_signals(self):
        """
        Connect all Feeder signals to their corresponding cache updaters.
        This ensures we listen to all relevant backend → GUI events.
        """
        s = self.feeder.s

        # === Market Data ===
        if hasattr(s, "ohlc"):             s.ohlc.connect(self._on_ohlc)
        if hasattr(s, "price_update"):     s.price_update.connect(self._on_price)
        if hasattr(s, "regime_breakdown"): s.regime_breakdown.connect(self._on_regime)
        if hasattr(s, "news"):             s.news.connect(self._on_news)

        # === Execution / Portfolio ===
        if hasattr(s, "symbols"):          s.symbols.connect(self._on_positions)
        if hasattr(s, "orders"):           s.orders.connect(self._on_orders)
        if hasattr(s, "trades"):           s.trades.connect(self._on_trades)
        if hasattr(s, "pnl_update"):       s.pnl_update.connect(self._on_pnl)

        # === System / Health / Alerts ===
        if hasattr(s, "health"):           s.health.connect(self._on_health)
        if hasattr(s, "alerts"):           s.alerts.connect(self._on_alerts)

        # === Performance / Analytics ===
        if hasattr(s, "performance_metrics"): s.performance_metrics.connect(self._on_performance)
        if hasattr(s, "heatmap"):             s.heatmap.connect(self._on_heatmap)
        if hasattr(s, "distribution"):        s.distribution.connect(self._on_distribution)
        if hasattr(s, "trade_stats"):         s.trade_stats.connect(self._on_trade_stats)
        if hasattr(s, "regime_perf"):         s.regime_perf.connect(self._on_regime_perf)

        # === History / Replay ===
        if hasattr(s, "history"):             s.history.connect(self._on_history)
        if hasattr(s, "benchmark"):           s.benchmark.connect(self._on_benchmark)
        if hasattr(s, "replay_frame"):        s.replay_frame.connect(self._on_replay_frame)

        # === Session / Config ===
        if hasattr(s, "session"):             s.session.connect(self._on_session)
        if hasattr(s, "config_snapshot"):     s.config_snapshot.connect(self._on_config_snapshot)

        # === Strategy ===
        if hasattr(s, "strategy_signals"):    s.strategy_signals.connect(self._on_strategy_signal)

    # ============================================================
    # Helpers
    # ============================================================
    @staticmethod
    def _iso_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _mark_seen(self, sym: Optional[str]):
        if not sym:
            return
        self._last_seen[sym] = time.time()

    def _append_price_point(self, sym: str, ts: str, price: float,
                            ma20: Optional[float] = None, ma50: Optional[float] = None):
        dq: Deque[dict] = self.buffers["price"].setdefault(sym, deque(maxlen=self.PRICE_MAXLEN))
        dq.append({"t": ts, "p": float(price), "ma20": ma20, "ma50": ma50})
        self._mark_seen(sym)

    def _append_ohlc_point(self, sym: str, ts: str, o: float, h: float, l: float, c: float, v: float):
        dq: Deque[dict] = self.buffers["ohlc"].setdefault(sym, deque(maxlen=self.OHLC_MAXLEN))
        dq.append({"t": ts, "o": float(o), "h": float(h), "l": float(l), "c": float(c), "v": float(v)})
        self._mark_seen(sym)

    # ============================================================
    # Event handlers (update cache + buffers)
    # ============================================================
    def _on_positions(self, rows):
        for r in rows:
            sym = r.get("symbol")
            if not sym:
                continue
            self.cache["positions"][sym] = r

    def _on_orders(self, rows):
        for r in rows:
            oid = r.get("order_id")
            if not oid:
                continue
            self.cache["orders"][oid] = r

    def _on_trades(self, rows):
        if not isinstance(rows, list):
            rows = [rows]
        # small cache for latest N
        self.cache["trades"].extend(rows)
        self.cache["trades"] = self.cache["trades"][-self.TRADES_MAX_CACHE:]
        # rolling buffer (larger)
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

    def _on_pnl(self, data):
        """Handle live PnL updates and record returns."""
        if not isinstance(data, dict):
            return

        prev_val = float(self.cache["pnl"].get("portfolio_value", 0.0) or 0.0)
        new_val = float(data.get("portfolio_value", prev_val or 0.0))
        self.cache["pnl"].update(data)

        # compute rolling returns for ratios
        if prev_val > 0 and new_val > 0 and new_val != prev_val:
            ret = (new_val - prev_val) / max(prev_val, 1e-9)
            self._returns.append(ret)

        # store equity history for drawdown
        self._equity_hist.append(new_val)
        ts = data.get("timestamp") or self._iso_now()
        self.cache["pnl"]["timestamp"] = ts
        self.buffers["equity"].append({"t": ts, "v": new_val})
        self._update_drawdown()

    def _on_alerts(self, alerts):
        if not isinstance(alerts, list):
            alerts = [alerts]
        self.cache["alerts"].extend(alerts)
        self.cache["alerts"] = self.cache["alerts"][-self.ALERTS_MAX_CACHE:]

    def _on_health(self, payload):
        self.cache["health"] = payload

    def _on_regime(self, payload):
        if isinstance(payload, dict):
            self.cache["regime"].update(payload)

    def _on_ohlc(self, sym: str, df):
        """
        Handle OHLC dataframe updates.
        Expects a DataFrame with columns like ['Open','High','Low','Close','Volume'] (case-insensitive).
        """
        try:
            if not isinstance(df, pd.DataFrame) or df.empty:
                return

            cols = {c.lower(): c for c in df.columns}
            need = all(k in cols for k in ("open", "high", "low", "close"))
            if not need:
                return

            # last bar
            o = float(df[cols["open"]].iloc[-1])
            h = float(df[cols["high"]].iloc[-1])
            l = float(df[cols["low"]].iloc[-1])
            c = float(df[cols["close"]].iloc[-1])
            v = float(df[cols.get("volume", cols["close"])].iloc[-1]) if "volume" in cols else 0.0
            ts = str(df.index[-1]) if not df.index.empty else self._iso_now()

            # latest price point for cache
            self.cache["price"][sym] = {"price": c, "timestamp": ts}

            # rolling ohlc buffer (full bar)
            self._append_ohlc_point(sym, ts, o, h, l, c, v)

            # also drop a tick for a smooth line chart if GUI wants
            self._append_price_point(sym, ts, c)

        except Exception:
            # be silent; bad DF shouldn't break the aggregator
            pass

    def _on_price(self, payload):
        """Tick-level price update."""
        sym = payload.get("symbol")
        if not sym:
            return

        ts = payload.get("timestamp") or self._iso_now()
        price = float(payload.get("price", np.nan))
        ma20 = payload.get("ma20")
        ma50 = payload.get("ma50")

        # latest tick in cache
        self.cache.setdefault("price", {})[sym] = {
            "price": price,
            "ma20": ma20,
            "ma50": ma50,
            "timestamp": ts,
        }

        # rolling buffer
        self._append_price_point(sym, ts, price, ma20, ma50)

    def _on_news(self, news_list):
        """Handle news updates with sentiment."""
        if not isinstance(news_list, list):
            news_list = [news_list]

        self.cache.setdefault("news", [])
        self.cache["news"].extend(news_list)
        self.cache["news"] = self.cache["news"][-50:]

        for n in news_list:
            self.buffers["news"].append({
                "t": n.get("timestamp") or self._iso_now(),
                "headline": n.get("headline"),
                "source": n.get("source"),
                "sentiment": n.get("sentiment"),
            })

    # === Performance / Analytics ===
    def _on_performance(self, payload):
        # latest point
        self.cache["metrics"].update(payload)
        # rolling
        m = dict(payload)
        m["t"] = payload.get("timestamp") or self._iso_now()
        self.buffers["metrics"].append(m)

    def _on_heatmap(self, payload):        self.cache["heatmap"] = payload
    def _on_distribution(self, payload):   self.cache["distribution"] = payload
    def _on_trade_stats(self, payload):    self.cache["trade_stats"] = payload
    def _on_regime_perf(self, payload):    self.cache["regime_perf"] = payload

    # === History / Replay ===
    def _on_history(self, payload):        self.cache["history"] = payload
    def _on_benchmark(self, payload):      self.cache["benchmark"] = payload
    def _on_replay_frame(self, payload):   self.cache["replay"] = payload

    # === Session / Config ===
    def _on_session(self, payload):        self.cache["session"] = payload
    def _on_config_snapshot(self, payload):self.cache["config_snapshot"] = payload

    # === Strategy ===
    def _on_strategy_signal(self, signals):
        if not isinstance(signals, list):
            signals = [signals]
        self.cache.setdefault("strategy_signals", [])
        self.cache["strategy_signals"].extend(signals)
        self.cache["strategy_signals"] = self.cache["strategy_signals"][-50:]

    # ============================================================
    # Performance metrics
    # ============================================================
    def _update_drawdown(self):
        """Compute current drawdown from rolling equity."""
        if len(self._equity_hist) < 2:
            return
        peak = max(self._equity_hist)
        eq = self._equity_hist[-1]
        dd = (eq - peak) / peak if peak else 0.0
        self.cache["pnl"]["drawdown"] = float(dd)

    def _compute_metrics(self):
        """Compute rolling Sharpe, Sortino, Kelly, and risk stats."""
        if len(self._returns) < 5:
            return {}

        rets = np.array(self._returns, dtype=float)
        mean_ret = float(np.mean(rets))
        std_ret = float(np.std(rets, ddof=1))
        downside = float(np.std(rets[rets < 0], ddof=1)) if np.any(rets < 0) else 0.0

        # Annualized ratios
        sharpe  = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0.0
        sortino = mean_ret / downside * np.sqrt(252) if downside > 0 else 0.0
        kelly   = mean_ret / (std_ret ** 2) if std_ret > 0 else 0.0

        # Conditional VaR (CVaR)
        var = float(np.percentile(rets, 5))
        cvar = float(rets[rets <= var].mean()) if np.any(rets <= var) else 0.0

        # Higher moments
        series = pd.Series(rets)
        skew = float(series.skew())
        kurt = float(series.kurt())

        # Max drawdown on cumulative returns
        cum = np.cumsum(rets)
        roll_max = np.maximum.accumulate(cum)
        drawdowns = cum - roll_max
        max_dd = float(drawdowns.min())

        metrics = {
            "sharpe": round(float(sharpe), 3),
            "sortino": round(float(sortino), 3),
            "kelly": round(float(kelly), 3),
            "cvar": round(float(cvar), 4),
            "skew": round(float(skew), 4),
            "kurtosis": round(float(kurt), 4),
            "max_drawdown": round(float(max_dd), 4),
            "avg_return": round(float(mean_ret), 5),
            "volatility": round(float(std_ret), 5),
            "timestamp": self._iso_now(),
        }

        self.cache["metrics"].update(metrics)
        # keep rolling series of metric points too (handy for a “metrics over time” plot)
        self.buffers["metrics"].append(metrics)
        return metrics

    # ============================================================
    # Snapshot emission + pruning
    # ============================================================
    def _serialize_buffers(self) -> dict:
        """Convert deques to lists for JSON-friendly snapshots."""
        return {
            "price": {sym: list(dq) for sym, dq in self.buffers["price"].items()},
            "ohlc":  {sym: list(dq) for sym, dq in self.buffers["ohlc"].items()},
            "equity": list(self.buffers["equity"]),
            "metrics": list(self.buffers["metrics"]),
            "news": list(self.buffers["news"]),
            "trades": list(self.buffers["trades"]),
        }

    def _emit_snapshot(self):
        """Emit a deep copy of all state for GUI consumption."""
        # Only emit if we’ve seen at least one health update (feeder alive)
        if not self.cache["health"]:
            return

        self._compute_metrics()
        snap = deepcopy(self.cache)
        snap["buffers"] = self._serialize_buffers()
        snap["timestamp"] = self._iso_now()
        self.snapshot_ready.emit(snap)
        self._last_emit = time.time()

    def _prune_inactive_symbols(self):
        """Drop price/ohlc buffers for symbols that have gone stale."""
        if not self._last_seen:
            return
        now = time.time()
        stale = [sym for sym, last in self._last_seen.items()
                 if (now - last) > self.SYMBOL_STALE_SECS]
        if not stale:
            return

        for sym in stale:
            self.buffers["price"].pop(sym, None)
            self.buffers["ohlc"].pop(sym, None)
            # keep positions/orders intact; we only prune chart buffers
            self._last_seen.pop(sym, None)
