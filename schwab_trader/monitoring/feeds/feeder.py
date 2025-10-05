# monitoring/feeds/data_feeder.py
from PySide6 import QtCore
import pandas as pd
from typing import List, Dict
import asyncio
import time
from datetime import datetime, timezone

from core.events import events
from core.events.eventhandler import EventHandler, get_event_handler


class FeedSignals(QtCore.QObject):
    """
    Central signal hub bridging EventBus → Qt GUI.
    All signals use 'object' for safety (avoids QMetaType errors).
    """

    symbols          = QtCore.Signal(object)           # position updates [{symbol, qty, avg, ...}]
    orders           = QtCore.Signal(object)           # order status updates [{order_id, symbol, status, ...}]
    equity_point     = QtCore.Signal(str, float)       # timestamp, portfolio_value
    realized_point   = QtCore.Signal(str, float, float) # ts, realized, unrealized
    risk_stats       = QtCore.Signal(float, float, float) # unrealized, realized, drawdown
    log              = QtCore.Signal(str)
    health           = QtCore.Signal(object)
    queue            = QtCore.Signal(object)
    cooldown         = QtCore.Signal(bool)
    alerts           = QtCore.Signal(object)           # list[dict] or single alert dict
    news             = QtCore.Signal(object)
    benchmark_point  = QtCore.Signal(str, float)
    strategy_signals = QtCore.Signal(object)
    regime_breakdown = QtCore.Signal(object)
    trades           = QtCore.Signal(object)           # trade dict or list of trades
    ohlc             = QtCore.Signal(str, object)      # symbol, DataFrame/dict
    pnl_update       = QtCore.Signal(object)           # full PnL payload
    regime_state     = QtCore.Signal(object)
    regime_perf      = QtCore.Signal(object)
    



class DataFeeder(QtCore.QObject):
    """
    Bridges async EventBus (backend) → Qt signals (GUI).
    Thread-safe, resilient, and auto-reconnecting.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.s = FeedSignals()
        self.bus = get_event_handler()
        self._subs = []
        self._running = False
        self._reconnect_task = None


    # ------------------------------------------------------------
    # Thread-safe Qt emit helper
    # ------------------------------------------------------------
class DataFeeder(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.s = FeedSignals()             # ✅ create signal hub
        self.bus = get_event_handler()     # ✅ subscribe to the global bus
        self._running = False
        self._reconnect_task = None
        self._last_emit = time.time()
        self._heartbeat_timer = QtCore.QTimer()
        self._heartbeat_timer.timeout.connect(self._check_heartbeat)
        self._heartbeat_timer.start(5000)  # every 5 seconds

    def _emit_safe(self, signal: QtCore.Signal, *args):
        """
        Thread-safe Qt signal emission.
        Runs emit() on the GUI thread using a queued timer call.
        """
        self._last_emit = time.time()  # 🩺 record last GUI event emission
        QtCore.QTimer.singleShot(0, lambda: signal.emit(*args))

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------
    async def start_safe(self, retry_interval: float = 3.0):
        """
        Try to subscribe safely. If it fails, retry until successful.
        Designed for GUI dashboards that run indefinitely.
        """
        if self._running:
            print("[DataFeeder] Already running — skipping duplicate start.")
            return

        self._running = True

        async def _connect_loop():
            while self._running:
                try:
                    if not self.bus:
                        raise RuntimeError("No EventHandler bound")

                    await self.start()
                    print("[DataFeeder] Connected to EventBus and ready.")
                    break  # connected, stop retry loop

                except Exception as e:
                    print(f"[DataFeeder] Connection failed: {e!r}, retrying in {retry_interval}s...")
                    await asyncio.sleep(retry_interval)

        self._reconnect_task = asyncio.create_task(_connect_loop())
    
    def _check_heartbeat(self):
        now = time.time()
        lag = now - self._last_emit

        if lag > 5:  # feed silent for >5s
            payload = {
                "broker": "Simulation",
                "status": "stale",
                "details": {"last_emit_age": round(lag, 2)},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        else:
            payload = {
                "broker": "Simulation",
                "status": "healthy",
                "details": {"last_emit_age": round(lag, 2)},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # 🩺 Emit directly to GUI (so health panel updates)
        self._emit_safe(self.s.health, payload)

        # Optionally: also forward via bus so backend sees it
        if self.bus:
            asyncio.create_task(self.bus.emit("HEALTH_UPDATE", payload))


    async def stop(self):
        """Cleanly unsubscribe from all events."""
        self._running = False
        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None

        for unsub in self._subs:
            try:
                await unsub()
            except Exception as e:
                print(f"[DataFeeder] Failed to unsubscribe cleanly: {e!r}")
        self._subs.clear()

        print("[DataFeeder] Disconnected cleanly.")

    # ------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------
    async def start(self):
        """(Re)subscribe to all event topics on the EventBus."""
        self._subs = [
            await self.bus.subscribe(events.EVENT_POSITION_UPDATE, self._on_position),
            await self.bus.subscribe(events.EVENT_ORDER_STATUS, self._on_order),
            await self.bus.subscribe(events.EVENT_NEW_TRADE, self._on_trade),
            await self.bus.subscribe(events.EVENT_PNL_UPDATE, self._on_pnl_update),
            await self.bus.subscribe(events.EVENT_LOG, self._on_log),
            await self.bus.subscribe(events.EVENT_HEALTH_UPDATE, self._on_health),
            await self.bus.subscribe(events.EVENT_ORDER_QUEUE_UPDATE, self._on_queue),
            await self.bus.subscribe(events.EVENT_COOLDOWN_STATE, self._on_cooldown),
            await self.bus.subscribe(events.EVENT_ALERT, self._on_alert),
            await self.bus.subscribe(events.EVENT_NEWS_UPDATE, self._on_news),
            await self.bus.subscribe(events.EVENT_NEW_BAR, self._on_bar),
            await self.bus.subscribe(events.EVENT_BENCHMARK_UPDATE, self._on_benchmark),
            await self.bus.subscribe(events.EVENT_STRATEGY_SIGNAL, self._on_strategy),
            await self.bus.subscribe(events.EVENT_REGIME_UPDATE, self._on_regime),
            await self.bus.subscribe(events.EVENT_REGIME_PERF, self._on_regime_perf),
            await self.bus.subscribe(events.EVENT_PRICE_UPDATE, self._on_price_update),
            await self.bus.subscribe(events.EVENT_PERFORMANCE_METRICS, self._on_perf_metrics),
            await self.bus.subscribe(events.EVENT_HALT_STATE, self._on_halt_state),
            await self.bus.subscribe(events.EVENT_REGIME_UPDATE, self._on_regime_update)
        ]


    # ------------------------------------------------------------
    # Event → Qt Signal mappings
    # ------------------------------------------------------------
    async def _on_position(self, event: events.PositionPayload):
        data = getattr(event, "payload", event)
        rows = data if isinstance(data, list) else [data]
        self._emit_safe(self.s.symbols, rows)

    async def _on_order(self, event: events.OrderStatusPayload):
        data = getattr(event, "payload", event)
        rows = data if isinstance(data, list) else [data]
        self._emit_safe(self.s.orders, rows)

    async def _on_trade(self, event: events.TradePayload):
        data = getattr(event, "payload", event)
        rows = data if isinstance(data, list) else [data]
        self._emit_safe(self.s.trades, rows)

    async def _on_pnl_update(self, event: events.PnLPayload):
        data = getattr(event, "payload", event)
        if not isinstance(data, dict):
            return
        ts = data.get("timestamp", "")
        self._emit_safe(self.s.equity_point, ts, float(data.get("portfolio_value", 0.0)))
        self._emit_safe(
            self.s.realized_point,
            ts,
            float(data.get("realized", 0.0)),
            float(data.get("unrealized", 0.0)),
        )
        self._emit_safe(
            self.s.risk_stats,
            float(data.get("unrealized", 0.0)),
            float(data.get("realized", 0.0)),
            float(data.get("drawdown", 0.0)),
        )

    async def _on_log(self, event: events.LogPayload):
        data = getattr(event, "payload", event)
        msg = f"[{data.get('level', 'INFO').upper()}] {data.get('message', '')}"
        self._emit_safe(self.s.log, msg)

    async def _on_health(self, event: events.HealthPayload):
        self._emit_safe(self.s.health, getattr(event, "payload", event))

    async def _on_queue(self, event):
        self._emit_safe(self.s.queue, getattr(event, "payload", event))

    async def _on_cooldown(self, event):
        payload = getattr(event, "payload", event)
        self._emit_safe(self.s.cooldown, bool(payload))

    async def _on_alert(self, event: events.AlertPayload):
        data = getattr(event, "payload", event)
        alerts = data if isinstance(data, list) else [data]
        self._emit_safe(self.s.alerts, alerts)

    async def _on_news(self, event: events.NewsPayload):
        data = getattr(event, "payload", event)
        items = data if isinstance(data, list) else [data]
        self._emit_safe(self.s.news, items)

    async def _on_bar(self, event: events.BarPayload):
        bar = getattr(event, "payload", event)
        if not isinstance(bar, dict):
            return
        df = pd.DataFrame([bar])
        self._emit_safe(self.s.ohlc, bar.get("symbol", ""), df)

    async def _on_benchmark(self, event: events.BenchmarkPayload):
        data = getattr(event, "payload", event)
        ts = data.get("timestamp", "")
        benchmark_curve = data.get("benchmark_curve", [0.0])
        last_val = benchmark_curve[-1] if benchmark_curve else 0.0
        self._emit_safe(self.s.benchmark_point, ts, last_val)

    async def _on_strategy(self, event: events.StrategySignalPayload):
        data = getattr(event, "payload", event)
        rows = data if isinstance(data, list) else [data]
        self._emit_safe(self.s.strategy_signals, rows)

    async def _on_regime(self, event: events.RegimePayload):
        data = getattr(event, "payload", event)
        payload = {
            "symbol": data.get("symbol", ""),
            "volatility": data.get("volatility", ""),
            "trend": data.get("trend", ""),
            "market": data.get("market", ""),
            "timestamp": data.get("timestamp", ""),
        }
        self._emit_safe(self.s.regime_breakdown, payload)

    async def _on_regime_perf(self, event: events.RegimePerfPayload):
        data = getattr(event, "payload", event)
        regimes = data.get("regimes", {})
        self._emit_safe(self.s.regime_breakdown, regimes)

    async def _on_price_update(self, event: events.PricePayload):
        data = getattr(event, "payload", event)
        self._emit_safe(self.s.ohlc, data.get("symbol", ""), pd.DataFrame([data]))
    
    async def _on_perf_metrics(self, event: events.PerformanceMetricsPayload):
        data = getattr(event, "payload", event)
        self._emit_safe(self.s.pnl_update, data)

    async def _on_halt_state(self, event):
        payload = getattr(event, "payload", event)
        self._emit_safe(self.s.cooldown, bool(payload.get("halted", False)))
    
    async def _on_regime_update(self, event):
        data = getattr(event, "payload", event)
        self._emit_safe(self.s.regime_state, data)