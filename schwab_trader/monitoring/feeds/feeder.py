# monitoring/feeds/feeder.py
"""
DataFeeder - Bridges EventBus to Qt GUI.

CRITICAL: Subscriptions happen SYNCHRONOUSLY in __init__ to ensure
they are in place before any simulation can start emitting events.
"""
from PySide6 import QtCore
import asyncio
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

from core.events import events
from core.events.eventhandler import get_event_handler, Event
from loggers.logger import Logger

# Use Logger class for consistent logging with propagation to app.log
_logger_instance = Logger(
    log_file="data_feeder.log",
    logger_name="DataFeeder",
    propagate=True,  # Also log to app.log
    console=False,   # Console output handled separately
)
logger = _logger_instance.get_logger()


class FeedSignals(QtCore.QObject):
    """Qt signals for GUI updates."""
    pnl_update = QtCore.Signal(object)
    equity_update = QtCore.Signal(float)
    bar_update = QtCore.Signal(str, object)
    price_update = QtCore.Signal(str, float)
    health_update = QtCore.Signal(object)
    trade_update = QtCore.Signal(object)
    position_update = QtCore.Signal(object)
    order_update = QtCore.Signal(object)
    log_message = QtCore.Signal(str)
    alert = QtCore.Signal(object)


class DataFeeder(QtCore.QObject):
    """
    Bridges EventBus to Qt GUI.

    IMPORTANT: Subscriptions are made SYNCHRONOUSLY in __init__ to guarantee
    they are in place before any backend can start emitting events.
    """

    def __init__(self):
        super().__init__()
        self.s = FeedSignals()
        self.bus = get_event_handler()
        self._running = True  # Mark as running immediately
        self._event_count = 0
        self._last_event_time = time.time()

        logger.info(f"DataFeeder created, EventBus ID: {id(self.bus)}")
        print(f"[Feeder] DataFeeder created, EventBus ID: {id(self.bus)}")

        # === SYNCHRONOUS SUBSCRIPTION - CRITICAL ===
        # Subscribe immediately in constructor to guarantee subscriptions
        # are in place before any simulation can run
        self._subscribe_all()

        # Health check timer
        self._health_timer = QtCore.QTimer()
        self._health_timer.timeout.connect(self._emit_health)
        self._health_timer.start(2000)

    def _subscribe_all(self):
        """Subscribe to all events SYNCHRONOUSLY."""
        logger.info("Subscribing to events (SYNC)...")
        print("[Feeder] Subscribing to events (SYNC)...")

        subscriptions = [
            (events.EVENT_PNL_UPDATE, self._handle_pnl),
            (events.EVENT_NEW_BAR, self._handle_bar),
            (events.EVENT_NEW_TRADE, self._handle_trade),
            (events.EVENT_ORDER_STATUS, self._handle_order),
            (events.EVENT_POSITION_UPDATE, self._handle_position),
            (events.EVENT_HEALTH_UPDATE, self._handle_health),
            (events.EVENT_ALERT, self._handle_alert),
            (events.EVENT_PRICE_UPDATE, self._handle_price),
        ]

        for event_name, handler in subscriptions:
            self.bus.subscribe_sync(event_name, handler)
            logger.info(f"  Subscribed to {event_name}")

        # Verify and log
        for event_name, _ in subscriptions:
            count = len(self.bus.listeners.get(event_name, []))
            print(f"[Feeder] {event_name}: {count} listener(s)")

        logger.info("All subscriptions complete (SYNC)")
        print("[Feeder] All subscriptions complete (SYNC)")

    def _emit(self, signal, *args):
        """Thread-safe signal emission to Qt."""
        self._last_event_time = time.time()
        self._event_count += 1
        # Use QTimer.singleShot(0) to ensure emission happens in Qt's event loop
        QtCore.QTimer.singleShot(0, lambda: signal.emit(*args))

    def _emit_health(self):
        """Emit health status periodically."""
        elapsed = time.time() - self._last_event_time
        status = "healthy" if self._running and elapsed < 5 else "stale"

        payload = {
            "status": status,
            "details": {
                "last_emit_age": round(elapsed, 1),
                "event_count": self._event_count,
                "running": self._running,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._emit(self.s.health_update, payload)

    async def start(self):
        """
        Legacy async start method - now a no-op since subscriptions
        happen synchronously in __init__.
        """
        # Subscriptions already done in __init__
        if not self._running:
            self._running = True
        logger.info("start() called - subscriptions already active")

    async def stop(self):
        """Stop the feeder."""
        self._running = False
        logger.info("Stopped")

    # ================================================================
    # Event Handlers - Called by EventBus when events are emitted
    # ================================================================

    async def _handle_pnl(self, event: Event):
        """Handle PNL event from EventBus."""
        data = event.payload if hasattr(event, 'payload') else event
        value = data.get('portfolio_value', 0)
        logger.debug(f"PNL received: ${value:,.2f}")
        print(f"[Feeder] >>> PNL received: ${value:,.2f}")

        # Emit Qt signals to GUI
        self._emit(self.s.pnl_update, data)

        if value:
            self._emit(self.s.equity_update, float(value))

    async def _handle_bar(self, event: Event):
        """Handle bar event from EventBus."""
        data = event.payload if hasattr(event, 'payload') else event
        symbol = data.get('symbol', 'UNKNOWN')
        close = data.get('close', 0)
        logger.debug(f"BAR: {symbol} close={close}")

        self._emit(self.s.bar_update, symbol, data)

        if close:
            self._emit(self.s.price_update, symbol, float(close))

    async def _handle_trade(self, event: Event):
        """Handle trade event from EventBus."""
        data = event.payload if hasattr(event, 'payload') else event
        logger.debug(f"TRADE: {data}")
        print(f"[Feeder] TRADE: {data.get('symbol')} {data.get('side')} {data.get('qty')}")
        self._emit(self.s.trade_update, data)

    async def _handle_order(self, event: Event):
        """Handle order event from EventBus."""
        data = event.payload if hasattr(event, 'payload') else event
        logger.debug(f"ORDER: {data}")
        self._emit(self.s.order_update, data)

    async def _handle_position(self, event: Event):
        """Handle position event from EventBus."""
        data = event.payload if hasattr(event, 'payload') else event
        logger.debug(f"POSITION: {data}")
        self._emit(self.s.position_update, data)

    async def _handle_health(self, event: Event):
        """Handle external health event from EventBus."""
        data = event.payload if hasattr(event, 'payload') else event
        self._emit(self.s.health_update, data)

    async def _handle_alert(self, event: Event):
        """Handle alert event from EventBus."""
        data = event.payload if hasattr(event, 'payload') else event
        logger.debug(f"ALERT: {data}")
        self._emit(self.s.alert, data)

    async def _handle_price(self, event: Event):
        """Handle price event from EventBus."""
        data = event.payload if hasattr(event, 'payload') else event
        symbol = data.get('symbol', '')
        price = data.get('price', 0)
        if symbol and price:
            self._emit(self.s.price_update, symbol, float(price))
