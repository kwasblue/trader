import asyncio

from PySide6 import QtCore

from core.contracts.events import (
    EVENT_CANCEL_ALL,
    EVENT_FLATTEN_ALL,
    EVENT_FLATTEN_SYMBOL,
    EVENT_HALTED,
    EVENT_MANUAL_ORDER,
)
from core.events.eventhandler import EventHandler

# =====================================================
# CONTROL BUS (signals your engine will subscribe to)
# =====================================================


class ControlBridge(QtCore.QObject):
    """
    Emit UI -> Engine intents. Wire these to your EventHandler or Broker.
    qtIn production, forward to your async event bus.
    """

    halt_changed = QtCore.Signal(bool)  # True -> halted, False -> resumed
    flatten_all = QtCore.Signal()  # intent to close all positions
    cancel_all = QtCore.Signal()  # intent to cancel all working orders
    flatten_symbol = QtCore.Signal(str)  # intent to close a single symbol
    manual_order = QtCore.Signal(dict)  # {symbol, side, qty, type, price?, tif, reduce_only, sl?, tp?, route}

    def __init__(self, bus: EventHandler):
        super().__init__()
        self.bus = bus

        # --- Forward Qt signals into event bus ---
        self.halt_changed.connect(self._emit_halt)
        self.flatten_all.connect(lambda: asyncio.create_task(self.bus.emit(EVENT_FLATTEN_ALL, {})))
        self.cancel_all.connect(lambda: asyncio.create_task(self.bus.emit(EVENT_CANCEL_ALL, {})))
        self.flatten_symbol.connect(
            lambda symbol: asyncio.create_task(self.bus.emit(EVENT_FLATTEN_SYMBOL, {"symbol": symbol}))
        )
        self.manual_order.connect(lambda order: asyncio.create_task(self.bus.emit(EVENT_MANUAL_ORDER, order)))

    def _emit_halt(self, is_halted: bool):
        asyncio.create_task(self.bus.emit(EVENT_HALTED, {"halted": is_halted}))
