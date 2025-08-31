from PySide6 import QtCore

# =====================================================
# CONTROL BUS (signals your engine can subscribe to)
# =====================================================

class ControlBridge(QtCore.QObject):
    """
    Emit UI -> Engine intents. Wire these to your EventHandler or Broker.
    qtIn production, forward to your async event bus.
    """
    halt_changed = QtCore.Signal(bool)       # True -> halted, False -> resumed
    flatten_all  = QtCore.Signal()           # intent to close all positions
    cancel_all   = QtCore.Signal()           # intent to cancel all working orders
    flatten_symbol = QtCore.Signal(str)      # intent to close a single symbol
    manual_order = QtCore.Signal(dict)       # {symbol, side, qty, type, price?, tif, reduce_only, sl?, tp?, route}