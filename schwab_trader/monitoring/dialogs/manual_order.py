from PySide6 import QtWidgets
from typing import Optional, List, Dict

# =====================================================
# MANUAL ORDER DIALOG
# =====================================================
class ManualOrderDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, symbols: Optional[List[str]]=None):
        super().__init__(parent)
        self.setWindowTitle("Manual Order")
        self.setModal(True)
        self.setMinimumWidth(460)
        self._symbols = symbols or ["AAPL","MSFT","TSLA","AMD"]

        form = QtWidgets.QFormLayout(self)
        self.symbol = QtWidgets.QComboBox(); self.symbol.addItems(self._symbols)
        self.side = QtWidgets.QComboBox(); self.side.addItems(["BUY","SELL","SHORT","COVER"])
        self.qty = QtWidgets.QSpinBox(); self.qty.setRange(1, 1_000_000); self.qty.setValue(100)
        self.ord_type = QtWidgets.QComboBox(); self.ord_type.addItems(["MARKET","LIMIT"]) 
        self.price = QtWidgets.QDoubleSpinBox(); self.price.setRange(0.0, 1_000_000.0); self.price.setDecimals(4); self.price.setValue(0.0); self.price.setEnabled(False)
        self.tif = QtWidgets.QComboBox(); self.tif.addItems(["DAY","IOC","FOK","GTC"]) 
        self.route = QtWidgets.QComboBox(); self.route.addItems(["Auto","Alpaca","Schwab","Sim"])  # NEW: Route selector
        self.reduce_only = QtWidgets.QCheckBox("Reduce-only")
        self.sl = QtWidgets.QDoubleSpinBox(); self.sl.setRange(0.0, 1_000_000.0); self.sl.setDecimals(4); self.sl.setValue(0.0)
        self.tp = QtWidgets.QDoubleSpinBox(); self.tp.setRange(0.0, 1_000_000.0); self.tp.setDecimals(4); self.tp.setValue(0.0)

        self.ord_type.currentTextChanged.connect(lambda t: self.price.setEnabled(t=="LIMIT"))

        form.addRow("Symbol", self.symbol)
        form.addRow("Side", self.side)
        form.addRow("Qty", self.qty)
        form.addRow("Type", self.ord_type)
        form.addRow("Limit Price", self.price)
        form.addRow("TIF", self.tif)
        form.addRow("Route", self.route)
        form.addRow(self.reduce_only)
        form.addRow("Stop Loss", self.sl)
        form.addRow("Take Profit", self.tp)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def _on_accept(self):
        if self.ord_type.currentText()=="LIMIT" and self.price.value()<=0:
            QtWidgets.QMessageBox.warning(self, "Manual Order", "Limit price must be > 0 for LIMIT orders.")
            return
        self.accept()

    def payload(self) -> Dict:
        return {
            "symbol": self.symbol.currentText(),
            "side": self.side.currentText(),
            "qty": int(self.qty.value()),
            "type": self.ord_type.currentText().lower(),
            "price": float(self.price.value()) if self.ord_type.currentText()=="LIMIT" else None,
            "tif": self.tif.currentText(),
            "route": self.route.currentText(),
            "reduce_only": bool(self.reduce_only.isChecked()),
            "sl": float(self.sl.value()) or None,
            "tp": float(self.tp.value()) or None,
        }