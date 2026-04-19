import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui


# =====================================================
# WIDGET HELPERS
# =====================================================
class Candles(pg.GraphicsObject):
    """Simple candlestick item for pyqtgraph."""

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df
        self._picture = None
        self.generatePicture()

    def generatePicture(self):
        self._picture = QtGui.QPicture()
        p = QtGui.QPainter(self._picture)
        w = 0.6
        for i, row in self._df.reset_index().iterrows():
            open_, close, high, low = row["open"], row["close"], row["high"], row["low"]
            x = i
            top = max(open_, close)
            bottom = min(open_, close)
            p.setPen(QtGui.QPen(QtGui.QColor("#e5e5e5")))
            p.drawLine(QtCore.QPointF(x, low), QtCore.QPointF(x, high))
            brush = QtGui.QBrush(QtGui.QColor("#22c55e" if close >= open_ else "#f87171"))
            p.fillRect(QtCore.QRectF(x - w / 2, bottom, w, top - bottom or 0.001), brush)
        p.end()

    def paint(self, p, *args):
        if self._picture:
            p.drawPicture(0, self._picture)

    def boundingRect(self):
        return QtCore.QRectF(
            0,
            float(self._df.index.max() + 1),
            float(self._df["low"].min() * 0.99),
            float(self._df["high"].max() * 1.01),
        ).normalized()
