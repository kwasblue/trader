
import math
from typing import List, Dict
from PySide6 import QtCore, QtGui

# =====================================================
# TABLE MODELS
# =====================================================
class SymbolsTableModel(QtCore.QAbstractTableModel):
    HEADERS = ["Symbol","Side","Qty","Avg","Last","Unreal","Realized","Total","PnL %","ATR","Regime","Risk"]
    SIGN_COLS = {5,6,7,8}

    def __init__(self, rows: List[Dict] | None = None):
        super().__init__(); self._rows: List[Dict] = rows or []

    def rowCount(self, parent=QtCore.QModelIndex()): return len(self._rows)
    def columnCount(self, parent=QtCore.QModelIndex()): return len(self.HEADERS)
    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole and orientation == QtCore.Qt.Horizontal: return self.HEADERS[section]
        return None

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid(): return None
        r,c=index.row(), index.column(); row=self._rows[r]
        total = (row.get("unreal") or 0) + (row.get("realized") or 0)
        cols=[row.get("symbol"),row.get("side"),row.get("qty"),row.get("avg"),row.get("last"),row.get("unreal"),row.get("realized"),total,row.get("pnl_pct"),row.get("atr"),row.get("regime"),row.get("risk")]
        val = cols[c]

        if role == QtCore.Qt.DisplayRole:
            if isinstance(val, float):
                if c==2: return f"{val:,.0f}"
                if c in (3,4,5,6,7,9): return f"{val:,.2f}"
                if c==8: return f"{val*100:.2f}%"
            return str(val)
        if role == QtCore.Qt.ForegroundRole:
            if c in self.SIGN_COLS and isinstance(val,(int,float)) and not (isinstance(val,float) and math.isnan(val)):
                if val>0: return QtGui.QBrush(QtGui.QColor("#22c55e"))
                if val<0: return QtGui.QBrush(QtGui.QColor("#f87171"))
            return QtGui.QBrush(QtGui.QColor("#e5e5e5"))
        if role == QtCore.Qt.TextAlignmentRole:
            return int(QtCore.Qt.AlignVCenter | (QtCore.Qt.AlignLeft if c==0 else QtCore.Qt.AlignRight))
        if role == QtCore.Qt.BackgroundRole and c==11:
            risk=row.get("risk") or 0
            if risk>60: return QtGui.QBrush(QtGui.QColor(60,0,0,80))
            if risk>40: return QtGui.QBrush(QtGui.QColor(60,40,0,80))
            if risk>20: return QtGui.QBrush(QtGui.QColor(0,40,40,80))
            return QtGui.QBrush(QtGui.QColor(0,60,20,80))
        return None

    def replace_rows(self, rows: List[Dict]):
        self.beginResetModel(); self._rows=rows; self.endResetModel()

class OrdersTableModel(QtCore.QAbstractTableModel):
    HEADERS = ["Time","Symbol","Side","Qty","Price","Status"]
    def __init__(self, rows: List[Dict] | None=None): super().__init__(); self._rows=rows or []
    def rowCount(self, parent=QtCore.QModelIndex()): return len(self._rows)
    def columnCount(self, parent=QtCore.QModelIndex()): return len(self.HEADERS)
    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role==QtCore.Qt.DisplayRole and orientation==QtCore.Qt.Horizontal: return self.HEADERS[section]
        return None
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid(): return None
        r,c=index.row(), index.column(); row=self._rows[r]
        cols=[row.get("ts"),row.get("symbol"),row.get("side"),row.get("qty"),row.get("price"),row.get("status")]
        val=cols[c]
        if role==QtCore.Qt.DisplayRole:
            if c==3 and isinstance(val,(int,float)): return f"{val:,.0f}"
            if c==4 and isinstance(val,(int,float)): return f"{val:,.2f}"
            return str(val)
        
        if role==QtCore.Qt.ForegroundRole:
            if c==2 and isinstance(val,str):
                if any(k in val.upper() for k in ["BUY","COVER"]): return QtGui.QBrush(QtGui.QColor("#22c55e"))
                if any(k in val.upper() for k in ["SELL","SHORT"]): return QtGui.QBrush(QtGui.QColor("#f87171"))
            return QtGui.QBrush(QtGui.QColor("#e5e5e5"))
        if role==QtCore.Qt.BackgroundRole:
            st=str(row.get("status",""))
            if st=="REJECTED": return QtGui.QBrush(QtGui.QColor(80,0,0,100))
            if st=="CANCELED": return QtGui.QBrush(QtGui.QColor(30,30,30,100))
        if role==QtCore.Qt.TextAlignmentRole:
            return int(QtCore.Qt.AlignVCenter | (QtCore.Qt.AlignLeft if c in (0,1,2,5) else QtCore.Qt.AlignRight))
        return None
    def replace_rows(self, rows: List[Dict]): self.beginResetModel(); self._rows=rows; self.endResetModel()

class TradesTableModel(QtCore.QAbstractTableModel):
    HEADERS=["Time","Symbol","Side","Qty","Price","Fee","SL","TP"]
    def __init__(self, rows: List[Dict] | None=None): super().__init__(); self._rows=rows or []
    def rowCount(self, parent=QtCore.QModelIndex()): return len(self._rows)
    def columnCount(self, parent=QtCore.QModelIndex()): return len(self.HEADERS)
    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role==QtCore.Qt.DisplayRole and orientation==QtCore.Qt.Horizontal: return self.HEADERS[section]
        return None
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid(): return None
        r,c=index.row(), index.column(); row=self._rows[r]
        cols=[row.get("ts"),row.get("symbol"),row.get("side"),row.get("qty"),row.get("price"),row.get("fee"),row.get("sl"),row.get("tp")]
        val=cols[c]
        if role==QtCore.Qt.DisplayRole:
            if c in (3,): return f"{val:,.0f}" if isinstance(val,(int,float)) else str(val)
            if c in (4,5,6,7): return f"{val:,.2f}" if isinstance(val,(int,float)) else "–"
            return str(val)
        if role==QtCore.Qt.ForegroundRole:
            if c==2 and isinstance(val,str):
                if any(k in val.upper() for k in ["BUY","COVER"]): return QtGui.QBrush(QtGui.QColor("#22c55e"))
                if any(k in val.upper() for k in ["SELL","SHORT"]): return QtGui.QBrush(QtGui.QColor("#f87171"))
            return QtGui.QBrush(QtGui.QColor("#e5e5e5"))
        if role==QtCore.Qt.TextAlignmentRole:
            return int(QtCore.Qt.AlignVCenter | (QtCore.Qt.AlignLeft if c in (0,1,2) else QtCore.Qt.AlignRight))
        return None
    def replace_rows(self, rows: List[Dict]): self.beginResetModel(); self._rows=rows; self.endResetModel()
