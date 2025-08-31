"""
Desktop Trading Bot Monitor — Feature Parity with React UI
=========================================================
Single-file PySide6 + pyqtgraph desktop app that mirrors the features we had in the React version:

- Risk panel (high-contrast numbers)
- Positions grid → table with per-symbol PnL (Unreal/Realized/Total), PnL %, ATR, Regime, and Risk score (ATR/Last scaled)
- Equity chart (cumulative)
- Realized PnL (daily bars + cumulative line)
- Orders table (status-aware, row tinting for CANCELED/REJECTED)
- Trades table
- Logs console with Clear button
- Start/Stop button (simulates engine running)

Run:
  pip install PySide6 pyqtgraph pandas numpy
  python bot_monitor_desktop.py

Integration notes:
- Replace DataFeeder.run() mocks with your engine snapshot/stream.
- Signals to emit:
    s.symbols.emit(list_of_symbol_dicts)
    s.equity_point.emit(date_str, equity)
    s.realized_point.emit(date_str, daily_realized, cumulative_realized)
    s.risk_stats.emit(total_unreal, total_realized, max_drawdown_float)
    s.orders.emit(list_of_order_dicts)
    s.trades.emit(list_of_trade_dicts)
    s.log.emit(str_line)
- Dict shapes expected are defined below in the models.
"""

from __future__ import annotations
import sys
from typing import List, Dict

from PySide6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pandas as pd
import numpy as np
import datetime as dt
import random

# ===============================
# THEME (high-contrast dark)
# ===============================

def apply_dark_palette(app: QtWidgets.QApplication) -> None:
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#0a0a0a"))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#e5e5e5"))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#0f0f0f"))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#111111"))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor("#0f0f0f"))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor("#e5e5e5"))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor("#e5e5e5"))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor("#141414"))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#f0f0f0"))
    palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor("#ff4d4f"))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#2563eb"))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
    app.setPalette(palette)

# ===============================
# TABLE MODELS
# ===============================
class SymbolsTableModel(QtCore.QAbstractTableModel):
    """Per-symbol telemetry & PnL
    Keys expected per row:
      symbol, side, qty, avg, last, unreal, realized, pnl_pct, atr, regime, risk
    """
    HEADERS = [
        "Symbol","Side","Qty","Avg","Last",
        "Unreal","Realized","Total","PnL %","ATR","Regime","Risk"
    ]
    SIGN_COLS = {5,6,7,8}  # Unreal, Realized, Total, PnL %

    def __init__(self, rows: List[Dict] | None = None):
        super().__init__()
        self._rows: List[Dict] = rows or []

    def rowCount(self, parent=QtCore.QModelIndex()): return len(self._rows)
    def columnCount(self, parent=QtCore.QModelIndex()): return len(self.HEADERS)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole and orientation == QtCore.Qt.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid(): return None
        r, c = index.row(), index.column()
        row = self._rows[r]
        total = (row.get("unreal") or 0) + (row.get("realized") or 0)
        cols = [
            row.get("symbol"), row.get("side"), row.get("qty"), row.get("avg"), row.get("last"),
            row.get("unreal"), row.get("realized"), total, row.get("pnl_pct"), row.get("atr"), row.get("regime"), row.get("risk")
        ]
        val = cols[c]

        if role == QtCore.Qt.DisplayRole:
            if isinstance(val, float):
                if c == 2: return f"{val:,.0f}"  # qty
                if c in (3,4,5,6,7,9): return f"{val:,.2f}"  # money-ish
                if c == 8: return f"{val*100:.2f}%"  # pct
            return str(val)

        if role == QtCore.Qt.ForegroundRole:
            if c in self.SIGN_COLS and isinstance(val,(int,float)) and not np.isnan(val):
                if val > 0: return QtGui.QBrush(QtGui.QColor("#22c55e"))
                if val < 0: return QtGui.QBrush(QtGui.QColor("#f87171"))
            return QtGui.QBrush(QtGui.QColor("#e5e5e5"))

        if role == QtCore.Qt.TextAlignmentRole:
            return int(QtCore.Qt.AlignVCenter | (QtCore.Qt.AlignLeft if c==0 else QtCore.Qt.AlignRight))

        if role == QtCore.Qt.BackgroundRole and c == 11:  # Risk chip shading
            risk = row.get("risk") or 0
            if risk > 60: return QtGui.QBrush(QtGui.QColor(60,0,0,80))
            if risk > 40: return QtGui.QBrush(QtGui.QColor(60,40,0,80))
            if risk > 20: return QtGui.QBrush(QtGui.QColor(0,40,40,80))
            return QtGui.QBrush(QtGui.QColor(0,60,20,80))
        return None

    def replace_rows(self, rows: List[Dict]):
        self.beginResetModel(); self._rows = rows; self.endResetModel()

class OrdersTableModel(QtCore.QAbstractTableModel):
    """Orders with status highlighting.
    Keys: id, ts, symbol, side, qty, price, status
    """
    HEADERS = ["Time","Symbol","Side","Qty","Price","Status"]

    def __init__(self, rows: List[Dict] | None=None):
        super().__init__(); self._rows = rows or []

    def rowCount(self, parent=QtCore.QModelIndex()): return len(self._rows)
    def columnCount(self, parent=QtCore.QModelIndex()): return len(self.HEADERS)
    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role==QtCore.Qt.DisplayRole and orientation==QtCore.Qt.Horizontal: return self.HEADERS[section]
        return None

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid(): return None
        r,c=index.row(), index.column(); row=self._rows[r]
        cols=[row.get("ts"), row.get("symbol"), row.get("side"), row.get("qty"), row.get("price"), row.get("status")]
        val=cols[c]
        if role==QtCore.Qt.DisplayRole:
            if c==0: return str(val)
            if c==3 and isinstance(val,(int,float)): return f"{val:,.0f}"
            if c==4 and isinstance(val,(int,float)): return f"{val:,.2f}"
            return str(val)
        if role==QtCore.Qt.ForegroundRole:
            if c==2: # side
                if isinstance(val,str) and any(k in val.upper() for k in ["BUY","COVER"]):
                    return QtGui.QBrush(QtGui.QColor("#22c55e"))
                if isinstance(val,str) and any(k in val.upper() for k in ["SELL","SHORT"]):
                    return QtGui.QBrush(QtGui.QColor("#f87171"))
            return QtGui.QBrush(QtGui.QColor("#e5e5e5"))
        if role==QtCore.Qt.BackgroundRole:
            st=str(row.get("status",""))
            if st=="REJECTED": return QtGui.QBrush(QtGui.QColor(80,0,0,100))
            if st=="CANCELED": return QtGui.QBrush(QtGui.QColor(30,30,30,100))
        if role==QtCore.Qt.TextAlignmentRole:
            return int(QtCore.Qt.AlignVCenter | (QtCore.Qt.AlignLeft if c in (0,1,2,5) else QtCore.Qt.AlignRight))
        return None

    def replace_rows(self, rows: List[Dict]):
        self.beginResetModel(); self._rows=rows; self.endResetModel()

class TradesTableModel(QtCore.QAbstractTableModel):
    """Trades table.
    Keys: id, ts, symbol, side, qty, price, fee, sl, tp
    """
    HEADERS = ["Time","Symbol","Side","Qty","Price","Fee","SL","TP"]
    def __init__(self, rows: List[Dict] | None=None): super().__init__(); self._rows=rows or []
    def rowCount(self, parent=QtCore.QModelIndex()): return len(self._rows)
    def columnCount(self, parent=QtCore.QModelIndex()): return len(self.HEADERS)
    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role==QtCore.Qt.DisplayRole and orientation==QtCore.Qt.Horizontal: return self.HEADERS[section]
        return None
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid(): return None
        r,c=index.row(), index.column(); row=self._rows[r]
        cols=[row.get("ts"), row.get("symbol"), row.get("side"), row.get("qty"), row.get("price"), row.get("fee"), row.get("sl"), row.get("tp")]
        val=cols[c]
        if role==QtCore.Qt.DisplayRole:
            if c==0: return str(val)
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
    def replace_rows(self, rows: List[Dict]):
        self.beginResetModel(); self._rows=rows; self.endResetModel()

# ===============================
# DATA FEEDER (background thread)
# ===============================
class FeedSignals(QtCore.QObject):
    symbols = QtCore.Signal(list)
    orders  = QtCore.Signal(list)
    trades  = QtCore.Signal(list)
    equity_point   = QtCore.Signal(str, float)
    realized_point = QtCore.Signal(str, float, float)
    risk_stats     = QtCore.Signal(float, float, float)
    log = QtCore.Signal(str)

class DataFeeder(QtCore.QThread):
    """Replace internals with your engine reads. Emits ~1 Hz updates."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.s = FeedSignals()
        self._running = True
        # mock state
        self._equity = 100_000.0
        self._cum_realized = 0.0
        self._peak = self._equity
        self._max_dd = 0.0
        self._order_id = 4

    def stop(self): self._running = False

    def run(self):
        # initial snapshot push
        self.s.orders.emit(self._mock_orders())
        self.s.trades.emit(self._mock_trades())
        while self._running:
            # ---- symbols snapshot (add ATR/regime/risk) ----
            symbols = []
            for sym in ["AAPL","MSFT","TSLA","AMD"]:
                last = random.uniform(100, 900)
                avg = last * random.uniform(0.98, 1.02)
                qty = random.choice([0, 10, 25, 50, 100])
                side = random.choice(["long","short","flat"]) if qty else "flat"
                unreal = (last-avg) * qty * (1 if side=="long" else -1 if side=="short" else 0)
                realized = random.uniform(-500, 800)
                pnl_pct = (last/avg - 1) if avg else 0.0
                atr = last * random.uniform(0.005, 0.04)  # 0.5%-4%
                regime = random.choice(["low_vol","normal_vol","high_vol"])
                risk = max(0, min(100, int((atr/max(last,1e-6))*100*2)))
                symbols.append({
                    "symbol": sym, "side": side, "qty": float(qty), "avg": float(avg), "last": float(last),
                    "unreal": float(unreal), "realized": float(realized), "pnl_pct": float(pnl_pct),
                    "atr": float(atr), "regime": regime, "risk": float(risk)
                })
            self.s.symbols.emit(symbols)

            # ---- equity & realized points ----
            shock = (random.random() - 0.5) * 0.006
            self._equity *= (1.0 + shock)
            self._peak = max(self._peak, self._equity)
            self._max_dd = min(self._max_dd, (self._equity - self._peak) / self._peak) if self._peak else 0.0

            daily_realized = (random.random() - 0.48) * 120
            self._cum_realized += daily_realized

            d = dt.date.today().isoformat()
            self.s.equity_point.emit(d, float(self._equity))
            self.s.realized_point.emit(d, float(daily_realized), float(self._cum_realized))

            # ---- risk totals ----
            total_unreal = float(sum(s["unreal"] for s in symbols))
            total_realized = float(sum(s["realized"] for s in symbols))
            self.s.risk_stats.emit(total_unreal, total_realized, float(self._max_dd))

            # ---- occasional order/log events ----
            if random.random() < 0.25:
                self._order_id += 1
                new_order = {
                    "id": f"o{self._order_id}",
                    "ts": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": random.choice(["AAPL","MSFT","TSLA","AMD"]),
                    "side": random.choice(["BUY","SELL","SHORT SELL","COVER"]),
                    "qty": random.choice([5,10,25,50]),
                    "price": round(random.uniform(100,900),2),
                    "status": random.choice(["FILLED","PARTIAL","CANCELED","REJECTED"])
                }
                orders = self._mock_orders() + [new_order]
                self.s.orders.emit(orders[-50:])
                self.s.log.emit(f"[{new_order['ts']}] {new_order['symbol']} {new_order['side']} {new_order['qty']} @ {new_order['price']}")

            self.msleep(1000)

    def _mock_orders(self) -> List[Dict]:
        return [
            {"id":"o1","ts":"2025-08-27 14:22:09","symbol":"AAPL","side":"BUY","qty":25,"price":219.10,"status":"FILLED"},
            {"id":"o2","ts":"2025-08-27 14:05:41","symbol":"TSLA","side":"SELL","qty":10,"price":239.90,"status":"PARTIAL"},
            {"id":"o3","ts":"2025-08-27 13:02:10","symbol":"MSFT","side":"SHORT SELL","qty":10,"price":422.40,"status":"CANCELED"},
            {"id":"o4","ts":"2025-08-27 12:41:00","symbol":"NVDA","side":"BUY","qty":5,"price":850.00,"status":"REJECTED"},
        ]

    def _mock_trades(self) -> List[Dict]:
        return [
            {"id":1,"ts":"2025-08-27 13:44:02","symbol":"AAPL","side":"BUY","qty":50,"price":218.70,"fee":0.75,"sl":213.30,"tp":224.20},
            {"id":2,"ts":"2025-08-27 10:15:10","symbol":"TSLA","side":"SELL","qty":10,"price":240.10,"fee":0.75,"sl":232.50,"tp":252.00},
            {"id":3,"ts":"2025-08-26 15:20:57","symbol":"MSFT","side":"SHORT SELL","qty":20,"price":423.20,"fee":0.75,"sl":436.00,"tp":412.70},
        ]

# ===============================
# MAIN WINDOW (tabs + charts + tables)
# ===============================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading Bot Monitor")
        self.resize(1380, 860)

        # pyqtgraph theme
        pg.setConfigOption("background", "#0a0a0a")
        pg.setConfigOption("foreground", "#e5e5e5")
        pg.setConfigOptions(antialias=True)

        # Toolbar
        toolbar = QtWidgets.QToolBar("Controls")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self.start_btn = QtGui.QAction("Start", self)
        self.stop_btn = QtGui.QAction("Stop", self)
        self.stop_btn.setEnabled(True)
        toolbar.addAction(self.start_btn)
        toolbar.addAction(self.stop_btn)
        toolbar.addSeparator()
        self.clear_logs_btn = QtGui.QAction("Clear Logs", self)
        toolbar.addAction(self.clear_logs_btn)

        # Tabs
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # DASHBOARD TAB (Risk + Positions + Charts)
        dash = QtWidgets.QWidget(); dash_grid = QtWidgets.QGridLayout(dash)
        dash_grid.setContentsMargins(12,12,12,12); dash_grid.setHorizontalSpacing(12); dash_grid.setVerticalSpacing(12)

        # Risk panel
        risk_box = QtWidgets.QGroupBox("Risk Panel")
        risk_layout = QtWidgets.QGridLayout(risk_box)
        risk_layout.setVerticalSpacing(6)
        self.unreal_lbl = self._kpi_label(); self.realized_lbl = self._kpi_label(); self.dd_lbl = self._kpi_label()
        risk_layout.addWidget(QtWidgets.QLabel("Unrealized PnL"), 0,0); risk_layout.addWidget(self.unreal_lbl, 0,1)
        risk_layout.addWidget(QtWidgets.QLabel("Realized PnL"),   1,0); risk_layout.addWidget(self.realized_lbl, 1,1)
        risk_layout.addWidget(QtWidgets.QLabel("Drawdown (max)"), 2,0); risk_layout.addWidget(self.dd_lbl, 2,1)

        # Positions table
        self.pos_table = QtWidgets.QTableView(); self.pos_table.setAlternatingRowColors(True)
        self.pos_table.setSortingEnabled(True); self.pos_table.horizontalHeader().setStretchLastSection(True)
        self.pos_table.verticalHeader().setVisible(False); self.pos_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.pos_model = SymbolsTableModel([]); self.pos_table.setModel(self.pos_model)

        # Charts
        self.equity_plot = pg.PlotWidget(); self.equity_plot.setTitle("Cumulative Equity"); self.equity_plot.showGrid(x=True, y=True, alpha=0.15)
        self._eq_x=[]; self._eq_y=[]; self.equity_curve = self.equity_plot.plot([], [], pen=pg.mkPen(width=2))
        self.realized_plot = pg.PlotWidget(); self.realized_plot.setTitle("Realized PnL (Daily & Cumulative)"); self.realized_plot.showGrid(x=True, y=True, alpha=0.15)
        self._rz_x=[]; self._rz_daily=[]; self._rz_cum=[]; self.rz_bars = pg.BarGraphItem(x=[], height=[], width=0.8, brush=(100,180,255,150)); self.realized_plot.addItem(self.rz_bars)
        self.rz_line = self.realized_plot.plot([], [], pen=pg.mkPen(255, 180, 0, width=2))

        left_col = QtWidgets.QVBoxLayout(); left_col.addWidget(risk_box); left_col.addWidget(self.pos_table)
        left_widget = QtWidgets.QWidget(); left_widget.setLayout(left_col)
        right_col = QtWidgets.QVBoxLayout(); right_col.addWidget(self.equity_plot, stretch=1); right_col.addWidget(self.realized_plot, stretch=1)
        right_widget = QtWidgets.QWidget(); right_widget.setLayout(right_col)

        dash_grid.addWidget(left_widget, 0,0, 2,1)
        dash_grid.addWidget(right_widget, 0,1, 2,2)
        self.tabs.addTab(dash, "Dashboard")

        # ORDERS TAB
        orders_tab = QtWidgets.QWidget(); orders_layout = QtWidgets.QVBoxLayout(orders_tab)
        self.orders_table = QtWidgets.QTableView(); self.orders_table.setAlternatingRowColors(True)
        self.orders_table.setSortingEnabled(True); self.orders_table.horizontalHeader().setStretchLastSection(True)
        self.orders_table.verticalHeader().setVisible(False); self.orders_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.orders_model = OrdersTableModel([]); self.orders_table.setModel(self.orders_model)
        orders_layout.addWidget(self.orders_table)
        self.tabs.addTab(orders_tab, "Orders")

        # TRADES TAB
        trades_tab = QtWidgets.QWidget(); trades_layout = QtWidgets.QVBoxLayout(trades_tab)
        self.trades_table = QtWidgets.QTableView(); self.trades_table.setAlternatingRowColors(True)
        self.trades_table.setSortingEnabled(True); self.trades_table.horizontalHeader().setStretchLastSection(True)
        self.trades_table.verticalHeader().setVisible(False); self.trades_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.trades_model = TradesTableModel([]); self.trades_table.setModel(self.trades_model)
        trades_layout.addWidget(self.trades_table)
        self.tabs.addTab(trades_tab, "Trades")

        # LOGS TAB
        logs_tab = QtWidgets.QWidget(); logs_layout = QtWidgets.QVBoxLayout(logs_tab)
        self.logs_view = QtWidgets.QPlainTextEdit(); self.logs_view.setReadOnly(True)
        self.logs_view.setMaximumBlockCount(5000)
        logs_layout.addWidget(self.logs_view)
        self.tabs.addTab(logs_tab, "Logs")

        # Wiring toolbar actions (start/stop just enable/disable feeder externally)
        # Actual engine control commands can be connected here.
        self.start_btn.triggered.connect(lambda: self._append_log("[UI] Start clicked"))
        self.stop_btn.triggered.connect(lambda: self._append_log("[UI] Stop clicked"))
        self.clear_logs_btn.triggered.connect(self.logs_view.clear)

    # -------- slots (connected in main()) --------
    @QtCore.Slot(list)
    def on_symbols(self, rows: List[Dict]): self.pos_model.replace_rows(rows)

    @QtCore.Slot(list)
    def on_orders(self, rows: List[Dict]): self.orders_model.replace_rows(rows)

    @QtCore.Slot(list)
    def on_trades(self, rows: List[Dict]): self.trades_model.replace_rows(rows)

    @QtCore.Slot(str, float)
    def on_equity_point(self, date_str: str, equity: float):
        self._eq_x.append(len(self._eq_x)); self._eq_y.append(equity)
        self.equity_curve.setData(self._eq_x, self._eq_y)

    @QtCore.Slot(str, float, float)
    def on_realized_point(self, date_str: str, daily: float, cum: float):
        idx = len(self._rz_x); self._rz_x.append(idx); self._rz_daily.append(daily); self._rz_cum.append(cum)
        self.rz_bars.setOpts(x=self._rz_x, height=self._rz_daily); self.rz_line.setData(self._rz_x, self._rz_cum)

    @QtCore.Slot(float, float, float)
    def on_risk_stats(self, total_unreal: float, total_realized: float, max_dd: float):
        self._set_kpi(self.unreal_lbl, total_unreal, money=True)
        self._set_kpi(self.realized_lbl, total_realized, money=True)
        self._set_kpi(self.dd_lbl, max_dd, pct=True)

    @QtCore.Slot(str)
    def on_log(self, line: str): self._append_log(line)

    # -------- helpers --------
    def _kpi_label(self) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel("--"); lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lbl.setStyleSheet("font-weight:700; font-size:16px; color:#e5e5e5;"); return lbl

    def _set_kpi(self, lbl: QtWidgets.QLabel, val: float, money: bool=False, pct: bool=False):
        color = "#e5e5e5"
        if pct: text = f"{val*100:.2f}%"; color = "#22c55e" if val >= 0 else "#f87171"
        elif money: text = f"{val:,.2f}"; color = "#22c55e" if val >= 0 else "#f87171"
        else: text = f"{val}"
        lbl.setText(text); lbl.setStyleSheet(f"font-weight:700; font-size:16px; color:{color};")

    def _append_log(self, text: str):
        self.logs_view.appendPlainText(text)
        self.logs_view.verticalScrollBar().setValue(self.logs_view.verticalScrollBar().maximum())

# ===============================
# APP ENTRY
# ===============================

def main():
    app = QtWidgets.QApplication(sys.argv)
    apply_dark_palette(app)

    win = MainWindow(); win.show()

    feeder = DataFeeder()
    feeder.s.symbols.connect(win.on_symbols)
    feeder.s.orders.connect(win.on_orders)
    feeder.s.trades.connect(win.on_trades)
    feeder.s.equity_point.connect(win.on_equity_point)
    feeder.s.realized_point.connect(win.on_realized_point)
    feeder.s.risk_stats.connect(win.on_risk_stats)
    feeder.s.log.connect(win.on_log)
    feeder.start()

    rc = app.exec()
    feeder.stop(); feeder.wait(1500)
    sys.exit(rc)

if __name__ == "__main__":
    main()
