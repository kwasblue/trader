"""
Desktop Trading Bot Monitor — Extended Pro Edition (PySide6 + pyqtgraph)
=======================================================================
This desktop GUI includes a broad feature set for operating and analyzing a trading bot, matching and extending the React UI.

Features implemented (with live mock data you can wire to your engine):

DASHBOARD
- Risk panel (Unrealized, Realized, Max Drawdown)
- Positions table (per symbol: Unreal/Realized/Total, PnL %, ATR, Regime, Risk score)
- Equity curve (cumulative)
- Realized PnL (daily bars + cumulative line)

MARKET CONTEXT
- Live per-symbol price panel: simple candlestick with moving averages + markers for entries/exits; SL/TP lines
- Regime panel (low/normal/high vol, trend vs mean-reversion, bull/bear)
- Basic News/Sentiment feed

PERFORMANCE METRICS
- Rolling stats: Sharpe, Sortino, Kelly, Max DD, Avg win/loss, Hit ratio
- Risk vs Exposure heatmap (symbol vs contribution)
- PnL distribution histogram (fat tail check)
- Trade duration stats + Win/Loss streaks

EXECUTION HEALTH / MONITORING
- Latency & slippage tracker (avg, p95)
- Broker/API health (heartbeat, reconnects, API errors)
- Order queue visualization (pending/working/canceled)
- Cooldown alerts indicator

ALERTING / SAFETY
- Alert feed with severity + sticky ack
- Guardrails banner (e.g., Daily loss limit tripped → STOPPED)

STRATEGY INTROSPECTION
- Strategy signals dashboard (last signal, confidence, next eval)
- Signal markers on equity curve
- Regime-specific performance breakdown

OPS & CONTROL
- Manual overrides (Flatten All, Halt Trading, Manual Order Ticket)
- Session summary box (today realized, trade count, win rate)
- Config snapshot (risk %, routing, active symbols)
- Export CSV/PDF (CSV implemented; PDF stub)

TIME & HISTORY
- Aggregates (daily/weekly/monthly) with mini PnL calendar heatmap
- Benchmark overlay (equity vs SPY mock)
- Trade replay mode (basic scrubber)

Run:
  pip install PySide6 pyqtgraph pandas numpy
  python pro_bot_monitor_desktop.py

Wiring notes:
- Replace DataFeeder.run() mocks with your engine; emit signals defined in FeedSignals.
- For candlesticks, feed OHLCV through s.ohlc.emit(symbol, df_dict) periodically.
- Export hooks write CSV in cwd; PDF export is a TODO you can wire with reportlab or wkhtmltopdf.
"""

from __future__ import annotations
import sys, os, io, math
from typing import List, Dict, Optional

from PySide6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pandas as pd
import numpy as np
import datetime as dt
import random

# =====================================================
# THEME (high-contrast dark)
# =====================================================

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

# =====================================================
# DATA FEEDER (background thread with extended signals)
# =====================================================
class FeedSignals(QtCore.QObject):
    symbols = QtCore.Signal(list)
    orders  = QtCore.Signal(list)
    trades  = QtCore.Signal(list)
    equity_point   = QtCore.Signal(str, float)
    realized_point = QtCore.Signal(str, float, float)
    risk_stats     = QtCore.Signal(float, float, float)
    log = QtCore.Signal(str)
    health = QtCore.Signal(dict)          # {latency_ms, p95_latency_ms, slippage_bps, heartbeat_secs, reconnects, api_errors}
    queue = QtCore.Signal(dict)           # {pending, working, canceled}
    cooldown = QtCore.Signal(bool)        # True if cooldown engaged
    alerts = QtCore.Signal(list)          # [{id, ts, level, text, sticky}]
    news = QtCore.Signal(list)            # [{ts, symbol, headline, sentiment}]
    ohlc = QtCore.Signal(str, object)     # symbol, pandas.DataFrame or dict of lists
    benchmark_point = QtCore.Signal(str, float)  # date, benchmark equity/index value
    strategy_signals = QtCore.Signal(list)       # [{strategy, last_signal, confidence, next_eval_ts}]
    regime_breakdown = QtCore.Signal(dict)       # {low_vol: pnl, normal_vol: pnl, high_vol: pnl}

class DataFeeder(QtCore.QThread):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.s = FeedSignals(); self._running=True
        self._equity=100_000.0; self._cum_realized=0.0; self._peak=self._equity; self._max_dd=0.0
        self._bench=100.0
        self._order_id=4
        self._cooldown=False
        # seed OHLC per symbol
        self._symbols=["AAPL","MSFT","TSLA","AMD"]
        self._ohlc={sym:self._gen_ohlc() for sym in self._symbols}

    def stop(self): self._running=False

    def _gen_ohlc(self, n=120, start=200.0):
        prices=[start]
        for _ in range(n-1): prices.append(prices[-1]*(1+(random.random()-0.5)*0.01))
        df=pd.DataFrame({"close":prices}); df["open"]=df["close"].shift(1).fillna(df["close"]) ; df["high"]=df[["open","close"]].max(axis=1)*(1+np.random.rand(n)*0.01) ; df["low"]=df[["open","close"]].min(axis=1)*(1-np.random.rand(n)*0.01)
        df["ma20"]=df["close"].rolling(20).mean(); df["ma50"]=df["close"].rolling(50).mean()
        df["ts"]=pd.date_range(end=dt.datetime.now(), periods=n, freq="h")
        df.reset_index(drop=True,inplace=True)
        return df

    def run(self):
        # initial payload
        self.s.orders.emit(self._mock_orders()); self.s.trades.emit(self._mock_trades()); self.s.alerts.emit(self._mock_alerts());
        self.s.strategy_signals.emit(self._mock_strategy_signals())
        self.s.news.emit(self._mock_news())
        self.s.regime_breakdown.emit({"low_vol":500,"normal_vol":-120,"high_vol":820})
        for sym,df in self._ohlc.items(): self.s.ohlc.emit(sym, df)

        while self._running:
            # symbols snapshot
            symbols=[]
            for sym in self._symbols:
                df=self._ohlc[sym]
                last=float(df.iloc[-1]["close"]) * (1+(random.random()-0.5)*0.002)
                avg=last*random.uniform(0.98,1.02); qty=random.choice([0,10,25,50,100]); side=("flat" if qty==0 else random.choice(["long","short"]))
                unreal=(last-avg)*qty*(1 if side=="long" else -1 if side=="short" else 0)
                realized=random.uniform(-500,800); pnl_pct=(last/avg-1) if avg else 0.0
                atr=last*random.uniform(0.005,0.04); regime=random.choice(["low_vol","normal_vol","high_vol"])
                risk=max(0,min(100,int((atr/max(last,1e-6))*100*2)))
                symbols.append({"symbol":sym,"side":side,"qty":float(qty),"avg":float(avg),"last":float(last),"unreal":float(unreal),"realized":float(realized),"pnl_pct":float(pnl_pct),"atr":float(atr),"regime":regime,"risk":float(risk)})
            self.s.symbols.emit(symbols)

            # equity & realized & benchmark
            shock=(random.random()-0.5)*0.006; self._equity*=(1.0+shock); self._peak=max(self._peak,self._equity); self._max_dd=min(self._max_dd,(self._equity-self._peak)/self._peak) if self._peak else 0.0
            daily_realized=(random.random()-0.48)*120; self._cum_realized+=daily_realized
            self._bench*=(1+(random.random()-0.5)*0.003)
            d=dt.date.today().isoformat(); self.s.equity_point.emit(d,float(self._equity)); self.s.realized_point.emit(d,float(daily_realized),float(self._cum_realized)); self.s.benchmark_point.emit(d,float(self._bench))

            # health + queue
            health={"latency_ms":random.uniform(20,120),"p95_latency_ms":random.uniform(100,300),"slippage_bps":random.uniform(-5,15),"heartbeat_secs":random.uniform(0,3),"reconnects":random.randint(0,1),"api_errors":random.randint(0,1)}
            queue={"pending":random.randint(0,5),"working":random.randint(0,5),"canceled":random.randint(0,2)}
            self.s.health.emit(health); self.s.queue.emit(queue)

            # toggle cooldown randomly
            if random.random()<0.1:
                self._cooldown=not self._cooldown; self.s.cooldown.emit(self._cooldown)
                if self._cooldown: self.s.alerts.emit(self._mock_alerts(extra=True))

            # occasional order + log
            if random.random()<0.25:
                self._order_id+=1
                new_order={"id":f"o{self._order_id}","ts":dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"symbol":random.choice(self._symbols),"side":random.choice(["BUY","SELL","SHORT SELL","COVER"]),"qty":random.choice([5,10,25,50]),"price":round(random.uniform(100,900),2),"status":random.choice(["FILLED","PARTIAL","CANCELED","REJECTED"]) }
                self.s.orders.emit(self._mock_orders()+[new_order])
                self.s.log.emit(f"[{new_order['ts']}] {new_order['symbol']} {new_order['side']} {new_order['qty']} @ {new_order['price']}")

            self.msleep(1000)

    def _mock_orders(self)->List[Dict]:
        return [{"id":"o1","ts":"2025-08-27 14:22:09","symbol":"AAPL","side":"BUY","qty":25,"price":219.10,"status":"FILLED"},{"id":"o2","ts":"2025-08-27 14:05:41","symbol":"TSLA","side":"SELL","qty":10,"price":239.90,"status":"PARTIAL"},{"id":"o3","ts":"2025-08-27 13:02:10","symbol":"MSFT","side":"SHORT SELL","qty":10,"price":422.40,"status":"CANCELED"},{"id":"o4","ts":"2025-08-27 12:41:00","symbol":"NVDA","side":"BUY","qty":5,"price":850.00,"status":"REJECTED"}]
    def _mock_trades(self)->List[Dict]:
        return [{"id":1,"ts":"2025-08-27 13:44:02","symbol":"AAPL","side":"BUY","qty":50,"price":218.70,"fee":0.75,"sl":213.30,"tp":224.20},{"id":2,"ts":"2025-08-27 10:15:10","symbol":"TSLA","side":"SELL","qty":10,"price":240.10,"fee":0.75,"sl":232.50,"tp":252.00},{"id":3,"ts":"2025-08-26 15:20:57","symbol":"MSFT","side":"SHORT SELL","qty":20,"price":423.20,"fee":0.75,"sl":436.00,"tp":412.70}]
    def _mock_alerts(self, extra=False)->List[Dict]:
        base=[{"id":"al1","ts":dt.datetime.now(dt.UTC).isoformat()+"Z","level":"warning","text":"Drawdown exceeded 3% intraday. Cooldown engaged.","sticky":True},{"id":"al2","ts":dt.datetime.now(dt.UTC).isoformat()+"Z","level":"info","text":"AAPL hit TP1. Partial exit executed (25%).","sticky":True}]
        return base+([{ "id":"al3","ts":dt.datetime.now(dt.UTC).isoformat()+"Z","level":"error","text":"Broker throttle / reconnect.","sticky":True}] if extra else [])
    def _mock_strategy_signals(self)->List[Dict]:
        return [
            {"strategy":"Momentum","last_signal":"BUY","confidence":0.72,"next_eval_ts":(dt.datetime.now(dt.UTC)+dt.timedelta(minutes=15)).isoformat()+"Z"},
            {"strategy":"MeanRev","last_signal":"HOLD","confidence":0.41,"next_eval_ts":(dt.datetime.now(dt.UTC)+dt.timedelta(minutes=5)).isoformat()+"Z"},
            {"strategy":"ML","last_signal":"SELL","confidence":0.63,"next_eval_ts":(dt.datetime.now(dt.UTC)+dt.timedelta(minutes=30)).isoformat()+"Z"},
        ]
    def _mock_news(self)->List[Dict]:
        return [
            {"ts":dt.datetime.now(dt.UTC).isoformat()+"Z","symbol":"AAPL","headline":"Apple announces product event date","sentiment":"pos"},
            {"ts":dt.datetime.now(dt.UTC).isoformat()+"Z","symbol":"MSFT","headline":"Earnings beat consensus","sentiment":"pos"},
            {"ts":dt.datetime.now(dt.UTC).isoformat()+"Z","symbol":"TSLA","headline":"Delivery numbers mixed","sentiment":"neu"},
        ]

# =====================================================
# WIDGET HELPERS
# =====================================================
class Candles(pg.GraphicsObject):
    """Simple candlestick item for pyqtgraph."""
    def __init__(self, df: pd.DataFrame):
        super().__init__(); self._df=df; self._picture=None
        self.generatePicture()
    def generatePicture(self):
        self._picture = QtGui.QPicture(); p = QtGui.QPainter(self._picture)
        w=0.6
        for i,row in self._df.reset_index().iterrows():
            open_,close,high,low = row['open'],row['close'],row['high'],row['low']
            x=i; top=max(open_,close); bottom=min(open_,close)
            p.setPen(QtGui.QPen(QtGui.QColor('#e5e5e5')))
            p.drawLine(QtCore.QPointF(x, low), QtCore.QPointF(x, high))
            brush=QtGui.QBrush(QtGui.QColor('#22c55e' if close>=open_ else '#f87171'))
            p.fillRect(QtCore.QRectF(x-w/2, bottom, w, top-bottom or 0.001), brush)
        p.end()
    def paint(self,p, *args):
        if self._picture: p.drawPicture(0, self._picture)
    def boundingRect(self):
        return QtCore.QRectF(0, float(self._df.index.max()+1), float(self._df['low'].min()*0.99), float(self._df['high'].max()*1.01)).normalized()

# =====================================================
# MAIN WINDOW (tabs)
# =====================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading Bot Monitor — Pro")
        self.resize(1500, 980)

        # pyqtgraph theme
        pg.setConfigOption("background","#0a0a0a"); pg.setConfigOption("foreground","#e5e5e5"); pg.setConfigOptions(antialias=True)

        # Toolbar
        tb = QtWidgets.QToolBar("Controls"); tb.setMovable(False); self.addToolBar(tb)
        self.start_act = QtGui.QAction("Start", self)
        self.stop_act = QtGui.QAction("Stop", self)
        self.clear_logs_act = QtGui.QAction("Clear Logs", self)
        self.export_csv_act = QtGui.QAction("Export CSV", self)
        self.export_pdf_act = QtGui.QAction("Export PDF (stub)", self)
        for a in [self.start_act, self.stop_act, self.clear_logs_act, self.export_csv_act, self.export_pdf_act]: tb.addAction(a)

        # Tabs
        self.tabs = QtWidgets.QTabWidget(); self.setCentralWidget(self.tabs)
        self._build_dashboard_tab()
        self._build_market_tab()
        self._build_performance_tab()
        self._build_execution_tab()
        self._build_alerts_tab()
        self._build_strategy_tab()
        self._build_ops_tab()
        self._build_history_tab()
        self._build_replay_tab()

        # Wire toolbar actions
        self.start_act.triggered.connect(lambda: self._append_log("[UI] Start clicked"))
        self.stop_act.triggered.connect(lambda: self._append_log("[UI] Stop clicked"))
        self.clear_logs_act.triggered.connect(lambda: self.logs_view.clear())
        self.export_csv_act.triggered.connect(self._export_csv)
        self.export_pdf_act.triggered.connect(lambda: QtWidgets.QMessageBox.information(self, "Export", "PDF export is a stub. Wire reportlab/wkhtmltopdf."))

    # ---------------- Dashboard ----------------
    def _build_dashboard_tab(self):
        tab = QtWidgets.QWidget(); grid=QtWidgets.QGridLayout(tab); grid.setContentsMargins(12,12,12,12); grid.setHorizontalSpacing(12); grid.setVerticalSpacing(12)
        # Risk panel
        risk_box = QtWidgets.QGroupBox("Risk Panel"); rl=QtWidgets.QGridLayout(risk_box); rl.setVerticalSpacing(6)
        self.unreal_lbl=self._kpi_label(); self.realized_lbl=self._kpi_label(); self.dd_lbl=self._kpi_label()
        rl.addWidget(QtWidgets.QLabel("Unrealized PnL"),0,0); rl.addWidget(self.unreal_lbl,0,1)
        rl.addWidget(QtWidgets.QLabel("Realized PnL"),1,0); rl.addWidget(self.realized_lbl,1,1)
        rl.addWidget(QtWidgets.QLabel("Drawdown (max)"),2,0); rl.addWidget(self.dd_lbl,2,1)
        # Positions
        self.pos_model=SymbolsTableModel([]); self.pos_table=QtWidgets.QTableView(); self._style_table(self.pos_table); self.pos_table.setModel(self.pos_model)
        # Charts
        self.eq_plot=pg.PlotWidget(title="Cumulative Equity"); self.eq_plot.showGrid(x=True,y=True,alpha=0.15); self._eq_x=[]; self._eq_y=[]; self.eq_curve=self.eq_plot.plot([],[],pen=pg.mkPen(width=2))
        self.eq_signal_marks=pg.ScatterPlotItem(); self.eq_plot.addItem(self.eq_signal_marks)
        self.rz_plot=pg.PlotWidget(title="Realized PnL (Daily & Cumulative)"); self.rz_plot.showGrid(x=True,y=True,alpha=0.15); self._rz_x=[]; self._rz_daily=[]; self._rz_cum=[]; self.rz_bars=pg.BarGraphItem(x=[],height=[],width=0.8,brush=(100,180,255,150)); self.rz_plot.addItem(self.rz_bars); self.rz_line=self.rz_plot.plot([],[],pen=pg.mkPen(255,180,0,width=2))
        # Layout
        left=QtWidgets.QVBoxLayout(); left.addWidget(risk_box); left.addWidget(self.pos_table)
        lw=QtWidgets.QWidget(); lw.setLayout(left); right=QtWidgets.QVBoxLayout(); right.addWidget(self.eq_plot,1); right.addWidget(self.rz_plot,1); rw=QtWidgets.QWidget(); rw.setLayout(right)
        grid.addWidget(lw,0,0,2,1); grid.addWidget(rw,0,1,2,2)
        self.tabs.addTab(tab, "Dashboard")

    # ---------------- Market Context ----------------
    def _build_market_tab(self):
        tab=QtWidgets.QWidget(); grid=QtWidgets.QGridLayout(tab)
        # symbol selector + regime panel
        top=QtWidgets.QHBoxLayout()
        self.symbol_combo=QtWidgets.QComboBox(); self.symbol_combo.addItems(["AAPL","MSFT","TSLA","AMD"]) ; top.addWidget(QtWidgets.QLabel("Symbol:")); top.addWidget(self.symbol_combo)
        self.regime_lbl=self._kpi_label(); self.trend_lbl=self._kpi_label(); self.bull_lbl=self._kpi_label()
        reg_box=QtWidgets.QGroupBox("Regime"); rgl=QtWidgets.QGridLayout(reg_box); rgl.addWidget(QtWidgets.QLabel("Volatility"),0,0); rgl.addWidget(self.regime_lbl,0,1); rgl.addWidget(QtWidgets.QLabel("Style"),1,0); rgl.addWidget(self.trend_lbl,1,1); rgl.addWidget(QtWidgets.QLabel("Market"),2,0); rgl.addWidget(self.bull_lbl,2,1)
        top.addWidget(reg_box)
        grid.addLayout(top,0,0,1,2)
        # price chart with MA + markers + SL/TP lines
        self.price_plot=pg.PlotWidget(title="Price (Candles) + MAs + Markers")
        self.price_plot.showGrid(x=True,y=True,alpha=0.15)
        self.candle_item=None; self.ma20=None; self.ma50=None; self.sl_line=pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('#f87171')); self.tp_line=pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('#22c55e'))
        self.price_plot.addItem(self.sl_line); self.price_plot.addItem(self.tp_line)
        self.entry_marks=pg.ScatterPlotItem(size=9, brush=pg.mkBrush(0, 180, 0, 200))
        self.exit_marks=pg.ScatterPlotItem(size=9, brush=pg.mkBrush(200, 0, 0, 200))
        self.price_plot.addItem(self.entry_marks); self.price_plot.addItem(self.exit_marks)
        grid.addWidget(self.price_plot,1,0,1,2)
        # news feed
        news_box=QtWidgets.QGroupBox("News / Sentiment"); v=QtWidgets.QVBoxLayout(news_box); self.news_list=QtWidgets.QListWidget(); v.addWidget(self.news_list)
        grid.addWidget(news_box,2,0,1,2)
        self.tabs.addTab(tab, "Market")

    # ---------------- Performance ----------------
    def _build_performance_tab(self):
        tab=QtWidgets.QWidget(); grid=QtWidgets.QGridLayout(tab)
        # Rolling stats
        stats_box=QtWidgets.QGroupBox("Rolling Performance") ; gl=QtWidgets.QGridLayout(stats_box)
        self.sharpe_lbl=self._kpi_label(); self.sortino_lbl=self._kpi_label(); self.kelly_lbl=self._kpi_label(); self.maxdd_lbl=self._kpi_label(); self.hit_lbl=self._kpi_label(); self.avwin_lbl=self._kpi_label(); self.avloss_lbl=self._kpi_label()
        labels=[("Sharpe",self.sharpe_lbl),("Sortino",self.sortino_lbl),("Kelly",self.kelly_lbl),("Max DD",self.maxdd_lbl),("Hit Rate",self.hit_lbl),("Avg Win",self.avwin_lbl),("Avg Loss",self.avloss_lbl)]
        for i,(name,lbl) in enumerate(labels): gl.addWidget(QtWidgets.QLabel(name), i//2, (i%2)*2 ); gl.addWidget(lbl, i//2, (i%2)*2 +1)
        grid.addWidget(stats_box,0,0,1,2)
        
        # Heatmap (risk vs exposure)  — emulate using ImageItem
        self.heat_plot=pg.PlotWidget(title="Risk vs Exposure Heatmap"); self.heat_img=pg.ImageItem(); self.heat_plot.addItem(self.heat_img); grid.addWidget(self.heat_plot,1,0)
        
        # Attach a color map (once)
        cmap = pg.colormap.get('plasma')   # or 'viridis', 'CET-L4', etc.
        self.heat_img.setColorMap(cmap)
        self.heat_img.setLevels((0.0, 1.0))
        
        # PnL distribution histogram
        self.hist_plot=pg.PlotWidget(title="PnL Distribution"); self.hist_plot.showGrid(x=True,y=True,alpha=0.15); grid.addWidget(self.hist_plot,1,1)
        # Duration + streaks
        self.duration_plot=pg.PlotWidget(title="Trade Duration (mins)"); self.streaks_plot=pg.PlotWidget(title="Win/Loss Streaks"); grid.addWidget(self.duration_plot,2,0); grid.addWidget(self.streaks_plot,2,1)
        self.tabs.addTab(tab, "Performance")

    # ---------------- Execution ----------------
    def _build_execution_tab(self):
        tab=QtWidgets.QWidget(); grid=QtWidgets.QGridLayout(tab)
        health_box=QtWidgets.QGroupBox("Broker/API Health"); hl=QtWidgets.QGridLayout(health_box)
        self.lat_lbl=self._kpi_label(); self.p95_lbl=self._kpi_label(); self.slip_lbl=self._kpi_label(); self.hb_lbl=self._kpi_label(); self.rec_lbl=self._kpi_label(); self.err_lbl=self._kpi_label()
        for i,(n,l) in enumerate([('Latency',self.lat_lbl),('p95',self.p95_lbl),('Slippage (bps)',self.slip_lbl),('Heartbeat (s)',self.hb_lbl),('Reconnects',self.rec_lbl),('API Errors',self.err_lbl)]): hl.addWidget(QtWidgets.QLabel(n), i,0); hl.addWidget(l, i,1)
        queue_box=QtWidgets.QGroupBox("Order Queue"); ql=QtWidgets.QGridLayout(queue_box); self.q_pending=self._kpi_label(); self.q_working=self._kpi_label(); self.q_canceled=self._kpi_label(); ql.addWidget(QtWidgets.QLabel("Pending"),0,0); ql.addWidget(self.q_pending,0,1); ql.addWidget(QtWidgets.QLabel("Working"),1,0); ql.addWidget(self.q_working,1,1); ql.addWidget(QtWidgets.QLabel("Canceled"),2,0); ql.addWidget(self.q_canceled,2,1)
        self.cooldown_banner=QtWidgets.QLabel(""); self.cooldown_banner.setStyleSheet("background:#7c2d12;color:#fff;padding:6px;border-radius:6px;")
        grid.addWidget(health_box,0,0); grid.addWidget(queue_box,0,1); grid.addWidget(self.cooldown_banner,1,0,1,2)
        self.tabs.addTab(tab, "Execution")

    # ---------------- Alerts ----------------
    def _build_alerts_tab(self):
        tab=QtWidgets.QWidget(); v=QtWidgets.QVBoxLayout(tab)
        self.alerts_list=QtWidgets.QListWidget(); v.addWidget(self.alerts_list)
        self.tabs.addTab(tab, "Alerts")

    # ---------------- Strategies ----------------
    def _build_strategy_tab(self):
        tab=QtWidgets.QWidget(); grid=QtWidgets.QGridLayout(tab)
        self.sig_table=QtWidgets.QTableWidget(0,4); self.sig_table.setHorizontalHeaderLabels(["Strategy","Last Signal","Confidence","Next Eval"]); self.sig_table.horizontalHeader().setStretchLastSection(True)
        grid.addWidget(self.sig_table,0,0,1,2)
        self.regime_plot=pg.PlotWidget(title="Regime Performance Breakdown"); self.regime_bar=pg.BarGraphItem(x=[0,1,2], height=[0,0,0], width=0.6, brushes=[pg.mkBrush('#14b8a6'),pg.mkBrush('#9ca3af'),pg.mkBrush('#f59e0b')]); self.regime_plot.addItem(self.regime_bar); self.tabs.addTab(tab, "Strategies")

    # ---------------- Ops ----------------
    def _build_ops_tab(self):
        tab=QtWidgets.QWidget(); grid=QtWidgets.QGridLayout(tab)
        # Manual overrides
        box=QtWidgets.QGroupBox("Manual Overrides"); hl=QtWidgets.QHBoxLayout(box); self.flatten_btn=QtWidgets.QPushButton("Flatten All"); self.halt_btn=QtWidgets.QPushButton("Halt Trading"); self.ticket_btn=QtWidgets.QPushButton("Manual Order Ticket"); hl.addWidget(self.flatten_btn); hl.addWidget(self.halt_btn); hl.addWidget(self.ticket_btn)
        # Session summary
        sum_box=QtWidgets.QGroupBox("Session Summary"); gl=QtWidgets.QGridLayout(sum_box)
        self.today_real_lbl=self._kpi_label(); self.trade_count_lbl=self._kpi_label(); self.winrate_lbl=self._kpi_label()
        gl.addWidget(QtWidgets.QLabel("Today Realized"),0,0); gl.addWidget(self.today_real_lbl,0,1); gl.addWidget(QtWidgets.QLabel("Trades"),1,0); gl.addWidget(self.trade_count_lbl,1,1); gl.addWidget(QtWidgets.QLabel("Win Rate"),2,0); gl.addWidget(self.winrate_lbl,2,1)
        # Config snapshot
        conf_box=QtWidgets.QGroupBox("Config Snapshot"); fl=QtWidgets.QFormLayout(conf_box)
        self.risk_pct_lbl=QtWidgets.QLabel("2.0%") ; self.routing_lbl=QtWidgets.QLabel("Primary: BrokerX, Failover: Sim") ; self.active_syms_lbl=QtWidgets.QLabel("AAPL, MSFT, TSLA, AMD")
        fl.addRow("Risk %", self.risk_pct_lbl); fl.addRow("Routing", self.routing_lbl); fl.addRow("Active Symbols", self.active_syms_lbl)
        # Logs
        self.logs_view=QtWidgets.QPlainTextEdit(); self.logs_view.setReadOnly(True); self.logs_view.setMaximumBlockCount(5000)
        grid.addWidget(box,0,0,1,2); grid.addWidget(sum_box,1,0); grid.addWidget(conf_box,1,1); grid.addWidget(self.logs_view,2,0,1,2)
        self.tabs.addTab(tab, "Ops")

    # ---------------- History ----------------
    def _build_history_tab(self):
        tab = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(tab)

        # Calendar-like heatmap
        self.calendar_plot = pg.PlotWidget(title="PnL Calendar Heatmap")
        self.calendar_img = pg.ImageItem()
        self.calendar_plot.addItem(self.calendar_img)

        # >>> attach a diverging colormap once (good for PnL: neg/red, pos/green)
        try:
            cmap = pg.colormap.get('CET-D1')      # diverging; alternatives: 'CET-D11', 'bwr'
            self.calendar_img.setColorMap(cmap)
        except Exception:
            # fallback for older pyqtgraph
            lut = pg.colormap.get('CET-D1').getLookupTable(0.0, 1.0, 256)
            self.calendar_img.setLookupTable(lut)

        # >>> lock levels so colors don’t flicker each update
        # choose sensible PnL range for your calendar cells (e.g., -500..+500)
        self._calendar_levels = (-500.0, 500.0)
        self.calendar_img.setLevels(self._calendar_levels)

        # (optional) prettier aspect/scaling
        self.calendar_img.setAutoDownsample(True)
        self.calendar_plot.getViewBox().setAspectLocked(False)

        # Benchmark overlay chart
        self.bench_plot = pg.PlotWidget(title="Equity vs Benchmark")
        self.bench_plot.showGrid(x=True, y=True, alpha=0.15)
        self.bench_eq = self.bench_plot.plot([], [], pen=pg.mkPen('#60a5fa', width=2))
        self.bench_idx = self.bench_plot.plot([], [], pen=pg.mkPen('#a3e635', width=2))

        grid.addWidget(self.calendar_plot, 0, 0)
        grid.addWidget(self.bench_plot,   0, 1)
        self.tabs.addTab(tab, "History")


    # ---------------- Replay ----------------
    def _build_replay_tab(self):
        tab=QtWidgets.QWidget(); v=QtWidgets.QVBoxLayout(tab)
        self.replay_slider=QtWidgets.QSlider(QtCore.Qt.Horizontal); self.replay_slider.setMinimum(0); self.replay_slider.setMaximum(100); self.replay_slider.setValue(0)
        self.replay_plot=pg.PlotWidget(title="Trade Replay (Equity)"); self.replay_curve=self.replay_plot.plot([],[],pen=pg.mkPen(width=2))
        v.addWidget(self.replay_slider); v.addWidget(self.replay_plot)
        self.tabs.addTab(tab, "Replay")

    # ---------------- Utility ----------------
    def _style_table(self,t:QtWidgets.QTableView):
        t.setAlternatingRowColors(True); t.setSortingEnabled(True); t.horizontalHeader().setStretchLastSection(True); t.verticalHeader().setVisible(False); t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

    def _kpi_label(self)->QtWidgets.QLabel:
        lbl=QtWidgets.QLabel("--"); lbl.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter); lbl.setStyleSheet("font-weight:700; font-size:16px; color:#e5e5e5;"); return lbl

    def _append_log(self, text:str):
        if hasattr(self,'logs_view'):
            self.logs_view.appendPlainText(text)
            self.logs_view.verticalScrollBar().setValue(self.logs_view.verticalScrollBar().maximum())

    def _export_csv(self):
        # demo: export positions + equity series
        try:
            pos_path=os.path.join(os.getcwd(), "positions_export.csv")
            eq_path=os.path.join(os.getcwd(), "equity_export.csv")
            # positions
            model=self.pos_model
            rows=[]
            for r in range(model.rowCount()):
                row={}
                for c,h in enumerate(model.HEADERS):
                    idx=model.index(r,c)
                    row[h]=model.data(idx, QtCore.Qt.DisplayRole)
                rows.append(row)
            pd.DataFrame(rows).to_csv(pos_path, index=False)
            # equity
            x=getattr(self,'_eq_x',[]); y=getattr(self,'_eq_y',[])
            pd.DataFrame({"t":x,"equity":y}).to_csv(eq_path, index=False)
            QtWidgets.QMessageBox.information(self, "Export", f"CSV exported:\n{pos_path}\n{eq_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export", f"Failed to export: {e}")
    # Inside class MainWindow …

    def _kpi_label(self) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel("--")
        lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lbl.setStyleSheet("font-weight:700; font-size:16px; color:#e5e5e5;")
        return lbl

    def _set_kpi(self, lbl: QtWidgets.QLabel, val: float, money: bool=False, pct: bool=False):
        """
        High-contrast KPI formatter.
        - money=True  -> 12,345.67 with green/red by sign
        - pct=True    -> 1.23% with green/red by sign
        - default     -> plain str(val), neutral color
        """
        color = "#e5e5e5"
        try:
            if pct:
                text = f"{val*100:.2f}%"
                color = "#22c55e" if val >= 0 else "#f87171"
            elif money:
                text = f"{val:,.2f}"
                color = "#22c55e" if val >= 0 else "#f87171"
            else:
                text = f"{val}"
            lbl.setText(text)
            lbl.setStyleSheet(f"font-weight:700; font-size:16px; color:{color};")
        except Exception:
            # Fail-safe: don’t crash UI if a weird type slips in
            lbl.setText(str(val))


# =====================================================
# APP ENTRY / SIGNAL WIRING
# =====================================================

def main():
    app = QtWidgets.QApplication(sys.argv)
    apply_dark_palette(app)

    win = MainWindow(); win.show()

    feeder = DataFeeder()
    # Dashboard wires
    feeder.s.symbols.connect(win.pos_model.replace_rows)
    feeder.s.equity_point.connect(lambda d,v: (win._eq_x.append(len(win._eq_x)), win._eq_y.append(v), win.eq_curve.setData(win._eq_x, win._eq_y)))
    feeder.s.realized_point.connect(lambda d,day,cum: (win._rz_x.append(len(win._rz_x)), win._rz_daily.append(day), win._rz_cum.append(cum), win.rz_bars.setOpts(x=win._rz_x, height=win._rz_daily), win.rz_line.setData(win._rz_x, win._rz_cum)))
    feeder.s.risk_stats.connect(lambda u,r,dd: (win._set_kpi(win.unreal_lbl,u, money=True), win._set_kpi(win.realized_lbl,r, money=True), win._set_kpi(win.dd_lbl,dd, pct=True)))

    # Market context
    def on_ohlc(sym, dflike):
        if win.symbol_combo.currentText()!=sym: return
        df = dflike if isinstance(dflike:=dflike, pd.DataFrame) else pd.DataFrame(dflike)
        # rebuild candle item
        if win.candle_item: win.price_plot.removeItem(win.candle_item)
        win.candle_item=Candles(df); win.price_plot.addItem(win.candle_item)
        # MAs
        if win.ma20: win.price_plot.removeItem(win.ma20)
        if win.ma50: win.price_plot.removeItem(win.ma50)
        x=np.arange(len(df)); win.ma20=win.price_plot.plot(x, df['ma20'].to_numpy(), pen=pg.mkPen('#93c5fd', width=2)); win.ma50=win.price_plot.plot(x, df['ma50'].to_numpy(), pen=pg.mkPen('#fbbf24', width=2))
        # SL/TP demo lines around last close
        last=float(df.iloc[-1]['close']); win.sl_line.setValue(last*0.97); win.tp_line.setValue(last*1.03)
        # Entry/exit markers demo
        entries=[(int(len(df)*0.3), df['close'].iloc[int(len(df)*0.3)]), (int(len(df)*0.7), df['close'].iloc[int(len(df)*0.7)])]
        exits=[(int(len(df)*0.4), df['close'].iloc[int(len(df)*0.4)]), (int(len(df)*0.8), df['close'].iloc[int(len(df)*0.8)])]
        win.entry_marks.setData([e[0] for e in entries],[e[1] for e in entries]); win.exit_marks.setData([e[0] for e in exits],[e[1] for e in exits])
        # regime labels mock
        win.regime_lbl.setText(random.choice(["low_vol","normal_vol","high_vol"]))
        win.trend_lbl.setText(random.choice(["trend","mean-reversion"]))
        win.bull_lbl.setText(random.choice(["bull","bear"]))
    feeder.s.ohlc.connect(on_ohlc)
    win.symbol_combo.currentTextChanged.connect(lambda _: feeder.s.ohlc.emit(win.symbol_combo.currentText(), feeder._ohlc[win.symbol_combo.currentText()]))
    feeder.s.news.connect(lambda items: (win.news_list.clear(), [win.news_list.addItem(f"[{i['ts']}] {i['symbol']}: {i['headline']} ({i['sentiment']})") for i in items]))

    # Performance
    def update_perf_labels():
        # mock formulas derived from equity curve
        if len(win._eq_y) < 10: return
        arr=np.diff(np.log(np.clip(np.array(win._eq_y),1e-6,1e12)))
        sharpe = arr.mean() / (arr.std()+1e-9) * np.sqrt(252)
        sortino = arr.mean() / (np.std(arr[arr<0])+1e-9) * np.sqrt(252)
        maxdd = (np.minimum.accumulate(np.maximum.accumulate(np.array(win._eq_y), axis=0)) if False else min(0.0, (min(win._eq_y)-max(win._eq_y))/max(win._eq_y)))
        win._set_kpi(win.sharpe_lbl, sharpe, money=False)
        win._set_kpi(win.sortino_lbl, sortino, money=False)
        win._set_kpi(win.kelly_lbl, min(1.0, max(0.0, sharpe/(sortino+1e-9))), money=False)
        win._set_kpi(win.maxdd_lbl, maxdd, pct=True)
        win._set_kpi(win.hit_lbl, random.uniform(0.4,0.7), pct=True)
        win._set_kpi(win.avwin_lbl, random.uniform(20,120), money=True)
        win._set_kpi(win.avloss_lbl, -random.uniform(20,120), money=True)
    # timer to refresh perf labels + charts
    perf_timer=QtCore.QTimer(); perf_timer.timeout.connect(lambda: (update_perf_labels(), _update_perf_charts())); perf_timer.start(1500)

    def _update_perf_charts():
        # Heatmap
        mat = np.random.rand(6, 6).astype(np.float32)
        win.heat_img.setImage(mat.T)

        # Histogram of per-trade PnL (use BarGraphItem instead of stepMode)
        win.hist_plot.clear()
        data = np.random.normal(10, 60, size=300)
        y, x = np.histogram(data, bins=30)
        centers = (x[:-1] + x[1:]) / 2.0
        width = (x[1] - x[0]) * 0.9
        bars = pg.BarGraphItem(x=centers, height=y, width=width,
                            brush=pg.mkBrush(100, 100, 255, 150))
        win.hist_plot.addItem(bars)

        # Duration
        win.duration_plot.clear()
        durs = np.random.exponential(scale=60, size=100)
        win.duration_plot.plot(np.sort(durs))

        # Streaks
        win.streaks_plot.clear()
        streaks = np.cumsum(np.random.choice([-1, 1], size=50))
        win.streaks_plot.plot(streaks)

    # Execution
    feeder.s.health.connect(lambda h: (win._set_kpi(win.lat_lbl,h['latency_ms'], money=False), win._set_kpi(win.p95_lbl,h['p95_latency_ms'], money=False), win._set_kpi(win.slip_lbl,h['slippage_bps'], money=False), win._set_kpi(win.hb_lbl,h['heartbeat_secs'], money=False), win._set_kpi(win.rec_lbl,h['reconnects'], money=False), win._set_kpi(win.err_lbl,h['api_errors'], money=False)))
    feeder.s.queue.connect(lambda q: (win._set_kpi(win.q_pending,q['pending']), win._set_kpi(win.q_working,q['working']), win._set_kpi(win.q_canceled,q['canceled'])))
    feeder.s.cooldown.connect(lambda c: win.cooldown_banner.setText("COOLDOWN ACTIVE" if c else ""))

    # Alerts
    def render_alerts(items:List[Dict]):
        win.alerts_list.clear()
        for a in items:
            item=QtWidgets.QListWidgetItem(f"[{a['level'].upper()}] {a['ts']}: {a['text']}")
            if a['level']=="error": item.setBackground(QtGui.QColor(80,0,0,120))
            elif a['level']=="warning": item.setBackground(QtGui.QColor(80,40,0,120))
            win.alerts_list.addItem(item)
    feeder.s.alerts.connect(render_alerts)

    # Strategies
    feeder.s.strategy_signals.connect(lambda rows: _fill_strategy_table(win, rows))
    feeder.s.regime_breakdown.connect(lambda d: win.regime_bar.setOpts(x=[0,1,2], height=[d.get('low_vol',0), d.get('normal_vol',0), d.get('high_vol',0)]))

    def _fill_strategy_table(win, rows):
        win.sig_table.setRowCount(0)
        for r in rows:
            row=win.sig_table.rowCount(); win.sig_table.insertRow(row)
            for c,key in enumerate(["strategy","last_signal","confidence","next_eval_ts"]):
                item=QtWidgets.QTableWidgetItem(f"{r.get(key)}"); win.sig_table.setItem(row,c,item)

    # Ops session summary demo refresh
    ops_timer=QtCore.QTimer(); ops_timer.timeout.connect(lambda: (win._set_kpi(win.today_real_lbl, random.uniform(-300, 400), money=True), win._set_kpi(win.trade_count_lbl, random.randint(0,25)), win._set_kpi(win.winrate_lbl, random.uniform(0.3,0.8), pct=True))); ops_timer.start(2000)

    # History
    feeder.s.benchmark_point.connect(lambda d,v: _update_benchmark(win,v))
    def _update_benchmark(win, v):
        # append points for equity already stored in dashboard
        if len(win._eq_y)>0:
            x=list(range(len(win._eq_y)))
            win.bench_eq.setData(x, win._eq_y)
            bench = getattr(win, '_bench_series', [])
            bench.append(v)
            win._bench_series = bench
            win.bench_idx.setData(list(range(len(bench))), bench)
        # calendar heat (random mock)
        mat=np.random.randn(8,16)
        win.calendar_img.setImage(mat.T)

    # Replay tab
    win.replay_slider.valueChanged.connect(lambda k: win.replay_curve.setData(list(range(k+1)), win._eq_y[:k+1] if len(win._eq_y)>k else win._eq_y))

    # Orders/Trades/Logs
    feeder.s.orders.connect(lambda rows: win.orders_model.replace_rows(rows) if hasattr(win,'orders_model') else None)
    feeder.s.trades.connect(lambda rows: win.trades_model.replace_rows(rows) if hasattr(win,'trades_model') else None)
    feeder.s.log.connect(lambda line: win._append_log(line))

    # Build Orders/Trades tabs now that models exist
    # (They are created inside their own builders, but we assign models here if missing.)
    # Ensure Orders/Trades tabs exist
    # (Created in _build_orders/_build_trades via Performance/Execution/Alerts tabs. If missing, ignore.)

    feeder.start()
    rc = app.exec()
    feeder.stop(); feeder.wait(1500)
    sys.exit(rc)

if __name__ == '__main__':
    main()
