
from PySide6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import os
import pandas as pd
import numpy as np
import random
from ..bus import ControlBridge
from monitoring.models import SymbolsTableModel
from monitoring.widgets.candles import Candles
from monitoring.dialogs.manual_order import ManualOrderDialog
from core.events.eventhandler import get_event_handler
# =====================================================
# MAIN WINDOW (tabs)
# =====================================================
class MainWindowDemo(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading Bot Monitor — Pro")
        self.resize(1540, 1000)

        self.ctrl = ControlBridge(bus=get_event_handler())
        self._halted = False

        # pyqtgraph theme
        pg.setConfigOption("background","#0a0a0a"); pg.setConfigOption("foreground","#e5e5e5"); pg.setConfigOptions(antialias=True)

        # Toolbar
        tb = QtWidgets.QToolBar("Controls"); tb.setMovable(False); self.addToolBar(tb)
        self.start_act = QtGui.QAction("Start", self)
        self.stop_act = QtGui.QAction("Stop", self)
        self.clear_logs_act = QtGui.QAction("Clear Logs", self)
        self.export_csv_act = QtGui.QAction("Export CSV", self)
        self.export_pdf_act = QtGui.QAction("Export PDF (stub)", self)

        # NEW: Panic Switch (toggle), Flatten All, Cancel All, Manual Order
        self.panic_btn = QtWidgets.QToolButton()
        # Ensure banner exists before _style_panic touches it
        self.halt_banner = QtWidgets.QLabel("")
        self.halt_banner.setObjectName("haltBanner")
        self.halt_banner.setStyleSheet("background:#991b1b;color:#fff;padding:6px;border-radius:6px;")
        self.panic_btn.setCheckable(True)
        self.panic_btn.setText("HALT ✖")
        self.panic_btn.setToolTip("Panic / Kill Switch (Shift+Esc)")
        self._style_panic(False)

        self.flatten_btn_tb = QtWidgets.QToolButton(); self.flatten_btn_tb.setText("Flatten All"); self.flatten_btn_tb.setToolTip("Close all positions immediately")
        self.cancel_all_btn_tb = QtWidgets.QToolButton(); self.cancel_all_btn_tb.setText("Cancel All"); self.cancel_all_btn_tb.setToolTip("Cancel all working orders")
        self.manual_order_btn_tb = QtWidgets.QToolButton(); self.manual_order_btn_tb.setText("Manual Order")

        for a in [self.start_act, self.stop_act, self.clear_logs_act, self.export_csv_act, self.export_pdf_act]:
            tb.addAction(a)
        tb.addSeparator()
        tb.addWidget(self.panic_btn)
        tb.addWidget(self.flatten_btn_tb)
        tb.addWidget(self.cancel_all_btn_tb)
        tb.addWidget(self.manual_order_btn_tb)

        # Global shortcuts
        QtGui.QShortcut(QtGui.QKeySequence("Shift+Esc"), self, activated=self._toggle_panic)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+M"), self, activated=self._show_manual_order)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+L"), self, activated=self._confirm_flatten)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+K"), self, activated=self._confirm_cancel_all)

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

        # Wire new controls
        self.panic_btn.clicked.connect(self._toggle_panic)
        self.flatten_btn_tb.clicked.connect(self._confirm_flatten)
        self.cancel_all_btn_tb.clicked.connect(self._confirm_cancel_all)
        self.manual_order_btn_tb.clicked.connect(self._show_manual_order)

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
        # Context menu: per-symbol flatten
        self.pos_table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.pos_table.customContextMenuRequested.connect(self._on_pos_context)
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
        
        # Heatmap (risk vs exposure)
        self.heat_plot=pg.PlotWidget(title="Risk vs Exposure Heatmap"); self.heat_img=pg.ImageItem(); self.heat_plot.addItem(self.heat_img); grid.addWidget(self.heat_plot,1,0)
        cmap = pg.colormap.get('plasma')
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
        self.halt_banner=QtWidgets.QLabel(""); self.halt_banner.setStyleSheet("background:#991b1b;color:#fff;padding:6px;border-radius:6px;")
        grid.addWidget(health_box,0,0); grid.addWidget(queue_box,0,1); grid.addWidget(self.cooldown_banner,1,0,1,2); grid.addWidget(self.halt_banner,2,0,1,2)
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
        # Manual overrides (duplicated in toolbar, but here for discoverability)
        box=QtWidgets.QGroupBox("Manual Overrides"); hl=QtWidgets.QHBoxLayout(box)
        self.flatten_btn=QtWidgets.QPushButton("Flatten All")
        self.cancel_all_btn=QtWidgets.QPushButton("Cancel All")
        self.halt_btn=QtWidgets.QPushButton("Halt Trading")
        self.ticket_btn=QtWidgets.QPushButton("Manual Order Ticket")
        hl.addWidget(self.flatten_btn); hl.addWidget(self.cancel_all_btn); hl.addWidget(self.halt_btn); hl.addWidget(self.ticket_btn)
        self.flatten_btn.clicked.connect(self._confirm_flatten)
        self.cancel_all_btn.clicked.connect(self._confirm_cancel_all)
        self.halt_btn.clicked.connect(self._toggle_panic)
        self.ticket_btn.clicked.connect(self._show_manual_order)

        # Guardrails
        guard=QtWidgets.QGroupBox("Guardrails"); glr=QtWidgets.QGridLayout(guard)
        self.daily_loss_limit_spin=QtWidgets.QDoubleSpinBox(); self.daily_loss_limit_spin.setRange(0.0, 1_000_000.0); self.daily_loss_limit_spin.setDecimals(2); self.daily_loss_limit_spin.setValue(500.0)
        self.guard_status=QtWidgets.QLabel("Auto-halt when day PnL ≤ -$500.00")
        glr.addWidget(QtWidgets.QLabel("Daily Loss Limit ($)"),0,0); glr.addWidget(self.daily_loss_limit_spin,0,1); glr.addWidget(self.guard_status,1,0,1,2)

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
        grid.addWidget(box,0,0,1,2); grid.addWidget(guard,1,0,1,2); grid.addWidget(sum_box,2,0); grid.addWidget(conf_box,2,1); grid.addWidget(self.logs_view,3,0,1,2)
        self.tabs.addTab(tab, "Ops")

    # ---------------- History ----------------
    def _build_history_tab(self):
        tab = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(tab)

        # Calendar-like heatmap
        self.calendar_plot = pg.PlotWidget(title="PnL Calendar Heatmap")
        self.calendar_img = pg.ImageItem()
        self.calendar_plot.addItem(self.calendar_img)

        try:
            cmap = pg.colormap.get('CET-D1')
            self.calendar_img.setColorMap(cmap)
        except Exception:
            lut = pg.colormap.get('CET-D1').getLookupTable(0.0, 1.0, 256)
            self.calendar_img.setLookupTable(lut)

        self._calendar_levels = (-500.0, 500.0)
        self.calendar_img.setLevels(self._calendar_levels)
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

    # ---------------- Panic / Flatten / Manual Order handlers ----------------
    def _style_panic(self, halted: bool):
        if halted:
            self.panic_btn.setChecked(True)
            self.panic_btn.setText("RESUME ▶")
            self.panic_btn.setStyleSheet("QToolButton{background:#b91c1c;color:#fff;font-weight:700;padding:6px 10px;border-radius:8px;}")
            self.halt_banner.setText("TRADING HALTED — Manual intervention required")
        else:
            self.panic_btn.setChecked(False)
            self.panic_btn.setText("HALT ✖")
            self.panic_btn.setStyleSheet("QToolButton{background:#1f2937;color:#e5e5e5;font-weight:700;padding:6px 10px;border-radius:8px;}")
            self.halt_banner.setText("")

    def _toggle_panic(self):
        self._halted = not self._halted
        self._style_panic(self._halted)
        self._append_log(f"[UI] {'HALT' if self._halted else 'RESUME'} pressed")
        self.ctrl.halt_changed.emit(self._halted)

    def _confirm_flatten(self):
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Flatten All — Confirm")
        msg.setText("This will submit market orders to close ALL positions across ALL symbols.")
        msg.setInformativeText("Are you absolutely sure?")
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        yes = msg.exec() == QtWidgets.QMessageBox.Yes
        if yes:
            self._append_log("[UI] Flatten All confirmed")
            self.ctrl.flatten_all.emit()
        else:
            self._append_log("[UI] Flatten All canceled")

    def _confirm_cancel_all(self):
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Cancel All — Confirm")
        msg.setText("Cancel all WORKING orders across ALL symbols?")
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if msg.exec() == QtWidgets.QMessageBox.Yes:
            self._append_log("[UI] Cancel All confirmed")
            self.ctrl.cancel_all.emit()

    def _show_manual_order(self):
        dlg = ManualOrderDialog(self, symbols=[self.symbol_combo.itemText(i) for i in range(self.symbol_combo.count())])
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            payload = dlg.payload()
            self._append_log(f"[UI] Manual Order -> {payload}")
            self.ctrl.manual_order.emit(payload)

    def _on_pos_context(self, pos: QtCore.QPoint):
        idx = self.pos_table.indexAt(pos)
        if not idx.isValid():
            return
        row = idx.row()
        sym = self.pos_model._rows[row].get("symbol")
        menu = QtWidgets.QMenu(self)
        act_flat = menu.addAction(f"Flatten {sym}")
        act_flat25 = menu.addAction(f"Close 25% of {sym}")
        act_flat50 = menu.addAction(f"Close 50% of {sym}")
        chosen = menu.exec(self.pos_table.viewport().mapToGlobal(pos))
        if chosen == act_flat:
            self._append_log(f"[UI] Flatten {sym} requested")
            self.ctrl.flatten_symbol.emit(sym)
        elif chosen in (act_flat25, act_flat50):
            pct = 0.25 if chosen==act_flat25 else 0.50
            self._append_log(f"[UI] Partial close {sym} {int(pct*100)}% (route via engine)")
            # Extend ControlBridge to support partials if you want.

    # ---------------- Utility ----------------
    def _style_table(self,t:QtWidgets.QTableView):
        t.setAlternatingRowColors(True); t.setSortingEnabled(True); t.horizontalHeader().setStretchLastSection(True); t.verticalHeader().setVisible(False); t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

    def _kpi_label(self)->QtWidgets.QLabel:
        lbl=QtWidgets.QLabel("--"); lbl.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter); lbl.setStyleSheet("font-weight:700; font-size:16px; color:#e5e5e5;"); return lbl

    def _set_kpi(self, lbl: QtWidgets.QLabel, val: float, money: bool=False, pct: bool=False):
        color = "#e5e5e5"
        try:
            if pct:
                text = f"{val*100:.2f}%"; color = "#22c55e" if val >= 0 else "#f87171"
            elif money:
                text = f"{val:,.2f}"; color = "#22c55e" if val >= 0 else "#f87171"
            else:
                text = f"{val}"
            lbl.setText(text); lbl.setStyleSheet(f"font-weight:700; font-size:16px; color:{color};")
        except Exception:
            lbl.setText(str(val))

    def _append_log(self, text:str):
        if hasattr(self,'logs_view'):
            self.logs_view.appendPlainText(text)
            self.logs_view.verticalScrollBar().setValue(self.logs_view.verticalScrollBar().maximum())

    def _export_csv(self):
        try:
            pos_path=os.path.join(os.getcwd(), "positions_export.csv")
            eq_path=os.path.join(os.getcwd(), "equity_export.csv")
            model=self.pos_model
            rows=[]
            for r in range(model.rowCount()):
                row={}
                for c,h in enumerate(model.HEADERS):
                    idx=model.index(r,c)
                    row[h]=model.data(idx, QtCore.Qt.DisplayRole)
                rows.append(row)
            pd.DataFrame(rows).to_csv(pos_path, index=False)
            x=getattr(self,'_eq_x',[]); y=getattr(self,'_eq_y',[])
            pd.DataFrame({"t":x,"equity":y}).to_csv(eq_path, index=False)
            QtWidgets.QMessageBox.information(self, "Export", f"CSV exported:\n{pos_path}\n{eq_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export", f"Failed to export: {e}")

    def _confirm_cancel_all(self):
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Cancel All Orders")
        msg.setText("This will CANCEL all working/pending orders.")
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if msg.exec() == QtWidgets.QMessageBox.Yes:
            self._append_log("[UI] Cancel All confirmed")
            self.ctrl.cancel_all.emit()
        else:
            self._append_log("[UI] Cancel All canceled")


    def _confirm_flatten_symbol(self, symbol: str):
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Flatten Symbol")
        msg.setText(f"Close ALL positions for {symbol}?")
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if msg.exec() == QtWidgets.QMessageBox.Yes:
            self._append_log(f"[UI] Flatten {symbol} confirmed")
            self.ctrl.flatten_symbol.emit(symbol)
        else:
            self._append_log(f"[UI] Flatten {symbol} canceled")