# monitoring/views/main_window.py
from PySide6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import asyncio, os, numpy as np, pandas as pd
from datetime import datetime
from monitoring.bus import ControlBridge
from monitoring.feeds.feeder import DataFeeder
from monitoring.feeds.state_aggregator import StateAggregator
from monitoring.models import SymbolsTableModel
from monitoring.dialogs.manual_order import ManualOrderDialog
from core.events.eventhandler import get_event_handler
from core.events.events import EVENT_PRICE_UPDATE


class MainWindow(QtWidgets.QMainWindow):
    """Full production version of the GUI with live data and all demo visuals."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading Bot Monitor — Pro (Live)")
        self.resize(1540, 1000)

        # ------------------------------------------------------------
        # CORE BACKEND CONNECTIONS
        # ------------------------------------------------------------
        self.bus = get_event_handler()
        self.ctrl = ControlBridge(self.bus)
        self.feeder = DataFeeder()
        self.aggregator = StateAggregator(self.feeder, interval_ms=1000)
        self._halted = False

        # ------------------------------------------------------------
        # GUI INITIALIZATION
        # ------------------------------------------------------------
        pg.setConfigOption("background", "#0a0a0a")
        pg.setConfigOption("foreground", "#e5e5e5")
        pg.setConfigOptions(antialias=True)

        self._setup_toolbar()
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # --- Build tabs (from demo) ---
        self._build_dashboard_tab()
        self._build_market_tab()
        self._build_performance_tab()
        self._build_execution_tab()
        self._build_alerts_tab()
        self._build_ops_tab()
        self._build_history_tab()
        self._build_replay_tab()

        # ------------------------------------------------------------
        # SIGNAL WIRING
        # ------------------------------------------------------------
        self.aggregator.snapshot_ready.connect(self._update_from_snapshot)
        QtCore.QTimer.singleShot(0, lambda: asyncio.create_task(self.feeder.start_safe()))
        self._append_log("[INIT] Feeder + Aggregator connected and running.")



    # ================================================================
    # SNAPSHOT HANDLER
    # ================================================================
    def _update_from_snapshot(self, snap: dict):
        """Merged snapshot emitted by StateAggregator every ~1s."""
        try:
            pnl = snap.get("pnl", {})
            health = snap.get("health", {})
            metrics = snap.get("metrics", {})
            positions = snap.get("positions", {})
            alerts = snap.get("alerts", [])
            trades = snap.get("trades", [])

            self._update_perf_dashboard(pnl)
            self._update_health_panel(health)
            self._update_performance_tab(metrics)
            for sym, info in positions.items():
                self._update_position_row(info)

            self.alerts_list.clear()
            for a in alerts[-10:]:
                self.alerts_list.addItem(a.get("text", str(a)))

            for t in trades[-5:]:
                self._handle_new_trade(t)
            
            # Dynamically add new symbols to dropdown as they appear
            symbols_in_state = sorted(
                set(list(positions.keys()) + list(snap.get("price", {}).keys()))
            )
            for sym in symbols_in_state:
                if self.symbol_combo.findText(sym) == -1:
                    self.symbol_combo.addItem(sym)

        except Exception as e:
            import traceback
            self._append_log(f"[ERR] Snapshot update failed: {e}\n{traceback.format_exc()}")

    # ================================================================
    # TOOLBAR
    # ================================================================
    def _setup_toolbar(self):
        tb = QtWidgets.QToolBar("Controls")
        tb.setMovable(False)
        self.addToolBar(tb)

        self.start_act = QtGui.QAction("Start", self)
        self.stop_act = QtGui.QAction("Stop", self)
        self.clear_logs_act = QtGui.QAction("Clear Logs", self)
        self.export_csv_act = QtGui.QAction("Export CSV", self)

        self.panic_btn = QtWidgets.QToolButton()
        self.panic_btn.setCheckable(True)
        self.panic_btn.setText("HALT ✖")
        self._style_panic(False)

        self.flatten_btn = QtWidgets.QToolButton()
        self.flatten_btn.setText("Flatten All")
        self.cancel_all_btn = QtWidgets.QToolButton()
        self.cancel_all_btn.setText("Cancel All")
        self.manual_order_btn = QtWidgets.QToolButton()
        self.manual_order_btn.setText("Manual Order")

        #simulation controls:
        self.start_act.triggered.connect(lambda: asyncio.create_task(self._start_sim()))
        self.stop_act.triggered.connect(self._stop_simulation)


        for a in [self.start_act, self.stop_act, self.clear_logs_act, self.export_csv_act]:
            tb.addAction(a)
        tb.addSeparator()
        tb.addWidget(self.panic_btn)
        tb.addWidget(self.flatten_btn)
        tb.addWidget(self.cancel_all_btn)
        tb.addWidget(self.manual_order_btn)

        self.panic_btn.clicked.connect(self._toggle_panic)
        self.flatten_btn.clicked.connect(lambda: self._append_log("[UI] Flatten All requested"))
        self.cancel_all_btn.clicked.connect(lambda: self._append_log("[UI] Cancel All requested"))
        self.manual_order_btn.clicked.connect(self._show_manual_order)

    # ================================================================
    # DASHBOARD TAB
    # ================================================================
    def _build_dashboard_tab(self):
        tab = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(tab)

        self.unreal_lbl = self._kpi_label()
        self.realized_lbl = self._kpi_label()
        self.dd_lbl = self._kpi_label()

        risk_box = QtWidgets.QGroupBox("Risk Panel")
        rl = QtWidgets.QGridLayout(risk_box)
        rl.addWidget(QtWidgets.QLabel("Unrealized PnL"), 0, 0)
        rl.addWidget(self.unreal_lbl, 0, 1)
        rl.addWidget(QtWidgets.QLabel("Realized PnL"), 1, 0)
        rl.addWidget(self.realized_lbl, 1, 1)
        rl.addWidget(QtWidgets.QLabel("Drawdown"), 2, 0)
        rl.addWidget(self.dd_lbl, 2, 1)

        self.pos_model = SymbolsTableModel([])
        self.pos_table = QtWidgets.QTableView()
        self.pos_table.setModel(self.pos_model)

        self.eq_plot = pg.PlotWidget(title="Cumulative Equity")
        self.eq_curve = self.eq_plot.plot([], [], pen=pg.mkPen('#22c55e', width=2))

        # entry markers
        self.entry_marks = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(0, 180, 0, 200))
        self.exit_marks = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(200, 0, 0, 200))
        self.eq_plot.addItem(self.entry_marks)
        self.eq_plot.addItem(self.exit_marks)

        grid.addWidget(risk_box, 0, 0)
        grid.addWidget(self.pos_table, 1, 0)
        grid.addWidget(self.eq_plot, 0, 1, 2, 1)
        self.tabs.addTab(tab, "Dashboard")

        grid.addWidget(risk_box, 0, 0)
        grid.addWidget(self.pos_table, 1, 0)
        grid.addWidget(self.eq_plot, 0, 1, 2, 1)
        self.tabs.addTab(tab, "Dashboard")


    # ================================================================
    # MARKET TAB
    # ================================================================
    def _build_market_tab(self):
        tab = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(tab)

        # === Top bar: symbol selector + refresh ===
        top_bar = QtWidgets.QHBoxLayout()
        self.symbol_combo = QtWidgets.QComboBox()
        self.symbol_combo.setEditable(False)
        self.symbol_combo.addItem("AAPL")  # default
        self.symbol_combo.currentTextChanged.connect(self._on_symbol_changed)

        refresh_btn = QtWidgets.QPushButton("↻ Refresh")
        refresh_btn.setFixedWidth(80)
        refresh_btn.clicked.connect(lambda: self._refresh_market_chart())

        # Optional dark styling
        self.symbol_combo.setStyleSheet(
            "QComboBox { background:#1e1e1e; color:#e5e5e5; border:1px solid #333; padding:2px 6px; }"
        )
        refresh_btn.setStyleSheet(
            "QPushButton { background:#1e1e1e; color:#22c55e; border:1px solid #333; padding:3px 6px; }"
        )

        top_bar.addWidget(QtWidgets.QLabel("Symbol:"))
        top_bar.addWidget(self.symbol_combo)
        top_bar.addStretch(1)
        top_bar.addWidget(refresh_btn)
        grid.addLayout(top_bar, 0, 0)

        # === Price Chart ===
        self.price_plot = pg.PlotWidget(title="Price Chart — AAPL")
        grid.addWidget(self.price_plot, 1, 0)

        # === News Feed ===
        self.news_list = QtWidgets.QListWidget()
        self.news_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        grid.addWidget(self.news_list, 2, 0)

        self.tabs.addTab(tab, "Market")

    # ================================================================
    # PERFORMANCE TAB
    # ================================================================
    def _build_performance_tab(self):
        tab = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(tab)
        self.sharpe_lbl = self._kpi_label()
        self.sortino_lbl = self._kpi_label()
        self.kelly_lbl = self._kpi_label()
        for i, (n, lbl) in enumerate([
            ("Sharpe", self.sharpe_lbl),
            ("Sortino", self.sortino_lbl),
            ("Kelly", self.kelly_lbl),
        ]):
            grid.addWidget(QtWidgets.QLabel(n), i, 0)
            grid.addWidget(lbl, i, 1)
        self.tabs.addTab(tab, "Performance")

    # ================================================================
    # EXECUTION TAB
    # ================================================================
    def _build_execution_tab(self):
        tab = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(tab)
        self.halt_banner = QtWidgets.QLabel("")
        self.halt_banner.setStyleSheet("background:#991b1b;color:#fff;padding:6px;border-radius:6px;")
        grid.addWidget(self.halt_banner, 0, 0)
        self.tabs.addTab(tab, "Execution")

    # ================================================================
    # ALERTS TAB
    # ================================================================
    def _build_alerts_tab(self):
        tab = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab)
        self.alerts_list = QtWidgets.QListWidget()
        v.addWidget(self.alerts_list)
        self.tabs.addTab(tab, "Alerts")

    # ================================================================
    # OPS TAB (logs)
    # ================================================================
    def _build_ops_tab(self):
        tab = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(tab)

        # === Core Controls ===
        self.flatten_btn = QtWidgets.QPushButton("Flatten All")
        self.cancel_all_btn = QtWidgets.QPushButton("Cancel All")
        self.halt_btn = QtWidgets.QPushButton("Halt")
        self.ticket_btn = QtWidgets.QPushButton("Manual Order")

        self.flatten_btn.clicked.connect(self._confirm_flatten)
        self.cancel_all_btn.clicked.connect(self._confirm_cancel_all)
        self.halt_btn.clicked.connect(self._toggle_panic)
        self.ticket_btn.clicked.connect(self._show_manual_order)

        hl = QtWidgets.QHBoxLayout()
        [hl.addWidget(b) for b in [self.flatten_btn, self.cancel_all_btn, self.halt_btn, self.ticket_btn]]

        # === Mode + Symbol Input ===
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Live", "Simulation"])
        self.mode_combo.setCurrentText("Live")
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)

        self.symbol_input = QtWidgets.QLineEdit()
        self.symbol_input.setPlaceholderText("Enter symbols (e.g. AAPL, TSLA, MSFT)")

        hl.addWidget(QtWidgets.QLabel("Mode:"))
        hl.addWidget(self.mode_combo)
        hl.addWidget(QtWidgets.QLabel("Symbols:"))
        hl.addWidget(self.symbol_input)

        grid.addLayout(hl, 0, 0)

        # === Simulation Controls ===
        sim_box = QtWidgets.QGroupBox("Simulation Controls")
        sim_layout = QtWidgets.QGridLayout(sim_box)

        # Parameters
        self.sim_steps_spin = QtWidgets.QSpinBox()
        self.sim_steps_spin.setRange(10, 100000)
        self.sim_steps_spin.setValue(2000)

        self.sim_speed_spin = QtWidgets.QDoubleSpinBox()
        self.sim_speed_spin.setRange(0.01, 5.0)
        self.sim_speed_spin.setSingleStep(0.05)
        self.sim_speed_spin.setValue(0.1)

        self.sim_mu_spin = QtWidgets.QDoubleSpinBox()
        self.sim_mu_spin.setRange(-0.5, 0.5)
        self.sim_mu_spin.setSingleStep(0.01)
        self.sim_mu_spin.setValue(0.05)

        self.sim_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.sim_sigma_spin.setRange(0.0, 1.0)
        self.sim_sigma_spin.setSingleStep(0.01)
        self.sim_sigma_spin.setValue(0.2)

        # Buttons
        self.sim_start_btn = QtWidgets.QPushButton("Start Simulation")
        self.sim_stop_btn = QtWidgets.QPushButton("Stop Simulation")
        self.sim_start_btn.clicked.connect(lambda: asyncio.create_task(self._start_sim()))
        self.sim_stop_btn.clicked.connect(self._stop_simulation)

        # Layout
        sim_layout.addWidget(QtWidgets.QLabel("Steps"), 0, 0)
        sim_layout.addWidget(self.sim_steps_spin, 0, 1)
        sim_layout.addWidget(QtWidgets.QLabel("Speed (sec/bar)"), 1, 0)
        sim_layout.addWidget(self.sim_speed_spin, 1, 1)
        sim_layout.addWidget(QtWidgets.QLabel("Drift μ"), 2, 0)
        sim_layout.addWidget(self.sim_mu_spin, 2, 1)
        sim_layout.addWidget(QtWidgets.QLabel("Volatility σ"), 3, 0)
        sim_layout.addWidget(self.sim_sigma_spin, 3, 1)
        sim_layout.addWidget(self.sim_start_btn, 4, 0)
        sim_layout.addWidget(self.sim_stop_btn, 4, 1)

        grid.addWidget(sim_box, 2, 0)

        # === Logs ===
        self.logs_view = QtWidgets.QPlainTextEdit()
        self.logs_view.setReadOnly(True)
        grid.addWidget(self.logs_view, 1, 0)

        self.tabs.addTab(tab, "Ops")

    # ================================================================
    # HISTORY + REPLAY (placeholders)
    # ================================================================
    def _build_history_tab(self):
        tab = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab)
        v.addWidget(QtWidgets.QLabel("PnL Calendar Heatmap (TBD)"))
        tab.setLayout(v)
        self.tabs.addTab(tab, "History")

    def _build_replay_tab(self):
        tab = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab)
        v.addWidget(QtWidgets.QLabel("Replay Mode (TBD)"))
        tab.setLayout(v)
        self.tabs.addTab(tab, "Replay")

    # ================================================================
    # UPDATE HELPERS
    # ================================================================
    def _update_perf_dashboard(self, pnl: dict):
        if not pnl:
            return
        self._set_kpi(self.unreal_lbl, pnl.get("unrealized", 0), money=True)
        self._set_kpi(self.realized_lbl, pnl.get("realized", 0), money=True)
        self._set_kpi(self.dd_lbl, pnl.get("drawdown", 0), pct=True)

        val = pnl.get("portfolio_value", np.nan)
        ts = pnl.get("timestamp", len(getattr(self, "_eq_x", [])))
        if not hasattr(self, "_eq_x"):
            self._eq_x, self._eq_y = [], []
        self._eq_x.append(len(self._eq_x))
        self._eq_y.append(val)
        mask = np.isfinite(self._eq_y)
        self.eq_curve.setData(np.array(self._eq_x)[mask], np.array(self._eq_y)[mask])

    def _update_health_panel(self, health: dict):
        if not health:
            return
        status = health.get("status", "unknown")
        color = "#166534" if status == "healthy" else "#991b1b"
        self.halt_banner.setText(f"Feed: {status.upper()}")
        self.halt_banner.setStyleSheet(f"background:{color};color:#fff;padding:6px;border-radius:6px;")
    
    def _on_mode_changed(self, mode: str):
        """Switch between Live and Simulation modes."""
        if mode == "Live":
            self._append_log("[MODE] Live trading mode activated.")
        elif mode == "Simulation":
            self._append_log("[MODE] Simulation mode activated.")
        else:
            self._append_log(f"[MODE] Unknown mode: {mode}")


    def _update_performance_tab(self, metrics: dict):
        if not metrics:
            return
        self._set_kpi(self.sharpe_lbl, metrics.get("sharpe", 0))
        self._set_kpi(self.sortino_lbl, metrics.get("sortino", 0))
        self._set_kpi(self.kelly_lbl, metrics.get("kelly", 0))

    def _update_position_row(self, info: dict):
        try:
            df = pd.DataFrame([info])
            self.pos_model.update_from_df(df)
        except Exception as e:
            self._append_log(f"[WARN] Position update failed: {e}")

    def _update_price_chart(self, price: dict):
        """Update Market tab price chart with latest simulated or live prices."""
        sym = price.get("symbol")
        px = float(price.get("price", np.nan))
        ts = price.get("timestamp")

        if not sym:
            return

        # Initialize storage
        if not hasattr(self, "_price_data"):
            self._price_data = {}

        d = self._price_data.setdefault(sym, {"x": [], "y": []})

        # Handle timestamp
        try:
            t = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
        except Exception:
            t = len(d["x"])

        d["x"].append(t)
        d["y"].append(px)

        # Keep rolling window
        if len(d["x"]) > 1000:
            d["x"], d["y"] = d["x"][-1000:], d["y"][-1000:]

        # If this symbol is the one displayed, update the chart immediately
        current_sym = getattr(self, "symbol_combo", None)
        if current_sym and current_sym.currentText() == sym:
            self.price_plot.clear()
            self.price_plot.plot(d["x"], d["y"], pen=pg.mkPen("#22c55e", width=2))
            self.price_plot.setTitle(f"Price Chart — {sym}")

            self.price_plot.plot(d["x"], d["y"], pen=pg.mkPen("#22c55e", width=2))
    
    def _on_symbol_changed(self, sym: str):
        """Triggered when user selects a different symbol from dropdown."""
        self._append_log(f"[UI] Market tab symbol changed → {sym}")
        if hasattr(self, "_price_data") and sym in self._price_data:
            d = self._price_data[sym]
            self.price_plot.clear()
            self.price_plot.plot(d["x"], d["y"], pen=pg.mkPen("#22c55e", width=2))
            self.price_plot.setTitle(f"Price Chart — {sym}")
        else:
            self.price_plot.clear()
            self.price_plot.setTitle(f"Price Chart — {sym} (no data yet)")


    def _refresh_market_chart(self, sym: str = None):
        """Refresh the market price plot for the selected symbol."""
        try:
            sym = sym or self.symbol_combo.currentText()
            if not hasattr(self, "aggregator") or not hasattr(self.aggregator, "cache"):
                self._append_log("[WARN] Aggregator not ready for chart refresh.")
                return

            # Retrieve from aggregator cache if available
            snap = getattr(self.aggregator, "cache", {})
            buffers = snap.get("buffers", {})
            prices = buffers.get("price", {})

            if sym not in prices:
                self._append_log(f"[WARN] No price data available for {sym}")
                return

            data = prices[sym]
            if not data:
                return

            x = np.arange(len(data))
            y = [p.get("p", np.nan) for p in data]

            self.price_plot.clear()
            self.price_plot.plot(x, y, pen=pg.mkPen("#22c55e", width=2))
            self.price_plot.setTitle(f"Price Chart — {sym}")

        except Exception as e:
            self._append_log(f"[WARN] Market chart refresh failed: {e}")

    def _handle_new_trade(self, trade: dict):
        """Plot trade markers on the equity curve."""
        try:
            sym = trade.get("symbol", "?")
            px = float(trade.get("price", np.nan))
            x = len(getattr(self, "_eq_x", []))

            if not hasattr(self, "entry_marks"):
                return  # avoid crashing if chart isn't ready

            if trade.get("side") in ("buy", "long"):
                self.entry_marks.addPoints([{"pos": (x, px)}])
            else:
                self.exit_marks.addPoints([{"pos": (x, px)}])

            self._append_log(f"[TRADE] {trade['side'].upper()} {sym} @ {px}")
        except Exception as e:
            self._append_log(f"[WARN] Trade marker failed: {e}")

    
    # ------------------------------------------------------------
    #  CONTROL ACTIONS (Flatten / Cancel / Halt / Manual Order)
    # ------------------------------------------------------------
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
            if hasattr(self, "ctrl"):
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
            if hasattr(self, "ctrl"):
                self.ctrl.cancel_all.emit()
        else:
            self._append_log("[UI] Cancel All canceled")

    def _toggle_panic(self):
        """Toggle the panic/kill switch."""
        self._halted = not getattr(self, "_halted", False)
        state = "HALT" if self._halted else "RESUME"
        self._append_log(f"[UI] {state} pressed")
        if hasattr(self, "ctrl"):
            self.ctrl.halt_changed.emit(self._halted)

    def _show_manual_order(self):
        """Open the manual order dialog."""
        from monitoring.dialogs.manual_order import ManualOrderDialog

        dlg = ManualOrderDialog(self, symbols=[s.strip().upper() for s in self.symbol_input.text().split(",") if s.strip()])
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            payload = dlg.payload()
            self._append_log(f"[UI] Manual Order -> {payload}")
            if hasattr(self, "ctrl"):
                self.ctrl.manual_order.emit(payload)


    # ================================================================
    # UTILITIES
    # ================================================================
    def _kpi_label(self):
        lbl = QtWidgets.QLabel("--")
        lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lbl.setStyleSheet("font-weight:700; font-size:16px; color:#e5e5e5;")
        return lbl

    def _set_kpi(self, lbl, val, money=False, pct=False):
        color = "#22c55e" if val >= 0 else "#f87171"
        if pct:
            lbl.setText(f"{val*100:.2f}%")
        elif money:
            lbl.setText(f"${val:,.2f}")
        else:
            lbl.setText(str(round(val, 2)))
        lbl.setStyleSheet(f"font-weight:700; font-size:16px; color:{color};")

    def _append_log(self, msg: str):
        ts = datetime.utcnow().strftime("%H:%M:%S")
        if hasattr(self, "logs_view"):
            self.logs_view.appendPlainText(f"[{ts}] {msg}")

    # Panic toggle
    def _style_panic(self, halted: bool):
        if halted:
            self.panic_btn.setChecked(True)
            self.panic_btn.setText("RESUME ▶")
            self.panic_btn.setStyleSheet("QToolButton{background:#b91c1c;color:#fff;font-weight:700;padding:6px 10px;border-radius:8px;}")
        else:
            self.panic_btn.setChecked(False)
            self.panic_btn.setText("HALT ✖")
            self.panic_btn.setStyleSheet("QToolButton{background:#1f2937;color:#e5e5e5;font-weight:700;padding:6px 10px;border-radius:8px;}")

    def _toggle_panic(self):
        self._halted = not self._halted
        self._style_panic(self._halted)
        self._append_log(f"[UI] {'HALT' if self._halted else 'RESUME'} pressed")
        self.ctrl.halt_changed.emit(self._halted)

    def _show_manual_order(self):
        dlg = ManualOrderDialog(self, symbols=["AAPL", "MSFT", "TSLA"])
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            payload = dlg.payload()
            self._append_log(f"[UI] Manual Order -> {payload}")
            self.ctrl.manual_order.emit(payload)

    # ================================================================
    # SIMULATION CONTROL
    # ================================================================
    async def _start_sim(self):
        """Launch SimulationRunner (GBM-based) using GUI parameters."""

        try:
            # --- Gather GUI parameters (add widgets to GUI for these) ---
            raw = getattr(self, "symbol_input", None)
            if raw and hasattr(raw, "text"):
                symbols = [s.strip().upper() for s in raw.text().split(",") if s.strip()] or ["AAPL"]
            else:
                symbols = ["AAPL"]

            steps = getattr(self, "sim_steps_spin", None)
            steps = steps.value() if steps else 1000

            mu_spin = getattr(self, "sim_mu_spin", None)
            mu = mu_spin.value() if mu_spin else 0.0005

            sigma_spin = getattr(self, "sim_sigma_spin", None)
            sigma = sigma_spin.value() if sigma_spin else 0.02

            sleep_time = 0.5  # fixed or link to spinbox

            self._append_log(
                f"[SIM] Launching SimulationRunner | symbols={symbols} | steps={steps}, μ={mu}, σ={sigma}, Δt={sleep_time}s"
            )

            # --- Build SimConfig (imported at top) ---
            from core.simulator.simulation import SimulationRunner, SimConfig

            cfg = SimConfig(
                symbols=symbols,
                steps=steps,
                bar_sleep=sleep_time,
            )

            self._sim_runner = SimulationRunner(cfg)
            if hasattr(self, "bus"):
                self._sim_runner.events = self.bus  # share GUI EventBus

            self._sim_running = True
            await self._sim_runner.run()

            final_equity = getattr(self._sim_runner.portfolio, "total_equity", lambda: 0.0)()
            self._append_log(f"[SIM] Completed. Final equity: ${final_equity:,.2f}")

        except asyncio.CancelledError:
            self._append_log("[SIM] Simulation cancelled.")
        except Exception as e:
            import traceback
            self._append_log(f"[SIM] Error: {e}\n{traceback.format_exc()}")
        finally:
            self._sim_running = False

    def _stop_simulation(self):
        """Stop running simulation cleanly."""
        if getattr(self, "_sim_running", False):
            self._sim_running = False
            self._append_log("[SIM] Simulation manually stopped by user.")
            if hasattr(self, "_sim_runner"):
                self._sim_runner.stop()
        else:
            self._append_log("[SIM] No active simulation to stop.")
