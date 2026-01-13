# ============================================================================
# 3. main_window.py - Main Window (Refactored)
# ============================================================================
from PySide6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging


class MainWindow(QtWidgets.QMainWindow):
    """
    Production trading GUI with proper error handling and state management.
    All critical issues from original fixed.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading Bot Monitor — Pro (Live)")
        self.resize(1540, 1000)
        
        self._logger = logging.getLogger("MainWindow")

        # Backend connections
        from core.events.eventhandler import get_event_handler
        from monitoring.bus import ControlBridge
        from monitoring.feeds.feeder import DataFeeder
        from monitoring.feeds.state_aggregator import StateAggregator
        
        self.bus = get_event_handler()
        self.ctrl = ControlBridge(self.bus)
        self.feeder = DataFeeder()
        self.aggregator = StateAggregator(self.feeder, interval_ms=1000)
        self._halted = False

        # State tracking
        self._eq_x, self._eq_y = [], []
        self._price_data: Dict[str, Dict[str, list]] = {}
        self._sim_runner = None
        self._sim_running = False

        # GUI setup
        self._setup_pyqtgraph()
        self._setup_toolbar()
        self._setup_tabs()

        # Signal wiring
        self.aggregator.snapshot_ready.connect(self._update_from_snapshot)
        
        self._logger.info("MainWindow initialized")

    def _setup_pyqtgraph(self):
        """Configure pyqtgraph theme"""
        pg.setConfigOption("background", "#0a0a0a")
        pg.setConfigOption("foreground", "#e5e5e5")
        pg.setConfigOptions(antialias=True)

    def _setup_toolbar(self):
        """Create toolbar with controls"""
        tb = QtWidgets.QToolBar("Controls")
        tb.setMovable(False)
        self.addToolBar(tb)

        # Actions
        self.start_act = QtGui.QAction("Start", self)
        self.stop_act = QtGui.QAction("Stop", self)
        self.clear_logs_act = QtGui.QAction("Clear Logs", self)
        self.export_csv_act = QtGui.QAction("Export CSV", self)

        self.start_act.triggered.connect(
            lambda: asyncio.create_task(self._start_sim())
        )
        self.stop_act.triggered.connect(self._stop_simulation)
        self.clear_logs_act.triggered.connect(
            lambda: self.logs_view.clear() if hasattr(self, 'logs_view') else None
        )

        # Panic button
        self.panic_btn = QtWidgets.QToolButton()
        self.panic_btn.setCheckable(True)
        self.panic_btn.setText("HALT ✖")
        self._style_panic(False)
        self.panic_btn.clicked.connect(self._toggle_panic)

        # Action buttons
        self.flatten_btn = QtWidgets.QToolButton()
        self.flatten_btn.setText("Flatten All")
        self.flatten_btn.clicked.connect(self._confirm_flatten)

        self.cancel_all_btn = QtWidgets.QToolButton()
        self.cancel_all_btn.setText("Cancel All")
        self.cancel_all_btn.clicked.connect(self._confirm_cancel_all)

        self.manual_order_btn = QtWidgets.QToolButton()
        self.manual_order_btn.setText("Manual Order")
        self.manual_order_btn.clicked.connect(self._show_manual_order)

        # Add to toolbar
        for a in [self.start_act, self.stop_act, self.clear_logs_act, self.export_csv_act]:
            tb.addAction(a)
        tb.addSeparator()
        tb.addWidget(self.panic_btn)
        tb.addWidget(self.flatten_btn)
        tb.addWidget(self.cancel_all_btn)
        tb.addWidget(self.manual_order_btn)

    def _setup_tabs(self):
        """Create all tabs"""
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        self._build_dashboard_tab()
        self._build_market_tab()
        self._build_performance_tab()
        self._build_execution_tab()
        self._build_alerts_tab()
        self._build_ops_tab()
        self._build_history_tab()

    # ========================================================================
    # Tab Builders
    # ========================================================================

    def _build_dashboard_tab(self):
        """Dashboard with positions, equity, and risk metrics"""
        tab = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(tab)

        # KPI labels
        self.unreal_lbl = self._kpi_label()
        self.realized_lbl = self._kpi_label()
        self.dd_lbl = self._kpi_label()

        # Risk panel
        risk_box = QtWidgets.QGroupBox("Risk Panel")
        rl = QtWidgets.QGridLayout(risk_box)
        rl.addWidget(QtWidgets.QLabel("Unrealized PnL"), 0, 0)
        rl.addWidget(self.unreal_lbl, 0, 1)
        rl.addWidget(QtWidgets.QLabel("Realized PnL"), 1, 0)
        rl.addWidget(self.realized_lbl, 1, 1)
        rl.addWidget(QtWidgets.QLabel("Drawdown"), 2, 0)
        rl.addWidget(self.dd_lbl, 2, 1)

        # Positions table
        from monitoring.models import SymbolsTableModel
        self.pos_model = SymbolsTableModel([])
        self.pos_table = QtWidgets.QTableView()
        self.pos_table.setModel(self.pos_model)

        # Equity chart
        self.eq_plot = pg.PlotWidget(title="Cumulative Equity")
        self.eq_curve = self.eq_plot.plot([], [], pen=pg.mkPen('#22c55e', width=2))

        # Trade markers
        self.entry_marks = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(0, 180, 0, 200))
        self.exit_marks = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(200, 0, 0, 200))
        self.eq_plot.addItem(self.entry_marks)
        self.eq_plot.addItem(self.exit_marks)

        # Layout (FIXED - removed duplicate)
        grid.addWidget(risk_box, 0, 0)
        grid.addWidget(self.pos_table, 1, 0)
        grid.addWidget(self.eq_plot, 0, 1, 2, 1)
        
        self.tabs.addTab(tab, "Dashboard")

    def _build_market_tab(self):
        """Market data with symbol selector"""
        tab = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(tab)

        # Symbol selector bar
        top_bar = QtWidgets.QHBoxLayout()
        self.symbol_combo = QtWidgets.QComboBox()
        self.symbol_combo.setEditable(False)
        self.symbol_combo.addItem("AAPL")
        self.symbol_combo.currentTextChanged.connect(self._on_symbol_changed)

        refresh_btn = QtWidgets.QPushButton("↻ Refresh")
        refresh_btn.setFixedWidth(80)
        refresh_btn.clicked.connect(lambda: self._refresh_market_chart())

        top_bar.addWidget(QtWidgets.QLabel("Symbol:"))
        top_bar.addWidget(self.symbol_combo)
        top_bar.addStretch(1)
        top_bar.addWidget(refresh_btn)
        
        grid.addLayout(top_bar, 0, 0)

        # Price chart
        self.price_plot = pg.PlotWidget(title="Price Chart — AAPL")
        grid.addWidget(self.price_plot, 1, 0)

        # News feed
        self.news_list = QtWidgets.QListWidget()
        self.news_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        grid.addWidget(self.news_list, 2, 0)

        self.tabs.addTab(tab, "Market")

    def _build_performance_tab(self):
        """Performance metrics"""
        tab = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(tab)

        self.sharpe_lbl = self._kpi_label()
        self.sortino_lbl = self._kpi_label()
        self.kelly_lbl = self._kpi_label()

        for i, (name, lbl) in enumerate([
            ("Sharpe Ratio", self.sharpe_lbl),
            ("Sortino Ratio", self.sortino_lbl),
            ("Kelly Criterion", self.kelly_lbl),
        ]):
            grid.addWidget(QtWidgets.QLabel(name), i, 0)
            grid.addWidget(lbl, i, 1)
        
        grid.setRowStretch(3, 1)
        self.tabs.addTab(tab, "Performance")

    def _build_execution_tab(self):
        """Execution status"""
        tab = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(tab)
        
        self.halt_banner = QtWidgets.QLabel("")
        self.halt_banner.setStyleSheet(
            "background:#991b1b;color:#fff;padding:6px;border-radius:6px;"
        )
        grid.addWidget(self.halt_banner, 0, 0)
        grid.setRowStretch(1, 1)
        
        self.tabs.addTab(tab, "Execution")

    def _build_alerts_tab(self):
        """Alerts feed"""
        tab = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab)
        self.alerts_list = QtWidgets.QListWidget()
        v.addWidget(self.alerts_list)
        self.tabs.addTab(tab, "Alerts")

    def _build_ops_tab(self):
        """Operations & logs"""
        tab = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(tab)

        # Mode selector
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Live", "Simulation"])
        self.mode_combo.setCurrentText("Live")
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)

        # Symbol input
        self.symbol_input = QtWidgets.QLineEdit()
        self.symbol_input.setPlaceholderText("Enter symbols (e.g. AAPL, TSLA)")
        self.symbol_input.setText("AAPL")

        top_bar = QtWidgets.QHBoxLayout()
        top_bar.addWidget(QtWidgets.QLabel("Mode:"))
        top_bar.addWidget(self.mode_combo)
        top_bar.addWidget(QtWidgets.QLabel("Symbols:"))
        top_bar.addWidget(self.symbol_input)
        grid.addLayout(top_bar, 0, 0)

        # Simulation controls
        sim_box = self._build_simulation_controls()
        grid.addWidget(sim_box, 1, 0)

        # Logs
        self.logs_view = QtWidgets.QPlainTextEdit()
        self.logs_view.setReadOnly(True)
        grid.addWidget(self.logs_view, 2, 0)

        self.tabs.addTab(tab, "Ops")

    def _build_simulation_controls(self) -> QtWidgets.QGroupBox:
        """Build simulation parameter controls"""
        sim_box = QtWidgets.QGroupBox("Simulation Controls")
        layout = QtWidgets.QGridLayout(sim_box)

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
        self.sim_start_btn.clicked.connect(
            lambda: asyncio.create_task(self._start_sim())
        )
        self.sim_stop_btn.clicked.connect(self._stop_simulation)

        # Layout
        layout.addWidget(QtWidgets.QLabel("Steps"), 0, 0)
        layout.addWidget(self.sim_steps_spin, 0, 1)
        layout.addWidget(QtWidgets.QLabel("Speed (sec/bar)"), 1, 0)
        layout.addWidget(self.sim_speed_spin, 1, 1)
        layout.addWidget(QtWidgets.QLabel("Drift μ"), 2, 0)
        layout.addWidget(self.sim_mu_spin, 2, 1)
        layout.addWidget(QtWidgets.QLabel("Volatility σ"), 3, 0)
        layout.addWidget(self.sim_sigma_spin, 3, 1)
        layout.addWidget(self.sim_start_btn, 4, 0)
        layout.addWidget(self.sim_stop_btn, 4, 1)

        return sim_box

    def _build_history_tab(self):
        """History/calendar view"""
        tab = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab)
        v.addWidget(QtWidgets.QLabel("PnL Calendar Heatmap (TBD)"))
        self.tabs.addTab(tab, "History")

    # ========================================================================
    # Snapshot Handler (Main Update Entry Point)
    # ========================================================================

    def _update_from_snapshot(self, snap: Dict[str, Any]):
        """Handle merged state snapshot from aggregator"""
        try:
            pnl = snap.get("pnl", {})
            health = snap.get("health", {})
            metrics = snap.get("metrics", {})
            positions = snap.get("positions", {})
            alerts = snap.get("alerts", [])
            trades = snap.get("trades", [])
            buffers = snap.get("buffers", {})

            # Update dashboard
            self._update_perf_dashboard(pnl)
            self._update_health_panel(health)
            self._update_performance_tab(metrics)

            # Update positions table
            if positions:
                df = pd.DataFrame(list(positions.values()))
                self.pos_model.update_from_df(df)

            # Update alerts
            self.alerts_list.clear()
            for a in alerts[-10:]:
                text = a.get("text", str(a))
                self.alerts_list.addItem(text)

            # Handle new trades
            for t in trades[-5:]:
                self._handle_new_trade(t)

            # Update market chart with price data
            price_buffers = buffers.get("price", {})
            for sym, data in price_buffers.items():
                if data:
                    self._update_price_data(sym, data)

            # Add new symbols to dropdown dynamically
            all_symbols = sorted(set(list(positions.keys()) + list(price_buffers.keys())))
            for sym in all_symbols:
                if self.symbol_combo.findText(sym) == -1:
                    self.symbol_combo.addItem(sym)

        except Exception as e:
            self._logger.exception(f"Snapshot update failed: {e}")
            self._append_log(f"[ERR] Snapshot processing error: {e}")

    # ========================================================================
    # Update Helpers
    # ========================================================================

    def _update_perf_dashboard(self, pnl: Dict):
        """Update PnL KPIs and equity chart"""
        try:
            if not pnl:
                return

            self._set_kpi(self.unreal_lbl, pnl.get("unrealized", 0), money=True)
            self._set_kpi(self.realized_lbl, pnl.get("realized", 0), money=True)
            self._set_kpi(self.dd_lbl, pnl.get("drawdown", 0), pct=True)

            # Update equity curve
            val = pnl.get("portfolio_value")
            if val is not None and not np.isnan(val):
                self._eq_x.append(len(self._eq_x))
                self._eq_y.append(float(val))
                
                # Keep last 5000 points
                if len(self._eq_x) > 5000:
                    self._eq_x = self._eq_x[-5000:]
                    self._eq_y = self._eq_y[-5000:]
                
                # Filter NaNs
                mask = np.isfinite(self._eq_y)
                self.eq_curve.setData(
                    np.array(self._eq_x)[mask],
                    np.array(self._eq_y)[mask]
                )
        except Exception as e:
            self._logger.error(f"Performance dashboard update error: {e}")

    def _update_health_panel(self, health: Dict):
        """Update system health banner"""
        try:
            if not health:
                return

            status = health.get("status", "unknown")
            color = "#166534" if status == "healthy" else "#991b1b"
            self.halt_banner.setText(f"Feed: {status.upper()}")
            self.halt_banner.setStyleSheet(
                f"background:{color};color:#fff;padding:6px;border-radius:6px;"
            )
        except Exception as e:
            self._logger.error(f"Health panel update error: {e}")

    def _update_performance_tab(self, metrics: Dict):
        """Update performance metrics KPIs"""
        try:
            if not metrics:
                return

            self._set_kpi(self.sharpe_lbl, metrics.get("sharpe", 0))
            self._set_kpi(self.sortino_lbl, metrics.get("sortino", 0))
            self._set_kpi(self.kelly_lbl, metrics.get("kelly", 0))
        except Exception as e:
            self._logger.error(f"Performance metrics update error: {e}")

    def _update_price_data(self, symbol: str, data: List[Dict]):
        """Update internal price storage"""
        try:
            if not data:
                return

            # Store in internal cache
            self._price_data.setdefault(symbol, {"x": [], "y": []})
            
            for point in data[-1000:]:  # Keep last 1000
                t = point.get("t", len(self._price_data[symbol]["x"]))
                p = point.get("p")
                
                if p is not None and not np.isnan(p):
                    self._price_data[symbol]["x"].append(t)
                    self._price_data[symbol]["y"].append(float(p))

            # If this is the currently selected symbol, update chart
            if self.symbol_combo.currentText() == symbol:
                self._render_price_chart(symbol)

        except Exception as e:
            self._logger.error(f"Price data update error for {symbol}: {e}")

    def _render_price_chart(self, symbol: str):
        """Render price chart for given symbol"""
        try:
            if symbol not in self._price_data:
                self.price_plot.clear()
                self.price_plot.setTitle(f"Price Chart — {symbol} (no data)")
                return

            data = self._price_data[symbol]
            self.price_plot.clear()
            self.price_plot.plot(
                data["x"],
                data["y"],
                pen=pg.mkPen("#22c55e", width=2)
            )
            self.price_plot.setTitle(f"Price Chart — {symbol}")

        except Exception as e:
            self._logger.error(f"Chart rendering error for {symbol}: {e}")

    def _handle_new_trade(self, trade: Dict):
        """Plot trade markers on equity curve"""
        try:
            side = trade.get("side", "").lower()
            x = len(self._eq_x)
            y = self._eq_y[-1] if self._eq_y else 0

            if "buy" in side or "long" in side:
                self.entry_marks.addPoints([{"pos": (x, y)}])
            else:
                self.exit_marks.addPoints([{"pos": (x, y)}])

            self._append_log(
                f"[TRADE] {trade.get('symbol')} {side.upper()} "
                f"{trade.get('qty')} @ {trade.get('price')}"
            )
        except Exception as e:
            self._logger.error(f"Trade marker error: {e}")

    # ========================================================================
    # Event Handlers
    # ========================================================================

    def _on_symbol_changed(self, symbol: str):
        """Handle symbol dropdown change"""
        try:
            self._append_log(f"[UI] Market tab symbol changed → {symbol}")
            self._render_price_chart(symbol)
        except Exception as e:
            self._logger.error(f"Symbol change error: {e}")

    def _refresh_market_chart(self):
        """Refresh the currently selected chart"""
        try:
            symbol = self.symbol_combo.currentText()
            self._render_price_chart(symbol)
        except Exception as e:
            self._logger.error(f"Chart refresh error: {e}")

    def _on_mode_changed(self, mode: str):
        """Handle mode change (Live/Simulation)"""
        self._append_log(f"[MODE] {mode} mode activated")

    def _confirm_flatten(self):
        """Confirm flatten all positions"""
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Flatten All — Confirm")
        msg.setText("Close ALL positions across ALL symbols?")
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setStandardButtons(
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if msg.exec() == QtWidgets.QMessageBox.Yes:
            self._append_log("[UI] Flatten All confirmed")
            self.ctrl.flatten_all.emit()
        else:
            self._append_log("[UI] Flatten All canceled")

    def _confirm_cancel_all(self):
        """Confirm cancel all orders"""
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Cancel All — Confirm")
        msg.setText("Cancel all WORKING orders?")
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setStandardButtons(
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if msg.exec() == QtWidgets.QMessageBox.Yes:
            self._append_log("[UI] Cancel All confirmed")
            self.ctrl.cancel_all.emit()
        else:
            self._append_log("[UI] Cancel All canceled")

    def _toggle_panic(self):
        """Toggle halt/resume"""
        self._halted = not self._halted
        self._style_panic(self._halted)
        self._append_log(f"[UI] {'HALT' if self._halted else 'RESUME'} pressed")
        self.ctrl.halt_changed.emit(self._halted)

    def _show_manual_order(self):
        """Open manual order dialog"""
        try:
            from monitoring.dialogs.manual_order import ManualOrderDialog
            
            symbols = [
                s.strip().upper()
                for s in self.symbol_input.text().split(",")
                if s.strip()
            ] or ["AAPL"]
            
            dlg = ManualOrderDialog(self, symbols=symbols)
            if dlg.exec() == QtWidgets.QDialog.Accepted:
                payload = dlg.payload()
                self._append_log(f"[UI] Manual Order → {payload}")
                self.ctrl.manual_order.emit(payload)
        except Exception as e:
            self._logger.error(f"Manual order dialog error: {e}")
            self._append_log(f"[ERR] Manual order failed: {e}")

    # ========================================================================
    # Simulation
    # ========================================================================

    async def _start_sim(self):
        """Start simulation with parameters from GUI"""
        try:
            # Parse symbols
            symbols = [
                s.strip().upper()
                for s in self.symbol_input.text().split(",")
                if s.strip()
            ] or ["AAPL"]

            steps = self.sim_steps_spin.value()
            sleep_time = self.sim_speed_spin.value()
            mu = self.sim_mu_spin.value()
            sigma = self.sim_sigma_spin.value()

            self._append_log(
                f"[SIM] Starting simulation: {symbols} | "
                f"steps={steps}, μ={mu}, σ={sigma}, Δt={sleep_time}s"
            )

            # Import and configure simulator
            from core.simulator.simulation import SimulationRunner, SimConfig

            cfg = SimConfig(
                symbols=symbols,
                steps=steps,
                bar_sleep=sleep_time,
            )

            self._sim_runner = SimulationRunner(cfg)
            self._sim_runner.events = self.bus

            self._sim_running = True
            await self._sim_runner.run()

            equity = getattr(self._sim_runner.portfolio, "total_equity", lambda: 0.0)()
            self._append_log(f"[SIM] Complete. Final equity: ${equity:,.2f}")

        except asyncio.CancelledError:
            self._append_log("[SIM] Simulation cancelled")
        except Exception as e:
            self._logger.exception(f"Simulation error: {e}")
            self._append_log(f"[SIM] Error: {e}")
        finally:
            self._sim_running = False

    def _stop_simulation(self):
        """Stop running simulation"""
        if self._sim_running and self._sim_runner:
            self._sim_runner.stop()
            self._sim_running = False
            self._append_log("[SIM] Simulation stopped by user")
        else:
            self._append_log("[SIM] No active simulation to stop")

    # ========================================================================
    # Utilities
    # ========================================================================

    def _kpi_label(self) -> QtWidgets.QLabel:
        """Create styled KPI label"""
        lbl = QtWidgets.QLabel("--")
        lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lbl.setStyleSheet("font-weight:700; font-size:16px; color:#e5e5e5;")
        return lbl

    def _set_kpi(self, lbl: QtWidgets.QLabel, val: float, 
                money: bool = False, pct: bool = False):
        """Update KPI label with color coding"""
        try:
            color = "#22c55e" if val >= 0 else "#f87171"
            
            if pct:
                text = f"{val*100:.2f}%"
            elif money:
                text = f"${val:,.2f}"
            else:
                text = f"{val:.2f}"
            
            lbl.setText(text)
            lbl.setStyleSheet(f"font-weight:700; font-size:16px; color:{color};")
        except Exception as e:
            self._logger.error(f"KPI update error: {e}")

    def _append_log(self, msg: str):
        """Append message to logs view"""
        try:
            ts = datetime.utcnow().strftime("%H:%M:%S")
            self.logs_view.appendPlainText(f"[{ts}] {msg}")
        except Exception as e:
            self._logger.error(f"Log append error: {e}")

    def _style_panic(self, halted: bool):
        """Style panic button based on state"""
        if halted:
            self.panic_btn.setChecked(True)
            self.panic_btn.setText("RESUME ▶")
            self.panic_btn.setStyleSheet(
                "QToolButton{background:#b91c1c;color:#fff;"
                "font-weight:700;padding:6px 10px;border-radius:8px;}"
            )
        else:
            self.panic_btn.setChecked(False)
            self.panic_btn.setText("HALT ✖")
            self.panic_btn.setStyleSheet(
                "QToolButton{background:#1f2937;color:#e5e5e5;"
                "font-weight:700;padding:6px 10px;border-radius:8px;}"
            )
