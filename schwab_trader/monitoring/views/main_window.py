from PySide6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import os
import pandas as pd
import numpy as np

from datetime import datetime, timezone
from core.simulator.gbm_simulator import GBMSimulator
from core.historical_loader import HistoricalBarLoader
from core.mock_executor import MockExecutor
from core.simulator.simulation import SimulationRunner, SimConfig
import random

from monitoring.bus import ControlBridge
from monitoring.models import SymbolsTableModel
from monitoring.dialogs.manual_order import ManualOrderDialog

from core.events.eventhandler import EventHandler, get_event_handler
from monitoring.feeds.feeder import DataFeeder
from core.events import events
import asyncio





class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trading Bot Monitor — Pro")
        self.resize(1540, 1000)

        # === Core wiring ===
        self.bus = get_event_handler()
        self.ctrl = ControlBridge(self.bus)
        self.feeder = DataFeeder()
        self._halted = False
        self.executor = None

        # === Session state ===
        self._eq_x, self._eq_y = [], []
        self._csv_dir = os.path.join(os.getcwd(), "csv")
        self._session_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        os.makedirs(self._csv_dir, exist_ok=True)

        # Event → CSV map
        self._event_csv_map = {
            events.EVENT_NEW_TRADE:      "trades.csv",
            events.EVENT_PNL_UPDATE:     "pnl_updates.csv",
            events.EVENT_ORDER_STATUS:   "orders.csv",
            events.EVENT_ALERT:          "alerts.csv",
            events.EVENT_POSITION_UPDATE:"positions.csv",
            events.EVENT_PRICE_UPDATE:   "prices.csv",
            events.EVENT_FLATTEN_ALL:    "ui_actions.csv",
            events.EVENT_CANCEL_ALL:     "ui_actions.csv",
            events.EVENT_FLATTEN_SYMBOL: "ui_actions.csv",
            events.EVENT_MANUAL_ORDER:   "ui_actions.csv",
            events.EVENT_HALTED:         "ui_actions.csv",
        }
        self._buffers = {fname: [] for fname in set(self._event_csv_map.values())}
        self._buffer_limit = 50
        self._csv_path = lambda fname: os.path.join(self._csv_dir, f"{self._session_id}_{fname}")


        # === Theme ===
        pg.setConfigOption("background", "#0a0a0a")
        pg.setConfigOption("foreground", "#e5e5e5")
        pg.setConfigOptions(antialias=True)

        # === Toolbar ===
        tb = QtWidgets.QToolBar("Controls")
        tb.setMovable(False)
        self.addToolBar(tb)

        self.start_act = QtGui.QAction("Start", self)
        self.stop_act = QtGui.QAction("Stop", self)
        self.clear_logs_act = QtGui.QAction("Clear Logs", self)
        self.export_csv_act = QtGui.QAction("Export CSV", self)
        self.export_pdf_act = QtGui.QAction("Export PDF (stub)", self)

        self.panic_btn = QtWidgets.QToolButton()
        self.halt_banner = QtWidgets.QLabel("")
        self.halt_banner.setObjectName("haltBanner")
        self.halt_banner.setStyleSheet("background:#991b1b;color:#fff;padding:6px;border-radius:6px;")
        self.panic_btn.setCheckable(True)
        self.panic_btn.setText("HALT ✖")
        self._style_panic(False)

        self.flatten_btn_tb = QtWidgets.QToolButton(); self.flatten_btn_tb.setText("Flatten All")
        self.cancel_all_btn_tb = QtWidgets.QToolButton(); self.cancel_all_btn_tb.setText("Cancel All")
        self.manual_order_btn_tb = QtWidgets.QToolButton(); self.manual_order_btn_tb.setText("Manual Order")

        for a in [self.start_act, self.stop_act, self.clear_logs_act, self.export_csv_act, self.export_pdf_act]:
            tb.addAction(a)
        tb.addSeparator()
        tb.addWidget(self.panic_btn)
        tb.addWidget(self.flatten_btn_tb)
        tb.addWidget(self.cancel_all_btn_tb)
        tb.addWidget(self.manual_order_btn_tb)

        # Shortcuts
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

        # Toolbar actions
        self.start_act.triggered.connect(lambda: self._append_log("[UI] Start clicked"))
        self.stop_act.triggered.connect(lambda: self._append_log("[UI] Stop clicked"))
        self.clear_logs_act.triggered.connect(lambda: self.logs_view.clear())
        self.export_csv_act.triggered.connect(self._export_all_csvs)
        self.export_pdf_act.triggered.connect(lambda: QtWidgets.QMessageBox.information(
            self, "Export", "PDF export is a stub. Wire reportlab/wkhtmltopdf."))

        self.panic_btn.clicked.connect(self._toggle_panic)
        self.flatten_btn_tb.clicked.connect(self._confirm_flatten)
        self.cancel_all_btn_tb.clicked.connect(self._confirm_cancel_all)
        self.manual_order_btn_tb.clicked.connect(self._show_manual_order)

        # heart beat indicator
        self.heartbeat_indicator = QtWidgets.QLabel("●")
        self.heartbeat_indicator.setStyleSheet("color: #22c55e; font-size: 18px;")
        self.statusBar().addPermanentWidget(self.heartbeat_indicator)

        # === Wire bus ===
        self._wire_gui_to_bus()
        self._subscribe_backend_events()

        # Connect feeder → GUI slots
# --- Connect feeder → GUI slots ---
        s = self.feeder.s

        # OHLC bars → price chart updates
        s.ohlc.connect(lambda sym, df: self._update_price_chart({"symbol": sym, "data": df}))

        # Trades → trade markers and log
        s.trades.connect(lambda rows: [self._handle_new_trade(r) for r in rows])

        # Positions → table model updates
        s.symbols.connect(lambda rows: [self._update_position_row(r) for r in rows])

        # PnL → equity curve and labels
        s.equity_point.connect(lambda ts, val: self._update_perf_dashboard({"timestamp": ts, "portfolio_value": val}))
        s.realized_point.connect(lambda ts, realized, unrealized: self._update_perf_dashboard({"timestamp": ts, "realized": realized, "unrealized": unrealized}))
        s.risk_stats.connect(lambda u, r, dd: self._update_perf_dashboard({"unrealized": u, "realized": r, "drawdown": dd}))

        # Alerts → alerts list
        s.alerts.connect(lambda alerts: [self.alerts_list.addItem(a["text"]) for a in alerts])
        # 7️⃣ Cooldown / halt → panic button + disable
        s.cooldown.connect(self._on_cooldown_state)

        # 8️⃣ Health → log or status bar
        s.health.connect(lambda h: self._append_log(f"[HEALTH] {h.get('broker','?')} - {h.get('status','?')}"))

        # 9️⃣ Regime breakdown (optional future use)
        s.regime_breakdown.connect(lambda d: self._append_log(f"[REGIME] {d}"))
        # Logs → Ops tab
        s.log.connect(self._append_log)
        #asyncio.create_task(self.feeder.start_safe())
        self._append_log("[INIT] Feeder started and subscribed to EventBus.")
    # ---------------- GUI -> Bus ----------------
    def _wire_gui_to_bus(self):
        self.ctrl.halt_changed.connect(lambda halted: self._emit_and_log(events.EVENT_HALTED, {"halted": bool(halted)}))
        self.ctrl.flatten_all.connect(lambda: self._emit_and_log(events.EVENT_FLATTEN_ALL, {}))
        self.ctrl.cancel_all.connect(lambda: self._emit_and_log(events.EVENT_CANCEL_ALL, {}))
        self.ctrl.flatten_symbol.connect(lambda sym: self._emit_and_log(events.EVENT_FLATTEN_SYMBOL, {"symbol": sym}))
        self.ctrl.manual_order.connect(lambda payload: self._emit_and_log(events.EVENT_MANUAL_ORDER, payload))

    def _emit_and_log(self, event_name: str, payload: dict):
        import asyncio
        asyncio.create_task(self.bus.emit(event_name, payload))
        self.log_event(event_name, payload)

    # ---------------- Bus -> GUI ----------------
    def _subscribe_backend_events(self):
        """Subscribe GUI to backend EventBus and route updates to GUI slots."""
        async def sub(event_name, handler):
            await self.bus.subscribe(event_name, handler)

        async def setup_subs():
            await asyncio.gather(
                sub(events.EVENT_PNL_UPDATE, self._on_pnl_update),
                sub(events.EVENT_NEW_TRADE, self._on_new_trade),
                sub(events.EVENT_ORDER_STATUS, self._on_order_status),
                sub(events.EVENT_ALERT, self._on_alert),
                sub(events.EVENT_POSITION_UPDATE, self._on_position_update),
                sub(events.EVENT_PRICE_UPDATE, self._on_price_update),
                sub(events.EVENT_MANUAL_ORDER, self._on_manual_order),
            )
            self._append_log("[INIT] Subscribed to backend EventBus events.")

        QtCore.QTimer.singleShot(0, lambda: asyncio.create_task(setup_subs()))
    # ---------------- Event Handlers ----------------
    async def _on_pnl_update(self, event):
        QtCore.QTimer.singleShot(0, lambda: self._update_perf_dashboard(event.payload))
        self.log_event(events.EVENT_PNL_UPDATE, event.payload)

    async def _on_new_trade(self, event):
        QtCore.QTimer.singleShot(0, lambda: self._handle_new_trade(event.payload))
        self.log_event(events.EVENT_NEW_TRADE, event.payload)

    async def _on_order_status(self, event):
        QtCore.QTimer.singleShot(0, lambda: self._update_order_kpis(event.payload))
        self.log_event(events.EVENT_ORDER_STATUS, event.payload)

    async def _on_alert(self, event):
        QtCore.QTimer.singleShot(0, lambda: self.alerts_list.addItem(
            f"{event.payload['level'].upper()}: {event.payload['message']}"))
        self.log_event(events.EVENT_ALERT, event.payload)

    async def _on_position_update(self, event):
        QtCore.QTimer.singleShot(0, lambda: self._update_position_row(event.payload))
        self.log_event(events.EVENT_POSITION_UPDATE, event.payload)

    async def _on_price_update(self, event):
        QtCore.QTimer.singleShot(0, lambda: self._update_price_chart(event.payload))
        self.log_event(events.EVENT_PRICE_UPDATE, event.payload)
    
    async def _on_manual_order(self, event):
        """Handle manual orders coming from the dialog (UI -> Backend)."""
        payload = event.payload
        sym = payload.get("symbol", "?")
        side = payload.get("side", "?")
        qty = payload.get("qty", "?")
        order_type = payload.get("type", "?")

        # Log to GUI
        self._append_log(f"[UI] Manual order → {side.upper()} {qty} {sym} ({order_type})")

        # Persist to CSV
        #self.log_event(events.EVENT_MANUAL_ORDER, payload)
    
    def _on_mode_changed(self, mode: str):
        self.sim_mode = (mode == "Simulation")
        self._append_log(f"[UI] Mode switched → {mode}")
        if self.sim_mode:
            # kick off sim loop
            self.executor = MockExecutor()
            asyncio.create_task(self.executor.bus.subscribe(events.EVENT_MANUAL_ORDER, self.executor._on_manual_order))
            asyncio.create_task(self._start_sim())
        else:
            self.sim_mode = False  # ensures _start_sim() loop exits

    # ---------------- Update Helpers ----------------


    # def _update_perf_dashboard(self, pnl: dict):
    #     # Update KPI labels
    #     if 'unrealized' in pnl:
    #         self._set_kpi(self.unreal_lbl, pnl['unrealized'], money=True)
    #     if 'realized' in pnl:
    #         self._set_kpi(self.realized_lbl, pnl['realized'], money=True)
    #     if 'drawdown' in pnl:
    #         self._set_kpi(self.dd_lbl, pnl['drawdown'], pct=True)

    #     # Update equity curve
    #     if 'portfolio_value' in pnl and 'timestamp' in pnl:
    #         ts = pnl['timestamp']

    #         # --- Convert timestamp to float for plotting ---
    #         if isinstance(ts, str):
    #             try:
    #                 ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    #             except Exception:
    #                 ts = float(len(self._eq_x))  # fallback sequential index
    #         elif isinstance(ts, datetime):
    #             ts = ts.timestamp()

    #         try:
    #             y_val = float(pnl['portfolio_value'])
    #         except (ValueError, TypeError):
    #             y_val = np.nan

    #         self._eq_x.append(ts)
    #         self._eq_y.append(y_val)

    #         # --- Ensure numeric numpy arrays for pyqtgraph ---
    #         x = np.asarray(self._eq_x, dtype=float)
    #         y = np.asarray(self._eq_y, dtype=float)
    #         self.eq_curve.setData(x, y)

    def _update_perf_dashboard(self, pnl: dict):
        # === Update KPI labels ===
        if 'unrealized' in pnl:
            self._set_kpi(self.unreal_lbl, pnl['unrealized'], money=True)
        if 'realized' in pnl:
            self._set_kpi(self.realized_lbl, pnl['realized'], money=True)
        if 'drawdown' in pnl:
            self._set_kpi(self.dd_lbl, pnl['drawdown'], pct=True)

        # === Update equity curve ===
        if 'portfolio_value' in pnl and 'timestamp' in pnl:
            ts = pnl['timestamp']

            # Convert timestamp → float (safe)
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                except Exception:
                    ts = float(len(self._eq_x))
            elif isinstance(ts, datetime):
                ts = ts.timestamp()

            try:
                y_val = float(pnl['portfolio_value'])
            except (ValueError, TypeError):
                y_val = np.nan

            self._eq_x.append(ts)
            self._eq_y.append(y_val)

            # --- 🧩 Fix 1: trim runaway array length ---
            MAX_POINTS = 10000
            if len(self._eq_x) > MAX_POINTS:
                self._eq_x = self._eq_x[-MAX_POINTS:]
                self._eq_y = self._eq_y[-MAX_POINTS:]

            # --- 🧩 Fix 2: remove NaN / Inf values ---
            x = np.asarray(self._eq_x, dtype=float)
            y = np.asarray(self._eq_y, dtype=float)
            mask = np.isfinite(y)
            if not mask.all():
                bad = np.sum(~mask)
                self._append_log(f"[WARN] Dropped {bad} bad PnL points (NaN/inf)")
                x, y = x[mask], y[mask]

            # --- 🧩 Fix 3: ensure monotonic timestamps ---
            # If any out-of-order data snuck in, sort all points by timestamp.
            if len(x) > 1:
                order = np.argsort(x)
                x, y = x[order], y[order]

                # Ensure uniqueness (optional safety)
                _, unique_idx = np.unique(x, return_index=True)
                x, y = x[unique_idx], y[unique_idx]
            
            # --- 🧩 Normalize timestamps (FIX huge x-axis) ---
            if len(x) > 0:
                x = x - x[0]

            # --- 🧩 Optional: downsample for GUI performance ---
            if len(x) > 2000:
                step = max(1, len(x) // 2000)
                x, y = x[::step], y[::step]

            # Safe plot call
            try:
                self.eq_curve.setData(x, y)
            except Exception as e:
                self._append_log(f"[ERR] Plot update failed: {e}")

    
    def _update_health_panel(self, payload: dict):
        msg = f"[HEALTH] {payload['status'].upper()} | " \
            f"Age: {payload['details']['last_emit_age']}s"
        #self._append_log(msg)

        status = payload.get("status", "unknown")
        age = payload.get("details", {}).get("last_emit_age", 0)

        # === Gentle pulse when healthy ===
        if not hasattr(self, "_health_anim"):
            self._health_anim = QtCore.QPropertyAnimation(self.halt_banner, b"windowOpacity")
            self._health_anim.setDuration(500)
            self._health_anim.setStartValue(1.0)
            self._health_anim.setEndValue(0.3)
            self._health_anim.setEasingCurve(QtCore.QEasingCurve.InOutQuad)
            self._health_anim.setLoopCount(2)  # one quick fade in/out pulse

        if status == "healthy":
            self.halt_banner.setStyleSheet("background:#166534;color:#fff;padding:6px;border-radius:6px;")
            self.halt_banner.setText(f"Feed OK ({age:.1f}s)")
            self.halt_banner.show()

            # restart gentle pulse
            self._health_anim.stop()
            self.halt_banner.setWindowOpacity(1.0)
            self._health_anim.start()

        elif status == "stale":
            self.halt_banner.setStyleSheet("background:#991b1b;color:#fff;padding:6px;border-radius:6px;")
            self.halt_banner.setText("Feed stalled!")
            self.halt_banner.show()

        else:
            self.halt_banner.hide()


    def _handle_new_trade(self, trade: dict):
        sym, px = trade['symbol'], float(trade['price'])
        x = len(self._eq_x)
        if trade['side'] in ('buy', 'long'):
            self.entry_marks.addPoints([{'pos': (x, px)}])
        else:
            self.exit_marks.addPoints([{'pos': (x, px)}])
        self._append_log(f"[TRADE] {trade['side'].upper()} {sym} {trade['qty']} @ {px}")

    def _update_position_row(self, pos: dict):
        if hasattr(self.pos_model, 'update_position'):
            self.pos_model.update_position(pos)

    def _update_order_kpis(self, order: dict):
        status = order.get('status', '').lower()
        if status == 'submitted':
            self.q_pending.setText(str(int(self.q_pending.text() or '0') + 1))
        elif status in ('canceled', 'rejected'):
            self.q_canceled.setText(str(int(self.q_canceled.text() or '0') + 1))

    def _update_price_chart(self, price: dict):
        """Update Market tab price chart with latest simulated prices."""
        sym = price.get("symbol")
        px = float(price.get("price", np.nan))
        ts = price.get("timestamp")

        if not hasattr(self, "_price_data"):
            self._price_data = {}

        self._price_data.setdefault(sym, {"x": [], "y": []})
        d = self._price_data[sym]

        try:
            t = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
        except Exception:
            t = len(d["x"])

        d["x"].append(t)
        d["y"].append(px)

        # Keep last 500 points for performance
        if len(d["x"]) > 500:
            d["x"] = d["x"][-500:]
            d["y"] = d["y"][-500:]

        self.price_plot.clear()
        self.price_plot.plot(d["x"], d["y"], pen=pg.mkPen("#22c55e", width=2))


    def _append_log(self, msg: str):
        """
        Append a log message to the logs_view in Ops tab.
        """
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        if hasattr(self, "logs_view"):
            self.logs_view.appendPlainText(f"[{ts}] {msg}")
    
    def _update_performance_tab(self, metrics: dict):
        """Update Sharpe, Sortino, and Kelly labels."""
        if not metrics:
            return
        sharpe = metrics.get("sharpe")
        sortino = metrics.get("sortino")
        kelly = metrics.get("kelly")

        if sharpe is not None:
            self._set_kpi(self.sharpe_lbl, sharpe)
        if sortino is not None:
            self._set_kpi(self.sortino_lbl, sortino)
        if kelly is not None:
            self._set_kpi(self.kelly_lbl, kelly)

    def _on_cooldown_state(self, cooldown: bool):
        """Toggle HALT state visual + disable trade buttons."""
        self._append_log(f"[STATE] Cooldown active: {cooldown}")
        self._style_panic(cooldown)
        for b in [self.flatten_btn, self.cancel_all_btn, self.halt_btn, self.ticket_btn]:
            b.setEnabled(not cooldown)
        for tb_btn in [self.flatten_btn_tb, self.cancel_all_btn_tb, self.manual_order_btn_tb]:
            tb_btn.setEnabled(not cooldown)



    

    # ---------------- CSV Logging ----------------
    def log_event(self, event_name: str, payload: dict):
        fname = self._event_csv_map.get(event_name)
        if not fname: return
        row = dict(payload)
        row.setdefault("timestamp", datetime.utcnow().isoformat())
        row.setdefault("event", event_name)
        buf = self._buffers[fname]; buf.append(row)
        if len(buf) >= self._buffer_limit:
            self._flush_csv(fname)

    def _flush_csv(self, fname: str):
        buf = self._buffers.get(fname, [])
        if not buf: return
        path = self._csv_path(fname)
        all_keys = set().union(*[row.keys() for row in buf])
        df = pd.DataFrame(buf, columns=sorted(all_keys))
        try:
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"# SESSION STARTED {datetime.utcnow().isoformat()}\n")
                df.to_csv(path, mode="a", header=True, index=False)
            else:
                df.to_csv(path, mode="a", header=False, index=False)
            self._buffers[fname] = []
        except Exception as e:
            self._append_log(f"[ERR] CSV flush failed for {fname}: {e}")

    def _write_footer(self, fname: str):
        path = self._csv_path(fname)
        if os.path.exists(path):
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"# SESSION ENDED {datetime.utcnow().isoformat()}\n")

    def _export_all_csvs(self):
        for fname in self._buffers.keys():
            self._flush_csv(fname)
            self._write_footer(fname)
        QtWidgets.QMessageBox.information(self, "Export", f"CSV logs exported to {self._csv_dir}")

    def closeEvent(self, event: QtGui.QCloseEvent):
        for fname in self._buffers.keys():
            if self._buffers[fname]:
                self._flush_csv(fname)
            self._write_footer(fname)
        event.accept()
        asyncio.create_task(self.feeder.stop())
        super().closeEvent(event)
    #-----------------Simulator---------------------
    
    async def _start_sim(self):

        # load symbols from GUI input (fallback to AAPL if empty)
        raw = self.symbol_input.text() if hasattr(self, "symbol_input") else "AAPL"
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]

        # use your real historical data path
        hist_path = r"C:\Users\kwasi\OneDrive\Documents\Personal Projects\schwab_trader\data\data_storage\proc_data"
        hist_loader = HistoricalBarLoader(hist_path)

        # use latest historical close as base price (fallback = 100.0)
        base_prices = {
            s: hist_loader.get_latest_close_price(s) or 100.0
            for s in symbols
        }

        sim = GBMSimulator(symbols, base_price=base_prices, log_prices=True)
        executor = MockExecutor()

        self._append_log(f"[SIM] Starting simulation for {symbols} with base {base_prices}")

        while self.sim_mode:  # keep running until user switches mode
            bars = sim.update_all()
            for sym, bar in bars.items():
                # emit price to GUI
                await self.bus.emit(events.EVENT_PRICE_UPDATE, bar)

                # TEMP: random signals (replace with real strategy later)
                signal = random.choice([-1, 0, 1])
                atr_val = max(bar["high"] - bar["low"], 0.01)
                executor.execute(sym, None, signal, bar["close"], atr_val)

            await asyncio.sleep(0.1)  # controls sim speed

    # ---------------- Tab Builders ----------------
    def _build_dashboard_tab(self):
        tab = QtWidgets.QWidget(); grid=QtWidgets.QGridLayout(tab)
        risk_box = QtWidgets.QGroupBox("Risk Panel"); rl=QtWidgets.QGridLayout(risk_box)
        self.unreal_lbl=self._kpi_label(); self.realized_lbl=self._kpi_label(); self.dd_lbl=self._kpi_label()
        rl.addWidget(QtWidgets.QLabel("Unrealized PnL"),0,0); rl.addWidget(self.unreal_lbl,0,1)
        rl.addWidget(QtWidgets.QLabel("Realized PnL"),1,0); rl.addWidget(self.realized_lbl,1,1)
        rl.addWidget(QtWidgets.QLabel("Drawdown"),2,0); rl.addWidget(self.dd_lbl,2,1)
        self.pos_model=SymbolsTableModel([]); self.pos_table=QtWidgets.QTableView(); self.pos_table.setModel(self.pos_model)
        left=QtWidgets.QVBoxLayout(); left.addWidget(risk_box); left.addWidget(self.pos_table)
        lw=QtWidgets.QWidget(); lw.setLayout(left)
        right=QtWidgets.QVBoxLayout()
        self.eq_plot=pg.PlotWidget(title="Equity Curve")
        self.eq_curve=self.eq_plot.plot([],[],pen=pg.mkPen(width=2))
        self.entry_marks=pg.ScatterPlotItem(size=9, brush=pg.mkBrush(0,180,0,200))
        self.exit_marks=pg.ScatterPlotItem(size=9, brush=pg.mkBrush(200,0,0,200))
        self.eq_plot.addItem(self.entry_marks); self.eq_plot.addItem(self.exit_marks)
        right.addWidget(self.eq_plot,1)
        rw=QtWidgets.QWidget(); rw.setLayout(right)
        grid.addWidget(lw,0,0,2,1); grid.addWidget(rw,0,1,2,2)
        self.tabs.addTab(tab,"Dashboard")

    def _build_market_tab(self):
        tab=QtWidgets.QWidget(); grid=QtWidgets.QGridLayout(tab)
        self.price_plot=pg.PlotWidget(title="Price Chart")
        self.price_plot.addItem(pg.ScatterPlotItem())  # stub
        grid.addWidget(self.price_plot,0,0)
        self.news_list=QtWidgets.QListWidget(); grid.addWidget(self.news_list,1,0)
        self.tabs.addTab(tab,"Market")

    def _build_performance_tab(self):
        tab=QtWidgets.QWidget(); grid=QtWidgets.QGridLayout(tab)
        self.sharpe_lbl=self._kpi_label(); self.sortino_lbl=self._kpi_label(); self.kelly_lbl=self._kpi_label()
        for i,(n,l) in enumerate([("Sharpe",self.sharpe_lbl),("Sortino",self.sortino_lbl),("Kelly",self.kelly_lbl)]):
            grid.addWidget(QtWidgets.QLabel(n),i,0); grid.addWidget(l,i,1)
        self.tabs.addTab(tab,"Performance")

    def _build_execution_tab(self):
        tab=QtWidgets.QWidget(); grid=QtWidgets.QGridLayout(tab)
        self.q_pending=self._kpi_label(); self.q_canceled=self._kpi_label()
        grid.addWidget(QtWidgets.QLabel("Pending Orders"),0,0); grid.addWidget(self.q_pending,0,1)
        grid.addWidget(QtWidgets.QLabel("Canceled Orders"),1,0); grid.addWidget(self.q_canceled,1,1)
        self.tabs.addTab(tab,"Execution")

    def _build_alerts_tab(self):
        tab=QtWidgets.QWidget(); v=QtWidgets.QVBoxLayout(tab)
        self.alerts_list=QtWidgets.QListWidget(); v.addWidget(self.alerts_list)
        self.tabs.addTab(tab,"Alerts")

    def _build_strategy_tab(self):
        tab=QtWidgets.QWidget(); grid=QtWidgets.QGridLayout(tab)
        self.sig_table=QtWidgets.QTableWidget(0,4)
        self.sig_table.setHorizontalHeaderLabels(["Strategy","Last Signal","Confidence","Next Eval"])
        grid.addWidget(self.sig_table,0,0)
        self.tabs.addTab(tab,"Strategies")

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

        # Create widgets first
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

        # Create simulation control buttons
        self.sim_start_btn = QtWidgets.QPushButton("Start Simulation")
        self.sim_stop_btn = QtWidgets.QPushButton("Stop Simulation")

        # Now connect signals AFTER creation
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

    def _build_history_tab(self):
        tab=QtWidgets.QWidget(); grid=QtWidgets.QGridLayout(tab)
        self.calendar_plot=pg.PlotWidget(title="PnL Calendar"); grid.addWidget(self.calendar_plot,0,0)
        self.bench_plot=pg.PlotWidget(title="Equity vs Benchmark"); grid.addWidget(self.bench_plot,0,1)
        self.tabs.addTab(tab,"History")

    def _build_replay_tab(self):
        tab=QtWidgets.QWidget(); v=QtWidgets.QVBoxLayout(tab)
        self.replay_slider=QtWidgets.QSlider(QtCore.Qt.Horizontal); v.addWidget(self.replay_slider)
        self.replay_plot=pg.PlotWidget(title="Replay"); self.replay_curve=self.replay_plot.plot([],[],pen=pg.mkPen(width=2))
        v.addWidget(self.replay_plot); self.tabs.addTab(tab,"Replay")

    # ---------------- Utility ----------------
    def _style_panic(self, halted: bool):
        if halted: self.panic_btn.setChecked(True); self.panic_btn.setText("RESUME ▶")
        else: self.panic_btn.setChecked(False); self.panic_btn

    def _show_manual_order(self):
        dlg = ManualOrderDialog(self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            payload = dlg.payload()
            self._emit_and_log(events.EVENT_MANUAL_ORDER, payload)

    def _confirm_flatten(self):
        if QtWidgets.QMessageBox.question(self, "Confirm Flatten", 
            "Close all positions?") == QtWidgets.QMessageBox.Yes:
            self._emit_and_log(events.EVENT_FLATTEN_ALL, {})

    def _confirm_cancel_all(self):
        if QtWidgets.QMessageBox.question(self, "Confirm Cancel All", 
            "Cancel all orders?") == QtWidgets.QMessageBox.Yes:
            self._emit_and_log(events.EVENT_CANCEL_ALL, {})

    def _toggle_panic(self):
        self._halted = not self._halted
        self.ctrl.halt_changed.emit(self._halted)
        self._style_panic(self._halted)
        self._emit_and_log(events.EVENT_HALTED, {"halted": self._halted})
    

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

    # async def _start_sim(self):
    #     """Run GBM-based simulation using GUI parameters."""

    #     raw = self.symbol_input.text().strip()
    #     symbols = [s.strip().upper() for s in raw.split(",") if s.strip()] or ["AAPL"]

    #     steps = self.sim_steps_spin.value()
    #     sleep_time = self.sim_speed_spin.value()
    #     mu = self.sim_mu_spin.value()
    #     sigma = self.sim_sigma_spin.value()

    #     self._append_log(f"[SIM] Starting {steps}-step sim | μ={mu}, σ={sigma}, Δt={sleep_time}s")

    #     sim = GBMSimulator(
    #         symbols,
    #         base_price={s: 100.0 for s in symbols},
    #         log_prices=True,
    #         drift_scale=mu,
    #         vol_scale=sigma,
    #         dt=sleep_time / 390.0,
    #     )
    #     executor = MockExecutor()
    #     self._sim_running = True

    #     for i in range(steps):
    #         if not self._sim_running or getattr(self, "_halted", False):
    #             self._append_log("[SIM] Simulation stopped.")
    #             break

    #         bars = sim.update_all()

    #         for sym, bar in bars.items():
    #             # ensure timestamp is ISO8601 string
    #             ts = (
    #                 bar["timestamp"].isoformat()
    #                 if not isinstance(bar["timestamp"], str)
    #                 else bar["timestamp"]
    #             )

    #             # --- 1️⃣ Emit full OHLC bar (BarPayload)
    #             await self.bus.emit(events.EVENT_NEW_BAR, {
    #                 "symbol": sym,
    #                 "open": bar["open"],
    #                 "high": bar["high"],
    #                 "low": bar["low"],
    #                 "close": bar["close"],
    #                 "volume": bar["volume"],
    #                 "timestamp": ts,
    #             })

    #             # --- 2️⃣ Emit price update (PricePayload)
    #             await self.bus.emit(events.EVENT_PRICE_UPDATE, {
    #                 "symbol": sym,
    #                 "price": bar["close"],
    #                 "ma20": None,
    #                 "ma50": None,
    #                 "timestamp": ts,
    #             })

    #             # --- 3️⃣ Emit dummy position (PositionPayload)
    #             await self.bus.emit(events.EVENT_POSITION_UPDATE, {
    #                 "symbol": sym,
    #                 "qty": 100,
    #                 "avg_price": bar["close"],
    #                 "unrealized": 0.0,
    #                 "realized": 0.0,
    #                 "timestamp": ts,
    #             })

    #             # --- 4️⃣ Random trade signal (optional)
    #             signal = random.choice([-1, 0, 1])
    #             atr_val = max(bar["high"] - bar["low"], 0.01)
    #             executor.execute(sym, None, signal, bar["close"], atr_val)
            
    #         #  pacing per bar (controls speed)
    #         await asyncio.sleep(sleep_time)

    #         # yield occasionally to keep GUI responsive
    #         if i % 10 == 0:
    #             await asyncio.sleep(0)

    #     await asyncio.sleep(sleep_time)

    #     self._append_log("[SIM] Simulation complete.")
    #     self._sim_running = False

    async def _start_sim(self):
        """Launch SimulationRunner (GBM-based) using GUI parameters."""

        # --- GUI parameters ---
        raw = self.symbol_input.text().strip()
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()] or ["AAPL"]

        steps = self.sim_steps_spin.value()
        #sleep_time = self.sim_speed_spin.value()
        sleep_time = 0.5
        mu = self.sim_mu_spin.value()
        sigma = self.sim_sigma_spin.value()

        self._append_log(
            f"[SIM] Launching SimulationRunner | symbols={symbols} | steps={steps}, μ={mu}, σ={sigma}, Δt={sleep_time}s"
        )

        # --- Build SimConfig (so the runner uses GUI values) ---
        cfg = SimConfig(
            symbols=symbols,
            steps=steps,
            bar_sleep=sleep_time,
            # the rest (drawdown config etc.) stays default
        )

        # --- Instantiate the SimulationRunner ---
        self._sim_runner = SimulationRunner(cfg)

        # (optional) attach to GUI’s event bus so signals, PnL, etc. flow to frontend
        if hasattr(self, "bus"):
            self._sim_runner.events = self.bus  # share same EventHandler

        self._sim_running = True

        try:
            # Run the actual simulation pipeline
            await self._sim_runner.run()

            self._append_log(
                f"[SIM] Completed. Final equity: ${self._sim_runner.portfolio.total_equity():,.2f}"
            )
        except asyncio.CancelledError:
            self._append_log("[SIM] Simulation cancelled.")
        except Exception as e:
            self._append_log(f"[SIM] Error: {e}")
        finally:
            self._sim_running = False

    def _stop_simulation(self):
        if getattr(self, "_sim_running", False):
            self._sim_running = False
            self._append_log("[SIM] Simulation manually stopped by user.")
        else:
            self._append_log("[SIM] No active simulation to stop.")