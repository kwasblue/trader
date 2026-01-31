from PySide6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import os
from pathlib import Path
import pandas as pd
import numpy as np

from datetime import datetime, timezone

# Project root for relative path resolution
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
from core.simulator.gbm_simulator import GBMSimulator
from core.historical_loader import HistoricalBarLoader
from core.mock_executor import MockExecutor
from core.simulator.simulation import SimulationRunner, SimConfig
import random

from monitoring.bus import ControlBridge
from monitoring.models import SymbolsTableModel
from monitoring.dialogs.manual_order import ManualOrderDialog
# StateAggregator removed - using direct feeder connections now

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

        # Mode selector
        self.mode_label = QtWidgets.QLabel("Mode:")
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Simulation", "Alpaca", "Schwab"])
        self.mode_combo.setMinimumWidth(100)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)

        # Symbol input
        self.symbol_label = QtWidgets.QLabel("Symbols:")
        self.symbol_input = QtWidgets.QLineEdit("AAPL,MSFT")
        self.symbol_input.setMinimumWidth(120)
        self.symbol_input.setPlaceholderText("AAPL,MSFT,GOOGL")

        for a in [self.start_act, self.stop_act, self.clear_logs_act, self.export_csv_act, self.export_pdf_act]:
            tb.addAction(a)
        tb.addSeparator()
        tb.addWidget(self.mode_label)
        tb.addWidget(self.mode_combo)
        tb.addWidget(self.symbol_label)
        tb.addWidget(self.symbol_input)
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

        # Toolbar actions - wire to trading controls
        self.start_act.triggered.connect(self._start_trading)
        self.stop_act.triggered.connect(self._stop_trading)
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

        # === Direct feeder → GUI connections ===
        # NOTE: Feeder subscribes to EventBus SYNCHRONOUSLY in its __init__,
        # so subscriptions are already in place when we connect Qt signals here.
        self._connect_feeder_signals()

        # Log subscription status
        pnl_count = len(self.bus.listeners.get("PNL_UPDATE", []))
        bar_count = len(self.bus.listeners.get("NEW_BAR", []))
        self._append_log(f"[INIT] Feeder ready: PNL_UPDATE={pnl_count}, NEW_BAR={bar_count} listeners")
        self._append_log("[INIT] GUI signal connections established.")

    def _connect_feeder_signals(self):
        """Connect DataFeeder Qt signals to GUI update methods.

        IMPORTANT: These handlers must be SYNC functions (not async).
        They receive data from the feeder's Qt signals.
        """
        # P&L and equity updates
        self.feeder.s.pnl_update.connect(self._gui_on_pnl)
        self.feeder.s.equity_update.connect(self._gui_on_equity)

        # Market data
        self.feeder.s.bar_update.connect(self._gui_on_bar)
        self.feeder.s.price_update.connect(self._gui_on_price)

        # Health status
        self.feeder.s.health_update.connect(self._gui_on_health)

        # Trades and positions
        self.feeder.s.trade_update.connect(self._gui_on_trade)
        self.feeder.s.position_update.connect(self._gui_on_position)
        self.feeder.s.order_update.connect(self._gui_on_order)

        # Logs and alerts
        self.feeder.s.log_message.connect(self._append_log)
        self.feeder.s.alert.connect(self._gui_on_alert)

        self._append_log("[INIT] Feeder Qt signals connected to GUI handlers.")

    # ================================================================
    # GUI HANDLERS (SYNC) - Connected to DataFeeder Qt signals
    # ================================================================

    def _gui_on_pnl(self, data: dict):
        """Handle P&L update from feeder Qt signal (SYNC)."""
        try:
            self._append_log(f"[PNL] Portfolio: ${data.get('portfolio_value', 0):,.2f}")

            # Update KPI labels
            if 'unrealized' in data:
                self._set_kpi(self.unreal_lbl, data['unrealized'], money=True)
            if 'realized' in data:
                self._set_kpi(self.realized_lbl, data['realized'], money=True)
            if 'drawdown' in data:
                self._set_kpi(self.dd_lbl, data['drawdown'], pct=True)

            # Update equity curve
            value = data.get('portfolio_value', 0)
            if value:
                self._eq_x.append(len(self._eq_x))
                self._eq_y.append(float(value))
                self._update_equity_chart()

            # Log to CSV
            self.log_event(events.EVENT_PNL_UPDATE, data)
        except Exception as e:
            self._append_log(f"[ERR] PnL update failed: {e}")

    def _gui_on_equity(self, value: float):
        """Handle simple equity update (SYNC)."""
        self._eq_x.append(len(self._eq_x))
        self._eq_y.append(value)
        self._update_equity_chart()

    def _gui_on_bar(self, symbol: str, bar: dict):
        """Handle bar update from feeder (SYNC)."""
        try:
            close = bar.get('close', 0)
            self._append_log(f"[BAR] {symbol}: ${close:.2f}")
            self._update_price_chart({"symbol": symbol, "data": bar})
        except Exception as e:
            self._append_log(f"[ERR] Bar update failed: {e}")

    def _gui_on_price(self, symbol: str, price: float):
        """Handle price update from feeder (SYNC)."""
        # Update price display if needed
        pass

    def _gui_on_health(self, data: dict):
        """Handle health status update (SYNC)."""
        try:
            status = data.get('status', 'unknown')
            details = data.get('details', {})
            age = details.get('last_emit_age', 0) if isinstance(details, dict) else 0
            count = details.get('event_count', 0) if isinstance(details, dict) else 0

            if status == 'healthy':
                self.heartbeat_indicator.setStyleSheet("color: #22c55e; font-size: 18px;")
                self.halt_banner.setStyleSheet("background:#166534;color:#fff;padding:6px;border-radius:6px;")
                self.halt_banner.setText(f"Feed OK | Events: {count}")
                self.halt_banner.show()
            else:
                self.heartbeat_indicator.setStyleSheet("color: #ef4444; font-size: 18px;")
                self.halt_banner.setStyleSheet("background:#991b1b;color:#fff;padding:6px;border-radius:6px;")
                self.halt_banner.setText(f"Feed STALE ({age:.0f}s)")
                self.halt_banner.show()
        except Exception as e:
            self._append_log(f"[ERR] Health update failed: {e}")

    def _gui_on_trade(self, data: dict):
        """Handle trade update from feeder (SYNC)."""
        try:
            symbol = data.get('symbol', 'UNKNOWN')
            side = data.get('side', 'unknown')
            qty = data.get('qty', 0)
            price = data.get('price', 0)
            self._append_log(f"[TRADE] {side.upper()} {qty} {symbol} @ ${price:.2f}")
            self._handle_new_trade(data)
            self.log_event(events.EVENT_NEW_TRADE, data)
        except Exception as e:
            self._append_log(f"[ERR] Trade update failed: {e}")

    def _gui_on_position(self, data: dict):
        """Handle position update from feeder (SYNC)."""
        try:
            if hasattr(self, 'pos_model') and hasattr(self.pos_model, 'update_position'):
                self.pos_model.update_position(data)
            self.log_event(events.EVENT_POSITION_UPDATE, data)
        except Exception as e:
            self._append_log(f"[ERR] Position update failed: {e}")

    def _gui_on_order(self, data: dict):
        """Handle order status update from feeder (SYNC)."""
        try:
            order_id = data.get('order_id', 'N/A')
            status = data.get('status', 'unknown')
            self._append_log(f"[ORDER] {order_id}: {status}")
            self._update_order_kpis(data)
            self.log_event(events.EVENT_ORDER_STATUS, data)
        except Exception as e:
            self._append_log(f"[ERR] Order update failed: {e}")

    def _gui_on_alert(self, data: dict):
        """Handle alert from feeder (SYNC)."""
        try:
            level = data.get('level', 'info') if isinstance(data, dict) else 'info'
            message = data.get('message', str(data)) if isinstance(data, dict) else str(data)
            self._append_log(f"[ALERT] {level.upper()}: {message}")
            if hasattr(self, 'alerts_list'):
                self.alerts_list.addItem(f"{level.upper()}: {message}")
            self.log_event(events.EVENT_ALERT, data)
        except Exception as e:
            self._append_log(f"[ERR] Alert update failed: {e}")

    def _update_equity_chart(self):
        """Update the equity curve chart."""
        try:
            if not self._eq_x or not self._eq_y:
                return

            x = np.asarray(self._eq_x[-1000:], dtype=float)  # Limit to last 1000 points
            y = np.asarray(self._eq_y[-1000:], dtype=float)

            if hasattr(self, 'eq_curve'):
                self.eq_curve.setData(x, y)
        except Exception as e:
            self._append_log(f"[ERR] Equity chart update failed: {e}")

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
        """
        Subscribe to backend events.

        NOTE: The DataFeeder handles EventBus → Qt signal bridging.
        This method now only handles events that aren't covered by the feeder.
        """
        async def sub(event_name, handler):
            await self.bus.subscribe(event_name, handler)

        async def setup_subs():
            # Only subscribe to events not handled by the feeder
            await sub(events.EVENT_MANUAL_ORDER, self._async_on_manual_order)
            self._append_log("[INIT] Backend event subscriptions ready.")

        QtCore.QTimer.singleShot(0, lambda: asyncio.create_task(setup_subs()))

    # ---------------- Async Event Handlers (for events not handled by feeder) ----------------
    async def _async_on_manual_order(self, event):
        """Handle manual orders coming from the dialog (UI -> Backend)."""
        payload = event.payload
        sym = payload.get("symbol", "?")
        side = payload.get("side", "?")
        qty = payload.get("qty", "?")
        order_type = payload.get("type", "?")

        # Log to GUI (thread-safe via QTimer)
        QtCore.QTimer.singleShot(0, lambda: self._append_log(
            f"[UI] Manual order → {side.upper()} {qty} {sym} ({order_type})"
        ))
    
    def _on_mode_changed(self, mode: str):
        """Handle mode change from the combo box."""
        self._current_mode = mode
        self._append_log(f"[UI] Mode selected → {mode}")
        # Mode change just selects - Start button actually starts trading

    def _start_trading(self):
        """Start trading with the selected mode."""
        mode = getattr(self, '_current_mode', 'Simulation')
        symbols_text = self.symbol_input.text() if hasattr(self, 'symbol_input') else "AAPL,MSFT"
        symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]

        if not symbols:
            self._append_log("[ERROR] No symbols specified")
            return

        self._trading_active = True
        self._append_log(f"[START] Starting {mode} mode for: {', '.join(symbols)}")

        asyncio.create_task(self._run_backend(mode, symbols))

    async def _run_backend(self, mode: str, symbols: list):
        """Run the selected backend."""
        try:
            # Verify feeder subscriptions (they were set up synchronously at init)
            pnl_listeners = len(self.bus.listeners.get("PNL_UPDATE", []))
            bar_listeners = len(self.bus.listeners.get("NEW_BAR", []))
            self._append_log(f"[INIT] EventBus: PNL={pnl_listeners}, BAR={bar_listeners} listeners")

            if pnl_listeners == 0:
                self._append_log("[ERROR] No PNL listeners! Feeder not initialized properly.")
                return

            # Start the backend
            if mode == "Simulation":
                await self._run_simulation_backend(symbols)
            elif mode == "Alpaca":
                await self._run_alpaca_backend(symbols)
            elif mode == "Schwab":
                await self._run_schwab_backend(symbols)
        except asyncio.CancelledError:
            self._append_log(f"[STOP] {mode} backend stopped")
        except Exception as e:
            import traceback
            self._append_log(f"[ERROR] Backend error: {e}")
            self._append_log(traceback.format_exc())

    async def _run_simulation_backend(self, symbols: list):
        """Run GBM simulation backend."""
        from core.simulator.simulation import SimConfig, SimulationRunner

        config = SimConfig(
            symbols=symbols,
            steps=999999,  # Run indefinitely
            bar_sleep=0.1,
        )
        self._sim_runner = SimulationRunner(config)
        self._append_log("[SIM] Starting simulation...")
        await self._sim_runner.run()

    async def _run_alpaca_backend(self, symbols: list):
        """Run Alpaca live/paper trading."""
        from utils.settings import Settings
        from core.alpaca_runner import AlpacaLiveRunner

        settings = Settings(root="config", include_root=True)
        self._alpaca_runner = AlpacaLiveRunner(settings, symbols)
        self._append_log("[ALPACA] Connecting to Alpaca...")
        await self._alpaca_runner.run()

    async def _run_schwab_backend(self, symbols: list):
        """Run Schwab live trading using SchwabLiveRunner."""
        from utils.settings import Settings
        from core.schwab_runner import SchwabLiveRunner

        try:
            settings = Settings(root="config", include_root=True)
            self._schwab_runner = SchwabLiveRunner(settings, symbols)
            self._append_log("[SCHWAB] Connecting to Schwab...")
            await self._schwab_runner.run()
        except ValueError as e:
            self._append_log(f"[SCHWAB] Configuration error: {e}")
            self._append_log("[SCHWAB] Please set SCHWAB_API_KEY and SCHWAB_SECRET in .env file")
        except Exception as e:
            self._append_log(f"[SCHWAB] Error: {e}")

    def _stop_trading(self):
        """Stop the current trading backend."""
        self._trading_active = False
        self._append_log("[STOP] Stopping trading...")

        # Cancel any running backend tasks
        if hasattr(self, '_sim_runner'):
            self._sim_runner = None
        if hasattr(self, '_alpaca_runner'):
            self._alpaca_runner = None

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
        status = payload.get("status", "unknown")
        details = payload.get("details", {})
        age = details.get("last_emit_age", 0) if isinstance(details, dict) else 0

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
        sym = trade.get('symbol', 'UNKNOWN')
        try:
            px = float(trade.get('price', 0))
        except (ValueError, TypeError):
            px = 0.0
        side = trade.get('side', 'unknown')
        qty = trade.get('qty', 0)

        x = len(self._eq_x)
        if side in ('buy', 'long'):
            self.entry_marks.addPoints([{'pos': (x, px)}])
        else:
            self.exit_marks.addPoints([{'pos': (x, px)}])
        self._append_log(f"[TRADE] {side.upper()} {sym} {qty} @ {px}")

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
        """Update Market tab price chart with latest simulated prices.

        Handles two input formats:
        1. Direct price: {"symbol": str, "price": float, "timestamp": str}
        2. OHLC data: {"symbol": str, "data": DataFrame}
        """
        sym = price.get("symbol")
        if not sym:
            return

        if not hasattr(self, "_price_data"):
            self._price_data = {}

        self._price_data.setdefault(sym, {"x": [], "y": []})
        d = self._price_data[sym]

        # Handle OHLC DataFrame format
        if "data" in price:
            df = price["data"]
            # Check if it's a DataFrame with .empty attribute, otherwise skip
            if df is not None and hasattr(df, 'empty') and not df.empty:
                close_col = "Close" if "Close" in df.columns else "close"
                if close_col in df.columns:
                    px = float(df[close_col].iloc[-1])
                    # Use index as timestamp if available
                    if hasattr(df.index, 'to_pydatetime'):
                        try:
                            ts = df.index[-1].timestamp()
                        except Exception:
                            ts = len(d["x"])
                    else:
                        ts = len(d["x"])
                    d["x"].append(ts)
                    d["y"].append(px)
        else:
            # Handle direct price format
            px = float(price.get("price", np.nan))
            ts = price.get("timestamp")

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


    def _update_from_snapshot(self, snap: dict):
        """
        Receive merged state snapshot from aggregator.
        Runs once per interval (~1s) with unified data from all subsystems.
        """
        try:
            pnl = snap.get("pnl", {})
            health = snap.get("health", {})
            metrics = snap.get("metrics", {})
            positions = snap.get("positions", {})
            alerts = snap.get("alerts", [])
            trades = snap.get("trades", [])

            # === 1️⃣ Performance Dashboard (equity, drawdown, etc.) ===
            self._update_perf_dashboard({
                "portfolio_value": pnl.get("portfolio_value", 0.0),
                "realized": pnl.get("realized", 0.0),
                "unrealized": pnl.get("unrealized", 0.0),
                "drawdown": pnl.get("drawdown", 0.0),
                "timestamp": pnl.get("timestamp", None),
            })

            # === 2️⃣ Performance Metrics (Sharpe, Sortino, Kelly, etc.) ===
            if metrics:
                self._update_performance_tab(metrics)

            # === 3️⃣ Health Indicator ===
            if health and isinstance(health, dict) and health.get("status"):
                self._update_health_panel(health)
            else:
                # graceful fallback
                self.halt_banner.hide()

            # === 4️⃣ Positions Table ===
            if hasattr(self, "pos_model"):
                for sym, info in positions.items():
                    self._update_position_row(info)

            # === 5️⃣ Alerts Tab ===
            if hasattr(self, "alerts_list"):
                self.alerts_list.clear()
                for a in alerts[-10:]:
                    msg = a.get("text") or a.get("message") or str(a)
                    self.alerts_list.addItem(msg)

            # === 6️⃣ Trades (visual markers on equity chart) ===
            if hasattr(self, "_eq_x") and trades:
                # Avoid repeated marking of same trade
                recent_ids = getattr(self, "_recent_trade_ids", set())
                for t in trades[-5:]:
                    tid = t.get("id") or (t.get("symbol"), t.get("timestamp"))
                    if tid in recent_ids:
                        continue
                    self._handle_new_trade(t)
                    recent_ids.add(tid)
                # Trim stored IDs
                self._recent_trade_ids = set(list(recent_ids)[-50:])

            # === 7️⃣ Optional: regime display (future use) ===
            if "regime" in snap and snap["regime"]:
                self._append_log(f"[REGIME] {snap['regime']}")

        except Exception as e:
            import traceback
            self._append_log(f"[ERR] Snapshot update failed: {e}\n{traceback.format_exc()}")


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

        # === Mode + Symbol Input (use existing from toolbar) ===
        # Note: mode_combo and symbol_input are already created in the toolbar
        # We add references here for ops tab context
        ops_mode_label = QtWidgets.QLabel("Mode:")
        ops_symbol_label = QtWidgets.QLabel("Symbols:")
        ops_mode_display = QtWidgets.QLabel(self.mode_combo.currentText())
        ops_symbol_display = QtWidgets.QLabel(self.symbol_input.text())

        # Update displays when main widgets change
        self.mode_combo.currentTextChanged.connect(ops_mode_display.setText)
        self.symbol_input.textChanged.connect(ops_symbol_display.setText)

        hl.addWidget(ops_mode_label)
        hl.addWidget(ops_mode_display)
        hl.addWidget(ops_symbol_label)
        hl.addWidget(ops_symbol_display)

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
        else: self.panic_btn.setChecked(False); self.panic_btn.setText("HALT ✖")

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

    async def _start_sim(self):
        """Launch SimulationRunner (GBM-based) using GUI parameters."""

        # Verify feeder is ready (subscriptions done synchronously at init)
        pnl_count = len(self.bus.listeners.get("PNL_UPDATE", []))
        bar_count = len(self.bus.listeners.get("NEW_BAR", []))
        self._append_log(f"[SIM] EventBus: PNL={pnl_count}, BAR={bar_count} listeners")

        if pnl_count == 0:
            self._append_log("[ERROR] No PNL listeners! Feeder not initialized properly.")
            return

        # --- GUI parameters ---
        raw = self.symbol_input.text().strip()
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()] or ["AAPL"]

        steps = self.sim_steps_spin.value()
        sleep_time = self.sim_speed_spin.value()
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
        )

        # --- Instantiate the SimulationRunner ---
        self._sim_runner = SimulationRunner(cfg)
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
            import traceback
            self._append_log(f"[SIM] Error: {e}")
            self._append_log(traceback.format_exc())
        finally:
            self._sim_running = False

    def _stop_simulation(self):
        if getattr(self, "_sim_running", False):
            self._sim_running = False
            self._append_log("[SIM] Simulation manually stopped by user.")
        else:
            self._append_log("[SIM] No active simulation to stop.")