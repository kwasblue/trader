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
from monitoring.views.symbol_list_widget import SymbolListWidget
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

        # === Global Dark Theme ===
        self.setStyleSheet("""
            QMainWindow {
                background: #0a0a0a;
            }
            QWidget {
                background: #0a0a0a;
                color: #e5e5e5;
                font-family: 'Segoe UI', 'SF Pro Display', -apple-system, sans-serif;
            }
            QTabWidget::pane {
                border: none;
                background: #0a0a0a;
            }
            QTabBar::tab {
                background: #1a1a2e;
                color: #94a3b8;
                padding: 10px 20px;
                border: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #16213e;
                color: #e5e5e5;
                border-bottom: 2px solid #3b82f6;
            }
            QTabBar::tab:hover:!selected {
                background: #1e293b;
            }
            QToolBar {
                background: #1a1a2e;
                border: none;
                spacing: 8px;
                padding: 6px;
            }
            QToolButton {
                background: #374151;
                color: #e5e5e5;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QToolButton:hover {
                background: #4b5563;
            }
            QToolButton:pressed {
                background: #1f2937;
            }
            QPushButton {
                background: #374151;
                color: #e5e5e5;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #4b5563;
            }
            QPushButton:pressed {
                background: #1f2937;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background: #1f2937;
                color: #e5e5e5;
                border: 1px solid #374151;
                border-radius: 4px;
                padding: 6px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 1px solid #3b82f6;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 8px;
            }
            QComboBox QAbstractItemView {
                background: #1f2937;
                color: #e5e5e5;
                selection-background-color: #3b82f6;
            }
            QScrollBar:vertical {
                background: #1a1a2e;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #374151;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #4b5563;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QStatusBar {
                background: #1a1a2e;
                color: #94a3b8;
            }
            QPlainTextEdit {
                background: #0f0f0f;
                color: #e5e5e5;
                border: 1px solid #333;
                border-radius: 4px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
            QSlider::groove:horizontal {
                background: #374151;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #3b82f6;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #60a5fa;
            }
        """)

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

        # Day trade checkbox
        self.day_trade_checkbox = QtWidgets.QCheckBox("Day Trade")
        self.day_trade_checkbox.setToolTip("Enable day trading (allow same-day exits)")

        for a in [self.start_act, self.stop_act, self.clear_logs_act, self.export_csv_act, self.export_pdf_act]:
            tb.addAction(a)
        tb.addSeparator()
        tb.addWidget(self.mode_label)
        tb.addWidget(self.mode_combo)
        tb.addWidget(self.symbol_label)
        tb.addWidget(self.symbol_input)
        tb.addWidget(self.day_trade_checkbox)
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
        self._build_lists_tab()
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

        # Strategy signals
        self.feeder.s.strategy_signal.connect(self._gui_on_strategy_signal)

        # History and Replay tabs
        self.feeder.s.history_update.connect(self._gui_on_history)
        self.feeder.s.benchmark_update.connect(self._gui_on_benchmark)
        self.feeder.s.replay_frame.connect(self._gui_on_replay_frame)

        # Market tab
        self.feeder.s.news_update.connect(self._gui_on_news)
        self.feeder.s.regime_update.connect(self._gui_on_regime)

        self._append_log("[INIT] Feeder Qt signals connected to GUI handlers.")

    # ================================================================
    # GUI HANDLERS (SYNC) - Connected to DataFeeder Qt signals
    # ================================================================

    def _gui_on_pnl(self, data: dict):
        """Handle P&L update from feeder Qt signal (SYNC)."""
        try:
            value = data.get('portfolio_value', 0)
            unrealized = data.get('unrealized', 0)
            realized = data.get('realized', 0)
            drawdown = data.get('drawdown', 0)

            # Log every 10th update to avoid spam
            if len(self._eq_y) % 10 == 0:
                self._append_log(f"[PNL] Portfolio: ${value:,.2f} (update #{len(self._eq_y)})")

            # === Update Dashboard KPIs ===
            # Portfolio Value (big label)
            if hasattr(self, 'portfolio_value_lbl') and value:
                color = "#22c55e" if value >= 100000 else "#ef4444"
                self.portfolio_value_lbl.setText(f"${value:,.2f}")
                self.portfolio_value_lbl.setStyleSheet(f"font-weight: 700; font-size: 20px; color: {color};")

            # Total P&L
            if hasattr(self, 'total_pnl_lbl'):
                total_pnl = unrealized + realized
                self._set_kpi(self.total_pnl_lbl, total_pnl, money=True)

            # Daily P&L (use realized for now)
            if hasattr(self, 'daily_pnl_lbl'):
                self._set_kpi(self.daily_pnl_lbl, realized, money=True)

            # Standard KPI labels
            if 'unrealized' in data:
                self._set_kpi(self.unreal_lbl, unrealized, money=True)
            if 'realized' in data:
                self._set_kpi(self.realized_lbl, realized, money=True)
            if 'drawdown' in data:
                self._set_kpi(self.dd_lbl, drawdown, pct=True)

            # Buying Power and Cash from broker
            if hasattr(self, 'buying_power_lbl'):
                buying_power = data.get('buying_power')
                if buying_power is not None:
                    self.buying_power_lbl.setText(f"${buying_power:,.0f}")
                else:
                    # Fallback: estimate from portfolio value - unrealized
                    cash = value - unrealized if value and unrealized else value
                    self.buying_power_lbl.setText(f"${cash:,.0f}")

            # Cash label if exists
            if hasattr(self, 'cash_lbl'):
                cash = data.get('cash')
                if cash is not None:
                    self.cash_lbl.setText(f"${cash:,.0f}")

            # Update equity curve
            if value:
                self._eq_x.append(len(self._eq_x))
                self._eq_y.append(float(value))
                self._update_equity_chart()

                # Update status bar
                if hasattr(self, 'status_bars_lbl'):
                    self.status_bars_lbl.setText(f"Bars: {len(self._eq_y)}")

                # Calculate win rate from equity changes
                if len(self._eq_y) > 2 and hasattr(self, 'win_rate_lbl'):
                    returns = np.diff(self._eq_y[-100:])
                    wins = np.sum(returns > 0)
                    total = len(returns)
                    win_rate = wins / total if total > 0 else 0
                    self._set_kpi(self.win_rate_lbl, win_rate, pct=True)

                # Update performance metrics every 20 updates
                if len(self._eq_y) % 20 == 0 and len(self._eq_y) > 20:
                    self._calculate_and_update_performance()

                # Auto-update History tab every 30 updates
                if len(self._eq_y) % 30 == 0 and len(self._eq_y) > 30:
                    self._refresh_history_from_session()

                # Auto-update Replay tab
                if hasattr(self, 'replay_slider') and len(self._eq_y) > 10:
                    self.replay_slider.setMaximum(len(self._eq_y) - 1)
                    self.replay_frame_lbl.setText(f"Frame: {self.replay_slider.value()} / {len(self._eq_y)}")

                    # Update replay history and curve
                    self._replay_history = list(zip(range(len(self._eq_y)), self._eq_y))
                    if len(self._eq_y) % 20 == 0:  # Update curve every 20 points
                        x = [p[0] for p in self._replay_history]
                        y = [p[1] for p in self._replay_history]
                        self.replay_curve.setData(x, y)

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
            open_price = bar.get('open', close)

            # Debug: Count and log bars
            if not hasattr(self, '_gui_bar_count'):
                self._gui_bar_count = 0
            self._gui_bar_count += 1

            if self._gui_bar_count % 20 == 0:
                self._append_log(f"[BAR] {symbol}: ${close:.2f} (bar #{self._gui_bar_count})")
                print(f"[MainWindow] BAR #{self._gui_bar_count}: {symbol} ${close:.2f}")

            # Update dashboard symbol prices
            if hasattr(self, '_update_symbol_price') and close > 0:
                # Calculate change percentage from open
                change_pct = ((close - open_price) / open_price * 100) if open_price > 0 else 0
                self._update_symbol_price(symbol, close, change_pct)

            # Update status bar symbols
            if hasattr(self, 'status_symbols_lbl'):
                if not hasattr(self, '_tracked_symbols'):
                    self._tracked_symbols = set()
                self._tracked_symbols.add(symbol)
                self.status_symbols_lbl.setText(f"Symbols: {', '.join(sorted(self._tracked_symbols))}")

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
            timestamp = data.get('timestamp', '')

            self._append_log(f"[TRADE] {side.upper()} {qty} {symbol} @ ${price:.2f}")

            # Update trade counter
            if not hasattr(self, '_trade_count'):
                self._trade_count = 0
            self._trade_count += 1

            # Update status bar
            if hasattr(self, 'status_trades_lbl'):
                self.status_trades_lbl.setText(f"Trades: {self._trade_count}")

            # Add to recent trades table on Dashboard
            if hasattr(self, 'recent_trades_table'):
                row = 0  # Insert at top
                self.recent_trades_table.insertRow(row)

                # Format timestamp
                time_str = timestamp[-8:] if timestamp else datetime.now().strftime("%H:%M:%S")

                self.recent_trades_table.setItem(row, 0, QtWidgets.QTableWidgetItem(time_str))
                self.recent_trades_table.setItem(row, 1, QtWidgets.QTableWidgetItem(symbol))

                # Color-code side
                side_item = QtWidgets.QTableWidgetItem(side.upper())
                if side.lower() in ('buy', 'long'):
                    side_item.setForeground(QtGui.QColor(34, 197, 94))  # Green
                else:
                    side_item.setForeground(QtGui.QColor(239, 68, 68))  # Red
                self.recent_trades_table.setItem(row, 2, side_item)

                self.recent_trades_table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(qty)))
                self.recent_trades_table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"${price:.2f}"))

                # Keep only last 10 trades in the mini table
                while self.recent_trades_table.rowCount() > 10:
                    self.recent_trades_table.removeRow(self.recent_trades_table.rowCount() - 1)

            self._handle_new_trade(data)
            self.log_event(events.EVENT_NEW_TRADE, data)
        except Exception as e:
            self._append_log(f"[ERR] Trade update failed: {e}")

    def _gui_on_position(self, data: dict):
        """Handle position update from feeder (SYNC)."""
        try:
            symbol = data.get('symbol', '')
            qty = data.get('qty', 0)

            if hasattr(self, 'pos_model'):
                if qty != 0:
                    # Update or add position
                    if hasattr(self.pos_model, 'update_position'):
                        self.pos_model.update_position(data)
                else:
                    # Remove position when qty is 0
                    if hasattr(self.pos_model, 'remove_position'):
                        self.pos_model.remove_position(symbol)

            # Track open positions
            if not hasattr(self, '_open_positions'):
                self._open_positions = {}

            if symbol:
                if qty != 0:
                    self._open_positions[symbol] = data
                elif symbol in self._open_positions:
                    del self._open_positions[symbol]

                # Update open positions count on dashboard
                if hasattr(self, 'open_positions_lbl'):
                    self.open_positions_lbl.setText(str(len(self._open_positions)))

            self.log_event(events.EVENT_POSITION_UPDATE, data)
        except Exception as e:
            self._append_log(f"[ERR] Position update failed: {e}")

    def _gui_on_order(self, data: dict):
        """Handle order status update from feeder (SYNC)."""
        try:
            order_id = data.get('order_id', 'N/A')
            status = data.get('status', 'unknown')
            symbol = data.get('symbol', 'N/A')
            side = data.get('side', 'N/A')
            qty = data.get('filled_qty', data.get('qty', 0))
            price = data.get('avg_price', data.get('price', 0))
            timestamp = data.get('timestamp', '')

            self._append_log(f"[ORDER] {symbol} {side} {qty} @ ${price:.2f} - {status}")

            # Update order counts
            self._update_order_kpis(data)

            # Add to order history table
            if hasattr(self, 'order_table'):
                row = self.order_table.rowCount()
                self.order_table.insertRow(row)

                # Format timestamp
                time_str = timestamp[-8:] if timestamp else 'N/A'

                self.order_table.setItem(row, 0, QtWidgets.QTableWidgetItem(time_str))
                self.order_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(symbol)))
                self.order_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(side).upper()))
                self.order_table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(qty)))
                self.order_table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"${price:.2f}"))

                # Color-code status
                status_item = QtWidgets.QTableWidgetItem(status)
                if status == 'filled':
                    status_item.setForeground(QtGui.QColor(34, 197, 94))  # Green
                elif status in ('canceled', 'rejected'):
                    status_item.setForeground(QtGui.QColor(248, 113, 113))  # Red
                self.order_table.setItem(row, 5, status_item)

                # Scroll to bottom
                self.order_table.scrollToBottom()

                # Keep only last 100 orders
                while self.order_table.rowCount() > 100:
                    self.order_table.removeRow(0)

            self.log_event(events.EVENT_ORDER_STATUS, data)
        except Exception as e:
            self._append_log(f"[ERR] Order update failed: {e}")

    def _gui_on_alert(self, data: dict):
        """Handle alert from feeder (SYNC)."""
        try:
            level = data.get('level', 'info') if isinstance(data, dict) else 'info'
            message = data.get('message', str(data)) if isinstance(data, dict) else str(data)
            symbol = data.get('symbol', '') if isinstance(data, dict) else ''
            timestamp = data.get('timestamp', '') if isinstance(data, dict) else ''

            self._append_log(f"[ALERT] {level.upper()}: {message}")

            # Update alert counts
            if not hasattr(self, '_alert_counts'):
                self._alert_counts = {"error": 0, "warning": 0, "info": 0, "total": 0}

            self._alert_counts["total"] += 1
            level_lower = level.lower()
            if level_lower in self._alert_counts:
                self._alert_counts[level_lower] += 1

            self._update_alert_counts()

            # Add to alerts list with formatting
            if hasattr(self, 'alerts_list'):
                time_str = timestamp[-8:] if timestamp else datetime.now().strftime("%H:%M:%S")
                sym_str = f"[{symbol}] " if symbol else ""
                item_text = f"[{time_str}] {level.upper()}: {sym_str}{message}"

                item = QtWidgets.QListWidgetItem(item_text)

                # Color-code by level
                if level_lower == 'error':
                    item.setForeground(QtGui.QColor(239, 68, 68))  # Red
                    item.setBackground(QtGui.QColor(50, 20, 20))
                elif level_lower == 'warning':
                    item.setForeground(QtGui.QColor(245, 158, 11))  # Orange
                    item.setBackground(QtGui.QColor(50, 40, 10))
                else:
                    item.setForeground(QtGui.QColor(59, 130, 246))  # Blue

                self.alerts_list.insertItem(0, item)  # Add to top

                # Keep only last 200 alerts
                while self.alerts_list.count() > 200:
                    self.alerts_list.takeItem(self.alerts_list.count() - 1)

            self.log_event(events.EVENT_ALERT, data)
        except Exception as e:
            self._append_log(f"[ERR] Alert update failed: {e}")

    def _update_alert_counts(self):
        """Update alert count labels."""
        if hasattr(self, 'alert_total_lbl'):
            self.alert_total_lbl.setText(str(self._alert_counts.get("total", 0)))
        if hasattr(self, 'alert_error_lbl'):
            self.alert_error_lbl.setText(str(self._alert_counts.get("error", 0)))
        if hasattr(self, 'alert_warning_lbl'):
            self.alert_warning_lbl.setText(str(self._alert_counts.get("warning", 0)))
        if hasattr(self, 'alert_info_lbl'):
            self.alert_info_lbl.setText(str(self._alert_counts.get("info", 0)))

    def _gui_on_strategy_signal(self, data: dict):
        """Handle strategy signal from feeder (SYNC)."""
        try:
            symbol = data.get('symbol', 'N/A')
            strategy = data.get('strategy', 'unknown')
            signal_raw = data.get('signal', 0)
            regime = data.get('regime', 'N/A')
            timestamp = data.get('timestamp', '')

            # Map signal to int and text
            if signal_raw in (1, 'buy'):
                signal = 1
            elif signal_raw in (-1, 'sell'):
                signal = -1
            else:
                signal = 0
            signal_text = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}.get(signal, str(signal_raw))

            # Count signals
            if not hasattr(self, '_signal_count'):
                self._signal_count = 0
            self._signal_count += 1

            # Update signal counters
            if not hasattr(self, '_signal_counts'):
                self._signal_counts = {"buy": 0, "sell": 0, "hold": 0, "total": 0}
            self._signal_counts["total"] += 1
            if signal == 1:
                self._signal_counts["buy"] += 1
            elif signal == -1:
                self._signal_counts["sell"] += 1
            else:
                self._signal_counts["hold"] += 1

            # Update Strategy tab KPIs
            if hasattr(self, 'strat_signals_lbl'):
                self.strat_signals_lbl.setText(str(self._signal_counts["total"]))
            if hasattr(self, 'strat_buy_lbl'):
                self.strat_buy_lbl.setText(str(self._signal_counts["buy"]))
            if hasattr(self, 'strat_sell_lbl'):
                self.strat_sell_lbl.setText(str(self._signal_counts["sell"]))

            # Log signal (only non-HOLD to reduce spam, or every 50th)
            if signal != 0:
                self._append_log(f"[SIGNAL] {symbol}/{strategy}: {signal_text} (regime={regime})")

                # Add to signal history list
                if hasattr(self, 'signal_history_list'):
                    time_str = timestamp[-8:] if timestamp else datetime.now().strftime("%H:%M:%S")
                    item_text = f"[{time_str}] {symbol}/{strategy}: {signal_text}"
                    item = QtWidgets.QListWidgetItem(item_text)
                    if signal == 1:
                        item.setForeground(QtGui.QColor(34, 197, 94))  # Green
                    else:
                        item.setForeground(QtGui.QColor(239, 68, 68))  # Red
                    self.signal_history_list.insertItem(0, item)

                    # Keep only last 50
                    while self.signal_history_list.count() > 50:
                        self.signal_history_list.takeItem(self.signal_history_list.count() - 1)

            # Track per-strategy signal counts
            if not hasattr(self, '_strategy_signal_counts'):
                self._strategy_signal_counts = {}
            key = f"{symbol}/{strategy}"
            if key not in self._strategy_signal_counts:
                self._strategy_signal_counts[key] = 0
            if signal != 0:
                self._strategy_signal_counts[key] += 1

            # Update strategies table
            if hasattr(self, 'sig_table'):
                # Find or create row for this symbol/strategy
                row = -1
                for i in range(self.sig_table.rowCount()):
                    item = self.sig_table.item(i, 0)
                    if item and item.text() == key:
                        row = i
                        break

                if row < 0:
                    row = self.sig_table.rowCount()
                    self.sig_table.insertRow(row)

                    # Update active strategies count
                    if hasattr(self, 'strat_active_lbl'):
                        self.strat_active_lbl.setText(str(self.sig_table.rowCount()))

                # Update cells
                self.sig_table.setItem(row, 0, QtWidgets.QTableWidgetItem(key))

                signal_item = QtWidgets.QTableWidgetItem(signal_text)
                if signal == 1:
                    signal_item.setBackground(QtGui.QColor(22, 101, 52))  # Dark green
                    signal_item.setForeground(QtGui.QColor(255, 255, 255))
                elif signal == -1:
                    signal_item.setBackground(QtGui.QColor(153, 27, 27))  # Dark red
                    signal_item.setForeground(QtGui.QColor(255, 255, 255))
                self.sig_table.setItem(row, 1, signal_item)

                # Regime with color
                regime_item = QtWidgets.QTableWidgetItem(regime)
                if 'high' in regime.lower():
                    regime_item.setForeground(QtGui.QColor(239, 68, 68))
                elif 'low' in regime.lower():
                    regime_item.setForeground(QtGui.QColor(34, 197, 94))
                self.sig_table.setItem(row, 2, regime_item)

                self.sig_table.setItem(row, 3, QtWidgets.QTableWidgetItem(timestamp[-8:] if timestamp else 'N/A'))
                self.sig_table.setItem(row, 4, QtWidgets.QTableWidgetItem(str(self._strategy_signal_counts.get(key, 0))))

            self.log_event(events.EVENT_STRATEGY_SIGNAL, data)
        except Exception as e:
            self._append_log(f"[ERR] Strategy signal update failed: {e}")

    def _gui_on_history(self, data: dict):
        """Handle history update from feeder (SYNC) - updates History tab."""
        try:
            pnl_by_day = data.get('pnl_by_day', {})
            if not pnl_by_day:
                return

            # Store history data for calendar plot
            if not hasattr(self, '_history_data'):
                self._history_data = {}
            self._history_data.update(pnl_by_day)

            # Update calendar plot
            self._update_calendar_plot()
        except Exception as e:
            self._append_log(f"[ERR] History update failed: {e}")

    def _gui_on_benchmark(self, data: dict):
        """Handle benchmark update from feeder (SYNC) - updates History tab."""
        try:
            equity = data.get('equity_curve', [])
            benchmark = data.get('benchmark_curve', [])

            if not equity or not benchmark:
                return

            # Plot equity vs benchmark
            x = list(range(len(equity)))
            self.bench_plot.clear()
            self.bench_plot.plot(x, equity, pen=pg.mkPen("#22c55e", width=2), name="Strategy")
            self.bench_plot.plot(x, benchmark, pen=pg.mkPen("#64748b", width=2), name="Benchmark")
            self.bench_plot.addLegend()
        except Exception as e:
            self._append_log(f"[ERR] Benchmark update failed: {e}")

    def _gui_on_replay_frame(self, data: dict):
        """Handle replay frame from feeder (SYNC) - updates Replay tab."""
        try:
            frame_idx = data.get('frame_idx', 0)
            equity = data.get('equity', 0)

            # Update replay curve
            if not hasattr(self, '_replay_data'):
                self._replay_data = {'x': [], 'y': []}

            self._replay_data['x'].append(frame_idx)
            self._replay_data['y'].append(equity)

            # Limit to last 1000 points
            if len(self._replay_data['x']) > 1000:
                self._replay_data['x'] = self._replay_data['x'][-1000:]
                self._replay_data['y'] = self._replay_data['y'][-1000:]

            self.replay_curve.setData(self._replay_data['x'], self._replay_data['y'])
        except Exception as e:
            self._append_log(f"[ERR] Replay frame update failed: {e}")

    def _gui_on_news(self, data: dict):
        """Handle news update from feeder (SYNC) - updates Market tab."""
        try:
            headline = data.get('headline', 'No headline')
            source = data.get('source', 'Unknown')
            sentiment = data.get('sentiment', 'neutral')
            timestamp = data.get('timestamp', '')

            # Add to news list
            if hasattr(self, 'news_list'):
                # Color based on sentiment
                item_text = f"[{source}] {headline}"
                item = QtWidgets.QListWidgetItem(item_text)
                if sentiment == 'positive':
                    item.setForeground(QtGui.QColor(34, 197, 94))  # Green
                elif sentiment == 'negative':
                    item.setForeground(QtGui.QColor(248, 113, 113))  # Red
                else:
                    item.setForeground(QtGui.QColor(148, 163, 184))  # Gray

                self.news_list.insertItem(0, item)  # Add to top

                # Keep only last 50 items
                while self.news_list.count() > 50:
                    self.news_list.takeItem(self.news_list.count() - 1)
        except Exception as e:
            self._append_log(f"[ERR] News update failed: {e}")

    def _gui_on_regime(self, data: dict):
        """Handle regime update from feeder (SYNC)."""
        try:
            symbol = data.get('symbol', 'N/A')
            volatility = data.get('volatility', 'unknown')
            trend = data.get('trend', 'unknown')
            market = data.get('market', 'unknown')
            timestamp = data.get('timestamp', '')

            # Store last regime per symbol to detect changes
            if not hasattr(self, '_last_regimes'):
                self._last_regimes = {}

            last = self._last_regimes.get(symbol)
            current = (volatility, trend)

            # Only log when regime changes
            if last != current:
                self._last_regimes[symbol] = current
                self._append_log(f"[REGIME] {symbol}: vol={volatility}, trend={trend}, market={market}")

                # Add to regime history list on Market tab
                if hasattr(self, 'regime_list'):
                    time_str = timestamp[-8:] if timestamp else datetime.now().strftime("%H:%M:%S")
                    item_text = f"[{time_str}] {symbol}: {volatility} / {trend}"
                    item = QtWidgets.QListWidgetItem(item_text)

                    # Color-code based on volatility
                    if 'high' in volatility.lower():
                        item.setForeground(QtGui.QColor(239, 68, 68))  # Red
                    elif 'low' in volatility.lower():
                        item.setForeground(QtGui.QColor(34, 197, 94))  # Green
                    else:
                        item.setForeground(QtGui.QColor(148, 163, 184))  # Gray

                    self.regime_list.insertItem(0, item)  # Add to top

                    # Keep only last 50 items
                    while self.regime_list.count() > 50:
                        self.regime_list.takeItem(self.regime_list.count() - 1)

            # Update Market tab KPIs
            if hasattr(self, 'market_regime_lbl'):
                self.market_regime_lbl.setText(volatility.upper())
                color = "#ef4444" if 'high' in volatility.lower() else "#22c55e" if 'low' in volatility.lower() else "#94a3b8"
                self.market_regime_lbl.setStyleSheet(f"font-weight: 700; font-size: 16px; color: {color};")

            if hasattr(self, 'market_volatility_lbl'):
                self.market_volatility_lbl.setText(volatility)

            if hasattr(self, 'market_trend_lbl'):
                self.market_trend_lbl.setText(trend.capitalize())
                color = "#22c55e" if trend == 'bullish' else "#ef4444" if trend == 'bearish' else "#94a3b8"
                self.market_trend_lbl.setStyleSheet(f"font-weight: 700; font-size: 16px; color: {color};")

            # Update strategy table regime column if symbol exists
            if hasattr(self, 'sig_table'):
                for i in range(self.sig_table.rowCount()):
                    item = self.sig_table.item(i, 0)
                    if item and item.text().startswith(symbol):
                        self.sig_table.setItem(i, 2, QtWidgets.QTableWidgetItem(volatility))
        except Exception as e:
            self._append_log(f"[ERR] Regime update failed: {e}")

    def _update_calendar_plot(self):
        """Update the PnL calendar heat map."""
        try:
            if not hasattr(self, '_history_data') or not self._history_data:
                return

            # Convert to sorted lists for plotting
            dates = sorted(self._history_data.keys())
            pnls = [self._history_data[d] for d in dates]

            # Simple bar chart representation
            x = list(range(len(dates)))
            colors = ['g' if p >= 0 else 'r' for p in pnls]

            self.calendar_plot.clear()
            bar = pg.BarGraphItem(x=x, height=pnls, width=0.8, brushes=colors)
            self.calendar_plot.addItem(bar)
            self.calendar_plot.setLabel('bottom', 'Day')
            self.calendar_plot.setLabel('left', 'PnL ($)')
        except Exception as e:
            self._append_log(f"[ERR] Calendar plot update failed: {e}")

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

        # Update status bar
        if hasattr(self, 'status_mode_lbl'):
            self.status_mode_lbl.setText(f"Mode: {mode}")

        # Mode change just selects - Start button actually starts trading

    def _start_trading(self):
        """Start trading with the selected mode."""
        # Check if already running
        if getattr(self, '_trading_active', False) or getattr(self, '_sim_running', False):
            self._append_log("[ERROR] Trading/Simulation already running!")
            return

        mode = getattr(self, '_current_mode', 'Simulation')
        symbols_text = self.symbol_input.text() if hasattr(self, 'symbol_input') else "AAPL,MSFT"
        symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]

        if not symbols:
            self._append_log("[ERROR] No symbols specified")
            return

        # Enable day trade mode if checkbox is checked
        day_trade = self.day_trade_checkbox.isChecked() if hasattr(self, 'day_trade_checkbox') else False
        if day_trade:
            from core.config_loader import enable_day_trade_mode
            enable_day_trade_mode()
            self._append_log("[CONFIG] Day trade mode enabled (swing mode disabled)")

        self._trading_active = True
        self._sim_running = True  # Also set sim_running for consistency
        self._append_log(f"[START] Starting {mode} mode for: {', '.join(symbols)}")

        # Store the task so we can cancel it later
        self._backend_task = asyncio.create_task(self._run_backend(mode, symbols))

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
        try:
            await self._sim_runner.run()
            self._append_log(f"[SIM] Completed. Final equity: ${self._sim_runner.portfolio.total_equity():,.2f}")
        finally:
            self._sim_running = False
            self._trading_active = False

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
        if not getattr(self, '_trading_active', False) and not getattr(self, '_sim_running', False):
            self._append_log("[STOP] No active trading to stop.")
            return

        self._append_log("[STOP] Stopping trading...")

        # Stop the simulation runner if it exists
        if hasattr(self, '_sim_runner') and self._sim_runner is not None:
            self._sim_runner.stop()
            self._append_log("[STOP] Stop signal sent to SimulationRunner.")

        # Cancel the backend task if it exists
        if hasattr(self, '_backend_task') and self._backend_task is not None:
            self._backend_task.cancel()
            self._append_log("[STOP] Backend task cancellation requested.")

        # Stop alpaca runner if it exists
        if hasattr(self, '_alpaca_runner') and self._alpaca_runner is not None:
            if hasattr(self._alpaca_runner, 'stop'):
                self._alpaca_runner.stop()

        self._trading_active = False
        self._sim_running = False

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

        # Initialize counters if needed
        if not hasattr(self, '_order_counts'):
            self._order_counts = {"pending": 0, "filled": 0, "canceled": 0, "total": 0}

        self._order_counts["total"] += 1

        if status == 'submitted':
            self._order_counts["pending"] += 1
        elif status == 'filled':
            self._order_counts["filled"] += 1
            # Decrement pending if we had one
            if self._order_counts["pending"] > 0:
                self._order_counts["pending"] -= 1
        elif status in ('canceled', 'rejected'):
            self._order_counts["canceled"] += 1
            if self._order_counts["pending"] > 0:
                self._order_counts["pending"] -= 1

        # Update labels
        self.q_pending.setText(str(self._order_counts["pending"]))
        if hasattr(self, 'q_filled'):
            self.q_filled.setText(str(self._order_counts["filled"]))
        self.q_canceled.setText(str(self._order_counts["canceled"]))
        if hasattr(self, 'q_total'):
            self.q_total.setText(str(self._order_counts["total"]))

    def _update_price_chart(self, price: dict):
        """Update Market tab price chart with latest simulated prices.

        Handles multiple input formats:
        1. Direct price: {"symbol": str, "price": float, "timestamp": str}
        2. OHLC data: {"symbol": str, "data": DataFrame or dict}
        """
        sym = price.get("symbol")
        if not sym:
            return

        if not hasattr(self, "_price_data"):
            self._price_data = {}

        self._price_data.setdefault(sym, {"x": [], "y": []})
        d = self._price_data[sym]

        px = None
        ts = None

        # Handle "data" field (can be DataFrame or dict)
        if "data" in price:
            data = price["data"]

            # Check if it's a DataFrame
            if data is not None and hasattr(data, 'empty') and not data.empty:
                close_col = "Close" if "Close" in data.columns else "close"
                if close_col in data.columns:
                    px = float(data[close_col].iloc[-1])
                    if hasattr(data.index, 'to_pydatetime'):
                        try:
                            ts = data.index[-1].timestamp()
                        except Exception:
                            ts = len(d["x"])
                    else:
                        ts = len(d["x"])

            # Check if it's a dict with OHLC data
            elif isinstance(data, dict):
                px = float(data.get('close', data.get('Close', 0)))
                ts_raw = data.get('timestamp', '')
                if ts_raw:
                    try:
                        if isinstance(ts_raw, str):
                            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).timestamp()
                        else:
                            ts = len(d["x"])
                    except Exception:
                        ts = len(d["x"])
                else:
                    ts = len(d["x"])

        # Handle direct price format (no "data" field)
        if px is None:
            px = float(price.get("price", price.get("close", np.nan)))
            ts_raw = price.get("timestamp")
            try:
                ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00")).timestamp()
            except Exception:
                ts = len(d["x"])

        # Add to data if we have a valid price
        if px is not None and not np.isnan(px) and px > 0:
            d["x"].append(ts if ts is not None else len(d["x"]))
            d["y"].append(px)

        # Keep last 500 points for performance
        if len(d["x"]) > 500:
            d["x"] = d["x"][-500:]
            d["y"] = d["y"][-500:]

        # Debug: Log chart updates periodically
        if len(d["x"]) % 20 == 0 and len(d["x"]) > 0:
            print(f"[MainWindow] Price chart {sym}: {len(d['x'])} points, last=${d['y'][-1]:.2f}")

        # Update chart
        self.price_plot.clear()
        self.price_plot.plot(d["x"], d["y"], pen=pg.mkPen("#22c55e", width=2))
        self.price_plot.setTitle(f"Price: {sym} (${d['y'][-1]:.2f})" if d["y"] else "Price Chart")


    def _append_log(self, msg: str):
        """
        Append a log message to the logs_view in Ops tab.
        """
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        if hasattr(self, "logs_view"):
            self.logs_view.appendPlainText(f"[{ts}] {msg}")
    
    def _calculate_and_update_performance(self):
        """Calculate and update performance metrics from equity curve."""
        import numpy as np
        try:
            if len(self._eq_y) < 20:
                return

            equity = np.array(self._eq_y[-500:])  # Use last 500 points
            returns = np.diff(equity) / equity[:-1]

            if len(returns) < 2:
                return

            # === Risk-Adjusted Metrics ===
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)

            # Sharpe Ratio (annualized)
            sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

            # Sortino Ratio (downside deviation only)
            downside = returns[returns < 0]
            downside_std = np.std(downside) if len(downside) > 1 else std_ret
            sortino = (mean_ret / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

            # Calmar Ratio (return / max drawdown)
            peak = np.maximum.accumulate(equity)
            drawdown = (equity - peak) / peak
            max_dd = abs(np.min(drawdown))
            total_return = (equity[-1] - equity[0]) / equity[0]
            calmar = (total_return / max_dd) if max_dd > 0 else 0.0

            # === Trade Statistics ===
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            total_trades = len(returns)
            win_count = len(wins)
            loss_count = len(losses)

            win_rate = win_count / total_trades if total_trades > 0 else 0
            avg_win = np.mean(wins) * 100 if len(wins) > 0 else 0  # As percentage
            avg_loss = np.mean(losses) * 100 if len(losses) > 0 else 0
            max_win = np.max(wins) * 100 if len(wins) > 0 else 0
            max_loss = np.min(losses) * 100 if len(losses) > 0 else 0

            # Profit Factor
            gross_profit = np.sum(wins) if len(wins) > 0 else 0
            gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

            # Expectancy
            expectancy = (win_rate * avg_win + (1 - win_rate) * avg_loss)

            # Kelly Criterion
            if len(wins) > 0 and len(losses) > 0 and abs(avg_loss) > 0:
                kelly = (win_rate - (1 - win_rate) / (abs(avg_win / avg_loss))) if avg_loss != 0 else 0.0
                kelly = max(0.0, min(1.0, kelly))
            else:
                kelly = 0.0

            # Debug output (reduced frequency)
            if len(self._eq_y) % 100 == 0:
                self._append_log(f"[PERF] Sharpe={sharpe:.2f}, Sortino={sortino:.2f}, Kelly={kelly:.2%}")

            # === Update UI ===
            self._update_performance_tab({
                "sharpe": round(sharpe, 2),
                "sortino": round(sortino, 2),
                "kelly": round(kelly, 2),
                "calmar": round(calmar, 2),
                "total_trades": total_trades,
                "win_rate": win_rate,
                "profit_factor": round(profit_factor, 2),
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "max_win": max_win,
                "max_loss": max_loss,
                "max_dd": max_dd,
                "expectancy": expectancy,
                "returns": returns,
                "drawdown": drawdown,
            })
        except Exception as e:
            self._append_log(f"[ERR] Performance calc failed: {e}")
            import traceback
            print(f"[MainWindow] Performance error: {e}\n{traceback.format_exc()}")

    def _update_performance_tab(self, metrics: dict):
        """Update all performance metrics."""
        if not metrics:
            return

        # === Risk-Adjusted Returns ===
        if metrics.get("sharpe") is not None:
            val = metrics["sharpe"]
            color = "#22c55e" if val > 1 else "#f59e0b" if val > 0 else "#ef4444"
            self.sharpe_lbl.setText(f"{val:.2f}")
            self.sharpe_lbl.setStyleSheet(f"font-weight: 700; font-size: 20px; color: {color};")

        if metrics.get("sortino") is not None:
            val = metrics["sortino"]
            color = "#22c55e" if val > 1.5 else "#f59e0b" if val > 0 else "#ef4444"
            self.sortino_lbl.setText(f"{val:.2f}")
            self.sortino_lbl.setStyleSheet(f"font-weight: 700; font-size: 20px; color: {color};")

        if metrics.get("kelly") is not None:
            val = metrics["kelly"]
            self.kelly_lbl.setText(f"{val*100:.1f}%")
            color = "#22c55e" if val > 0.1 else "#f59e0b" if val > 0 else "#ef4444"
            self.kelly_lbl.setStyleSheet(f"font-weight: 700; font-size: 20px; color: {color};")

        if metrics.get("calmar") is not None and hasattr(self, 'calmar_lbl'):
            val = metrics["calmar"]
            color = "#22c55e" if val > 1 else "#f59e0b" if val > 0 else "#ef4444"
            self.calmar_lbl.setText(f"{val:.2f}")
            self.calmar_lbl.setStyleSheet(f"font-weight: 700; font-size: 20px; color: {color};")

        # === Trade Statistics ===
        if metrics.get("total_trades") is not None and hasattr(self, 'perf_total_trades_lbl'):
            self.perf_total_trades_lbl.setText(str(metrics["total_trades"]))

        if metrics.get("win_rate") is not None and hasattr(self, 'perf_win_rate_lbl'):
            self._set_kpi(self.perf_win_rate_lbl, metrics["win_rate"], pct=True)

        if metrics.get("profit_factor") is not None and hasattr(self, 'perf_profit_factor_lbl'):
            val = metrics["profit_factor"]
            color = "#22c55e" if val > 1 else "#ef4444"
            self.perf_profit_factor_lbl.setText(f"{val:.2f}")
            self.perf_profit_factor_lbl.setStyleSheet(f"font-weight: 700; font-size: 16px; color: {color};")

        if metrics.get("avg_win") is not None and hasattr(self, 'perf_avg_win_lbl'):
            self._set_kpi(self.perf_avg_win_lbl, metrics["avg_win"], pct=True)

        if metrics.get("avg_loss") is not None and hasattr(self, 'perf_avg_loss_lbl'):
            self._set_kpi(self.perf_avg_loss_lbl, metrics["avg_loss"], pct=True)

        if metrics.get("expectancy") is not None and hasattr(self, 'perf_expectancy_lbl'):
            self._set_kpi(self.perf_expectancy_lbl, metrics["expectancy"], pct=True)

        if metrics.get("max_win") is not None and hasattr(self, 'perf_max_win_lbl'):
            self._set_kpi(self.perf_max_win_lbl, metrics["max_win"], pct=True)

        if metrics.get("max_loss") is not None and hasattr(self, 'perf_max_loss_lbl'):
            self._set_kpi(self.perf_max_loss_lbl, metrics["max_loss"], pct=True)

        if metrics.get("max_dd") is not None and hasattr(self, 'perf_max_dd_lbl'):
            self._set_kpi(self.perf_max_dd_lbl, -metrics["max_dd"], pct=True)

        # === Update Charts ===
        # Returns distribution histogram
        if "returns" in metrics and hasattr(self, 'returns_plot'):
            try:
                returns = metrics["returns"] * 100  # Convert to percentage
                y, x = np.histogram(returns, bins=30)
                self.returns_plot.clear()
                # Create bar chart from histogram
                bar = pg.BarGraphItem(x=x[:-1], height=y, width=(x[1]-x[0])*0.8,
                                      brush=pg.mkBrush('#3b82f6'))
                self.returns_plot.addItem(bar)
            except Exception:
                pass

        # Drawdown chart
        if "drawdown" in metrics and hasattr(self, 'dd_plot'):
            try:
                dd = metrics["drawdown"] * 100  # Convert to percentage
                x = list(range(len(dd)))
                self.dd_plot.clear()
                fill = pg.FillBetweenItem(
                    pg.PlotDataItem(x, dd),
                    pg.PlotDataItem(x, [0] * len(dd)),
                    brush=pg.mkBrush(239, 68, 68, 100)
                )
                self.dd_plot.addItem(fill)
                self.dd_plot.plot(x, dd, pen=pg.mkPen('#ef4444', width=1))
            except Exception:
                pass

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
        tab = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(tab)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # === TOP KPI BAR ===
        kpi_bar = QtWidgets.QFrame()
        kpi_bar.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a1a2e, stop:1 #16213e);
                border-radius: 8px;
                padding: 8px;
            }
        """)
        kpi_layout = QtWidgets.QHBoxLayout(kpi_bar)
        kpi_layout.setSpacing(20)

        # Portfolio Value (large, prominent)
        self.portfolio_value_lbl = self._big_kpi_label()
        self.portfolio_value_lbl.setText("$100,000.00")
        kpi_layout.addWidget(self._kpi_card("Portfolio Value", self.portfolio_value_lbl))

        # Daily P&L
        self.daily_pnl_lbl = self._kpi_label()
        kpi_layout.addWidget(self._kpi_card("Daily P&L", self.daily_pnl_lbl))

        # Total P&L
        self.total_pnl_lbl = self._kpi_label()
        kpi_layout.addWidget(self._kpi_card("Total P&L", self.total_pnl_lbl))

        # Win Rate
        self.win_rate_lbl = self._kpi_label()
        kpi_layout.addWidget(self._kpi_card("Win Rate", self.win_rate_lbl))

        # Open Positions
        self.open_positions_lbl = self._kpi_label()
        self.open_positions_lbl.setText("0")
        kpi_layout.addWidget(self._kpi_card("Positions", self.open_positions_lbl))

        # Drawdown
        self.dd_lbl = self._kpi_label()
        kpi_layout.addWidget(self._kpi_card("Drawdown", self.dd_lbl))

        kpi_layout.addStretch()
        main_layout.addWidget(kpi_bar)

        # === MIDDLE SECTION (Charts + Risk) ===
        middle = QtWidgets.QHBoxLayout()
        middle.setSpacing(10)

        # LEFT COLUMN: Risk Panel + Positions
        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(10)

        # Risk Panel
        risk_box = QtWidgets.QGroupBox("Risk & P&L")
        risk_box.setStyleSheet(self._group_box_style())
        rl = QtWidgets.QGridLayout(risk_box)
        rl.setSpacing(8)

        self.unreal_lbl = self._kpi_label()
        self.realized_lbl = self._kpi_label()
        self.cash_lbl = self._kpi_label()
        self.buying_power_lbl = self._kpi_label()

        rl.addWidget(self._metric_label("Unrealized P&L"), 0, 0)
        rl.addWidget(self.unreal_lbl, 0, 1)
        rl.addWidget(self._metric_label("Realized P&L"), 1, 0)
        rl.addWidget(self.realized_lbl, 1, 1)
        rl.addWidget(self._metric_label("Cash"), 2, 0)
        rl.addWidget(self.cash_lbl, 2, 1)
        rl.addWidget(self._metric_label("Buying Power"), 3, 0)
        rl.addWidget(self.buying_power_lbl, 3, 1)

        left_col.addWidget(risk_box)

        # Current Prices
        prices_box = QtWidgets.QGroupBox("Watched Symbols")
        prices_box.setStyleSheet(self._group_box_style())
        prices_layout = QtWidgets.QVBoxLayout(prices_box)

        self.symbol_prices_widget = QtWidgets.QWidget()
        self.symbol_prices_layout = QtWidgets.QGridLayout(self.symbol_prices_widget)
        self.symbol_prices_layout.setSpacing(4)
        self._symbol_price_labels = {}

        prices_layout.addWidget(self.symbol_prices_widget)
        left_col.addWidget(prices_box)

        # Positions Table
        pos_box = QtWidgets.QGroupBox("Open Positions")
        pos_box.setStyleSheet(self._group_box_style())
        pos_layout = QtWidgets.QVBoxLayout(pos_box)

        self.pos_model = SymbolsTableModel([])
        self.pos_table = QtWidgets.QTableView()
        self.pos_table.setModel(self.pos_model)
        self.pos_table.setStyleSheet("""
            QTableView {
                background: #0f0f0f;
                color: #e5e5e5;
                gridline-color: #333;
                border: none;
                font-size: 12px;
            }
            QHeaderView::section {
                background: #1a1a2e;
                color: #94a3b8;
                padding: 6px;
                border: none;
                font-weight: bold;
            }
        """)
        self.pos_table.horizontalHeader().setStretchLastSection(True)
        pos_layout.addWidget(self.pos_table)
        left_col.addWidget(pos_box, 1)

        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left_col)
        left_widget.setFixedWidth(320)

        # RIGHT COLUMN: Equity Chart
        right_col = QtWidgets.QVBoxLayout()

        # Equity Curve
        eq_box = QtWidgets.QGroupBox("Equity Curve")
        eq_box.setStyleSheet(self._group_box_style())
        eq_layout = QtWidgets.QVBoxLayout(eq_box)

        self.eq_plot = pg.PlotWidget()
        self.eq_plot.setBackground('#0a0a0a')
        self.eq_plot.showGrid(x=True, y=True, alpha=0.3)
        self.eq_plot.setLabel('left', 'Equity ($)')
        self.eq_plot.setLabel('bottom', 'Bar #')

        # Gradient fill under curve
        self.eq_curve = self.eq_plot.plot([], [], pen=pg.mkPen('#22c55e', width=2))
        self.entry_marks = pg.ScatterPlotItem(size=10, symbol='t', brush=pg.mkBrush(34, 197, 94, 200))
        self.exit_marks = pg.ScatterPlotItem(size=10, symbol='t1', brush=pg.mkBrush(239, 68, 68, 200))
        self.eq_plot.addItem(self.entry_marks)
        self.eq_plot.addItem(self.exit_marks)

        eq_layout.addWidget(self.eq_plot)
        right_col.addWidget(eq_box, 2)

        # Recent Trades (mini table)
        trades_box = QtWidgets.QGroupBox("Recent Trades")
        trades_box.setStyleSheet(self._group_box_style())
        trades_layout = QtWidgets.QVBoxLayout(trades_box)

        self.recent_trades_table = QtWidgets.QTableWidget(0, 5)
        self.recent_trades_table.setHorizontalHeaderLabels(["Time", "Symbol", "Side", "Qty", "Price"])
        self.recent_trades_table.setStyleSheet("""
            QTableWidget {
                background: #0f0f0f;
                color: #e5e5e5;
                gridline-color: #333;
                border: none;
                font-size: 11px;
            }
            QHeaderView::section {
                background: #1a1a2e;
                color: #94a3b8;
                padding: 4px;
                border: none;
            }
        """)
        self.recent_trades_table.horizontalHeader().setStretchLastSection(True)
        self.recent_trades_table.setMaximumHeight(150)
        trades_layout.addWidget(self.recent_trades_table)
        right_col.addWidget(trades_box)

        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right_col)

        middle.addWidget(left_widget)
        middle.addWidget(right_widget, 1)
        main_layout.addLayout(middle, 1)

        # === BOTTOM STATUS BAR ===
        status_bar = QtWidgets.QFrame()
        status_bar.setStyleSheet("""
            QFrame {
                background: #1a1a2e;
                border-radius: 6px;
                padding: 4px;
            }
        """)
        status_layout = QtWidgets.QHBoxLayout(status_bar)
        status_layout.setContentsMargins(10, 4, 10, 4)

        self.status_mode_lbl = QtWidgets.QLabel("Mode: Simulation")
        self.status_mode_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")

        self.status_symbols_lbl = QtWidgets.QLabel("Symbols: AAPL, MSFT")
        self.status_symbols_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")

        self.status_bars_lbl = QtWidgets.QLabel("Bars: 0")
        self.status_bars_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")

        self.status_trades_lbl = QtWidgets.QLabel("Trades: 0")
        self.status_trades_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")

        status_layout.addWidget(self.status_mode_lbl)
        status_layout.addWidget(QtWidgets.QLabel("|"))
        status_layout.addWidget(self.status_symbols_lbl)
        status_layout.addWidget(QtWidgets.QLabel("|"))
        status_layout.addWidget(self.status_bars_lbl)
        status_layout.addWidget(QtWidgets.QLabel("|"))
        status_layout.addWidget(self.status_trades_lbl)
        status_layout.addStretch()

        main_layout.addWidget(status_bar)

        self.tabs.addTab(tab, "Dashboard")

    def _group_box_style(self):
        """Return consistent GroupBox styling."""
        return """
            QGroupBox {
                background: #0f0f0f;
                border: 1px solid #333;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 8px;
                font-weight: bold;
                color: #94a3b8;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #e5e5e5;
            }
        """

    def _kpi_card(self, title: str, value_label: QtWidgets.QLabel) -> QtWidgets.QWidget:
        """Create a styled KPI card widget."""
        card = QtWidgets.QFrame()
        card.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 6px;
                padding: 4px;
            }
        """)
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        title_lbl = QtWidgets.QLabel(title)
        title_lbl.setStyleSheet("color: #64748b; font-size: 10px; font-weight: normal;")
        title_lbl.setAlignment(QtCore.Qt.AlignCenter)

        layout.addWidget(title_lbl)
        layout.addWidget(value_label)

        return card

    def _big_kpi_label(self) -> QtWidgets.QLabel:
        """Create a large KPI label for prominent values."""
        lbl = QtWidgets.QLabel("--")
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setStyleSheet("font-weight: 700; font-size: 20px; color: #22c55e;")
        return lbl

    def _metric_label(self, text: str) -> QtWidgets.QLabel:
        """Create a styled metric label."""
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet("color: #94a3b8; font-size: 12px;")
        return lbl

    def _update_symbol_price(self, symbol: str, price: float, change_pct: float = 0.0):
        """Update or add a symbol price display."""
        if symbol not in self._symbol_price_labels:
            row = len(self._symbol_price_labels)
            sym_lbl = QtWidgets.QLabel(symbol)
            sym_lbl.setStyleSheet("color: #e5e5e5; font-weight: bold; font-size: 12px;")
            price_lbl = QtWidgets.QLabel(f"${price:.2f}")
            price_lbl.setAlignment(QtCore.Qt.AlignRight)
            price_lbl.setStyleSheet("color: #22c55e; font-size: 12px;")
            change_lbl = QtWidgets.QLabel(f"{change_pct:+.2f}%")
            change_lbl.setAlignment(QtCore.Qt.AlignRight)
            change_lbl.setStyleSheet("color: #22c55e; font-size: 10px;")

            self.symbol_prices_layout.addWidget(sym_lbl, row, 0)
            self.symbol_prices_layout.addWidget(price_lbl, row, 1)
            self.symbol_prices_layout.addWidget(change_lbl, row, 2)
            self._symbol_price_labels[symbol] = (sym_lbl, price_lbl, change_lbl)
        else:
            _, price_lbl, change_lbl = self._symbol_price_labels[symbol]
            price_lbl.setText(f"${price:.2f}")
            color = "#22c55e" if change_pct >= 0 else "#ef4444"
            price_lbl.setStyleSheet(f"color: {color}; font-size: 12px;")
            change_lbl.setText(f"{change_pct:+.2f}%")
            change_lbl.setStyleSheet(f"color: {color}; font-size: 10px;")

    def _build_market_tab(self):
        tab = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(tab)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # === TOP: Market Overview KPIs ===
        market_kpi_bar = QtWidgets.QFrame()
        market_kpi_bar.setStyleSheet("""
            QFrame {
                background: #1a1a2e;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        kpi_layout = QtWidgets.QHBoxLayout(market_kpi_bar)

        self.market_regime_lbl = self._kpi_label()
        self.market_regime_lbl.setText("NORMAL")
        kpi_layout.addWidget(self._kpi_card("Market Regime", self.market_regime_lbl))

        self.market_volatility_lbl = self._kpi_label()
        self.market_volatility_lbl.setText("--")
        kpi_layout.addWidget(self._kpi_card("Volatility", self.market_volatility_lbl))

        self.market_trend_lbl = self._kpi_label()
        self.market_trend_lbl.setText("--")
        kpi_layout.addWidget(self._kpi_card("Trend", self.market_trend_lbl))

        kpi_layout.addStretch()
        main_layout.addWidget(market_kpi_bar)

        # === MIDDLE: Charts ===
        charts_layout = QtWidgets.QHBoxLayout()

        # Price Chart
        price_box = QtWidgets.QGroupBox("Price Chart")
        price_box.setStyleSheet(self._group_box_style())
        price_layout = QtWidgets.QVBoxLayout(price_box)

        self.price_plot = pg.PlotWidget()
        self.price_plot.setBackground('#0a0a0a')
        self.price_plot.showGrid(x=True, y=True, alpha=0.3)
        self.price_plot.setLabel('left', 'Price ($)')
        self.price_plot.setLabel('bottom', 'Time')
        price_layout.addWidget(self.price_plot)

        charts_layout.addWidget(price_box, 2)

        # Volume / ATR Chart (placeholder)
        indicator_box = QtWidgets.QGroupBox("Indicators")
        indicator_box.setStyleSheet(self._group_box_style())
        indicator_layout = QtWidgets.QVBoxLayout(indicator_box)

        self.indicator_plot = pg.PlotWidget()
        self.indicator_plot.setBackground('#0a0a0a')
        self.indicator_plot.showGrid(x=True, y=True, alpha=0.3)
        self.indicator_plot.setLabel('left', 'ATR')
        self.indicator_plot.setLabel('bottom', 'Time')
        indicator_layout.addWidget(self.indicator_plot)

        charts_layout.addWidget(indicator_box, 1)

        main_layout.addLayout(charts_layout, 2)

        # === BOTTOM: News & Regime Info ===
        bottom_layout = QtWidgets.QHBoxLayout()

        # Regime History
        regime_box = QtWidgets.QGroupBox("Regime History")
        regime_box.setStyleSheet(self._group_box_style())
        regime_layout = QtWidgets.QVBoxLayout(regime_box)

        self.regime_list = QtWidgets.QListWidget()
        self.regime_list.setStyleSheet("""
            QListWidget {
                background: #0f0f0f;
                color: #e5e5e5;
                border: none;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #333;
            }
        """)
        regime_layout.addWidget(self.regime_list)
        bottom_layout.addWidget(regime_box)

        # News Feed
        news_box = QtWidgets.QGroupBox("News & Alerts")
        news_box.setStyleSheet(self._group_box_style())
        news_layout = QtWidgets.QVBoxLayout(news_box)

        self.news_list = QtWidgets.QListWidget()
        self.news_list.setStyleSheet("""
            QListWidget {
                background: #0f0f0f;
                color: #e5e5e5;
                border: none;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #333;
            }
        """)
        news_layout.addWidget(self.news_list)
        bottom_layout.addWidget(news_box)

        main_layout.addLayout(bottom_layout, 1)

        self.tabs.addTab(tab, "Market")

    def _build_performance_tab(self):
        tab = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(tab)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # === Risk-Adjusted Returns ===
        risk_box = QtWidgets.QGroupBox("Risk-Adjusted Returns")
        risk_box.setStyleSheet(self._group_box_style())
        risk_layout = QtWidgets.QHBoxLayout(risk_box)

        self.sharpe_lbl = self._big_kpi_label()
        self.sharpe_lbl.setText("--")
        risk_layout.addWidget(self._kpi_card("Sharpe Ratio", self.sharpe_lbl))

        self.sortino_lbl = self._big_kpi_label()
        self.sortino_lbl.setText("--")
        risk_layout.addWidget(self._kpi_card("Sortino Ratio", self.sortino_lbl))

        self.kelly_lbl = self._big_kpi_label()
        self.kelly_lbl.setText("--")
        risk_layout.addWidget(self._kpi_card("Kelly %", self.kelly_lbl))

        self.calmar_lbl = self._big_kpi_label()
        self.calmar_lbl.setText("--")
        risk_layout.addWidget(self._kpi_card("Calmar Ratio", self.calmar_lbl))

        risk_layout.addStretch()
        main_layout.addWidget(risk_box)

        # === Trade Statistics ===
        trade_box = QtWidgets.QGroupBox("Trade Statistics")
        trade_box.setStyleSheet(self._group_box_style())
        trade_layout = QtWidgets.QGridLayout(trade_box)
        trade_layout.setSpacing(15)

        # Row 1
        self.perf_total_trades_lbl = self._kpi_label()
        trade_layout.addWidget(self._metric_label("Total Trades"), 0, 0)
        trade_layout.addWidget(self.perf_total_trades_lbl, 0, 1)

        self.perf_win_rate_lbl = self._kpi_label()
        trade_layout.addWidget(self._metric_label("Win Rate"), 0, 2)
        trade_layout.addWidget(self.perf_win_rate_lbl, 0, 3)

        self.perf_profit_factor_lbl = self._kpi_label()
        trade_layout.addWidget(self._metric_label("Profit Factor"), 0, 4)
        trade_layout.addWidget(self.perf_profit_factor_lbl, 0, 5)

        # Row 2
        self.perf_avg_win_lbl = self._kpi_label()
        trade_layout.addWidget(self._metric_label("Avg Win"), 1, 0)
        trade_layout.addWidget(self.perf_avg_win_lbl, 1, 1)

        self.perf_avg_loss_lbl = self._kpi_label()
        trade_layout.addWidget(self._metric_label("Avg Loss"), 1, 2)
        trade_layout.addWidget(self.perf_avg_loss_lbl, 1, 3)

        self.perf_expectancy_lbl = self._kpi_label()
        trade_layout.addWidget(self._metric_label("Expectancy"), 1, 4)
        trade_layout.addWidget(self.perf_expectancy_lbl, 1, 5)

        # Row 3
        self.perf_max_win_lbl = self._kpi_label()
        trade_layout.addWidget(self._metric_label("Max Win"), 2, 0)
        trade_layout.addWidget(self.perf_max_win_lbl, 2, 1)

        self.perf_max_loss_lbl = self._kpi_label()
        trade_layout.addWidget(self._metric_label("Max Loss"), 2, 2)
        trade_layout.addWidget(self.perf_max_loss_lbl, 2, 3)

        self.perf_max_dd_lbl = self._kpi_label()
        trade_layout.addWidget(self._metric_label("Max Drawdown"), 2, 4)
        trade_layout.addWidget(self.perf_max_dd_lbl, 2, 5)

        main_layout.addWidget(trade_box)

        # === Returns Distribution Chart ===
        charts_layout = QtWidgets.QHBoxLayout()

        returns_box = QtWidgets.QGroupBox("Returns Distribution")
        returns_box.setStyleSheet(self._group_box_style())
        returns_layout = QtWidgets.QVBoxLayout(returns_box)

        self.returns_plot = pg.PlotWidget()
        self.returns_plot.setBackground('#0a0a0a')
        self.returns_plot.showGrid(x=True, y=True, alpha=0.3)
        self.returns_plot.setLabel('left', 'Frequency')
        self.returns_plot.setLabel('bottom', 'Return %')
        returns_layout.addWidget(self.returns_plot)

        charts_layout.addWidget(returns_box)

        # Drawdown Chart
        dd_box = QtWidgets.QGroupBox("Drawdown Over Time")
        dd_box.setStyleSheet(self._group_box_style())
        dd_layout = QtWidgets.QVBoxLayout(dd_box)

        self.dd_plot = pg.PlotWidget()
        self.dd_plot.setBackground('#0a0a0a')
        self.dd_plot.showGrid(x=True, y=True, alpha=0.3)
        self.dd_plot.setLabel('left', 'Drawdown %')
        self.dd_plot.setLabel('bottom', 'Bar #')
        dd_layout.addWidget(self.dd_plot)

        charts_layout.addWidget(dd_box)

        main_layout.addLayout(charts_layout, 1)

        self.tabs.addTab(tab, "Performance")

    def _build_execution_tab(self):
        tab = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(tab)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # === Order Statistics KPIs ===
        kpi_bar = QtWidgets.QFrame()
        kpi_bar.setStyleSheet("""
            QFrame {
                background: #1a1a2e;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        kpi_layout = QtWidgets.QHBoxLayout(kpi_bar)

        self.q_pending = self._kpi_label()
        self.q_pending.setText("0")
        self.q_pending.setStyleSheet("font-weight: 700; font-size: 16px; color: #f59e0b;")
        kpi_layout.addWidget(self._kpi_card("Pending", self.q_pending))

        self.q_filled = self._kpi_label()
        self.q_filled.setText("0")
        self.q_filled.setStyleSheet("font-weight: 700; font-size: 16px; color: #22c55e;")
        kpi_layout.addWidget(self._kpi_card("Filled", self.q_filled))

        self.q_canceled = self._kpi_label()
        self.q_canceled.setText("0")
        self.q_canceled.setStyleSheet("font-weight: 700; font-size: 16px; color: #ef4444;")
        kpi_layout.addWidget(self._kpi_card("Canceled", self.q_canceled))

        self.q_total = self._kpi_label()
        self.q_total.setText("0")
        kpi_layout.addWidget(self._kpi_card("Total Orders", self.q_total))

        kpi_layout.addStretch()
        main_layout.addWidget(kpi_bar)

        # === Order History Table ===
        order_box = QtWidgets.QGroupBox("Order History")
        order_box.setStyleSheet(self._group_box_style())
        order_layout = QtWidgets.QVBoxLayout(order_box)

        self.order_table = QtWidgets.QTableWidget(0, 6)
        self.order_table.setHorizontalHeaderLabels([
            "Time", "Symbol", "Side", "Qty", "Price", "Status"
        ])
        self.order_table.setStyleSheet("""
            QTableWidget {
                background: #0f0f0f;
                color: #e5e5e5;
                gridline-color: #333;
                border: none;
                font-size: 12px;
            }
            QHeaderView::section {
                background: #1a1a2e;
                color: #94a3b8;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
            QTableWidget::item {
                padding: 6px;
            }
        """)
        self.order_table.horizontalHeader().setStretchLastSection(True)
        self.order_table.setAlternatingRowColors(True)
        self.order_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        order_layout.addWidget(self.order_table)

        main_layout.addWidget(order_box, 1)

        # Initialize counters
        self._order_counts = {"pending": 0, "filled": 0, "canceled": 0, "total": 0}

        self.tabs.addTab(tab, "Execution")

    def _build_alerts_tab(self):
        tab = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(tab)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # === Alert Summary ===
        summary_bar = QtWidgets.QFrame()
        summary_bar.setStyleSheet("""
            QFrame {
                background: #1a1a2e;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        summary_layout = QtWidgets.QHBoxLayout(summary_bar)

        self.alert_total_lbl = self._kpi_label()
        self.alert_total_lbl.setText("0")
        summary_layout.addWidget(self._kpi_card("Total Alerts", self.alert_total_lbl))

        self.alert_error_lbl = self._kpi_label()
        self.alert_error_lbl.setText("0")
        self.alert_error_lbl.setStyleSheet("font-weight: 700; font-size: 16px; color: #ef4444;")
        summary_layout.addWidget(self._kpi_card("Errors", self.alert_error_lbl))

        self.alert_warning_lbl = self._kpi_label()
        self.alert_warning_lbl.setText("0")
        self.alert_warning_lbl.setStyleSheet("font-weight: 700; font-size: 16px; color: #f59e0b;")
        summary_layout.addWidget(self._kpi_card("Warnings", self.alert_warning_lbl))

        self.alert_info_lbl = self._kpi_label()
        self.alert_info_lbl.setText("0")
        self.alert_info_lbl.setStyleSheet("font-weight: 700; font-size: 16px; color: #3b82f6;")
        summary_layout.addWidget(self._kpi_card("Info", self.alert_info_lbl))

        # Clear button
        self.clear_alerts_btn = QtWidgets.QPushButton("Clear All")
        self.clear_alerts_btn.setStyleSheet("""
            QPushButton {
                background: #374151;
                color: #e5e5e5;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background: #4b5563;
            }
        """)
        self.clear_alerts_btn.clicked.connect(self._clear_alerts)
        summary_layout.addStretch()
        summary_layout.addWidget(self.clear_alerts_btn)

        main_layout.addWidget(summary_bar)

        # === Alerts List ===
        alerts_box = QtWidgets.QGroupBox("Alert History")
        alerts_box.setStyleSheet(self._group_box_style())
        alerts_layout = QtWidgets.QVBoxLayout(alerts_box)

        self.alerts_list = QtWidgets.QListWidget()
        self.alerts_list.setStyleSheet("""
            QListWidget {
                background: #0f0f0f;
                color: #e5e5e5;
                border: none;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #333;
            }
            QListWidget::item:selected {
                background: #1e3a5f;
            }
        """)
        alerts_layout.addWidget(self.alerts_list)

        main_layout.addWidget(alerts_box, 1)

        # Initialize alert counters
        self._alert_counts = {"error": 0, "warning": 0, "info": 0, "total": 0}

        self.tabs.addTab(tab, "Alerts")

    def _clear_alerts(self):
        """Clear all alerts."""
        if hasattr(self, 'alerts_list'):
            self.alerts_list.clear()
        self._alert_counts = {"error": 0, "warning": 0, "info": 0, "total": 0}
        self._update_alert_counts()

    def _build_strategy_tab(self):
        tab = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(tab)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # === Strategy Summary KPIs ===
        summary_bar = QtWidgets.QFrame()
        summary_bar.setStyleSheet("""
            QFrame {
                background: #1a1a2e;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        summary_layout = QtWidgets.QHBoxLayout(summary_bar)

        self.strat_active_lbl = self._kpi_label()
        self.strat_active_lbl.setText("0")
        summary_layout.addWidget(self._kpi_card("Active Strategies", self.strat_active_lbl))

        self.strat_signals_lbl = self._kpi_label()
        self.strat_signals_lbl.setText("0")
        summary_layout.addWidget(self._kpi_card("Signals Today", self.strat_signals_lbl))

        self.strat_buy_lbl = self._kpi_label()
        self.strat_buy_lbl.setText("0")
        self.strat_buy_lbl.setStyleSheet("font-weight: 700; font-size: 16px; color: #22c55e;")
        summary_layout.addWidget(self._kpi_card("Buy Signals", self.strat_buy_lbl))

        self.strat_sell_lbl = self._kpi_label()
        self.strat_sell_lbl.setText("0")
        self.strat_sell_lbl.setStyleSheet("font-weight: 700; font-size: 16px; color: #ef4444;")
        summary_layout.addWidget(self._kpi_card("Sell Signals", self.strat_sell_lbl))

        summary_layout.addStretch()
        main_layout.addWidget(summary_bar)

        # === Active Strategies Table ===
        strat_box = QtWidgets.QGroupBox("Strategy Signals")
        strat_box.setStyleSheet(self._group_box_style())
        strat_layout = QtWidgets.QVBoxLayout(strat_box)

        self.sig_table = QtWidgets.QTableWidget(0, 5)
        self.sig_table.setHorizontalHeaderLabels([
            "Symbol/Strategy", "Last Signal", "Regime", "Timestamp", "Count"
        ])
        self.sig_table.setStyleSheet("""
            QTableWidget {
                background: #0f0f0f;
                color: #e5e5e5;
                gridline-color: #333;
                border: none;
                font-size: 12px;
            }
            QHeaderView::section {
                background: #1a1a2e;
                color: #94a3b8;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
            QTableWidget::item {
                padding: 6px;
            }
        """)
        self.sig_table.horizontalHeader().setStretchLastSection(True)
        self.sig_table.setAlternatingRowColors(True)
        self.sig_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        strat_layout.addWidget(self.sig_table)

        main_layout.addWidget(strat_box, 1)

        # === Signal History ===
        history_box = QtWidgets.QGroupBox("Recent Signal History")
        history_box.setStyleSheet(self._group_box_style())
        history_layout = QtWidgets.QVBoxLayout(history_box)

        self.signal_history_list = QtWidgets.QListWidget()
        self.signal_history_list.setStyleSheet("""
            QListWidget {
                background: #0f0f0f;
                color: #e5e5e5;
                border: none;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 4px;
                border-bottom: 1px solid #333;
            }
        """)
        self.signal_history_list.setMaximumHeight(150)
        history_layout.addWidget(self.signal_history_list)

        main_layout.addWidget(history_box)

        # Initialize signal counters
        self._signal_counts = {"buy": 0, "sell": 0, "hold": 0, "total": 0}
        self._strategy_signal_counts = {}

        self.tabs.addTab(tab, "Strategies")

    def _build_lists_tab(self):
        """Build the symbol lists management tab."""
        self.symbol_list_widget = SymbolListWidget()
        self.symbol_list_widget.symbolMoved.connect(self._on_symbol_moved)
        self.tabs.addTab(self.symbol_list_widget, "Lists")

    def _on_symbol_moved(self, symbol: str, list_type: str):
        """Handle symbol moved between lists."""
        action = "trade" if list_type == "trade" else "watch"
        self._log_event(f"Symbol {symbol} moved to {action} list")

    def _build_ops_tab(self):
        tab = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(tab)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # === Trading Controls Bar ===
        controls_bar = QtWidgets.QFrame()
        controls_bar.setStyleSheet("""
            QFrame {
                background: #1a1a2e;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        controls_layout = QtWidgets.QHBoxLayout(controls_bar)

        # Emergency buttons with distinct styling
        self.flatten_btn = QtWidgets.QPushButton("Flatten All")
        self.flatten_btn.setStyleSheet("""
            QPushButton {
                background: #dc2626;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover { background: #ef4444; }
        """)

        self.cancel_all_btn = QtWidgets.QPushButton("Cancel All")
        self.cancel_all_btn.setStyleSheet("""
            QPushButton {
                background: #f59e0b;
                color: black;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover { background: #fbbf24; }
        """)

        self.halt_btn = QtWidgets.QPushButton("HALT")
        self.halt_btn.setStyleSheet("""
            QPushButton {
                background: #7c3aed;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover { background: #8b5cf6; }
        """)

        self.ticket_btn = QtWidgets.QPushButton("Manual Order")
        self.ticket_btn.setStyleSheet("""
            QPushButton {
                background: #3b82f6;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover { background: #60a5fa; }
        """)

        self.flatten_btn.clicked.connect(self._confirm_flatten)
        self.cancel_all_btn.clicked.connect(self._confirm_cancel_all)
        self.halt_btn.clicked.connect(self._toggle_panic)
        self.ticket_btn.clicked.connect(self._show_manual_order)

        controls_layout.addWidget(self.flatten_btn)
        controls_layout.addWidget(self.cancel_all_btn)
        controls_layout.addWidget(self.halt_btn)
        controls_layout.addWidget(self.ticket_btn)
        controls_layout.addStretch()

        # Status display
        ops_mode_label = QtWidgets.QLabel("Mode:")
        ops_mode_label.setStyleSheet("color: #94a3b8;")
        ops_mode_display = QtWidgets.QLabel(self.mode_combo.currentText())
        ops_mode_display.setStyleSheet("color: #22c55e; font-weight: bold;")
        ops_symbol_label = QtWidgets.QLabel("Symbols:")
        ops_symbol_label.setStyleSheet("color: #94a3b8;")
        ops_symbol_display = QtWidgets.QLabel(self.symbol_input.text())
        ops_symbol_display.setStyleSheet("color: #22c55e; font-weight: bold;")

        self.mode_combo.currentTextChanged.connect(ops_mode_display.setText)
        self.symbol_input.textChanged.connect(ops_symbol_display.setText)

        controls_layout.addWidget(ops_mode_label)
        controls_layout.addWidget(ops_mode_display)
        controls_layout.addWidget(ops_symbol_label)
        controls_layout.addWidget(ops_symbol_display)

        main_layout.addWidget(controls_bar)

        # === Middle Section: Simulation + Logs ===
        middle_layout = QtWidgets.QHBoxLayout()

        # Simulation Controls
        sim_box = QtWidgets.QGroupBox("Simulation Controls")
        sim_box.setStyleSheet(self._group_box_style())
        sim_box.setFixedWidth(300)
        sim_layout = QtWidgets.QGridLayout(sim_box)
        sim_layout.setSpacing(10)

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

        self.sim_start_btn = QtWidgets.QPushButton("Start Simulation")
        self.sim_start_btn.setStyleSheet("""
            QPushButton {
                background: #16a34a;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover { background: #22c55e; }
        """)

        self.sim_stop_btn = QtWidgets.QPushButton("Stop Simulation")
        self.sim_stop_btn.setStyleSheet("""
            QPushButton {
                background: #dc2626;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover { background: #ef4444; }
        """)

        self.sim_start_btn.clicked.connect(self._start_sim_clicked)
        self.sim_stop_btn.clicked.connect(self._stop_simulation)

        sim_layout.addWidget(self._metric_label("Steps"), 0, 0)
        sim_layout.addWidget(self.sim_steps_spin, 0, 1)
        sim_layout.addWidget(self._metric_label("Speed (sec/bar)"), 1, 0)
        sim_layout.addWidget(self.sim_speed_spin, 1, 1)
        sim_layout.addWidget(self._metric_label("Drift (μ)"), 2, 0)
        sim_layout.addWidget(self.sim_mu_spin, 2, 1)
        sim_layout.addWidget(self._metric_label("Volatility (σ)"), 3, 0)
        sim_layout.addWidget(self.sim_sigma_spin, 3, 1)
        sim_layout.addWidget(self.sim_start_btn, 4, 0, 1, 2)
        sim_layout.addWidget(self.sim_stop_btn, 5, 0, 1, 2)

        middle_layout.addWidget(sim_box)

        # Logs
        logs_box = QtWidgets.QGroupBox("System Logs")
        logs_box.setStyleSheet(self._group_box_style())
        logs_layout = QtWidgets.QVBoxLayout(logs_box)

        self.logs_view = QtWidgets.QPlainTextEdit()
        self.logs_view.setReadOnly(True)
        self.logs_view.setStyleSheet("""
            QPlainTextEdit {
                background: #0a0a0a;
                color: #22c55e;
                border: none;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
        """)
        logs_layout.addWidget(self.logs_view)

        middle_layout.addWidget(logs_box, 1)

        main_layout.addLayout(middle_layout, 1)

        self.tabs.addTab(tab, "Ops")

    def _build_history_tab(self):
        tab = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(tab)

        # Controls row
        controls = QtWidgets.QHBoxLayout()

        self.history_refresh_btn = QtWidgets.QPushButton("Refresh from Session")
        self.history_refresh_btn.clicked.connect(self._refresh_history_from_session)

        self.history_export_btn = QtWidgets.QPushButton("Export History")
        self.history_export_btn.clicked.connect(self._export_history)

        controls.addWidget(self.history_refresh_btn)
        controls.addWidget(self.history_export_btn)
        controls.addStretch()

        grid.addLayout(controls, 0, 0, 1, 2)

        # PnL Calendar (bar chart showing daily returns)
        self.calendar_plot = pg.PlotWidget(title="Daily PnL")
        self.calendar_plot.setLabel('bottom', 'Day')
        self.calendar_plot.setLabel('left', 'PnL ($)')
        grid.addWidget(self.calendar_plot, 1, 0)

        # Equity vs Benchmark
        self.bench_plot = pg.PlotWidget(title="Equity vs Benchmark")
        self.bench_plot.setLabel('bottom', 'Time')
        self.bench_plot.setLabel('left', 'Value ($)')
        self.bench_plot.addLegend()
        grid.addWidget(self.bench_plot, 1, 1)

        # Trade statistics
        stats_box = QtWidgets.QGroupBox("Session Statistics")
        stats_layout = QtWidgets.QGridLayout(stats_box)

        self.hist_trades_lbl = self._kpi_label()
        self.hist_winrate_lbl = self._kpi_label()
        self.hist_avgwin_lbl = self._kpi_label()
        self.hist_avgloss_lbl = self._kpi_label()
        self.hist_maxdd_lbl = self._kpi_label()
        self.hist_total_pnl_lbl = self._kpi_label()

        stats_layout.addWidget(QtWidgets.QLabel("Total Trades"), 0, 0)
        stats_layout.addWidget(self.hist_trades_lbl, 0, 1)
        stats_layout.addWidget(QtWidgets.QLabel("Win Rate"), 0, 2)
        stats_layout.addWidget(self.hist_winrate_lbl, 0, 3)
        stats_layout.addWidget(QtWidgets.QLabel("Avg Win"), 1, 0)
        stats_layout.addWidget(self.hist_avgwin_lbl, 1, 1)
        stats_layout.addWidget(QtWidgets.QLabel("Avg Loss"), 1, 2)
        stats_layout.addWidget(self.hist_avgloss_lbl, 1, 3)
        stats_layout.addWidget(QtWidgets.QLabel("Max Drawdown"), 2, 0)
        stats_layout.addWidget(self.hist_maxdd_lbl, 2, 1)
        stats_layout.addWidget(QtWidgets.QLabel("Total PnL"), 2, 2)
        stats_layout.addWidget(self.hist_total_pnl_lbl, 2, 3)

        grid.addWidget(stats_box, 2, 0, 1, 2)

        self.tabs.addTab(tab, "History")

    def _refresh_history_from_session(self):
        """Populate History tab with current session data."""
        try:
            if len(self._eq_y) < 2:
                return  # Silent return - not enough data yet

            print(f"[MainWindow] HISTORY: Refreshing with {len(self._eq_y)} equity points")

            equity = np.array(self._eq_y)
            returns = np.diff(equity)

            # Calculate daily PnL (group by chunks of ~100 bars as "days")
            chunk_size = max(1, len(returns) // 20)  # ~20 days
            daily_pnl = []
            for i in range(0, len(returns), chunk_size):
                chunk = returns[i:i+chunk_size]
                daily_pnl.append(np.sum(chunk))

            # Update calendar plot
            x = list(range(len(daily_pnl)))
            colors = ['#22c55e' if p >= 0 else '#ef4444' for p in daily_pnl]
            brushes = [pg.mkBrush(c) for c in colors]

            self.calendar_plot.clear()
            bar = pg.BarGraphItem(x=x, height=daily_pnl, width=0.8, brushes=brushes)
            self.calendar_plot.addItem(bar)

            # Update benchmark plot (compare to buy-and-hold)
            initial = equity[0] if equity[0] > 0 else 10000
            benchmark = np.linspace(initial, initial * 1.05, len(equity))  # Simple 5% growth line

            self.bench_plot.clear()
            self.bench_plot.plot(list(range(len(equity))), equity.tolist(),
                                 pen=pg.mkPen("#22c55e", width=2), name="Strategy")
            self.bench_plot.plot(list(range(len(benchmark))), benchmark.tolist(),
                                 pen=pg.mkPen("#64748b", width=2), name="Benchmark")

            # Calculate statistics
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            total_trades = len(returns)
            win_rate = len(wins) / total_trades if total_trades > 0 else 0
            avg_win = np.mean(wins) if len(wins) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0

            # Max drawdown
            peak = np.maximum.accumulate(equity)
            drawdown = (equity - peak) / peak
            max_dd = np.min(drawdown)

            total_pnl = equity[-1] - equity[0]

            # Update labels
            self.hist_trades_lbl.setText(str(total_trades))
            self._set_kpi(self.hist_winrate_lbl, win_rate, pct=True)
            self._set_kpi(self.hist_avgwin_lbl, avg_win, money=True)
            self._set_kpi(self.hist_avgloss_lbl, avg_loss, money=True)
            self._set_kpi(self.hist_maxdd_lbl, max_dd, pct=True)
            self._set_kpi(self.hist_total_pnl_lbl, total_pnl, money=True)

            self._append_log(f"[HISTORY] Refreshed with {len(equity)} data points")
        except Exception as e:
            self._append_log(f"[ERR] History refresh failed: {e}")

    def _export_history(self):
        """Export history data to CSV."""
        try:
            if len(self._eq_y) < 2:
                self._append_log("[HISTORY] No data to export")
                return

            path = os.path.join(self._csv_dir, f"{self._session_id}_history.csv")
            df = pd.DataFrame({
                'frame': range(len(self._eq_y)),
                'equity': self._eq_y
            })
            df.to_csv(path, index=False)
            self._append_log(f"[HISTORY] Exported to {path}")
            QtWidgets.QMessageBox.information(self, "Export", f"History exported to:\n{path}")
        except Exception as e:
            self._append_log(f"[ERR] History export failed: {e}")

    def _build_replay_tab(self):
        tab = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab)

        # Controls row
        controls = QtWidgets.QHBoxLayout()

        self.replay_load_btn = QtWidgets.QPushButton("Load Session")
        self.replay_load_btn.clicked.connect(self._load_replay_session)

        self.replay_play_btn = QtWidgets.QPushButton("▶ Play")
        self.replay_play_btn.clicked.connect(self._toggle_replay)

        self.replay_speed_spin = QtWidgets.QDoubleSpinBox()
        self.replay_speed_spin.setRange(0.1, 10.0)
        self.replay_speed_spin.setValue(1.0)
        self.replay_speed_spin.setSuffix("x")

        self.replay_frame_lbl = QtWidgets.QLabel("Frame: 0 / 0")

        controls.addWidget(self.replay_load_btn)
        controls.addWidget(self.replay_play_btn)
        controls.addWidget(QtWidgets.QLabel("Speed:"))
        controls.addWidget(self.replay_speed_spin)
        controls.addStretch()
        controls.addWidget(self.replay_frame_lbl)

        v.addLayout(controls)

        # Slider
        self.replay_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.replay_slider.setMinimum(0)
        self.replay_slider.setMaximum(100)
        self.replay_slider.setValue(0)
        self.replay_slider.valueChanged.connect(self._on_replay_slider_changed)
        v.addWidget(self.replay_slider)

        # Plot
        self.replay_plot = pg.PlotWidget(title="Equity Replay")
        self.replay_curve = self.replay_plot.plot([], [], pen=pg.mkPen("#22c55e", width=2))
        self.replay_marker = pg.ScatterPlotItem(size=12, brush=pg.mkBrush(255, 200, 0, 200))
        self.replay_plot.addItem(self.replay_marker)
        v.addWidget(self.replay_plot)

        # Initialize replay state
        self._replay_playing = False
        self._replay_timer = QtCore.QTimer()
        self._replay_timer.timeout.connect(self._replay_step)
        self._replay_history = []  # Stored equity history for replay

        self.tabs.addTab(tab, "Replay")

    def _load_replay_session(self):
        """Load a session for replay from equity history or CSV."""
        print(f"[MainWindow] REPLAY: Load requested, have {len(self._eq_y)} equity points")
        self._append_log(f"[REPLAY] Loading session with {len(self._eq_y)} data points")

        # Use current session's equity data
        if len(self._eq_y) > 10:
            self._replay_history = list(zip(range(len(self._eq_y)), self._eq_y))
            self.replay_slider.setMaximum(len(self._replay_history) - 1)
            self.replay_slider.setValue(0)
            self.replay_frame_lbl.setText(f"Frame: 0 / {len(self._replay_history)}")

            # Plot full history
            x = [p[0] for p in self._replay_history]
            y = [p[1] for p in self._replay_history]
            self.replay_curve.setData(x, y)

            # Update replay plot appearance
            self.replay_plot.setTitle(f"Equity Replay ({len(self._replay_history)} frames)")

            self._append_log(f"[REPLAY] Loaded {len(self._replay_history)} frames from current session")
        else:
            self._append_log("[REPLAY] Not enough data to replay. Run a simulation first.")
            QtWidgets.QMessageBox.information(
                self, "Replay",
                "Not enough data to replay.\nRun a simulation first to generate equity data."
            )

    def _toggle_replay(self):
        """Toggle replay playback."""
        if not self._replay_history:
            self._append_log("[REPLAY] Load a session first")
            return

        self._replay_playing = not self._replay_playing
        if self._replay_playing:
            self.replay_play_btn.setText("⏸ Pause")
            interval = int(100 / self.replay_speed_spin.value())  # ms per frame
            self._replay_timer.start(interval)
        else:
            self.replay_play_btn.setText("▶ Play")
            self._replay_timer.stop()

    def _replay_step(self):
        """Advance replay by one frame."""
        current = self.replay_slider.value()
        if current < self.replay_slider.maximum():
            self.replay_slider.setValue(current + 1)
        else:
            # Reached end
            self._replay_playing = False
            self.replay_play_btn.setText("▶ Play")
            self._replay_timer.stop()

    def _on_replay_slider_changed(self, value):
        """Handle replay slider position change."""
        if not self._replay_history or value >= len(self._replay_history):
            return

        frame_idx, equity = self._replay_history[value]
        self.replay_frame_lbl.setText(f"Frame: {value} / {len(self._replay_history)}")

        # Update marker position
        self.replay_marker.setData([frame_idx], [equity])

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
    

    def _kpi_label(self) -> QtWidgets.QLabel:
        """Create a styled KPI value label."""
        lbl = QtWidgets.QLabel("--")
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setStyleSheet("""
            font-weight: 700;
            font-size: 16px;
            color: #e5e5e5;
            padding: 2px;
        """)
        return lbl

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

    def _start_sim_clicked(self):
        """Handler for Start Simulation button - creates and stores the task."""
        if getattr(self, "_sim_running", False):
            self._append_log("[SIM] Simulation already running!")
            return
        self._sim_task = asyncio.create_task(self._start_sim())

    async def _start_sim(self):
        """Launch SimulationRunner (GBM-based) using GUI parameters."""

        # Check if already running (double-check)
        if getattr(self, "_sim_running", False):
            self._append_log("[SIM] Simulation already running!")
            return

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
            self._sim_task = None

    def _stop_simulation(self):
        """Stop the running simulation."""
        if not getattr(self, "_sim_running", False):
            self._append_log("[SIM] No active simulation to stop.")
            return

        self._append_log("[SIM] Stopping simulation...")

        # Set stop flag on the runner
        if hasattr(self, '_sim_runner') and self._sim_runner is not None:
            self._sim_runner.stop()
            self._append_log("[SIM] Stop signal sent to SimulationRunner.")

        # Cancel the task if it exists
        if hasattr(self, '_sim_task') and self._sim_task is not None:
            self._sim_task.cancel()
            self._append_log("[SIM] Task cancellation requested.")

        self._sim_running = False