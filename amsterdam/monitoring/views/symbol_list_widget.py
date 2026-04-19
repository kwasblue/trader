"""
Symbol List Widget - GUI for managing trade and watch lists.

Provides a two-column view for managing symbols between trade and watch lists
with drag-and-drop support and context menus.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from core.symbol_list_manager import LIST_TYPE_TRADE, LIST_TYPE_WATCH, get_list_manager


class SymbolListWidget(QtWidgets.QWidget):
    """Widget for managing trade and watch lists."""

    symbolMoved = QtCore.Signal(str, str)  # symbol, new_list_type

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._manager = get_list_manager()
        self._setup_ui()
        self._load_lists()

    def _setup_ui(self):
        """Set up the widget UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # === Header with Add Symbol ===
        header = QtWidgets.QFrame()
        header.setStyleSheet("""
            QFrame {
                background: #1a1a2e;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        header_layout = QtWidgets.QHBoxLayout(header)

        # Add symbol input
        self.symbol_input = QtWidgets.QLineEdit()
        self.symbol_input.setPlaceholderText("Enter symbol (e.g., TSLA)")
        self.symbol_input.setMaximumWidth(200)
        self.symbol_input.returnPressed.connect(self._add_to_trade)
        header_layout.addWidget(self.symbol_input)

        # Add buttons
        self.add_trade_btn = QtWidgets.QPushButton("Add to Trade")
        self.add_trade_btn.setStyleSheet("""
            QPushButton {
                background: #22c55e;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #16a34a;
            }
        """)
        self.add_trade_btn.clicked.connect(self._add_to_trade)
        header_layout.addWidget(self.add_trade_btn)

        self.add_watch_btn = QtWidgets.QPushButton("Add to Watch")
        self.add_watch_btn.setStyleSheet("""
            QPushButton {
                background: #3b82f6;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #2563eb;
            }
        """)
        self.add_watch_btn.clicked.connect(self._add_to_watch)
        header_layout.addWidget(self.add_watch_btn)

        header_layout.addStretch()

        # Refresh button
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._load_lists)
        header_layout.addWidget(self.refresh_btn)

        layout.addWidget(header)

        # === Lists Container ===
        lists_container = QtWidgets.QHBoxLayout()
        lists_container.setSpacing(10)

        # Trade List
        trade_box = QtWidgets.QGroupBox("Trade List (Active Trading)")
        trade_box.setStyleSheet(self._group_box_style("#22c55e"))
        trade_layout = QtWidgets.QVBoxLayout(trade_box)

        self.trade_list = QtWidgets.QListWidget()
        self.trade_list.setStyleSheet(self._list_style())
        self.trade_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.trade_list.customContextMenuRequested.connect(lambda pos: self._show_context_menu(pos, LIST_TYPE_TRADE))
        self.trade_list.setDragDropMode(QtWidgets.QAbstractItemView.DragDrop)
        self.trade_list.setDefaultDropAction(QtCore.Qt.MoveAction)
        trade_layout.addWidget(self.trade_list)

        self.trade_count_lbl = QtWidgets.QLabel("0 symbols")
        self.trade_count_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        trade_layout.addWidget(self.trade_count_lbl)

        lists_container.addWidget(trade_box)

        # Move buttons
        move_container = QtWidgets.QVBoxLayout()
        move_container.addStretch()

        self.move_to_watch_btn = QtWidgets.QPushButton(">>")
        self.move_to_watch_btn.setToolTip("Move to Watch List")
        self.move_to_watch_btn.setFixedWidth(40)
        self.move_to_watch_btn.clicked.connect(self._move_to_watch)
        move_container.addWidget(self.move_to_watch_btn)

        self.move_to_trade_btn = QtWidgets.QPushButton("<<")
        self.move_to_trade_btn.setToolTip("Move to Trade List")
        self.move_to_trade_btn.setFixedWidth(40)
        self.move_to_trade_btn.clicked.connect(self._move_to_trade)
        move_container.addWidget(self.move_to_trade_btn)

        move_container.addStretch()
        lists_container.addLayout(move_container)

        # Watch List
        watch_box = QtWidgets.QGroupBox("Watch List (Data Only)")
        watch_box.setStyleSheet(self._group_box_style("#3b82f6"))
        watch_layout = QtWidgets.QVBoxLayout(watch_box)

        self.watch_list = QtWidgets.QListWidget()
        self.watch_list.setStyleSheet(self._list_style())
        self.watch_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.watch_list.customContextMenuRequested.connect(lambda pos: self._show_context_menu(pos, LIST_TYPE_WATCH))
        self.watch_list.setDragDropMode(QtWidgets.QAbstractItemView.DragDrop)
        self.watch_list.setDefaultDropAction(QtCore.Qt.MoveAction)
        watch_layout.addWidget(self.watch_list)

        self.watch_count_lbl = QtWidgets.QLabel("0 symbols")
        self.watch_count_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        watch_layout.addWidget(self.watch_count_lbl)

        lists_container.addWidget(watch_box)

        layout.addLayout(lists_container, 1)

        # === Info Panel ===
        info_box = QtWidgets.QGroupBox("Symbol Info")
        info_box.setStyleSheet(self._group_box_style("#64748b"))
        info_layout = QtWidgets.QVBoxLayout(info_box)

        self.info_text = QtWidgets.QLabel(
            "Select a symbol to view details. Use the buttons or right-click to move symbols between lists."
        )
        self.info_text.setStyleSheet("color: #94a3b8; font-size: 12px;")
        self.info_text.setWordWrap(True)
        info_layout.addWidget(self.info_text)

        layout.addWidget(info_box)

    def _group_box_style(self, accent_color: str) -> str:
        """Return GroupBox style with accent color."""
        return f"""
            QGroupBox {{
                background: #0f0f0f;
                border: 1px solid #333;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 8px;
                font-weight: bold;
                color: #94a3b8;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {accent_color};
            }}
        """

    def _list_style(self) -> str:
        """Return list widget style."""
        return """
            QListWidget {
                background: #0f0f0f;
                color: #e5e5e5;
                border: none;
                font-size: 13px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #333;
            }
            QListWidget::item:selected {
                background: #374151;
            }
            QListWidget::item:hover {
                background: #1f2937;
            }
        """

    def _load_lists(self):
        """Load symbols into both lists."""
        self.trade_list.clear()
        self.watch_list.clear()

        trade_symbols = self._manager.get_trade_list()
        watch_symbols = self._manager.get_watch_list()

        for symbol in trade_symbols:
            item = QtWidgets.QListWidgetItem(symbol)
            item.setData(QtCore.Qt.UserRole, symbol)
            self.trade_list.addItem(item)

        for symbol in watch_symbols:
            item = QtWidgets.QListWidgetItem(symbol)
            item.setData(QtCore.Qt.UserRole, symbol)
            self.watch_list.addItem(item)

        self.trade_count_lbl.setText(f"{len(trade_symbols)} symbols")
        self.watch_count_lbl.setText(f"{len(watch_symbols)} symbols")

    def _add_to_trade(self):
        """Add symbol from input to trade list."""
        symbol = self.symbol_input.text().strip().upper()
        if not symbol:
            return

        self._manager.add_to_trade_list(symbol)
        self.symbol_input.clear()
        self._load_lists()
        self.symbolMoved.emit(symbol, LIST_TYPE_TRADE)

    def _add_to_watch(self):
        """Add symbol from input to watch list."""
        symbol = self.symbol_input.text().strip().upper()
        if not symbol:
            return

        self._manager.add_to_watch_list(symbol)
        self.symbol_input.clear()
        self._load_lists()
        self.symbolMoved.emit(symbol, LIST_TYPE_WATCH)

    def _move_to_watch(self):
        """Move selected trade list symbol to watch list."""
        item = self.trade_list.currentItem()
        if not item:
            return

        symbol = item.data(QtCore.Qt.UserRole)
        if self._manager.move_to_watch_list(symbol):
            self._load_lists()
            self.symbolMoved.emit(symbol, LIST_TYPE_WATCH)

    def _move_to_trade(self):
        """Move selected watch list symbol to trade list."""
        item = self.watch_list.currentItem()
        if not item:
            return

        symbol = item.data(QtCore.Qt.UserRole)
        if self._manager.move_to_trade_list(symbol):
            self._load_lists()
            self.symbolMoved.emit(symbol, LIST_TYPE_TRADE)

    def _show_context_menu(self, pos: QtCore.QPoint, list_type: str):
        """Show context menu for list item."""
        if list_type == LIST_TYPE_TRADE:
            list_widget = self.trade_list
        else:
            list_widget = self.watch_list

        item = list_widget.itemAt(pos)
        if not item:
            return

        symbol = item.data(QtCore.Qt.UserRole)

        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background: #1f2937;
                color: #e5e5e5;
                border: 1px solid #374151;
            }
            QMenu::item {
                padding: 8px 20px;
            }
            QMenu::item:selected {
                background: #374151;
            }
        """)

        if list_type == LIST_TYPE_TRADE:
            move_action = menu.addAction("Move to Watch List")
            move_action.triggered.connect(lambda: self._context_move(symbol, LIST_TYPE_WATCH))
        else:
            move_action = menu.addAction("Move to Trade List")
            move_action.triggered.connect(lambda: self._context_move(symbol, LIST_TYPE_TRADE))

        menu.addSeparator()

        remove_action = menu.addAction("Remove")
        remove_action.triggered.connect(lambda: self._remove_symbol(symbol))

        menu.exec_(list_widget.mapToGlobal(pos))

    def _context_move(self, symbol: str, target_list: str):
        """Move symbol via context menu."""
        if target_list == LIST_TYPE_WATCH:
            self._manager.move_to_watch_list(symbol)
        else:
            self._manager.move_to_trade_list(symbol)
        self._load_lists()
        self.symbolMoved.emit(symbol, target_list)

    def _remove_symbol(self, symbol: str):
        """Remove symbol from all lists."""
        reply = QtWidgets.QMessageBox.question(
            self,
            "Remove Symbol",
            f"Remove {symbol} from all lists?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self._manager.remove_symbol(symbol)
            self._load_lists()

    def get_trade_symbols(self) -> list[str]:
        """Get current trade list symbols."""
        return self._manager.get_trade_list()

    def get_watch_symbols(self) -> list[str]:
        """Get current watch list symbols."""
        return self._manager.get_watch_list()
