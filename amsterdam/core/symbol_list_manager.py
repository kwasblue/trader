"""
Symbol List Manager - Manages trade and watch lists with SQLite persistence.

Provides functionality to:
- Maintain a trade list (actively traded symbols)
- Maintain a watch list (monitored symbols, data only)
- Move symbols between lists
- Persist lists to database

Usage:
    from core.symbol_list_manager import get_list_manager

    manager = get_list_manager()

    # Get lists
    trade_symbols = manager.get_trade_list()
    watch_symbols = manager.get_watch_list()

    # Add symbols
    manager.add_to_trade_list("TSLA")
    manager.add_to_watch_list("NVDA")

    # Move between lists
    manager.move_to_watch_list("TSLA")
    manager.move_to_trade_list("NVDA")

    # Remove
    manager.remove_symbol("AAPL")
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

LIST_TYPE_TRADE = "trade"
LIST_TYPE_WATCH = "watch"


@dataclass
class SymbolEntry:
    """A symbol entry in a list."""
    symbol: str
    list_type: str  # 'trade' or 'watch'
    added_at: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "SymbolEntry":
        return cls(
            symbol=row["symbol"],
            list_type=row["list_type"],
            added_at=row["added_at"],
            notes=row["notes"] or "",
        )


class SymbolListManager:
    """
    Manages trade and watch lists with SQLite persistence.

    Features:
    - Persistent storage of symbol lists
    - Thread-safe operations
    - Migration from config defaults
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the symbol list manager.

        Args:
            db_path: Path to SQLite database. Defaults to 'data/trading_state.db'
        """
        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "data" / "trading_state.db")

        self.db_path = db_path
        self._local = threading.local()
        self._ensure_directory()
        self._init_schema()
        self._migrate_defaults_if_empty()

    def _ensure_directory(self) -> None:
        """Ensure database directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(self.db_path)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    @contextmanager
    def _transaction(self):
        """Context manager for database transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_schema(self) -> None:
        """Initialize database schema for symbol lists."""
        with self._transaction() as conn:
            conn.executescript("""
                -- Symbol lists table
                CREATE TABLE IF NOT EXISTS symbol_lists (
                    symbol TEXT PRIMARY KEY,
                    list_type TEXT NOT NULL CHECK(list_type IN ('trade', 'watch')),
                    added_at TEXT NOT NULL,
                    notes TEXT DEFAULT ''
                );

                -- Index for list type queries
                CREATE INDEX IF NOT EXISTS idx_symbol_lists_type ON symbol_lists(list_type);
            """)
        logger.info(f"[SymbolListManager] Initialized database at {self.db_path}")

    def _migrate_defaults_if_empty(self) -> None:
        """Migrate default symbols from config if lists are empty."""
        conn = self._get_connection()
        count = conn.execute("SELECT COUNT(*) FROM symbol_lists").fetchone()[0]

        if count > 0:
            return  # Already has data, don't migrate

        # Try to load defaults from trading_config.json
        try:
            config_path = Path(__file__).parent.parent / "config" / "trading_config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                default_symbols = config.get("general", {}).get("default_symbols", [])

                if default_symbols:
                    logger.info(f"[SymbolListManager] Migrating default symbols to trade list: {default_symbols}")
                    for symbol in default_symbols:
                        self.add_to_trade_list(symbol.upper())
        except Exception as e:
            logger.warning(f"[SymbolListManager] Could not migrate defaults: {e}")

    # ========== Trade List Operations ==========

    def get_trade_list(self) -> List[str]:
        """Get list of symbols in the trade list."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT symbol FROM symbol_lists WHERE list_type = ? ORDER BY symbol",
            (LIST_TYPE_TRADE,)
        ).fetchall()
        return [row["symbol"] for row in rows]

    def add_to_trade_list(self, symbol: str, notes: str = "") -> bool:
        """
        Add a symbol to the trade list.

        Args:
            symbol: Stock symbol (will be uppercased)
            notes: Optional notes about the symbol

        Returns:
            True if added, False if already exists
        """
        symbol = symbol.upper()
        now = datetime.now(timezone.utc).isoformat()

        with self._transaction() as conn:
            # Check if exists
            existing = conn.execute(
                "SELECT list_type FROM symbol_lists WHERE symbol = ?", (symbol,)
            ).fetchone()

            if existing:
                if existing["list_type"] == LIST_TYPE_TRADE:
                    return False  # Already in trade list
                # Move from watch to trade
                conn.execute(
                    "UPDATE symbol_lists SET list_type = ?, added_at = ?, notes = ? WHERE symbol = ?",
                    (LIST_TYPE_TRADE, now, notes, symbol)
                )
                logger.info(f"[SymbolListManager] Moved {symbol} from watch to trade list")
            else:
                conn.execute(
                    "INSERT INTO symbol_lists (symbol, list_type, added_at, notes) VALUES (?, ?, ?, ?)",
                    (symbol, LIST_TYPE_TRADE, now, notes)
                )
                logger.info(f"[SymbolListManager] Added {symbol} to trade list")

        return True

    # ========== Watch List Operations ==========

    def get_watch_list(self) -> List[str]:
        """Get list of symbols in the watch list."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT symbol FROM symbol_lists WHERE list_type = ? ORDER BY symbol",
            (LIST_TYPE_WATCH,)
        ).fetchall()
        return [row["symbol"] for row in rows]

    def add_to_watch_list(self, symbol: str, notes: str = "") -> bool:
        """
        Add a symbol to the watch list.

        Args:
            symbol: Stock symbol (will be uppercased)
            notes: Optional notes about the symbol

        Returns:
            True if added, False if already exists
        """
        symbol = symbol.upper()
        now = datetime.now(timezone.utc).isoformat()

        with self._transaction() as conn:
            # Check if exists
            existing = conn.execute(
                "SELECT list_type FROM symbol_lists WHERE symbol = ?", (symbol,)
            ).fetchone()

            if existing:
                if existing["list_type"] == LIST_TYPE_WATCH:
                    return False  # Already in watch list
                # Move from trade to watch
                conn.execute(
                    "UPDATE symbol_lists SET list_type = ?, added_at = ?, notes = ? WHERE symbol = ?",
                    (LIST_TYPE_WATCH, now, notes, symbol)
                )
                logger.info(f"[SymbolListManager] Moved {symbol} from trade to watch list")
            else:
                conn.execute(
                    "INSERT INTO symbol_lists (symbol, list_type, added_at, notes) VALUES (?, ?, ?, ?)",
                    (symbol, LIST_TYPE_WATCH, now, notes)
                )
                logger.info(f"[SymbolListManager] Added {symbol} to watch list")

        return True

    # ========== Move Operations ==========

    def move_to_trade_list(self, symbol: str) -> bool:
        """
        Move a symbol to the trade list.

        Args:
            symbol: Stock symbol

        Returns:
            True if moved, False if not found or already in trade list
        """
        symbol = symbol.upper()
        now = datetime.now(timezone.utc).isoformat()

        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE symbol_lists SET list_type = ?, added_at = ? WHERE symbol = ? AND list_type = ?",
                (LIST_TYPE_TRADE, now, symbol, LIST_TYPE_WATCH)
            )
            if cursor.rowcount > 0:
                logger.info(f"[SymbolListManager] Moved {symbol} to trade list")
                return True
            return False

    def move_to_watch_list(self, symbol: str) -> bool:
        """
        Move a symbol to the watch list.

        Args:
            symbol: Stock symbol

        Returns:
            True if moved, False if not found or already in watch list
        """
        symbol = symbol.upper()
        now = datetime.now(timezone.utc).isoformat()

        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE symbol_lists SET list_type = ?, added_at = ? WHERE symbol = ? AND list_type = ?",
                (LIST_TYPE_WATCH, now, symbol, LIST_TYPE_TRADE)
            )
            if cursor.rowcount > 0:
                logger.info(f"[SymbolListManager] Moved {symbol} to watch list")
                return True
            return False

    # ========== General Operations ==========

    def get_symbol(self, symbol: str) -> Optional[SymbolEntry]:
        """Get details for a specific symbol."""
        symbol = symbol.upper()
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM symbol_lists WHERE symbol = ?", (symbol,)
        ).fetchone()
        if row:
            return SymbolEntry.from_row(row)
        return None

    def get_all_symbols(self) -> List[SymbolEntry]:
        """Get all symbols from both lists."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM symbol_lists ORDER BY list_type, symbol"
        ).fetchall()
        return [SymbolEntry.from_row(row) for row in rows]

    def remove_symbol(self, symbol: str) -> bool:
        """
        Remove a symbol from all lists.

        Args:
            symbol: Stock symbol

        Returns:
            True if removed, False if not found
        """
        symbol = symbol.upper()

        with self._transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM symbol_lists WHERE symbol = ?", (symbol,)
            )
            if cursor.rowcount > 0:
                logger.info(f"[SymbolListManager] Removed {symbol} from lists")
                return True
            return False

    def set_notes(self, symbol: str, notes: str) -> bool:
        """
        Set notes for a symbol.

        Args:
            symbol: Stock symbol
            notes: Notes to set

        Returns:
            True if updated, False if symbol not found
        """
        symbol = symbol.upper()

        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE symbol_lists SET notes = ? WHERE symbol = ?",
                (notes, symbol)
            )
            return cursor.rowcount > 0

    def symbol_exists(self, symbol: str) -> bool:
        """Check if a symbol exists in any list."""
        symbol = symbol.upper()
        conn = self._get_connection()
        row = conn.execute(
            "SELECT 1 FROM symbol_lists WHERE symbol = ?", (symbol,)
        ).fetchone()
        return row is not None

    def get_list_type(self, symbol: str) -> Optional[str]:
        """Get which list a symbol is in."""
        symbol = symbol.upper()
        conn = self._get_connection()
        row = conn.execute(
            "SELECT list_type FROM symbol_lists WHERE symbol = ?", (symbol,)
        ).fetchone()
        if row:
            return row["list_type"]
        return None

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None


# Singleton instance
_list_manager: Optional[SymbolListManager] = None


def get_list_manager(db_path: Optional[str] = None) -> SymbolListManager:
    """
    Get singleton SymbolListManager instance.

    Args:
        db_path: Optional database path (only used on first call)

    Returns:
        SymbolListManager instance
    """
    global _list_manager
    if _list_manager is None:
        _list_manager = SymbolListManager(db_path)
    return _list_manager


def reset_list_manager() -> None:
    """Reset the singleton instance (useful for testing)."""
    global _list_manager
    if _list_manager is not None:
        _list_manager.close()
        _list_manager = None
