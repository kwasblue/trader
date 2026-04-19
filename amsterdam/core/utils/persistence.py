"""
State persistence for trading system recovery.

Provides SQLite-based persistence for positions, orders, and system state.
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
from typing import Any, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class PersistedPosition:
    """Persisted position record."""
    symbol: str
    quantity: float
    avg_price: float
    side: str  # 'long' or 'short'
    opened_at: str
    last_updated: str
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    metadata: str = "{}"  # JSON string

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['metadata'] = json.loads(d['metadata'])
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PersistedPosition":
        if isinstance(d.get('metadata'), dict):
            d['metadata'] = json.dumps(d['metadata'])
        return cls(**d)


@dataclass
class PersistedOrder:
    """Persisted order record."""
    order_id: str
    symbol: str
    side: str
    quantity: float
    order_type: str
    status: str
    created_at: str
    updated_at: str
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    metadata: str = "{}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['metadata'] = json.loads(d['metadata'])
        return d


@dataclass
class SystemState:
    """System state snapshot."""
    state_id: str
    timestamp: str
    cash_balance: float
    total_equity: float
    positions_json: str
    pending_orders_json: str
    metadata: str = "{}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'state_id': self.state_id,
            'timestamp': self.timestamp,
            'cash_balance': self.cash_balance,
            'total_equity': self.total_equity,
            'positions': json.loads(self.positions_json),
            'pending_orders': json.loads(self.pending_orders_json),
            'metadata': json.loads(self.metadata),
        }


class StatePersistence:
    """
    SQLite-based state persistence for trading system.

    Features:
    - Position tracking and recovery
    - Order history
    - System state snapshots
    - Thread-safe operations
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize persistence layer.

        Args:
            db_path: Path to SQLite database. Defaults to 'data/trading_state.db'
        """
        if db_path is None:
            db_path = str(Path(__file__).parent.parent.parent / "data" / "trading_state.db")

        self.db_path = db_path
        self._local = threading.local()
        self._ensure_directory()
        self._init_schema()

    def _ensure_directory(self) -> None:
        """Ensure database directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
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
        """Initialize database schema."""
        with self._transaction() as conn:
            conn.executescript("""
                -- Positions table
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity REAL NOT NULL,
                    avg_price REAL NOT NULL,
                    side TEXT NOT NULL,
                    opened_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    unrealized_pnl REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    metadata TEXT DEFAULT '{}'
                );

                -- Orders table
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    order_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    filled_quantity REAL DEFAULT 0,
                    avg_fill_price REAL,
                    limit_price REAL,
                    stop_price REAL,
                    metadata TEXT DEFAULT '{}'
                );

                -- System state snapshots
                CREATE TABLE IF NOT EXISTS system_state (
                    state_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    cash_balance REAL NOT NULL,
                    total_equity REAL NOT NULL,
                    positions_json TEXT NOT NULL,
                    pending_orders_json TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                );

                -- Trade history
                CREATE TABLE IF NOT EXISTS trade_history (
                    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    pnl REAL,
                    executed_at TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}'
                );

                -- Indexes
                CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
                CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
                CREATE INDEX IF NOT EXISTS idx_trade_history_symbol ON trade_history(symbol);
                CREATE INDEX IF NOT EXISTS idx_system_state_timestamp ON system_state(timestamp);
            """)

        logger.info(f"[Persistence] Initialized database at {self.db_path}")

    # ========== Position Operations ==========

    def save_position(self, position: PersistedPosition) -> None:
        """Save or update a position."""
        with self._transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO positions
                (symbol, quantity, avg_price, side, opened_at, last_updated,
                 unrealized_pnl, realized_pnl, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position.symbol, position.quantity, position.avg_price,
                position.side, position.opened_at, position.last_updated,
                position.unrealized_pnl, position.realized_pnl, position.metadata
            ))

    def get_position(self, symbol: str) -> Optional[PersistedPosition]:
        """Get position by symbol."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM positions WHERE symbol = ?", (symbol,)
        ).fetchone()
        if row:
            return PersistedPosition(**dict(row))
        return None

    def get_all_positions(self) -> List[PersistedPosition]:
        """Get all open positions."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM positions WHERE quantity != 0"
        ).fetchall()
        return [PersistedPosition(**dict(row)) for row in rows]

    def delete_position(self, symbol: str) -> None:
        """Delete a position (when closed)."""
        with self._transaction() as conn:
            conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))

    # ========== Order Operations ==========

    def save_order(self, order: PersistedOrder) -> None:
        """Save or update an order."""
        with self._transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO orders
                (order_id, symbol, side, quantity, order_type, status,
                 created_at, updated_at, filled_quantity, avg_fill_price,
                 limit_price, stop_price, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order.order_id, order.symbol, order.side, order.quantity,
                order.order_type, order.status, order.created_at, order.updated_at,
                order.filled_quantity, order.avg_fill_price, order.limit_price,
                order.stop_price, order.metadata
            ))

    def get_pending_orders(self) -> List[PersistedOrder]:
        """Get all pending orders."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM orders WHERE status IN ('pending', 'open', 'partial')"
        ).fetchall()
        return [PersistedOrder(**dict(row)) for row in rows]

    def get_order(self, order_id: str) -> Optional[PersistedOrder]:
        """Get order by ID."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        if row:
            return PersistedOrder(**dict(row))
        return None

    # ========== System State Operations ==========

    def save_system_state(
        self,
        state_id: str,
        cash_balance: float,
        total_equity: float,
        positions: List[Dict],
        pending_orders: List[Dict],
        metadata: Optional[Dict] = None,
    ) -> None:
        """Save a system state snapshot."""
        with self._transaction() as conn:
            conn.execute("""
                INSERT INTO system_state
                (state_id, timestamp, cash_balance, total_equity,
                 positions_json, pending_orders_json, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                state_id,
                datetime.now(timezone.utc).isoformat(),
                cash_balance,
                total_equity,
                json.dumps(positions),
                json.dumps(pending_orders),
                json.dumps(metadata or {}),
            ))

    def get_latest_state(self) -> Optional[SystemState]:
        """Get the most recent system state snapshot."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM system_state ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if row:
            return SystemState(**dict(row))
        return None

    def get_state_by_id(self, state_id: str) -> Optional[SystemState]:
        """Get system state by ID."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM system_state WHERE state_id = ?", (state_id,)
        ).fetchone()
        if row:
            return SystemState(**dict(row))
        return None

    # ========== Trade History ==========

    def record_trade(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        pnl: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Record a completed trade."""
        with self._transaction() as conn:
            conn.execute("""
                INSERT INTO trade_history
                (order_id, symbol, side, quantity, price, pnl, executed_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                order_id, symbol, side, quantity, price, pnl,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(metadata or {}),
            ))

    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get trade history."""
        conn = self._get_connection()
        if symbol:
            rows = conn.execute(
                "SELECT * FROM trade_history WHERE symbol = ? ORDER BY executed_at DESC LIMIT ?",
                (symbol, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trade_history ORDER BY executed_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    # ========== Cleanup ==========

    def cleanup_old_states(self, keep_days: int = 7) -> int:
        """Remove old state snapshots."""
        with self._transaction() as conn:
            cursor = conn.execute("""
                DELETE FROM system_state
                WHERE timestamp < datetime('now', ?)
            """, (f'-{keep_days} days',))
            return cursor.rowcount

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None


# Singleton instance
_persistence: Optional[StatePersistence] = None


def get_persistence() -> StatePersistence:
    """Get singleton persistence instance."""
    global _persistence
    if _persistence is None:
        _persistence = StatePersistence()
    return _persistence
