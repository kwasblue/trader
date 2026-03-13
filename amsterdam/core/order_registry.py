"""
Order Registry - Local order lifecycle tracking

Maintains a local cache of orders to avoid repeated broker queries
and enable fast lookups for pending orders and conflict detection.

Features:
- Thread-safe order tracking with asyncio.Lock
- Fast symbol-based lookups
- Conflict detection for wash trade prevention
- Order lifecycle management (register, update, remove)
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

from core.enums import OrderStatus


@dataclass
class TrackedOrder:
    """
    A locally tracked order with metadata for lifecycle management.

    Attributes:
        order_id: Unique broker order ID
        symbol: Trading symbol
        side: Order side ("buy" or "sell")
        qty: Order quantity
        status: Current order status (OrderStatus enum)
        correlation_id: ID linking related operations
        created_at: When order was placed
        filled_qty: Quantity filled so far (for partial fills)
    """
    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    qty: int
    status: OrderStatus
    correlation_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filled_qty: int = 0

    @property
    def is_open(self) -> bool:
        """Check if order is still open (not filled/cancelled/rejected)."""
        return self.status.is_open

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED

    @property
    def is_cancelled(self) -> bool:
        """Check if order was cancelled."""
        return self.status == OrderStatus.CANCELLED

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "status": self.status.value,
            "correlation_id": self.correlation_id,
            "created_at": self.created_at.isoformat(),
            "filled_qty": self.filled_qty,
        }


class OrderRegistry:
    """
    Local order tracking registry with thread-safe operations.

    Provides fast lookups by order ID or symbol without broker queries.
    Essential for:
    - Detecting pending orders before position deletion
    - Finding conflicting orders for wash trade prevention
    - Tracking order lifecycle for state management
    """

    # Valid order status transitions
    VALID_TRANSITIONS: Dict[OrderStatus, set] = {
        OrderStatus.PENDING: {OrderStatus.ACCEPTED, OrderStatus.REJECTED, OrderStatus.FILLED},
        OrderStatus.ACCEPTED: {OrderStatus.PARTIAL, OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED},
        OrderStatus.PARTIAL: {OrderStatus.FILLED, OrderStatus.CANCELLED},
        OrderStatus.FILLED: set(),      # Terminal
        OrderStatus.CANCELLED: set(),   # Terminal
        OrderStatus.REJECTED: set(),    # Terminal
        OrderStatus.EXPIRED: set(),     # Terminal
    }

    def __init__(self):
        """Initialize empty order registry."""
        self._orders: Dict[str, TrackedOrder] = {}
        self._by_symbol: Dict[str, List[str]] = {}  # symbol -> [order_ids]
        self._lock = asyncio.Lock()

    @staticmethod
    def _normalize_status(status: str | OrderStatus) -> OrderStatus:
        """Convert string status to OrderStatus enum."""
        if isinstance(status, OrderStatus):
            return status
        status_str = status.lower().strip()
        # Map common variations
        status_map = {
            "pending": OrderStatus.PENDING,
            "new": OrderStatus.PENDING,
            "open": OrderStatus.ACCEPTED,
            "accepted": OrderStatus.ACCEPTED,
            "partial": OrderStatus.PARTIAL,
            "partially_filled": OrderStatus.PARTIAL,
            "filled": OrderStatus.FILLED,
            "completed": OrderStatus.FILLED,
            "cancelled": OrderStatus.CANCELLED,
            "canceled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.EXPIRED,
        }
        return status_map.get(status_str, OrderStatus.PENDING)

    def _is_valid_transition(self, current: OrderStatus, new: OrderStatus) -> bool:
        """Check if a status transition is valid."""
        # Allow same-state updates (idempotent)
        if current == new:
            return True
        allowed = self.VALID_TRANSITIONS.get(current, set())
        return new in allowed

    async def register(
        self,
        order_id: str,
        symbol: str,
        side: str,
        qty: int,
        correlation_id: str,
        status: str | OrderStatus = OrderStatus.PENDING
    ) -> TrackedOrder:
        """
        Register a new order in the registry.

        Args:
            order_id: Broker order ID
            symbol: Trading symbol
            side: Order side ("buy" or "sell")
            qty: Order quantity
            correlation_id: Correlation ID for tracking
            status: Initial order status (string or OrderStatus)

        Returns:
            The tracked order object
        """
        async with self._lock:
            order = TrackedOrder(
                order_id=order_id,
                symbol=symbol,
                side=side.lower(),
                qty=qty,
                status=self._normalize_status(status),
                correlation_id=correlation_id,
            )

            self._orders[order_id] = order

            if symbol not in self._by_symbol:
                self._by_symbol[symbol] = []
            self._by_symbol[symbol].append(order_id)

            return order

    async def update_status(
        self,
        order_id: str,
        status: str | OrderStatus,
        filled_qty: Optional[int] = None
    ) -> Optional[TrackedOrder]:
        """
        Update the status of a tracked order.

        Validates that the transition is allowed per order lifecycle rules.

        Args:
            order_id: Order ID to update
            status: New status (string or OrderStatus)
            filled_qty: Updated filled quantity (optional)

        Returns:
            Updated order or None if not found or transition invalid
        """
        new_status = self._normalize_status(status)

        async with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                return None

            # Validate transition
            if not self._is_valid_transition(order.status, new_status):
                # Log warning but allow - broker is source of truth
                import logging
                logging.getLogger("OrderRegistry").warning(
                    f"Invalid order status transition for {order_id}: "
                    f"{order.status.value} -> {new_status.value}"
                )

            order.status = new_status
            if filled_qty is not None:
                order.filled_qty = filled_qty

            return order

    async def get(self, order_id: str) -> Optional[TrackedOrder]:
        """
        Get a tracked order by ID.

        Args:
            order_id: Order ID to look up

        Returns:
            TrackedOrder or None if not found
        """
        async with self._lock:
            return self._orders.get(order_id)

    async def get_pending_for_symbol(self, symbol: str) -> List[TrackedOrder]:
        """
        Get all pending (open) orders for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of open orders for the symbol
        """
        async with self._lock:
            order_ids = self._by_symbol.get(symbol, [])
            pending = []
            for oid in order_ids:
                order = self._orders.get(oid)
                if order and order.is_open:
                    pending.append(order)
            return pending

    async def has_pending_orders(self, symbol: str) -> bool:
        """
        Check if a symbol has any pending orders.

        Fast check without returning the actual orders.

        Args:
            symbol: Trading symbol

        Returns:
            True if any open orders exist for symbol
        """
        async with self._lock:
            order_ids = self._by_symbol.get(symbol, [])
            for oid in order_ids:
                order = self._orders.get(oid)
                if order and order.is_open:
                    return True
            return False

    async def get_conflicting_orders(
        self,
        symbol: str,
        side: str
    ) -> List[TrackedOrder]:
        """
        Get orders that would conflict with a new order (opposite side).

        Used for wash trade prevention - finds open orders on the opposite
        side that need to be cancelled before placing a new order.

        Args:
            symbol: Trading symbol
            side: Side of the new order ("buy" or "sell")

        Returns:
            List of conflicting open orders
        """
        async with self._lock:
            order_ids = self._by_symbol.get(symbol, [])
            opposite_side = "sell" if side.lower() == "buy" else "buy"
            conflicting = []

            for oid in order_ids:
                order = self._orders.get(oid)
                if order and order.is_open and order.side == opposite_side:
                    conflicting.append(order)

            return conflicting

    async def remove(self, order_id: str) -> Optional[TrackedOrder]:
        """
        Remove an order from the registry.

        Call this when an order is filled, cancelled, or no longer relevant.

        Args:
            order_id: Order ID to remove

        Returns:
            Removed order or None if not found
        """
        async with self._lock:
            order = self._orders.pop(order_id, None)
            if order:
                symbol_orders = self._by_symbol.get(order.symbol, [])
                if order_id in symbol_orders:
                    symbol_orders.remove(order_id)
            return order

    async def clear_symbol(self, symbol: str) -> List[TrackedOrder]:
        """
        Remove all orders for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of removed orders
        """
        async with self._lock:
            order_ids = self._by_symbol.pop(symbol, [])
            removed = []
            for oid in order_ids:
                order = self._orders.pop(oid, None)
                if order:
                    removed.append(order)
            return removed

    async def get_all_pending(self) -> List[TrackedOrder]:
        """
        Get all pending orders across all symbols.

        Returns:
            List of all open orders
        """
        async with self._lock:
            return [o for o in self._orders.values() if o.is_open]

    async def cleanup_closed(self) -> int:
        """
        Remove all closed (filled/cancelled) orders.

        Returns:
            Number of orders removed
        """
        async with self._lock:
            closed_ids = [
                oid for oid, order in self._orders.items()
                if not order.is_open
            ]

            for oid in closed_ids:
                order = self._orders.pop(oid, None)
                if order:
                    symbol_orders = self._by_symbol.get(order.symbol, [])
                    if oid in symbol_orders:
                        symbol_orders.remove(oid)

            return len(closed_ids)

    def __len__(self) -> int:
        """Return total number of tracked orders."""
        return len(self._orders)

    def __repr__(self) -> str:
        """String representation."""
        pending = sum(1 for o in self._orders.values() if o.is_open)
        return f"OrderRegistry(total={len(self._orders)}, pending={pending})"
