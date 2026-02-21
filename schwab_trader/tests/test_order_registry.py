"""
Tests for OrderRegistry - Local order lifecycle tracking

Tests cover:
- Order registration and retrieval
- Status updates
- Pending order checks
- Conflicting order detection
- Concurrent access safety
"""
import asyncio
import pytest
from datetime import datetime, timezone

from core.order_registry import OrderRegistry, TrackedOrder


class TestTrackedOrder:
    """Test TrackedOrder dataclass."""

    def test_tracked_order_creation(self):
        """Test creating a TrackedOrder."""
        order = TrackedOrder(
            order_id="ORD123",
            symbol="AAPL",
            side="buy",
            qty=10,
            status="pending",
            correlation_id="corr123",
        )

        assert order.order_id == "ORD123"
        assert order.symbol == "AAPL"
        assert order.side == "buy"
        assert order.qty == 10
        assert order.status == "pending"
        assert order.correlation_id == "corr123"
        assert order.filled_qty == 0

    def test_is_open_pending(self):
        """Test is_open for pending orders."""
        order = TrackedOrder(
            order_id="ORD123",
            symbol="AAPL",
            side="buy",
            qty=10,
            status="pending",
            correlation_id="corr123",
        )
        assert order.is_open is True

    def test_is_open_accepted(self):
        """Test is_open for accepted orders."""
        order = TrackedOrder(
            order_id="ORD123",
            symbol="AAPL",
            side="buy",
            qty=10,
            status="accepted",
            correlation_id="corr123",
        )
        assert order.is_open is True

    def test_is_open_filled(self):
        """Test is_open for filled orders."""
        order = TrackedOrder(
            order_id="ORD123",
            symbol="AAPL",
            side="buy",
            qty=10,
            status="filled",
            correlation_id="corr123",
        )
        assert order.is_open is False

    def test_is_filled(self):
        """Test is_filled property."""
        order = TrackedOrder(
            order_id="ORD123",
            symbol="AAPL",
            side="buy",
            qty=10,
            status="filled",
            correlation_id="corr123",
        )
        assert order.is_filled is True

    def test_is_cancelled(self):
        """Test is_cancelled property."""
        order = TrackedOrder(
            order_id="ORD123",
            symbol="AAPL",
            side="buy",
            qty=10,
            status="cancelled",
            correlation_id="corr123",
        )
        assert order.is_cancelled is True

    def test_to_dict(self):
        """Test converting order to dictionary."""
        order = TrackedOrder(
            order_id="ORD123",
            symbol="AAPL",
            side="buy",
            qty=10,
            status="pending",
            correlation_id="corr123",
        )
        d = order.to_dict()

        assert d["order_id"] == "ORD123"
        assert d["symbol"] == "AAPL"
        assert d["side"] == "buy"
        assert "created_at" in d


class TestOrderRegistry:
    """Test OrderRegistry functionality."""

    @pytest.fixture
    def registry(self):
        """Create fresh registry for each test."""
        return OrderRegistry()

    @pytest.mark.asyncio
    async def test_register_order(self, registry):
        """Test registering an order."""
        order = await registry.register(
            order_id="ORD123",
            symbol="AAPL",
            side="buy",
            qty=10,
            correlation_id="corr123",
        )

        assert order.order_id == "ORD123"
        assert order.symbol == "AAPL"
        assert len(registry) == 1

    @pytest.mark.asyncio
    async def test_get_order(self, registry):
        """Test getting an order by ID."""
        await registry.register(
            order_id="ORD123",
            symbol="AAPL",
            side="buy",
            qty=10,
            correlation_id="corr123",
        )

        order = await registry.get("ORD123")
        assert order is not None
        assert order.symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_get_nonexistent_order(self, registry):
        """Test getting nonexistent order returns None."""
        order = await registry.get("NONEXISTENT")
        assert order is None

    @pytest.mark.asyncio
    async def test_update_status(self, registry):
        """Test updating order status."""
        await registry.register(
            order_id="ORD123",
            symbol="AAPL",
            side="buy",
            qty=10,
            correlation_id="corr123",
        )

        updated = await registry.update_status("ORD123", "filled", 10)
        assert updated is not None
        assert updated.status == "filled"
        assert updated.filled_qty == 10

    @pytest.mark.asyncio
    async def test_update_nonexistent_order(self, registry):
        """Test updating nonexistent order returns None."""
        result = await registry.update_status("NONEXISTENT", "filled")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_pending_for_symbol(self, registry):
        """Test getting pending orders for a symbol."""
        await registry.register("ORD1", "AAPL", "buy", 10, "corr1")
        await registry.register("ORD2", "AAPL", "sell", 5, "corr2")
        await registry.register("ORD3", "GOOG", "buy", 20, "corr3")

        # Mark one as filled
        await registry.update_status("ORD2", "filled")

        pending = await registry.get_pending_for_symbol("AAPL")
        assert len(pending) == 1
        assert pending[0].order_id == "ORD1"

    @pytest.mark.asyncio
    async def test_has_pending_orders(self, registry):
        """Test checking for pending orders."""
        await registry.register("ORD1", "AAPL", "buy", 10, "corr1")

        assert await registry.has_pending_orders("AAPL") is True
        assert await registry.has_pending_orders("GOOG") is False

        # Mark as filled
        await registry.update_status("ORD1", "filled")
        assert await registry.has_pending_orders("AAPL") is False

    @pytest.mark.asyncio
    async def test_get_conflicting_orders(self, registry):
        """Test detecting conflicting orders."""
        # Register a buy order
        await registry.register("ORD1", "AAPL", "buy", 10, "corr1")

        # Check for conflicts when placing a sell order
        conflicting = await registry.get_conflicting_orders("AAPL", "sell")
        assert len(conflicting) == 1
        assert conflicting[0].order_id == "ORD1"

        # No conflicts for same side
        no_conflict = await registry.get_conflicting_orders("AAPL", "buy")
        assert len(no_conflict) == 0

    @pytest.mark.asyncio
    async def test_remove_order(self, registry):
        """Test removing an order."""
        await registry.register("ORD1", "AAPL", "buy", 10, "corr1")
        assert len(registry) == 1

        removed = await registry.remove("ORD1")
        assert removed is not None
        assert removed.order_id == "ORD1"
        assert len(registry) == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent_order(self, registry):
        """Test removing nonexistent order."""
        removed = await registry.remove("NONEXISTENT")
        assert removed is None

    @pytest.mark.asyncio
    async def test_clear_symbol(self, registry):
        """Test clearing all orders for a symbol."""
        await registry.register("ORD1", "AAPL", "buy", 10, "corr1")
        await registry.register("ORD2", "AAPL", "sell", 5, "corr2")
        await registry.register("ORD3", "GOOG", "buy", 20, "corr3")

        removed = await registry.clear_symbol("AAPL")
        assert len(removed) == 2
        assert len(registry) == 1

    @pytest.mark.asyncio
    async def test_get_all_pending(self, registry):
        """Test getting all pending orders."""
        await registry.register("ORD1", "AAPL", "buy", 10, "corr1")
        await registry.register("ORD2", "GOOG", "sell", 5, "corr2")
        await registry.update_status("ORD2", "filled")

        pending = await registry.get_all_pending()
        assert len(pending) == 1
        assert pending[0].order_id == "ORD1"

    @pytest.mark.asyncio
    async def test_cleanup_closed(self, registry):
        """Test cleaning up closed orders."""
        await registry.register("ORD1", "AAPL", "buy", 10, "corr1")
        await registry.register("ORD2", "GOOG", "sell", 5, "corr2")
        await registry.update_status("ORD1", "filled")
        await registry.update_status("ORD2", "cancelled")

        count = await registry.cleanup_closed()
        assert count == 2
        assert len(registry) == 0

    @pytest.mark.asyncio
    async def test_concurrent_access(self, registry):
        """Test concurrent access to registry."""
        async def register_order(i):
            await registry.register(
                order_id=f"ORD{i}",
                symbol="AAPL",
                side="buy" if i % 2 == 0 else "sell",
                qty=10,
                correlation_id=f"corr{i}",
            )

        # Register 100 orders concurrently
        await asyncio.gather(*[register_order(i) for i in range(100)])

        assert len(registry) == 100

    @pytest.mark.asyncio
    async def test_concurrent_updates(self, registry):
        """Test concurrent status updates."""
        # Register orders first
        for i in range(50):
            await registry.register(f"ORD{i}", "AAPL", "buy", 10, f"corr{i}")

        async def update_order(i):
            await registry.update_status(f"ORD{i}", "filled", 10)

        # Update all concurrently
        await asyncio.gather(*[update_order(i) for i in range(50)])

        # All should be filled
        pending = await registry.get_all_pending()
        assert len(pending) == 0

    def test_repr(self, registry):
        """Test string representation."""
        assert "OrderRegistry" in repr(registry)
        assert "total=0" in repr(registry)
