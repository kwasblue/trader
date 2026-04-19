"""
Tests for Portfolio Concurrency - Thread-safe operations

Tests cover:
- Lock behavior under concurrent access
- Thread-safe fill application
- Position state management
- State transition validation
"""

import asyncio

import pytest

from core.enums import PositionState
from core.logic.portfolio_state import PortfolioState


class TestPortfolioStatePositionStates:
    """Test position state management in PortfolioState."""

    @pytest.fixture
    def portfolio(self):
        """Create fresh portfolio for each test."""
        return PortfolioState(cash=100000.0)

    def test_get_position_state_default(self, portfolio):
        """Test default position state is NONE."""
        state = portfolio.get_position_state("AAPL")
        assert state == PositionState.NONE

    @pytest.mark.asyncio
    async def test_set_position_state(self, portfolio):
        """Test setting position state."""
        result = await portfolio.set_position_state("AAPL", PositionState.PENDING_ENTRY)
        assert result is True
        assert portfolio.get_position_state("AAPL") == PositionState.PENDING_ENTRY

    @pytest.mark.asyncio
    async def test_valid_state_transition_none_to_pending(self, portfolio):
        """Test valid transition from NONE to PENDING_ENTRY."""
        result = await portfolio.set_position_state("AAPL", PositionState.PENDING_ENTRY)
        assert result is True

    @pytest.mark.asyncio
    async def test_valid_state_transition_pending_to_open(self, portfolio):
        """Test valid transition from PENDING_ENTRY to OPEN."""
        await portfolio.set_position_state("AAPL", PositionState.PENDING_ENTRY)
        result = await portfolio.set_position_state("AAPL", PositionState.OPEN)
        assert result is True

    @pytest.mark.asyncio
    async def test_valid_state_transition_open_to_pending_exit(self, portfolio):
        """Test valid transition from OPEN to PENDING_EXIT."""
        await portfolio.set_position_state("AAPL", PositionState.PENDING_ENTRY)
        await portfolio.set_position_state("AAPL", PositionState.OPEN)
        result = await portfolio.set_position_state("AAPL", PositionState.PENDING_EXIT)
        assert result is True

    @pytest.mark.asyncio
    async def test_invalid_state_transition(self, portfolio):
        """Test invalid state transition is rejected."""
        # Can't go from NONE directly to OPEN
        result = await portfolio.set_position_state("AAPL", PositionState.OPEN)
        assert result is False

    @pytest.mark.asyncio
    async def test_invalid_transition_pending_to_pending_exit(self, portfolio):
        """Test invalid transition from PENDING_ENTRY to PENDING_EXIT."""
        await portfolio.set_position_state("AAPL", PositionState.PENDING_ENTRY)
        result = await portfolio.set_position_state("AAPL", PositionState.PENDING_EXIT)
        assert result is False

    @pytest.mark.asyncio
    async def test_clear_position_state(self, portfolio):
        """Test clearing position state."""
        await portfolio.set_position_state("AAPL", PositionState.PENDING_ENTRY)
        await portfolio.set_position_state("AAPL", PositionState.OPEN)

        await portfolio.clear_position_state("AAPL")
        assert portfolio.get_position_state("AAPL") == PositionState.NONE

    @pytest.mark.asyncio
    async def test_position_state_with_order_id(self, portfolio):
        """Test position state with associated order ID."""
        result = await portfolio.set_position_state("AAPL", PositionState.PENDING_ENTRY, order_id="ORD123")
        assert result is True


class TestApplyFillSafe:
    """Test thread-safe fill application."""

    @pytest.fixture
    def portfolio(self):
        """Create fresh portfolio for each test."""
        return PortfolioState(cash=100000.0)

    @pytest.mark.asyncio
    async def test_apply_fill_safe_buy(self, portfolio):
        """Test thread-safe buy fill."""
        await portfolio.apply_fill_safe("AAPL", "buy", 10, 150.0)

        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"].qty == 10
        assert portfolio.positions["AAPL"].avg_price == 150.0

    @pytest.mark.asyncio
    async def test_apply_fill_safe_sell(self, portfolio):
        """Test thread-safe sell fill (short)."""
        await portfolio.apply_fill_safe("AAPL", "sell", 10, 150.0)

        assert "AAPL" in portfolio.positions
        assert portfolio.positions["AAPL"].qty == -10

    @pytest.mark.asyncio
    async def test_apply_fill_safe_close_position(self, portfolio):
        """Test thread-safe position close."""
        # Open position
        await portfolio.apply_fill_safe("AAPL", "buy", 10, 150.0)

        # Close position
        await portfolio.apply_fill_safe("AAPL", "sell", 10, 155.0)

        assert portfolio.positions["AAPL"].qty == 0

    @pytest.mark.asyncio
    async def test_concurrent_fills_same_symbol(self, portfolio):
        """Test concurrent fills for same symbol are serialized."""

        # This tests that the lock prevents race conditions
        async def apply_fill(side, qty, price):
            await portfolio.apply_fill_safe("AAPL", side, qty, price)

        # Run multiple fills concurrently
        await asyncio.gather(
            apply_fill("buy", 10, 150.0),
            apply_fill("buy", 20, 151.0),
            apply_fill("buy", 30, 152.0),
        )

        # Total should be 60 shares
        assert portfolio.positions["AAPL"].qty == 60

    @pytest.mark.asyncio
    async def test_concurrent_fills_different_symbols(self, portfolio):
        """Test concurrent fills for different symbols."""

        async def apply_fill(symbol, qty, price):
            await portfolio.apply_fill_safe(symbol, "buy", qty, price)

        # Run fills for different symbols concurrently
        await asyncio.gather(
            apply_fill("AAPL", 10, 150.0),
            apply_fill("GOOG", 5, 2500.0),
            apply_fill("MSFT", 20, 300.0),
        )

        assert portfolio.positions["AAPL"].qty == 10
        assert portfolio.positions["GOOG"].qty == 5
        assert portfolio.positions["MSFT"].qty == 20


class TestConcurrentStateTransitions:
    """Test concurrent position state transitions."""

    @pytest.fixture
    def portfolio(self):
        """Create fresh portfolio for each test."""
        return PortfolioState(cash=100000.0)

    @pytest.mark.asyncio
    async def test_concurrent_state_updates_same_symbol(self, portfolio):
        """Test concurrent state updates for same symbol are serialized."""
        # First transition to PENDING_ENTRY
        await portfolio.set_position_state("AAPL", PositionState.PENDING_ENTRY)

        # Try to do concurrent transitions
        results = await asyncio.gather(
            portfolio.set_position_state("AAPL", PositionState.OPEN),
            portfolio.set_position_state("AAPL", PositionState.OPEN),
            return_exceptions=True,
        )

        # At least one should succeed
        assert any(r is True for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_state_updates_different_symbols(self, portfolio):
        """Test concurrent state updates for different symbols."""

        async def transition_to_pending(symbol):
            return await portfolio.set_position_state(symbol, PositionState.PENDING_ENTRY)

        results = await asyncio.gather(
            transition_to_pending("AAPL"),
            transition_to_pending("GOOG"),
            transition_to_pending("MSFT"),
        )

        # All should succeed (different symbols)
        assert all(results)
        assert portfolio.get_position_state("AAPL") == PositionState.PENDING_ENTRY
        assert portfolio.get_position_state("GOOG") == PositionState.PENDING_ENTRY
        assert portfolio.get_position_state("MSFT") == PositionState.PENDING_ENTRY


class TestPortfolioLockBehavior:
    """Test lock behavior in PortfolioState."""

    @pytest.fixture
    def portfolio(self):
        """Create fresh portfolio for each test."""
        return PortfolioState(cash=100000.0)

    @pytest.mark.asyncio
    async def test_lock_exists(self, portfolio):
        """Test that portfolio has a lock."""
        assert hasattr(portfolio, "_lock")
        assert isinstance(portfolio._lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_lock_not_acquired_by_default(self, portfolio):
        """Test that lock is not held by default."""
        assert not portfolio._lock.locked()

    @pytest.mark.asyncio
    async def test_lock_acquired_during_fill(self, portfolio):
        """Test that lock is acquired during fill operation."""
        # We can test this indirectly by checking concurrent access works
        lock_was_used = False

        original_apply_fill = portfolio.apply_fill

        def tracking_apply_fill(*args, **kwargs):
            nonlocal lock_was_used
            lock_was_used = portfolio._lock.locked()
            return original_apply_fill(*args, **kwargs)

        portfolio.apply_fill = tracking_apply_fill

        await portfolio.apply_fill_safe("AAPL", "buy", 10, 150.0)

        # Lock should have been acquired during the operation
        assert lock_was_used

    @pytest.mark.asyncio
    async def test_lock_released_after_operation(self, portfolio):
        """Test that lock is released after operation."""
        await portfolio.apply_fill_safe("AAPL", "buy", 10, 150.0)
        assert not portfolio._lock.locked()

    @pytest.mark.asyncio
    async def test_lock_released_on_error(self, portfolio):
        """Test that lock is released even on error."""
        try:
            # This should raise an error (invalid qty)
            await portfolio.apply_fill_safe("AAPL", "buy", -10, 150.0)
        except ValueError:
            pass

        # Lock should still be released
        assert not portfolio._lock.locked()


class TestPortfolioStateIntegration:
    """Integration tests for portfolio state management."""

    @pytest.fixture
    def portfolio(self):
        """Create fresh portfolio for each test."""
        return PortfolioState(cash=100000.0)

    @pytest.mark.asyncio
    async def test_full_trade_lifecycle(self, portfolio):
        """Test complete trade lifecycle with state tracking."""
        symbol = "AAPL"

        # 1. Start with no position
        assert portfolio.get_position_state(symbol) == PositionState.NONE

        # 2. Place entry order
        result = await portfolio.set_position_state(symbol, PositionState.PENDING_ENTRY)
        assert result is True
        assert portfolio.get_position_state(symbol) == PositionState.PENDING_ENTRY

        # 3. Order fills
        await portfolio.apply_fill_safe(symbol, "buy", 10, 150.0)
        result = await portfolio.set_position_state(symbol, PositionState.OPEN)
        assert result is True
        assert portfolio.get_position_state(symbol) == PositionState.OPEN

        # 4. Place exit order
        result = await portfolio.set_position_state(symbol, PositionState.PENDING_EXIT)
        assert result is True

        # 5. Exit fills
        await portfolio.apply_fill_safe(symbol, "sell", 10, 155.0)
        result = await portfolio.set_position_state(symbol, PositionState.NONE)
        assert result is True

        # 6. Back to no position
        assert portfolio.get_position_state(symbol) == PositionState.NONE

    @pytest.mark.asyncio
    async def test_cancelled_order_lifecycle(self, portfolio):
        """Test order cancellation lifecycle."""
        symbol = "AAPL"

        # Place entry order
        await portfolio.set_position_state(symbol, PositionState.PENDING_ENTRY)

        # Order cancelled - go back to NONE
        result = await portfolio.set_position_state(symbol, PositionState.NONE)
        assert result is True
        assert portfolio.get_position_state(symbol) == PositionState.NONE

    @pytest.mark.asyncio
    async def test_add_to_position_lifecycle(self, portfolio):
        """Test adding to existing position lifecycle."""
        symbol = "AAPL"

        # Setup: Get to OPEN state
        await portfolio.set_position_state(symbol, PositionState.PENDING_ENTRY)
        await portfolio.apply_fill_safe(symbol, "buy", 10, 150.0)
        await portfolio.set_position_state(symbol, PositionState.OPEN)

        # Add to position
        result = await portfolio.set_position_state(symbol, PositionState.PENDING_ADD)
        assert result is True

        # Add fills
        await portfolio.apply_fill_safe(symbol, "buy", 10, 152.0)
        result = await portfolio.set_position_state(symbol, PositionState.OPEN)
        assert result is True

        # Should have 20 shares now
        assert portfolio.positions[symbol].qty == 20
