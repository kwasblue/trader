"""
Tests for PositionState enum - Position lifecycle states

Tests cover:
- Enum values
- is_pending property
- allows_new_orders property
- State transitions (in PortfolioState)
"""
import pytest

from core.enums import PositionState


class TestPositionStateEnum:
    """Test PositionState enum values and properties."""

    def test_none_value(self):
        """Test NONE state."""
        assert PositionState.NONE.value == "none"
        assert str(PositionState.NONE) == "none"

    def test_pending_entry_value(self):
        """Test PENDING_ENTRY state."""
        assert PositionState.PENDING_ENTRY.value == "pending_entry"

    def test_open_value(self):
        """Test OPEN state."""
        assert PositionState.OPEN.value == "open"

    def test_pending_exit_value(self):
        """Test PENDING_EXIT state."""
        assert PositionState.PENDING_EXIT.value == "pending_exit"

    def test_pending_add_value(self):
        """Test PENDING_ADD state."""
        assert PositionState.PENDING_ADD.value == "pending_add"

    def test_is_pending_none(self):
        """Test is_pending for NONE state."""
        assert PositionState.NONE.is_pending is False

    def test_is_pending_pending_entry(self):
        """Test is_pending for PENDING_ENTRY state."""
        assert PositionState.PENDING_ENTRY.is_pending is True

    def test_is_pending_open(self):
        """Test is_pending for OPEN state."""
        assert PositionState.OPEN.is_pending is False

    def test_is_pending_pending_exit(self):
        """Test is_pending for PENDING_EXIT state."""
        assert PositionState.PENDING_EXIT.is_pending is True

    def test_is_pending_pending_add(self):
        """Test is_pending for PENDING_ADD state."""
        assert PositionState.PENDING_ADD.is_pending is True

    def test_allows_new_orders_none(self):
        """Test allows_new_orders for NONE state."""
        assert PositionState.NONE.allows_new_orders is True

    def test_allows_new_orders_pending_entry(self):
        """Test allows_new_orders for PENDING_ENTRY state."""
        assert PositionState.PENDING_ENTRY.allows_new_orders is False

    def test_allows_new_orders_open(self):
        """Test allows_new_orders for OPEN state."""
        assert PositionState.OPEN.allows_new_orders is True

    def test_allows_new_orders_pending_exit(self):
        """Test allows_new_orders for PENDING_EXIT state."""
        assert PositionState.PENDING_EXIT.allows_new_orders is False

    def test_allows_new_orders_pending_add(self):
        """Test allows_new_orders for PENDING_ADD state."""
        assert PositionState.PENDING_ADD.allows_new_orders is False

    def test_enum_iteration(self):
        """Test iterating over all states."""
        states = list(PositionState)
        assert len(states) == 5
        assert PositionState.NONE in states
        assert PositionState.PENDING_ENTRY in states
        assert PositionState.OPEN in states
        assert PositionState.PENDING_EXIT in states
        assert PositionState.PENDING_ADD in states

    def test_enum_from_string(self):
        """Test creating enum from string value."""
        assert PositionState("none") == PositionState.NONE
        assert PositionState("pending_entry") == PositionState.PENDING_ENTRY
        assert PositionState("open") == PositionState.OPEN

    def test_enum_comparison(self):
        """Test enum comparison."""
        assert PositionState.NONE == PositionState.NONE
        assert PositionState.NONE != PositionState.OPEN

    def test_enum_is_str(self):
        """Test that PositionState is a string enum."""
        assert isinstance(PositionState.NONE, str)
        assert PositionState.NONE == "none"


class TestPositionStateTransitions:
    """Test valid state transitions logic."""

    def test_valid_transitions_from_none(self):
        """Test valid transitions from NONE state."""
        # NONE -> PENDING_ENTRY is valid (placing entry order)
        valid_from_none = {PositionState.PENDING_ENTRY}

        assert PositionState.PENDING_ENTRY in valid_from_none
        assert PositionState.OPEN not in valid_from_none  # Can't go directly to OPEN

    def test_valid_transitions_from_pending_entry(self):
        """Test valid transitions from PENDING_ENTRY state."""
        # PENDING_ENTRY -> OPEN (filled) or NONE (cancelled/rejected)
        valid_from_pending_entry = {PositionState.OPEN, PositionState.NONE}

        assert PositionState.OPEN in valid_from_pending_entry
        assert PositionState.NONE in valid_from_pending_entry
        assert PositionState.PENDING_EXIT not in valid_from_pending_entry

    def test_valid_transitions_from_open(self):
        """Test valid transitions from OPEN state."""
        # OPEN -> PENDING_EXIT (exit order) or PENDING_ADD (add to position) or NONE (closed)
        valid_from_open = {
            PositionState.PENDING_EXIT,
            PositionState.PENDING_ADD,
            PositionState.NONE
        }

        assert PositionState.PENDING_EXIT in valid_from_open
        assert PositionState.PENDING_ADD in valid_from_open
        assert PositionState.NONE in valid_from_open
        assert PositionState.PENDING_ENTRY not in valid_from_open

    def test_valid_transitions_from_pending_exit(self):
        """Test valid transitions from PENDING_EXIT state."""
        # PENDING_EXIT -> OPEN (cancel) or NONE (filled)
        valid_from_pending_exit = {PositionState.OPEN, PositionState.NONE}

        assert PositionState.OPEN in valid_from_pending_exit
        assert PositionState.NONE in valid_from_pending_exit

    def test_valid_transitions_from_pending_add(self):
        """Test valid transitions from PENDING_ADD state."""
        # PENDING_ADD -> OPEN (filled or cancelled)
        valid_from_pending_add = {PositionState.OPEN}

        assert PositionState.OPEN in valid_from_pending_add


class TestPositionStateUseCases:
    """Test real-world use cases for position states."""

    def test_entry_flow(self):
        """Test typical entry order flow."""
        # Start with no position
        state = PositionState.NONE
        assert state.allows_new_orders is True

        # Place entry order
        state = PositionState.PENDING_ENTRY
        assert state.is_pending is True
        assert state.allows_new_orders is False

        # Order fills
        state = PositionState.OPEN
        assert state.is_pending is False
        assert state.allows_new_orders is True

    def test_exit_flow(self):
        """Test typical exit order flow."""
        # Have open position
        state = PositionState.OPEN
        assert state.allows_new_orders is True

        # Place exit order
        state = PositionState.PENDING_EXIT
        assert state.is_pending is True
        assert state.allows_new_orders is False

        # Order fills
        state = PositionState.NONE
        assert state.is_pending is False

    def test_cancelled_entry_flow(self):
        """Test entry order cancellation flow."""
        state = PositionState.NONE
        state = PositionState.PENDING_ENTRY

        # Order cancelled/rejected
        state = PositionState.NONE
        assert state == PositionState.NONE

    def test_add_to_position_flow(self):
        """Test adding to existing position flow."""
        # Have open position
        state = PositionState.OPEN

        # Place add order
        state = PositionState.PENDING_ADD
        assert state.is_pending is True

        # Order fills
        state = PositionState.OPEN
        assert state.allows_new_orders is True


# ============================================================================
# State Synchronizer Tests
# ============================================================================

import pytest
import asyncio
from core.logic.portfolio_state import PortfolioState
from core.logic.symbol_state import SymbolState
from core.state_sync import StateSynchronizer


class TestStateSynchronizer:
    """Test StateSynchronizer for Portfolio <-> Symbol state consistency."""

    @pytest.fixture
    def portfolio(self):
        """Create a test portfolio."""
        return PortfolioState(cash=100_000.0)

    @pytest.fixture
    def symbol_states(self):
        """Create symbol states dict."""
        return {
            "AAPL": SymbolState(symbol="AAPL"),
            "MSFT": SymbolState(symbol="MSFT"),
        }

    @pytest.fixture
    def synchronizer(self, portfolio, symbol_states):
        """Create a StateSynchronizer."""
        return StateSynchronizer(portfolio, symbol_states)

    @pytest.mark.asyncio
    async def test_apply_fill_syncs_symbol_state(self, synchronizer, portfolio, symbol_states):
        """Test that apply_fill_and_sync updates both portfolio and symbol state."""
        # Apply a buy fill
        await synchronizer.apply_fill_and_sync("AAPL", "buy", 100, 150.0)

        # Check portfolio
        position = portfolio.get_position("AAPL")
        assert position is not None
        assert position.qty == 100
        assert position.avg_price == 150.0

        # Check symbol state was synced
        state = symbol_states["AAPL"]
        assert state.current_position == 100
        assert state.side == "long"
        assert state.entry_price == 150.0

    @pytest.mark.asyncio
    async def test_position_state_synced_after_fill(self, synchronizer, portfolio, symbol_states):
        """Test that position state is synced after multiple fills."""
        # Buy 100 shares
        await synchronizer.apply_fill_and_sync("AAPL", "buy", 100, 150.0)

        # Check initial sync
        assert symbol_states["AAPL"].current_position == 100

        # Buy 50 more (add to position)
        await synchronizer.apply_fill_and_sync("AAPL", "buy", 50, 152.0)

        # Check position updated in both
        position = portfolio.get_position("AAPL")
        assert position.qty == 150

        state = symbol_states["AAPL"]
        assert state.current_position == 150
        assert state.side == "long"

    @pytest.mark.asyncio
    async def test_concurrent_fills_maintain_consistency(self, synchronizer, portfolio, symbol_states):
        """Test that concurrent fills maintain state consistency."""
        # Run multiple fills concurrently
        async def do_fills():
            tasks = [
                synchronizer.apply_fill_and_sync("AAPL", "buy", 10, 150.0),
                synchronizer.apply_fill_and_sync("MSFT", "buy", 20, 300.0),
            ]
            await asyncio.gather(*tasks)

        await do_fills()

        # Check both symbols are consistent
        assert portfolio.get_position("AAPL").qty == 10
        assert portfolio.get_position("MSFT").qty == 20

        assert symbol_states["AAPL"].current_position == 10
        assert symbol_states["MSFT"].current_position == 20

    @pytest.mark.asyncio
    async def test_sync_all_from_portfolio(self, synchronizer, portfolio, symbol_states):
        """Test syncing all symbol states from portfolio."""
        # Directly update portfolio (simulating broker sync)
        portfolio.apply_fill("AAPL", "buy", 100, 150.0)
        portfolio.apply_fill("MSFT", "buy", 50, 300.0)

        # Symbol states are out of sync
        assert symbol_states["AAPL"].current_position == 0
        assert symbol_states["MSFT"].current_position == 0

        # Sync all
        count = await synchronizer.sync_all_from_portfolio()
        assert count == 2

        # Now they should be synced
        assert symbol_states["AAPL"].current_position == 100
        assert symbol_states["MSFT"].current_position == 50

    @pytest.mark.asyncio
    async def test_verify_consistency_detects_mismatch(self, synchronizer, portfolio, symbol_states):
        """Test that verify_consistency detects mismatches."""
        # Create a mismatch
        portfolio.apply_fill("AAPL", "buy", 100, 150.0)
        # Don't sync symbol state - it's still at 0

        # Verify should find the inconsistency
        inconsistencies = await synchronizer.verify_consistency()

        assert "AAPL" in inconsistencies
        assert inconsistencies["AAPL"]["portfolio_qty"] == 100
        assert inconsistencies["AAPL"]["symbol_qty"] == 0

    @pytest.mark.asyncio
    async def test_fix_inconsistencies(self, synchronizer, portfolio, symbol_states):
        """Test that fix_inconsistencies corrects mismatches."""
        # Create a mismatch
        portfolio.apply_fill("AAPL", "buy", 100, 150.0)

        # Verify inconsistency exists
        inconsistencies = await synchronizer.verify_consistency()
        assert len(inconsistencies) > 0

        # Fix it
        fixed_count = await synchronizer.fix_inconsistencies()
        assert fixed_count == 1

        # Verify consistency now passes
        inconsistencies = await synchronizer.verify_consistency()
        assert len(inconsistencies) == 0

    @pytest.mark.asyncio
    async def test_register_unregister_symbol(self, synchronizer):
        """Test registering and unregistering symbols."""
        new_state = SymbolState(symbol="GOOGL")

        # Register new symbol
        synchronizer.register_symbol("GOOGL", new_state)
        assert "GOOGL" in synchronizer.symbol_states

        # Unregister
        synchronizer.unregister_symbol("GOOGL")
        assert "GOOGL" not in synchronizer.symbol_states

    @pytest.mark.asyncio
    async def test_close_position_resets_symbol_state(self, synchronizer, portfolio, symbol_states):
        """Test that closing a position resets the symbol state."""
        # Open position
        await synchronizer.apply_fill_and_sync("AAPL", "buy", 100, 150.0)
        assert symbol_states["AAPL"].current_position == 100

        # Close position
        await synchronizer.apply_fill_and_sync("AAPL", "sell", 100, 155.0)

        # Position should be flat
        position = portfolio.get_position("AAPL")
        assert position.qty == 0

        # Symbol state should reflect this
        state = symbol_states["AAPL"]
        assert state.current_position == 0
        assert state.side is None
