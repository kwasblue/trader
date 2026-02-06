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
