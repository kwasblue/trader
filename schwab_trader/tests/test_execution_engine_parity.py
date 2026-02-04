"""
Test that LiveExecutionEngine and MockExecutionEngine produce identical results
for the same input signals when using equivalent configurations.

These tests ensure backtest-live parity by verifying that:
1. _determine_action() produces identical results for same inputs
2. _setup_approval_state() produces identical state setup
3. _get_exit_quantity() produces identical exit quantities
4. Same signal stream produces same trade decisions
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone

from core.logic.live_execution_engine import LiveExecutionEngine
from core.logic.mock_execution_engine import MockExecutionEngine
from core.logic.portfolio_state import PortfolioState, SymbolPosition
from core.logic.symbol_state import SymbolState
from core.enums import OrderSide


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def shared_components():
    """Create shared components for both engines."""
    # Mock broker
    broker = MagicMock()
    broker.is_market_open = AsyncMock(return_value=True)
    broker.get_position = AsyncMock(return_value=None)
    broker.get_buying_power = MagicMock(return_value=50000.0)
    broker.get_available_funds = MagicMock(return_value=50000.0)
    broker.place_order = AsyncMock()

    # Mock executor
    executor = MagicMock()

    # Mock sizer - returns fixed quantity for testing
    sizer = MagicMock()
    sizer.calculate_position_size = MagicMock(return_value=10)

    # Portfolio state
    portfolio = PortfolioState(cash=100000)

    # Mock trade logic manager
    trade_logic_manager = MagicMock()
    mock_logic = MagicMock()
    mock_logic.should_trade = MagicMock(return_value=(True, "approved"))
    mock_logic.get_param = MagicMock(return_value=0.25)
    mock_logic.sl_mults = {"normal": 1.5, "trending": 2.0}
    trade_logic_manager.get = MagicMock(return_value=mock_logic)

    # Mock performance tracker
    performance_tracker = MagicMock()
    performance_tracker.log_trade = MagicMock()
    performance_tracker.log_error = MagicMock()

    return {
        'broker': broker,
        'executor': executor,
        'sizer': sizer,
        'portfolio': portfolio,
        'trade_logic_manager': trade_logic_manager,
        'performance_tracker': performance_tracker,
        'mock_logic': mock_logic,
    }


@pytest.fixture
def live_engine(shared_components):
    """Create LiveExecutionEngine with shared components."""
    with patch.object(LiveExecutionEngine, '_sync_portfolio_with_broker', return_value=None):
        engine = LiveExecutionEngine(
            broker=shared_components['broker'],
            executor=shared_components['executor'],
            sizer=shared_components['sizer'],
            performance_tracker=shared_components['performance_tracker'],
            trade_logic_manager=shared_components['trade_logic_manager'],
            portfolio=shared_components['portfolio'],
            sync_on_start=False
        )
    return engine


@pytest.fixture
def mock_engine(shared_components):
    """Create MockExecutionEngine with shared components."""
    engine = MockExecutionEngine(
        broker=shared_components['broker'],
        executor=shared_components['executor'],
        sizer=shared_components['sizer'],
        performance_tracker=shared_components['performance_tracker'],
        trade_logic_manager=shared_components['trade_logic_manager'],
        portfolio=shared_components['portfolio'],
    )
    return engine


# ============================================================================
# _determine_action() Parity Tests
# ============================================================================

class TestDetermineActionParity:
    """Test that _determine_action produces identical results in both engines."""

    def test_entry_long_no_position(self, live_engine, mock_engine):
        """Entry with long signal when not in position."""
        state = SymbolState(symbol="AAPL")
        state.side = None

        live_action, live_side = live_engine._determine_action("AAPL", state, 1, None)
        mock_action, mock_side = mock_engine._determine_action("AAPL", state, 1, None)

        assert live_action == mock_action == "entry"
        assert live_side == mock_side == OrderSide.BUY

    def test_entry_short_no_position(self, live_engine, mock_engine):
        """Entry with short signal when not in position."""
        state = SymbolState(symbol="AAPL")
        state.side = None

        live_action, live_side = live_engine._determine_action("AAPL", state, -1, None)
        mock_action, mock_side = mock_engine._determine_action("AAPL", state, -1, None)

        assert live_action == mock_action == "entry"
        assert live_side == mock_side == OrderSide.SELL

    def test_exit_long_position(self, live_engine, mock_engine):
        """Exit signal when in long position."""
        state = SymbolState(symbol="AAPL")
        state.side = "long"

        live_action, live_side = live_engine._determine_action("AAPL", state, -1, "exit")
        mock_action, mock_side = mock_engine._determine_action("AAPL", state, -1, "exit")

        assert live_action == mock_action == "exit"
        assert live_side == mock_side == OrderSide.SELL

    def test_exit_short_position(self, live_engine, mock_engine):
        """Exit signal when in short position."""
        state = SymbolState(symbol="AAPL")
        state.side = "short"

        live_action, live_side = live_engine._determine_action("AAPL", state, 1, "exit")
        mock_action, mock_side = mock_engine._determine_action("AAPL", state, 1, "exit")

        assert live_action == mock_action == "exit"
        assert live_side == mock_side == OrderSide.BUY

    def test_partial_exit_long(self, live_engine, mock_engine):
        """Partial exit from long position."""
        state = SymbolState(symbol="AAPL")
        state.side = "long"

        live_action, live_side = live_engine._determine_action("AAPL", state, -1, "partial exit")
        mock_action, mock_side = mock_engine._determine_action("AAPL", state, -1, "partial exit")

        assert live_action == mock_action == "partial_exit"
        assert live_side == mock_side == OrderSide.SELL

    def test_partial_exit_short(self, live_engine, mock_engine):
        """Partial exit from short position."""
        state = SymbolState(symbol="AAPL")
        state.side = "short"

        live_action, live_side = live_engine._determine_action("AAPL", state, 1, "partial exit")
        mock_action, mock_side = mock_engine._determine_action("AAPL", state, 1, "partial exit")

        assert live_action == mock_action == "partial_exit"
        assert live_side == mock_side == OrderSide.BUY

    def test_reversal_long_to_short(self, live_engine, mock_engine):
        """Reversal from long to short."""
        state = SymbolState(symbol="AAPL")
        state.side = "long"

        live_action, live_side = live_engine._determine_action("AAPL", state, -1, "reversal signal")
        mock_action, mock_side = mock_engine._determine_action("AAPL", state, -1, "reversal signal")

        assert live_action == mock_action == "reversal"
        assert live_side == mock_side == OrderSide.SELL

    def test_reversal_short_to_long(self, live_engine, mock_engine):
        """Reversal from short to long."""
        state = SymbolState(symbol="AAPL")
        state.side = "short"

        live_action, live_side = live_engine._determine_action("AAPL", state, 1, "reversal")
        mock_action, mock_side = mock_engine._determine_action("AAPL", state, 1, "reversal")

        assert live_action == mock_action == "reversal"
        assert live_side == mock_side == OrderSide.BUY


# ============================================================================
# _setup_approval_state() Parity Tests
# ============================================================================

class TestSetupApprovalStateParity:
    """Test that _setup_approval_state produces identical state setup."""

    def test_no_position(self, live_engine, mock_engine):
        """State setup with no position."""
        state_live = SymbolState(symbol="AAPL")
        state_mock = SymbolState(symbol="AAPL")

        live_qty, live_price = live_engine._setup_approval_state("AAPL", state_live)
        mock_qty, mock_price = mock_engine._setup_approval_state("AAPL", state_mock)

        assert live_qty == mock_qty == 0
        assert live_price == mock_price == None
        assert state_live.side == state_mock.side == None
        assert state_live.current_position == state_mock.current_position == 0

    def test_with_long_position(self, live_engine, mock_engine):
        """State setup with existing long position."""
        # Add position to portfolio
        live_engine.portfolio.positions["AAPL"] = SymbolPosition(
            qty=100, avg_price=150.0, last_price=152.0
        )
        mock_engine.portfolio.positions["AAPL"] = SymbolPosition(
            qty=100, avg_price=150.0, last_price=152.0
        )

        state_live = SymbolState(symbol="AAPL")
        state_mock = SymbolState(symbol="AAPL")

        live_qty, live_price = live_engine._setup_approval_state("AAPL", state_live)
        mock_qty, mock_price = mock_engine._setup_approval_state("AAPL", state_mock)

        assert live_qty == mock_qty == 100
        assert live_price == mock_price == 150.0
        assert state_live.side == state_mock.side == "long"
        assert state_live.current_position == state_mock.current_position == 100

    def test_with_short_position(self, live_engine, mock_engine):
        """State setup with existing short position."""
        # Add position to portfolio
        live_engine.portfolio.positions["AAPL"] = SymbolPosition(
            qty=-50, avg_price=150.0, last_price=148.0
        )
        mock_engine.portfolio.positions["AAPL"] = SymbolPosition(
            qty=-50, avg_price=150.0, last_price=148.0
        )

        state_live = SymbolState(symbol="AAPL")
        state_mock = SymbolState(symbol="AAPL")

        live_qty, live_price = live_engine._setup_approval_state("AAPL", state_live)
        mock_qty, mock_price = mock_engine._setup_approval_state("AAPL", state_mock)

        assert live_qty == mock_qty == -50
        assert live_price == mock_price == 150.0
        assert state_live.side == state_mock.side == "short"
        assert state_live.current_position == state_mock.current_position == -50


# ============================================================================
# _get_exit_quantity() Parity Tests
# ============================================================================

class TestGetExitQuantityParity:
    """Test that _get_exit_quantity produces identical results."""

    def test_full_exit_long(self, live_engine, mock_engine, shared_components):
        """Full exit from long position."""
        # Add position
        live_engine.portfolio.positions["AAPL"] = SymbolPosition(qty=100)
        mock_engine.portfolio.positions["AAPL"] = SymbolPosition(qty=100)

        mock_logic = shared_components['mock_logic']

        live_qty = live_engine._get_exit_quantity("AAPL", "exit", mock_logic)
        mock_qty = mock_engine._get_exit_quantity("AAPL", "exit", mock_logic)

        assert live_qty == mock_qty == 100

    def test_full_exit_short(self, live_engine, mock_engine, shared_components):
        """Full exit from short position."""
        # Add position
        live_engine.portfolio.positions["AAPL"] = SymbolPosition(qty=-75)
        mock_engine.portfolio.positions["AAPL"] = SymbolPosition(qty=-75)

        mock_logic = shared_components['mock_logic']

        live_qty = live_engine._get_exit_quantity("AAPL", "exit", mock_logic)
        mock_qty = mock_engine._get_exit_quantity("AAPL", "exit", mock_logic)

        assert live_qty == mock_qty == 75

    def test_reversal_quantity(self, live_engine, mock_engine, shared_components):
        """Reversal uses full position size."""
        live_engine.portfolio.positions["AAPL"] = SymbolPosition(qty=100)
        mock_engine.portfolio.positions["AAPL"] = SymbolPosition(qty=100)

        mock_logic = shared_components['mock_logic']

        live_qty = live_engine._get_exit_quantity("AAPL", "reversal", mock_logic)
        mock_qty = mock_engine._get_exit_quantity("AAPL", "reversal", mock_logic)

        assert live_qty == mock_qty == 100

    def test_partial_exit_default_fraction(self, live_engine, mock_engine, shared_components):
        """Partial exit uses default 25% fraction."""
        live_engine.portfolio.positions["AAPL"] = SymbolPosition(qty=100)
        mock_engine.portfolio.positions["AAPL"] = SymbolPosition(qty=100)

        # Create a mock logic without get_exit_quantity to force get_param path
        mock_logic = MagicMock(spec=['get_param'])
        mock_logic.get_param = MagicMock(return_value=0.25)  # 25%

        live_qty = live_engine._get_exit_quantity("AAPL", "partial_exit", mock_logic)
        mock_qty = mock_engine._get_exit_quantity("AAPL", "partial_exit", mock_logic)

        assert live_qty == mock_qty == 25  # 25% of 100

    def test_no_position_returns_zero(self, live_engine, mock_engine, shared_components):
        """Exit with no position returns 0."""
        # Ensure no position
        live_engine.portfolio.positions.pop("AAPL", None)
        mock_engine.portfolio.positions.pop("AAPL", None)

        mock_logic = shared_components['mock_logic']

        live_qty = live_engine._get_exit_quantity("AAPL", "exit", mock_logic)
        mock_qty = mock_engine._get_exit_quantity("AAPL", "exit", mock_logic)

        assert live_qty == mock_qty == 0


# ============================================================================
# Signal-to-Decision Parity Tests
# ============================================================================

class TestSignalToDecisionParity:
    """Test that same signals produce same trade decisions."""

    def test_entry_decision_parity(self, live_engine, mock_engine, shared_components):
        """Same entry signal produces same action and side."""
        state_live = SymbolState(symbol="AAPL")
        state_mock = SymbolState(symbol="AAPL")

        # Simulate what handle_signal does for action determination
        signal = 1  # Buy signal
        reason = "entry approved"

        live_action, live_side = live_engine._determine_action("AAPL", state_live, signal, reason)
        mock_action, mock_side = mock_engine._determine_action("AAPL", state_mock, signal, reason)

        assert live_action == mock_action
        assert live_side == mock_side

    def test_exit_decision_parity(self, live_engine, mock_engine, shared_components):
        """Same exit signal produces same action and side."""
        # Set up position
        state_live = SymbolState(symbol="AAPL")
        state_live.side = "long"
        state_mock = SymbolState(symbol="AAPL")
        state_mock.side = "long"

        signal = -1  # Sell signal
        reason = "exit signal"

        live_action, live_side = live_engine._determine_action("AAPL", state_live, signal, reason)
        mock_action, mock_side = mock_engine._determine_action("AAPL", state_mock, signal, reason)

        assert live_action == mock_action
        assert live_side == mock_side


# ============================================================================
# Base Class Method Inheritance Tests
# ============================================================================

class TestBaseClassInheritance:
    """Verify both engines use the same base class methods."""

    def test_both_inherit_from_base(self, live_engine, mock_engine):
        """Both engines should inherit from ExecutionEngineBase."""
        from core.base.execution_engine_base import ExecutionEngineBase

        assert isinstance(live_engine, ExecutionEngineBase)
        assert isinstance(mock_engine, ExecutionEngineBase)

    def test_determine_action_is_inherited(self, live_engine, mock_engine):
        """_determine_action should come from base class."""
        from core.base.execution_engine_base import ExecutionEngineBase

        # Both should use the base class implementation
        # We verify by checking method resolution order
        live_method = type(live_engine)._determine_action
        mock_method = type(mock_engine)._determine_action

        # Both should resolve to ExecutionEngineBase
        assert live_method is ExecutionEngineBase._determine_action
        assert mock_method is ExecutionEngineBase._determine_action

    def test_setup_approval_state_is_inherited(self, live_engine, mock_engine):
        """_setup_approval_state should come from base class."""
        from core.base.execution_engine_base import ExecutionEngineBase

        live_method = type(live_engine)._setup_approval_state
        mock_method = type(mock_engine)._setup_approval_state

        assert live_method is ExecutionEngineBase._setup_approval_state
        assert mock_method is ExecutionEngineBase._setup_approval_state

    def test_get_exit_quantity_is_inherited(self, live_engine, mock_engine):
        """_get_exit_quantity should come from base class."""
        from core.base.execution_engine_base import ExecutionEngineBase

        live_method = type(live_engine)._get_exit_quantity
        mock_method = type(mock_engine)._get_exit_quantity

        assert live_method is ExecutionEngineBase._get_exit_quantity
        assert mock_method is ExecutionEngineBase._get_exit_quantity


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
