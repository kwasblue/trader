"""
Tests for SignalContext and OrderResult enhancements.

Tests Phase 0 implementation:
- SignalContext creation and factory methods
- OrderResult commission field and filled_value property
"""

import pytest
from datetime import datetime, timezone

from core.app_types import SignalContext, OrderResult


class TestSignalContext:
    """Tests for SignalContext dataclass."""

    def test_signal_context_creation(self):
        """Test basic SignalContext creation with all fields."""
        ts = datetime.now(timezone.utc)
        context = SignalContext(
            symbol="AAPL",
            signal=1,
            price=150.25,
            atr=2.5,
            regime="trending",
            timestamp=ts,
            strategy_name="momentum",
            confidence=0.85,
            market_open=True
        )

        assert context.symbol == "AAPL"
        assert context.signal == 1
        assert context.price == 150.25
        assert context.atr == 2.5
        assert context.regime == "trending"
        assert context.timestamp == ts
        assert context.strategy_name == "momentum"
        assert context.confidence == 0.85
        assert context.market_open is True
        assert context.df is None
        assert context.metadata == {}

    def test_signal_context_defaults(self):
        """Test SignalContext default values."""
        ts = datetime.now(timezone.utc)
        context = SignalContext(
            symbol="MSFT",
            signal=0,
            price=300.0,
            atr=3.0,
            regime="normal",
            timestamp=ts
        )

        # Check defaults
        assert context.strategy_name is None
        assert context.confidence == 1.0
        assert context.df is None
        assert context.market_open is True
        assert context.metadata == {}

    def test_signal_context_from_kwargs(self):
        """Test SignalContext.from_kwargs factory method."""
        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,
            price=150.25,
            atr=2.5,
            regime="trending",
            strategy_name="momentum"
        )

        assert context.symbol == "AAPL"
        assert context.signal == 1
        assert context.price == 150.25
        assert context.atr == 2.5
        assert context.regime == "trending"
        assert context.strategy_name == "momentum"
        assert context.timestamp is not None  # Auto-generated

    def test_signal_context_from_kwargs_with_defaults(self):
        """Test from_kwargs with minimal arguments."""
        context = SignalContext.from_kwargs(
            symbol="TSLA",
            signal=-1
        )

        assert context.symbol == "TSLA"
        assert context.signal == -1
        assert context.price == 0.0
        assert context.atr == 0.0
        assert context.regime == "normal"
        assert context.confidence == 1.0
        assert context.market_open is True

    def test_signal_context_from_kwargs_with_extra_metadata(self):
        """Test from_kwargs stores extra kwargs in metadata."""
        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,
            price=150.0,
            atr=2.0,
            regime="normal",
            # Extra kwargs that should go to metadata
            bar_id=12345,
            trade_gate_passed=True,
            custom_indicator=0.75
        )

        assert context.metadata["bar_id"] == 12345
        assert context.metadata["trade_gate_passed"] is True
        assert context.metadata["custom_indicator"] == 0.75

    def test_signal_context_is_buy(self):
        """Test is_buy() helper method."""
        buy_context = SignalContext.from_kwargs(symbol="AAPL", signal=1)
        sell_context = SignalContext.from_kwargs(symbol="AAPL", signal=-1)
        hold_context = SignalContext.from_kwargs(symbol="AAPL", signal=0)

        assert buy_context.is_buy() is True
        assert sell_context.is_buy() is False
        assert hold_context.is_buy() is False

    def test_signal_context_is_sell(self):
        """Test is_sell() helper method."""
        buy_context = SignalContext.from_kwargs(symbol="AAPL", signal=1)
        sell_context = SignalContext.from_kwargs(symbol="AAPL", signal=-1)
        hold_context = SignalContext.from_kwargs(symbol="AAPL", signal=0)

        assert buy_context.is_sell() is False
        assert sell_context.is_sell() is True
        assert hold_context.is_sell() is False

    def test_signal_context_is_hold(self):
        """Test is_hold() helper method."""
        buy_context = SignalContext.from_kwargs(symbol="AAPL", signal=1)
        sell_context = SignalContext.from_kwargs(symbol="AAPL", signal=-1)
        hold_context = SignalContext.from_kwargs(symbol="AAPL", signal=0)

        assert buy_context.is_hold() is False
        assert sell_context.is_hold() is False
        assert hold_context.is_hold() is True


class TestOrderResultEnhancements:
    """Tests for OrderResult enhancements."""

    def test_order_result_commission_field(self):
        """Test commission field on OrderResult."""
        result = OrderResult(
            order_id="ORD123",
            symbol="AAPL",
            side="buy",
            filled_qty=100,
            avg_fill_price=150.0,
            commission=1.50,
            status="filled"
        )

        assert result.commission == 1.50

    def test_order_result_commission_default(self):
        """Test commission defaults to None."""
        result = OrderResult(
            order_id="ORD123",
            symbol="AAPL",
            filled_qty=100,
            avg_fill_price=150.0
        )

        assert result.commission is None

    def test_order_result_filled_value(self):
        """Test filled_value property calculation."""
        result = OrderResult(
            order_id="ORD123",
            symbol="AAPL",
            filled_qty=100,
            avg_fill_price=150.25,
            status="filled"
        )

        assert result.filled_value == 100 * 150.25
        assert result.filled_value == 15025.0

    def test_order_result_filled_value_partial_fill(self):
        """Test filled_value with partial fill."""
        result = OrderResult(
            order_id="ORD123",
            symbol="AAPL",
            qty=100,
            filled_qty=50,
            avg_fill_price=150.0,
            status="partial_fill"
        )

        assert result.filled_value == 50 * 150.0
        assert result.filled_value == 7500.0

    def test_order_result_filled_value_no_fill(self):
        """Test filled_value when no fill occurred."""
        result = OrderResult(
            order_id="ORD123",
            symbol="AAPL",
            qty=100,
            status="pending"
        )

        # No filled_qty or avg_fill_price
        assert result.filled_value == 0.0

    def test_order_result_filled_value_legacy_avg_price(self):
        """Test filled_value falls back to legacy avg_price field."""
        result = OrderResult(
            order_id="ORD123",
            symbol="AAPL",
            filled_qty=100,
            avg_price=150.0,  # Legacy field
            status="filled"
        )

        # Should use avg_price when avg_fill_price is None
        assert result.filled_value == 15000.0

    def test_order_result_to_dict_includes_commission(self):
        """Test to_dict includes commission field."""
        result = OrderResult(
            order_id="ORD123",
            symbol="AAPL",
            filled_qty=100,
            avg_fill_price=150.0,
            commission=2.50
        )

        d = result.to_dict()
        assert "commission" in d
        assert d["commission"] == 2.50

    def test_order_result_bool_behavior_unchanged(self):
        """Test that bool behavior is unchanged."""
        # Success based on explicit success field
        result_success = OrderResult(success=True)
        result_fail = OrderResult(success=False)

        assert bool(result_success) is True
        assert bool(result_fail) is False

        # Success inferred from status
        result_filled = OrderResult(status="filled")
        result_rejected = OrderResult(status="rejected")

        assert bool(result_filled) is True
        assert bool(result_rejected) is False


# ============================================================================
# Phase 3 Tests: Signal Context Integration
# ============================================================================


class TestHandleSignalContextIntegration:
    """Tests for handle_signal_context in execution engines."""

    @pytest.fixture
    def mock_engine_components(self):
        """Create minimal components for mock engine testing."""
        from unittest.mock import Mock, AsyncMock, MagicMock
        from core.logic.portfolio_state import PortfolioState

        # Create mocks
        broker = MagicMock()
        broker.get_position = AsyncMock(return_value=None)
        broker.is_market_open = AsyncMock(return_value=True)

        executor = Mock()
        sizer = Mock()
        sizer.calculate_position_size = Mock(return_value=10)

        performance_tracker = Mock()
        performance_tracker.log_trade = Mock()
        performance_tracker.log_error = Mock()

        trade_logic_manager = Mock()
        trade_logic = Mock()
        trade_logic.should_trade = Mock(return_value=(True, "approved"))
        trade_logic.get_param = Mock(return_value=0.25)
        # Add sl_mults dict for position sizing
        trade_logic.sl_mults = {"normal": 1.5, "trending": 2.0}
        trade_logic_manager.get = Mock(return_value=trade_logic)

        portfolio = PortfolioState(cash=100000, total_value=100000)

        return {
            'broker': broker,
            'executor': executor,
            'sizer': sizer,
            'performance_tracker': performance_tracker,
            'trade_logic_manager': trade_logic_manager,
            'trade_logic': trade_logic,
            'portfolio': portfolio
        }

    @pytest.mark.asyncio
    async def test_handle_signal_context_filters_low_confidence(self, mock_engine_components):
        """Test that low confidence signals are filtered."""
        from core.logic.mock_execution_engine import MockExecutionEngine

        engine = MockExecutionEngine(
            broker=mock_engine_components['broker'],
            executor=mock_engine_components['executor'],
            sizer=mock_engine_components['sizer'],
            performance_tracker=mock_engine_components['performance_tracker'],
            trade_logic_manager=mock_engine_components['trade_logic_manager'],
            portfolio=mock_engine_components['portfolio']
        )

        # Set minimum confidence threshold
        engine.min_signal_confidence = 0.5

        # Create low confidence signal
        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,
            price=150.0,
            atr=2.0,
            regime="normal",
            confidence=0.3  # Below threshold
        )

        result = await engine.handle_signal_context(context)

        # Should be filtered
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_signal_context_accepts_high_confidence(self, mock_engine_components):
        """Test that high confidence signals are accepted."""
        from core.logic.mock_execution_engine import MockExecutionEngine

        engine = MockExecutionEngine(
            broker=mock_engine_components['broker'],
            executor=mock_engine_components['executor'],
            sizer=mock_engine_components['sizer'],
            performance_tracker=mock_engine_components['performance_tracker'],
            trade_logic_manager=mock_engine_components['trade_logic_manager'],
            portfolio=mock_engine_components['portfolio']
        )

        # Set minimum confidence threshold
        engine.min_signal_confidence = 0.5

        # Create high confidence signal
        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,
            price=150.0,
            atr=2.0,
            regime="normal",
            confidence=0.8  # Above threshold
        )

        result = await engine.handle_signal_context(context)

        # Should execute (returns OrderResult)
        assert result is not None

    @pytest.mark.asyncio
    async def test_handle_signal_context_skips_hold_signals(self, mock_engine_components):
        """Test that hold signals (signal=0) are skipped."""
        from core.logic.mock_execution_engine import MockExecutionEngine

        engine = MockExecutionEngine(
            broker=mock_engine_components['broker'],
            executor=mock_engine_components['executor'],
            sizer=mock_engine_components['sizer'],
            performance_tracker=mock_engine_components['performance_tracker'],
            trade_logic_manager=mock_engine_components['trade_logic_manager'],
            portfolio=mock_engine_components['portfolio']
        )

        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=0,  # Hold
            price=150.0,
            atr=2.0,
            regime="normal"
        )

        result = await engine.handle_signal_context(context)

        # Should skip
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_signal_with_context_and_state(self, mock_engine_components):
        """Test that handle_signal works with context and state arguments."""
        from core.logic.mock_execution_engine import MockExecutionEngine
        from core.logic.symbol_state import SymbolState
        from core.app_types import SignalContext
        from datetime import datetime, timezone

        engine = MockExecutionEngine(
            broker=mock_engine_components['broker'],
            executor=mock_engine_components['executor'],
            sizer=mock_engine_components['sizer'],
            performance_tracker=mock_engine_components['performance_tracker'],
            trade_logic_manager=mock_engine_components['trade_logic_manager'],
            portfolio=mock_engine_components['portfolio']
        )

        state = SymbolState(symbol="AAPL")

        # Create context and call handle_signal directly
        context = SignalContext(
            symbol="AAPL",
            signal=1,
            price=150.0,
            atr=2.0,
            regime="normal",
            timestamp=datetime.now(timezone.utc),
            strategy_name="test_strategy"
        )
        result = await engine.handle_signal(context, state)

        # Should work and return result
        assert result is not None
        assert result.symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_handle_signal_context_uses_state_from_metadata(self, mock_engine_components):
        """Test that state is extracted from metadata if provided."""
        from core.logic.mock_execution_engine import MockExecutionEngine
        from core.logic.symbol_state import SymbolState

        engine = MockExecutionEngine(
            broker=mock_engine_components['broker'],
            executor=mock_engine_components['executor'],
            sizer=mock_engine_components['sizer'],
            performance_tracker=mock_engine_components['performance_tracker'],
            trade_logic_manager=mock_engine_components['trade_logic_manager'],
            portfolio=mock_engine_components['portfolio']
        )

        # Create state with specific values
        state = SymbolState(symbol="AAPL")
        state.strategy_name = "original_strategy"

        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,
            price=150.0,
            atr=2.0,
            regime="normal",
            strategy_name="new_strategy",
            state=state  # Pass state in metadata
        )

        result = await engine.handle_signal_context(context)

        # State should be updated with new strategy name
        assert state.strategy_name == "new_strategy"

    @pytest.mark.asyncio
    async def test_handle_signal_context_creates_state_if_missing(self, mock_engine_components):
        """Test that state is created if not in metadata."""
        from core.logic.mock_execution_engine import MockExecutionEngine

        engine = MockExecutionEngine(
            broker=mock_engine_components['broker'],
            executor=mock_engine_components['executor'],
            sizer=mock_engine_components['sizer'],
            performance_tracker=mock_engine_components['performance_tracker'],
            trade_logic_manager=mock_engine_components['trade_logic_manager'],
            portfolio=mock_engine_components['portfolio']
        )

        # Don't provide state
        context = SignalContext.from_kwargs(
            symbol="TSLA",  # New symbol
            signal=1,
            price=200.0,
            atr=3.0,
            regime="normal"
        )

        # Before call, no state exists
        assert "TSLA" not in engine.symbol_states

        result = await engine.handle_signal_context(context)

        # State should be created
        assert "TSLA" in engine.symbol_states

    @pytest.mark.asyncio
    async def test_handle_signal_context_passes_market_open(self, mock_engine_components):
        """Test that market_open from context is passed to trade logic."""
        from unittest.mock import call
        from core.logic.mock_execution_engine import MockExecutionEngine

        engine = MockExecutionEngine(
            broker=mock_engine_components['broker'],
            executor=mock_engine_components['executor'],
            sizer=mock_engine_components['sizer'],
            performance_tracker=mock_engine_components['performance_tracker'],
            trade_logic_manager=mock_engine_components['trade_logic_manager'],
            portfolio=mock_engine_components['portfolio']
        )

        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,
            price=150.0,
            atr=2.0,
            regime="normal",
            market_open=False  # Market closed
        )

        await engine.handle_signal_context(context)

        # Check that should_trade was called with context containing market_open=False
        trade_logic = mock_engine_components['trade_logic']
        call_kwargs = trade_logic.should_trade.call_args.kwargs
        # New API passes context object, check market_open from context
        passed_context = call_kwargs.get('context')
        assert passed_context is not None
        assert passed_context.market_open is False


class TestSignalContextAbstractMethod:
    """Test that handle_signal_context is properly defined in base class."""

    def test_handle_signal_context_is_abstract(self):
        """Test that handle_signal_context is abstract in base class."""
        from core.base.execution_engine_base import ExecutionEngineBase
        import inspect

        # Check that handle_signal_context is abstract
        assert hasattr(ExecutionEngineBase, 'handle_signal_context')
        method = getattr(ExecutionEngineBase, 'handle_signal_context')
        assert getattr(method, '__isabstractmethod__', False)

    def test_handle_signal_context_signature(self):
        """Test that handle_signal_context has correct signature."""
        from core.base.execution_engine_base import ExecutionEngineBase
        import inspect

        sig = inspect.signature(ExecutionEngineBase.handle_signal_context)
        params = list(sig.parameters.keys())

        # Should have self and context
        assert 'self' in params
        assert 'context' in params

    def test_live_engine_implements_handle_signal_context(self):
        """Test that LiveExecutionEngine implements handle_signal_context."""
        from core.logic.live_execution_engine import LiveExecutionEngine
        import inspect

        assert hasattr(LiveExecutionEngine, 'handle_signal_context')
        # Check it's not abstract (overridden)
        method = getattr(LiveExecutionEngine, 'handle_signal_context')
        assert not getattr(method, '__isabstractmethod__', False)

    def test_mock_engine_implements_handle_signal_context(self):
        """Test that MockExecutionEngine implements handle_signal_context."""
        from core.logic.mock_execution_engine import MockExecutionEngine
        import inspect

        assert hasattr(MockExecutionEngine, 'handle_signal_context')
        # Check it's not abstract (overridden)
        method = getattr(MockExecutionEngine, 'handle_signal_context')
        assert not getattr(method, '__isabstractmethod__', False)
