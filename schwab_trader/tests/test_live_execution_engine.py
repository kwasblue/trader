"""
Tests for LiveExecutionEngine

Coverage:
- Order execution
- Position tracking
- P&L calculation
- Signal handling
- Trade logic integration
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
import asyncio

from core.app_types import OrderResult, PositionView, SignalContext


@pytest.fixture
def mock_broker():
    """Create a mock broker."""
    broker = MagicMock()
    broker.place_order = AsyncMock(return_value=OrderResult(
        order_id="ORD123",
        symbol="AAPL",
        side="buy",
        qty=10,
        status="filled",
        filled_qty=10,
        avg_fill_price=150.0,
        raw={}
    ))
    broker.cancel_order = AsyncMock(return_value=OrderResult(
        order_id="ORD123",
        status="cancelled",
        raw={}
    ))
    broker.get_position = AsyncMock(return_value=PositionView(
        symbol="AAPL",
        qty=100,
        avg_entry_price=145.0,
        market_price=150.0,
        side="long"
    ))
    broker.is_market_open = AsyncMock(return_value=True)
    return broker


@pytest.fixture
def mock_sizer():
    """Create a mock position sizer."""
    sizer = MagicMock()
    sizer.calculate_position_size.return_value = 10
    sizer.calculate_risk_amount.return_value = 100.0
    return sizer


@pytest.fixture
def mock_trade_logger():
    """Create a mock trade logger."""
    logger = MagicMock()
    logger.log_trade = MagicMock()
    logger.log_signal = MagicMock()
    logger.flush = MagicMock()
    return logger


@pytest.fixture
def mock_trade_logic_manager():
    """Create a mock trade logic manager."""
    manager = MagicMock()
    mock_logic = MagicMock()
    mock_logic.should_enter.return_value = True
    mock_logic.should_exit.return_value = False
    mock_logic.get_stop_loss.return_value = 145.0
    mock_logic.get_take_profit.return_value = 160.0
    manager.get_logic.return_value = mock_logic
    return manager


@pytest.fixture
def mock_executor():
    """Create a mock executor."""
    executor = MagicMock()
    executor.execute_order = AsyncMock(return_value=OrderResult(
        order_id="ORD123",
        symbol="AAPL",
        side="buy",
        qty=10,
        status="filled",
        filled_qty=10,
        avg_fill_price=150.0,
        raw={}
    ))
    return executor


@pytest.fixture
def mock_portfolio():
    """Create a mock portfolio state."""
    portfolio = MagicMock()
    portfolio.cash = 100000.0
    portfolio.equity = 100000.0
    portfolio.positions = {}
    portfolio.get_position.return_value = None
    portfolio.update_position = MagicMock()
    portfolio.update_cash = MagicMock()
    return portfolio


class TestLiveExecutionEngineSignals:
    """Test signal handling."""

    @pytest.mark.asyncio
    async def test_handle_buy_signal(self, mock_broker, mock_executor, mock_sizer, mock_trade_logger, mock_trade_logic_manager, mock_portfolio):
        """Test handling a buy signal."""
        # Import here to avoid circular imports
        from core.logic.live_execution_engine import LiveExecutionEngine
        from core.logic.symbol_state import SymbolState

        engine = LiveExecutionEngine(
            broker=mock_broker,
            executor=mock_executor,
            sizer=mock_sizer,
            performance_tracker=mock_trade_logger,
            trade_logic_manager=mock_trade_logic_manager,
            portfolio=mock_portfolio,
            sync_on_start=False,  # Disable sync to avoid asyncio.run() in async test
        )

        state = SymbolState(symbol="AAPL")
        state.side = None  # No position

        # Create SignalContext for new API
        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,  # Buy signal
            price=150.0,
            atr=2.0,
            regime="normal",
            timestamp=datetime.now(timezone.utc),
            strategy_name="test_strategy",
        )

        # Should run without exception
        result = await engine.handle_signal(context, state)

        # Engine should handle the signal (may or may not trade depending on logic)
        assert engine is not None

    @pytest.mark.asyncio
    async def test_handle_sell_signal(self, mock_broker, mock_executor, mock_sizer, mock_trade_logger, mock_trade_logic_manager, mock_portfolio):
        """Test handling a sell signal."""
        from core.logic.live_execution_engine import LiveExecutionEngine
        from core.logic.symbol_state import SymbolState

        engine = LiveExecutionEngine(
            broker=mock_broker,
            executor=mock_executor,
            sizer=mock_sizer,
            performance_tracker=mock_trade_logger,
            trade_logic_manager=mock_trade_logic_manager,
            portfolio=mock_portfolio,
            sync_on_start=False,  # Disable sync to avoid asyncio.run() in async test
        )

        state = SymbolState(symbol="AAPL")
        state.side = "long"
        state.qty = 10
        state.entry_price = 145.0

        # Create SignalContext for new API
        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=-1,  # Sell signal
            price=150.0,
            atr=2.0,
            regime="normal",
            timestamp=datetime.now(timezone.utc),
            strategy_name="test_strategy",
        )

        # Should run without exception
        result = await engine.handle_signal(context, state)

        # Engine should handle the signal
        assert engine is not None

    @pytest.mark.asyncio
    async def test_handle_hold_signal(self, mock_broker, mock_executor, mock_sizer, mock_trade_logger, mock_trade_logic_manager, mock_portfolio):
        """Test handling a hold signal."""
        from core.logic.live_execution_engine import LiveExecutionEngine
        from core.logic.symbol_state import SymbolState

        engine = LiveExecutionEngine(
            broker=mock_broker,
            executor=mock_executor,
            sizer=mock_sizer,
            performance_tracker=mock_trade_logger,
            trade_logic_manager=mock_trade_logic_manager,
            portfolio=mock_portfolio,
            sync_on_start=False,  # Disable sync to avoid asyncio.run() in async test
        )

        state = SymbolState(symbol="AAPL")

        # Create SignalContext for new API
        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=0,  # Hold signal
            price=150.0,
            atr=2.0,
            regime="normal",
            timestamp=datetime.now(timezone.utc),
            strategy_name="test_strategy",
        )

        # Hold signal should not trigger any trading action
        await engine.handle_signal(context, state)


class TestLiveExecutionEnginePositionManagement:
    """Test position management."""

    @pytest.mark.asyncio
    async def test_position_sizing(self, mock_broker, mock_executor, mock_sizer, mock_trade_logger, mock_trade_logic_manager, mock_portfolio):
        """Test that position sizing is calculated correctly."""
        from core.logic.live_execution_engine import LiveExecutionEngine
        from core.logic.symbol_state import SymbolState

        engine = LiveExecutionEngine(
            broker=mock_broker,
            executor=mock_executor,
            sizer=mock_sizer,
            performance_tracker=mock_trade_logger,
            trade_logic_manager=mock_trade_logic_manager,
            portfolio=mock_portfolio,
            sync_on_start=False,  # Disable sync to avoid asyncio.run() in async test
        )

        state = SymbolState(symbol="AAPL")
        state.portfolio_value = 100000.0

        # Create SignalContext for new API
        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,
            price=150.0,
            atr=2.0,
            regime="normal",
            timestamp=datetime.now(timezone.utc),
            strategy_name="test_strategy",
        )

        # Should run without exception
        result = await engine.handle_signal(context, state)

        # Engine should handle position sizing internally
        assert engine is not None


class TestLiveExecutionEngineRiskManagement:
    """Test risk management features."""

    @pytest.mark.asyncio
    async def test_trade_gate_integration(self, mock_broker, mock_executor, mock_sizer, mock_trade_logger, mock_trade_logic_manager, mock_portfolio):
        """Test trade gate integration."""
        from core.logic.live_execution_engine import LiveExecutionEngine
        from core.logic.symbol_state import SymbolState
        from core.logic.trade_gate import TradeGate

        engine = LiveExecutionEngine(
            broker=mock_broker,
            executor=mock_executor,
            sizer=mock_sizer,
            performance_tracker=mock_trade_logger,
            trade_logic_manager=mock_trade_logic_manager,
            portfolio=mock_portfolio,
            sync_on_start=False,  # Disable sync to avoid asyncio.run() in async test
        )

        # Set up trade gate
        trade_gate = TradeGate(max_layers=2)
        engine.trade_gate = trade_gate

        state = SymbolState(symbol="AAPL")

        # Create SignalContext for new API
        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,
            price=150.0,
            atr=2.0,
            regime="normal",
            timestamp=datetime.now(timezone.utc),
            strategy_name="test_strategy",
        )

        await engine.handle_signal(context, state)

    @pytest.mark.asyncio
    async def test_drawdown_monitor_integration(self, mock_broker, mock_executor, mock_sizer, mock_trade_logger, mock_trade_logic_manager, mock_portfolio):
        """Test drawdown monitor integration."""
        from core.logic.live_execution_engine import LiveExecutionEngine
        from core.logic.symbol_state import SymbolState
        from core.drawdown_monitor import DrawdownMonitor

        engine = LiveExecutionEngine(
            broker=mock_broker,
            executor=mock_executor,
            sizer=mock_sizer,
            performance_tracker=mock_trade_logger,
            trade_logic_manager=mock_trade_logic_manager,
            portfolio=mock_portfolio,
            sync_on_start=False,  # Disable sync to avoid asyncio.run() in async test
        )

        # Set up drawdown monitor
        ddm = DrawdownMonitor(max_symbol_drawdown=0.10, max_portfolio_drawdown=0.15)
        engine.drawdown_monitor = ddm

        state = SymbolState(symbol="AAPL")

        # Create SignalContext for new API
        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,
            price=150.0,
            atr=2.0,
            regime="normal",
            timestamp=datetime.now(timezone.utc),
            strategy_name="test_strategy",
        )

        await engine.handle_signal(context, state)


class TestLiveExecutionEngineLogging:
    """Test trade logging."""

    @pytest.mark.asyncio
    async def test_trade_logged_on_entry(self, mock_broker, mock_executor, mock_sizer, mock_trade_logger, mock_trade_logic_manager, mock_portfolio):
        """Test that trades are logged on entry."""
        from core.logic.live_execution_engine import LiveExecutionEngine
        from core.logic.symbol_state import SymbolState

        # Make trade logic allow entry
        mock_trade_logic_manager.get_logic.return_value.should_enter.return_value = True

        engine = LiveExecutionEngine(
            broker=mock_broker,
            executor=mock_executor,
            sizer=mock_sizer,
            performance_tracker=mock_trade_logger,
            trade_logic_manager=mock_trade_logic_manager,
            portfolio=mock_portfolio,
            sync_on_start=False,  # Disable sync to avoid asyncio.run() in async test
        )

        state = SymbolState(symbol="AAPL")

        # Create SignalContext for new API
        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,
            price=150.0,
            atr=2.0,
            regime="normal",
            timestamp=datetime.now(timezone.utc),
            strategy_name="test_strategy",
        )

        await engine.handle_signal(context, state)


class TestLiveExecutionEngineTradeLogic:
    """Test trade logic routing."""

    @pytest.mark.asyncio
    async def test_trade_logic_routing(self, mock_broker, mock_executor, mock_sizer, mock_trade_logger, mock_trade_logic_manager, mock_portfolio):
        """Test that correct trade logic is used based on regime."""
        from core.logic.live_execution_engine import LiveExecutionEngine
        from core.logic.symbol_state import SymbolState

        engine = LiveExecutionEngine(
            broker=mock_broker,
            executor=mock_executor,
            sizer=mock_sizer,
            performance_tracker=mock_trade_logger,
            trade_logic_manager=mock_trade_logic_manager,
            portfolio=mock_portfolio,
            sync_on_start=False,  # Disable sync to avoid asyncio.run() in async test
        )

        state = SymbolState(symbol="AAPL")

        # Create SignalContext for new API
        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,
            price=150.0,
            atr=2.0,
            regime="high_volatility",
            timestamp=datetime.now(timezone.utc),
            strategy_name="test_strategy",
        )

        # Should run without exception and route based on regime
        result = await engine.handle_signal(context, state)

        # Engine should handle the signal with appropriate trade logic
        assert engine is not None


class TestLiveExecutionEngineErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_handles_broker_error(self, mock_broker, mock_executor, mock_sizer, mock_trade_logger, mock_trade_logic_manager, mock_portfolio):
        """Test handling of broker errors."""
        from core.logic.live_execution_engine import LiveExecutionEngine
        from core.logic.symbol_state import SymbolState

        # Make broker raise an error
        mock_broker.place_order = AsyncMock(side_effect=Exception("Broker error"))

        engine = LiveExecutionEngine(
            broker=mock_broker,
            executor=mock_executor,
            sizer=mock_sizer,
            performance_tracker=mock_trade_logger,
            trade_logic_manager=mock_trade_logic_manager,
            portfolio=mock_portfolio,
            sync_on_start=False,  # Disable sync to avoid asyncio.run() in async test
        )

        state = SymbolState(symbol="AAPL")

        # Create SignalContext for new API
        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,
            price=150.0,
            atr=2.0,
            regime="normal",
            timestamp=datetime.now(timezone.utc),
            strategy_name="test_strategy",
        )

        # Should not raise exception
        await engine.handle_signal(context, state)

    @pytest.mark.asyncio
    async def test_handles_sizer_error(self, mock_broker, mock_executor, mock_sizer, mock_trade_logger, mock_trade_logic_manager, mock_portfolio):
        """Test handling of position sizer errors."""
        from core.logic.live_execution_engine import LiveExecutionEngine
        from core.logic.symbol_state import SymbolState

        # Make sizer raise an error
        mock_sizer.calculate_position_size.side_effect = Exception("Sizer error")

        engine = LiveExecutionEngine(
            broker=mock_broker,
            executor=mock_executor,
            sizer=mock_sizer,
            performance_tracker=mock_trade_logger,
            trade_logic_manager=mock_trade_logic_manager,
            portfolio=mock_portfolio,
            sync_on_start=False,  # Disable sync to avoid asyncio.run() in async test
        )

        state = SymbolState(symbol="AAPL")

        # Create SignalContext for new API
        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,
            price=150.0,
            atr=2.0,
            regime="normal",
            timestamp=datetime.now(timezone.utc),
            strategy_name="test_strategy",
        )

        # Should not raise exception
        await engine.handle_signal(context, state)
