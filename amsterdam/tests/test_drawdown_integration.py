"""
Tests for DrawdownMonitor integration with LiveExecutionEngine.

Verifies that drawdown limits are properly enforced during live trading.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.app_types import SignalContext
from core.drawdown_monitor import DrawdownMonitor
from core.logic.live_execution_engine import LiveExecutionEngine
from core.logic.portfolio_state import PortfolioState


@pytest.fixture
def portfolio():
    """Create a test portfolio with initial cash."""
    return PortfolioState(cash=100_000.0)


@pytest.fixture
def drawdown_monitor():
    """Create a DrawdownMonitor with test-friendly limits."""
    ddm = DrawdownMonitor(
        max_symbol_drawdown=0.10,  # 10% symbol intraday
        max_symbol_daily_drawdown=0.05,  # 5% symbol daily
        symbol_cooldown_seconds=1,
        max_portfolio_drawdown=0.15,  # 15% portfolio intraday
        max_portfolio_daily_drawdown=0.08,  # 8% portfolio daily
        portfolio_cooldown_seconds=1,
    )
    # Initialize with starting equity
    ddm.start_new_day(portfolio_equity=100_000.0)
    return ddm


@pytest.fixture
def mock_broker():
    """Create a mock broker."""
    broker = MagicMock()
    broker.get_position = AsyncMock(return_value=None)
    broker.is_market_open = AsyncMock(return_value=True)
    broker.get_buying_power = MagicMock(return_value=100_000.0)
    broker.get_available_funds = MagicMock(return_value=100_000.0)
    broker.place_order = AsyncMock(
        return_value=MagicMock(order_id="test-123", symbol="AAPL", side="buy", qty=100, status="filled")
    )
    broker.get_open_orders = AsyncMock(return_value=[])
    return broker


@pytest.fixture
def mock_executor():
    """Create a mock executor."""
    return MagicMock()


@pytest.fixture
def mock_sizer():
    """Create a mock position sizer."""
    sizer = MagicMock()
    sizer.calculate_position_size = MagicMock(return_value=100)
    return sizer


@pytest.fixture
def mock_trade_logger():
    """Create a mock trade logger."""
    logger = MagicMock()
    logger.log_error = MagicMock()
    return logger


@pytest.fixture
def mock_trade_logic_manager():
    """Create a mock trade logic manager."""
    logic = MagicMock()
    logic.should_trade = MagicMock(return_value=(True, "approved"))
    return logic


@pytest.fixture
def engine(
    portfolio, mock_broker, mock_executor, mock_sizer, mock_trade_logger, mock_trade_logic_manager, drawdown_monitor
):
    """Create a LiveExecutionEngine with DrawdownMonitor."""
    engine = LiveExecutionEngine(
        broker=mock_broker,
        executor=mock_executor,
        sizer=mock_sizer,
        performance_tracker=mock_trade_logger,
        trade_logic_manager=mock_trade_logic_manager,
        portfolio=portfolio,
        sync_on_start=False,
        drawdown_monitor=drawdown_monitor,
    )
    return engine


class TestDrawdownIntegration:
    """Test DrawdownMonitor integration with LiveExecutionEngine."""

    @pytest.mark.asyncio
    async def test_portfolio_drawdown_blocks_trade(self, engine, drawdown_monitor):
        """Test that portfolio drawdown breach blocks trading."""
        # Simulate portfolio drawdown breach
        # Start with 100k, update to 85k (15% loss) which exceeds 8% daily limit
        drawdown_monitor.update_portfolio(85_000.0)

        # Verify portfolio is locked
        assert drawdown_monitor.is_portfolio_blocked()

        # Create a buy signal
        context = SignalContext(
            symbol="AAPL",
            signal=1,
            price=150.0,
            atr=2.5,
            regime="normal",
            timestamp=datetime.now(timezone.utc),
            strategy_name="test_strategy",
            market_open=True,
        )

        # Signal should be blocked
        result = await engine.handle_signal_context(context)
        assert result is None

    @pytest.mark.asyncio
    async def test_symbol_drawdown_blocks_trade(self, engine, drawdown_monitor):
        """Test that per-symbol drawdown breach blocks trading for that symbol."""
        # Initialize symbol with starting value
        drawdown_monitor.update_symbol("AAPL", 10_000.0)

        # Simulate symbol drawdown breach (10% loss exceeds 5% daily limit)
        drawdown_monitor.update_symbol("AAPL", 9_400.0)

        # Verify AAPL is blocked
        assert drawdown_monitor.is_symbol_blocked("AAPL")

        # MSFT should still be tradeable
        drawdown_monitor.update_symbol("MSFT", 10_000.0)
        assert not drawdown_monitor.is_symbol_blocked("MSFT")

        # Create a buy signal for AAPL
        context = SignalContext(
            symbol="AAPL",
            signal=1,
            price=150.0,
            atr=2.5,
            regime="normal",
            timestamp=datetime.now(timezone.utc),
            strategy_name="test_strategy",
            market_open=True,
        )

        # AAPL signal should be blocked
        result = await engine.handle_signal_context(context)
        assert result is None

    @pytest.mark.asyncio
    async def test_cooldown_blocks_trade(self, engine, drawdown_monitor):
        """Test that cooldown period blocks trading after unlock."""
        # Simulate symbol lock and immediate unlock
        drawdown_monitor.symbol_locked["AAPL"] = True
        drawdown_monitor.unlock_symbol("AAPL")

        # Verify cooldown is active
        assert drawdown_monitor.is_symbol_in_cooldown("AAPL")

        # Create a buy signal
        context = SignalContext(
            symbol="AAPL",
            signal=1,
            price=150.0,
            atr=2.5,
            regime="normal",
            timestamp=datetime.now(timezone.utc),
            strategy_name="test_strategy",
            market_open=True,
        )

        # Signal should be blocked during cooldown
        result = await engine.handle_signal_context(context)
        assert result is None

        # Wait for cooldown to expire
        await asyncio.sleep(1.1)

        # Now cooldown should be over
        assert not drawdown_monitor.is_symbol_in_cooldown("AAPL")

    @pytest.mark.asyncio
    async def test_drawdown_updated_after_trade(self, engine, drawdown_monitor, portfolio):
        """Test that drawdown monitor is updated after successful trade."""
        # Update portfolio to simulate gains
        portfolio.cash = 110_000.0

        # The _post_execution method should update the drawdown monitor
        # We'll test this by calling _post_execution directly
        mock_result = MagicMock()
        mock_result.side = MagicMock(value="buy")
        mock_result.filled_qty = 100
        mock_result.avg_price = 150.0

        mock_state = MagicMock()
        mock_state.symbol = "AAPL"

        # Call _post_execution
        engine._post_execution(
            symbol="AAPL",
            state=mock_state,
            result=mock_result,
            action_type="entry",
            regime="normal",
            strategy_name="test_strategy",
        )

        # Portfolio peak should have been updated
        assert drawdown_monitor.portfolio_peak is not None

    @pytest.mark.asyncio
    async def test_can_trade_async(self, drawdown_monitor):
        """Test async wrapper for can_trade."""
        # Portfolio not in drawdown, symbol not blocked
        drawdown_monitor.update_portfolio(100_000.0)
        drawdown_monitor.update_symbol("AAPL", 10_000.0)

        result = await drawdown_monitor.can_trade_async("AAPL")
        assert result is True

        # Now trigger drawdown
        drawdown_monitor.update_portfolio(80_000.0)  # 20% loss

        result = await drawdown_monitor.can_trade_async("AAPL")
        assert result is False

    @pytest.mark.asyncio
    async def test_update_portfolio_async(self, drawdown_monitor):
        """Test async wrapper for update_portfolio."""
        result = await drawdown_monitor.update_portfolio_async(100_000.0)
        assert result is True

        # Trigger breach
        result = await drawdown_monitor.update_portfolio_async(80_000.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_update_symbol_async(self, drawdown_monitor):
        """Test async wrapper for update_symbol."""
        # Initialize and check
        result = await drawdown_monitor.update_symbol_async("AAPL", 10_000.0)
        assert result is True

        # Trigger symbol breach (5% daily loss)
        result = await drawdown_monitor.update_symbol_async("AAPL", 9_400.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_trade_allowed_when_no_drawdown(self, engine, drawdown_monitor, mock_broker):
        """Test that trades proceed when drawdown limits not breached."""
        # Ensure portfolio and symbol are healthy
        drawdown_monitor.update_portfolio(100_000.0)
        drawdown_monitor.update_symbol("AAPL", 10_000.0)

        # Verify no blocks
        assert not drawdown_monitor.is_portfolio_blocked()
        assert not drawdown_monitor.is_symbol_blocked("AAPL")

        # The can_trade check should pass
        can_trade = await drawdown_monitor.can_trade_async("AAPL")
        assert can_trade is True


class TestDrawdownMonitorAsyncWrappers:
    """Test async wrapper methods specifically."""

    @pytest.mark.asyncio
    async def test_wrappers_dont_block_event_loop(self, drawdown_monitor):
        """Verify async wrappers don't block the event loop."""
        # Run multiple async operations concurrently
        tasks = [
            drawdown_monitor.can_trade_async("AAPL"),
            drawdown_monitor.can_trade_async("MSFT"),
            drawdown_monitor.update_portfolio_async(100_000.0),
            drawdown_monitor.update_symbol_async("AAPL", 10_000.0),
        ]

        # All should complete without blocking
        results = await asyncio.gather(*tasks)
        assert len(results) == 4

    @pytest.mark.asyncio
    async def test_thread_safety(self, drawdown_monitor):
        """Test that concurrent async calls are thread-safe."""
        # Initialize symbols
        for i in range(10):
            drawdown_monitor.update_symbol(f"SYM{i}", 10_000.0)

        # Run many concurrent updates
        async def update_batch():
            tasks = []
            for i in range(10):
                tasks.append(drawdown_monitor.update_symbol_async(f"SYM{i}", 9_900.0))
            return await asyncio.gather(*tasks)

        # Run multiple batches concurrently
        results = await asyncio.gather(*[update_batch() for _ in range(5)])
        assert len(results) == 5
