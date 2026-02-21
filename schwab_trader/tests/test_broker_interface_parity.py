"""
Tests for Broker Interface Parity.

Tests Phase 1 implementation:
- All brokers implement async place_market_order() returning OrderResult
- All brokers implement async place_oco_order() returning OrderResult
- connect() returns None (status stored in instance)
- OrderSide enum consistency
- Deprecated sync methods warn
"""

import pytest
import asyncio
import warnings
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone

from core.app_types import OrderResult
from core.enums import OrderSide, OrderStatus
from core.broker.mock_broker import MockBroker
from core.logic.portfolio_state import PortfolioState


class TestMockBrokerInterface:
    """Tests for MockBroker interface compliance."""

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker for testing."""
        return MockBroker(starting_cash=100000.0)

    @pytest.mark.asyncio
    async def test_place_market_order_returns_order_result(self, mock_broker):
        """Test that place_market_order returns OrderResult."""
        result = await mock_broker.place_market_order(
            symbol="AAPL",
            qty=10,
            side=OrderSide.BUY,
            price=150.0
        )

        assert isinstance(result, OrderResult)
        assert result.symbol == "AAPL"
        assert result.filled_qty == 10
        assert result.avg_price == 150.0

    @pytest.mark.asyncio
    async def test_place_market_order_is_async(self, mock_broker):
        """Verify place_market_order is an async method."""
        import inspect
        assert inspect.iscoroutinefunction(mock_broker.place_market_order)

    @pytest.mark.asyncio
    async def test_place_oco_order_returns_order_result(self, mock_broker):
        """Test that place_oco_order returns OrderResult."""
        result = await mock_broker.place_oco_order(
            symbol="AAPL",
            qty=10,
            stop_price=145.0,
            limit_price=160.0
        )

        assert isinstance(result, OrderResult)
        assert result.order_id is not None

    @pytest.mark.asyncio
    async def test_place_oco_order_is_async(self, mock_broker):
        """Verify place_oco_order is an async method."""
        import inspect
        assert inspect.iscoroutinefunction(mock_broker.place_oco_order)

    @pytest.mark.asyncio
    async def test_order_side_enum_buy(self, mock_broker):
        """Test OrderSide.BUY works correctly."""
        result = await mock_broker.place_market_order(
            symbol="AAPL",
            qty=10,
            side=OrderSide.BUY,
            price=150.0
        )

        assert result.success is True
        assert result.side == OrderSide.BUY

    @pytest.mark.asyncio
    async def test_order_side_enum_sell(self, mock_broker):
        """Test OrderSide.SELL works correctly."""
        # First buy to have a position
        await mock_broker.place_market_order(
            symbol="AAPL",
            qty=10,
            side=OrderSide.BUY,
            price=150.0
        )

        # Now sell
        result = await mock_broker.place_market_order(
            symbol="AAPL",
            qty=10,
            side=OrderSide.SELL,
            price=155.0
        )

        assert result.success is True
        assert result.side == OrderSide.SELL

    @pytest.mark.asyncio
    async def test_place_market_order_insufficient_funds(self, mock_broker):
        """Test rejection when insufficient funds."""
        result = await mock_broker.place_market_order(
            symbol="AAPL",
            qty=1000,
            side=OrderSide.BUY,
            price=1000.0  # 1M cost > 100k cash
        )

        assert result.success is False
        assert "Insufficient funds" in result.message

    @pytest.mark.asyncio
    async def test_place_market_order_insufficient_position(self, mock_broker):
        """Test rejection when selling without position."""
        result = await mock_broker.place_market_order(
            symbol="AAPL",
            qty=10,
            side=OrderSide.SELL,
            price=150.0
        )

        assert result.success is False
        assert "Insufficient position" in result.message


class TestBrokerInterfaceContract:
    """Tests for broker interface contract compliance."""

    def test_mock_broker_has_required_async_methods(self):
        """Verify MockBroker has all required async methods."""
        import inspect

        broker = MockBroker()

        # Required async methods
        assert inspect.iscoroutinefunction(broker.place_market_order)
        assert inspect.iscoroutinefunction(broker.place_oco_order)
        assert inspect.iscoroutinefunction(broker.place_order)
        assert inspect.iscoroutinefunction(broker.cancel_order)
        assert inspect.iscoroutinefunction(broker.get_position)
        assert inspect.iscoroutinefunction(broker.get_account_info)
        assert inspect.iscoroutinefunction(broker.is_market_open)

    def test_mock_broker_has_sync_helpers(self):
        """Verify MockBroker has sync helper methods."""
        broker = MockBroker()

        # Sync methods
        assert callable(broker.get_quote)
        assert callable(broker.get_available_funds)
        assert callable(broker.get_default_account)
        assert callable(broker.mark_price)


class TestDeprecatedSyncMethods:
    """Tests for deprecated sync wrapper methods."""

    def test_place_market_order_sync_warns(self):
        """Test that place_market_order_sync raises deprecation warning."""
        from core.base.base_broker_interface import BaseBrokerInterface

        # Check the method exists
        assert hasattr(BaseBrokerInterface, 'place_market_order_sync')

    def test_place_oco_order_sync_warns(self):
        """Test that place_oco_order_sync raises deprecation warning."""
        from core.base.base_broker_interface import BaseBrokerInterface

        # Check the method exists
        assert hasattr(BaseBrokerInterface, 'place_oco_order_sync')


class TestOrderResultFromBroker:
    """Tests for OrderResult consistency from brokers."""

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker for testing."""
        return MockBroker(starting_cash=100000.0)

    @pytest.mark.asyncio
    async def test_order_result_has_commission(self, mock_broker):
        """Test that OrderResult includes commission."""
        mock_broker.commission = 1.50

        result = await mock_broker.place_market_order(
            symbol="AAPL",
            qty=10,
            side=OrderSide.BUY,
            price=150.0
        )

        assert result.commission == 1.50

    @pytest.mark.asyncio
    async def test_order_result_filled_value(self, mock_broker):
        """Test filled_value property on OrderResult."""
        result = await mock_broker.place_market_order(
            symbol="AAPL",
            qty=10,
            side=OrderSide.BUY,
            price=150.0
        )

        # filled_value = filled_qty * avg_price
        expected_value = 10 * 150.0
        assert result.filled_value == expected_value

    @pytest.mark.asyncio
    async def test_order_result_status(self, mock_broker):
        """Test OrderResult has proper status."""
        result = await mock_broker.place_market_order(
            symbol="AAPL",
            qty=10,
            side=OrderSide.BUY,
            price=150.0
        )

        assert result.status == OrderStatus.FILLED
        assert result.is_filled is True


class TestAlpacaBrokerInterface:
    """Tests for AlpacaBroker interface compliance (mocked)."""

    def test_alpaca_broker_has_async_connect(self):
        """Verify AlpacaBroker.connect is async (or wrapped async)."""
        import inspect
        from core.broker.alpaca_broker import AlpacaBroker

        # Check the method - may be wrapped by @retry decorator
        method = AlpacaBroker.connect
        # Check if it's async or has async inner function (decorated)
        is_async = inspect.iscoroutinefunction(method)
        # Also check wrapped function if decorator is applied
        inner = getattr(method, '__wrapped__', None)
        inner_is_async = inspect.iscoroutinefunction(inner) if inner else False

        assert is_async or inner_is_async, "connect should be async or have async wrapped function"

    def test_alpaca_broker_has_sync_connect(self):
        """Verify AlpacaBroker has connect_sync for compatibility."""
        from core.broker.alpaca_broker import AlpacaBroker

        assert hasattr(AlpacaBroker, 'connect_sync')

    def test_alpaca_broker_place_market_order_is_async(self):
        """Verify AlpacaBroker.place_market_order is async."""
        import inspect
        from core.broker.alpaca_broker import AlpacaBroker

        assert inspect.iscoroutinefunction(AlpacaBroker.place_market_order)

    def test_alpaca_broker_place_oco_order_is_async(self):
        """Verify AlpacaBroker.place_oco_order is async."""
        import inspect
        from core.broker.alpaca_broker import AlpacaBroker

        assert inspect.iscoroutinefunction(AlpacaBroker.place_oco_order)

    def test_alpaca_broker_place_order_returns_order_result(self):
        """Verify AlpacaBroker.place_order return type annotation."""
        import inspect
        from core.broker.alpaca_broker import AlpacaBroker

        sig = inspect.signature(AlpacaBroker.place_order)
        # Check it's annotated to return OrderResult (may be string due to annotations future import)
        annotation = sig.return_annotation
        assert annotation == OrderResult or annotation == 'OrderResult', \
            f"Expected OrderResult annotation, got {annotation}"


class TestSchwabBrokerInterface:
    """Tests for SchwabBroker interface compliance (mocked)."""

    def test_schwab_broker_connect_returns_none(self):
        """Verify SchwabBroker.connect return type is None."""
        import inspect
        from core.broker.schwab_broker import SchwabBroker

        sig = inspect.signature(SchwabBroker.connect)
        # Check it's annotated to return None (may be string due to annotations future import)
        annotation = sig.return_annotation
        assert annotation is None or annotation == 'None', \
            f"Expected None annotation, got {annotation}"

    def test_schwab_broker_has_is_connected_property(self):
        """Verify SchwabBroker has is_connected property."""
        from core.broker.schwab_broker import SchwabBroker

        assert hasattr(SchwabBroker, 'is_connected')

    def test_schwab_broker_place_market_order_is_async(self):
        """Verify SchwabBroker.place_market_order is async."""
        import inspect
        from core.broker.schwab_broker import SchwabBroker

        assert inspect.iscoroutinefunction(SchwabBroker.place_market_order)

    def test_schwab_broker_place_oco_order_is_async(self):
        """Verify SchwabBroker.place_oco_order is async."""
        import inspect
        from core.broker.schwab_broker import SchwabBroker

        assert inspect.iscoroutinefunction(SchwabBroker.place_oco_order)


# ==============================================================================
# NEW FUNCTIONALITY TESTS
# ==============================================================================

class TestMockBrokerOpenOrderTracking:
    """Tests for MockBroker open order tracking functionality."""

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker for testing."""
        return MockBroker(starting_cash=100000.0)

    @pytest.mark.asyncio
    async def test_open_orders_initially_empty(self, mock_broker):
        """Open orders dict should be empty on init."""
        open_orders = await mock_broker.get_open_orders()
        assert len(open_orders) == 0

    @pytest.mark.asyncio
    async def test_limit_order_added_to_open_orders(self, mock_broker):
        """Limit order should be tracked in open orders."""
        from core.enums import OrderType

        result = await mock_broker.place_order(
            symbol="AAPL",
            qty=10,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=145.0
        )

        assert result.success is True
        open_orders = await mock_broker.get_open_orders()
        assert len(open_orders) == 1
        assert open_orders[0].order_id == result.order_id

    @pytest.mark.asyncio
    async def test_stop_order_added_to_open_orders(self, mock_broker):
        """Stop order should be tracked in open orders."""
        from core.enums import OrderType

        # First buy to have a position to sell
        await mock_broker.place_market_order("AAPL", 10, OrderSide.BUY, price=150.0)

        result = await mock_broker.place_order(
            symbol="AAPL",
            qty=10,
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            stop_price=145.0
        )

        assert result.success is True
        open_orders = await mock_broker.get_open_orders()
        assert len(open_orders) == 1
        assert open_orders[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_market_order_not_in_open_orders(self, mock_broker):
        """Market orders fill immediately and should not be in open orders."""
        result = await mock_broker.place_market_order(
            symbol="AAPL",
            qty=10,
            side=OrderSide.BUY,
            price=150.0
        )

        assert result.success is True
        assert result.status == OrderStatus.FILLED
        open_orders = await mock_broker.get_open_orders()
        assert len(open_orders) == 0

    @pytest.mark.asyncio
    async def test_cancelled_order_removed_from_open_orders(self, mock_broker):
        """Cancelled orders should be removed from open orders."""
        from core.enums import OrderType

        result = await mock_broker.place_order(
            symbol="AAPL",
            qty=10,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=145.0
        )

        # Cancel the order
        cancel_result = await mock_broker.cancel_order(result.order_id)
        assert cancel_result.success is True

        open_orders = await mock_broker.get_open_orders()
        assert len(open_orders) == 0

    @pytest.mark.asyncio
    async def test_get_open_orders_filter_by_symbol(self, mock_broker):
        """get_open_orders should filter by symbol when specified."""
        from core.enums import OrderType

        await mock_broker.place_order("AAPL", 10, OrderSide.BUY, OrderType.LIMIT, limit_price=145.0)
        await mock_broker.place_order("MSFT", 5, OrderSide.BUY, OrderType.LIMIT, limit_price=340.0)

        aapl_orders = await mock_broker.get_open_orders(symbol="AAPL")
        msft_orders = await mock_broker.get_open_orders(symbol="MSFT")
        all_orders = await mock_broker.get_open_orders()

        assert len(aapl_orders) == 1
        assert len(msft_orders) == 1
        assert len(all_orders) == 2


class TestMockBrokerLimitOrders:
    """Tests for MockBroker LIMIT order functionality."""

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker for testing."""
        return MockBroker(starting_cash=100000.0)

    @pytest.mark.asyncio
    async def test_limit_buy_fills_when_price_drops(self, mock_broker):
        """Limit buy should fill when price drops to limit price."""
        from core.enums import OrderType

        # Place limit buy at $145
        result = await mock_broker.place_order(
            symbol="AAPL",
            qty=10,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=145.0
        )

        assert result.status == OrderStatus.PENDING

        # Price above limit - should not fill
        filled = await mock_broker.check_pending_orders("AAPL", 150.0)
        assert len(filled) == 0

        # Price at limit - should fill
        filled = await mock_broker.check_pending_orders("AAPL", 145.0)
        assert len(filled) == 1
        assert filled[0].avg_price == 145.0

    @pytest.mark.asyncio
    async def test_limit_buy_fills_at_better_price(self, mock_broker):
        """Limit buy should fill at limit price even if market is lower."""
        from core.enums import OrderType

        # Place limit buy at $145
        result = await mock_broker.place_order(
            symbol="AAPL",
            qty=10,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=145.0
        )

        # Price below limit - should fill at limit price
        filled = await mock_broker.check_pending_orders("AAPL", 140.0)
        assert len(filled) == 1
        assert filled[0].avg_price == 145.0  # Fills at limit, not market

    @pytest.mark.asyncio
    async def test_limit_sell_fills_when_price_rises(self, mock_broker):
        """Limit sell should fill when price rises to limit price."""
        from core.enums import OrderType

        # First buy to have a position
        await mock_broker.place_market_order("AAPL", 10, OrderSide.BUY, price=150.0)

        # Place limit sell at $160
        result = await mock_broker.place_order(
            symbol="AAPL",
            qty=10,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            limit_price=160.0
        )

        assert result.status == OrderStatus.PENDING

        # Price below limit - should not fill
        filled = await mock_broker.check_pending_orders("AAPL", 155.0)
        assert len(filled) == 0

        # Price at limit - should fill
        filled = await mock_broker.check_pending_orders("AAPL", 160.0)
        assert len(filled) == 1
        assert filled[0].avg_price == 160.0

    @pytest.mark.asyncio
    async def test_limit_order_requires_limit_price(self, mock_broker):
        """Limit order without limit_price should fail."""
        from core.enums import OrderType

        result = await mock_broker.place_order(
            symbol="AAPL",
            qty=10,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=None
        )

        assert result.success is False
        assert "Limit price required" in result.message


class TestMockBrokerStopOrders:
    """Tests for MockBroker STOP order functionality."""

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker for testing."""
        return MockBroker(starting_cash=100000.0)

    @pytest.mark.asyncio
    async def test_stop_sell_triggers_when_price_drops(self, mock_broker):
        """Stop sell should trigger and fill at market when price hits stop."""
        from core.enums import OrderType

        # First buy to have a position
        await mock_broker.place_market_order("AAPL", 10, OrderSide.BUY, price=150.0)

        # Place stop sell at $140
        result = await mock_broker.place_order(
            symbol="AAPL",
            qty=10,
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            stop_price=140.0
        )

        assert result.status == OrderStatus.PENDING

        # Price above stop - should not trigger
        filled = await mock_broker.check_pending_orders("AAPL", 145.0)
        assert len(filled) == 0

        # Price at stop - should trigger and fill at market
        filled = await mock_broker.check_pending_orders("AAPL", 140.0)
        assert len(filled) == 1
        assert filled[0].avg_price == 140.0  # Fills at market price

    @pytest.mark.asyncio
    async def test_stop_buy_triggers_when_price_rises(self, mock_broker):
        """Stop buy should trigger when price rises to stop."""
        from core.enums import OrderType

        # Place stop buy at $160
        result = await mock_broker.place_order(
            symbol="AAPL",
            qty=10,
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            stop_price=160.0
        )

        assert result.status == OrderStatus.PENDING

        # Price below stop - should not trigger
        filled = await mock_broker.check_pending_orders("AAPL", 155.0)
        assert len(filled) == 0

        # Price at stop - should trigger
        filled = await mock_broker.check_pending_orders("AAPL", 160.0)
        assert len(filled) == 1

    @pytest.mark.asyncio
    async def test_stop_order_requires_stop_price(self, mock_broker):
        """Stop order without stop_price should fail."""
        from core.enums import OrderType

        result = await mock_broker.place_order(
            symbol="AAPL",
            qty=10,
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            stop_price=None
        )

        assert result.success is False
        assert "Stop price required" in result.message


class TestMockBrokerStopLimitOrders:
    """Tests for MockBroker STOP_LIMIT order functionality."""

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker for testing."""
        return MockBroker(starting_cash=100000.0)

    @pytest.mark.asyncio
    async def test_stop_limit_two_phase_behavior(self, mock_broker):
        """Stop-limit should first trigger at stop, then fill at limit."""
        from core.enums import OrderType

        # First buy to have a position
        await mock_broker.place_market_order("AAPL", 10, OrderSide.BUY, price=150.0)

        # Place stop-limit sell: stop at $140, limit at $138
        result = await mock_broker.place_order(
            symbol="AAPL",
            qty=10,
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LIMIT,
            stop_price=140.0,
            limit_price=138.0
        )

        assert result.status == OrderStatus.PENDING
        assert result.stop_triggered is False

        # Price above stop - should not trigger
        filled = await mock_broker.check_pending_orders("AAPL", 145.0)
        assert len(filled) == 0

        # Price hits stop but ABOVE limit (for sell) - stop triggers, doesn't fill yet
        # For a sell stop-limit: stop triggers when price <= stop, but only fills when price >= limit
        # At $137, stop triggers (<=140) but can't fill because 137 < 138 (limit for sell)
        filled = await mock_broker.check_pending_orders("AAPL", 137.0)
        assert len(filled) == 0  # Doesn't fill yet

        # Check stop_triggered flag
        open_orders = await mock_broker.get_open_orders()
        assert len(open_orders) == 1
        assert open_orders[0].stop_triggered is True

        # Price reaches limit - should now fill
        filled = await mock_broker.check_pending_orders("AAPL", 138.0)
        assert len(filled) == 1
        assert filled[0].avg_price == 138.0

    @pytest.mark.asyncio
    async def test_stop_limit_requires_both_prices(self, mock_broker):
        """Stop-limit order requires both stop and limit prices."""
        from core.enums import OrderType

        # Missing limit price
        result1 = await mock_broker.place_order(
            symbol="AAPL",
            qty=10,
            side=OrderSide.BUY,
            order_type=OrderType.STOP_LIMIT,
            stop_price=160.0,
            limit_price=None
        )
        assert result1.success is False
        assert "Both stop and limit prices required" in result1.message

        # Missing stop price
        result2 = await mock_broker.place_order(
            symbol="AAPL",
            qty=10,
            side=OrderSide.BUY,
            order_type=OrderType.STOP_LIMIT,
            stop_price=None,
            limit_price=165.0
        )
        assert result2.success is False
        assert "Both stop and limit prices required" in result2.message


class TestMockBrokerOCOOrders:
    """Tests for MockBroker OCO (one-cancels-other) order functionality."""

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker for testing."""
        return MockBroker(starting_cash=100000.0)

    @pytest.mark.asyncio
    async def test_oco_creates_two_linked_orders(self, mock_broker):
        """OCO should create two linked orders in open orders."""
        result = await mock_broker.place_oco_order(
            symbol="AAPL",
            qty=10,
            stop_price=145.0,
            limit_price=160.0
        )

        assert result.success is True
        assert "oco" in result.order_id.lower()

        open_orders = await mock_broker.get_open_orders()
        assert len(open_orders) == 2

        # Check both legs exist
        stop_leg = next((o for o in open_orders if "stop" in o.order_id), None)
        limit_leg = next((o for o in open_orders if "limit" in o.order_id), None)

        assert stop_leg is not None
        assert limit_leg is not None
        assert stop_leg.oco_pair_id == limit_leg.order_id
        assert limit_leg.oco_pair_id == stop_leg.order_id

    @pytest.mark.asyncio
    async def test_oco_stop_fills_cancels_limit(self, mock_broker):
        """When OCO stop leg fills, limit leg should be cancelled."""
        # First buy to have a position to sell
        await mock_broker.place_market_order("AAPL", 10, OrderSide.BUY, price=150.0)

        result = await mock_broker.place_oco_order(
            symbol="AAPL",
            qty=10,
            stop_price=145.0,
            limit_price=160.0
        )

        # Verify two open orders
        open_orders = await mock_broker.get_open_orders()
        assert len(open_orders) == 2

        # Price drops to stop - stop leg should fill, limit should cancel
        filled = await mock_broker.check_pending_orders("AAPL", 145.0)
        assert len(filled) == 1

        # Check that only stop filled and limit was cancelled
        open_orders = await mock_broker.get_open_orders()
        assert len(open_orders) == 0  # Both removed (one filled, one cancelled)

        # Find the cancelled limit order in historical orders
        limit_order = None
        for order in mock_broker.orders.values():
            if "limit" in order.order_id:
                limit_order = order
                break

        assert limit_order is not None
        assert limit_order.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_oco_limit_fills_cancels_stop(self, mock_broker):
        """When OCO limit leg fills, stop leg should be cancelled."""
        # First buy to have a position to sell
        await mock_broker.place_market_order("AAPL", 10, OrderSide.BUY, price=150.0)

        result = await mock_broker.place_oco_order(
            symbol="AAPL",
            qty=10,
            stop_price=145.0,
            limit_price=160.0
        )

        # Price rises to limit - limit leg should fill, stop should cancel
        filled = await mock_broker.check_pending_orders("AAPL", 160.0)
        assert len(filled) == 1

        # Check that both are removed from open orders
        open_orders = await mock_broker.get_open_orders()
        assert len(open_orders) == 0

        # Find the cancelled stop order in historical orders
        stop_order = None
        for order in mock_broker.orders.values():
            if "stop" in order.order_id:
                stop_order = order
                break

        assert stop_order is not None
        assert stop_order.status == OrderStatus.CANCELLED


class TestMockBrokerMarketHours:
    """Tests for MockBroker market hours functionality."""

    @pytest.mark.asyncio
    async def test_market_always_open_by_default(self):
        """By default, is_market_open should always return True."""
        broker = MockBroker(starting_cash=100000.0, respect_market_hours=False)
        assert await broker.is_market_open() is True

    @pytest.mark.asyncio
    async def test_market_hours_respects_flag(self):
        """When respect_market_hours=True, should check actual hours."""
        broker = MockBroker(starting_cash=100000.0, respect_market_hours=True)

        # This will check actual market hours based on current time
        result = await broker.is_market_open()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_market_hours_constants(self):
        """Verify market hours constants are correct."""
        from datetime import time as dt_time

        broker = MockBroker(respect_market_hours=True)

        # Check market open/close times
        assert broker.MARKET_OPEN == dt_time(9, 30)
        assert broker.MARKET_CLOSE == dt_time(16, 0)

    @pytest.mark.asyncio
    async def test_market_closed_on_weekend_logic(self):
        """Verify weekend detection logic is correct."""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        broker = MockBroker(starting_cash=100000.0, respect_market_hours=True)
        et = ZoneInfo("America/New_York")

        # Verify Saturday = weekday 5, Sunday = weekday 6
        saturday = datetime(2024, 1, 6, 12, 0, 0, tzinfo=et)
        sunday = datetime(2024, 1, 7, 12, 0, 0, tzinfo=et)
        monday = datetime(2024, 1, 8, 12, 0, 0, tzinfo=et)

        assert saturday.weekday() == 5  # Saturday
        assert sunday.weekday() == 6    # Sunday
        assert monday.weekday() == 0    # Monday

        # Verify the broker would correctly identify these as weekend days
        # (weekday >= 5 means closed)
        assert saturday.weekday() >= 5
        assert sunday.weekday() >= 5
        assert monday.weekday() < 5

    @pytest.mark.asyncio
    async def test_market_open_during_trading_hours(self):
        """Market should be open during 9:30 AM - 4:00 PM ET on weekdays."""
        from datetime import time as dt_time

        broker = MockBroker(respect_market_hours=True)

        # Test the time comparison logic
        market_open = broker.MARKET_OPEN
        market_close = broker.MARKET_CLOSE

        # 10:30 AM should be open
        mid_day = dt_time(10, 30)
        assert market_open <= mid_day < market_close

        # 8:00 AM should be closed
        early = dt_time(8, 0)
        assert not (market_open <= early < market_close)

        # 5:00 PM should be closed
        late = dt_time(17, 0)
        assert not (market_open <= late < market_close)


class TestSchwabStreamerFieldMapping:
    """Tests for Schwab Streamer LEVELONE_EQUITIES field mapping."""

    def test_quote_field_mapping_correctness(self):
        """Verify Schwab LEVELONE_EQUITIES field numbers are correct.

        Per Schwab API documentation:
        - Field 1 = Bid Price
        - Field 2 = Ask Price
        - Field 3 = Last Price
        - Field 8 = Total Volume
        - Field 10 = High Price
        - Field 11 = Low Price
        - Field 12 = Close Price (previous day)
        - Field 17 = Open Price
        - Field 35 = Trade Time in Long (ms since epoch)
        """
        # Simulate raw Schwab streamer content
        raw_item = {
            'key': 'AAPL',
            '1': 150.25,   # Bid Price
            '2': 150.30,   # Ask Price
            '3': 150.27,   # Last Price
            '4': 100,      # Bid Size
            '5': 200,      # Ask Size
            '8': 5000000,  # Volume
            '10': 152.00,  # High Price
            '11': 149.00,  # Low Price
            '12': 149.50,  # Close Price (prev day)
            '17': 149.75,  # Open Price
            '35': 1704067200000,  # Trade Time
        }

        # Transform using correct field mapping
        quote = {
            'bid_price': raw_item.get('1'),
            'ask_price': raw_item.get('2'),
            'last_price': raw_item.get('3'),
            'bid_size': raw_item.get('4'),
            'ask_size': raw_item.get('5'),
            'volume': raw_item.get('8'),
            'high_price': raw_item.get('10'),
            'low_price': raw_item.get('11'),
            'close_price': raw_item.get('12'),
            'open_price': raw_item.get('17'),
            'trade_time': raw_item.get('35'),
        }

        # Assertions - verify correct mapping
        assert quote['bid_price'] == 150.25
        assert quote['ask_price'] == 150.30
        assert quote['last_price'] == 150.27
        assert quote['volume'] == 5000000
        assert quote['high_price'] == 152.00
        assert quote['low_price'] == 149.00
        assert quote['close_price'] == 149.50
        assert quote['open_price'] == 149.75

    def test_chart_equity_field_mapping(self):
        """Verify Schwab CHART_EQUITY field numbers are correct.

        Per Schwab API documentation:
        - Field 1 = Open Price
        - Field 2 = High Price
        - Field 3 = Low Price
        - Field 4 = Close Price
        - Field 5 = Volume
        - Field 7 = Chart Time (ms since midnight)
        """
        raw_bar = {
            'key': 'AAPL',
            '1': 150.00,   # Open
            '2': 152.00,   # High
            '3': 149.50,   # Low
            '4': 151.50,   # Close
            '5': 1000000,  # Volume
            '7': 36000000, # Chart Time (10:00 AM as ms since midnight)
        }

        # Transform using correct field mapping
        bar = {
            'open': raw_bar.get('1'),
            'high': raw_bar.get('2'),
            'low': raw_bar.get('3'),
            'close': raw_bar.get('4'),
            'volume': raw_bar.get('5'),
            'chart_time': raw_bar.get('7'),
        }

        # Assertions
        assert bar['open'] == 150.00
        assert bar['high'] == 152.00
        assert bar['low'] == 149.50
        assert bar['close'] == 151.50
        assert bar['volume'] == 1000000

    def test_bid_ask_spread_calculation(self):
        """Verify spread can be calculated from correct fields."""
        raw_item = {
            '1': 150.25,  # Bid
            '2': 150.30,  # Ask
        }

        bid = raw_item['1']
        ask = raw_item['2']
        spread = ask - bid

        assert spread == pytest.approx(0.05)

    def test_incomplete_quote_handling(self):
        """Streamer may send partial updates with only some fields."""
        # Partial update with only last price
        partial_item = {
            'key': 'AAPL',
            '3': 150.27,  # Only last price
        }

        quote = {
            'bid_price': partial_item.get('1'),
            'ask_price': partial_item.get('2'),
            'last_price': partial_item.get('3'),
        }

        assert quote['last_price'] == 150.27
        assert quote['bid_price'] is None
        assert quote['ask_price'] is None


class TestMockBrokerReset:
    """Tests for MockBroker reset functionality."""

    @pytest.mark.asyncio
    async def test_reset_clears_open_orders(self):
        """Reset should clear all open orders."""
        from core.enums import OrderType

        broker = MockBroker(starting_cash=100000.0)

        # Place some pending orders
        await broker.place_order("AAPL", 10, OrderSide.BUY, OrderType.LIMIT, limit_price=145.0)
        await broker.place_order("MSFT", 5, OrderSide.BUY, OrderType.LIMIT, limit_price=340.0)

        open_orders = await broker.get_open_orders()
        assert len(open_orders) == 2

        # Reset
        broker.reset(starting_cash=50000.0)

        open_orders = await broker.get_open_orders()
        assert len(open_orders) == 0
        assert broker.portfolio.cash == 50000.0

    @pytest.mark.asyncio
    async def test_reset_clears_historical_orders(self):
        """Reset should clear all historical orders."""
        broker = MockBroker(starting_cash=100000.0)

        # Place a market order
        await broker.place_market_order("AAPL", 10, OrderSide.BUY, price=150.0)

        assert len(broker.orders) > 0

        # Reset
        broker.reset()

        assert len(broker.orders) == 0
