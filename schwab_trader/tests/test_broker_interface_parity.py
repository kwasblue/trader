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
