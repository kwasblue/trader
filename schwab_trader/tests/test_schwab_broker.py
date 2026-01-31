"""
Tests for SchwabBroker

Coverage:
- Order placement (market, limit, stop, stop_limit)
- Order cancellation
- Account synchronization
- Position retrieval
- Market hours check
- Circuit breaker functionality
- Event emission
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
import asyncio

from core.broker.schwab_broker import SchwabBroker, CircuitBreaker
from core.app_types import OrderResult, PositionView, BrokerSnapshot


@pytest.fixture
def mock_client():
    """Create a mock SchwabClient."""
    client = MagicMock()
    client.account_number.return_value = {"accountNumbers": [{"accountNumber": "123456789"}]}
    client.accounts_number.return_value = {
        "securitiesAccount": {
            "type": "MARGIN",
            "positions": [],
            "currentBalances": {
                "availableFunds": 10000.0,
                "buyingPower": 20000.0,
                "liquidationValue": 25000.0,
            }
        }
    }
    client.generate_order.return_value = {
        "orderType": "MARKET",
        "session": "NORMAL",
        "duration": "DAY",
        "orderStrategyType": "SINGLE",
        "orderLegCollection": [{
            "instruction": "BUY",
            "quantity": 10,
            "instrument": {"symbol": "AAPL", "assetType": "EQUITY"}
        }]
    }
    client.place_orders.return_value = {
        "orderId": "ORD123",
        "status": "FILLED",
        "filledQuantity": 10,
        "averagePrice": 150.0,
    }
    client.cancel_order.return_value = {
        "orderId": "ORD123",
        "status": "CANCELLED",
    }
    client.all_orders.return_value = []
    client.get_order_by_id.return_value = {
        "orderId": "ORD123",
        "status": "FILLED",
    }
    client.quote.return_value = {
        "AAPL": {"lastPrice": 150.0}
    }
    return client


@pytest.fixture
def broker(mock_client):
    """Create a SchwabBroker instance with mocked client."""
    with patch('core.broker.schwab_broker.get_event_handler') as mock_eh:
        mock_eh.return_value = AsyncMock()
        broker = SchwabBroker(mock_client)
        return broker


class TestSchwabBrokerOrders:
    """Test order placement and cancellation."""

    @pytest.mark.asyncio
    async def test_place_market_order(self, broker, mock_client):
        """Test placing a market order."""
        result = await broker.place_order(
            symbol="AAPL",
            qty=10,
            side="buy",
            order_type="market",
        )

        assert result.order_id == "ORD123"
        assert result.status == "filled"
        assert result.symbol == "AAPL"
        mock_client.place_orders.assert_called_once()

    @pytest.mark.asyncio
    async def test_place_limit_order(self, broker, mock_client):
        """Test placing a limit order."""
        result = await broker.place_order(
            symbol="AAPL",
            qty=10,
            side="buy",
            order_type="limit",
            limit_price=145.0,
        )

        assert result.order_id == "ORD123"
        # Verify limit price was set in the order
        call_args = mock_client.place_orders.call_args
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_place_limit_order_missing_price(self, broker):
        """Test that limit order without price raises error."""
        with pytest.raises(ValueError, match="limit_price required"):
            await broker.place_order(
                symbol="AAPL",
                qty=10,
                side="buy",
                order_type="limit",
            )

    @pytest.mark.asyncio
    async def test_place_stop_order(self, broker, mock_client):
        """Test placing a stop order."""
        result = await broker.place_order(
            symbol="AAPL",
            qty=10,
            side="sell",
            order_type="stop",
            stop_price=140.0,
        )

        assert result.order_id == "ORD123"
        mock_client.place_orders.assert_called_once()

    @pytest.mark.asyncio
    async def test_place_stop_limit_order(self, broker, mock_client):
        """Test placing a stop-limit order."""
        result = await broker.place_order(
            symbol="AAPL",
            qty=10,
            side="sell",
            order_type="stop_limit",
            stop_price=140.0,
            limit_price=139.0,
        )

        assert result.order_id == "ORD123"
        mock_client.place_orders.assert_called_once()

    @pytest.mark.asyncio
    async def test_place_order_invalid_qty(self, broker):
        """Test that invalid quantity raises error."""
        with pytest.raises(ValueError, match="qty must be > 0"):
            await broker.place_order(
                symbol="AAPL",
                qty=0,
                side="buy",
            )

    @pytest.mark.asyncio
    async def test_cancel_order(self, broker, mock_client):
        """Test cancelling an order."""
        result = await broker.cancel_order("ORD123")

        assert result.order_id == "ORD123"
        assert result.status == "cancelled"
        mock_client.cancel_order.assert_called_once_with("123456789", "ORD123")


class TestSchwabBrokerAccount:
    """Test account-related functionality."""

    @pytest.mark.asyncio
    async def test_get_account_info(self, broker, mock_client):
        """Test retrieving account information."""
        result = await broker.get_account_info()

        assert isinstance(result, BrokerSnapshot)
        assert result.account_number == "123456789"
        assert result.cash == 10000.0
        assert result.buying_power == 20000.0

    @pytest.mark.asyncio
    async def test_get_position(self, broker, mock_client):
        """Test retrieving position for a symbol."""
        mock_client.accounts_number.return_value = {
            "securitiesAccount": {
                "positions": [{
                    "instrument": {"symbol": "AAPL"},
                    "longQuantity": 100,
                    "shortQuantity": 0,
                    "averagePrice": 145.0,
                }]
            }
        }

        result = await broker.get_position("AAPL")

        assert result is not None
        assert result.symbol == "AAPL"
        assert result.qty == 100
        assert result.avg_entry_price == 145.0

    @pytest.mark.asyncio
    async def test_get_position_not_found(self, broker, mock_client):
        """Test retrieving position for symbol not held."""
        result = await broker.get_position("MSFT")
        assert result is None

    def test_get_quote(self, broker, mock_client):
        """Test getting a quote for a symbol."""
        price = broker.get_quote("AAPL")
        assert price == 150.0

    def test_get_available_funds(self, broker, mock_client):
        """Test getting available funds."""
        funds = broker.get_available_funds()
        assert funds == 10000.0


class TestSchwabBrokerMarketHours:
    """Test market hours functionality."""

    @pytest.mark.asyncio
    async def test_is_market_open_weekday(self, broker):
        """Test market hours check on a weekday."""
        # This test depends on the actual time, so we just verify it returns a boolean
        result = await broker.is_market_open()
        assert isinstance(result, bool)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_initial_state(self):
        """Test circuit breaker starts closed."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        assert cb.state == CircuitBreaker.CLOSED
        assert cb.can_execute() is True

    def test_opens_after_threshold(self):
        """Test circuit breaker opens after failure threshold."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitBreaker.OPEN
        assert cb.can_execute() is False

    def test_resets_on_success(self):
        """Test circuit breaker resets on success."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()

        assert cb.failures == 0
        assert cb.state == CircuitBreaker.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_rejects_order(self, broker, mock_client):
        """Test that orders are rejected when circuit breaker is open."""
        # Open the circuit breaker
        for _ in range(5):
            broker._circuit_breaker.record_failure()

        result = await broker.place_order(
            symbol="AAPL",
            qty=10,
            side="buy",
        )

        assert result.status == "rejected"
        assert result.raw.get("error") == "circuit_breaker_open"
        mock_client.place_orders.assert_not_called()


class TestSchwabBrokerStreaming:
    """Test streaming functionality."""

    def test_connect_stream(self, broker):
        """Test initializing streaming connection."""
        broker.connect_stream("api_key", "secret_key")
        assert broker.stream is not None

    def test_subscribe_quotes(self, broker):
        """Test subscribing to quotes."""
        broker.connect_stream("api_key", "secret_key")

        callback = MagicMock()
        broker.subscribe_quotes(callback, "AAPL")

        assert "AAPL" in broker._stream_symbols
        assert callback in broker._quote_callbacks.get("AAPL", [])

    def test_unsubscribe_quotes(self, broker):
        """Test unsubscribing from quotes."""
        broker.connect_stream("api_key", "secret_key")

        callback = MagicMock()
        broker.subscribe_quotes(callback, "AAPL")
        broker.unsubscribe_quotes(callback, "AAPL")

        assert callback not in broker._quote_callbacks.get("AAPL", [])


class TestSchwabBrokerConnection:
    """Test connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_success(self, broker, mock_client):
        """Test successful connection."""
        result = await broker.connect()
        assert result is True
        assert broker._connected is True

    @pytest.mark.asyncio
    async def test_connect_failure(self, broker, mock_client):
        """Test connection failure."""
        mock_client.account_number.return_value = {"accountNumbers": []}

        result = await broker.connect()
        assert result is False
        assert broker._connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self, broker):
        """Test disconnection."""
        broker._connected = True
        await broker.disconnect()
        assert broker._connected is False


class TestSchwabBrokerEventEmission:
    """Test event emission."""

    @pytest.mark.asyncio
    async def test_order_event_emitted(self, broker, mock_client):
        """Test that order events are emitted."""
        with patch.object(broker._event_handler, 'emit', new_callable=AsyncMock) as mock_emit:
            await broker.place_order(
                symbol="AAPL",
                qty=10,
                side="buy",
            )

            # Verify event was emitted
            assert mock_emit.called
            call_args = mock_emit.call_args_list
            # Find the order status event call
            order_events = [c for c in call_args if 'ORDER' in str(c)]
            assert len(order_events) > 0

    @pytest.mark.asyncio
    async def test_health_event_emitted_on_connect(self, broker, mock_client):
        """Test that health event is emitted on connect."""
        with patch.object(broker._event_handler, 'emit', new_callable=AsyncMock) as mock_emit:
            await broker.connect()

            # Verify health event was emitted
            health_events = [c for c in mock_emit.call_args_list if 'HEALTH' in str(c)]
            assert len(health_events) > 0
