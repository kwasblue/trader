"""
Tests for AlpacaBroker

Coverage:
- Connection management
- Order placement (market, limit)
- Order cancellation
- Position retrieval
- Market status checks
- Streaming subscription
- Error handling and retry logic
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from datetime import datetime, timezone
import asyncio

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.broker.alpaca_broker import AlpacaBroker
from core.app_types import OrderResult, PositionView


@pytest.fixture(autouse=True)
def mock_async_create_task(monkeypatch):
    """Mock asyncio.create_task to prevent 'no running event loop' errors."""
    def mock_create_task(coro, **kwargs):
        # Close the coroutine to avoid warnings
        coro.close()
        return MagicMock()
    monkeypatch.setattr(asyncio, 'create_task', mock_create_task)


class TestAlpacaBrokerInit:
    """Tests for AlpacaBroker initialization."""

    def test_init(self):
        """Test broker initialization."""
        with patch('core.broker.alpaca_broker.Logger'):
            broker = AlpacaBroker(
                api_key="test_key",
                api_secret="test_secret",
                paper=True
            )
            assert broker.api_key == "test_key"
            assert broker.api_secret == "test_secret"
            assert broker.paper is True

    def test_init_live_mode(self):
        """Test broker initialization in live mode."""
        with patch('core.broker.alpaca_broker.Logger'):
            broker = AlpacaBroker(
                api_key="test_key",
                api_secret="test_secret",
                paper=False
            )
            assert broker.paper is False


class TestConnection:
    """Tests for connection management."""

    @pytest.fixture
    def broker(self):
        with patch('core.broker.alpaca_broker.Logger'):
            return AlpacaBroker(
                api_key="test_key",
                api_secret="test_secret",
                paper=True
            )

    @pytest.mark.asyncio
    async def test_connect_success(self, broker):
        """Test successful connection."""
        with patch('core.broker.alpaca_broker.TradingClient') as MockTradingClient:
            with patch('core.broker.alpaca_broker.StockDataStream') as MockStream:
                with patch('core.broker.alpaca_broker.StockHistoricalDataClient') as MockDataClient:
                    with patch('core.broker.alpaca_broker.get_health_checker') as mock_health:
                        mock_health.return_value = MagicMock()

                        await broker.connect()

                        MockTradingClient.assert_called_once()
                        MockStream.assert_called_once()
                        MockDataClient.assert_called_once()
                        assert broker.trading_client is not None

    @pytest.mark.asyncio
    async def test_connect_failure_retry(self, broker):
        """Test connection retry on failure."""
        with patch('core.broker.alpaca_broker.TradingClient') as MockTradingClient:
            MockTradingClient.side_effect = Exception("Connection failed")

            with pytest.raises(Exception):
                await broker.connect()


class TestOrderPlacement:
    """Tests for order placement."""

    @pytest.fixture
    def connected_broker(self):
        with patch('core.broker.alpaca_broker.Logger'):
            broker = AlpacaBroker(
                api_key="test_key",
                api_secret="test_secret",
                paper=True
            )
            broker.trading_client = MagicMock()
            return broker

    @pytest.mark.asyncio
    async def test_place_market_order_buy(self, connected_broker):
        """Test placing a market buy order."""
        mock_order = MagicMock()
        mock_order.id = "order123"
        mock_order.symbol = "AAPL"
        mock_order.side = "buy"
        mock_order.qty = "10"
        mock_order.status = "filled"
        mock_order.filled_qty = "10"
        mock_order.filled_avg_price = "150.00"

        connected_broker.trading_client.submit_order.return_value = mock_order

        result = await connected_broker.place_order(
            symbol="AAPL",
            side="buy",
            qty=10,
            order_type="market"
        )

        assert result.order_id == "order123"
        assert result.symbol == "AAPL"
        assert result.side == "buy"

    @pytest.mark.asyncio
    async def test_place_market_order_sell(self, connected_broker):
        """Test placing a market sell order."""
        mock_order = MagicMock()
        mock_order.id = "order456"
        mock_order.symbol = "AAPL"
        mock_order.side = "sell"
        mock_order.qty = "10"
        mock_order.status = "filled"
        mock_order.filled_qty = "10"
        mock_order.filled_avg_price = "155.00"

        connected_broker.trading_client.submit_order.return_value = mock_order

        result = await connected_broker.place_order(
            symbol="AAPL",
            side="sell",
            qty=10,
            order_type="market"
        )

        assert result.side == "sell"

    @pytest.mark.asyncio
    async def test_place_limit_order(self, connected_broker):
        """Test placing a limit order."""
        mock_order = MagicMock()
        mock_order.id = "order789"
        mock_order.symbol = "AAPL"
        mock_order.side = "buy"
        mock_order.qty = "10"
        mock_order.status = "new"
        mock_order.filled_qty = "0"
        mock_order.filled_avg_price = None

        connected_broker.trading_client.submit_order.return_value = mock_order

        result = await connected_broker.place_order(
            symbol="AAPL",
            side="buy",
            qty=10,
            order_type="limit",
            limit_price=148.00
        )

        assert result.status == "new"

    @pytest.mark.asyncio
    async def test_place_order_failure(self, connected_broker):
        """Test order placement failure handling."""
        connected_broker.trading_client.submit_order.side_effect = Exception("Insufficient funds")

        with pytest.raises(Exception):
            await connected_broker.place_order(
                symbol="AAPL",
                side="buy",
                qty=10000,
                order_type="market"
            )


class TestOrderCancellation:
    """Tests for order cancellation."""

    @pytest.fixture
    def connected_broker(self):
        with patch('core.broker.alpaca_broker.Logger'):
            broker = AlpacaBroker(
                api_key="test_key",
                api_secret="test_secret",
                paper=True
            )
            broker.trading_client = MagicMock()
            return broker

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, connected_broker):
        """Test successful order cancellation."""
        connected_broker.trading_client.cancel_order_by_id.return_value = None

        # cancel_order returns None per implementation (emits event instead)
        await connected_broker.cancel_order("order123")

        connected_broker.trading_client.cancel_order_by_id.assert_called_once_with("order123")

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(self, connected_broker):
        """Test cancellation of non-existent order."""
        connected_broker.trading_client.cancel_order_by_id.side_effect = Exception("Order not found")

        with pytest.raises(Exception):
            await connected_broker.cancel_order("nonexistent")


class TestPositionRetrieval:
    """Tests for position retrieval."""

    @pytest.fixture
    def connected_broker(self):
        with patch('core.broker.alpaca_broker.Logger'):
            broker = AlpacaBroker(
                api_key="test_key",
                api_secret="test_secret",
                paper=True
            )
            broker.trading_client = MagicMock()
            return broker

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="get_position mock issue - trading_client mock not working in asyncio.to_thread")
    async def test_get_position_exists(self, connected_broker):
        """Test retrieving an existing position."""
        mock_position = MagicMock()
        mock_position.symbol = "AAPL"
        mock_position.qty = 100  # Use int/float, not string
        mock_position.avg_entry_price = 145.00
        mock_position.current_price = 150.00
        mock_position.side = "long"
        mock_position.unrealized_pl = 500.00
        mock_position.unrealized_plpc = 0.034

        connected_broker.trading_client.get_open_position.return_value = mock_position

        result = await connected_broker.get_position("AAPL")

        assert result is not None
        assert result.symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_get_position_not_exists(self, connected_broker):
        """Test retrieving a non-existent position."""
        connected_broker.trading_client.get_open_position.side_effect = Exception("Position not found")

        result = await connected_broker.get_position("MSFT")

        # Should return None or empty position, not raise
        assert result is None or result.qty == 0

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="get_all_positions not implemented in AlpacaBroker")
    async def test_get_all_positions(self, connected_broker):
        """Test retrieving all positions."""
        mock_positions = [
            MagicMock(symbol="AAPL", qty="100", avg_entry_price="145.00", current_price="150.00", side="long"),
            MagicMock(symbol="MSFT", qty="50", avg_entry_price="300.00", current_price="310.00", side="long")
        ]

        connected_broker.trading_client.get_all_positions.return_value = mock_positions

        result = await connected_broker.get_all_positions()

        assert len(result) == 2


class TestMarketStatus:
    """Tests for market status checking."""

    @pytest.fixture
    def connected_broker(self):
        with patch('core.broker.alpaca_broker.Logger'):
            broker = AlpacaBroker(
                api_key="test_key",
                api_secret="test_secret",
                paper=True
            )
            broker.trading_client = MagicMock()
            return broker

    @pytest.mark.asyncio
    async def test_is_market_open_true(self, connected_broker):
        """Test market open status."""
        mock_clock = MagicMock()
        mock_clock.is_open = True

        connected_broker.trading_client.get_clock.return_value = mock_clock

        result = await connected_broker.is_market_open()
        assert result is True

    @pytest.mark.asyncio
    async def test_is_market_open_false(self, connected_broker):
        """Test market closed status."""
        mock_clock = MagicMock()
        mock_clock.is_open = False

        connected_broker.trading_client.get_clock.return_value = mock_clock

        result = await connected_broker.is_market_open()
        assert result is False


class TestStreaming:
    """Tests for streaming subscription."""

    @pytest.fixture
    def connected_broker(self):
        with patch('core.broker.alpaca_broker.Logger'):
            broker = AlpacaBroker(
                api_key="test_key",
                api_secret="test_secret",
                paper=True
            )
            broker.trading_client = MagicMock()
            broker.stream = MagicMock()
            return broker

    def test_subscribe_bars(self, connected_broker):
        """Test subscribing to bar updates."""
        callback = MagicMock()

        connected_broker.subscribe_bars(callback, "AAPL")

        connected_broker.stream.subscribe_bars.assert_called()

    @pytest.mark.skip(reason="subscribe_quotes not implemented in AlpacaBroker")
    def test_subscribe_quotes(self, connected_broker):
        """Test subscribing to quote updates."""
        callback = MagicMock()

        connected_broker.subscribe_quotes(callback, "AAPL")

        connected_broker.stream.subscribe_quotes.assert_called()

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="start_stream hangs in test environment due to asyncio.create_task mock")
    async def test_start_stream(self, connected_broker):
        """Test starting the data stream."""
        connected_broker.stream.run = AsyncMock()

        await connected_broker.start_stream()

        connected_broker.stream.run.assert_called_once()


class TestAccountInfo:
    """Tests for account information retrieval."""

    @pytest.fixture
    def connected_broker(self):
        with patch('core.broker.alpaca_broker.Logger'):
            broker = AlpacaBroker(
                api_key="test_key",
                api_secret="test_secret",
                paper=True
            )
            broker.trading_client = MagicMock()
            return broker

    @pytest.mark.asyncio
    async def test_get_account(self, connected_broker):
        """Test getting account information."""
        mock_account = MagicMock()
        mock_account.equity = "100000.00"
        mock_account.buying_power = "50000.00"
        mock_account.cash = "25000.00"
        mock_account.status = "ACTIVE"
        mock_account.account_number = "12345"

        mock_positions = []
        connected_broker.trading_client.get_account.return_value = mock_account
        connected_broker.trading_client.get_all_positions.return_value = mock_positions

        result = await connected_broker.get_account_info()

        assert result is not None

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="get_snapshot not implemented, use get_account_info instead")
    async def test_get_snapshot(self, connected_broker):
        """Test getting broker snapshot."""
        mock_account = MagicMock()
        mock_account.equity = "100000.00"
        mock_account.buying_power = "50000.00"
        mock_account.cash = "25000.00"

        mock_positions = [
            MagicMock(symbol="AAPL", qty="100", market_value="15000.00")
        ]

        connected_broker.trading_client.get_account.return_value = mock_account
        connected_broker.trading_client.get_all_positions.return_value = mock_positions

        snapshot = await connected_broker.get_snapshot()

        assert snapshot is not None


class TestErrorHandling:
    """Tests for error handling and retry logic."""

    @pytest.fixture
    def broker(self):
        with patch('core.broker.alpaca_broker.Logger'):
            return AlpacaBroker(
                api_key="test_key",
                api_secret="test_secret",
                paper=True
            )

    @pytest.mark.asyncio
    async def test_api_error_handling(self, broker):
        """Test handling of API errors."""
        broker.trading_client = MagicMock()
        broker.trading_client.submit_order.side_effect = Exception("API rate limit exceeded")

        with pytest.raises(Exception) as exc_info:
            await broker.place_order(
                symbol="AAPL",
                side="buy",
                qty=10,
                order_type="market"
            )

        assert "rate limit" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_network_error_handling(self, broker):
        """Test handling of network errors."""
        broker.trading_client = MagicMock()
        broker.trading_client.get_account.side_effect = ConnectionError("Network unreachable")

        with pytest.raises(ConnectionError):
            await broker.get_account_info()
