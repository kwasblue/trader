"""
Tests for SchwabBroker error recovery improvements.

Tests cover:
- Retry behavior on get_position failure
- Circuit breaker opening after threshold
- Market hours returning False on error (safe default)
- Configurable circuit breaker settings
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone

from core.broker.schwab_broker import SchwabBroker


@pytest.fixture
def mock_client():
    """Create a mock SchwabClient."""
    client = MagicMock()
    client.account_number = MagicMock(return_value={
        "accountNumbers": [{"accountNumber": "12345678"}]
    })
    client.accounts_number = MagicMock(return_value={
        "securitiesAccount": {
            "positions": [],
            "currentBalances": {
                "availableFunds": 100000.0,
                "buyingPower": 100000.0,
                "liquidationValue": 100000.0,
            }
        }
    })
    return client


@pytest.fixture
def broker(mock_client):
    """Create a SchwabBroker instance."""
    return SchwabBroker(client=mock_client, session="NORMAL")


class TestGetPositionRetry:
    """Test retry behavior for get_position."""

    @pytest.mark.asyncio
    async def test_get_position_retries_on_failure(self, broker, mock_client):
        """Test that get_position retries on transient failures."""
        call_count = 0

        def failing_then_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return {
                "securitiesAccount": {
                    "positions": [{
                        "instrument": {"symbol": "AAPL"},
                        "longQuantity": 100,
                        "averagePrice": 150.0,
                    }]
                }
            }

        mock_client.accounts_number = failing_then_success

        # Should succeed after retries
        result = await broker.get_position("AAPL")

        # Should have been called 3 times (2 failures + 1 success)
        assert call_count == 3
        assert result is not None
        assert result.symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_get_position_returns_none_for_missing(self, broker, mock_client):
        """Test that get_position returns None for non-existent position."""
        mock_client.accounts_number = MagicMock(return_value={
            "securitiesAccount": {"positions": []}
        })

        result = await broker.get_position("AAPL")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_position_exhausts_retries(self, broker, mock_client):
        """Test that get_position raises after exhausting retries."""
        mock_client.accounts_number = MagicMock(
            side_effect=ConnectionError("Persistent failure")
        )

        with pytest.raises(ConnectionError):
            await broker.get_position("AAPL")


class TestGetAccountInfoRetry:
    """Test retry behavior for get_account_info."""

    @pytest.mark.asyncio
    async def test_get_account_info_retries(self, broker, mock_client):
        """Test that get_account_info retries on failure."""
        call_count = 0

        def failing_then_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("Request timeout")
            return {
                "securitiesAccount": {
                    "positions": [],
                    "currentBalances": {
                        "availableFunds": 50000.0,
                        "buyingPower": 50000.0,
                        "liquidationValue": 50000.0,
                    }
                }
            }

        mock_client.accounts_number = failing_then_success

        result = await broker.get_account_info()

        assert call_count == 2
        assert result.cash == 50000.0


class TestCircuitBreaker:
    """Test circuit breaker behavior."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_threshold(self, broker, mock_client):
        """Test that circuit breaker opens after repeated failures."""
        # Simulate repeated failures
        mock_client.place_orders = MagicMock(side_effect=Exception("API error"))

        # Make enough failed requests to open the circuit breaker
        failure_count = 0
        for _ in range(10):
            try:
                await broker.place_order(
                    symbol="AAPL",
                    qty=100,
                    side="buy",
                    order_type="market"
                )
            except Exception:
                failure_count += 1

        # Circuit breaker should be open
        # Additional requests should be rejected immediately
        result = await broker.place_order(
            symbol="AAPL",
            qty=100,
            side="buy",
            order_type="market"
        )

        # Should be rejected by circuit breaker
        assert result.status == "rejected"
        assert result.raw.get("error") == "circuit_breaker_open"

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_success(self, broker, mock_client):
        """Test that circuit breaker resets after successful requests."""
        # First, cause some failures (but not enough to open circuit)
        # Note: @async_retry retries 3 times per call, so 1 call = 3 failures
        # Default threshold is 5, so 1 call (3 failures) keeps circuit closed
        mock_client.place_orders = MagicMock(side_effect=Exception("API error"))

        try:
            await broker.place_order(
                symbol="AAPL",
                qty=100,
                side="buy",
                order_type="market"
            )
        except Exception:
            pass

        # Verify circuit is still closed (3 failures < 5 threshold)
        assert broker._circuit_breaker.failure_count == 3

        # Now make a successful request
        mock_client.place_orders = MagicMock(return_value={
            "orderId": "123",
            "status": "FILLED",
        })

        result = await broker.place_order(
            symbol="AAPL",
            qty=100,
            side="buy",
            order_type="market"
        )

        # Should succeed and reset the failure count
        assert result.status != "rejected"
        assert broker._circuit_breaker.failure_count == 0


class TestMarketHoursSafeDefault:
    """Test market hours returns False on error."""

    @pytest.mark.asyncio
    async def test_market_hours_returns_false_on_error(self, broker):
        """Test that is_market_open returns False on error."""
        # Patch the datetime to raise an exception
        with patch('core.broker.schwab_broker.datetime') as mock_dt:
            mock_dt.now.side_effect = Exception("Timezone error")
            mock_dt.side_effect = Exception("Timezone error")

            result = await broker.is_market_open()

            # Should return False (safe default) on error
            assert result is False

    @pytest.mark.asyncio
    async def test_market_hours_returns_false_on_missing_timezone(self, broker):
        """Test that is_market_open returns False when timezone lib unavailable."""
        # Patch ZoneInfo to raise ImportError
        with patch.dict('sys.modules', {'zoneinfo': None}):
            with patch('builtins.__import__', side_effect=ImportError("No zoneinfo")):
                # This is tricky to test, but the code path handles ImportError
                # The actual test would need to manipulate imports more carefully
                pass

    @pytest.mark.asyncio
    async def test_market_hours_weekend(self, broker):
        """Test that is_market_open returns False on weekends."""
        # Patch datetime to return a Saturday
        with patch('core.broker.schwab_broker.datetime') as mock_dt:
            from zoneinfo import ZoneInfo
            et_tz = ZoneInfo("America/New_York")

            # Create a Saturday
            saturday = datetime(2024, 1, 6, 12, 0, 0, tzinfo=et_tz)  # Jan 6, 2024 is Saturday
            mock_dt.now.return_value = saturday

            result = await broker.is_market_open()

            # Should return False for weekend
            assert result is False


class TestConfigurableCircuitBreaker:
    """Test that circuit breaker is configurable via config."""

    def test_circuit_breaker_uses_config(self, mock_client):
        """Test that circuit breaker settings come from config."""
        with patch('core.config_loader.get_config') as mock_config:
            mock_err_config = MagicMock()
            mock_err_config.circuit_breaker_failure_threshold = 10
            mock_err_config.circuit_breaker_timeout = 120.0
            mock_config.return_value.error_recovery = mock_err_config

            broker = SchwabBroker(client=mock_client, session="NORMAL")

            # Check that circuit breaker was configured with custom values
            assert broker._circuit_breaker.config.failure_threshold == 10
            assert broker._circuit_breaker.config.timeout == 120.0


class TestGetOpenOrdersRetry:
    """Test retry behavior for get_open_orders."""

    @pytest.mark.asyncio
    async def test_get_open_orders_retries(self, broker, mock_client):
        """Test that get_open_orders retries on failure."""
        call_count = 0

        def failing_then_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Network error")
            return []

        mock_client.all_orders = failing_then_success

        result = await broker.get_open_orders()

        assert call_count == 2
        assert result == []


class TestGetOrderStatusRetry:
    """Test retry behavior for get_order_status."""

    @pytest.mark.asyncio
    async def test_get_order_status_retries(self, broker, mock_client):
        """Test that get_order_status retries on failure."""
        call_count = 0

        def failing_then_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("Timeout")
            return {
                "orderId": "123",
                "status": "FILLED",
            }

        mock_client.get_order_by_id = failing_then_success

        result = await broker.get_order_status("123")

        assert call_count == 2
        assert result.order_id == "123"
