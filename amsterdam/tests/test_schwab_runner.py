"""
Tests for SchwabLiveRunner

Coverage:
- Quote canonicalization (LEVELONE_EQUITIES field mapping)
- Chart bar canonicalization (CHART_EQUITY field mapping)
- Bar aggregation from quotes
- Bar bucket calculation
- Periodic data update scheduling
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestCanonicalizeSchwabQuote:
    """Tests for SchwabLiveRunner._canonicalize_schwab_quote static method."""

    def test_canonicalize_with_numeric_fields(self):
        """Test canonicalization using Schwab numeric field IDs."""
        from core.schwab_runner import SchwabLiveRunner

        raw_quote = {
            "1": 150.25,  # Bid Price
            "2": 150.30,  # Ask Price
            "3": 150.27,  # Last Price
            "8": 5000000,  # Volume
            "10": 152.00,  # High Price
            "11": 149.00,  # Low Price
            "12": 149.50,  # Close Price (prev day)
            "17": 149.75,  # Open Price
            "35": 1704067200000,  # Trade Time (ms since epoch)
        }

        result = SchwabLiveRunner._canonicalize_schwab_quote(raw_quote, "AAPL")

        assert result["symbol"] == "AAPL"
        assert result["Close"] == 150.27  # Uses last price
        assert result["Open"] == 149.75
        assert result["High"] == 152.00
        assert result["Low"] == 149.00
        assert result["Volume"] == 5000000
        assert result["prev_close"] == 149.50
        assert isinstance(result["timestamp"], datetime)

    def test_canonicalize_with_named_fields(self):
        """Test canonicalization using named fields (fallback)."""
        from core.schwab_runner import SchwabLiveRunner

        raw_quote = {
            "bid_price": 150.25,
            "ask_price": 150.30,
            "last_price": 150.27,
            "volume": 5000000,
            "high_price": 152.00,
            "low_price": 149.00,
            "close_price": 149.50,
            "open_price": 149.75,
        }

        result = SchwabLiveRunner._canonicalize_schwab_quote(raw_quote, "MSFT")

        assert result["symbol"] == "MSFT"
        assert result["Close"] == 150.27
        assert result["Open"] == 149.75
        assert result["High"] == 152.00
        assert result["Low"] == 149.00

    def test_canonicalize_uses_mid_price_when_no_last(self):
        """Test that mid price is used when last price is not available."""
        from core.schwab_runner import SchwabLiveRunner

        raw_quote = {
            "1": 150.00,  # Bid
            "2": 151.00,  # Ask
            # No field 3 (last price)
        }

        result = SchwabLiveRunner._canonicalize_schwab_quote(raw_quote, "AAPL")

        # Should use mid price: (150 + 151) / 2 = 150.5
        assert result["Close"] == 150.5

    def test_canonicalize_partial_quote(self):
        """Test canonicalization with partial data (only some fields present)."""
        from core.schwab_runner import SchwabLiveRunner

        raw_quote = {
            "3": 150.27,  # Only last price
        }

        result = SchwabLiveRunner._canonicalize_schwab_quote(raw_quote, "AAPL")

        assert result["Close"] == 150.27
        # Other OHLC fields fall back to Close when not available
        assert result["Open"] == 150.27
        assert result["High"] == 150.27
        assert result["Low"] == 150.27

    def test_canonicalize_timestamp_from_trade_time(self):
        """Test that timestamp is parsed from trade time field."""
        from core.schwab_runner import SchwabLiveRunner

        # Jan 1, 2024 12:00:00 UTC in milliseconds
        trade_time_ms = 1704110400000

        raw_quote = {
            "3": 150.27,
            "35": trade_time_ms,
        }

        result = SchwabLiveRunner._canonicalize_schwab_quote(raw_quote, "AAPL")

        expected_ts = datetime.fromtimestamp(trade_time_ms / 1000, tz=timezone.utc)
        assert result["timestamp"] == expected_ts


class TestCanonicalizeSchwabChartBar:
    """Tests for SchwabLiveRunner._canonicalize_schwab_chart_bar static method."""

    def test_canonicalize_chart_bar(self):
        """Test CHART_EQUITY field mapping (different from LEVELONE_EQUITIES)."""
        from core.schwab_runner import SchwabLiveRunner

        raw_bar = {
            "1": 150.00,  # Open (different from LEVELONE where 1=Bid)
            "2": 152.00,  # High (different from LEVELONE where 2=Ask)
            "3": 149.50,  # Low (different from LEVELONE where 3=Last)
            "4": 151.50,  # Close
            "5": 1000000,  # Volume
            "6": 123,  # Sequence
            "7": 1704110400000,  # Chart Time
        }

        result = SchwabLiveRunner._canonicalize_schwab_chart_bar(raw_bar, "AAPL")

        assert result["symbol"] == "AAPL"
        assert result["Open"] == 150.00
        assert result["High"] == 152.00
        assert result["Low"] == 149.50
        assert result["Close"] == 151.50
        assert result["Volume"] == 1000000
        assert result["sequence"] == 123

    def test_chart_vs_levelone_field_difference(self):
        """Verify CHART_EQUITY and LEVELONE_EQUITIES use different field meanings."""
        from core.schwab_runner import SchwabLiveRunner

        # Same raw fields but different interpretation
        raw_data = {
            "1": 150.00,
            "2": 152.00,
            "3": 149.50,
        }

        # As LEVELONE_EQUITIES quote
        quote_result = SchwabLiveRunner._canonicalize_schwab_quote(raw_data, "AAPL")
        # Field 1 = Bid, Field 2 = Ask, Field 3 = Last
        assert quote_result["Close"] == 149.50  # Field 3 = Last price

        # As CHART_EQUITY bar
        chart_result = SchwabLiveRunner._canonicalize_schwab_chart_bar(raw_data, "AAPL")
        # Field 1 = Open, Field 2 = High, Field 3 = Low
        assert chart_result["Open"] == 150.00
        assert chart_result["High"] == 152.00
        assert chart_result["Low"] == 149.50


class TestBarBucket:
    """Tests for bar bucket ID calculation."""

    def _create_mock_runner(self, bar_interval_ms=60000):
        """Create a mock runner for testing _bar_bucket."""
        from unittest.mock import MagicMock

        from core.schwab_runner import SchwabLiveRunner

        # Create instance without __init__
        runner = object.__new__(SchwabLiveRunner)

        # Mock the config attribute that _bar_bucket uses
        mock_config = MagicMock()
        mock_config.streaming.bar_interval_ms = bar_interval_ms
        runner.config = mock_config

        return runner

    def test_bar_bucket_60_second_default(self):
        """Test 60-second bar bucketing."""
        runner = self._create_mock_runner(bar_interval_ms=60000)

        # Two timestamps 30 seconds apart should be in same bucket
        ts1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 1, 12, 0, 30, tzinfo=timezone.utc)

        bucket1 = runner._bar_bucket(ts1)
        bucket2 = runner._bar_bucket(ts2)

        assert bucket1 == bucket2

    def test_bar_bucket_different_minutes(self):
        """Test that different minutes get different buckets."""
        runner = self._create_mock_runner(bar_interval_ms=60000)

        ts1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 1, 12, 1, 0, tzinfo=timezone.utc)

        bucket1 = runner._bar_bucket(ts1)
        bucket2 = runner._bar_bucket(ts2)

        assert bucket1 != bucket2
        assert bucket2 == bucket1 + 1  # Consecutive buckets

    def test_bar_bucket_custom_timeframe(self):
        """Test custom timeframe bucketing (e.g., 5-minute bars)."""
        runner = self._create_mock_runner(bar_interval_ms=60000)

        ts1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 1, 12, 4, 59, tzinfo=timezone.utc)
        ts3 = datetime(2024, 1, 1, 12, 5, 0, tzinfo=timezone.utc)

        # 5-minute bars (300 seconds) - use override parameter
        bucket1 = runner._bar_bucket(ts1, timeframe_sec=300)
        bucket2 = runner._bar_bucket(ts2, timeframe_sec=300)
        bucket3 = runner._bar_bucket(ts3, timeframe_sec=300)

        assert bucket1 == bucket2  # Same 5-min bucket
        assert bucket2 != bucket3  # Different 5-min buckets

    def test_bar_bucket_configurable_interval(self):
        """Test that bar interval from config is used."""
        # Create runner with 500ms (0.5 second) bars
        runner = self._create_mock_runner(bar_interval_ms=500)

        ts1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 1, 12, 0, 0, 400000, tzinfo=timezone.utc)  # +400ms
        ts3 = datetime(2024, 1, 1, 12, 0, 0, 600000, tzinfo=timezone.utc)  # +600ms

        bucket1 = runner._bar_bucket(ts1)
        bucket2 = runner._bar_bucket(ts2)
        bucket3 = runner._bar_bucket(ts3)

        assert bucket1 == bucket2  # Same 500ms bucket
        assert bucket1 != bucket3  # Different 500ms bucket


class TestBarAggregation:
    """Tests for quote-to-bar aggregation logic."""

    @pytest.fixture
    def mock_runner(self):
        """Create a minimal runner for testing aggregation."""
        with (
            patch("core.base.base_live_runner.get_event_handler"),
            patch("core.base.base_live_runner.get_module_logger"),
            patch("core.base.base_live_runner.FileTradeLogger"),
            patch("core.schwab_runner.SchwabClient"),
            patch("core.schwab_runner.SchwabBroker"),
            patch("core.base.base_live_runner.get_config"),
            patch("core.base.base_live_runner.StrategyRoutingManager"),
            patch("core.base.base_live_runner.LiveExecutor"),
            patch("core.base.base_live_runner.LiveExecutionEngine"),
            patch("core.base.base_live_runner.StateReconciler"),
            patch("core.base.base_live_runner.UnifiedDataPipeline"),
            patch("core.base.base_live_runner.HistoricalDataUpdater"),
            patch("core.base.base_live_runner.DynamicTradeLogicManager"),
        ):
            from core.schwab_runner import SchwabLiveRunner

            # Create runner with mocked dependencies
            runner = SchwabLiveRunner.__new__(SchwabLiveRunner)
            runner._bar_aggregation = {}

            # Add the method we're testing
            from collections import defaultdict

            runner._bar_aggregation = defaultdict(
                lambda: {"open": None, "high": None, "low": None, "close": None, "volume": 0, "bar_id": None}
            )

            return runner

    def test_aggregation_first_quote_starts_bar(self, mock_runner):
        """First quote should start a new bar."""
        result = mock_runner._aggregate_quote_to_bar("AAPL", 150.0, 1000, bar_id=100)

        assert result["Open"] == 150.0
        assert result["High"] == 150.0
        assert result["Low"] == 150.0
        assert result["Close"] == 150.0
        assert result["Volume"] == 1000
        assert result["bar_closed"] is False

    def test_aggregation_updates_ohlc(self, mock_runner):
        """Subsequent quotes should update OHLC correctly."""
        # First quote
        mock_runner._aggregate_quote_to_bar("AAPL", 150.0, 1000, bar_id=100)

        # Higher price - should update high
        result = mock_runner._aggregate_quote_to_bar("AAPL", 152.0, 500, bar_id=100)
        assert result["High"] == 152.0
        assert result["Open"] == 150.0  # Unchanged
        assert result["Close"] == 152.0

        # Lower price - should update low
        result = mock_runner._aggregate_quote_to_bar("AAPL", 149.0, 500, bar_id=100)
        assert result["Low"] == 149.0
        assert result["High"] == 152.0  # Unchanged
        assert result["Close"] == 149.0

    def test_aggregation_accumulates_volume(self, mock_runner):
        """Volume should accumulate across quotes in same bar."""
        mock_runner._aggregate_quote_to_bar("AAPL", 150.0, 1000, bar_id=100)
        mock_runner._aggregate_quote_to_bar("AAPL", 151.0, 500, bar_id=100)
        result = mock_runner._aggregate_quote_to_bar("AAPL", 152.0, 200, bar_id=100)

        assert result["Volume"] == 1700  # 1000 + 500 + 200

    def test_aggregation_new_bar_closes_previous(self, mock_runner):
        """New bar ID should close previous bar and start new one."""
        # First bar
        mock_runner._aggregate_quote_to_bar("AAPL", 150.0, 1000, bar_id=100)
        mock_runner._aggregate_quote_to_bar("AAPL", 152.0, 500, bar_id=100)

        # New bar - should close previous
        result = mock_runner._aggregate_quote_to_bar("AAPL", 153.0, 800, bar_id=101)

        assert result["bar_closed"] is True
        # Closed bar values (from bar 100)
        assert result["Open"] == 150.0
        assert result["High"] == 152.0
        assert result["Close"] == 152.0
        assert result["Volume"] == 1500

    def test_aggregation_after_bar_close(self, mock_runner):
        """After bar closes, next call should show new bar in progress."""
        # Build first bar
        mock_runner._aggregate_quote_to_bar("AAPL", 150.0, 1000, bar_id=100)

        # Close first bar and start second
        mock_runner._aggregate_quote_to_bar("AAPL", 153.0, 800, bar_id=101)

        # Continue second bar
        result = mock_runner._aggregate_quote_to_bar("AAPL", 154.0, 200, bar_id=101)

        assert result["bar_closed"] is False
        assert result["Open"] == 153.0  # New bar started at 153
        assert result["High"] == 154.0
        assert result["Volume"] == 1000  # 800 + 200

    def test_aggregation_multiple_symbols_independent(self, mock_runner):
        """Different symbols should have independent aggregation."""
        mock_runner._aggregate_quote_to_bar("AAPL", 150.0, 1000, bar_id=100)
        mock_runner._aggregate_quote_to_bar("MSFT", 350.0, 2000, bar_id=100)

        aapl_result = mock_runner._aggregate_quote_to_bar("AAPL", 152.0, 500, bar_id=100)
        msft_result = mock_runner._aggregate_quote_to_bar("MSFT", 352.0, 500, bar_id=100)

        # AAPL aggregation
        assert aapl_result["Open"] == 150.0
        assert aapl_result["High"] == 152.0
        assert aapl_result["Volume"] == 1500

        # MSFT aggregation - independent
        assert msft_result["Open"] == 350.0
        assert msft_result["High"] == 352.0
        assert msft_result["Volume"] == 2500


class TestPeriodicDataUpdate:
    """Tests for periodic data update scheduling."""

    @pytest.mark.asyncio
    async def test_periodic_update_sleeps_then_updates(self):
        """Test that periodic update waits for interval then fetches data."""
        from core.schwab_runner import SchwabLiveRunner

        # Create a mock runner
        runner = MagicMock(spec=SchwabLiveRunner)
        runner.symbols = ["AAPL", "MSFT"]
        runner.logger = MagicMock()

        # Mock the data pipeline
        runner.data_pipeline = AsyncMock()
        runner.data_pipeline.update_symbols = AsyncMock(return_value={"AAPL": 100, "MSFT": 100})

        # Bind the actual method to our mock
        runner._periodic_data_update = SchwabLiveRunner._periodic_data_update.__get__(runner, SchwabLiveRunner)

        # Run for a short time
        update_task = asyncio.create_task(runner._periodic_data_update(interval_minutes=0.001))  # 60ms

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Cancel and verify
        update_task.cancel()
        try:
            await update_task
        except asyncio.CancelledError:
            pass

        # Should have called update_symbols at least once
        runner.data_pipeline.update_symbols.assert_called()


class TestSchwabRunnerFieldMappingIntegration:
    """Integration tests verifying field mappings work end-to-end."""

    def test_field_1_is_bid_not_last(self):
        """Verify field 1 is bid price (common mistake is treating it as last)."""
        from core.schwab_runner import SchwabLiveRunner

        quote = {
            "1": 100.00,  # If this were last price, Close would be 100
            "2": 101.00,
            "3": 100.50,  # This is actually the last price
        }

        result = SchwabLiveRunner._canonicalize_schwab_quote(quote, "TEST")

        # Close should be field 3 (last price), NOT field 1 (bid)
        assert result["Close"] == 100.50
        assert result["Close"] != 100.00

    def test_field_3_is_last_not_low(self):
        """Verify field 3 is last price, not low (important distinction)."""
        from core.schwab_runner import SchwabLiveRunner

        quote = {
            "3": 150.27,  # Last price (LEVELONE)
            "11": 149.00,  # Low price
        }

        result = SchwabLiveRunner._canonicalize_schwab_quote(quote, "TEST")

        assert result["Close"] == 150.27  # Field 3 = Last
        assert result["Low"] == 149.00  # Field 11 = Low
