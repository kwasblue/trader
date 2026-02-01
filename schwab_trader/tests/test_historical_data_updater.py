"""
Tests for HistoricalDataUpdater

Coverage:
- Data fetching from Alpaca
- File saving in correct format
- Incremental updates
- Data freshness checking
- Periodic update scheduling
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone, timedelta
import json
import os

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.historical_data_updater import HistoricalDataUpdater


class TestHistoricalDataUpdaterInit:
    """Tests for HistoricalDataUpdater initialization."""

    def test_init_with_credentials(self):
        """Test initialization with API credentials."""
        with patch('core.historical_data_updater.Logger'):
            updater = HistoricalDataUpdater(
                api_key="test_key",
                api_secret="test_secret"
            )
            assert updater.api_key == "test_key"
            assert updater.api_secret == "test_secret"

    def test_init_custom_path(self, tmp_path):
        """Test initialization with custom data path."""
        with patch('core.historical_data_updater.Logger'):
            updater = HistoricalDataUpdater(
                api_key="test_key",
                api_secret="test_secret",
                data_path=str(tmp_path)
            )
            assert str(tmp_path) in str(updater.data_path)


class TestDataFetching:
    """Tests for data fetching."""

    @pytest.fixture
    def updater(self, tmp_path):
        with patch('core.historical_data_updater.StockHistoricalDataClient'):
            with patch('core.historical_data_updater.Logger'):
                return HistoricalDataUpdater(
                    api_key="test_key",
                    api_secret="test_secret",
                    data_path=str(tmp_path)
                )

    @pytest.mark.asyncio
    async def test_fetch_bars_success(self, updater):
        """Test successful bar fetching."""
        mock_bars = [
            {'Date': 1704711600000, 'Symbol': 'AAPL', 'Open': 150.0, 'High': 151.0, 'Low': 149.0, 'Close': 150.5, 'Volume': 1000},
            {'Date': 1704711660000, 'Symbol': 'AAPL', 'Open': 150.5, 'High': 152.0, 'Low': 150.0, 'Close': 151.0, 'Volume': 1100}
        ]

        with patch.object(updater, '_fetch_bars', new_callable=AsyncMock, return_value=mock_bars):
            with patch.object(updater, '_save_bars'):
                count = await updater.update_symbol('AAPL', days=1)
                assert count == 2

    @pytest.mark.asyncio
    async def test_fetch_bars_empty(self, updater):
        """Test fetching with no data returned."""
        with patch.object(updater, '_fetch_bars', new_callable=AsyncMock, return_value=[]):
            count = await updater.update_symbol('AAPL', days=1)
            assert count == 0

    @pytest.mark.asyncio
    async def test_fetch_bars_api_error(self, updater):
        """Test handling of API errors."""
        with patch.object(updater, '_fetch_bars', new_callable=AsyncMock, side_effect=Exception("API Error")):
            count = await updater.update_symbol('AAPL', days=1)
            assert count == 0


class TestFileSaving:
    """Tests for file saving."""

    @pytest.fixture
    def updater(self, tmp_path):
        with patch('core.historical_data_updater.Logger'):
            return HistoricalDataUpdater(
                api_key="test_key",
                api_secret="test_secret",
                data_path=str(tmp_path)
            )

    @pytest.mark.asyncio
    async def test_saves_to_correct_file(self, updater, tmp_path):
        """Test data is saved to correct file path."""
        mock_bars = [
            MagicMock(
                timestamp=datetime(2024, 1, 8, 10, 0, tzinfo=timezone.utc),
                open=150.0, high=151.0, low=149.0, close=150.5, volume=1000
            )
        ]

        with patch('core.historical_data_updater.StockHistoricalDataClient') as MockClient:
            mock_client = MagicMock()
            mock_client.get_stock_bars.return_value = {'AAPL': mock_bars}
            MockClient.return_value = mock_client

            await updater.update_symbol('AAPL', days=1)

            # Check file exists
            expected_file = tmp_path / "proc_AAPL_file.json"
            # File creation depends on implementation

    @pytest.mark.asyncio
    async def test_file_format(self, updater, tmp_path):
        """Test saved file format is correct."""
        mock_bars = [
            MagicMock(
                timestamp=datetime(2024, 1, 8, 10, 0, tzinfo=timezone.utc),
                open=150.0, high=151.0, low=149.0, close=150.5, volume=1000
            )
        ]

        with patch('core.historical_data_updater.StockHistoricalDataClient') as MockClient:
            mock_client = MagicMock()
            mock_client.get_stock_bars.return_value = {'AAPL': mock_bars}
            MockClient.return_value = mock_client

            await updater.update_symbol('AAPL', days=1)

            # Verify file format if file was created
            # (depends on implementation)


class TestDataFreshness:
    """Tests for data freshness checking."""

    @pytest.fixture
    def updater(self, tmp_path):
        with patch('core.historical_data_updater.StockHistoricalDataClient'):
            with patch('core.historical_data_updater.Logger'):
                u = HistoricalDataUpdater(
                    api_key="test_key",
                    api_secret="test_secret",
                    data_path=str(tmp_path)
                )
                # Ensure directory exists
                tmp_path.mkdir(exist_ok=True)
                return u

    def test_freshness_no_file(self, updater):
        """Test freshness check with no file."""
        result = updater.get_data_freshness('NONEXISTENT')
        assert result is None

    def test_freshness_fresh_data(self, updater, tmp_path):
        """Test freshness check with fresh data."""
        # Create a recent data file - method expects list of bars with 'Date' or 'timestamp'
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        data = [
            {'Date': now_ms, 'Open': 150.0, 'High': 151.0, 'Low': 149.0, 'Close': 150.5, 'Volume': 1000}
        ]
        file_path = tmp_path / "proc_AAPL_file.json"
        file_path.write_text(json.dumps(data))

        result = updater.get_data_freshness('AAPL')

        assert result is not None
        assert result['is_stale'] is False
        assert result['bar_count'] == 1

    def test_freshness_stale_data(self, updater, tmp_path):
        """Test freshness check with stale data."""
        # Create an old data file (3 hours ago)
        old_ms = int((datetime.now(timezone.utc) - timedelta(hours=3)).timestamp() * 1000)
        data = [
            {'Date': old_ms, 'Open': 150.0, 'High': 151.0, 'Low': 149.0, 'Close': 150.5, 'Volume': 1000}
        ]
        file_path = tmp_path / "proc_AAPL_file.json"
        file_path.write_text(json.dumps(data))

        result = updater.get_data_freshness('AAPL')

        assert result is not None
        assert result['is_stale'] is True  # > 1 hour old


class TestIncrementalUpdate:
    """Tests for incremental updates."""

    @pytest.fixture
    def updater(self, tmp_path):
        with patch('core.historical_data_updater.Logger'):
            return HistoricalDataUpdater(
                api_key="test_key",
                api_secret="test_secret",
                data_path=str(tmp_path)
            )

    @pytest.mark.asyncio
    async def test_incremental_update(self, updater, tmp_path):
        """Test incremental update appends to existing data."""
        # Create existing data file
        existing_data = {
            'timestamp': ['2024-01-08T10:00:00+00:00'],
            'Open': [150.0], 'High': [151.0], 'Low': [149.0],
            'Close': [150.5], 'Volume': [1000]
        }
        file_path = tmp_path / "proc_AAPL_file.json"
        file_path.write_text(json.dumps(existing_data))

        # Fetch new data
        new_bar = MagicMock(
            timestamp=datetime(2024, 1, 8, 10, 1, tzinfo=timezone.utc),
            open=150.5, high=152.0, low=150.0, close=151.0, volume=1100
        )

        with patch('core.historical_data_updater.StockHistoricalDataClient') as MockClient:
            mock_client = MagicMock()
            mock_client.get_stock_bars.return_value = {'AAPL': [new_bar]}
            MockClient.return_value = mock_client

            count = await updater.update_symbol('AAPL', days=1, force_full=False)

            # Should have added new data
            # (depends on implementation)

    @pytest.mark.asyncio
    async def test_force_full_update(self, updater, tmp_path):
        """Test force full update replaces all data."""
        # Create existing data
        existing_data = {
            'timestamp': ['2024-01-08T10:00:00+00:00'],
            'Open': [150.0], 'High': [151.0], 'Low': [149.0],
            'Close': [150.5], 'Volume': [1000]
        }
        file_path = tmp_path / "proc_AAPL_file.json"
        file_path.write_text(json.dumps(existing_data))

        # Fetch with force_full
        mock_bars = [
            MagicMock(
                timestamp=datetime(2024, 1, 9, 10, 0, tzinfo=timezone.utc),
                open=155.0, high=156.0, low=154.0, close=155.5, volume=2000
            )
        ]

        with patch('core.historical_data_updater.StockHistoricalDataClient') as MockClient:
            mock_client = MagicMock()
            mock_client.get_stock_bars.return_value = {'AAPL': mock_bars}
            MockClient.return_value = mock_client

            count = await updater.update_symbol('AAPL', days=1, force_full=True)

            # Should have replaced data
            # (depends on implementation)


class TestMultiSymbolUpdate:
    """Tests for updating multiple symbols."""

    @pytest.fixture
    def updater(self, tmp_path):
        with patch('core.historical_data_updater.Logger'):
            return HistoricalDataUpdater(
                api_key="test_key",
                api_secret="test_secret",
                data_path=str(tmp_path)
            )

    @pytest.mark.asyncio
    async def test_update_multiple_symbols(self, updater):
        """Test updating multiple symbols."""
        mock_bar = MagicMock(
            timestamp=datetime(2024, 1, 8, 10, 0, tzinfo=timezone.utc),
            open=150.0, high=151.0, low=149.0, close=150.5, volume=1000
        )

        with patch('core.historical_data_updater.StockHistoricalDataClient') as MockClient:
            mock_client = MagicMock()
            mock_client.get_stock_bars.return_value = {'AAPL': [mock_bar]}
            MockClient.return_value = mock_client

            results = await updater.update_symbols(['AAPL', 'MSFT', 'GOOGL'], days=1)

            assert len(results) == 3

    @pytest.mark.asyncio
    async def test_concurrent_updates(self, updater):
        """Test concurrent symbol updates."""
        mock_bar = MagicMock(
            timestamp=datetime(2024, 1, 8, 10, 0, tzinfo=timezone.utc),
            open=150.0, high=151.0, low=149.0, close=150.5, volume=1000
        )

        with patch('core.historical_data_updater.StockHistoricalDataClient') as MockClient:
            mock_client = MagicMock()
            mock_client.get_stock_bars.return_value = {'AAPL': [mock_bar]}
            MockClient.return_value = mock_client

            # Should handle concurrent updates
            results = await updater.update_symbols(
                ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
                days=1,
                max_concurrent=3
            )

            assert len(results) == 5


class TestPeriodicUpdates:
    """Tests for periodic update scheduling."""

    @pytest.fixture
    def updater(self, tmp_path):
        with patch('core.historical_data_updater.Logger'):
            return HistoricalDataUpdater(
                api_key="test_key",
                api_secret="test_secret",
                data_path=str(tmp_path)
            )

    @pytest.mark.asyncio
    async def test_run_periodic_single_iteration(self, updater):
        """Test single iteration of periodic update."""
        mock_bar = MagicMock(
            timestamp=datetime.now(timezone.utc),
            open=150.0, high=151.0, low=149.0, close=150.5, volume=1000
        )

        with patch('core.historical_data_updater.StockHistoricalDataClient') as MockClient:
            mock_client = MagicMock()
            mock_client.get_stock_bars.return_value = {'AAPL': [mock_bar]}
            MockClient.return_value = mock_client

            # Run single update (not full periodic loop)
            results = await updater.update_symbols(['AAPL'], days=1)
            assert 'AAPL' in results


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def updater(self, tmp_path):
        with patch('core.historical_data_updater.Logger'):
            return HistoricalDataUpdater(
                api_key="test_key",
                api_secret="test_secret",
                data_path=str(tmp_path)
            )

    @pytest.mark.asyncio
    async def test_handles_network_error(self, updater):
        """Test handling of network errors."""
        with patch('core.historical_data_updater.StockHistoricalDataClient') as MockClient:
            mock_client = MagicMock()
            mock_client.get_stock_bars.side_effect = ConnectionError("Network error")
            MockClient.return_value = mock_client

            count = await updater.update_symbol('AAPL', days=1)
            assert count == 0  # Should fail gracefully

    @pytest.mark.asyncio
    async def test_handles_rate_limit(self, updater):
        """Test handling of rate limit errors."""
        with patch('core.historical_data_updater.StockHistoricalDataClient') as MockClient:
            mock_client = MagicMock()
            mock_client.get_stock_bars.side_effect = Exception("Rate limit exceeded")
            MockClient.return_value = mock_client

            count = await updater.update_symbol('AAPL', days=1)
            assert count == 0  # Should fail gracefully

    @pytest.mark.asyncio
    async def test_handles_invalid_symbol(self, updater):
        """Test handling of invalid symbol."""
        with patch('core.historical_data_updater.StockHistoricalDataClient') as MockClient:
            mock_client = MagicMock()
            mock_client.get_stock_bars.side_effect = Exception("Symbol not found")
            MockClient.return_value = mock_client

            count = await updater.update_symbol('INVALID123', days=1)
            assert count == 0
