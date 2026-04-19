"""
Test suite for the Aggregator class.

Tests data aggregation, storage, and retrieval operations.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd


class TestAggregatorInit(unittest.TestCase):
    """Test Aggregator initialization."""

    @patch("data.aggregate.SchwabClient")
    @patch("data.aggregate.get_logs_path", return_value="/tmp/logs")
    @patch("data.aggregate.Authenticator")
    @patch("data.aggregate.DataStore")
    @patch("data.aggregate.Logger")
    @patch("data.aggregate.FileWriter")
    @patch("data.aggregate.CacheManager")
    def test_init(self, mock_cache, mock_writer, mock_logger, mock_store, mock_auth, mock_logs_path, mock_client):
        """Aggregator should initialize with required dependencies."""
        from data.aggregate import Aggregator

        agg = Aggregator(apikey="test_key", secret="test_secret")

        mock_client.assert_called_once_with("test_key", "test_secret")
        self.assertIsNotNone(agg.session)


class TestIsStockOutdated(unittest.TestCase):
    """Test stock outdated detection."""

    @patch("data.aggregate.SchwabClient")
    @patch("data.aggregate.get_logs_path", return_value="/tmp/logs")
    @patch("data.aggregate.Authenticator")
    @patch("data.aggregate.DataStore")
    @patch("data.aggregate.Logger")
    @patch("data.aggregate.FileWriter")
    @patch("data.aggregate.CacheManager")
    def setUp(self, mock_cache, mock_writer, mock_logger, mock_store, mock_auth, mock_logs_path, mock_client):
        mock_logger.return_value.get_logger.return_value = MagicMock()

        from data.aggregate import Aggregator

        self.aggregator = Aggregator(apikey="test_key", secret="test_secret")

    def test_no_timestamp_is_outdated(self):
        """No timestamp should be considered outdated."""
        result = self.aggregator.is_stock_outdated(None)
        self.assertTrue(result)

    def test_old_timestamp_is_outdated(self):
        """Timestamp from last week should be outdated."""
        # 7 days ago in milliseconds
        old_timestamp = int((datetime.utcnow() - timedelta(days=7)).timestamp() * 1000)
        result = self.aggregator.is_stock_outdated(old_timestamp)
        self.assertTrue(result)

    def test_recent_timestamp_not_outdated(self):
        """Timestamp from today should not be outdated."""
        # Current time in milliseconds
        recent_timestamp = int(datetime.utcnow().timestamp() * 1000)
        today = pd.Timestamp.utcnow().normalize().tz_localize(None)
        result = self.aggregator.is_stock_outdated(recent_timestamp, today)
        self.assertFalse(result)


class TestRawDataStore(unittest.TestCase):
    """Test raw data storage operations."""

    @patch("data.aggregate.SchwabClient")
    @patch("data.aggregate.get_logs_path", return_value="/tmp/logs")
    @patch("data.aggregate.Authenticator")
    @patch("data.aggregate.DataStore")
    @patch("data.aggregate.Logger")
    @patch("data.aggregate.FileWriter")
    @patch("data.aggregate.CacheManager")
    def setUp(self, mock_cache, mock_writer, mock_logger, mock_store, mock_auth, mock_logs_path, mock_client):
        mock_logger.return_value.get_logger.return_value = MagicMock()
        mock_auth.return_value.program_path = "/tmp"

        from data.aggregate import Aggregator

        self.aggregator = Aggregator(apikey="test_key", secret="test_secret")
        self.mock_cache = mock_cache
        self.mock_writer = mock_writer
        self.mock_client = mock_client

    def test_up_to_date_returns_no_update(self):
        """Should return 'No Update Needed' if data is current."""
        # Mock cache to return recent timestamp
        self.aggregator.cache.get_last_processed_date.return_value = int(datetime.utcnow().timestamp() * 1000)
        self.aggregator.is_stock_outdated = MagicMock(return_value=False)

        result = self.aggregator.raw_data_store("AAPL")
        self.assertEqual(result, "No Update Needed")

    def test_no_candles_returns_no_update(self):
        """Should return 'No Update Needed' if API returns no candles."""
        self.aggregator.cache.get_last_processed_date.return_value = None
        self.aggregator.session.daily_price_history.return_value = {"candles": []}

        result = self.aggregator.raw_data_store("AAPL")
        self.assertEqual(result, "No Update Needed")

    @patch("os.path.exists", return_value=False)
    def test_new_file_created(self, mock_exists):
        """Should create new file when none exists."""
        self.aggregator.cache.get_last_processed_date.return_value = None
        self.aggregator.session.daily_price_history.return_value = {
            "candles": [{"datetime": 1672531200000, "open": 150.0, "close": 152.0}]
        }

        result = self.aggregator.raw_data_store("AAPL")

        self.assertEqual(result, "File Generated!")
        self.aggregator.writer.write_json.assert_called_once()


class TestStoreProcessedData(unittest.TestCase):
    """Test processed data storage."""

    @patch("data.aggregate.SchwabClient")
    @patch("data.aggregate.get_logs_path", return_value="/tmp/logs")
    @patch("data.aggregate.Authenticator")
    @patch("data.aggregate.DataStore")
    @patch("data.aggregate.Logger")
    @patch("data.aggregate.FileWriter")
    @patch("data.aggregate.CacheManager")
    def setUp(self, mock_cache, mock_writer, mock_logger, mock_store, mock_auth, mock_logs_path, mock_client):
        mock_logger.return_value.get_logger.return_value = MagicMock()

        from data.aggregate import Aggregator

        self.aggregator = Aggregator(apikey="test_key", secret="test_secret")

    def test_store_processed_data_success(self):
        """Should store processed data in database."""
        test_data = pd.DataFrame({"datetime": [1672531200000], "open": [150.0], "close": [152.0]})

        # Mock the datastore context manager
        mock_store_instance = MagicMock()
        self.aggregator.datastore.__enter__ = MagicMock(return_value=mock_store_instance)
        self.aggregator.datastore.__exit__ = MagicMock(return_value=False)

        result = self.aggregator.store_processed_data("AAPL", test_data)

        self.assertEqual(result, "Processed data stored!")

    def test_store_empty_dataframe(self):
        """Should handle empty dataframe gracefully."""
        test_data = pd.DataFrame()

        # Should not raise, just log warning
        self.aggregator.store_processed_data_files("AAPL", test_data)


class TestFetchData(unittest.TestCase):
    """Test data fetching from database."""

    @patch("data.aggregate.SchwabClient")
    @patch("data.aggregate.get_logs_path", return_value="/tmp/logs")
    @patch("data.aggregate.Authenticator")
    @patch("data.aggregate.DataStore")
    @patch("data.aggregate.Logger")
    @patch("data.aggregate.FileWriter")
    @patch("data.aggregate.CacheManager")
    def setUp(self, mock_cache, mock_writer, mock_logger, mock_store, mock_auth, mock_logs_path, mock_client):
        mock_logger.return_value.get_logger.return_value = MagicMock()

        from data.aggregate import Aggregator

        self.aggregator = Aggregator(apikey="test_key", secret="test_secret")

    def test_fetch_data_returns_dataframe(self):
        """Should return DataFrame from database."""
        mock_df = pd.DataFrame({"symbol": ["AAPL"], "close": [150.0]})
        self.aggregator.datastore.get_data_by_symbol.return_value = mock_df

        result = self.aggregator.fetch_data("AAPL")

        self.assertIsInstance(result, pd.DataFrame)

    def test_fetch_data_error_returns_empty(self):
        """Should return empty DataFrame on error."""
        self.aggregator.datastore.get_data_by_symbol.side_effect = Exception("DB error")

        result = self.aggregator.fetch_data("AAPL")

        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
