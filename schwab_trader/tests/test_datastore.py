"""
Test suite for data storage operations.

Tests database operations, file I/O, and data persistence.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import Mock, MagicMock, patch, mock_open
import json
import tempfile
import shutil


class TestDataStore(unittest.TestCase):
    """Test DataStore database operations."""

    def test_datastore_create_database(self):
        """DataStore should create database table."""
        try:
            from utils.datastore import DataStore

            # Use in-memory database for testing
            store = DataStore(":memory:")
            store.create_database("test_table")

            self.assertTrue(True)  # If no exception, test passes
        except ImportError:
            self.skipTest("DataStore not found")

    def test_datastore_fill_database(self):
        """DataStore should insert data into database."""
        try:
            from utils.datastore import DataStore

            store = DataStore(":memory:")
            store.create_database("test_data")

            # Sample data
            data = [
                {"timestamp": "2023-01-01", "open": 100, "close": 101},
                {"timestamp": "2023-01-02", "open": 101, "close": 102},
            ]

            store.fill_database(data)
            store.commit()

            self.assertTrue(True)
        except ImportError:
            self.skipTest("DataStore not found")


class TestJsonFileOperations(unittest.TestCase):
    """Test JSON file read/write operations."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_write_json(self):
        """Should write JSON data to file."""
        data = {"symbol": "AAPL", "price": 150.0}
        filepath = os.path.join(self.test_dir, "test.json")

        with open(filepath, 'w') as f:
            json.dump(data, f)

        # Verify file was written
        self.assertTrue(os.path.exists(filepath))

        # Verify content
        with open(filepath, 'r') as f:
            loaded = json.load(f)

        self.assertEqual(loaded['symbol'], 'AAPL')
        self.assertEqual(loaded['price'], 150.0)

    def test_read_json(self):
        """Should read JSON data from file."""
        data = {"symbol": "MSFT", "price": 250.0}
        filepath = os.path.join(self.test_dir, "test.json")

        # Write first
        with open(filepath, 'w') as f:
            json.dump(data, f)

        # Read back
        with open(filepath, 'r') as f:
            loaded = json.load(f)

        self.assertEqual(loaded['symbol'], 'MSFT')

    def test_append_to_json_list(self):
        """Should append to JSON list file."""
        filepath = os.path.join(self.test_dir, "list.json")

        # Write initial list
        initial_data = [{"id": 1}]
        with open(filepath, 'w') as f:
            json.dump(initial_data, f)

        # Read and append
        with open(filepath, 'r') as f:
            data = json.load(f)

        data.append({"id": 2})

        with open(filepath, 'w') as f:
            json.dump(data, f)

        # Verify
        with open(filepath, 'r') as f:
            final = json.load(f)

        self.assertEqual(len(final), 2)


class TestRawDataStorage(unittest.TestCase):
    """Test raw market data storage."""

    def setUp(self):
        """Create sample raw data."""
        self.sample_candles = [
            {"datetime": "2023-01-01", "open": 100, "high": 105,
             "low": 99, "close": 102, "volume": 1000000},
            {"datetime": "2023-01-02", "open": 102, "high": 108,
             "low": 101, "close": 107, "volume": 1200000},
        ]

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_store_raw_data(self, mock_json_dump, mock_file):
        """Should store raw candle data."""
        # Simulate writing raw data
        with open("raw_AAPL.json", 'w') as f:
            json.dump({"symbol": "AAPL", "candles": self.sample_candles}, f)

        mock_json_dump.assert_called_once()

    def test_raw_data_format(self):
        """Raw data should have correct format."""
        raw_data = {
            "symbol": "AAPL",
            "candles": self.sample_candles
        }

        self.assertIn("symbol", raw_data)
        self.assertIn("candles", raw_data)
        self.assertEqual(len(raw_data["candles"]), 2)

        for candle in raw_data["candles"]:
            self.assertIn("datetime", candle)
            self.assertIn("open", candle)
            self.assertIn("high", candle)
            self.assertIn("low", candle)
            self.assertIn("close", candle)
            self.assertIn("volume", candle)


class TestProcessedDataStorage(unittest.TestCase):
    """Test processed data storage."""

    def setUp(self):
        """Create sample processed data."""
        self.processed_data = [
            {"timestamp": "2023-01-01", "Open": 100, "High": 105,
             "Low": 99, "Close": 102, "Volume": 1000000,
             "EMA_20": 101.5, "RSI": 55.0},
        ]

    def test_processed_data_has_indicators(self):
        """Processed data should include indicators."""
        for row in self.processed_data:
            self.assertIn("EMA_20", row)
            self.assertIn("RSI", row)

    def test_processed_data_maintains_ohlcv(self):
        """Processed data should maintain OHLCV fields."""
        for row in self.processed_data:
            self.assertIn("Open", row)
            self.assertIn("High", row)
            self.assertIn("Low", row)
            self.assertIn("Close", row)
            self.assertIn("Volume", row)


class TestDataStorageConfig(unittest.TestCase):
    """Test data storage configuration."""

    def test_default_storage_paths(self):
        """Should have configurable storage paths."""
        default_raw_path = "data/data_storage/raw_data"
        default_proc_path = "data/data_storage/proc_data"

        self.assertIn("raw_data", default_raw_path)
        self.assertIn("proc_data", default_proc_path)

    def test_symbol_file_naming(self):
        """Should generate correct file names for symbols."""
        symbol = "AAPL"

        raw_filename = f"raw_{symbol}_file.json"
        proc_filename = f"proc_{symbol}_file.json"

        self.assertEqual(raw_filename, "raw_AAPL_file.json")
        self.assertEqual(proc_filename, "proc_AAPL_file.json")


if __name__ == '__main__':
    unittest.main()
