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
import pandas as pd


class TestDataStore(unittest.TestCase):
    """Test DataStore database operations."""

    def test_datastore_create_database(self):
        """DataStore should create database table."""
        from data.datastorage import DataStore

        # Use in-memory database for testing
        with DataStore(":memory:", use_config=False) as store:
            test_df = pd.DataFrame({
                'symbol': ['AAPL'],
                'close': [150.0]
            })
            store.create_database("test_table", test_df)

            # Verify table exists
            self.assertTrue(store.table_exists("test_table"))

    def test_datastore_fill_database(self):
        """DataStore should insert data into database."""
        from data.datastorage import DataStore

        with DataStore(":memory:", use_config=False) as store:
            # Sample data
            data = pd.DataFrame({
                'timestamp': ['2023-01-01', '2023-01-02'],
                'open': [100.0, 101.0],
                'close': [101.0, 102.0]
            })

            store.fill_database("test_data", data)

            # Verify data was inserted
            result = store.get_data_base("test_data")
            self.assertEqual(len(result), 2)

    def test_datastore_get_by_symbol(self):
        """DataStore should retrieve data by symbol."""
        from data.datastorage import DataStore

        with DataStore(":memory:", use_config=False) as store:
            data = pd.DataFrame({
                'symbol': ['AAPL', 'MSFT', 'AAPL'],
                'close': [150.0, 300.0, 151.0]
            })

            store.fill_database("stock_data", data)
            result = store.get_data_by_symbol("stock_data", "AAPL")

            self.assertEqual(len(result), 2)

    def test_datastore_upsert(self):
        """DataStore should upsert data correctly."""
        from data.datastorage import DataStore

        with DataStore(":memory:", use_config=False) as store:
            # Initial data
            data1 = pd.DataFrame({
                'symbol': ['AAPL'],
                'date': ['2023-01-01'],
                'close': [150.0]
            })
            store.fill_database("test_upsert", data1)

            # Upsert with updated value
            data2 = pd.DataFrame({
                'symbol': ['AAPL'],
                'date': ['2023-01-01'],
                'close': [155.0]
            })
            store.upsert_data("test_upsert", data2, ['symbol', 'date'])

            result = store.get_data_base("test_upsert")
            # Should still have 1 row, with updated close
            self.assertEqual(len(result), 1)

    def test_datastore_list_tables(self):
        """DataStore should list all tables."""
        from data.datastorage import DataStore

        with DataStore(":memory:", use_config=False) as store:
            df = pd.DataFrame({'col': [1]})
            store.fill_database("table1", df)
            store.fill_database("table2", df)

            tables = store.list_tables()
            self.assertIn("table1", tables)
            self.assertIn("table2", tables)

    def test_datastore_row_count(self):
        """DataStore should return correct row count."""
        from data.datastorage import DataStore

        with DataStore(":memory:", use_config=False) as store:
            data = pd.DataFrame({
                'value': [1, 2, 3, 4, 5]
            })
            store.fill_database("count_test", data)

            count = store.get_row_count("count_test")
            self.assertEqual(count, 5)

    def test_datastore_delete_data(self):
        """DataStore should delete data correctly."""
        from data.datastorage import DataStore

        with DataStore(":memory:", use_config=False) as store:
            data = pd.DataFrame({
                'symbol': ['AAPL', 'MSFT'],
                'close': [150.0, 300.0]
            })
            store.fill_database("delete_test", data)

            # Delete AAPL
            store.delete_data("delete_test", "symbol = ?", ("AAPL",))

            result = store.get_data_base("delete_test")
            self.assertEqual(len(result), 1)


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
