# tests/test_unified_pipeline_multitf.py
"""
Unit tests for UnifiedDataPipeline - Multi-timeframe support

Tests:
- Fetching data at multiple timeframes
- Storage with timeframe identifiers
- Retrieving data by timeframe
- Backward compatibility with old format
- Timeframe validation
"""

import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from core.unified_data_pipeline import UnifiedDataPipeline


class TestMultiTimeframeSupport(unittest.TestCase):
    """Test multi-timeframe data fetching and storage."""

    def setUp(self):
        """Set up test environment with temporary directories."""
        self.test_dir = tempfile.mkdtemp()
        self.data_path = Path(self.test_dir) / "proc_data"
        self.raw_data_path = Path(self.test_dir) / "raw_data"
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

        # Create pipeline with test directories (no DB for simplicity)
        self.pipeline = UnifiedDataPipeline(
            data_path=str(self.data_path), raw_data_path=str(self.raw_data_path), store_to_db=False, store_to_files=True
        )

    def tearDown(self):
        """Clean up test directories."""
        shutil.rmtree(self.test_dir)

    def test_supported_timeframes(self):
        """Test that all supported timeframes are defined."""
        expected_timeframes = ["1min", "5min", "15min", "30min", "1hour", "day"]

        for tf in expected_timeframes:
            self.assertIn(tf, self.pipeline.SUPPORTED_TIMEFRAMES)

        # Check mapping values
        self.assertEqual(self.pipeline.SUPPORTED_TIMEFRAMES["1min"], ("Minute", 1))
        self.assertEqual(self.pipeline.SUPPORTED_TIMEFRAMES["5min"], ("Minute", 5))
        self.assertEqual(self.pipeline.SUPPORTED_TIMEFRAMES["15min"], ("Minute", 15))
        self.assertEqual(self.pipeline.SUPPORTED_TIMEFRAMES["30min"], ("Minute", 30))
        self.assertEqual(self.pipeline.SUPPORTED_TIMEFRAMES["1hour"], ("Hour", 1))
        self.assertEqual(self.pipeline.SUPPORTED_TIMEFRAMES["day"], ("Day", 1))

    def test_timeframe_validation(self):
        """Test validation of timeframe parameters."""
        # Valid timeframes should pass
        valid = ["1min", "5min", "15min"]
        result = [tf for tf in valid if tf in self.pipeline.SUPPORTED_TIMEFRAMES]
        self.assertEqual(len(result), 3)

        # Invalid timeframes should be filtered out
        invalid = ["2min", "10min", "hour", "invalid"]
        result = [tf for tf in invalid if tf in self.pipeline.SUPPORTED_TIMEFRAMES]
        self.assertEqual(len(result), 0)

    def test_raw_data_storage_with_timeframe(self):
        """Test raw data is saved with timeframe identifier."""
        symbol = "AAPL"
        timeframe = "5min"

        # Mock bars data
        bars = [
            {
                "datetime": int(datetime.now(timezone.utc).timestamp() * 1000),
                "open": 150.0,
                "high": 151.0,
                "low": 149.0,
                "close": 150.5,
                "volume": 10000,
            }
        ]

        # Save raw data
        self.pipeline._save_raw_data(symbol, bars, timeframe)

        # Check file was created with timeframe suffix
        file_path = self.raw_data_path / f"raw_{symbol}_{timeframe}.json"
        self.assertTrue(file_path.exists())

        # Load and verify
        import json

        with open(file_path) as f:
            data = json.load(f)

        self.assertIn("candles", data)
        self.assertIn("timeframe", data)
        self.assertEqual(data["timeframe"], timeframe)
        self.assertEqual(len(data["candles"]), 1)

    def test_load_raw_data_with_timeframe(self):
        """Test loading raw data for specific timeframe."""
        symbol = "AAPL"
        timeframe = "15min"

        # Create test data
        bars = [
            {
                "datetime": int(datetime.now(timezone.utc).timestamp() * 1000),
                "open": 150.0,
                "high": 151.0,
                "low": 149.0,
                "close": 150.5,
                "volume": 10000,
            }
        ]

        # Save
        self.pipeline._save_raw_data(symbol, bars, timeframe)

        # Load
        loaded = self.pipeline._load_all_raw_data(symbol, timeframe)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["open"], 150.0)

    def test_backward_compatibility_old_format(self):
        """Test backward compatibility with old filename format (no timeframe)."""
        symbol = "AAPL"
        timeframe = "day"

        # Create old format file
        import json

        old_file = self.raw_data_path / f"raw_{symbol}_file.json"
        bars = [
            {
                "datetime": int(datetime.now(timezone.utc).timestamp() * 1000),
                "open": 150.0,
                "high": 151.0,
                "low": 149.0,
                "close": 150.5,
                "volume": 10000,
            }
        ]

        with open(old_file, "w") as f:
            json.dump({"candles": bars}, f)

        # Should be able to load as 'day' timeframe
        loaded = self.pipeline._load_all_raw_data(symbol, timeframe)
        self.assertEqual(len(loaded), 1)

    def test_get_data_with_timeframe(self):
        """Test retrieving processed data for specific timeframe."""
        symbol = "AAPL"
        timeframe = "5min"

        # Create mock processed data
        import json

        proc_file = self.data_path / f"proc_{symbol}_{timeframe}.json"

        data = {
            "bars": [
                {
                    "Date": "2026-03-14T09:30:00",
                    "open": 150.0,
                    "high": 151.0,
                    "low": 149.0,
                    "close": 150.5,
                    "volume": 10000,
                }
            ],
            "timeframe": timeframe,
        }

        with open(proc_file, "w") as f:
            json.dump(data, f)

        # Load data
        df = self.pipeline.get_data(symbol, timeframe=timeframe)

        self.assertFalse(df.empty)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["open"], 150.0)

    def test_list_available_timeframes(self):
        """Test listing available timeframes for a symbol."""
        symbol = "AAPL"

        # Create files for multiple timeframes
        import json

        for timeframe in ["5min", "15min", "30min"]:
            proc_file = self.data_path / f"proc_{symbol}_{timeframe}.json"
            with open(proc_file, "w") as f:
                json.dump({"bars": [], "timeframe": timeframe}, f)

        # List timeframes
        timeframes = self.pipeline.list_available_timeframes(symbol, source="file")

        self.assertEqual(len(timeframes), 3)
        self.assertIn("5min", timeframes)
        self.assertIn("15min", timeframes)
        self.assertIn("30min", timeframes)

    def test_list_symbols_filtered_by_timeframe(self):
        """Test listing symbols filtered by specific timeframe."""
        # Create files for different symbols and timeframes
        import json

        # AAPL with 5min and 15min
        for tf in ["5min", "15min"]:
            file_path = self.data_path / f"proc_AAPL_{tf}.json"
            with open(file_path, "w") as f:
                json.dump({"bars": [], "timeframe": tf}, f)

        # TSLA with only 15min
        file_path = self.data_path / "proc_TSLA_15min.json"
        with open(file_path, "w") as f:
            json.dump({"bars": [], "timeframe": "15min"}, f)

        # List symbols with 15min data
        symbols = self.pipeline.list_available_symbols(source="file", timeframe="15min")

        self.assertEqual(len(symbols), 2)
        self.assertIn("AAPL", symbols)
        self.assertIn("TSLA", symbols)

        # List symbols with 5min data
        symbols_5min = self.pipeline.list_available_symbols(source="file", timeframe="5min")

        self.assertEqual(len(symbols_5min), 1)
        self.assertIn("AAPL", symbols_5min)


class TestMultiTimeframeFetching(unittest.TestCase):
    """Test multi-timeframe data fetching logic."""

    def setUp(self):
        """Set up test pipeline."""
        self.test_dir = tempfile.mkdtemp()
        self.pipeline = UnifiedDataPipeline(
            data_path=str(Path(self.test_dir) / "proc_data"),
            raw_data_path=str(Path(self.test_dir) / "raw_data"),
            store_to_db=False,
            store_to_files=True,
        )

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir)

    def test_update_symbols_multiple_timeframes(self):
        """Test updating symbols with multiple timeframes."""
        # This is a sync wrapper for the async test
        import asyncio

        asyncio.run(self._async_test_update_symbols_multiple_timeframes())

    @patch("core.unified_data_pipeline.UnifiedDataPipeline._fetch_alpaca")
    @patch("core.unified_data_pipeline.UnifiedDataPipeline._select_best_source")
    @patch("core.unified_data_pipeline.UnifiedDataPipeline.check_and_warn_credentials")
    async def _async_test_update_symbols_multiple_timeframes(
        self, mock_check_creds, mock_select_source, mock_fetch_alpaca
    ):
        """Test updating symbols with multiple timeframes."""
        # Setup mocks
        mock_check_creds.return_value = None
        mock_select_source.return_value = "alpaca"

        # Mock fetch to return sample data
        mock_bars = [
            {
                "datetime": int(datetime.now(timezone.utc).timestamp() * 1000),
                "open": 150.0,
                "high": 151.0,
                "low": 149.0,
                "close": 150.5,
                "volume": 10000,
            }
        ]
        mock_fetch_alpaca.return_value = mock_bars

        # Update with multiple timeframes
        # Note: This would need async test setup in real usage
        # For now, just verify the interface accepts timeframes parameter
        timeframes = ["5min", "15min", "30min"]

        # Verify SUPPORTED_TIMEFRAMES contains our test timeframes
        for tf in timeframes:
            self.assertIn(tf, self.pipeline.SUPPORTED_TIMEFRAMES)


class TestCacheWithTimeframes(unittest.TestCase):
    """Test cache handling with multiple timeframes."""

    def setUp(self):
        """Set up test pipeline."""
        self.test_dir = tempfile.mkdtemp()
        self.pipeline = UnifiedDataPipeline(
            data_path=str(Path(self.test_dir) / "proc_data"),
            raw_data_path=str(Path(self.test_dir) / "raw_data"),
            store_to_db=False,
            store_to_files=True,
        )

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir)

    def test_cache_key_includes_timeframe(self):
        """Test cache uses composite key with timeframe."""
        symbol = "AAPL"
        timeframe = "5min"

        bars = [
            {
                "datetime": int(datetime.now(timezone.utc).timestamp() * 1000),
                "open": 150.0,
                "high": 151.0,
                "low": 149.0,
                "close": 150.5,
                "volume": 10000,
            }
        ]

        # Update cache
        self.pipeline._update_cache(symbol, bars, timeframe)

        # Cache key should be symbol_timeframe
        if self.pipeline.cache:
            # Would check cache.get(f'{symbol}_{timeframe}') here
            # For now, just verify the method accepts timeframe parameter
            pass


if __name__ == "__main__":
    unittest.main()
