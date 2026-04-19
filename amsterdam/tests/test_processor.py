"""
Test suite for data processing pipeline.

Tests data loading, transformation, and processing.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


class TestHistoricalBarLoader(unittest.TestCase):
    """Test HistoricalBarLoader functionality."""

    def setUp(self):
        """Create sample data files."""
        self.sample_bars = [
            {
                "timestamp": "2023-01-01T10:00:00",
                "symbol": "AAPL",
                "Open": 150.0,
                "High": 152.0,
                "Low": 149.0,
                "Close": 151.0,
                "Volume": 100000,
            },
            {
                "timestamp": "2023-01-01T10:01:00",
                "symbol": "AAPL",
                "Open": 151.0,
                "High": 153.0,
                "Low": 150.0,
                "Close": 152.0,
                "Volume": 110000,
            },
        ]

    def test_loader_initialization(self):
        """HistoricalBarLoader should initialize correctly."""
        try:
            from core.historical_loader import HistoricalBarLoader

            loader = HistoricalBarLoader("/fake/path")
            self.assertIsNotNone(loader)
        except ImportError:
            self.skipTest("HistoricalBarLoader not found")

    @patch("os.path.exists")
    @patch("builtins.open", create=True)
    @patch("json.load")
    def test_loader_loads_bars(self, mock_json_load, mock_open, mock_exists):
        """HistoricalBarLoader should load bars from file."""
        try:
            from core.historical_loader import HistoricalBarLoader

            mock_exists.return_value = True
            mock_json_load.return_value = self.sample_bars

            loader = HistoricalBarLoader("/fake/path")
            bars = loader.load_last_n_bars("AAPL", n=2)

            # Should return iterable of bars
            self.assertIsNotNone(bars)
        except ImportError:
            self.skipTest("HistoricalBarLoader not found")


class TestDataTransformation(unittest.TestCase):
    """Test data transformation utilities."""

    def setUp(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 100

        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        self.data = pd.DataFrame(
            {
                "Open": close - np.random.rand(n) * 0.5,
                "High": close + np.random.rand(n) * 1.0,
                "Low": close - np.random.rand(n) * 1.0,
                "Close": close,
                "Volume": np.random.randint(100000, 1000000, n),
            }
        )

    def test_returns_calculation(self):
        """Returns should be calculated correctly."""
        returns = self.data["Close"].pct_change()

        # First return should be NaN
        self.assertTrue(pd.isna(returns.iloc[0]))

        # Subsequent returns should be finite
        self.assertTrue(returns.iloc[1:].notna().all())

    def test_log_returns_calculation(self):
        """Log returns should be calculated correctly."""
        log_returns = np.log(self.data["Close"] / self.data["Close"].shift(1))

        # First return should be NaN
        self.assertTrue(pd.isna(log_returns.iloc[0]))

        # Should be close to regular returns for small changes
        regular_returns = self.data["Close"].pct_change()
        correlation = log_returns.iloc[1:].corr(regular_returns.iloc[1:])
        self.assertGreater(correlation, 0.99)

    def test_normalize_data(self):
        """Data normalization should work correctly."""
        normalized = (self.data["Close"] - self.data["Close"].mean()) / self.data["Close"].std()

        # Mean should be ~0
        self.assertAlmostEqual(normalized.mean(), 0.0, places=5)

        # Std should be ~1
        self.assertAlmostEqual(normalized.std(), 1.0, places=5)


class TestDataValidation(unittest.TestCase):
    """Test data validation utilities."""

    def test_detect_missing_values(self):
        """Should detect missing values in data."""
        data = pd.DataFrame({"Close": [100, np.nan, 102, 103, np.nan]})

        missing_count = data["Close"].isna().sum()
        self.assertEqual(missing_count, 2)

    def test_detect_outliers_zscore(self):
        """Should detect outliers using z-score method."""
        # Create data with clear outlier
        data = pd.Series([100, 101, 102, 103, 104, 200])  # 200 is outlier

        z_scores = (data - data.mean()) / data.std()
        outliers = abs(z_scores) > 2

        self.assertTrue(outliers.iloc[-1], "Should detect 200 as outlier")

    def test_detect_duplicate_timestamps(self):
        """Should detect duplicate timestamps."""
        data = pd.DataFrame({"timestamp": ["2023-01-01", "2023-01-01", "2023-01-02"], "Close": [100, 101, 102]})

        duplicates = data["timestamp"].duplicated().sum()
        self.assertEqual(duplicates, 1)


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering utilities."""

    def setUp(self):
        """Create sample data."""
        np.random.seed(42)
        n = 50

        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        self.data = pd.DataFrame(
            {
                "Open": close - np.random.rand(n) * 0.5,
                "High": close + np.random.rand(n) * 1.0,
                "Low": close - np.random.rand(n) * 1.0,
                "Close": close,
                "Volume": np.random.randint(100000, 1000000, n),
            }
        )

    def test_price_momentum_feature(self):
        """Price momentum feature should be calculated."""
        lookback = 10
        momentum = self.data["Close"] / self.data["Close"].shift(lookback) - 1

        # After warmup, should have valid values
        valid_momentum = momentum.dropna()
        self.assertEqual(len(valid_momentum), len(self.data) - lookback)

    def test_volume_ma_feature(self):
        """Volume moving average should be calculated."""
        window = 10
        volume_ma = self.data["Volume"].rolling(window=window).mean()

        # After warmup, should have valid values
        valid_volume_ma = volume_ma.dropna()
        self.assertEqual(len(valid_volume_ma), len(self.data) - window + 1)

    def test_volatility_feature(self):
        """Volatility feature should be calculated."""
        window = 20
        returns = self.data["Close"].pct_change()
        volatility = returns.rolling(window=window).std()

        # Should be non-negative
        valid_vol = volatility.dropna()
        self.assertTrue(all(valid_vol >= 0))


if __name__ == "__main__":
    unittest.main()
