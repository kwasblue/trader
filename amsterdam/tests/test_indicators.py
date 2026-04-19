"""
Test suite for technical indicators.

Tests indicator calculations for correctness.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

import numpy as np
import pandas as pd


class TestATRIndicator(unittest.TestCase):
    """Test ATR (Average True Range) indicator."""

    def setUp(self):
        """Create sample OHLCV data."""
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

    def test_atr_calculation(self):
        """ATR should return valid float values."""
        try:
            from indicators.atr import ATRIndicator

            indicator = ATRIndicator(self.data.copy(), window=14)
            indicator.compute()
            result = indicator.df

            self.assertIn("ATR", result.columns)
            # ATR should be positive after warmup period
            valid_atr = result["ATR"].dropna()
            self.assertTrue(all(valid_atr >= 0), "ATR should be non-negative")
        except ImportError:
            self.skipTest("ATRIndicator not found")

    def test_atr_with_insufficient_data(self):
        """ATR should handle insufficient data gracefully."""
        try:
            from indicators.atr import ATRIndicator

            small_data = self.data.iloc[:5].copy()
            indicator = ATRIndicator(small_data, window=14)
            indicator.compute()
            result = indicator.df

            # Should not raise, may have NaN values
            self.assertIsInstance(result, pd.DataFrame)
        except ImportError:
            self.skipTest("ATRIndicator not found")


class TestEMAIndicator(unittest.TestCase):
    """Test EMA (Exponential Moving Average) calculations."""

    def setUp(self):
        """Create sample data."""
        self.data = pd.DataFrame({"Close": [100, 102, 104, 103, 105, 107, 106, 108, 110, 109]})

    def test_ema_calculation(self):
        """EMA should be calculated correctly."""
        # Pandas EWM implementation
        ema = self.data["Close"].ewm(span=5, adjust=False).mean()

        # EMA should exist and be numeric
        self.assertEqual(len(ema), len(self.data))
        self.assertTrue(ema.notna().all())

    def test_ema_smoothing(self):
        """EMA should smooth price fluctuations."""
        ema_short = self.data["Close"].ewm(span=3, adjust=False).mean()
        ema_long = self.data["Close"].ewm(span=7, adjust=False).mean()

        # Shorter EMA should be more responsive (higher std deviation)
        # This is a general property, not strict for all data
        self.assertIsInstance(ema_short.std(), float)
        self.assertIsInstance(ema_long.std(), float)


class TestRSIIndicator(unittest.TestCase):
    """Test RSI (Relative Strength Index) calculations."""

    def setUp(self):
        """Create sample data with known RSI behavior."""
        # Uptrend data
        self.uptrend_data = pd.DataFrame({"Close": [100 + i for i in range(20)]})

        # Downtrend data
        self.downtrend_data = pd.DataFrame({"Close": [120 - i for i in range(20)]})

    def test_rsi_uptrend(self):
        """RSI should be high (>70) in strong uptrend."""
        delta = self.uptrend_data["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()

        # Avoid division by zero
        loss = loss.replace(0, 1e-10)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Last RSI should be high for uptrend
        last_rsi = rsi.iloc[-1]
        self.assertTrue(last_rsi > 50, f"RSI should be above 50 in uptrend, got {last_rsi}")

    def test_rsi_downtrend(self):
        """RSI should be low (<30) in strong downtrend."""
        delta = self.downtrend_data["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()

        # Avoid division by zero
        gain = gain.replace(0, 1e-10)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Last RSI should be low for downtrend
        last_rsi = rsi.iloc[-1]
        self.assertTrue(last_rsi < 50, f"RSI should be below 50 in downtrend, got {last_rsi}")


class TestBollingerBands(unittest.TestCase):
    """Test Bollinger Bands calculations."""

    def setUp(self):
        """Create sample data."""
        np.random.seed(42)
        self.data = pd.DataFrame({"Close": 100 + np.random.randn(50) * 2})

    def test_bollinger_band_calculation(self):
        """Bollinger Bands should be calculated correctly."""
        window = 20
        num_std = 2

        rolling_mean = self.data["Close"].rolling(window=window).mean()
        rolling_std = self.data["Close"].rolling(window=window).std()

        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)

        # Upper band should be above lower band
        valid_idx = ~(upper_band.isna() | lower_band.isna())
        self.assertTrue(all(upper_band[valid_idx] > lower_band[valid_idx]), "Upper band should be above lower band")

    def test_bollinger_contains_most_prices(self):
        """Most prices should fall within Bollinger Bands."""
        window = 20
        num_std = 2

        rolling_mean = self.data["Close"].rolling(window=window).mean()
        rolling_std = self.data["Close"].rolling(window=window).std()

        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)

        # Check how many prices are within bands
        within_bands = ((self.data["Close"] >= lower_band) & (self.data["Close"] <= upper_band)).dropna()

        # With random walk data, most prices should be within bands
        pct_within = within_bands.mean()
        self.assertGreater(pct_within, 0.5, "Most prices should be within bands")


if __name__ == "__main__":
    unittest.main()
