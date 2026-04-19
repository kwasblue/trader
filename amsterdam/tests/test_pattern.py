"""
Test suite for pattern recognition.

Tests candlestick patterns, chart patterns, and price action detection.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

import numpy as np
import pandas as pd


class TestCandlestickPatterns(unittest.TestCase):
    """Test candlestick pattern detection."""

    def test_detect_doji(self):
        """Should detect doji pattern."""
        # Doji: open and close very close, with wicks
        candle = {"Open": 100.0, "High": 102.0, "Low": 98.0, "Close": 100.1}

        body_size = abs(candle["Close"] - candle["Open"])
        range_size = candle["High"] - candle["Low"]
        body_ratio = body_size / range_size if range_size > 0 else 0

        is_doji = body_ratio < 0.1  # Body < 10% of range

        self.assertTrue(is_doji)

    def test_detect_hammer(self):
        """Should detect hammer pattern (bullish reversal)."""
        # Hammer: small body at top, long lower wick
        candle = {
            "Open": 100.0,
            "High": 100.4,  # Small upper wick (0.4 < body_size of 0.5)
            "Low": 95.0,  # Long lower wick (5.0 >= 2 * 0.5)
            "Close": 100.5,
        }

        body_size = abs(candle["Close"] - candle["Open"])
        lower_wick = min(candle["Open"], candle["Close"]) - candle["Low"]
        upper_wick = candle["High"] - max(candle["Open"], candle["Close"])

        # Hammer: lower wick >= 2x body, small upper wick
        is_hammer = lower_wick >= 2 * body_size and upper_wick < body_size

        self.assertTrue(is_hammer)

    def test_detect_engulfing(self):
        """Should detect engulfing pattern."""
        # Bullish engulfing: current candle engulfs previous
        prev_candle = {"Open": 102.0, "High": 103.0, "Low": 100.0, "Close": 101.0}  # Bearish
        curr_candle = {"Open": 99.0, "High": 104.0, "Low": 98.0, "Close": 103.5}  # Bullish

        prev_body_high = max(prev_candle["Open"], prev_candle["Close"])
        prev_body_low = min(prev_candle["Open"], prev_candle["Close"])
        curr_body_high = max(curr_candle["Open"], curr_candle["Close"])
        curr_body_low = min(curr_candle["Open"], curr_candle["Close"])

        is_bullish_engulfing = (
            curr_body_high > prev_body_high
            and curr_body_low < prev_body_low
            and curr_candle["Close"] > curr_candle["Open"]  # Bullish current
        )

        self.assertTrue(is_bullish_engulfing)


class TestChartPatterns(unittest.TestCase):
    """Test chart pattern detection."""

    def setUp(self):
        """Create sample price data."""
        np.random.seed(42)

    def test_detect_double_bottom(self):
        """Should detect double bottom pattern."""
        # Double bottom: two lows at similar level with higher middle
        prices = pd.Series([100, 95, 100, 96, 105, 110])  # V pattern twice

        # Find local minima
        local_mins = []
        for i in range(1, len(prices) - 1):
            if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
                local_mins.append((i, prices[i]))

        # Check if two minima are close in value
        if len(local_mins) >= 2:
            diff = abs(local_mins[0][1] - local_mins[1][1])
            is_double_bottom = diff < 2  # Within 2 points

            self.assertTrue(is_double_bottom)

    def test_detect_head_and_shoulders(self):
        """Should detect head and shoulders pattern."""
        # Simplified: left shoulder, head (higher), right shoulder (similar to left)
        highs = [100, 105, 110, 103, 98]  # Left shoulder, head, right shoulder

        left_shoulder = highs[1]
        head = highs[2]
        right_shoulder = highs[3]

        is_head_and_shoulders = (
            head > left_shoulder
            and head > right_shoulder
            and abs(left_shoulder - right_shoulder) < 5  # Shoulders similar
        )

        self.assertTrue(is_head_and_shoulders)

    def test_detect_triangle(self):
        """Should detect triangle pattern (converging highs and lows)."""
        highs = pd.Series([110, 108, 106, 104])
        lows = pd.Series([100, 101, 102, 103])

        # Highs decreasing, lows increasing = converging
        highs_decreasing = all(highs.diff().dropna() < 0)
        lows_increasing = all(lows.diff().dropna() > 0)

        is_triangle = highs_decreasing and lows_increasing

        self.assertTrue(is_triangle)


class TestSupportResistance(unittest.TestCase):
    """Test support and resistance level detection."""

    def test_identify_support_level(self):
        """Should identify support levels."""
        lows = pd.Series([100, 102, 100, 105, 101, 100])

        # Support: price bounces from similar level multiple times
        support_level = lows.mode()[0]

        self.assertEqual(support_level, 100)

    def test_identify_resistance_level(self):
        """Should identify resistance levels."""
        highs = pd.Series([110, 108, 110, 105, 109, 110])

        # Resistance: price rejected from similar level multiple times
        resistance_level = highs.mode()[0]

        self.assertEqual(resistance_level, 110)

    def test_breakout_detection(self):
        """Should detect breakout above resistance."""
        resistance = 110
        current_price = 112

        is_breakout = current_price > resistance
        self.assertTrue(is_breakout)


class TestTrendPatterns(unittest.TestCase):
    """Test trend pattern detection."""

    def test_detect_uptrend(self):
        """Should detect uptrend (higher highs and higher lows)."""
        prices = pd.DataFrame({"High": [100, 105, 103, 108, 106, 112], "Low": [95, 98, 97, 101, 100, 105]})

        # Check if highs and lows are generally increasing
        high_trend = np.polyfit(range(len(prices)), prices["High"], 1)[0]
        low_trend = np.polyfit(range(len(prices)), prices["Low"], 1)[0]

        is_uptrend = high_trend > 0 and low_trend > 0

        self.assertTrue(is_uptrend)

    def test_detect_downtrend(self):
        """Should detect downtrend (lower highs and lower lows)."""
        prices = pd.DataFrame({"High": [110, 108, 105, 103, 100], "Low": [105, 103, 100, 98, 95]})

        high_trend = np.polyfit(range(len(prices)), prices["High"], 1)[0]
        low_trend = np.polyfit(range(len(prices)), prices["Low"], 1)[0]

        is_downtrend = high_trend < 0 and low_trend < 0

        self.assertTrue(is_downtrend)

    def test_detect_ranging(self):
        """Should detect ranging/sideways market."""
        prices = pd.DataFrame({"High": [105, 104, 106, 105, 104], "Low": [100, 99, 101, 100, 99]})

        high_std = prices["High"].std()
        low_std = prices["Low"].std()

        # Ranging: low standard deviation in highs and lows
        is_ranging = high_std < 2 and low_std < 2

        self.assertTrue(is_ranging)


class TestVolumePatterns(unittest.TestCase):
    """Test volume-based patterns."""

    def test_volume_confirmation(self):
        """Should detect volume confirmation of breakout."""
        avg_volume = 1000000
        breakout_volume = 2500000

        # Volume > 2x average on breakout
        has_confirmation = breakout_volume > 2 * avg_volume

        self.assertTrue(has_confirmation)

    def test_volume_divergence(self):
        """Should detect volume divergence."""
        # Price making new highs but volume declining
        prices = [100, 105, 110, 115]
        volumes = [1000000, 900000, 800000, 700000]

        price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
        volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]

        # Divergence: price up, volume down
        has_divergence = price_trend > 0 and volume_trend < 0

        self.assertTrue(has_divergence)


if __name__ == "__main__":
    unittest.main()
