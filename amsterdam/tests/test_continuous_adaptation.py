#!/usr/bin/env python3
"""
Tests for Continuous Adaptation System

Run with:
    python tests/test_continuous_adaptation.py
    pytest tests/test_continuous_adaptation.py -v
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.continuous_metrics import ContinuousMetrics, RiskMetrics
from core.continuous_position_sizer import ContinuousPositionSizer


class TestContinuousMetrics(unittest.TestCase):
    """Test continuous metrics calculator."""

    def setUp(self):
        """Set up test fixtures."""
        self.metrics_calc = ContinuousMetrics()

        # Create test data
        np.random.seed(42)
        self.test_bars = self._create_test_bars(300)

    def _create_test_bars(self, n_bars: int) -> pd.DataFrame:
        """Create synthetic bar data for testing."""
        dates = pd.date_range(end=datetime.now(), periods=n_bars, freq="D")

        # Generate realistic price data with volatility
        base_price = 100
        returns = np.random.normal(0.001, 0.02, n_bars)  # 0.1% mean, 2% std
        prices = base_price * (1 + returns).cumprod()

        bars = pd.DataFrame(
            {
                "timestamp": dates,
                "open": prices * 0.995,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": np.random.randint(1000000, 5000000, n_bars),
            }
        )

        return bars

    def test_atr_calculation(self):
        """Test ATR calculation."""
        atr = self.metrics_calc.calculate_atr(self.test_bars, period=14)

        self.assertGreater(atr, 0, "ATR should be positive")
        self.assertIsInstance(atr, float, "ATR should be float")

    def test_atr_percentile(self):
        """Test ATR percentile calculation."""
        percentile = self.metrics_calc.calculate_atr_percentile("TEST", self.test_bars, lookback=250)

        self.assertGreaterEqual(percentile, 0, "Percentile should be >= 0")
        self.assertLessEqual(percentile, 100, "Percentile should be <= 100")

    def test_rolling_sharpe(self):
        """Test rolling Sharpe calculation."""
        # Create mock trades
        mock_trades = [
            {
                "pnl": 100 if i % 2 == 0 else -50,
                "entry_value": 10000,
                "return_pct": 1.0 if i % 2 == 0 else -0.5,
                "timestamp": datetime.now() - timedelta(days=i),
            }
            for i in range(20)
        ]

        sharpe = self.metrics_calc.calculate_rolling_sharpe("TEST", "rsi", mock_trades, lookback_days=30)

        self.assertIsInstance(sharpe, float, "Sharpe should be float")
        # With 67% win rate and positive mean, should be positive
        self.assertGreater(sharpe, 0, "Sharpe should be positive for winning strategy")

    def test_normalize_atr_percentile(self):
        """Test ATR percentile normalization."""
        # Low volatility (0th percentile) should give high score (1.0)
        score_low = self.metrics_calc.normalize_atr_percentile(0)
        self.assertAlmostEqual(score_low, 1.0, places=2)

        # High volatility (100th percentile) should give low score (0.0)
        score_high = self.metrics_calc.normalize_atr_percentile(100)
        self.assertAlmostEqual(score_high, 0.0, places=2)

        # Medium volatility (50th percentile) should give medium score (0.5)
        score_med = self.metrics_calc.normalize_atr_percentile(50)
        self.assertAlmostEqual(score_med, 0.5, places=2)

    def test_normalize_sharpe(self):
        """Test Sharpe normalization."""
        # Negative Sharpe should give 0
        score_neg = self.metrics_calc.normalize_sharpe(-0.5)
        self.assertAlmostEqual(score_neg, 0.0)

        # Sharpe = 2.0 should give 1.0 (max)
        score_max = self.metrics_calc.normalize_sharpe(2.0, max_sharpe=2.0)
        self.assertAlmostEqual(score_max, 1.0)

        # Sharpe = 1.0 should give 0.5
        score_med = self.metrics_calc.normalize_sharpe(1.0, max_sharpe=2.0)
        self.assertAlmostEqual(score_med, 0.5)


class TestContinuousPositionSizer(unittest.TestCase):
    """Test continuous position sizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.sizer = ContinuousPositionSizer()

        # Create test data
        np.random.seed(42)
        self.test_bars = self._create_test_bars(300)
        self.test_trades = self._create_test_trades(20, win_rate=0.6)

    def _create_test_bars(self, n_bars: int) -> pd.DataFrame:
        """Create synthetic bar data."""
        dates = pd.date_range(end=datetime.now(), periods=n_bars, freq="D")
        base_price = 100
        returns = np.random.normal(0.001, 0.02, n_bars)
        prices = base_price * (1 + returns).cumprod()

        return pd.DataFrame(
            {
                "timestamp": dates,
                "open": prices * 0.995,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": np.random.randint(1000000, 5000000, n_bars),
            }
        )

    def _create_test_trades(self, n_trades: int, win_rate: float) -> list:
        """Create mock trade history."""
        trades = []
        for i in range(n_trades):
            is_win = i < (n_trades * win_rate)
            pnl = 100 if is_win else -50
            trades.append(
                {
                    "pnl": pnl,
                    "entry_value": 10000,
                    "return_pct": 1.0 if is_win else -0.5,
                    "timestamp": datetime.now() - timedelta(days=i),
                }
            )
        return trades

    def test_calculate_position_size(self):
        """Test position size calculation."""
        result = self.sizer.calculate_position_size(
            symbol="TEST",
            strategy="rsi",
            base_capital=100000,
            max_position_pct=0.10,
            bars=self.test_bars,
            recent_trades=self.test_trades,
        )

        # Verify result structure
        self.assertEqual(result.symbol, "TEST")
        self.assertEqual(result.strategy, "rsi")
        self.assertGreater(result.base_size, 0)
        self.assertGreaterEqual(result.multiplier, 0)
        self.assertLessEqual(result.multiplier, 1.0)
        self.assertGreater(result.final_size, 0)
        self.assertIsInstance(result.metrics, RiskMetrics)

    def test_continuous_vs_discrete_multiplier(self):
        """Test both multiplier calculation methods."""
        # Create mock metrics
        from core.continuous_metrics import RiskMetrics

        metrics = RiskMetrics(
            atr_percentile=25.0,  # Low volatility
            rolling_sharpe=1.6,  # Good performance
            volatility_score=0.75,
            performance_score=0.8,
            combined_score=0.77,
        )

        # Test discrete
        self.sizer.config["use_continuous_formula"] = False
        mult_discrete, _ = self.sizer.calculate_discrete_multiplier(metrics)

        # Test continuous
        self.sizer.config["use_continuous_formula"] = True
        trade_count = 50  # Provide a reasonable trade count for testing
        mult_continuous, _, _ = self.sizer.calculate_continuous_multiplier(metrics, trade_count)

        # Both should be in valid range
        self.assertGreaterEqual(mult_discrete, 0)
        self.assertLessEqual(mult_discrete, 1.0)
        self.assertGreaterEqual(mult_continuous, 0)
        self.assertLessEqual(mult_continuous, 1.0)

    def test_should_skip_trade(self):
        """Test trade skip logic."""
        from core.continuous_metrics import RiskMetrics
        from core.continuous_position_sizer import PositionSizeResult

        # Create result with very low multiplier
        low_multiplier_result = PositionSizeResult(
            symbol="TEST",
            strategy="rsi",
            base_size=10000,
            raw_score=0.03,  # Below default skip_trade_threshold of 0.05
            multiplier=0.05,  # Very low
            final_size=500,
            metrics=RiskMetrics(0, 0, 0, 0, 0),
            rationale="Low multiplier test",
        )

        # Should skip (raw_score 0.03 is below default threshold of 0.05)
        should_skip = self.sizer.should_skip_trade(low_multiplier_result)
        self.assertTrue(should_skip, f"Expected to skip trade with raw_score={low_multiplier_result.raw_score}")

        # Create result with good multiplier
        good_multiplier_result = PositionSizeResult(
            symbol="TEST",
            strategy="rsi",
            base_size=10000,
            raw_score=0.75,
            multiplier=0.75,
            final_size=7500,
            metrics=RiskMetrics(0, 0, 0, 0, 0),
            rationale="Good multiplier test",
        )

        # Should not skip
        self.assertFalse(self.sizer.should_skip_trade(good_multiplier_result))


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestContinuousMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestContinuousPositionSizer))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
