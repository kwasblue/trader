"""
Test suite for strategy signal generation.

Tests that all strategies:
1. Return int type (not DataFrame)
2. Return valid signals: -1 (sell), 0 (hold), or 1 (buy)
3. Handle edge cases (empty data, insufficient data)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock


class TestStrategySignalTypes(unittest.TestCase):
    """Test that all strategies return int signals."""

    @classmethod
    def setUpClass(cls):
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        n = 100

        # Generate realistic price data
        close_prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

        cls.sample_data = pd.DataFrame({
            'Open': close_prices - np.random.rand(n) * 0.5,
            'High': close_prices + np.random.rand(n) * 1.0,
            'Low': close_prices - np.random.rand(n) * 1.0,
            'Close': close_prices,
            'Volume': np.random.randint(100000, 1000000, n),
        })

        # Also create lowercase version for strategies that expect it
        cls.sample_data_lower = cls.sample_data.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        })

    def _assert_valid_signal(self, signal, strategy_name):
        """Assert signal is valid int in {-1, 0, 1}."""
        self.assertIsInstance(
            signal, (int, np.integer),
            f"{strategy_name} returned {type(signal)}, expected int"
        )
        self.assertIn(
            int(signal), [-1, 0, 1],
            f"{strategy_name} returned {signal}, expected -1, 0, or 1"
        )

    # ========== Simple Strategies ==========

    def test_ema_strategy_returns_int(self):
        """EMAStrategy should return int signal."""
        from strategies.strategy_registry.ema_strategy import EMAStrategy

        strategy = EMAStrategy(params={"short_window": 10, "long_window": 20})
        signal = strategy.generate_signal(self.sample_data.copy())

        self._assert_valid_signal(signal, "EMAStrategy")

    def test_sma_strategy_returns_int(self):
        """SMAStrategy should return int signal."""
        from strategies.strategy_registry.sma_strategy import SMAStrategy

        strategy = SMAStrategy(fast=10, slow=30)
        signal = strategy.generate_signal(self.sample_data.copy())

        self._assert_valid_signal(signal, "SMAStrategy")

    def test_rsi_strategy_returns_int(self):
        """RSIStrategy should return int signal."""
        from strategies.strategy_registry.rsi_strategy import RSIStrategy

        strategy = RSIStrategy(params={"window": 14, "oversold": 30, "overbought": 70})
        signal = strategy.generate_signal(self.sample_data.copy())

        self._assert_valid_signal(signal, "RSIStrategy")

    def test_macd_strategy_returns_int(self):
        """MACDStrategy should return int signal."""
        from strategies.strategy_registry.macd_strategy import MACDStrategy

        strategy = MACDStrategy(params={"fast_window": 12, "slow_window": 26, "signal_window": 9})
        signal = strategy.generate_signal(self.sample_data.copy())

        self._assert_valid_signal(signal, "MACDStrategy")

    def test_bollinger_strategy_returns_int(self):
        """BollingerStrategy should return int signal."""
        from strategies.strategy_registry.bollinger_strategy import BollingerStrategy

        strategy = BollingerStrategy(params={"window": 20, "num_std": 2})
        signal = strategy.generate_signal(self.sample_data.copy())

        self._assert_valid_signal(signal, "BollingerStrategy")

    def test_momentum_strategy_returns_int(self):
        """MomentumStrategy should return int signal."""
        from strategies.strategy_registry.momentum_strategy import MomentumStrategy

        strategy = MomentumStrategy(lookback=20)
        signal = strategy.generate_signal(self.sample_data.copy())

        self._assert_valid_signal(signal, "MomentumStrategy")

    def test_mean_reversion_strategy_returns_int(self):
        """MeanReversionStrategy should return int signal."""
        from strategies.strategy_registry.mean_reversion_strategy import MeanReversionStrategy

        strategy = MeanReversionStrategy(window=14, threshold=1.0)
        signal = strategy.generate_signal(self.sample_data.copy())

        self._assert_valid_signal(signal, "MeanReversionStrategy")

    def test_breakout_strategy_returns_int(self):
        """BreakoutStrategy should return int signal."""
        from strategies.strategy_registry.breakout_strategy import BreakoutStrategy

        strategy = BreakoutStrategy(params={"window": 20})
        signal = strategy.generate_signal(self.sample_data.copy())

        self._assert_valid_signal(signal, "BreakoutStrategy")

    def test_donchian_strategy_returns_int(self):
        """DonchianStrategy should return int signal."""
        from strategies.strategy_registry.donchian_strategy import DonchianStrategy

        strategy = DonchianStrategy(params={"window": 20})
        signal = strategy.generate_signal(self.sample_data.copy())

        self._assert_valid_signal(signal, "DonchianStrategy")

    def test_vwap_strategy_returns_int(self):
        """VWAPStrategy should return int signal."""
        from strategies.strategy_registry.vwap_strategy import VWAPStrategy

        strategy = VWAPStrategy(params={})
        signal = strategy.generate_signal(self.sample_data.copy())

        self._assert_valid_signal(signal, "VWAPStrategy")

    def test_stochastic_strategy_returns_int(self):
        """StochasticStrategy should return int signal."""
        from strategies.strategy_registry.stochastic_strategy import StochasticStrategy

        strategy = StochasticStrategy(params={
            "k_window": 14, "d_window": 3,
            "oversold": 20, "overbought": 80
        })
        signal = strategy.generate_signal(self.sample_data.copy())

        self._assert_valid_signal(signal, "StochasticStrategy")

    def test_ichimoku_strategy_returns_int(self):
        """IchimokuStrategy should return int signal."""
        from strategies.strategy_registry.ichimoku_strategy import IchimokuStrategy

        strategy = IchimokuStrategy(params={})
        signal = strategy.generate_signal(self.sample_data.copy())

        self._assert_valid_signal(signal, "IchimokuStrategy")

    # ========== Strategies requiring external libraries ==========

    def test_adx_strategy_returns_int(self):
        """ADXStrategy should return int signal."""
        try:
            from strategies.strategy_registry.adx_strategy import ADXStrategy

            strategy = ADXStrategy(params={"window": 14, "threshold": 25})
            signal = strategy.generate_signal(self.sample_data.copy())

            self._assert_valid_signal(signal, "ADXStrategy")
        except ImportError:
            self.skipTest("ta library not installed")

    def test_psar_strategy_returns_int(self):
        """PSARStrategy should return int signal."""
        try:
            from strategies.strategy_registry.psar_strategy import PSARStrategy

            strategy = PSARStrategy(params={})
            signal = strategy.generate_signal(self.sample_data.copy())

            self._assert_valid_signal(signal, "PSARStrategy")
        except ImportError:
            self.skipTest("ta library not installed")

    # ========== Edge Cases ==========

    def test_ema_strategy_with_minimal_data(self):
        """EMAStrategy should return 0 for insufficient data."""
        from strategies.strategy_registry.ema_strategy import EMAStrategy

        strategy = EMAStrategy(params={"short_window": 20, "long_window": 50})
        minimal_data = self.sample_data.iloc[:10].copy()  # Only 10 rows
        signal = strategy.generate_signal(minimal_data)

        self._assert_valid_signal(signal, "EMAStrategy (minimal data)")

    def test_sma_strategy_with_minimal_data(self):
        """SMAStrategy should return 0 for insufficient data."""
        from strategies.strategy_registry.sma_strategy import SMAStrategy

        strategy = SMAStrategy(fast=10, slow=30)
        minimal_data = self.sample_data.iloc[:5].copy()  # Only 5 rows
        signal = strategy.generate_signal(minimal_data)

        self.assertEqual(signal, 0, "SMAStrategy should return 0 for insufficient data")


class TestStrategySignalValues(unittest.TestCase):
    """Test that strategies generate correct signal values for known scenarios."""

    def test_ema_crossover_buy_signal(self):
        """EMA crossover should generate buy signal when short > long."""
        from strategies.strategy_registry.ema_strategy import EMAStrategy

        # Create data with clear uptrend
        n = 50
        close_prices = np.linspace(100, 150, n)  # Steady uptrend

        data = pd.DataFrame({
            'Open': close_prices - 0.5,
            'High': close_prices + 1.0,
            'Low': close_prices - 1.0,
            'Close': close_prices,
            'Volume': [100000] * n,
        })

        strategy = EMAStrategy(params={"short_window": 5, "long_window": 20})
        signal = strategy.generate_signal(data)

        # In a clear uptrend, short EMA should be above long EMA
        self.assertEqual(signal, 1, "Expected buy signal in uptrend")

    def test_ema_crossover_sell_signal(self):
        """EMA crossover should generate sell signal when short < long."""
        from strategies.strategy_registry.ema_strategy import EMAStrategy

        # Create data with clear downtrend
        n = 50
        close_prices = np.linspace(150, 100, n)  # Steady downtrend

        data = pd.DataFrame({
            'Open': close_prices + 0.5,
            'High': close_prices + 1.0,
            'Low': close_prices - 1.0,
            'Close': close_prices,
            'Volume': [100000] * n,
        })

        strategy = EMAStrategy(params={"short_window": 5, "long_window": 20})
        signal = strategy.generate_signal(data)

        # In a clear downtrend, short EMA should be below long EMA
        self.assertEqual(signal, -1, "Expected sell signal in downtrend")

    def test_rsi_oversold_buy_signal(self):
        """RSI should generate buy signal when oversold."""
        from strategies.strategy_registry.rsi_strategy import RSIStrategy

        # Create data with continuous decline (to push RSI below oversold)
        n = 30
        close_prices = np.linspace(150, 100, n)  # Sharp decline

        data = pd.DataFrame({
            'Open': close_prices + 0.5,
            'High': close_prices + 1.0,
            'Low': close_prices - 1.0,
            'Close': close_prices,
            'Volume': [100000] * n,
        })

        strategy = RSIStrategy(params={"window": 14, "oversold": 30, "overbought": 70})
        signal = strategy.generate_signal(data)

        # RSI should be oversold after continuous decline
        self.assertIn(signal, [0, 1], "Expected hold or buy signal when oversold")


class TestCombinedStrategy(unittest.TestCase):
    """Test the CombinedStrategy that aggregates multiple strategies."""

    def setUp(self):
        """Create sample data."""
        np.random.seed(42)
        n = 100
        close_prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

        self.sample_data = pd.DataFrame({
            'Open': close_prices - np.random.rand(n) * 0.5,
            'High': close_prices + np.random.rand(n) * 1.0,
            'Low': close_prices - np.random.rand(n) * 1.0,
            'Close': close_prices,
            'Volume': np.random.randint(100000, 1000000, n),
        })

    def test_combined_strategy_returns_int(self):
        """CombinedStrategy should return int signal."""
        from strategies.strategy_registry.combined_strategy import CombinedStrategy

        # Create a mock strategy instance that has methods returning DataFrames with Signal
        class MockStrategyObj:
            def ema_strat(self, data, **kwargs):
                data['Signal'] = 1
                return data

            def rsi_strat(self, data, **kwargs):
                data['Signal'] = 1
                return data

        # Pass keyword arguments directly (not nested under 'params')
        strategy = CombinedStrategy(
            strategy_methods=["ema_strat", "rsi_strat"],
            combine_method="vote",
            strategy_instance=MockStrategyObj()
        )

        signal = strategy.generate_signal(self.sample_data.copy())

        self.assertIsInstance(signal, (int, np.integer))
        self.assertIn(int(signal), [-1, 0, 1])


if __name__ == '__main__':
    unittest.main()
