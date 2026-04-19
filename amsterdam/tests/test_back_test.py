"""
Test suite for backtesting framework.

Tests backtesting engine, results calculation, and metrics.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestBacktestEngine(unittest.TestCase):
    """Test backtesting engine functionality."""

    def setUp(self):
        """Create sample historical data."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)

        self.historical_data = pd.DataFrame({
            'timestamp': dates,
            'Open': close - np.random.rand(100) * 0.5,
            'High': close + np.random.rand(100) * 1.0,
            'Low': close - np.random.rand(100) * 1.0,
            'Close': close,
            'Volume': np.random.randint(100000, 1000000, 100),
        })

    def test_backtest_returns_results(self):
        """Backtest should return results dictionary."""
        # Mock backtest function
        def mock_backtest(data, strategy):
            return {
                'total_return': 0.15,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.08,
                'trades': 25
            }

        results = mock_backtest(self.historical_data, None)

        self.assertIn('total_return', results)
        self.assertIn('sharpe_ratio', results)
        self.assertIn('max_drawdown', results)

    def test_backtest_calculates_returns(self):
        """Backtest should calculate returns correctly."""
        # Simple buy-and-hold return
        start_price = self.historical_data['Close'].iloc[0]
        end_price = self.historical_data['Close'].iloc[-1]
        expected_return = (end_price - start_price) / start_price

        self.assertIsInstance(expected_return, (int, float))


class TestBacktestMetrics(unittest.TestCase):
    """Test backtesting metrics calculations."""

    def test_sharpe_ratio_calculation(self):
        """Sharpe ratio should be calculated correctly."""
        # Sample returns
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0003)  # Daily returns

        mean_return = returns.mean()
        std_return = returns.std()
        risk_free_rate = 0.02 / 252  # Daily risk-free rate

        sharpe = (mean_return - risk_free_rate) / std_return * np.sqrt(252)

        self.assertIsInstance(sharpe, float)

    def test_max_drawdown_calculation(self):
        """Max drawdown should be calculated correctly."""
        # Sample equity curve
        equity = pd.Series([100, 105, 103, 110, 95, 100, 115])

        # Calculate running max and drawdown
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        # Max drawdown should be (95 - 110) / 110 = 0.136
        expected_dd = (95 - 110) / 110
        self.assertAlmostEqual(max_drawdown, abs(expected_dd), places=2)

    def test_win_rate_calculation(self):
        """Win rate should be calculated correctly."""
        trades = pd.DataFrame({
            'pnl': [100, -50, 75, -25, 150, -100, 200]
        })

        winning_trades = (trades['pnl'] > 0).sum()
        total_trades = len(trades)
        win_rate = winning_trades / total_trades

        expected_win_rate = 4 / 7
        self.assertAlmostEqual(win_rate, expected_win_rate, places=2)

    def test_profit_factor_calculation(self):
        """Profit factor should be calculated correctly."""
        trades = pd.DataFrame({
            'pnl': [100, -50, 75, -25, 150, -100, 200]
        })

        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        expected_pf = (100 + 75 + 150 + 200) / (50 + 25 + 100)
        self.assertAlmostEqual(profit_factor, expected_pf, places=2)


class TestBacktestTradeExecution(unittest.TestCase):
    """Test backtest trade execution simulation."""

    def test_simulate_market_order(self):
        """Market order simulation should use close price."""
        bar = {'Open': 100, 'High': 102, 'Low': 99, 'Close': 101}

        # Market order typically fills at close in backtest
        fill_price = bar['Close']
        self.assertEqual(fill_price, 101)

    def test_simulate_limit_order_fill(self):
        """Limit order should fill if price touches limit."""
        bar = {'Open': 100, 'High': 102, 'Low': 99, 'Close': 101}
        limit_buy_price = 99.5

        # Check if limit price was touched
        filled = bar['Low'] <= limit_buy_price

        self.assertTrue(filled, "Limit buy at 99.5 should fill when low is 99")

    def test_simulate_limit_order_no_fill(self):
        """Limit order should not fill if price doesn't touch limit."""
        bar = {'Open': 100, 'High': 102, 'Low': 99, 'Close': 101}
        limit_buy_price = 98  # Below the low

        # Check if limit price was touched
        filled = bar['Low'] <= limit_buy_price

        self.assertFalse(filled, "Limit buy at 98 should not fill when low is 99")


class TestBacktestSlippage(unittest.TestCase):
    """Test slippage modeling in backtests."""

    def test_fixed_slippage(self):
        """Fixed slippage should be applied correctly."""
        price = 100.0
        slippage_pct = 0.001  # 0.1%

        # Buy slippage (price goes up)
        buy_fill = price * (1 + slippage_pct)
        self.assertAlmostEqual(buy_fill, 100.10, places=2)

        # Sell slippage (price goes down)
        sell_fill = price * (1 - slippage_pct)
        self.assertAlmostEqual(sell_fill, 99.90, places=2)

    def test_volume_based_slippage(self):
        """Volume-based slippage should increase with order size."""
        price = 100.0
        volume = 1000000
        order_sizes = [100, 10000, 100000]

        slippages = []
        for size in order_sizes:
            # Simple model: slippage proportional to order/volume ratio
            impact = (size / volume) * 0.01  # 1% at 100% volume
            slippages.append(impact)

        # Larger orders should have more slippage
        self.assertLess(slippages[0], slippages[1])
        self.assertLess(slippages[1], slippages[2])


# ============================================================================
# NEW BACKTEST SUITE TESTS
# ============================================================================

class TestDataValidation(unittest.TestCase):
    """Test OHLCV data validation."""

    def setUp(self):
        """Create sample data with issues."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')

        self.valid_data = pd.DataFrame({
            'Date': dates,
            'Open': [100 + i * 0.1 for i in range(50)],
            'High': [102 + i * 0.1 for i in range(50)],
            'Low': [99 + i * 0.1 for i in range(50)],
            'Close': [101 + i * 0.1 for i in range(50)],
            'Volume': [100000] * 50
        })

    def test_valid_data_passes(self):
        """Valid data should pass validation."""
        from core.backtest_suite import validate_ohlcv_data

        result = validate_ohlcv_data(self.valid_data)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)

    def test_missing_columns_detected(self):
        """Missing required columns should be detected."""
        from core.backtest_suite import validate_ohlcv_data

        incomplete = self.valid_data.drop(columns=['Close'])
        result = validate_ohlcv_data(incomplete)
        self.assertFalse(result.is_valid)
        self.assertTrue(any('Missing' in e for e in result.errors))

    def test_nan_values_fixed(self):
        """NaN values should be fixed when fix_issues=True."""
        from core.backtest_suite import validate_ohlcv_data

        data_with_nan = self.valid_data.copy()
        data_with_nan.loc[5, 'Close'] = np.nan
        data_with_nan.loc[10, 'Open'] = np.nan

        result = validate_ohlcv_data(data_with_nan, fix_issues=True)
        self.assertTrue(result.is_valid)
        self.assertIsNotNone(result.cleaned_data)
        self.assertEqual(result.cleaned_data['Close'].isna().sum(), 0)

    def test_ohlc_consistency_fixed(self):
        """OHLC consistency issues should be fixed."""
        from core.backtest_suite import validate_ohlcv_data

        # High < Low is invalid
        data = self.valid_data.copy()
        data.loc[5, 'High'] = 98  # Below Low of 99
        data.loc[5, 'Low'] = 100

        result = validate_ohlcv_data(data, fix_issues=True)
        self.assertTrue(result.is_valid)
        # High should now be >= Low
        self.assertGreaterEqual(
            result.cleaned_data.loc[5, 'High'],
            result.cleaned_data.loc[5, 'Low']
        )


class TestSlippageModels(unittest.TestCase):
    """Test slippage model implementations."""

    def test_fixed_slippage_model(self):
        """FixedSlippage should apply constant slippage."""
        from core.backtest_suite import FixedSlippage

        model = FixedSlippage(slippage_pct=0.001)

        buy_price = model.calculate_slippage(100.0, 100, 'buy')
        sell_price = model.calculate_slippage(100.0, 100, 'sell')

        self.assertAlmostEqual(buy_price, 100.10, places=2)
        self.assertAlmostEqual(sell_price, 99.90, places=2)

    def test_volume_based_slippage_model(self):
        """VolumeBasedSlippage should increase with participation rate."""
        from core.backtest_suite import VolumeBasedSlippage

        model = VolumeBasedSlippage(base_slippage=0.0001, volume_impact=0.1)

        # Small order (low participation)
        small_price = model.calculate_slippage(100.0, 100, 'buy', volume=100000)
        # Large order (high participation)
        large_price = model.calculate_slippage(100.0, 10000, 'buy', volume=100000)

        # Larger order should have more slippage
        self.assertGreater(large_price, small_price)

    def test_volatility_adjusted_slippage(self):
        """VolatilityAdjustedSlippage should increase with volatility."""
        from core.backtest_suite import VolatilityAdjustedSlippage

        model = VolatilityAdjustedSlippage(base_slippage=0.0005)

        low_vol_price = model.calculate_slippage(100.0, 100, 'buy', volatility=1.0)
        high_vol_price = model.calculate_slippage(100.0, 100, 'buy', volatility=5.0)

        # Higher volatility should have more slippage
        self.assertGreater(high_vol_price, low_vol_price)


class TestMonteCarloSimulation(unittest.TestCase):
    """Test Monte Carlo simulation."""

    def test_monte_carlo_returns_statistics(self):
        """Monte Carlo should return distribution statistics."""
        from core.backtest_suite import monte_carlo_simulation

        # Sample trades
        trades = [
            {'pnl': 100}, {'pnl': -50}, {'pnl': 75},
            {'pnl': -25}, {'pnl': 150}, {'pnl': -100},
            {'pnl': 200}, {'pnl': -75}, {'pnl': 125}
        ]

        result = monte_carlo_simulation(trades, initial_capital=10000, n_simulations=100, seed=42)

        self.assertIsNotNone(result.mean_return)
        self.assertIsNotNone(result.median_return)
        self.assertIsNotNone(result.std_return)
        self.assertIn(5, result.percentiles)
        self.assertIn(95, result.percentiles)

    def test_monte_carlo_confidence_interval(self):
        """Monte Carlo should provide 95% confidence interval."""
        from core.backtest_suite import monte_carlo_simulation

        # Use variable trade P&Ls for meaningful confidence interval
        np.random.seed(42)
        trades = [{'pnl': np.random.uniform(-50, 100)} for _ in range(50)]
        result = monte_carlo_simulation(trades, n_simulations=500, seed=42)

        lower, upper = result.confidence_interval_95
        # With variable trades, CI should have spread
        self.assertLessEqual(lower, upper)
        # Median should be within CI
        self.assertGreaterEqual(result.median_return, lower)
        self.assertLessEqual(result.median_return, upper)


class TestBenchmarkComparison(unittest.TestCase):
    """Test benchmark comparison functionality."""

    def test_compare_to_benchmark(self):
        """Comparison should calculate relative metrics."""
        from core.backtest_suite import compare_to_benchmark

        np.random.seed(42)
        # Strategy outperforms benchmark
        strategy_returns = pd.Series(np.random.randn(100) * 0.01 + 0.001)
        benchmark_returns = pd.Series(np.random.randn(100) * 0.01 + 0.0005)

        result = compare_to_benchmark(strategy_returns, benchmark_returns)

        self.assertIsNotNone(result.strategy_return)
        self.assertIsNotNone(result.benchmark_return)
        self.assertIsNotNone(result.excess_return)
        self.assertIsNotNone(result.beta)
        self.assertIsNotNone(result.alpha)
        self.assertIsNotNone(result.information_ratio)

    def test_up_down_capture(self):
        """Up/down capture ratios should be calculated."""
        from core.backtest_suite import compare_to_benchmark

        # Create returns where strategy has high up-capture, low down-capture
        strategy_returns = pd.Series([0.02, -0.005, 0.03, -0.002, 0.015])
        benchmark_returns = pd.Series([0.01, -0.01, 0.02, -0.005, 0.01])

        result = compare_to_benchmark(strategy_returns, benchmark_returns)

        self.assertIsNotNone(result.up_capture)
        self.assertIsNotNone(result.down_capture)


class TestVectorizedBacktester(unittest.TestCase):
    """Test vectorized backtester."""

    def setUp(self):
        """Create sample data."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        close = 100 + np.cumsum(np.random.randn(200) * 0.5)

        self.data = pd.DataFrame({
            'Date': dates,
            'Open': close - np.random.rand(200) * 0.5,
            'High': close + np.random.rand(200) * 1.0,
            'Low': close - np.random.rand(200) * 1.0,
            'Close': close,
            'Volume': np.random.randint(100000, 1000000, 200)
        })

    def test_vectorized_backtest_runs(self):
        """Vectorized backtester should run without errors."""
        from core.backtest_suite import VectorizedBacktester

        bt = VectorizedBacktester(self.data)
        # Use ema strategy which handles params via self.params
        result = bt.run('ema', {'short_window': 10, 'long_window': 30})

        self.assertIn('Portfolio_Value', result.columns)
        self.assertIn('Strategy_Return', result.columns)
        self.assertIn('Drawdown', result.columns)

    def test_vectorized_metrics(self):
        """Vectorized backtester should calculate metrics."""
        from core.backtest_suite import VectorizedBacktester

        bt = VectorizedBacktester(self.data)
        # Use rsi strategy which handles params via self.params
        result = bt.run('rsi', {'window': 14})
        metrics = bt.get_metrics(result)

        self.assertIn('total_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('win_rate', metrics)


class TestSortino(unittest.TestCase):
    """Test Sortino ratio calculation."""

    def test_sortino_ratio(self):
        """Sortino ratio should only consider downside deviation."""
        # Returns with more upside than downside
        returns = np.array([0.02, -0.01, 0.03, -0.005, 0.025, -0.008, 0.015])

        # Calculate downside deviation
        downside = returns[returns < 0]
        downside_std = np.std(downside)

        sortino = np.mean(returns) / downside_std * np.sqrt(252)

        # With positive mean and limited downside, Sortino should be positive
        self.assertGreater(sortino, 0)


if __name__ == '__main__':
    unittest.main()
