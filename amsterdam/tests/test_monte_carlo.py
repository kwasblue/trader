"""
Tests for Monte Carlo Simulation Module

Coverage:
- MonteCarloResult dataclass
- monte_carlo_simulation function
- bootstrap_returns function
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.backtest.monte_carlo import MonteCarloResult, bootstrap_returns, monte_carlo_simulation


class TestMonteCarloResult:
    """Tests for MonteCarloResult dataclass."""

    def test_dataclass_creation(self):
        """Test MonteCarloResult can be created."""
        result = MonteCarloResult(
            mean_return=0.05,
            median_return=0.04,
            std_return=0.10,
            percentiles={5: -0.15, 50: 0.04, 95: 0.25},
            max_drawdowns=[0.10, 0.15, 0.12],
            sharpe_ratios=[1.5, 1.2, 1.8],
            final_values=[10500, 10400, 10600],
            confidence_interval_95=(-0.10, 0.20),
        )

        assert result.mean_return == 0.05
        assert result.median_return == 0.04
        assert result.std_return == 0.10

    def test_dataclass_fields(self):
        """Test all expected fields are present."""
        result = MonteCarloResult(
            mean_return=0.0,
            median_return=0.0,
            std_return=0.0,
            percentiles={},
            max_drawdowns=[],
            sharpe_ratios=[],
            final_values=[],
            confidence_interval_95=(0, 0),
        )

        assert hasattr(result, "mean_return")
        assert hasattr(result, "median_return")
        assert hasattr(result, "std_return")
        assert hasattr(result, "percentiles")
        assert hasattr(result, "max_drawdowns")
        assert hasattr(result, "sharpe_ratios")
        assert hasattr(result, "final_values")
        assert hasattr(result, "confidence_interval_95")


class TestMonteCarloSimulation:
    """Tests for monte_carlo_simulation function."""

    @pytest.fixture
    def sample_trades(self):
        """Create sample trades with P&L."""
        return [
            {"pnl": 100, "symbol": "AAPL"},
            {"pnl": -50, "symbol": "AAPL"},
            {"pnl": 75, "symbol": "MSFT"},
            {"pnl": -25, "symbol": "MSFT"},
            {"pnl": 150, "symbol": "AAPL"},
            {"pnl": -100, "symbol": "GOOGL"},
            {"pnl": 200, "symbol": "AAPL"},
            {"pnl": -75, "symbol": "MSFT"},
            {"pnl": 125, "symbol": "AAPL"},
            {"pnl": 50, "symbol": "GOOGL"},
        ]

    def test_monte_carlo_basic(self, sample_trades):
        """Test basic Monte Carlo simulation."""
        result = monte_carlo_simulation(sample_trades, initial_capital=10000, n_simulations=100, seed=42)

        assert isinstance(result, MonteCarloResult)
        assert len(result.final_values) == 100
        assert len(result.max_drawdowns) == 100
        assert len(result.sharpe_ratios) == 100

    def test_monte_carlo_reproducibility(self, sample_trades):
        """Test that seed produces reproducible results."""
        result1 = monte_carlo_simulation(sample_trades, initial_capital=10000, n_simulations=50, seed=42)

        result2 = monte_carlo_simulation(sample_trades, initial_capital=10000, n_simulations=50, seed=42)

        assert result1.final_values == result2.final_values

    def test_monte_carlo_statistics(self, sample_trades):
        """Test statistics are calculated correctly."""
        result = monte_carlo_simulation(sample_trades, initial_capital=10000, n_simulations=1000, seed=42)

        # Mean and median should be reasonable
        assert -1 < result.mean_return < 1
        assert -1 < result.median_return < 1

        # Std should be non-negative
        assert result.std_return >= 0

        # Confidence interval should bracket median
        assert result.confidence_interval_95[0] <= result.median_return
        assert result.confidence_interval_95[1] >= result.median_return

    def test_monte_carlo_percentiles(self, sample_trades):
        """Test percentiles are calculated."""
        result = monte_carlo_simulation(sample_trades, initial_capital=10000, n_simulations=500, seed=42)

        assert 5 in result.percentiles
        assert 25 in result.percentiles
        assert 50 in result.percentiles
        assert 75 in result.percentiles
        assert 95 in result.percentiles

        # Percentiles should be ordered
        assert result.percentiles[5] <= result.percentiles[25]
        assert result.percentiles[25] <= result.percentiles[50]
        assert result.percentiles[50] <= result.percentiles[75]
        assert result.percentiles[75] <= result.percentiles[95]

    def test_monte_carlo_empty_trades_raises(self):
        """Test empty trades list raises error."""
        with pytest.raises(ValueError, match="No trades provided"):
            monte_carlo_simulation([], n_simulations=10)

    def test_monte_carlo_no_pnl_raises(self):
        """Test trades without P&L and without calculable P&L raises error."""
        trades = [{"symbol": "AAPL"}, {"symbol": "MSFT"}]

        with pytest.raises(ValueError, match="Could not extract P&L"):
            monte_carlo_simulation(trades, n_simulations=10)

    def test_monte_carlo_infers_pnl_from_actions(self):
        """Test P&L can be inferred from trade actions."""
        trades = [
            {"Action": "BUY", "Price": 100, "Quantity": 10},
            {"Action": "SELL", "Price": 105, "Quantity": 10},
            {"Action": "BUY", "Price": 95, "Quantity": 5},
            {"Action": "SELL", "Price": 100, "Quantity": 5},
        ]

        result = monte_carlo_simulation(trades, initial_capital=10000, n_simulations=50, seed=42)

        assert isinstance(result, MonteCarloResult)

    def test_monte_carlo_max_drawdowns(self, sample_trades):
        """Test max drawdowns are calculated."""
        result = monte_carlo_simulation(sample_trades, initial_capital=10000, n_simulations=100, seed=42)

        # All max drawdowns should be non-negative
        for mdd in result.max_drawdowns:
            assert mdd >= 0
            assert mdd <= 1  # Can't lose more than 100%


class TestBootstrapReturns:
    """Tests for bootstrap_returns function."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns array."""
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, 252)  # 1 year of daily returns

    def test_bootstrap_basic(self, sample_returns):
        """Test basic bootstrap simulation."""
        result = bootstrap_returns(sample_returns, n_simulations=100, block_size=20, seed=42)

        assert isinstance(result, MonteCarloResult)
        assert len(result.final_values) == 100
        assert len(result.max_drawdowns) == 100

    def test_bootstrap_reproducibility(self, sample_returns):
        """Test that seed produces reproducible results."""
        result1 = bootstrap_returns(sample_returns, n_simulations=50, block_size=20, seed=42)

        result2 = bootstrap_returns(sample_returns, n_simulations=50, block_size=20, seed=42)

        assert result1.final_values == result2.final_values

    def test_bootstrap_statistics(self, sample_returns):
        """Test statistics are calculated."""
        result = bootstrap_returns(sample_returns, n_simulations=500, block_size=10, seed=42)

        # Should have reasonable statistics
        assert isinstance(result.mean_return, float)
        assert isinstance(result.median_return, float)
        assert isinstance(result.std_return, float)
        assert result.std_return >= 0

    def test_bootstrap_different_block_sizes(self, sample_returns):
        """Test with different block sizes."""
        result_small = bootstrap_returns(sample_returns, n_simulations=100, block_size=5, seed=42)

        result_large = bootstrap_returns(sample_returns, n_simulations=100, block_size=50, seed=42)

        # Both should produce valid results
        assert isinstance(result_small, MonteCarloResult)
        assert isinstance(result_large, MonteCarloResult)

    def test_bootstrap_sharpe_ratios(self, sample_returns):
        """Test Sharpe ratios are calculated."""
        result = bootstrap_returns(sample_returns, n_simulations=100, block_size=20, seed=42)

        assert len(result.sharpe_ratios) == 100

        # Sharpe ratios should be finite
        for sr in result.sharpe_ratios:
            assert np.isfinite(sr)

    def test_bootstrap_confidence_interval(self, sample_returns):
        """Test confidence interval is calculated."""
        result = bootstrap_returns(sample_returns, n_simulations=1000, block_size=20, seed=42)

        lower, upper = result.confidence_interval_95

        # Lower should be less than upper
        assert lower <= upper

        # Median should be within interval (usually)
        # Note: This may occasionally fail due to skewed distributions
        assert lower <= result.percentiles[50] <= upper


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_trade(self):
        """Test with a single trade."""
        trades = [{"pnl": 100}]

        result = monte_carlo_simulation(trades, initial_capital=10000, n_simulations=50, seed=42)

        # With one trade, all simulations should be identical
        assert all(fv == result.final_values[0] for fv in result.final_values)

    def test_all_winning_trades(self):
        """Test with all winning trades."""
        trades = [{"pnl": 100} for _ in range(20)]

        result = monte_carlo_simulation(trades, initial_capital=10000, n_simulations=100, seed=42)

        # All final values should be above initial
        for fv in result.final_values:
            assert fv > 10000

    def test_all_losing_trades(self):
        """Test with all losing trades."""
        trades = [{"pnl": -50} for _ in range(20)]

        result = monte_carlo_simulation(trades, initial_capital=10000, n_simulations=100, seed=42)

        # All final values should be below initial
        for fv in result.final_values:
            assert fv < 10000

    def test_bootstrap_short_returns(self):
        """Test bootstrap with short returns array."""
        returns = np.array([0.01, -0.01, 0.02, -0.02, 0.01])

        result = bootstrap_returns(returns, n_simulations=50, block_size=2, seed=42)

        assert isinstance(result, MonteCarloResult)

    def test_monte_carlo_large_trades(self):
        """Test with large number of trades."""
        trades = [{"pnl": np.random.randint(-100, 100)} for _ in range(500)]

        result = monte_carlo_simulation(trades, initial_capital=10000, n_simulations=100, seed=42)

        assert len(result.final_values) == 100
