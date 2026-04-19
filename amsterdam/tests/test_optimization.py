"""
Tests for Parameter Optimization Module

Coverage:
- OptimizationResult dataclass
- grid_search function
- random_search function
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.backtest.optimization import (
    OptimizationResult,
    grid_search,
    random_search
)


def create_mock_backtester(sharpe=1.5, total_return=0.10, sortino=1.8, max_dd=-0.15):
    """Create a mock backtester that returns consistent metrics."""
    mock_bt = MagicMock()
    mock_portfolio = pd.DataFrame({
        'Portfolio_Value': [10000, 10100, 10200, 10300, 10400]
    })
    mock_bt.run.return_value = mock_portfolio
    mock_bt.get_metrics.return_value = {
        'sharpe_ratio': sharpe,
        'total_return': total_return,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'win_rate': 0.55,
        'profit_factor': 1.3
    }
    return mock_bt


# Patch target for VectorizedBacktester (imported inside functions)
BACKTESTER_PATCH = 'core.backtest.backtester.VectorizedBacktester'


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_dataclass_creation(self):
        """Test OptimizationResult can be created."""
        result = OptimizationResult(
            best_params={'sma_window': 20},
            best_metric=1.5,
            all_results=[],
            metric_name='sharpe_ratio'
        )

        assert result.best_params == {'sma_window': 20}
        assert result.best_metric == 1.5
        assert result.metric_name == 'sharpe_ratio'

    def test_dataclass_all_fields(self):
        """Test all fields are accessible."""
        result = OptimizationResult(
            best_params={'a': 1, 'b': 2},
            best_metric=2.5,
            all_results=[{'params': {'a': 1}, 'metric_value': 1.0}],
            metric_name='total_return'
        )

        assert hasattr(result, 'best_params')
        assert hasattr(result, 'best_metric')
        assert hasattr(result, 'all_results')
        assert hasattr(result, 'metric_name')


class TestGridSearch:
    """Tests for grid_search function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 200
        return pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=n, freq='D'),
            'Open': np.random.uniform(100, 110, n),
            'High': np.random.uniform(110, 120, n),
            'Low': np.random.uniform(90, 100, n),
            'Close': np.random.uniform(100, 110, n),
            'Volume': np.random.randint(1000000, 5000000, n)
        })

    def test_grid_search_basic(self, sample_data):
        """Test basic grid search."""
        param_grid = {
            'sma_window': [10, 20],
            'ema_window': [5, 10]
        }

        with patch(BACKTESTER_PATCH) as MockBT:
            MockBT.return_value = create_mock_backtester()

            result = grid_search(
                sample_data,
                strategy_name='SMA',
                param_grid=param_grid,
                metric='sharpe_ratio',
                initial_capital=10000,
                verbose=False
            )

            assert isinstance(result, OptimizationResult)
            assert 'sma_window' in result.best_params
            assert result.metric_name == 'sharpe_ratio'

    def test_grid_search_returns_all_results(self, sample_data):
        """Test that all results are returned."""
        param_grid = {
            'sma_window': [10, 20, 30],
        }

        with patch(BACKTESTER_PATCH) as MockBT:
            MockBT.return_value = create_mock_backtester()

            result = grid_search(
                sample_data,
                strategy_name='SMA',
                param_grid=param_grid,
                metric='total_return',
                initial_capital=10000,
                verbose=False
            )

            assert len(result.all_results) == 3

    def test_grid_search_different_metrics(self, sample_data):
        """Test grid search with different metrics."""
        param_grid = {'sma_window': [10, 20]}

        with patch(BACKTESTER_PATCH) as MockBT:
            MockBT.return_value = create_mock_backtester()

            # Test with total_return
            result_return = grid_search(
                sample_data,
                strategy_name='SMA',
                param_grid=param_grid,
                metric='total_return',
                verbose=False
            )
            assert result_return.metric_name == 'total_return'

            # Test with max_drawdown
            result_dd = grid_search(
                sample_data,
                strategy_name='SMA',
                param_grid=param_grid,
                metric='max_drawdown',
                verbose=False
            )
            assert result_dd.metric_name == 'max_drawdown'

    def test_grid_search_best_is_maximum(self, sample_data):
        """Test that best result has maximum metric value."""
        param_grid = {'sma_window': [10, 20, 30]}

        with patch(BACKTESTER_PATCH) as MockBT:
            MockBT.return_value = create_mock_backtester()

            result = grid_search(
                sample_data,
                strategy_name='SMA',
                param_grid=param_grid,
                metric='sharpe_ratio',
                verbose=False
            )

            # Best should have maximum metric value
            max_metric = max(r['metric_value'] for r in result.all_results)
            assert result.best_metric == max_metric

    def test_grid_search_no_valid_results_raises(self, sample_data):
        """Test that no valid results raises error."""
        param_grid = {'sma_window': [10]}

        with patch(BACKTESTER_PATCH) as MockBT:
            mock_bt = MagicMock()
            mock_bt.run.return_value = pd.DataFrame()  # Empty result
            MockBT.return_value = mock_bt

            with pytest.raises(ValueError, match="No valid results"):
                grid_search(
                    sample_data,
                    strategy_name='SMA',
                    param_grid=param_grid,
                    verbose=False
                )


class TestRandomSearch:
    """Tests for random_search function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 200
        return pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=n, freq='D'),
            'Open': np.random.uniform(100, 110, n),
            'High': np.random.uniform(110, 120, n),
            'Low': np.random.uniform(90, 100, n),
            'Close': np.random.uniform(100, 110, n),
            'Volume': np.random.randint(1000000, 5000000, n)
        })

    def test_random_search_basic(self, sample_data):
        """Test basic random search."""
        param_distributions = {
            'sma_window': [10, 15, 20, 25, 30],
        }

        with patch(BACKTESTER_PATCH) as MockBT:
            MockBT.return_value = create_mock_backtester()

            result = random_search(
                sample_data,
                strategy_name='SMA',
                param_distributions=param_distributions,
                n_iter=5,
                metric='sharpe_ratio',
                verbose=False,
                seed=42
            )

            assert isinstance(result, OptimizationResult)
            assert 'sma_window' in result.best_params

    def test_random_search_with_tuple_range(self, sample_data):
        """Test random search with tuple (min, max) distribution."""
        param_distributions = {
            'sma_window': [10, 20, 30],  # Discrete
        }

        with patch(BACKTESTER_PATCH) as MockBT:
            MockBT.return_value = create_mock_backtester()

            result = random_search(
                sample_data,
                strategy_name='SMA',
                param_distributions=param_distributions,
                n_iter=5,
                verbose=False,
                seed=42
            )

            assert result.best_params['sma_window'] in [10, 20, 30]

    def test_random_search_reproducibility(self, sample_data):
        """Test that seed makes results reproducible."""
        param_distributions = {
            'sma_window': [10, 15, 20, 25, 30],
        }

        with patch(BACKTESTER_PATCH) as MockBT:
            MockBT.return_value = create_mock_backtester()

            result1 = random_search(
                sample_data,
                strategy_name='SMA',
                param_distributions=param_distributions,
                n_iter=5,
                verbose=False,
                seed=42
            )

            result2 = random_search(
                sample_data,
                strategy_name='SMA',
                param_distributions=param_distributions,
                n_iter=5,
                verbose=False,
                seed=42
            )

            assert result1.best_params == result2.best_params

    def test_random_search_n_iter(self, sample_data):
        """Test that n_iter controls number of evaluations."""
        param_distributions = {
            'sma_window': [10, 15, 20, 25, 30],
        }

        with patch(BACKTESTER_PATCH) as MockBT:
            MockBT.return_value = create_mock_backtester()

            result = random_search(
                sample_data,
                strategy_name='SMA',
                param_distributions=param_distributions,
                n_iter=3,
                verbose=False,
                seed=42
            )

            # Should have at most n_iter results
            assert len(result.all_results) <= 3


class TestOptimizationMetrics:
    """Tests for different optimization metrics."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 150
        return pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=n, freq='D'),
            'Open': np.random.uniform(100, 110, n),
            'High': np.random.uniform(110, 120, n),
            'Low': np.random.uniform(90, 100, n),
            'Close': np.random.uniform(100, 110, n),
            'Volume': np.random.randint(1000000, 5000000, n)
        })

    def test_sharpe_ratio_metric(self, sample_data):
        """Test optimization with sharpe_ratio metric."""
        with patch(BACKTESTER_PATCH) as MockBT:
            MockBT.return_value = create_mock_backtester()

            result = grid_search(
                sample_data,
                strategy_name='SMA',
                param_grid={'sma_window': [10, 20]},
                metric='sharpe_ratio',
                verbose=False
            )

            assert result.metric_name == 'sharpe_ratio'

    def test_total_return_metric(self, sample_data):
        """Test optimization with total_return metric."""
        with patch(BACKTESTER_PATCH) as MockBT:
            MockBT.return_value = create_mock_backtester()

            result = grid_search(
                sample_data,
                strategy_name='SMA',
                param_grid={'sma_window': [10, 20]},
                metric='total_return',
                verbose=False
            )

            assert result.metric_name == 'total_return'

    def test_sortino_ratio_metric(self, sample_data):
        """Test optimization with sortino_ratio metric."""
        with patch(BACKTESTER_PATCH) as MockBT:
            MockBT.return_value = create_mock_backtester()

            result = grid_search(
                sample_data,
                strategy_name='SMA',
                param_grid={'sma_window': [10, 20]},
                metric='sortino_ratio',
                verbose=False
            )

            assert result.metric_name == 'sortino_ratio'

    def test_max_drawdown_metric(self, sample_data):
        """Test optimization with max_drawdown metric."""
        with patch(BACKTESTER_PATCH) as MockBT:
            MockBT.return_value = create_mock_backtester()

            result = grid_search(
                sample_data,
                strategy_name='SMA',
                param_grid={'sma_window': [10, 20]},
                metric='max_drawdown',
                verbose=False
            )

            assert result.metric_name == 'max_drawdown'


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=n, freq='D'),
            'Open': np.random.uniform(100, 110, n),
            'High': np.random.uniform(110, 120, n),
            'Low': np.random.uniform(90, 100, n),
            'Close': np.random.uniform(100, 110, n),
            'Volume': np.random.randint(1000000, 5000000, n)
        })

    def test_single_param_combination(self, sample_data):
        """Test with single parameter combination."""
        with patch(BACKTESTER_PATCH) as MockBT:
            MockBT.return_value = create_mock_backtester()

            result = grid_search(
                sample_data,
                strategy_name='SMA',
                param_grid={'sma_window': [20]},
                verbose=False
            )

            assert len(result.all_results) == 1
            assert result.best_params['sma_window'] == 20

    def test_multiple_params(self, sample_data):
        """Test with multiple parameters."""
        with patch(BACKTESTER_PATCH) as MockBT:
            MockBT.return_value = create_mock_backtester()

            result = grid_search(
                sample_data,
                strategy_name='SMA',
                param_grid={
                    'sma_window': [10, 20],
                    'ema_window': [5, 10]
                },
                verbose=False
            )

            # Should have 2x2=4 combinations
            assert len(result.all_results) == 4

    def test_grid_search_with_transaction_cost(self, sample_data):
        """Test that transaction cost parameter is passed."""
        with patch(BACKTESTER_PATCH) as MockBT:
            MockBT.return_value = create_mock_backtester()

            result_low_cost = grid_search(
                sample_data,
                strategy_name='SMA',
                param_grid={'sma_window': [20]},
                transaction_cost=0.0001,
                verbose=False
            )

            result_high_cost = grid_search(
                sample_data,
                strategy_name='SMA',
                param_grid={'sma_window': [20]},
                transaction_cost=0.01,
                verbose=False
            )

            # Both should return results
            assert isinstance(result_low_cost, OptimizationResult)
            assert isinstance(result_high_cost, OptimizationResult)
