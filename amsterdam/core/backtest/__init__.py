"""
Backtest Module

Advanced backtesting framework with:
- Data validation and cleaning
- Vectorized backtesting engine
- Parameter optimization (grid search, random search)
- Walk-forward analysis
- Monte Carlo simulation
- Slippage models
- Benchmark comparison
"""

from core.backtest.backtester import BacktestConfig, VectorizedBacktester
from core.backtest.benchmarking import (
    BenchmarkComparison,
    calculate_drawdown_analysis,
    calculate_rolling_metrics,
    compare_to_benchmark,
)
from core.backtest.monte_carlo import MonteCarloResult, bootstrap_returns, monte_carlo_simulation
from core.backtest.optimization import OptimizationResult, grid_search, random_search
from core.backtest.slippage_models import (
    CompositeSlippage,
    FixedSlippage,
    RandomSlippage,
    SlippageModel,
    VolatilityAdjustedSlippage,
    VolumeBasedSlippage,
)
from core.backtest.validation import ValidationResult, validate_ohlcv_data
from core.backtest.walk_forward import WalkForwardResult, anchored_walk_forward, walk_forward_analysis

__all__ = [
    # Validation
    "ValidationResult",
    "validate_ohlcv_data",
    # Slippage Models
    "SlippageModel",
    "FixedSlippage",
    "RandomSlippage",
    "VolumeBasedSlippage",
    "VolatilityAdjustedSlippage",
    "CompositeSlippage",
    # Optimization
    "OptimizationResult",
    "grid_search",
    "random_search",
    # Walk-Forward
    "WalkForwardResult",
    "walk_forward_analysis",
    "anchored_walk_forward",
    # Monte Carlo
    "MonteCarloResult",
    "monte_carlo_simulation",
    "bootstrap_returns",
    # Benchmarking
    "BenchmarkComparison",
    "compare_to_benchmark",
    "calculate_rolling_metrics",
    "calculate_drawdown_analysis",
    # Backtester
    "VectorizedBacktester",
    "BacktestConfig",
]
