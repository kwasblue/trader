"""
Backtest Suite - Legacy Compatibility Facade

This module re-exports from core.backtest for backward compatibility.
New code should import directly from core.backtest instead.

Example:
    # Legacy (still works)
    from core.backtest_suite import VectorizedBacktester, grid_search

    # Preferred
    from core.backtest import VectorizedBacktester, grid_search
"""

# Re-export everything from the backtest package
from core.backtest import (
    # Validation
    ValidationResult,
    validate_ohlcv_data,

    # Slippage Models
    SlippageModel,
    FixedSlippage,
    RandomSlippage,
    VolumeBasedSlippage,
    VolatilityAdjustedSlippage,
    CompositeSlippage,

    # Optimization
    OptimizationResult,
    grid_search,
    random_search,

    # Walk-Forward
    WalkForwardResult,
    walk_forward_analysis,
    anchored_walk_forward,

    # Monte Carlo
    MonteCarloResult,
    monte_carlo_simulation,
    bootstrap_returns,

    # Benchmarking
    BenchmarkComparison,
    compare_to_benchmark,
    calculate_rolling_metrics,
    calculate_drawdown_analysis,

    # Backtester
    VectorizedBacktester,
    BacktestConfig,
)

__all__ = [
    # Validation
    'ValidationResult',
    'validate_ohlcv_data',

    # Slippage Models
    'SlippageModel',
    'FixedSlippage',
    'RandomSlippage',
    'VolumeBasedSlippage',
    'VolatilityAdjustedSlippage',
    'CompositeSlippage',

    # Optimization
    'OptimizationResult',
    'grid_search',
    'random_search',

    # Walk-Forward
    'WalkForwardResult',
    'walk_forward_analysis',
    'anchored_walk_forward',

    # Monte Carlo
    'MonteCarloResult',
    'monte_carlo_simulation',
    'bootstrap_returns',

    # Benchmarking
    'BenchmarkComparison',
    'compare_to_benchmark',
    'calculate_rolling_metrics',
    'calculate_drawdown_analysis',

    # Backtester
    'VectorizedBacktester',
    'BacktestConfig',
]
