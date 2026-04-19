"""
Walk-Forward Analysis Module

Provides walk-forward analysis for out-of-sample strategy validation.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.backtest.optimization import grid_search

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    """Result of walk-forward analysis."""

    windows: list[dict[str, Any]]
    overall_return: float
    overall_sharpe: float
    out_of_sample_returns: list[float]
    in_sample_params: list[dict[str, Any]]


def walk_forward_analysis(
    data,
    strategy_name: str,
    param_grid: dict[str, list[Any]],
    train_size: int = 252,  # ~1 year of daily data
    test_size: int = 63,  # ~3 months
    step_size: int = 63,  # Roll forward by 3 months
    metric: str = "sharpe_ratio",
    initial_capital: float = 10000,
    verbose: bool = True,
) -> WalkForwardResult:
    """
    Walk-forward analysis with rolling train/test windows.

    1. Train on [0, train_size] → find best params
    2. Test on [train_size, train_size + test_size]
    3. Roll forward by step_size and repeat

    Args:
        data: OHLCV DataFrame
        strategy_name: Strategy to test
        param_grid: Parameters to optimize
        train_size: Training window size (bars)
        test_size: Testing window size (bars)
        step_size: How much to roll forward each iteration
        metric: Metric to optimize
        initial_capital: Starting capital
        verbose: Print progress

    Returns:
        WalkForwardResult with per-window and overall results
    """
    from core.backtest.backtester import VectorizedBacktester

    windows = []
    oos_returns = []
    is_params = []
    cumulative_capital = initial_capital

    n_bars = len(data)
    start_idx = 0
    window_num = 0

    while start_idx + train_size + test_size <= n_bars:
        window_num += 1
        train_end = start_idx + train_size
        test_end = train_end + test_size

        train_data = data.iloc[start_idx:train_end].copy()
        test_data = data.iloc[train_end:test_end].copy()

        if verbose:
            logger.info(f"Window {window_num}: Train [{start_idx}:{train_end}], Test [{train_end}:{test_end}]")

        # Optimize on training data
        try:
            opt_result = grid_search(
                train_data, strategy_name, param_grid, metric=metric, initial_capital=cumulative_capital, verbose=False
            )
            best_params = opt_result.best_params
            is_metric = opt_result.best_metric
        except Exception as e:
            if verbose:
                logger.warning(f"  Optimization failed: {e}")
            start_idx += step_size
            continue

        if verbose:
            logger.info(f"  Best in-sample params: {best_params}")
            logger.info(f"  In-sample {metric}: {is_metric:.4f}")

        # Test on out-of-sample data
        try:
            bt = VectorizedBacktester(test_data, cumulative_capital, 0.001)
            portfolio_df = bt.run(strategy_name, best_params)

            if not portfolio_df.empty:
                final_value = portfolio_df["Portfolio_Value"].iloc[-1]
                oos_return = (final_value - cumulative_capital) / cumulative_capital
                cumulative_capital = final_value

                metrics = bt.get_metrics(portfolio_df)
                oos_sharpe = metrics.get("sharpe_ratio", 0)
            else:
                oos_return = 0
                oos_sharpe = 0
        except Exception as e:
            if verbose:
                logger.warning(f"  Test failed: {e}")
            oos_return = 0
            oos_sharpe = 0

        if verbose:
            logger.info(f"  Out-of-sample return: {oos_return:.2%}")

        windows.append(
            {
                "window": window_num,
                "train_start": start_idx,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": test_end,
                "best_params": best_params,
                "is_metric": is_metric,
                "oos_return": oos_return,
                "oos_sharpe": oos_sharpe,
            }
        )

        oos_returns.append(oos_return)
        is_params.append(best_params)

        start_idx += step_size

    # Calculate overall metrics
    overall_return = (cumulative_capital - initial_capital) / initial_capital
    avg_oos_sharpe = np.mean([w["oos_sharpe"] for w in windows]) if windows else 0

    if verbose:
        logger.info("=== Walk-Forward Summary ===")
        logger.info(f"Windows tested: {len(windows)}")
        logger.info(f"Overall return: {overall_return:.2%}")
        logger.info(f"Average OOS Sharpe: {avg_oos_sharpe:.4f}")

    return WalkForwardResult(
        windows=windows,
        overall_return=overall_return,
        overall_sharpe=avg_oos_sharpe,
        out_of_sample_returns=oos_returns,
        in_sample_params=is_params,
    )


def anchored_walk_forward(
    data,
    strategy_name: str,
    param_grid: dict[str, list[Any]],
    initial_train_size: int = 252,
    test_size: int = 63,
    metric: str = "sharpe_ratio",
    initial_capital: float = 10000,
    verbose: bool = True,
) -> WalkForwardResult:
    """
    Anchored walk-forward analysis.

    Unlike rolling walk-forward, the training window always starts at the beginning
    and grows over time. This can provide more stable parameter estimates.

    Args:
        data: OHLCV DataFrame
        strategy_name: Strategy to test
        param_grid: Parameters to optimize
        initial_train_size: Initial training window size
        test_size: Testing window size
        metric: Metric to optimize
        initial_capital: Starting capital
        verbose: Print progress

    Returns:
        WalkForwardResult with per-window and overall results
    """
    from core.backtest.backtester import VectorizedBacktester

    windows = []
    oos_returns = []
    is_params = []
    cumulative_capital = initial_capital

    n_bars = len(data)
    train_end = initial_train_size
    window_num = 0

    while train_end + test_size <= n_bars:
        window_num += 1
        test_end = train_end + test_size

        # Training always starts from beginning
        train_data = data.iloc[:train_end].copy()
        test_data = data.iloc[train_end:test_end].copy()

        if verbose:
            logger.info(f"Window {window_num}: Train [0:{train_end}], Test [{train_end}:{test_end}]")

        try:
            opt_result = grid_search(
                train_data, strategy_name, param_grid, metric=metric, initial_capital=cumulative_capital, verbose=False
            )
            best_params = opt_result.best_params
            is_metric = opt_result.best_metric
        except Exception as e:
            if verbose:
                logger.warning(f"  Optimization failed: {e}")
            train_end += test_size
            continue

        try:
            bt = VectorizedBacktester(test_data, cumulative_capital, 0.001)
            portfolio_df = bt.run(strategy_name, best_params)

            if not portfolio_df.empty:
                final_value = portfolio_df["Portfolio_Value"].iloc[-1]
                oos_return = (final_value - cumulative_capital) / cumulative_capital
                cumulative_capital = final_value
                metrics = bt.get_metrics(portfolio_df)
                oos_sharpe = metrics.get("sharpe_ratio", 0)
            else:
                oos_return = 0
                oos_sharpe = 0
        except Exception as e:
            if verbose:
                logger.warning(f"  Test failed: {e}")
            oos_return = 0
            oos_sharpe = 0

        windows.append(
            {
                "window": window_num,
                "train_start": 0,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": test_end,
                "best_params": best_params,
                "is_metric": is_metric,
                "oos_return": oos_return,
                "oos_sharpe": oos_sharpe,
            }
        )

        oos_returns.append(oos_return)
        is_params.append(best_params)
        train_end += test_size

    overall_return = (cumulative_capital - initial_capital) / initial_capital
    avg_oos_sharpe = np.mean([w["oos_sharpe"] for w in windows]) if windows else 0

    return WalkForwardResult(
        windows=windows,
        overall_return=overall_return,
        overall_sharpe=avg_oos_sharpe,
        out_of_sample_returns=oos_returns,
        in_sample_params=is_params,
    )
