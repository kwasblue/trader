"""
Parameter Optimization Module

Provides grid search and other optimization methods for strategy parameters.
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""

    best_params: dict[str, Any]
    best_metric: float
    all_results: list[dict[str, Any]]
    metric_name: str


def grid_search(
    data,
    strategy_name: str,
    param_grid: dict[str, list[Any]],
    metric: str = "sharpe_ratio",
    initial_capital: float = 10000,
    transaction_cost: float = 0.001,
    n_jobs: int = 1,
    verbose: bool = True,
) -> OptimizationResult:
    """
    Grid search over strategy parameters.

    Args:
        data: OHLCV DataFrame
        strategy_name: Name of strategy to optimize
        param_grid: Dict mapping param names to lists of values
        metric: Metric to optimize ('sharpe_ratio', 'total_return', 'sortino_ratio', 'max_drawdown')
        initial_capital: Starting capital
        transaction_cost: Transaction cost fraction
        n_jobs: Number of parallel jobs (1 = sequential)
        verbose: Print progress

    Returns:
        OptimizationResult with best params and all results
    """
    # Import here to avoid circular imports
    from core.backtest.backtester import VectorizedBacktester

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    if verbose:
        logger.info(f"Testing {len(combinations)} parameter combinations...")

    results = []

    def evaluate_params(params_tuple):
        params = dict(zip(param_names, params_tuple))
        try:
            bt = VectorizedBacktester(data.copy(), initial_capital, transaction_cost)
            portfolio_df = bt.run(strategy_name, params)

            if portfolio_df.empty:
                return None

            metrics = bt.get_metrics(portfolio_df)

            # Calculate the target metric
            if metric == "sharpe_ratio":
                metric_value = metrics.get("sharpe_ratio", -999)
            elif metric == "total_return":
                metric_value = metrics.get("total_return", -999)
            elif metric == "sortino_ratio":
                metric_value = metrics.get("sortino_ratio", -999)
            elif metric == "max_drawdown":
                # Negate so higher is better
                metric_value = -metrics.get("max_drawdown", 1)
            else:
                metric_value = metrics.get(metric, -999)

            # Convert metrics to performance dict format for compatibility
            performance = {
                "Sharpe Ratio": metrics.get("sharpe_ratio", 0),
                "Sortino Ratio": metrics.get("sortino_ratio", 0),
                "Max Drawdown": metrics.get("max_drawdown", 0),
                "Total Return": metrics.get("total_return", 0),
                "Win Rate": metrics.get("win_rate", 0),
                "Profit Factor": metrics.get("profit_factor", 0),
            }

            return {"params": params, "metric_value": metric_value, "performance": performance}
        except Exception as e:
            if verbose:
                logger.warning(f"Error with params {params}: {e}")
            return None

    # Run evaluations
    if n_jobs == 1:
        for i, combo in enumerate(combinations):
            result = evaluate_params(combo)
            if result:
                results.append(result)
            if verbose and (i + 1) % 10 == 0:
                logger.info(f"  Completed {i + 1}/{len(combinations)}")
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(evaluate_params, combo): combo for combo in combinations}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

    if not results:
        raise ValueError("No valid results from optimization")

    # Find best
    best = max(results, key=lambda x: x["metric_value"])

    if verbose:
        logger.info(f"Best {metric}: {best['metric_value']:.4f}")
        logger.info(f"Best params: {best['params']}")

    return OptimizationResult(
        best_params=best["params"], best_metric=best["metric_value"], all_results=results, metric_name=metric
    )


def random_search(
    data,
    strategy_name: str,
    param_distributions: dict[str, Any],
    n_iter: int = 100,
    metric: str = "sharpe_ratio",
    initial_capital: float = 10000,
    transaction_cost: float = 0.001,
    verbose: bool = True,
    seed: int = None,
) -> OptimizationResult:
    """
    Random search over strategy parameters.

    More efficient than grid search for high-dimensional parameter spaces.

    Args:
        data: OHLCV DataFrame
        strategy_name: Name of strategy to optimize
        param_distributions: Dict mapping param names to distributions or lists
        n_iter: Number of random samples to try
        metric: Metric to optimize
        initial_capital: Starting capital
        transaction_cost: Transaction cost fraction
        verbose: Print progress
        seed: Random seed for reproducibility

    Returns:
        OptimizationResult with best params and all results
    """
    import numpy as np

    from core.backtest.backtester import VectorizedBacktester

    if seed is not None:
        np.random.seed(seed)

    results = []

    for i in range(n_iter):
        # Sample parameters
        params = {}
        for name, dist in param_distributions.items():
            if isinstance(dist, list):
                params[name] = np.random.choice(dist)
            elif isinstance(dist, tuple) and len(dist) == 2:
                # Assume (min, max) uniform distribution
                params[name] = np.random.uniform(dist[0], dist[1])
            else:
                params[name] = dist

        try:
            bt = VectorizedBacktester(data.copy(), initial_capital, transaction_cost)
            portfolio_df = bt.run(strategy_name, params)

            if portfolio_df.empty:
                continue

            metrics = bt.get_metrics(portfolio_df)
            metric_value = metrics.get(metric, -999)

            results.append({"params": params, "metric_value": metric_value, "performance": metrics})

            if verbose and (i + 1) % 10 == 0:
                logger.info(f"  Completed {i + 1}/{n_iter}")

        except Exception as e:
            if verbose:
                logger.warning(f"Error with params {params}: {e}")
            continue

    if not results:
        raise ValueError("No valid results from optimization")

    best = max(results, key=lambda x: x["metric_value"])

    if verbose:
        logger.info(f"Best {metric}: {best['metric_value']:.4f}")
        logger.info(f"Best params: {best['params']}")

    return OptimizationResult(
        best_params=best["params"], best_metric=best["metric_value"], all_results=results, metric_name=metric
    )
