"""
Monte Carlo Simulation Module

Provides Monte Carlo simulation for estimating strategy outcome distributions.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MonteCarloResult:
    """Result of Monte Carlo simulation."""

    mean_return: float
    median_return: float
    std_return: float
    percentiles: dict[int, float]
    max_drawdowns: list[float]
    sharpe_ratios: list[float]
    final_values: list[float]
    confidence_interval_95: tuple[float, float]


def monte_carlo_simulation(
    trades: list[dict], initial_capital: float = 10000, n_simulations: int = 1000, seed: int = None
) -> MonteCarloResult:
    """
    Monte Carlo simulation by randomizing trade order.

    Takes a list of historical trades and shuffles their order
    to estimate the distribution of possible outcomes.

    Args:
        trades: List of trade dicts with 'pnl' key
        initial_capital: Starting capital
        n_simulations: Number of simulations to run
        seed: Random seed for reproducibility

    Returns:
        MonteCarloResult with distribution statistics
    """
    if seed is not None:
        np.random.seed(seed)

    if not trades:
        raise ValueError("No trades provided for Monte Carlo simulation")

    # Extract P&L from trades
    trade_pnls = [t.get("pnl", 0) for t in trades if "pnl" in t]
    if not trade_pnls:
        # Try to calculate from price and quantity
        trade_pnls = []
        for t in trades:
            action = t.get("Action", "").upper()
            price = t.get("Price", 0)
            qty = t.get("Quantity", 0)
            if action == "SELL":
                # Approximate P&L (would need entry price for accuracy)
                trade_pnls.append(price * qty * 0.01)  # Assume 1% profit
            elif action == "BUY":
                trade_pnls.append(-price * qty * 0.01)  # Buying is cost

    if not trade_pnls:
        raise ValueError("Could not extract P&L from trades")

    final_values = []
    max_drawdowns = []
    sharpe_ratios = []

    for _ in range(n_simulations):
        # Shuffle trade order
        shuffled = np.random.permutation(trade_pnls)

        # Calculate equity curve
        equity = [initial_capital]
        for pnl in shuffled:
            equity.append(equity[-1] + pnl)

        equity = np.array(equity)
        final_values.append(equity[-1])

        # Calculate max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdowns.append(np.max(drawdown))

        # Calculate Sharpe (simplified)
        returns = np.diff(equity) / equity[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0
        sharpe_ratios.append(sharpe)

    final_values = np.array(final_values)
    returns = (final_values - initial_capital) / initial_capital

    return MonteCarloResult(
        mean_return=np.mean(returns),
        median_return=np.median(returns),
        std_return=np.std(returns),
        percentiles={
            5: np.percentile(returns, 5),
            25: np.percentile(returns, 25),
            50: np.percentile(returns, 50),
            75: np.percentile(returns, 75),
            95: np.percentile(returns, 95),
        },
        max_drawdowns=max_drawdowns,
        sharpe_ratios=sharpe_ratios,
        final_values=final_values.tolist(),
        confidence_interval_95=(np.percentile(returns, 2.5), np.percentile(returns, 97.5)),
    )


def bootstrap_returns(
    returns: np.ndarray, n_simulations: int = 1000, block_size: int = 20, seed: int = None
) -> MonteCarloResult:
    """
    Block bootstrap simulation for returns series.

    Preserves some autocorrelation structure by resampling blocks.

    Args:
        returns: Array of returns
        n_simulations: Number of simulations
        block_size: Size of blocks to resample
        seed: Random seed

    Returns:
        MonteCarloResult with bootstrap distribution
    """
    if seed is not None:
        np.random.seed(seed)

    n_returns = len(returns)
    n_blocks = n_returns // block_size + 1

    final_values = []
    max_drawdowns = []
    sharpe_ratios = []

    for _ in range(n_simulations):
        # Sample blocks with replacement
        block_indices = np.random.randint(0, n_returns - block_size + 1, size=n_blocks)
        sampled_returns = []

        for idx in block_indices:
            sampled_returns.extend(returns[idx : idx + block_size])

        sampled_returns = np.array(sampled_returns[:n_returns])

        # Calculate equity curve
        equity = np.cumprod(1 + sampled_returns)
        final_values.append(equity[-1])

        # Calculate max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdowns.append(np.max(drawdown))

        # Calculate Sharpe
        if np.std(sampled_returns) > 0:
            sharpe = np.mean(sampled_returns) / np.std(sampled_returns) * np.sqrt(252)
        else:
            sharpe = 0
        sharpe_ratios.append(sharpe)

    final_values = np.array(final_values)
    total_returns = final_values - 1  # Convert to returns

    return MonteCarloResult(
        mean_return=np.mean(total_returns),
        median_return=np.median(total_returns),
        std_return=np.std(total_returns),
        percentiles={
            5: np.percentile(total_returns, 5),
            25: np.percentile(total_returns, 25),
            50: np.percentile(total_returns, 50),
            75: np.percentile(total_returns, 75),
            95: np.percentile(total_returns, 95),
        },
        max_drawdowns=max_drawdowns,
        sharpe_ratios=sharpe_ratios,
        final_values=final_values.tolist(),
        confidence_interval_95=(np.percentile(total_returns, 2.5), np.percentile(total_returns, 97.5)),
    )
