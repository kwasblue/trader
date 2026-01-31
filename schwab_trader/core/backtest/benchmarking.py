"""
Benchmark Comparison Module

Provides tools for comparing strategy performance against benchmarks.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BenchmarkComparison:
    """Comparison of strategy vs benchmark."""
    strategy_return: float
    benchmark_return: float
    excess_return: float
    strategy_sharpe: float
    benchmark_sharpe: float
    beta: float
    alpha: float
    information_ratio: float
    tracking_error: float
    up_capture: float
    down_capture: float


def compare_to_benchmark(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02
) -> BenchmarkComparison:
    """
    Compare strategy performance to a benchmark.

    Args:
        strategy_returns: Series of strategy returns
        benchmark_returns: Series of benchmark returns
        risk_free_rate: Annualized risk-free rate

    Returns:
        BenchmarkComparison with various metrics
    """
    # Align the series
    aligned = pd.DataFrame({
        'strategy': strategy_returns,
        'benchmark': benchmark_returns
    }).dropna()

    strat = aligned['strategy']
    bench = aligned['benchmark']

    # Basic returns
    strategy_total = (1 + strat).prod() - 1
    benchmark_total = (1 + bench).prod() - 1
    excess_return = strategy_total - benchmark_total

    # Sharpe ratios (annualized)
    daily_rf = risk_free_rate / 252
    strat_sharpe = (strat.mean() - daily_rf) / strat.std() * np.sqrt(252) if strat.std() > 0 else 0
    bench_sharpe = (bench.mean() - daily_rf) / bench.std() * np.sqrt(252) if bench.std() > 0 else 0

    # Beta and Alpha
    cov = np.cov(strat, bench)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0
    alpha = (strat.mean() - daily_rf - beta * (bench.mean() - daily_rf)) * 252

    # Information Ratio
    active_returns = strat - bench
    tracking_error = active_returns.std() * np.sqrt(252)
    information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() > 0 else 0

    # Up/Down capture
    up_periods = bench > 0
    down_periods = bench < 0

    if up_periods.sum() > 0:
        up_capture = strat[up_periods].mean() / bench[up_periods].mean()
    else:
        up_capture = 0

    if down_periods.sum() > 0:
        down_capture = strat[down_periods].mean() / bench[down_periods].mean()
    else:
        down_capture = 0

    return BenchmarkComparison(
        strategy_return=strategy_total,
        benchmark_return=benchmark_total,
        excess_return=excess_return,
        strategy_sharpe=strat_sharpe,
        benchmark_sharpe=bench_sharpe,
        beta=beta,
        alpha=alpha,
        information_ratio=information_ratio,
        tracking_error=tracking_error,
        up_capture=up_capture,
        down_capture=down_capture
    )


def calculate_rolling_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 252,
    risk_free_rate: float = 0.02
) -> pd.DataFrame:
    """
    Calculate rolling performance metrics vs benchmark.

    Args:
        strategy_returns: Series of strategy returns
        benchmark_returns: Series of benchmark returns
        window: Rolling window size (default 252 = 1 year)
        risk_free_rate: Annualized risk-free rate

    Returns:
        DataFrame with rolling metrics
    """
    aligned = pd.DataFrame({
        'strategy': strategy_returns,
        'benchmark': benchmark_returns
    }).dropna()

    daily_rf = risk_free_rate / 252

    rolling_alpha = []
    rolling_beta = []
    rolling_sharpe = []
    rolling_info_ratio = []
    dates = []

    for i in range(window, len(aligned)):
        strat_window = aligned['strategy'].iloc[i - window:i]
        bench_window = aligned['benchmark'].iloc[i - window:i]

        # Beta
        cov = np.cov(strat_window, bench_window)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0

        # Alpha
        alpha = (strat_window.mean() - daily_rf - beta * (bench_window.mean() - daily_rf)) * 252

        # Sharpe
        sharpe = (strat_window.mean() - daily_rf) / strat_window.std() * np.sqrt(252) if strat_window.std() > 0 else 0

        # Information Ratio
        active = strat_window - bench_window
        ir = active.mean() / active.std() * np.sqrt(252) if active.std() > 0 else 0

        rolling_alpha.append(alpha)
        rolling_beta.append(beta)
        rolling_sharpe.append(sharpe)
        rolling_info_ratio.append(ir)
        dates.append(aligned.index[i])

    return pd.DataFrame({
        'date': dates,
        'alpha': rolling_alpha,
        'beta': rolling_beta,
        'sharpe': rolling_sharpe,
        'information_ratio': rolling_info_ratio
    }).set_index('date')


def calculate_drawdown_analysis(returns: pd.Series) -> dict:
    """
    Detailed drawdown analysis.

    Args:
        returns: Series of returns

    Returns:
        Dictionary with drawdown statistics
    """
    equity = (1 + returns).cumprod()
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak

    # Find drawdown periods
    in_drawdown = drawdown < 0
    drawdown_starts = in_drawdown & ~in_drawdown.shift(1).fillna(False)
    drawdown_ends = ~in_drawdown & in_drawdown.shift(1).fillna(False)

    # Calculate drawdown statistics
    drawdowns = []
    current_dd_start = None

    for i, (is_start, is_end, dd) in enumerate(zip(drawdown_starts, drawdown_ends, drawdown)):
        if is_start:
            current_dd_start = i
        if is_end and current_dd_start is not None:
            dd_slice = drawdown.iloc[current_dd_start:i]
            drawdowns.append({
                'start_idx': current_dd_start,
                'end_idx': i,
                'depth': dd_slice.min(),
                'duration': i - current_dd_start,
                'recovery_time': i - dd_slice.idxmin() if dd_slice.min() < 0 else 0
            })
            current_dd_start = None

    if drawdowns:
        max_dd = min(d['depth'] for d in drawdowns)
        avg_dd = np.mean([d['depth'] for d in drawdowns])
        avg_duration = np.mean([d['duration'] for d in drawdowns])
        avg_recovery = np.mean([d['recovery_time'] for d in drawdowns if d['recovery_time'] > 0])
    else:
        max_dd = 0
        avg_dd = 0
        avg_duration = 0
        avg_recovery = 0

    return {
        'max_drawdown': max_dd,
        'avg_drawdown': avg_dd,
        'avg_duration_bars': avg_duration,
        'avg_recovery_bars': avg_recovery if drawdowns else 0,
        'num_drawdowns': len(drawdowns),
        'current_drawdown': drawdown.iloc[-1] if len(drawdown) > 0 else 0,
        'drawdown_periods': drawdowns
    }
