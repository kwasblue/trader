#!/usr/bin/env python3
"""
Example 4: Walk-Forward Analysis

Demonstrates robust out-of-sample testing with rolling train/test windows.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime

from core.backtest_suite import walk_forward_analysis, validate_ohlcv_data


def generate_sample_data(days: int = 1000) -> pd.DataFrame:
    """Generate synthetic price data with regime changes."""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # Create price with different regimes
    returns = np.zeros(days)
    regime_length = days // 4

    # Regime 1: Low volatility uptrend
    returns[:regime_length] = np.random.randn(regime_length) * 0.01 + 0.0008

    # Regime 2: High volatility
    returns[regime_length:2*regime_length] = np.random.randn(regime_length) * 0.025

    # Regime 3: Downtrend
    returns[2*regime_length:3*regime_length] = np.random.randn(regime_length) * 0.015 - 0.0005

    # Regime 4: Recovery
    returns[3*regime_length:] = np.random.randn(days - 3*regime_length) * 0.012 + 0.0006

    prices = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.randn(days) * 0.005),
        'High': prices * (1 + np.abs(np.random.randn(days) * 0.01)),
        'Low': prices * (1 - np.abs(np.random.randn(days) * 0.01)),
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, days)
    })

    data['High'] = data[['Open', 'High', 'Low', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'High', 'Low', 'Close']].min(axis=1)

    return data


def main():
    print("=" * 70)
    print("WALK-FORWARD ANALYSIS EXAMPLE")
    print("=" * 70)

    # Generate data
    print("\n1. Generating sample data...")
    data = generate_sample_data(1000)

    validation = validate_ohlcv_data(data, fix_issues=True)
    data = validation.cleaned_data
    print(f"   Data: {len(data)} bars (~4 years of daily data)")

    # Define parameter grid
    param_grid = {
        'fast': [5, 10, 15, 20],
        'slow': [25, 35, 45]
    }

    print("\n2. Walk-Forward Configuration:")
    print("   Training window: 252 bars (~1 year)")
    print("   Testing window:  63 bars (~3 months)")
    print("   Step size:       63 bars (quarterly roll)")
    print(f"   Parameter grid:  {len(param_grid['fast']) * len(param_grid['slow'])} combinations")

    # Run walk-forward analysis
    print("\n3. Running walk-forward analysis...")
    print("   (Optimizing on each training window, testing on next period)")
    print()

    result = walk_forward_analysis(
        data=data,
        strategy_name='sma',
        param_grid=param_grid,
        train_size=252,    # 1 year training
        test_size=63,      # 3 months testing
        step_size=63,      # Roll quarterly
        metric='sharpe_ratio',
        initial_capital=10000,
        verbose=True
    )

    # Display results
    print("\n" + "=" * 70)
    print("WALK-FORWARD RESULTS")
    print("=" * 70)

    print(f"\nOverall Performance:")
    print(f"   Total Out-of-Sample Return: {result.overall_return:+.2%}")
    print(f"   Average OOS Sharpe Ratio:   {result.overall_sharpe:.4f}")
    print(f"   Windows Tested:             {len(result.windows)}")

    # Per-window breakdown
    print("\n\nPer-Window Breakdown:")
    print("-" * 70)
    print(f"{'Window':>6} {'Train Period':>20} {'Best Params':>20} {'OOS Return':>12}")
    print("-" * 70)

    for w in result.windows:
        params_str = f"fast={w['best_params']['fast']}, slow={w['best_params']['slow']}"
        train_period = f"[{w['train_start']}-{w['train_end']}]"
        print(f"{w['window']:>6} {train_period:>20} {params_str:>20} {w['oos_return']:>+11.2%}")

    # Parameter stability analysis
    print("\n\nParameter Stability Analysis:")
    print("-" * 40)

    fast_counts = {}
    slow_counts = {}
    for params in result.in_sample_params:
        fast = params['fast']
        slow = params['slow']
        fast_counts[fast] = fast_counts.get(fast, 0) + 1
        slow_counts[slow] = slow_counts.get(slow, 0) + 1

    print("\nMost Selected Fast Periods:")
    for fast, count in sorted(fast_counts.items(), key=lambda x: -x[1]):
        pct = count / len(result.windows) * 100
        bar = "█" * int(pct / 5)
        print(f"   {fast:>3}: {bar:<20} ({count} times, {pct:.0f}%)")

    print("\nMost Selected Slow Periods:")
    for slow, count in sorted(slow_counts.items(), key=lambda x: -x[1]):
        pct = count / len(result.windows) * 100
        bar = "█" * int(pct / 5)
        print(f"   {slow:>3}: {bar:<20} ({count} times, {pct:.0f}%)")

    # Return distribution
    print("\n\nOut-of-Sample Return Distribution:")
    print("-" * 40)

    oos_returns = result.out_of_sample_returns
    print(f"   Mean:    {np.mean(oos_returns):+.2%}")
    print(f"   Median:  {np.median(oos_returns):+.2%}")
    print(f"   Std Dev: {np.std(oos_returns):.2%}")
    print(f"   Min:     {np.min(oos_returns):+.2%}")
    print(f"   Max:     {np.max(oos_returns):+.2%}")

    positive = sum(1 for r in oos_returns if r > 0)
    print(f"   Positive: {positive}/{len(oos_returns)} ({positive/len(oos_returns)*100:.0f}%)")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
