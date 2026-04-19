#!/usr/bin/env python3
"""
Example 3: Parameter Optimization

Demonstrates grid search optimization to find optimal strategy parameters.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime

from core.backtest_suite import grid_search, validate_ohlcv_data


def generate_sample_data(days: int = 750) -> pd.DataFrame:
    """Generate synthetic price data with trends."""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # Create price with regime changes
    returns = np.zeros(days)

    # Bull market (first third)
    returns[:days//3] = np.random.randn(days//3) * 0.015 + 0.001
    # Bear market (middle third)
    returns[days//3:2*days//3] = np.random.randn(days//3) * 0.02 - 0.0005
    # Bull market (last third)
    returns[2*days//3:] = np.random.randn(days - 2*days//3) * 0.015 + 0.0008

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
    print("PARAMETER OPTIMIZATION EXAMPLE")
    print("=" * 70)

    # Generate data
    print("\n1. Generating sample data...")
    data = generate_sample_data(750)

    validation = validate_ohlcv_data(data, fix_issues=True)
    data = validation.cleaned_data
    print(f"   Data: {len(data)} bars")

    # Define parameter grid for SMA strategy
    print("\n2. Defining parameter grid...")
    param_grid = {
        'fast': [5, 10, 15, 20, 25],
        'slow': [20, 30, 40, 50, 60]
    }

    total_combinations = len(param_grid['fast']) * len(param_grid['slow'])
    print(f"   Fast periods: {param_grid['fast']}")
    print(f"   Slow periods: {param_grid['slow']}")
    print(f"   Total combinations: {total_combinations}")

    # Run optimization
    print("\n3. Running grid search optimization...")
    print("   Metric: Sharpe Ratio")
    print()

    result = grid_search(
        data=data,
        strategy_name='sma',
        param_grid=param_grid,
        metric='sharpe_ratio',
        initial_capital=10000,
        verbose=True
    )

    # Display results
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    print(f"\nBest Parameters:")
    print(f"   Fast Period: {result.best_params['fast']}")
    print(f"   Slow Period: {result.best_params['slow']}")
    print(f"   Best Sharpe: {result.best_metric:.4f}")

    # Create heatmap-style table
    print("\n\nSharpe Ratio by Parameter Combination:")
    print("-" * 50)

    # Build matrix
    fast_vals = sorted(set(r['params']['fast'] for r in result.all_results))
    slow_vals = sorted(set(r['params']['slow'] for r in result.all_results))

    # Header
    print(f"{'Fast\\Slow':>10}", end='')
    for slow in slow_vals:
        print(f"{slow:>8}", end='')
    print()
    print("-" * (10 + 8 * len(slow_vals)))

    # Rows
    for fast in fast_vals:
        print(f"{fast:>10}", end='')
        for slow in slow_vals:
            if slow <= fast:
                print(f"{'---':>8}", end='')  # Invalid: slow must be > fast
            else:
                # Find result
                for r in result.all_results:
                    if r['params']['fast'] == fast and r['params']['slow'] == slow:
                        sharpe = r['metric_value']
                        # Highlight best
                        if r['params'] == result.best_params:
                            print(f"*{sharpe:>6.2f}*", end='')
                        else:
                            print(f"{sharpe:>8.2f}", end='')
                        break
        print()

    print("\n* = Best combination")

    # Top 5 combinations
    print("\n\nTop 5 Parameter Combinations:")
    print("-" * 50)
    sorted_results = sorted(result.all_results, key=lambda x: x['metric_value'], reverse=True)[:5]

    for i, r in enumerate(sorted_results, 1):
        perf = r['performance']
        print(f"\n{i}. Fast={r['params']['fast']}, Slow={r['params']['slow']}")
        print(f"   Sharpe: {r['metric_value']:.4f}")
        print(f"   Max Drawdown: {perf.get('Max Drawdown', 0):.2%}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
