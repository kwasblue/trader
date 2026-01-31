#!/usr/bin/env python3
"""
Example 2: Strategy Comparison

Compares multiple trading strategies on the same dataset.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime

from core.backtest_suite import VectorizedBacktester, validate_ohlcv_data
from strategies.strategy_registry import list_strategies


def generate_sample_data(days: int = 500) -> pd.DataFrame:
    """Generate synthetic price data for demonstration."""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    returns = np.random.randn(days) * 0.02 + 0.0003
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
    print("STRATEGY COMPARISON EXAMPLE")
    print("=" * 70)

    # Generate and validate data
    print("\n1. Generating sample data...")
    data = generate_sample_data(500)

    validation = validate_ohlcv_data(data, fix_issues=True)
    if not validation.is_valid:
        print(f"   Data validation failed: {validation.errors}")
        return
    data = validation.cleaned_data
    print(f"   Data validated: {len(data)} bars")

    # Define strategies to compare
    strategies = {
        'SMA (10/30)': ('sma', {'fast': 10, 'slow': 30}),
        'EMA (12/26)': ('ema', {'short_window': 12, 'long_window': 26}),
        'MACD': ('macd', {'fast_window': 12, 'slow_window': 26, 'signal_window': 9}),
        'RSI (14)': ('rsi', {'window': 14, 'oversold': 30, 'overbought': 70}),
        'Bollinger': ('bollinger', {'window': 20, 'num_std': 2}),
        'Momentum (20)': ('momentum', {'lookback': 20}),
        'Mean Reversion': ('meanreversion', {'window': 14, 'threshold': 1.0}),
    }

    # Run backtests
    print("\n2. Running backtests...")
    results = {}

    for name, (strategy, params) in strategies.items():
        try:
            bt = VectorizedBacktester(data.copy(), initial_capital=10000)
            portfolio = bt.run(strategy, params)
            metrics = bt.get_metrics(portfolio)
            results[name] = metrics
            print(f"   ✓ {name}")
        except Exception as e:
            print(f"   ✗ {name}: {e}")

    # Create comparison table
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)

    # Header
    print(f"\n{'Strategy':<20} {'Return':>10} {'Sharpe':>10} {'Sortino':>10} {'MaxDD':>10} {'WinRate':>10}")
    print("-" * 70)

    # Sort by Sharpe ratio
    sorted_results = sorted(results.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True)

    for name, metrics in sorted_results:
        print(f"{name:<20} "
              f"{metrics['total_return']:>9.2%} "
              f"{metrics['sharpe_ratio']:>10.2f} "
              f"{metrics['sortino_ratio']:>10.2f} "
              f"{metrics['max_drawdown']:>9.2%} "
              f"{metrics['win_rate']:>9.2%}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    best_return = max(sorted_results, key=lambda x: x[1]['total_return'])
    best_sharpe = sorted_results[0]  # Already sorted by Sharpe
    lowest_dd = min(sorted_results, key=lambda x: x[1]['max_drawdown'])

    print(f"\n   Best Return:      {best_return[0]} ({best_return[1]['total_return']:.2%})")
    print(f"   Best Sharpe:      {best_sharpe[0]} ({best_sharpe[1]['sharpe_ratio']:.2f})")
    print(f"   Lowest Drawdown:  {lowest_dd[0]} ({lowest_dd[1]['max_drawdown']:.2%})")

    # Buy and hold comparison
    buy_hold_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
    print(f"\n   Buy & Hold Return: {buy_hold_return:.2%}")

    outperformers = [name for name, m in sorted_results if m['total_return'] > buy_hold_return]
    print(f"   Strategies beating B&H: {len(outperformers)}/{len(sorted_results)}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
