#!/usr/bin/env python3
"""
Example 5: Monte Carlo Simulation

Demonstrates confidence interval estimation by randomizing trade order.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime

from core.backtest_suite import (
    VectorizedBacktester,
    monte_carlo_simulation,
    validate_ohlcv_data
)


def generate_sample_data(days: int = 500) -> pd.DataFrame:
    """Generate synthetic price data."""
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
    print("MONTE CARLO SIMULATION EXAMPLE")
    print("=" * 70)

    # Generate data and run backtest
    print("\n1. Running initial backtest...")
    data = generate_sample_data(500)
    validation = validate_ohlcv_data(data, fix_issues=True)
    data = validation.cleaned_data

    initial_capital = 10000
    bt = VectorizedBacktester(data, initial_capital=initial_capital)
    results = bt.run('ema', {'short_window': 12, 'long_window': 26})
    metrics = bt.get_metrics(results)

    print(f"   Strategy: EMA Crossover (12/26)")
    print(f"   Bars: {len(data)}")
    print(f"   Actual Return: {metrics['total_return']:.2%}")
    print(f"   Actual Sharpe: {metrics['sharpe_ratio']:.2f}")

    # Create synthetic trades from returns
    returns = results['Strategy_Return'].values
    trades = []
    for r in returns:
        if r != 0:
            pnl = r * initial_capital
            trades.append({'pnl': pnl})

    print(f"   Total Trades: {len(trades)}")

    # Run Monte Carlo simulation
    print("\n2. Running Monte Carlo simulation...")
    print("   Simulations: 5000")
    print("   (Randomizing trade order to estimate outcome distribution)")

    mc = monte_carlo_simulation(
        trades=trades,
        initial_capital=initial_capital,
        n_simulations=5000,
        seed=42
    )

    # Display results
    print("\n" + "=" * 70)
    print("MONTE CARLO RESULTS")
    print("=" * 70)

    print("\nReturn Distribution:")
    print("-" * 40)
    print(f"   Mean Return:     {mc.mean_return:+.2%}")
    print(f"   Median Return:   {mc.median_return:+.2%}")
    print(f"   Std Deviation:   {mc.std_return:.2%}")

    print("\n   Percentiles:")
    for pct in [5, 25, 50, 75, 95]:
        print(f"      {pct:>3}th: {mc.percentiles[pct]:+.2%}")

    print(f"\n   95% Confidence Interval:")
    print(f"      Lower: {mc.confidence_interval_95[0]:+.2%}")
    print(f"      Upper: {mc.confidence_interval_95[1]:+.2%}")

    # Drawdown analysis
    print("\nMaximum Drawdown Distribution:")
    print("-" * 40)
    dd_mean = np.mean(mc.max_drawdowns)
    dd_median = np.median(mc.max_drawdowns)
    dd_95 = np.percentile(mc.max_drawdowns, 95)
    dd_worst = np.max(mc.max_drawdowns)

    print(f"   Mean Max DD:   {dd_mean:.2%}")
    print(f"   Median Max DD: {dd_median:.2%}")
    print(f"   95th pct DD:   {dd_95:.2%}")
    print(f"   Worst Case:    {dd_worst:.2%}")

    # Sharpe distribution
    print("\nSharpe Ratio Distribution:")
    print("-" * 40)
    sharpe_mean = np.mean(mc.sharpe_ratios)
    sharpe_median = np.median(mc.sharpe_ratios)
    sharpe_5 = np.percentile(mc.sharpe_ratios, 5)
    sharpe_95 = np.percentile(mc.sharpe_ratios, 95)

    print(f"   Mean Sharpe:   {sharpe_mean:.2f}")
    print(f"   Median Sharpe: {sharpe_median:.2f}")
    print(f"   5th pct:       {sharpe_5:.2f}")
    print(f"   95th pct:      {sharpe_95:.2f}")

    # Final value distribution
    print("\nFinal Portfolio Value Distribution:")
    print("-" * 40)
    fv = np.array(mc.final_values)
    print(f"   Mean:    ${np.mean(fv):,.2f}")
    print(f"   Median:  ${np.median(fv):,.2f}")
    print(f"   5th pct: ${np.percentile(fv, 5):,.2f}")
    print(f"   95th pct: ${np.percentile(fv, 95):,.2f}")

    # Probability analysis
    print("\nProbability Analysis:")
    print("-" * 40)
    prob_profit = sum(1 for v in fv if v > initial_capital) / len(fv)
    prob_10pct = sum(1 for v in fv if v > initial_capital * 1.10) / len(fv)
    prob_loss_10pct = sum(1 for v in fv if v < initial_capital * 0.90) / len(fv)

    print(f"   P(Profit):         {prob_profit:.1%}")
    print(f"   P(Return > 10%):   {prob_10pct:.1%}")
    print(f"   P(Loss > 10%):     {prob_loss_10pct:.1%}")

    # ASCII histogram
    print("\n\nReturn Distribution Histogram:")
    print("-" * 50)

    returns_pct = [(v / initial_capital - 1) * 100 for v in fv]
    bins = np.linspace(min(returns_pct), max(returns_pct), 21)
    hist, _ = np.histogram(returns_pct, bins=bins)
    max_count = max(hist)

    for i in range(len(hist)):
        bin_start = bins[i]
        bin_end = bins[i + 1]
        bar_length = int(hist[i] / max_count * 40)
        bar = "█" * bar_length
        print(f"   {bin_start:>6.1f}% - {bin_end:>6.1f}%: {bar}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
