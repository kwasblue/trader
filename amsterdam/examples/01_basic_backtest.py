#!/usr/bin/env python3
"""
Example 1: Basic Backtest

Demonstrates how to run a simple backtest with the SMA crossover strategy.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

import numpy as np
import pandas as pd

from core.backtest_suite import VectorizedBacktester, validate_ohlcv_data


def generate_sample_data(days: int = 500) -> pd.DataFrame:
    """Generate synthetic price data for demonstration."""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

    # Random walk with drift
    returns = np.random.randn(days) * 0.02 + 0.0003  # 2% daily vol, slight upward drift
    prices = 100 * np.exp(np.cumsum(returns))

    # Generate OHLC from close prices
    data = pd.DataFrame(
        {
            "Date": dates,
            "Open": prices * (1 + np.random.randn(days) * 0.005),
            "High": prices * (1 + np.abs(np.random.randn(days) * 0.01)),
            "Low": prices * (1 - np.abs(np.random.randn(days) * 0.01)),
            "Close": prices,
            "Volume": np.random.randint(100000, 1000000, days),
        }
    )

    # Ensure OHLC consistency
    data["High"] = data[["Open", "High", "Low", "Close"]].max(axis=1)
    data["Low"] = data[["Open", "High", "Low", "Close"]].min(axis=1)

    return data


def main():
    print("=" * 60)
    print("BASIC BACKTEST EXAMPLE")
    print("=" * 60)

    # Generate sample data
    print("\n1. Generating sample price data...")
    data = generate_sample_data(500)

    # Validate data
    validation = validate_ohlcv_data(data, fix_issues=True)
    if not validation.is_valid:
        print(f"   Data validation issues: {validation.errors}")
    data = validation.cleaned_data

    print(f"   Data range: {data['Date'].min().date()} to {data['Date'].max().date()}")
    print(f"   Total bars: {len(data)}")

    # Initialize backtester
    print("\n2. Initializing backtester...")
    initial_capital = 10000
    backtester = VectorizedBacktester(
        data=data,
        initial_capital=initial_capital,
        transaction_cost=0.001,  # 0.1% per trade
    )

    # Run backtest with SMA strategy
    print("\n3. Running SMA crossover strategy...")
    print("   Parameters: fast=10, slow=30")

    results = backtester.run(strategy_name="sma", strategy_params={"fast": 10, "slow": 30})

    if results.empty:
        print("   ERROR: Backtest returned no results")
        return

    # Calculate performance metrics
    print("\n4. Calculating performance metrics...")
    metrics = backtester.get_metrics(results)

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    final_value = results["Portfolio_Value"].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital

    print("\nPortfolio Performance:")
    print(f"   Initial Capital:    ${initial_capital:,.2f}")
    print(f"   Final Value:        ${final_value:,.2f}")
    print(f"   Total Return:       {total_return:+.2%}")

    print("\nRisk Metrics:")
    print(f"   Sharpe Ratio:       {metrics['sharpe_ratio']:.4f}")
    print(f"   Sortino Ratio:      {metrics['sortino_ratio']:.4f}")
    print(f"   Max Drawdown:       {metrics['max_drawdown']:.2%}")
    print(f"   Win Rate:           {metrics['win_rate']:.2%}")
    print(f"   Profit Factor:      {metrics['profit_factor']:.2f}")

    print("\nTrade Statistics:")
    print(f"   Total Trades:       {metrics['num_trades']}")

    print("\n" + "=" * 60)
    print("Backtest complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
