#!/usr/bin/env python3
"""
Quick Backtest - Adaptive Features Comparison

Simplified version that just compares the 4 configurations.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.unified_data_pipeline import UnifiedDataPipeline


def normalize_data(df):
    """Normalize column names to lowercase."""
    # Create copy with lowercase columns
    result = df.copy()

    # Map common column names
    column_map = {
        "Date": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }

    result = result.rename(columns=column_map)

    # Keep only OHLCV + timestamp
    keep_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    result = result[[c for c in keep_cols if c in result.columns]]

    return result


def simple_backtest(symbol, bars, regime_stable=False, sizing_continuous=False):
    """Run simple backtest comparing configurations."""

    # Normalize data
    bars = normalize_data(bars)

    # Simple strategy: buy when price > 20-day MA
    bars["ma20"] = bars["close"].rolling(20).mean()
    bars["signal"] = (bars["close"] > bars["ma20"]).astype(int)

    # Calculate returns
    bars["returns"] = bars["close"].pct_change()

    # Strategy returns
    bars["strategy_returns"] = bars["signal"].shift(1) * bars["returns"]

    # Apply position sizing if continuous
    if sizing_continuous:
        # Simple volatility-based sizing
        bars["vol"] = bars["returns"].rolling(20).std()
        bars["vol_percentile"] = bars["vol"].rank(pct=True) * 100

        # Size multiplier: lower vol = higher size
        bars["size_mult"] = 1.0 - (bars["vol_percentile"] / 200)  # 0.5 to 1.0
        bars["size_mult"] = bars["size_mult"].clip(0.25, 1.0)

        bars["strategy_returns"] = bars["strategy_returns"] * bars["size_mult"]

    # Calculate metrics
    total_ret = (1 + bars["strategy_returns"].dropna()).prod() - 1
    sharpe = bars["strategy_returns"].mean() / bars["strategy_returns"].std() * np.sqrt(252)

    # Max drawdown
    cum_returns = (1 + bars["strategy_returns"].fillna(0)).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()

    # Number of trades (signals changes)
    num_trades = (bars["signal"].diff() != 0).sum()

    return {
        "total_return": total_ret * 100,
        "sharpe": sharpe,
        "max_drawdown": abs(max_dd) * 100,
        "num_trades": num_trades,
    }


def main():
    """Run quick comparison."""
    logging.basicConfig(level=logging.WARNING)

    # Test symbols
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]

    pipeline = UnifiedDataPipeline()

    all_results = []

    for symbol in symbols:
        print(f"\nTesting {symbol}...")

        # Load data
        bars = pipeline.get_data(symbol, timeframe="day")

        if bars is None or bars.empty:
            print("  No data, skipping")
            continue

        # Limit to last 750 bars
        if len(bars) > 750:
            bars = bars.tail(750).copy()

        print(f"  Loaded {len(bars)} bars")

        # Test 4 configurations
        configs = [
            (False, False, "Baseline"),
            (True, False, "Stable Regimes"),
            (False, True, "Continuous Sizing"),
            (True, True, "Full Adaptive"),
        ]

        for stable, continuous, name in configs:
            result = simple_backtest(symbol, bars, stable, continuous)
            result["symbol"] = symbol
            result["config"] = name
            all_results.append(result)

    # Display results
    df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)

    for symbol in df["symbol"].unique():
        print(f"\n{symbol}:")
        symbol_df = df[df["symbol"] == symbol]

        for _, row in symbol_df.iterrows():
            print(
                f"  {row['config']:20s}  Return: {row['total_return']:6.1f}%  "
                f"Sharpe: {row['sharpe']:5.2f}  MaxDD: {row['max_drawdown']:5.1f}%  "
                f"Trades: {row['num_trades']:3.0f}"
            )

    # Summary comparison
    print("\n" + "=" * 80)
    print("AVERAGE ACROSS SYMBOLS")
    print("=" * 80)

    summary = df.groupby("config").agg(
        {"total_return": "mean", "sharpe": "mean", "max_drawdown": "mean", "num_trades": "mean"}
    )

    print(summary.to_string())

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    baseline = summary.loc["Baseline"]
    full = summary.loc["Full Adaptive"]

    sharpe_improve = (full["sharpe"] - baseline["sharpe"]) / baseline["sharpe"] * 100
    dd_improve = (baseline["max_drawdown"] - full["max_drawdown"]) / baseline["max_drawdown"] * 100

    print(f"Sharpe Improvement: {sharpe_improve:+.1f}%")
    print(f"Drawdown Reduction: {dd_improve:+.1f}%")

    if sharpe_improve > 20:
        print("\n✅ STRONG IMPROVEMENT - Consider deployment")
    elif sharpe_improve > 10:
        print("\n⚠️  MODERATE IMPROVEMENT - Tune and retest")
    else:
        print("\n❌ MINIMAL IMPROVEMENT - More work needed")


if __name__ == "__main__":
    main()
