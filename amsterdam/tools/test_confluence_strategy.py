#!/usr/bin/env python3
"""
Test the ConfluenceRegimeStrategy.

This script:
1. Loads historical data
2. Runs the strategy
3. Shows signal distribution and basic stats
4. Compares to a simple SMA strategy

Run: python tools/test_confluence_strategy.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def load_sample_data(symbol: str = "AAPL", days: int = 365) -> pd.DataFrame:
    """Load historical data from the data pipeline."""
    try:
        from data.unified_data_pipeline import UnifiedDataPipeline

        pipeline = UnifiedDataPipeline()
        df = pipeline.get_data(symbol, resample="1D")
        if df is not None and len(df) > 0:
            return df.tail(days)
    except Exception as e:
        print(f"Could not load real data: {e}")

    # Fallback: generate synthetic data
    print("Using synthetic data for testing...")
    np.random.seed(42)
    n = days

    # Generate realistic price series with trends and ranges
    returns = np.random.randn(n) * 0.02

    # Add some trending periods
    returns[50:100] += 0.005  # Uptrend
    returns[150:200] -= 0.005  # Downtrend
    returns[250:300] += 0.003  # Mild uptrend

    close = 100 * np.exp(np.cumsum(returns))

    return pd.DataFrame({
        "Open": close * (1 + np.random.randn(n) * 0.005),
        "High": close * (1 + np.abs(np.random.randn(n) * 0.01)),
        "Low": close * (1 - np.abs(np.random.randn(n) * 0.01)),
        "Close": close,
        "Volume": np.random.randint(1000000, 10000000, n),
    })


def simple_backtest(signals: list[int], prices: np.ndarray) -> dict:
    """
    Simple backtest: buy on 1, sell on -1, hold otherwise.
    Returns basic performance metrics.
    """
    position = 0
    entry_price = 0
    trades = []
    equity = [1.0]

    for i in range(1, len(signals)):
        signal = signals[i]
        price = prices[i]

        # Entry
        if signal == 1 and position == 0:
            position = 1
            entry_price = price
        elif signal == -1 and position == 0:
            position = -1
            entry_price = price

        # Exit on opposite signal
        elif signal == -1 and position == 1:
            pnl = (price - entry_price) / entry_price
            trades.append(pnl)
            equity.append(equity[-1] * (1 + pnl))
            position = 0
        elif signal == 1 and position == -1:
            pnl = (entry_price - price) / entry_price
            trades.append(pnl)
            equity.append(equity[-1] * (1 + pnl))
            position = 0
        else:
            equity.append(equity[-1])

    # Close any open position
    if position != 0:
        if position == 1:
            pnl = (prices[-1] - entry_price) / entry_price
        else:
            pnl = (entry_price - prices[-1]) / entry_price
        trades.append(pnl)
        equity.append(equity[-1] * (1 + pnl))

    trades = np.array(trades) if trades else np.array([0])
    equity = np.array(equity)

    # Calculate metrics
    total_return = equity[-1] - 1
    win_rate = np.mean(trades > 0) if len(trades) > 0 else 0
    avg_win = np.mean(trades[trades > 0]) if np.any(trades > 0) else 0
    avg_loss = np.mean(trades[trades < 0]) if np.any(trades < 0) else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    # Sharpe (annualized, assuming daily)
    returns = np.diff(equity) / equity[:-1]
    sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_dd = np.max(drawdown)

    return {
        "total_return": total_return,
        "num_trades": len(trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def main():
    print("=" * 60)
    print("  CONFLUENCE REGIME STRATEGY TEST")
    print("=" * 60)

    # Load data
    df = load_sample_data("AAPL", days=500)
    print(f"\nData: {len(df)} bars")
    print(f"Date range: {df.index[0] if hasattr(df.index[0], 'strftime') else 'N/A'} to {df.index[-1] if hasattr(df.index[-1], 'strftime') else 'N/A'}")

    # Test Confluence Strategy
    print("\n--- Confluence Regime Strategy ---")
    from strategies.strategy_registry.confluence_regime_strategy import ConfluenceRegimeStrategy

    strategy = ConfluenceRegimeStrategy(
        regime_lookback=50,
        trend_threshold=0.6,
        vol_percentile_max=40,
        min_confluence=3,
    )

    signals = strategy.generate_signals_vectorized(df)
    if signals is None:
        print("Not enough data for strategy")
        return

    # Signal distribution
    signals_arr = np.array(signals)
    buy_signals = np.sum(signals_arr == 1)
    sell_signals = np.sum(signals_arr == -1)
    hold_signals = np.sum(signals_arr == 0)

    print(f"Signal distribution:")
    print(f"  BUY:  {buy_signals:4d} ({100*buy_signals/len(signals):.1f}%)")
    print(f"  SELL: {sell_signals:4d} ({100*sell_signals/len(signals):.1f}%)")
    print(f"  HOLD: {hold_signals:4d} ({100*hold_signals/len(signals):.1f}%)")

    # Backtest
    prices = df["Close"].values
    results = simple_backtest(signals, prices)

    print(f"\nBacktest results:")
    print(f"  Total return:  {results['total_return']*100:+.2f}%")
    print(f"  Num trades:    {results['num_trades']}")
    print(f"  Win rate:      {results['win_rate']*100:.1f}%")
    print(f"  Profit factor: {results['profit_factor']:.2f}")
    print(f"  Sharpe ratio:  {results['sharpe']:.2f}")
    print(f"  Max drawdown:  {results['max_drawdown']*100:.1f}%")

    # Compare to simple SMA
    print("\n--- Simple SMA Crossover (baseline) ---")
    from strategies.strategy_registry.sma_strategy import SMAStrategy

    sma_strategy = SMAStrategy(fast=10, slow=30)
    sma_signals = sma_strategy.generate_signals_vectorized(df)

    if sma_signals:
        sma_arr = np.array(sma_signals)
        print(f"Signal distribution:")
        print(f"  BUY:  {np.sum(sma_arr == 1):4d} ({100*np.sum(sma_arr == 1)/len(sma_signals):.1f}%)")
        print(f"  SELL: {np.sum(sma_arr == -1):4d} ({100*np.sum(sma_arr == -1)/len(sma_signals):.1f}%)")

        sma_results = simple_backtest(sma_signals, prices)
        print(f"\nBacktest results:")
        print(f"  Total return:  {sma_results['total_return']*100:+.2f}%")
        print(f"  Num trades:    {sma_results['num_trades']}")
        print(f"  Win rate:      {sma_results['win_rate']*100:.1f}%")
        print(f"  Profit factor: {sma_results['profit_factor']:.2f}")
        print(f"  Sharpe ratio:  {sma_results['sharpe']:.2f}")
        print(f"  Max drawdown:  {sma_results['max_drawdown']*100:.1f}%")

    # Buy and hold comparison
    print("\n--- Buy and Hold (baseline) ---")
    bh_return = (prices[-1] - prices[0]) / prices[0]
    print(f"  Total return:  {bh_return*100:+.2f}%")

    print("\n" + "=" * 60)
    print("  KEY INSIGHT")
    print("=" * 60)
    print("""
The confluence strategy trades LESS frequently than SMA crossover.
That's the point - it's filtering for quality setups.

If it has:
  - Fewer trades but similar/better returns → better risk-adjusted
  - Higher win rate → better signal quality
  - Lower drawdown → better risk management

Then the filtering is working.

If it underperforms, adjust:
  - min_confluence (3 is strict, try 2)
  - vol_percentile_max (40 is strict, try 50-60)
  - trend_threshold (0.6 is moderate, try 0.4-0.8)
""")


if __name__ == "__main__":
    main()
