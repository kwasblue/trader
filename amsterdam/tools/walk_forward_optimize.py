#!/usr/bin/env python3
"""
Walk-Forward Optimization for ConfluenceRegimeStrategy

Walk-forward avoids overfitting by:
1. Train on window N
2. Test on window N+1 (out-of-sample)
3. Roll forward, repeat
4. Aggregate OOS results

This gives realistic performance expectations.

Usage:
    python tools/walk_forward_optimize.py --symbol AAPL
    python tools/walk_forward_optimize.py --symbol AAPL --train-days 120 --test-days 30
"""

import argparse
import sys
from pathlib import Path
from itertools import product
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


@dataclass
class TradeResult:
    """Single trade result."""
    entry_price: float
    exit_price: float
    direction: int  # 1 = long, -1 = short
    pnl_pct: float
    bars_held: int


def load_data(symbol: str) -> pd.DataFrame | None:
    """Load historical data for a symbol."""
    # Try multiple data sources

    # 1. Try processed data directory
    data_path = ROOT / "data" / "data_storage" / "proc_data" / f"{symbol}.parquet"
    if data_path.exists():
        df = pd.read_parquet(data_path)
        print(f"Loaded {len(df)} bars from {data_path}")
        return df

    # 2. Try CSV
    csv_path = ROOT / "data" / "data_storage" / "proc_data" / f"{symbol}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(f"Loaded {len(df)} bars from {csv_path}")
        return df

    # 3. Try fetching via Alpaca
    try:
        from core.broker.alpaca_broker import AlpacaBroker
        print(f"Fetching {symbol} data from Alpaca...")
        broker = AlpacaBroker()
        df = broker.get_historical_bars(symbol, days=500)
        if df is not None and len(df) > 0:
            print(f"Fetched {len(df)} bars from Alpaca")
            return df
    except Exception as e:
        print(f"Alpaca fetch failed: {e}")

    return None


def backtest_signals(signals: list[int], prices: np.ndarray) -> list[TradeResult]:
    """
    Backtest signals and return individual trade results.
    """
    trades = []
    position = 0
    entry_price = 0
    entry_bar = 0

    for i in range(1, len(signals)):
        signal = signals[i]
        price = prices[i]

        # Entry
        if signal == 1 and position == 0:
            position = 1
            entry_price = price
            entry_bar = i
        elif signal == -1 and position == 0:
            position = -1
            entry_price = price
            entry_bar = i

        # Exit on opposite signal
        elif signal == -1 and position == 1:
            pnl = (price - entry_price) / entry_price
            trades.append(TradeResult(entry_price, price, 1, pnl, i - entry_bar))
            position = 0
        elif signal == 1 and position == -1:
            pnl = (entry_price - price) / entry_price
            trades.append(TradeResult(entry_price, price, -1, pnl, i - entry_bar))
            position = 0

    # Close open position at end
    if position != 0:
        price = prices[-1]
        if position == 1:
            pnl = (price - entry_price) / entry_price
        else:
            pnl = (entry_price - price) / entry_price
        trades.append(TradeResult(entry_price, price, position, pnl, len(signals) - entry_bar))

    return trades


def calculate_metrics(trades: list[TradeResult]) -> dict:
    """Calculate performance metrics from trades."""
    if not trades:
        return {
            "total_return": 0,
            "num_trades": 0,
            "win_rate": 0,
            "avg_pnl": 0,
            "sharpe": 0,
            "profit_factor": 0,
            "avg_bars_held": 0,
        }

    pnls = np.array([t.pnl_pct for t in trades])

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    total_return = np.prod(1 + pnls) - 1
    win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    profit_factor = abs(np.sum(wins) / np.sum(losses)) if np.sum(losses) != 0 else float('inf')

    # Simple Sharpe approximation
    sharpe = np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(len(pnls))

    return {
        "total_return": total_return,
        "num_trades": len(trades),
        "win_rate": win_rate,
        "avg_pnl": np.mean(pnls),
        "sharpe": sharpe,
        "profit_factor": profit_factor,
        "avg_bars_held": np.mean([t.bars_held for t in trades]),
    }


def optimize_on_window(df: pd.DataFrame, param_grid: dict) -> tuple[dict, dict]:
    """
    Find best parameters on a training window.
    Returns (best_params, best_metrics).
    """
    from strategies.strategy_registry.confluence_regime_strategy import ConfluenceRegimeStrategy

    best_sharpe = -999
    best_params = None
    best_metrics = None

    prices = df["Close"].values if "Close" in df.columns else df["close"].values

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    for combo in product(*param_values):
        params = dict(zip(param_names, combo))

        try:
            strategy = ConfluenceRegimeStrategy(**params)
            signals = strategy.generate_signals_vectorized(df)

            if signals is None or sum(1 for s in signals if s != 0) == 0:
                continue

            trades = backtest_signals(signals, prices)
            metrics = calculate_metrics(trades)

            # Optimize for Sharpe, but require minimum trades
            if metrics["num_trades"] >= 3 and metrics["sharpe"] > best_sharpe:
                best_sharpe = metrics["sharpe"]
                best_params = params
                best_metrics = metrics

        except Exception as e:
            continue

    return best_params, best_metrics


def walk_forward_optimize(
    df: pd.DataFrame,
    train_days: int = 120,
    test_days: int = 30,
    param_grid: dict | None = None
) -> dict:
    """
    Perform walk-forward optimization.

    Args:
        df: Full historical data
        train_days: Days to use for training each window
        test_days: Days to test (out-of-sample)
        param_grid: Parameter search space

    Returns:
        Aggregated out-of-sample results
    """
    from strategies.strategy_registry.confluence_regime_strategy import ConfluenceRegimeStrategy

    if param_grid is None:
        param_grid = {
            "min_confluence": [2, 3],
            "vol_percentile_max": [40, 50, 60],
            "trend_threshold": [0.3, 0.4, 0.5, 0.6],
            "regime_lookback": [30, 50],
            "fast_period": [5, 8, 12],
            "slow_period": [15, 21, 30],
        }

    total_days = len(df)
    window_size = train_days + test_days

    if total_days < window_size:
        print(f"Not enough data: {total_days} days, need {window_size}")
        return {}

    all_oos_trades = []
    window_results = []

    prices = df["Close"].values if "Close" in df.columns else df["close"].values

    # Walk forward
    start = 0
    window_num = 0

    print(f"\nWalk-Forward Optimization")
    print(f"Train: {train_days} days, Test: {test_days} days")
    print(f"Total combinations per window: {np.prod([len(v) for v in param_grid.values()])}")
    print("=" * 70)

    while start + window_size <= total_days:
        window_num += 1
        train_end = start + train_days
        test_end = train_end + test_days

        train_df = df.iloc[start:train_end]
        test_df = df.iloc[train_end:test_end]

        print(f"\nWindow {window_num}: Train [{start}:{train_end}], Test [{train_end}:{test_end}]")

        # Optimize on training data
        best_params, train_metrics = optimize_on_window(train_df, param_grid)

        if best_params is None:
            print(f"  No valid parameters found in training")
            start += test_days
            continue

        print(f"  Best params: confluence={best_params['min_confluence']}, "
              f"vol={best_params['vol_percentile_max']}, trend={best_params['trend_threshold']:.1f}")
        print(f"  Train: {train_metrics['num_trades']} trades, "
              f"Sharpe={train_metrics['sharpe']:.2f}, WR={train_metrics['win_rate']*100:.0f}%")

        # Test on out-of-sample data
        strategy = ConfluenceRegimeStrategy(**best_params)

        # Need to include some lookback from training for indicators
        lookback = max(best_params.get("regime_lookback", 50),
                       best_params.get("slow_period", 21)) + 10
        test_with_lookback = df.iloc[train_end - lookback:test_end]

        signals = strategy.generate_signals_vectorized(test_with_lookback)

        if signals:
            # Only count signals in the actual test period
            test_signals = signals[lookback:]
            test_prices = prices[train_end:test_end]

            oos_trades = backtest_signals(test_signals, test_prices)
            oos_metrics = calculate_metrics(oos_trades)

            print(f"  OOS:   {oos_metrics['num_trades']} trades, "
                  f"Return={oos_metrics['total_return']*100:+.1f}%, "
                  f"WR={oos_metrics['win_rate']*100:.0f}%")

            all_oos_trades.extend(oos_trades)
            window_results.append({
                "window": window_num,
                "params": best_params,
                "train_sharpe": train_metrics["sharpe"],
                "oos_return": oos_metrics["total_return"],
                "oos_trades": oos_metrics["num_trades"],
            })

        # Roll forward
        start += test_days

    # Aggregate OOS results
    print("\n" + "=" * 70)
    print("AGGREGATED OUT-OF-SAMPLE RESULTS")
    print("=" * 70)

    if all_oos_trades:
        final_metrics = calculate_metrics(all_oos_trades)

        print(f"\nTotal OOS trades: {final_metrics['num_trades']}")
        print(f"Total OOS return: {final_metrics['total_return']*100:+.2f}%")
        print(f"Win rate:         {final_metrics['win_rate']*100:.1f}%")
        print(f"Profit factor:    {final_metrics['profit_factor']:.2f}")
        print(f"Sharpe ratio:     {final_metrics['sharpe']:.2f}")
        print(f"Avg bars held:    {final_metrics['avg_bars_held']:.1f}")

        # Compare to buy and hold
        bh_return = (prices[-1] - prices[0]) / prices[0]
        print(f"\nBuy & Hold return: {bh_return*100:+.2f}%")

        # Parameter stability
        print("\nParameter stability across windows:")
        if window_results:
            params_used = [r["params"] for r in window_results]
            for param in ["min_confluence", "vol_percentile_max", "trend_threshold"]:
                values = [p[param] for p in params_used]
                print(f"  {param}: {values} (stable)" if len(set(values)) == 1 else f"  {param}: {values} (varies)")

        return final_metrics
    else:
        print("No out-of-sample trades generated")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Optimization")
    parser.add_argument("--symbol", "-s", type=str, default="AAPL", help="Symbol to test")
    parser.add_argument("--train-days", type=int, default=120, help="Training window size")
    parser.add_argument("--test-days", type=int, default=30, help="Test window size")
    args = parser.parse_args()

    print(f"Walk-Forward Optimization for {args.symbol}")
    print("=" * 70)

    # Load data
    df = load_data(args.symbol)

    if df is None or len(df) == 0:
        print(f"Could not load data for {args.symbol}")
        print("\nGenerating synthetic data for demonstration...")

        np.random.seed(42)
        n = 500

        # More realistic synthetic data with regime changes
        returns = np.zeros(n)
        regime = "trending"

        for i in range(n):
            # Regime switches
            if np.random.random() < 0.02:  # 2% chance of regime change
                regime = "ranging" if regime == "trending" else "trending"

            if regime == "trending":
                returns[i] = np.random.randn() * 0.015 + 0.001  # Slight drift
            else:
                returns[i] = np.random.randn() * 0.02  # Mean reverting

        close = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            "Open": close * (1 + np.random.randn(n) * 0.003),
            "High": close * (1 + np.abs(np.random.randn(n) * 0.008)),
            "Low": close * (1 - np.abs(np.random.randn(n) * 0.008)),
            "Close": close,
            "Volume": np.random.randint(1000000, 10000000, n),
        })
        print(f"Generated {len(df)} bars of synthetic data")

    # Run walk-forward optimization
    results = walk_forward_optimize(
        df,
        train_days=args.train_days,
        test_days=args.test_days,
    )

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
If OOS Sharpe > 0.5 and Win Rate > 45%:
  → Strategy shows promise, worth paper trading

If parameters are stable across windows:
  → More likely to be robust, less likely overfit

If OOS << Train performance:
  → Overfitting detected, simplify the strategy

Next steps:
  1. Paper trade with the most stable parameters
  2. Track whether OOS predictions hold in live markets
  3. Adjust position sizing based on confidence
""")


if __name__ == "__main__":
    main()
