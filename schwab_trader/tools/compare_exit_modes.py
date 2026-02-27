#!/usr/bin/env python3
"""
Compare Exit Modes

Compare signal-based exits vs SL/TP-only exits to see the impact on R:R.

Usage:
    python tools/compare_exit_modes.py AAPL
    python tools/compare_exit_modes.py AAPL -s sma,macd,momentum
    python tools/compare_exit_modes.py AAPL --days 60
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asyncio
import pandas as pd
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def load_data(symbol: str, days: int) -> pd.DataFrame:
    """Load historical data for a symbol."""
    from core.unified_data_pipeline import UnifiedDataPipeline

    pipeline = UnifiedDataPipeline()
    data = pipeline.get_data(symbol)

    if data is None or data.empty:
        print(f"No cached data for {symbol}, fetching...")

        async def fetch():
            return await pipeline.update_symbols([symbol], days=days)

        asyncio.run(fetch())
        data = pipeline.get_data(symbol)

    if data is None or data.empty:
        raise ValueError(f"Could not load data for {symbol}")

    if len(data) > days:
        data = data.tail(days).reset_index(drop=True)

    return data


def run_comparison(symbol: str, strategies: list, days: int):
    """Run comparison between exit modes."""
    from core.backtest.unified_backtest_runner import UnifiedBacktestRunner, BacktestConfig
    from core.config_loader import get_config

    cfg = get_config()
    tl = cfg.trade_logic

    # Load data
    print(f"\nLoading {days} days of data for {symbol}...")
    data = load_data(symbol, days)
    print(f"Loaded {len(data)} bars")

    runner = UnifiedBacktestRunner(data)

    results = []

    for strategy in strategies:
        for signal_exits in [True, False]:
            mode = "Signal+SL/TP" if signal_exits else "SL/TP Only"

            config = BacktestConfig(
                strategy_name=strategy,
                strategy_params={},
                stop_loss_atr=tl.sl_mult_normal,
                take_profit_atr=tl.tp_mult_normal,
                signal_based_exits=signal_exits,
                initial_capital=100000,
                position_size=0.10,
            )

            try:
                result = runner.run(config)
                trades = result.trades

                if not trades:
                    continue

                # Calculate metrics
                wins = [t for t in trades if t.pnl > 0]
                losses = [t for t in trades if t.pnl <= 0]

                total_trades = len(trades)
                win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
                total_pnl = sum(t.pnl for t in trades)

                avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
                avg_loss = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 0
                rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0

                # Exit reason breakdown
                signal_exits_count = len([t for t in trades if t.exit_reason == "signal"])
                sl_exits = len([t for t in trades if t.exit_reason == "stop_loss"])
                tp_exits = len([t for t in trades if t.exit_reason == "take_profit"])
                eod_exits = len([t for t in trades if t.exit_reason == "end_of_data"])

                results.append({
                    "Strategy": strategy,
                    "Mode": mode,
                    "Trades": total_trades,
                    "Win%": win_rate,
                    "R:R": rr_ratio,
                    "Total PnL": total_pnl,
                    "Avg Win": avg_win,
                    "Avg Loss": avg_loss,
                    "Signal Exits": signal_exits_count,
                    "SL Exits": sl_exits,
                    "TP Exits": tp_exits,
                    "EOD Exits": eod_exits,
                })

            except Exception as e:
                print(f"  Error running {strategy} ({mode}): {e}")

    return results


def print_results(results: list, symbol: str):
    """Print formatted results."""
    if not results:
        print("No results to display")
        return

    print("\n" + "=" * 90)
    print(f"  EXIT MODE COMPARISON - {symbol}")
    print("=" * 90)

    # Group by strategy
    strategies = sorted(set(r["Strategy"] for r in results))

    for strategy in strategies:
        strat_results = [r for r in results if r["Strategy"] == strategy]

        print(f"\n  {strategy.upper()}")
        print("-" * 90)
        print(f"  {'Mode':<15} {'Trades':>7} {'Win%':>7} {'R:R':>6} {'Total PnL':>12} "
              f"{'Signal':>7} {'SL':>5} {'TP':>5} {'EOD':>5}")
        print("-" * 90)

        for r in strat_results:
            print(f"  {r['Mode']:<15} {r['Trades']:>7} {r['Win%']:>6.1f}% {r['R:R']:>6.2f} "
                  f"${r['Total PnL']:>11,.2f} {r['Signal Exits']:>7} {r['SL Exits']:>5} "
                  f"{r['TP Exits']:>5} {r['EOD Exits']:>5}")

        # Show delta
        if len(strat_results) == 2:
            signal_mode = next((r for r in strat_results if r["Mode"] == "Signal+SL/TP"), None)
            sltp_mode = next((r for r in strat_results if r["Mode"] == "SL/TP Only"), None)

            if signal_mode and sltp_mode:
                pnl_delta = sltp_mode["Total PnL"] - signal_mode["Total PnL"]
                rr_delta = sltp_mode["R:R"] - signal_mode["R:R"]
                wr_delta = sltp_mode["Win%"] - signal_mode["Win%"]

                print("-" * 90)
                improved = pnl_delta > 0
                symbol_char = "+" if improved else ""
                print(f"  {'DELTA':<15} {'':>7} {symbol_char}{wr_delta:>6.1f}% {symbol_char}{rr_delta:>6.2f} "
                      f"${symbol_char}{pnl_delta:>11,.2f}")
                print(f"  {'VERDICT':<15} {'SL/TP Only is BETTER' if improved else 'Signal+SL/TP is better'}")

    print("\n" + "=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Compare exit modes")
    parser.add_argument("symbol", help="Symbol to test")
    parser.add_argument("-s", "--strategies", default="sma,momentum,macd",
                        help="Strategies to test (comma-separated)")
    parser.add_argument("-d", "--days", type=int, default=90,
                        help="Days of historical data")

    args = parser.parse_args()

    strategies = [s.strip().lower() for s in args.strategies.split(",")]

    results = run_comparison(args.symbol, strategies, args.days)
    print_results(results, args.symbol)


if __name__ == "__main__":
    main()
