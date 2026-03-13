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


def run_comparison(symbols: list, strategies: list, days: int):
    """Run comparison between exit modes."""
    from core.backtest.unified_backtest_runner import UnifiedBacktestRunner, BacktestConfig
    from core.config_loader import get_config

    cfg = get_config()
    tl = cfg.trade_logic

    results = []

    for symbol in symbols:
        # Load data
        print(f"\nLoading {days} days of data for {symbol}...")
        try:
            data = load_data(symbol, days)
            print(f"  Loaded {len(data)} bars")
        except Exception as e:
            print(f"  Error loading data: {e}")
            continue

        runner = UnifiedBacktestRunner(data)

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
                        "Symbol": symbol,
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


def print_results(results: list, symbols: list):
    """Print formatted results."""
    if not results:
        print("No results to display")
        return

    print("\n" + "=" * 90)
    print(f"  EXIT MODE COMPARISON - {len(symbols)} symbols")
    print("=" * 90)

    # Aggregate by strategy and mode
    strategies = sorted(set(r["Strategy"] for r in results))

    for strategy in strategies:
        strat_results = [r for r in results if r["Strategy"] == strategy]

        # Aggregate by mode
        signal_results = [r for r in strat_results if r["Mode"] == "Signal+SL/TP"]
        sltp_results = [r for r in strat_results if r["Mode"] == "SL/TP Only"]

        print(f"\n  {strategy.upper()}")
        print("-" * 90)
        print(f"  {'Mode':<15} {'Trades':>7} {'Win%':>7} {'R:R':>6} {'Total PnL':>12} "
              f"{'Signal':>7} {'SL':>5} {'TP':>5} {'EOD':>5}")
        print("-" * 90)

        for mode_name, mode_results in [("Signal+SL/TP", signal_results), ("SL/TP Only", sltp_results)]:
            if not mode_results:
                continue

            total_trades = sum(r["Trades"] for r in mode_results)
            total_wins = sum(r["Trades"] * r["Win%"] / 100 for r in mode_results)
            win_pct = (total_wins / total_trades * 100) if total_trades > 0 else 0

            total_pnl = sum(r["Total PnL"] for r in mode_results)
            avg_rr = sum(r["R:R"] * r["Trades"] for r in mode_results) / total_trades if total_trades > 0 else 0

            signal_exits = sum(r["Signal Exits"] for r in mode_results)
            sl_exits = sum(r["SL Exits"] for r in mode_results)
            tp_exits = sum(r["TP Exits"] for r in mode_results)
            eod_exits = sum(r["EOD Exits"] for r in mode_results)

            print(f"  {mode_name:<15} {total_trades:>7} {win_pct:>6.1f}% {avg_rr:>6.2f} "
                  f"${total_pnl:>11,.2f} {signal_exits:>7} {sl_exits:>5} "
                  f"{tp_exits:>5} {eod_exits:>5}")

        # Show delta
        if signal_results and sltp_results:
            signal_pnl = sum(r["Total PnL"] for r in signal_results)
            sltp_pnl = sum(r["Total PnL"] for r in sltp_results)
            pnl_delta = sltp_pnl - signal_pnl

            signal_trades = sum(r["Trades"] for r in signal_results)
            sltp_trades = sum(r["Trades"] for r in sltp_results)

            signal_wr = sum(r["Trades"] * r["Win%"] / 100 for r in signal_results) / signal_trades * 100 if signal_trades > 0 else 0
            sltp_wr = sum(r["Trades"] * r["Win%"] / 100 for r in sltp_results) / sltp_trades * 100 if sltp_trades > 0 else 0
            wr_delta = sltp_wr - signal_wr

            print("-" * 90)
            improved = pnl_delta > 0
            symbol_char = "+" if pnl_delta >= 0 else ""
            wr_char = "+" if wr_delta >= 0 else ""
            print(f"  {'DELTA':<15} {'':>7} {wr_char}{wr_delta:>6.1f}% {'':>6} "
                  f"${symbol_char}{pnl_delta:>11,.2f}")
            print(f"  {'VERDICT':<15} {'SL/TP Only is BETTER' if improved else 'Signal+SL/TP is better'}")

    # Overall summary
    print("\n" + "=" * 90)
    print("  OVERALL SUMMARY")
    print("-" * 90)

    signal_total = sum(r["Total PnL"] for r in results if r["Mode"] == "Signal+SL/TP")
    sltp_total = sum(r["Total PnL"] for r in results if r["Mode"] == "SL/TP Only")
    delta = sltp_total - signal_total

    print(f"  Signal+SL/TP Total PnL: ${signal_total:>12,.2f}")
    print(f"  SL/TP Only Total PnL:   ${sltp_total:>12,.2f}")
    print(f"  Delta:                  ${delta:>12,.2f}")
    print(f"\n  {'>>> SL/TP Only WINS <<<' if delta > 0 else '>>> Signal+SL/TP WINS <<<'}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Compare exit modes")
    parser.add_argument("symbols", help="Symbol(s) to test (comma-separated)")
    parser.add_argument("-s", "--strategies", default="sma,momentum,macd",
                        help="Strategies to test (comma-separated)")
    parser.add_argument("-d", "--days", type=int, default=90,
                        help="Days of historical data")

    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    strategies = [s.strip().lower() for s in args.strategies.split(",")]

    results = run_comparison(symbols, strategies, args.days)
    print_results(results, symbols)


if __name__ == "__main__":
    main()
