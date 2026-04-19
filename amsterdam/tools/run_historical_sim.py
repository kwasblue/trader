#!/usr/bin/env python3
"""
Run Historical Data Simulation for ML Training Data Generation.

This script runs simulations using real historical market data instead of
synthetic GBM data, generating meta-trade logs for ML model training.

Usage:
    # Run with default settings (AAPL, MSFT, NVDA)
    python tools/run_historical_sim.py

    # Run with specific symbols
    python tools/run_historical_sim.py --symbols AAPL GOOGL

    # Run with custom number of bars
    python tools/run_historical_sim.py --bars 1000

    # Run with higher slippage for conservative estimates
    python tools/run_historical_sim.py --slippage 0.002

Output:
    - logs/meta_trades_historical.jsonl: Raw trade data
    - Use tools/convert_meta_trades.py to convert to Parquet for training
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.simulator.simulation import SimConfig, SimulationRunner


async def run_simulation(args: argparse.Namespace) -> None:
    """Run the historical data simulation."""

    # Build config
    cfg = SimConfig(
        symbols=args.symbols,
        steps=args.bars,
        bar_sleep=args.bar_sleep,
        starting_cash=args.cash,
        use_historical_data=True,
        historical_data_path=args.data_path,
        historical_start_index=args.start_index,
        slippage=args.slippage,
        commission=args.commission,
        meta_logging_enabled=True,
        meta_log_file=args.output,
        warmup_bars=args.warmup,
    )

    print("\n" + "=" * 60)
    print("HISTORICAL DATA SIMULATION")
    print("=" * 60)
    print(f"Symbols:    {cfg.symbols}")
    print(f"Bars:       {cfg.steps}")
    print(f"Data path:  {cfg.historical_data_path}")
    print(f"Slippage:   {cfg.slippage:.4f} ({cfg.slippage * 100:.2f}%)")
    print(f"Commission: ${cfg.commission:.2f}")
    print(f"Output:     logs/{cfg.meta_log_file}")
    print("=" * 60 + "\n")

    # Run simulation
    runner = SimulationRunner(cfg)
    await runner.run()

    # Print summary
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Convert to Parquet for training:")
    print("   python tools/convert_meta_trades.py \\")
    print(f"       --input logs/{cfg.meta_log_file} \\")
    print("       --output data/trades_historical.parquet \\")
    print("       --stats")
    print()
    print("2. Load in pandas for analysis:")
    print("   import pandas as pd")
    print("   df = pd.read_parquet('data/trades_historical.parquet')")
    print("   print(df.describe())")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run historical data simulation for ML training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--symbols", "-s", nargs="+", default=["AAPL", "MSFT", "NVDA"], help="Symbols to simulate")
    parser.add_argument("--bars", "-b", type=int, default=500, help="Number of bars to simulate")
    parser.add_argument(
        "--data-path",
        default="data/data_storage/proc_data",
        help="Path to historical data files (use proc_data for indicators)",
    )
    parser.add_argument(
        "--start-index", type=int, default=50, help="Starting bar index (skip early bars for indicator warmup)"
    )
    parser.add_argument("--warmup", type=int, default=200, help="Warmup bars for indicators")
    parser.add_argument("--slippage", type=float, default=0.001, help="Slippage as fraction (0.001 = 0.1%% = 10 bps)")
    parser.add_argument("--commission", type=float, default=0.0, help="Commission per trade in dollars")
    parser.add_argument("--cash", type=float, default=100_000.0, help="Starting cash")
    parser.add_argument("--bar-sleep", type=float, default=0.001, help="Sleep between bars (lower = faster)")
    parser.add_argument(
        "--output", "-o", default="meta_trades_historical.jsonl", help="Output JSONL filename (in logs/)"
    )

    args = parser.parse_args()

    # Validate symbols have data
    data_path = Path(args.data_path)
    if not data_path.exists():
        # Try relative to script
        data_path = ROOT / args.data_path

    if data_path.exists():
        available = []
        for sym in args.symbols:
            # Check for both proc_* and raw_* file naming conventions
            proc_file = data_path / f"proc_{sym}_file.json"
            raw_file = data_path / f"raw_{sym}_file.json"
            if proc_file.exists() or raw_file.exists():
                available.append(sym)
            else:
                print(f"Warning: No data file for {sym}, skipping")
        args.symbols = available

    if not args.symbols:
        print("Error: No valid symbols with data files")
        sys.exit(1)

    asyncio.run(run_simulation(args))


if __name__ == "__main__":
    main()
