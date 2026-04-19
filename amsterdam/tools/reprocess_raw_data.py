#!/usr/bin/env python3
"""
Reprocess Raw Data to Processed Data.

This script reads all raw data files and reprocesses them through the ML pipeline
to create complete processed data files.

Use this when:
- proc_data files are truncated/incomplete
- Raw data has more bars than processed data
- You want to regenerate all indicators

Usage:
    # Reprocess specific symbols
    python tools/reprocess_raw_data.py --symbols AAPL MSFT NVDA

    # Reprocess all symbols with raw data
    python tools/reprocess_raw_data.py --all

    # Dry run (check what would be processed)
    python tools/reprocess_raw_data.py --all --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from data.processor import Processor


def get_raw_data_files(raw_path: Path) -> dict[str, Path]:
    """Get all raw data files and their symbols."""
    files = {}
    for f in raw_path.glob("raw_*_file.json"):
        # Extract symbol: raw_AAPL_file.json -> AAPL
        parts = f.stem.split("_")
        if len(parts) >= 2:
            symbol = parts[1]
            files[symbol] = f
    return files


def load_raw_data(filepath: Path) -> list[dict]:
    """Load raw candle data from JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    # Handle different formats
    if isinstance(data, dict):
        if "candles" in data:
            return data["candles"]
        elif "bars" in data:
            return data["bars"]
    elif isinstance(data, list):
        return data

    return []


def process_symbol(symbol: str, raw_path: Path, proc_path: Path, dry_run: bool = False) -> int:
    """
    Process a single symbol's raw data.

    Returns:
        Number of bars processed
    """
    raw_file = raw_path / f"raw_{symbol}_file.json"
    proc_file = proc_path / f"proc_{symbol}_file.json"

    if not raw_file.exists():
        print(f"  [{symbol}] SKIP - No raw data file")
        return 0

    # Load raw data
    candles = load_raw_data(raw_file)
    if not candles:
        print(f"  [{symbol}] SKIP - Empty raw data")
        return 0

    # Check current proc data
    proc_bars = 0
    if proc_file.exists():
        try:
            with open(proc_file) as f:
                proc_data = json.load(f)
                proc_bars = len(proc_data.get("bars", proc_data) if isinstance(proc_data, dict) else proc_data)
        except Exception:
            proc_bars = 0

    raw_bars = len(candles)

    if dry_run:
        status = "NEEDS UPDATE" if raw_bars > proc_bars else "OK"
        print(f"  [{symbol}] {status}: raw={raw_bars}, proc={proc_bars}")
        return raw_bars

    print(f"  [{symbol}] Processing {raw_bars} bars (was {proc_bars})...")

    # Convert to DataFrame
    df = pd.DataFrame(candles)

    # Ensure datetime column exists
    if "datetime" not in df.columns and "Date" not in df.columns:
        print(f"  [{symbol}] ERROR - No datetime column")
        return 0

    # Process through ML pipeline
    try:
        processor = Processor(stock=symbol, frame=df)
        processed_df = processor.ml_process(
            sma_window=200,
            ema_window=50,
            scaling_method="standard",
            include_scaled=True,
            include_pca=False,
        )

        if processed_df is None or processed_df.empty:
            print(f"  [{symbol}] ERROR - Processing returned empty")
            return 0

        # Save to JSON
        save_processed_data(symbol, processed_df, proc_path)

        print(f"  [{symbol}] SUCCESS: {len(processed_df)} bars processed")
        return len(processed_df)

    except Exception as e:
        print(f"  [{symbol}] ERROR - {e}")
        return 0


def save_processed_data(symbol: str, df: pd.DataFrame, proc_path: Path) -> None:
    """Save processed DataFrame to JSON file."""
    file_path = proc_path / f"proc_{symbol}_file.json"

    # Make a copy
    df_copy = df.copy()

    # Reset index if DatetimeIndex
    if isinstance(df_copy.index, pd.DatetimeIndex):
        df_copy = df_copy.reset_index()

    # Convert datetime columns to ISO strings
    for col in df_copy.columns:
        if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
        elif df_copy[col].dtype == "object":
            df_copy[col] = df_copy[col].apply(lambda x: x.isoformat() if hasattr(x, "isoformat") else x)

    # Handle NaN and infinity
    df_copy = df_copy.where(df_copy.notna(), None)
    df_copy = df_copy.replace({float("inf"): None, float("-inf"): None})

    # Save
    data = df_copy.to_dict(orient="records")
    with open(file_path, "w") as f:
        json.dump({"bars": data}, f)


def main():
    parser = argparse.ArgumentParser(
        description="Reprocess raw data to create complete processed data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--symbols", "-s", nargs="+", help="Symbols to reprocess")
    parser.add_argument("--all", "-a", action="store_true", help="Process all symbols with raw data")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would be processed without doing it")
    parser.add_argument("--raw-path", default="data/data_storage/raw_data", help="Path to raw data files")
    parser.add_argument("--proc-path", default="data/data_storage/proc_data", help="Path to processed data files")

    args = parser.parse_args()

    # Resolve paths
    raw_path = Path(args.raw_path)
    proc_path = Path(args.proc_path)

    if not raw_path.is_absolute():
        raw_path = ROOT / raw_path
    if not proc_path.is_absolute():
        proc_path = ROOT / proc_path

    # Ensure output path exists
    proc_path.mkdir(parents=True, exist_ok=True)

    # Get symbols to process
    available = get_raw_data_files(raw_path)

    if args.all:
        symbols = sorted(available.keys())
    elif args.symbols:
        symbols = args.symbols
    else:
        print("Error: Specify --symbols or --all")
        sys.exit(1)

    if not symbols:
        print("No symbols to process")
        sys.exit(0)

    print(f"\n{'=' * 60}")
    print(f"{'DRY RUN - ' if args.dry_run else ''}REPROCESSING RAW DATA")
    print(f"{'=' * 60}")
    print(f"Raw path:  {raw_path}")
    print(f"Proc path: {proc_path}")
    print(f"Symbols:   {len(symbols)}")
    print(f"{'=' * 60}\n")

    # Process each symbol
    total_bars = 0
    success_count = 0

    for symbol in symbols:
        if symbol not in available:
            print(f"  [{symbol}] SKIP - No raw data file")
            continue

        bars = process_symbol(symbol, raw_path, proc_path, args.dry_run)
        if bars > 0:
            total_bars += bars
            success_count += 1

    # Summary
    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {success_count}/{len(symbols)} symbols, {total_bars} total bars")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
