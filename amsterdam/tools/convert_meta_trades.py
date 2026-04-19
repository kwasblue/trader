#!/usr/bin/env python3
"""
Convert Meta Trades JSONL to Parquet for ML Training.

Reads meta_trades.jsonl, joins entry/exit events by trade_id,
and outputs a training-ready Parquet file.

Usage:
    python convert_meta_trades.py
    python convert_meta_trades.py --input logs/meta_trades.jsonl --output data/trades_for_training.parquet

Output Schema:
    - trade_id: str
    - entry_timestamp: datetime
    - exit_timestamp: datetime
    - symbol: str
    - side: str
    - qty: int
    - entry_price: float
    - exit_price: float
    - All entry features (strategy, regime, atr, etc.)
    - All outcome metrics (pnl_dollars, pnl_percent, mae_percent, mfe_percent, etc.)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas")
    sys.exit(1)


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    Load events from JSONL file.

    Args:
        filepath: Path to JSONL file

    Returns:
        List of event dictionaries
    """
    events = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
    return events


def join_trades(events: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Join entry and exit events by trade_id.

    Args:
        events: List of entry/exit event dictionaries

    Returns:
        DataFrame with joined trade data
    """
    # Separate entries and exits
    entries: Dict[str, Dict] = {}
    exits: Dict[str, Dict] = {}

    for event in events:
        trade_id = event.get("trade_id")
        event_type = event.get("event")

        if not trade_id or not event_type:
            continue

        if event_type == "entry":
            entries[trade_id] = event
        elif event_type == "exit":
            exits[trade_id] = event

    # Join on trade_id
    joined_trades = []

    for trade_id, entry in entries.items():
        exit_event = exits.get(trade_id)

        # Start with entry data
        trade = {
            "trade_id": trade_id,
            "entry_timestamp": entry.get("timestamp"),
            "symbol": entry.get("symbol"),
            "side": entry.get("side"),
            "qty": entry.get("qty"),
            "entry_price": entry.get("price"),
        }

        # Add features
        features = entry.get("features", {})
        for key, value in features.items():
            trade[f"feat_{key}"] = value

        # Add exit data if available
        if exit_event:
            trade["exit_timestamp"] = exit_event.get("timestamp")
            trade["exit_price"] = exit_event.get("price")
            trade["is_complete"] = True

            # Add outcome metrics
            outcome = exit_event.get("outcome", {})
            for key, value in outcome.items():
                trade[f"out_{key}"] = value
        else:
            # Open trade - no exit yet
            trade["exit_timestamp"] = None
            trade["exit_price"] = None
            trade["is_complete"] = False
            trade["out_pnl_dollars"] = None
            trade["out_pnl_percent"] = None
            trade["out_hold_bars"] = None
            trade["out_mae_percent"] = None
            trade["out_mfe_percent"] = None
            trade["out_exit_reason"] = None

        joined_trades.append(trade)

    return pd.DataFrame(joined_trades)


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert column types for efficient storage.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with optimized types
    """
    if df.empty:
        return df

    # Convert timestamps
    if "entry_timestamp" in df.columns:
        df["entry_timestamp"] = pd.to_datetime(df["entry_timestamp"])
    if "exit_timestamp" in df.columns:
        df["exit_timestamp"] = pd.to_datetime(df["exit_timestamp"])

    # Convert numeric columns
    numeric_cols = [
        "qty", "entry_price", "exit_price",
        "feat_atr", "feat_atr_percentile",
        "feat_drawdown_portfolio_pct", "feat_drawdown_symbol_pct",
        "feat_position_size_pct", "feat_hours_since_last_trade",
        "out_pnl_dollars", "out_pnl_percent",
        "out_mae_percent", "out_mfe_percent",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert integer columns
    int_cols = [
        "feat_hour_of_day", "feat_day_of_week",
        "feat_minutes_since_open", "feat_bars_in_regime",
        "feat_signal_strength", "out_hold_bars",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Convert categorical columns
    cat_cols = ["symbol", "side", "feat_strategy", "feat_regime", "out_exit_reason"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def get_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate summary statistics for the trade data.

    Args:
        df: Trade DataFrame

    Returns:
        Dictionary of statistics
    """
    stats = {
        "total_trades": len(df),
        "complete_trades": df["is_complete"].sum() if "is_complete" in df.columns else 0,
        "open_trades": (~df["is_complete"]).sum() if "is_complete" in df.columns else 0,
    }

    # Only calculate P&L stats for complete trades
    complete = df[df["is_complete"]] if "is_complete" in df.columns else df

    if not complete.empty and "out_pnl_dollars" in complete.columns:
        pnl = complete["out_pnl_dollars"].dropna()
        if len(pnl) > 0:
            stats["total_pnl"] = pnl.sum()
            stats["avg_pnl"] = pnl.mean()
            stats["win_rate"] = (pnl > 0).mean()
            stats["profit_factor"] = abs(pnl[pnl > 0].sum() / pnl[pnl < 0].sum()) if pnl[pnl < 0].sum() != 0 else float("inf")

    if "out_hold_bars" in complete.columns:
        hold_bars = complete["out_hold_bars"].dropna()
        if len(hold_bars) > 0:
            stats["avg_hold_bars"] = hold_bars.mean()

    if "symbol" in df.columns:
        stats["symbols"] = df["symbol"].nunique()

    if "feat_strategy" in df.columns:
        stats["strategies"] = df["feat_strategy"].nunique()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert meta trades JSONL to Parquet for ML training"
    )
    parser.add_argument(
        "--input", "-i",
        default="logs/meta_trades.jsonl",
        help="Input JSONL file path (default: logs/meta_trades.jsonl)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/trades_for_training.parquet",
        help="Output Parquet file path (default: data/trades_for_training.parquet)"
    )
    parser.add_argument(
        "--complete-only",
        action="store_true",
        help="Only include complete trades (with both entry and exit)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print statistics after conversion"
    )
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent.parent
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_absolute():
        input_path = script_dir / input_path
    if not output_path.is_absolute():
        output_path = script_dir / output_path

    # Check input exists
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Load and process
    print(f"Loading events from: {input_path}")
    events = load_jsonl(str(input_path))
    print(f"Loaded {len(events)} events")

    # Join trades
    df = join_trades(events)
    print(f"Joined into {len(df)} trades")

    # Filter to complete trades if requested
    if args.complete_only and "is_complete" in df.columns:
        df = df[df["is_complete"]]
        print(f"Filtered to {len(df)} complete trades")

    # Convert types
    df = convert_types(df)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to Parquet
    try:
        df.to_parquet(str(output_path), index=False, engine="pyarrow")
    except ImportError:
        print("Error: pyarrow is required for Parquet output.")
        print("Install with: pip install pyarrow")
        sys.exit(1)
    print(f"Saved to: {output_path}")

    # Print stats if requested
    if args.stats:
        print("\nStatistics:")
        stats = get_stats(df)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
