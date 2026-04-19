#!/usr/bin/env python3
"""
Analyze ML Filter Shadow Mode Results

Compares ML filter predictions (shadow log) against actual trade outcomes
to measure how effective the filter would be.

Usage:
    python tools/analyze_ml_shadow.py
    python tools/analyze_ml_shadow.py --shadow logs/ml_filter_shadow.jsonl --trades logs/meta_trades_live.jsonl
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def load_shadow_log(path: str) -> list[dict]:
    """Load shadow predictions."""
    predictions = []
    with open(path) as f:
        for line in f:
            try:
                predictions.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return predictions


def load_trade_outcomes(path: str) -> dict[str, dict]:
    """Load trade outcomes, keyed by trade_id."""
    entries = {}
    exits = {}

    with open(path) as f:
        for line in f:
            try:
                event = json.loads(line.strip())
                trade_id = event.get("trade_id")

                if event.get("event") == "entry":
                    entries[trade_id] = event
                elif event.get("event") == "exit":
                    exits[trade_id] = event
            except json.JSONDecodeError:
                continue

    # Merge entries with exits
    trades = {}
    for trade_id, entry in entries.items():
        if trade_id in exits:
            exit_event = exits[trade_id]
            pnl = exit_event.get("outcome", {}).get("pnl_dollars", 0)
            trades[trade_id] = {
                "symbol": entry["symbol"],
                "entry_time": entry["timestamp"],
                "exit_time": exit_event["timestamp"],
                "pnl": pnl,
                "won": pnl > 0,
                "strategy": entry.get("features", {}).get("strategy", "unknown"),
            }

    return trades


def match_predictions_to_outcomes(
    predictions: list[dict], trades: dict[str, dict], time_window_minutes: int = 5
) -> list[dict]:
    """Match shadow predictions to actual trade outcomes."""
    matched = []

    # Index trades by symbol and approximate time
    trades_by_symbol = defaultdict(list)
    for trade_id, trade in trades.items():
        trades_by_symbol[trade["symbol"]].append({"trade_id": trade_id, **trade})

    for pred in predictions:
        symbol = pred["symbol"]
        pred_time = datetime.fromisoformat(pred["timestamp"].replace("Z", "+00:00"))

        # Find matching trade within time window
        best_match = None
        best_diff = timedelta(minutes=time_window_minutes + 1)

        for trade in trades_by_symbol.get(symbol, []):
            trade_time = datetime.fromisoformat(trade["entry_time"].replace("Z", "+00:00"))
            diff = abs(trade_time - pred_time)

            if diff < best_diff and diff <= timedelta(minutes=time_window_minutes):
                best_diff = diff
                best_match = trade

        if best_match:
            matched.append(
                {
                    "symbol": symbol,
                    "prediction_time": pred["timestamp"],
                    "score": pred["score"],
                    "threshold": pred["threshold"],
                    "would_approve": pred["would_approve"],
                    "would_reject": pred["would_reject"],
                    "strategy": pred.get("strategy", "unknown"),
                    "actual_pnl": best_match["pnl"],
                    "actual_won": best_match["won"],
                    "trade_id": best_match["trade_id"],
                }
            )

    return matched


def analyze_results(matched: list[dict]) -> dict[str, Any]:
    """Calculate effectiveness metrics."""
    if not matched:
        return {"error": "No matched predictions found"}

    # Split by filter decision
    would_approve = [m for m in matched if m["would_approve"]]
    would_reject = [m for m in matched if m["would_reject"]]

    # Calculate metrics
    total = len(matched)
    total_wins = sum(1 for m in matched if m["actual_won"])
    baseline_win_rate = total_wins / total if total > 0 else 0

    # Approved trades
    approved_wins = sum(1 for m in would_approve if m["actual_won"])
    approved_win_rate = approved_wins / len(would_approve) if would_approve else 0
    approved_pnl = sum(m["actual_pnl"] for m in would_approve)

    # Rejected trades
    rejected_wins = sum(1 for m in would_reject if m["actual_won"])
    rejected_losses = sum(1 for m in would_reject if not m["actual_won"])
    rejected_win_rate = rejected_wins / len(would_reject) if would_reject else 0
    rejected_pnl = sum(m["actual_pnl"] for m in would_reject)

    # Filter effectiveness
    correct_rejections = rejected_losses  # Rejected and actually lost
    incorrect_rejections = rejected_wins  # Rejected but actually won

    return {
        "total_predictions": total,
        "matched_to_outcomes": len(matched),
        "baseline": {
            "total_trades": total,
            "wins": total_wins,
            "win_rate": f"{baseline_win_rate:.1%}",
            "total_pnl": f"${sum(m['actual_pnl'] for m in matched):,.2f}",
        },
        "with_filter": {
            "trades_taken": len(would_approve),
            "trades_blocked": len(would_reject),
            "block_rate": f"{len(would_reject) / total:.1%}" if total > 0 else "0%",
        },
        "approved_trades": {
            "count": len(would_approve),
            "wins": approved_wins,
            "win_rate": f"{approved_win_rate:.1%}",
            "total_pnl": f"${approved_pnl:,.2f}",
            "avg_pnl": f"${approved_pnl / len(would_approve):,.2f}" if would_approve else "$0",
        },
        "rejected_trades": {
            "count": len(would_reject),
            "wins": rejected_wins,
            "losses": rejected_losses,
            "win_rate": f"{rejected_win_rate:.1%}",
            "total_pnl": f"${rejected_pnl:,.2f}",
            "avg_pnl": f"${rejected_pnl / len(would_reject):,.2f}" if would_reject else "$0",
        },
        "filter_effectiveness": {
            "correct_rejections": correct_rejections,
            "incorrect_rejections": incorrect_rejections,
            "rejection_accuracy": f"{correct_rejections / len(would_reject):.1%}" if would_reject else "N/A",
            "pnl_saved": f"${-rejected_pnl:,.2f}" if rejected_pnl < 0 else f"${0:,.2f}",
            "win_rate_improvement": f"{approved_win_rate - baseline_win_rate:+.1%}",
        },
    }


def print_report(analysis: dict[str, Any]) -> None:
    """Print formatted analysis report."""
    print("\n" + "=" * 60)
    print("ML FILTER SHADOW MODE ANALYSIS")
    print("=" * 60)

    if "error" in analysis:
        print(f"\nError: {analysis['error']}")
        return

    print(f"\nTotal predictions matched: {analysis['matched_to_outcomes']}")

    print("\n" + "-" * 40)
    print("BASELINE (No Filter)")
    print("-" * 40)
    b = analysis["baseline"]
    print(f"  Trades: {b['total_trades']}")
    print(f"  Wins: {b['wins']}")
    print(f"  Win Rate: {b['win_rate']}")
    print(f"  Total P&L: {b['total_pnl']}")

    print("\n" + "-" * 40)
    print("WITH ML FILTER")
    print("-" * 40)
    wf = analysis["with_filter"]
    print(f"  Trades Taken: {wf['trades_taken']}")
    print(f"  Trades Blocked: {wf['trades_blocked']} ({wf['block_rate']})")

    print("\n  APPROVED TRADES:")
    at = analysis["approved_trades"]
    print(f"    Count: {at['count']}")
    print(f"    Wins: {at['wins']}")
    print(f"    Win Rate: {at['win_rate']}")
    print(f"    Total P&L: {at['total_pnl']}")
    print(f"    Avg P&L: {at['avg_pnl']}")

    print("\n  REJECTED TRADES (would have blocked):")
    rt = analysis["rejected_trades"]
    print(f"    Count: {rt['count']}")
    print(f"    Actual Wins: {rt['wins']}")
    print(f"    Actual Losses: {rt['losses']}")
    print(f"    Win Rate: {rt['win_rate']}")
    print(f"    Total P&L: {rt['total_pnl']}")
    print(f"    Avg P&L: {rt['avg_pnl']}")

    print("\n" + "-" * 40)
    print("FILTER EFFECTIVENESS")
    print("-" * 40)
    fe = analysis["filter_effectiveness"]
    print(f"  Correct Rejections (blocked losers): {fe['correct_rejections']}")
    print(f"  Incorrect Rejections (blocked winners): {fe['incorrect_rejections']}")
    print(f"  Rejection Accuracy: {fe['rejection_accuracy']}")
    print(f"  P&L Saved by Blocking: {fe['pnl_saved']}")
    print(f"  Win Rate Improvement: {fe['win_rate_improvement']}")

    print("\n" + "=" * 60)

    # Recommendation
    print("\nRECOMMENDATION:")
    try:
        win_rate_delta = float(fe["win_rate_improvement"].replace("%", "").replace("+", "")) / 100
        if win_rate_delta > 0.05:
            print("  ✓ Filter shows >5% win rate improvement - consider enabling")
        elif win_rate_delta > 0:
            print("  ~ Filter shows modest improvement - may be worth enabling")
        else:
            print("  ✗ Filter does not improve win rate - keep disabled")
    except:
        print("  Unable to calculate recommendation")

    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze ML Filter shadow mode results")
    parser.add_argument("--shadow", default="logs/ml_filter_shadow.jsonl", help="Path to shadow log file")
    parser.add_argument("--trades", default="logs/meta_trades_live.jsonl", help="Path to trade outcomes file")
    parser.add_argument(
        "--window", type=int, default=5, help="Time window (minutes) for matching predictions to trades"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of formatted report")

    args = parser.parse_args()

    # Check files exist
    shadow_path = Path(args.shadow)
    trades_path = Path(args.trades)

    if not shadow_path.exists():
        print(f"Shadow log not found: {shadow_path}")
        print("Run the trading system with filter_shadow_mode=true first")
        return

    if not trades_path.exists():
        print(f"Trades file not found: {trades_path}")
        return

    # Load data
    print(f"Loading shadow predictions from {shadow_path}...")
    predictions = load_shadow_log(str(shadow_path))
    print(f"  Found {len(predictions)} predictions")

    print(f"Loading trade outcomes from {trades_path}...")
    trades = load_trade_outcomes(str(trades_path))
    print(f"  Found {len(trades)} completed trades")

    # Match predictions to outcomes
    print(f"Matching predictions to outcomes (within {args.window} min window)...")
    matched = match_predictions_to_outcomes(predictions, trades, args.window)
    print(f"  Matched {len(matched)} predictions to outcomes")

    # Analyze
    analysis = analyze_results(matched)

    # Output
    if args.json:
        print(json.dumps(analysis, indent=2))
    else:
        print_report(analysis)


if __name__ == "__main__":
    main()
