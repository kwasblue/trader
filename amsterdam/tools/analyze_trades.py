#!/usr/bin/env python3
"""
Trade Analysis Tool - Analyze win rate and performance from trade logs.

Usage:
    python tools/analyze_trades.py              # Full analysis
    python tools/analyze_trades.py --summary    # Quick summary only
    python tools/analyze_trades.py --by-day     # Daily breakdown
    python tools/analyze_trades.py --by-symbol  # Per-symbol stats
    python tools/analyze_trades.py --by-hour    # Hourly performance
    python tools/analyze_trades.py --worst 10   # Show worst N trades
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "logs" / "live_trades.csv"

# Load environment for broker credentials
load_dotenv(ROOT / ".env")


def load_and_match_trades(csv_path: Path) -> pd.DataFrame:
    """Load trades and match entries to exits to calculate PnL."""
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    positions = defaultdict(list)
    closed_trades = []

    for _, row in df.iterrows():
        symbol = row["symbol"]
        action = row["action"].lower()
        price = row["price"]
        qty = row["quantity"]
        ts = row["timestamp"]
        date = row["date"]
        hour = row["hour"]
        strategy = row.get("strategy", "unknown")
        regime = row.get("regime", "unknown")

        if action == "buy":
            if positions[symbol] and positions[symbol][0]["side"] == "short":
                entry = positions[symbol].pop(0)
                pnl = (entry["price"] - price) * min(qty, entry["qty"])
                closed_trades.append(
                    {
                        "symbol": symbol,
                        "side": "short",
                        "entry_price": entry["price"],
                        "exit_price": price,
                        "qty": min(qty, entry["qty"]),
                        "pnl": pnl,
                        "entry_date": entry["date"],
                        "exit_date": date,
                        "entry_hour": entry["hour"],
                        "exit_hour": hour,
                        "strategy": entry["strategy"],
                        "regime": entry["regime"],
                        "hold_minutes": (ts - entry["timestamp"]).total_seconds() / 60,
                    }
                )
            else:
                positions[symbol].append(
                    {
                        "qty": qty,
                        "price": price,
                        "side": "long",
                        "date": date,
                        "hour": hour,
                        "timestamp": ts,
                        "strategy": strategy,
                        "regime": regime,
                    }
                )

        elif action == "sell":
            if positions[symbol] and positions[symbol][0]["side"] == "long":
                entry = positions[symbol].pop(0)
                pnl = (price - entry["price"]) * min(qty, entry["qty"])
                closed_trades.append(
                    {
                        "symbol": symbol,
                        "side": "long",
                        "entry_price": entry["price"],
                        "exit_price": price,
                        "qty": min(qty, entry["qty"]),
                        "pnl": pnl,
                        "entry_date": entry["date"],
                        "exit_date": date,
                        "entry_hour": entry["hour"],
                        "exit_hour": hour,
                        "strategy": entry["strategy"],
                        "regime": entry["regime"],
                        "hold_minutes": (ts - entry["timestamp"]).total_seconds() / 60,
                    }
                )
            else:
                positions[symbol].append(
                    {
                        "qty": qty,
                        "price": price,
                        "side": "short",
                        "date": date,
                        "hour": hour,
                        "timestamp": ts,
                        "strategy": strategy,
                        "regime": regime,
                    }
                )

    trades_df = pd.DataFrame(closed_trades)
    if not trades_df.empty:
        trades_df["win"] = trades_df["pnl"] > 0

    # Count open positions
    open_count = sum(len(p) for p in positions.values())

    return trades_df, open_count, positions


def print_summary(trades_df: pd.DataFrame, open_count: int):
    """Print overall summary statistics."""
    if trades_df.empty:
        print("No closed trades found.")
        return

    total = len(trades_df)
    wins = trades_df["win"].sum()
    losses = total - wins
    total_pnl = trades_df["pnl"].sum()
    avg_win = trades_df[trades_df["win"]]["pnl"].mean() if wins > 0 else 0
    avg_loss = trades_df[~trades_df["win"]]["pnl"].mean() if losses > 0 else 0

    print("=" * 70)
    print("  TRADING PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"  Total Closed Trades: {total}")
    print(f"  Open Positions: {open_count}")
    print()
    print(f"  Wins: {wins} | Losses: {losses}")
    print(f"  Win Rate: {wins / total * 100:.1f}%")
    print()
    print(f"  Total PnL: ${total_pnl:,.2f}")
    print(f"  Avg Win: ${avg_win:,.2f}")
    print(f"  Avg Loss: ${avg_loss:,.2f}")
    if avg_loss != 0:
        print(f"  Risk/Reward: {abs(avg_win / avg_loss):.2f}")
    print("=" * 70)


def print_by_day(trades_df: pd.DataFrame):
    """Print performance breakdown by day."""
    if trades_df.empty:
        return

    print()
    print("=" * 70)
    print("  PERFORMANCE BY DAY")
    print("=" * 70)
    print(f"{'Date':<12} {'Trades':>7} {'Wins':>6} {'Losses':>7} {'Win%':>7} {'PnL':>12}")
    print("-" * 70)

    by_day = trades_df.groupby("entry_date").agg({"pnl": ["count", "sum"], "win": "sum"}).round(2)
    by_day.columns = ["trades", "pnl", "wins"]
    by_day["losses"] = by_day["trades"] - by_day["wins"]
    by_day["win_rate"] = (by_day["wins"] / by_day["trades"] * 100).round(1)

    for date, row in by_day.iterrows():
        pnl_str = f"${row['pnl']:+,.2f}"
        print(
            f"{str(date):<12} {int(row['trades']):>7} {int(row['wins']):>6} "
            f"{int(row['losses']):>7} {row['win_rate']:>6.1f}% {pnl_str:>12}"
        )

    print("-" * 70)

    # Best and worst days
    if len(by_day) > 1:
        best_day = by_day["pnl"].idxmax()
        worst_day = by_day["pnl"].idxmin()
        print()
        print(f"  Best Day:  {best_day} (${by_day.loc[best_day, 'pnl']:+,.2f})")
        print(f"  Worst Day: {worst_day} (${by_day.loc[worst_day, 'pnl']:+,.2f})")
    print("=" * 70)


def print_by_symbol(trades_df: pd.DataFrame):
    """Print performance breakdown by symbol."""
    if trades_df.empty:
        return

    print()
    print("=" * 70)
    print("  PERFORMANCE BY SYMBOL")
    print("=" * 70)
    print(f"{'Symbol':<8} {'Trades':>7} {'Wins':>6} {'Losses':>7} {'Win%':>7} {'PnL':>12}")
    print("-" * 70)

    by_symbol = trades_df.groupby("symbol").agg({"pnl": ["count", "sum"], "win": "sum"}).round(2)
    by_symbol.columns = ["trades", "pnl", "wins"]
    by_symbol["win_rate"] = (by_symbol["wins"] / by_symbol["trades"] * 100).round(1)
    by_symbol = by_symbol.sort_values("pnl", ascending=False)

    for symbol, row in by_symbol.iterrows():
        losses = int(row["trades"] - row["wins"])
        pnl_str = f"${row['pnl']:+,.2f}"
        print(
            f"{symbol:<8} {int(row['trades']):>7} {int(row['wins']):>6} "
            f"{losses:>7} {row['win_rate']:>6.1f}% {pnl_str:>12}"
        )

    print("-" * 70)
    print("=" * 70)


def print_by_hour(trades_df: pd.DataFrame):
    """Print performance breakdown by entry hour."""
    if trades_df.empty:
        return

    print()
    print("=" * 70)
    print("  PERFORMANCE BY HOUR (Entry Time UTC)")
    print("=" * 70)
    print(f"{'Hour':>6} {'Trades':>7} {'Wins':>6} {'Win%':>7} {'PnL':>12} {'Avg PnL':>10}")
    print("-" * 70)

    by_hour = trades_df.groupby("entry_hour").agg({"pnl": ["count", "sum", "mean"], "win": "sum"}).round(2)
    by_hour.columns = ["trades", "pnl", "avg_pnl", "wins"]
    by_hour["win_rate"] = (by_hour["wins"] / by_hour["trades"] * 100).round(1)

    for hour, row in by_hour.iterrows():
        # Convert UTC to ET (approximate)
        (hour - 5) % 24
        hour_str = f"{hour:02d}:00"
        pnl_str = f"${row['pnl']:+,.2f}"
        avg_str = f"${row['avg_pnl']:+,.2f}"
        print(
            f"{hour_str:>6} {int(row['trades']):>7} {int(row['wins']):>6} "
            f"{row['win_rate']:>6.1f}% {pnl_str:>12} {avg_str:>10}"
        )

    print("-" * 70)

    # Best and worst hours
    if len(by_hour) > 1:
        best_hour = by_hour["pnl"].idxmax()
        worst_hour = by_hour["pnl"].idxmin()
        print()
        print(f"  Best Hour:  {best_hour:02d}:00 UTC (${by_hour.loc[best_hour, 'pnl']:+,.2f})")
        print(f"  Worst Hour: {worst_hour:02d}:00 UTC (${by_hour.loc[worst_hour, 'pnl']:+,.2f})")
    print("=" * 70)


def print_by_strategy(trades_df: pd.DataFrame):
    """Print performance breakdown by strategy."""
    if trades_df.empty or "strategy" not in trades_df.columns:
        return

    print()
    print("=" * 70)
    print("  PERFORMANCE BY STRATEGY")
    print("=" * 70)
    print(f"{'Strategy':<20} {'Trades':>7} {'Wins':>6} {'Win%':>7} {'PnL':>12}")
    print("-" * 70)

    by_strat = trades_df.groupby("strategy").agg({"pnl": ["count", "sum"], "win": "sum"}).round(2)
    by_strat.columns = ["trades", "pnl", "wins"]
    by_strat["win_rate"] = (by_strat["wins"] / by_strat["trades"] * 100).round(1)
    by_strat = by_strat.sort_values("pnl", ascending=False)

    for strat, row in by_strat.iterrows():
        pnl_str = f"${row['pnl']:+,.2f}"
        print(f"{strat:<20} {int(row['trades']):>7} {int(row['wins']):>6} {row['win_rate']:>6.1f}% {pnl_str:>12}")

    print("-" * 70)
    print("=" * 70)


def print_by_hold_time(trades_df: pd.DataFrame):
    """Print performance breakdown by hold time."""
    if trades_df.empty or "hold_minutes" not in trades_df.columns:
        return

    print()
    print("=" * 70)
    print("  PERFORMANCE BY HOLD TIME")
    print("=" * 70)
    print(f"{'Hold Time':<12} {'Trades':>7} {'Wins':>6} {'Win%':>7} {'PnL':>12} {'Avg PnL':>10}")
    print("-" * 70)

    trades_df["hold_bucket"] = pd.cut(
        trades_df["hold_minutes"],
        bins=[0, 5, 15, 30, 60, 120, float("inf")],
        labels=["0-5min", "5-15min", "15-30min", "30-60min", "1-2hr", "2hr+"],
    )

    by_hold = (
        trades_df.groupby("hold_bucket", observed=True).agg({"pnl": ["count", "sum", "mean"], "win": "sum"}).round(2)
    )
    by_hold.columns = ["trades", "pnl", "avg_pnl", "wins"]
    by_hold["win_rate"] = (by_hold["wins"] / by_hold["trades"] * 100).round(1)

    for bucket, row in by_hold.iterrows():
        pnl_str = f"${row['pnl']:+,.2f}"
        avg_str = f"${row['avg_pnl']:+,.2f}"
        print(
            f"{str(bucket):<12} {int(row['trades']):>7} {int(row['wins']):>6} "
            f"{row['win_rate']:>6.1f}% {pnl_str:>12} {avg_str:>10}"
        )

    print("-" * 70)
    print("=" * 70)


def print_worst_trades(trades_df: pd.DataFrame, n: int = 10):
    """Print the worst N trades."""
    if trades_df.empty:
        return

    print()
    print("=" * 70)
    print(f"  WORST {n} TRADES")
    print("=" * 70)
    print(f"{'Symbol':<8} {'Side':<6} {'Entry':>8} {'Exit':>8} {'PnL':>12} {'Hold Time':<12}")
    print("-" * 70)

    worst = trades_df.nsmallest(n, "pnl")

    for _, row in worst.iterrows():
        pnl_str = f"${row['pnl']:+,.2f}"
        if row["hold_minutes"] >= 60:
            hold_str = f"{row['hold_minutes'] / 60:.1f}hr"
        else:
            hold_str = f"{row['hold_minutes']:.0f}min"

        print(
            f"{row['symbol']:<8} {row['side']:<6} ${row['entry_price']:>7.2f} "
            f"${row['exit_price']:>7.2f} {pnl_str:>12} {hold_str:<12}"
        )

    print("-" * 70)
    print("=" * 70)


def print_open_positions(positions: dict):
    """Print currently open positions (from CSV matching - may be inaccurate)."""
    open_list = []
    for symbol, pos_list in positions.items():
        for pos in pos_list:
            open_list.append(
                {
                    "symbol": symbol,
                    "side": pos["side"],
                    "qty": pos["qty"],
                    "price": pos["price"],
                    "date": pos["date"],
                }
            )

    if not open_list:
        return

    print()
    print("=" * 70)
    print("  OPEN POSITIONS (from CSV - may be stale)")
    print("=" * 70)
    print(f"{'Symbol':<8} {'Side':<6} {'Qty':>6} {'Entry Price':>12} {'Entry Date':<12}")
    print("-" * 70)

    for pos in open_list:
        print(f"{pos['symbol']:<8} {pos['side']:<6} {pos['qty']:>6} ${pos['price']:>10.2f} {str(pos['date']):<12}")

    print("-" * 70)
    print("=" * 70)


def fetch_live_positions():
    """Fetch actual positions from Alpaca broker."""
    try:
        from alpaca.trading.client import TradingClient

        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            print("Error: ALPACA_API_KEY or ALPACA_SECRET_KEY not set")
            return None

        paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
        client = TradingClient(api_key, secret_key, paper=paper)
        return client.get_all_positions()
    except Exception as e:
        print(f"Error fetching positions: {e}")
        return None


def print_live_positions():
    """Print actual positions from broker."""
    positions = fetch_live_positions()

    if positions is None:
        return 0

    print()
    print("=" * 70)
    print("  OPEN POSITIONS (LIVE from Alpaca)")
    print("=" * 70)

    if not positions:
        print("  No open positions")
        print("=" * 70)
        return 0

    print(f"{'Symbol':<8} {'Side':<6} {'Qty':>8} {'Avg Entry':>12} {'Market Val':>12} {'Unrealized':>12}")
    print("-" * 70)

    total_unrealized = 0.0
    for pos in positions:
        qty = int(float(pos.qty))
        avg_entry = float(pos.avg_entry_price)
        market_val = float(pos.market_value)
        unrealized = float(pos.unrealized_pl)
        total_unrealized += unrealized

        print(
            f"{pos.symbol:<8} {pos.side.value:<6} {qty:>8} "
            f"${avg_entry:>10.2f} ${market_val:>10.2f} ${unrealized:>+10.2f}"
        )

    print("-" * 70)
    print(f"{'Total':>36} ${sum(float(p.market_value) for p in positions):>10.2f} ${total_unrealized:>+10.2f}")
    print("=" * 70)

    return len(positions)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trading performance from CSV logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-f", "--file", default=str(DEFAULT_LOG), help=f"Path to trades CSV (default: {DEFAULT_LOG})")
    parser.add_argument("--summary", action="store_true", help="Show summary only")
    parser.add_argument("--by-day", action="store_true", help="Show daily breakdown")
    parser.add_argument("--by-symbol", action="store_true", help="Show per-symbol stats")
    parser.add_argument("--by-hour", action="store_true", help="Show hourly performance")
    parser.add_argument("--by-strategy", action="store_true", help="Show per-strategy stats")
    parser.add_argument("--by-hold-time", action="store_true", help="Show hold time analysis")
    parser.add_argument("--worst", type=int, metavar="N", help="Show worst N trades")
    parser.add_argument("--open", action="store_true", help="Show open positions")
    parser.add_argument("--live", action="store_true", help="Query live positions from broker (instead of CSV)")

    args = parser.parse_args()

    csv_path = Path(args.file)
    if not csv_path.exists():
        print(f"Error: Trade log not found: {csv_path}")
        sys.exit(1)

    trades_df, open_count, positions = load_and_match_trades(csv_path)

    # If no specific flags, show everything
    show_all = not any(
        [
            args.summary,
            args.by_day,
            args.by_symbol,
            args.by_hour,
            args.by_strategy,
            args.by_hold_time,
            args.worst,
            args.open,
            args.live,
        ]
    )

    # Use live position count for summary if --live flag is set
    if args.live or show_all:
        live_positions = fetch_live_positions()
        live_open_count = len(live_positions) if live_positions else 0
    else:
        live_open_count = None

    if show_all or args.summary:
        # Show live count in summary if available
        display_open_count = live_open_count if live_open_count is not None else open_count
        print_summary(trades_df, display_open_count)

    if show_all or args.by_day:
        print_by_day(trades_df)

    if show_all or args.by_symbol:
        print_by_symbol(trades_df)

    if show_all or args.by_hour:
        print_by_hour(trades_df)

    if show_all or args.by_strategy:
        print_by_strategy(trades_df)

    if show_all or args.by_hold_time:
        print_by_hold_time(trades_df)

    if args.worst:
        print_worst_trades(trades_df, args.worst)

    # Show live positions by default, fall back to CSV if --open explicitly requested
    if show_all or args.live:
        print_live_positions()
    elif args.open:
        print_open_positions(positions)


if __name__ == "__main__":
    main()
