#!/usr/bin/env python3
"""
Compare Strategies CLI Tool

Compare trading strategies, categories, and hybrid sizing approaches.

Usage:
    # Compare specific strategies
    python tools/compare_strategies.py AAPL -s sma,ema,rsi,macd

    # Compare categories
    python tools/compare_strategies.py AAPL --categories trend_following,mean_reversion

    # Compare hybrid vs standard sizing
    python tools/compare_strategies.py AAPL --hybrid-comparison -s sma,macd,rsi

    # Full comparison with report
    python tools/compare_strategies.py AAPL --full -o reports/comparison.md

    # List available strategies
    python tools/compare_strategies.py --list
"""

import sys
import argparse
import asyncio
from pathlib import Path

# Ensure project root is in path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def load_data(symbol: str, days: int) -> pd.DataFrame:
    """Load historical data for a symbol."""
    from core.unified_data_pipeline import UnifiedDataPipeline

    pipeline = UnifiedDataPipeline()

    # Try to get existing data
    data = pipeline.get_data(symbol)

    if data is None or data.empty:
        print(f"No cached data for {symbol}, fetching...")

        async def fetch():
            return await pipeline.update_symbols([symbol], days=days)

        asyncio.run(fetch())
        data = pipeline.get_data(symbol)

    if data is None or data.empty:
        raise ValueError(f"Could not load data for {symbol}")

    # Limit to requested days
    if len(data) > days:
        data = data.tail(days).reset_index(drop=True)

    return data


def list_strategies_cmd():
    """List all available strategies."""
    from core.backtest.unified_backtest_runner import (
        list_available_strategies,
        list_categories,
        get_strategy_category,
    )

    strategies = list_available_strategies()
    categories = list_categories()

    print("\n" + "=" * 60)
    print("  AVAILABLE STRATEGIES")
    print("=" * 60)

    # Group by category
    for category, cat_strategies in categories.items():
        available = [s for s in cat_strategies if s in strategies]
        if available:
            print(f"\n  {category.replace('_', ' ').title()}:")
            for s in available:
                print(f"    • {s}")

    # Show uncategorized
    categorized = set()
    for cat_strats in categories.values():
        categorized.update(cat_strats)

    uncategorized = [s for s in strategies if s not in categorized]
    if uncategorized:
        print("\n  Other:")
        for s in uncategorized:
            print(f"    • {s}")

    print(f"\n  Total: {len(strategies)} strategies")
    print("=" * 60)
    print()


def compare_strategies_cmd(args):
    """Run strategy comparison."""
    from core.backtest.backtest_comparison import (
        BacktestComparison,
        print_strategy_table,
    )

    print(f"\nLoading {args.days} days of data for {args.symbol}...")
    data = load_data(args.symbol, args.days)
    print(f"Loaded {len(data)} bars")

    comparison = BacktestComparison(
        data=data,
        initial_capital=args.capital,
        transaction_cost=args.cost,
        symbol=args.symbol,
    )

    strategies = [s.strip() for s in args.strategies.split(",")]

    print(f"\nComparing strategies: {', '.join(strategies)}")
    print(f"Metric: {args.metric}")

    result = comparison.compare_strategies(
        strategies=strategies,
        metric=args.metric,
        use_hybrid=args.use_hybrid,
    )

    print_strategy_table(result.results, args.symbol)

    return result


def compare_categories_cmd(args):
    """Run category comparison."""
    from core.backtest.backtest_comparison import BacktestComparison

    print(f"\nLoading {args.days} days of data for {args.symbol}...")
    data = load_data(args.symbol, args.days)
    print(f"Loaded {len(data)} bars")

    comparison = BacktestComparison(
        data=data,
        initial_capital=args.capital,
        transaction_cost=args.cost,
        symbol=args.symbol,
    )

    categories = [c.strip() for c in args.categories.split(",")]

    print(f"\nComparing categories: {', '.join(categories)}")

    result = comparison.compare_categories(
        categories=categories,
        metric=args.metric,
    )

    # Print results
    print()
    print("=" * 70)
    print(f"  CATEGORY COMPARISON - {args.symbol}")
    print("=" * 70)

    print()
    print(f"{'Category':<20} {'Avg Return':>12} {'Avg Sharpe':>12} {'Best Strategy':>18}")
    print("-" * 70)

    sorted_cats = sorted(result.categories, key=lambda c: result.rankings.get(c, 999))
    for cat in sorted_cats:
        cat_data = result.category_results.get(cat, {})
        avg_return = cat_data.get("avg_return", 0)
        avg_sharpe = cat_data.get("avg_sharpe", 0)
        best_strat = cat_data.get("best_strategy", "-")

        print(
            f"{cat.replace('_', ' ').title():<20} {avg_return:>+11.1%} "
            f"{avg_sharpe:>12.2f} {best_strat:>18}"
        )

    print("-" * 70)
    print(f"\nBest Category: {result.best_category.replace('_', ' ').title()}")
    print("=" * 70)
    print()

    return result


def compare_hybrid_cmd(args):
    """Run hybrid comparison."""
    from core.backtest.backtest_comparison import (
        BacktestComparison,
        print_hybrid_table,
    )

    print(f"\nLoading {args.days} days of data for {args.symbol}...")
    data = load_data(args.symbol, args.days)
    print(f"Loaded {len(data)} bars")

    comparison = BacktestComparison(
        data=data,
        initial_capital=args.capital,
        transaction_cost=args.cost,
        symbol=args.symbol,
    )

    strategies = [s.strip() for s in args.strategies.split(",")]

    print(f"\nComparing hybrid sizing for: {', '.join(strategies)}")

    result = comparison.compare_hybrid(strategies=strategies)

    print_hybrid_table(result, args.symbol)

    # Print insight
    from core.backtest.unified_backtest_runner import get_strategy_category

    tf_improved = sum(
        1 for s in strategies
        if get_strategy_category(s) == "trend_following" and result.improvements.get(s, 0) > 0
    )
    mr_improved = sum(
        1 for s in strategies
        if get_strategy_category(s) == "mean_reversion" and result.improvements.get(s, 0) > 0
    )

    print("INSIGHT:")
    if tf_improved > mr_improved:
        print("  Trend-following strategies benefit from hybrid sizing.")
    elif mr_improved > tf_improved:
        print("  Mean-reversion strategies may not benefit from hybrid sizing")
        print("  (they trade against trend by design).")
    else:
        print("  Mixed results - hybrid sizing impact varies by strategy.")

    print()
    return result


def full_comparison_cmd(args):
    """Run full comparison."""
    from core.backtest.backtest_comparison import BacktestComparison

    print(f"\nLoading {args.days} days of data for {args.symbol}...")
    data = load_data(args.symbol, args.days)
    print(f"Loaded {len(data)} bars")

    comparison = BacktestComparison(
        data=data,
        initial_capital=args.capital,
        transaction_cost=args.cost,
        symbol=args.symbol,
    )

    # Determine strategies
    strategies = None
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",")]

    # Determine categories
    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]

    print("\nRunning full comparison analysis...")
    print("This may take a while...\n")

    report = comparison.full_comparison(
        strategies=strategies,
        categories=categories,
        include_hybrid=True,
        metric=args.metric,
    )

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if args.output.endswith(".json"):
            content = report.to_json()
        else:
            content = report.to_markdown()

        output_path.write_text(content)
        print(f"Report saved to: {output_path}")
    else:
        # Print to console
        print(report.to_markdown())

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Compare trading strategies, categories, and sizing approaches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare specific strategies
    python tools/compare_strategies.py AAPL -s sma,ema,rsi,macd

    # Compare categories
    python tools/compare_strategies.py AAPL --categories trend_following,mean_reversion

    # Compare hybrid sizing
    python tools/compare_strategies.py AAPL --hybrid-comparison -s sma,macd,rsi

    # Full comparison with report
    python tools/compare_strategies.py AAPL --full -o reports/comparison.md

    # List available strategies
    python tools/compare_strategies.py --list
        """,
    )

    parser.add_argument(
        "symbol",
        nargs="?",
        help="Stock symbol to analyze (e.g., AAPL)",
    )
    parser.add_argument(
        "-s", "--strategies",
        default="sma,ema,macd,rsi",
        help="Comma-separated strategies to compare (default: sma,ema,macd,rsi)",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help="Comma-separated categories to compare (e.g., trend_following,mean_reversion)",
    )
    parser.add_argument(
        "--hybrid-comparison",
        action="store_true",
        help="Run hybrid vs standard sizing comparison",
    )
    parser.add_argument(
        "--use-hybrid",
        action="store_true",
        help="Use hybrid sizing for strategy comparison",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full comparison (strategies, categories, hybrid)",
    )
    parser.add_argument(
        "-d", "--days",
        type=int,
        default=365,
        help="Days of historical data (default: 365)",
    )
    parser.add_argument(
        "-m", "--metric",
        default="sharpe_ratio",
        choices=["sharpe_ratio", "total_return", "sortino_ratio", "win_rate"],
        help="Metric for ranking (default: sharpe_ratio)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital (default: 10000)",
    )
    parser.add_argument(
        "--cost",
        type=float,
        default=0.001,
        help="Transaction cost fraction (default: 0.001)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file for report (.md or .json)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available strategies",
    )

    args = parser.parse_args()

    # Handle list command (no symbol needed)
    if args.list:
        list_strategies_cmd()
        return

    # Require symbol for other commands
    if not args.symbol:
        parser.error("Symbol is required (e.g., python tools/compare_strategies.py AAPL)")

    args.symbol = args.symbol.upper()

    try:
        if args.full:
            full_comparison_cmd(args)
        elif args.categories:
            compare_categories_cmd(args)
        elif args.hybrid_comparison:
            compare_hybrid_cmd(args)
        else:
            compare_strategies_cmd(args)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
