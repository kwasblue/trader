#!/usr/bin/env python
"""
Analyze Sharpe Filter Impact

Shows which symbol/regime combinations would be blocked at different
Sharpe thresholds to help you choose the right min_sharpe setting.

Usage:
    python tools/analyze_sharpe_filter.py
    python tools/analyze_sharpe_filter.py --min-sharpe 0.3
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.logic.sharpe_filter import SharpeFilter


def analyze_thresholds():
    """Analyze impact of different Sharpe thresholds."""
    thresholds = [0.0, 0.3, 0.5, 0.7, 1.0]

    print("\n" + "=" * 80)
    print("SHARPE FILTER ANALYSIS")
    print("=" * 80)

    for threshold in thresholds:
        print(f"\n📊 Min Sharpe = {threshold}")
        print("-" * 80)

        filter = SharpeFilter(min_sharpe=threshold)
        blocked = filter.get_blocked_regimes()

        if not blocked:
            print("  ✅ No regimes blocked")
            continue

        for symbol, regimes in sorted(blocked.items()):
            sharpe_info = []
            for regime in regimes:
                sharpe = filter.get_sharpe(symbol, regime)
                sharpe_info.append(f"{regime}: {sharpe:.2f}")

            print(f"  {symbol:6} - {', '.join(sharpe_info)}")

        total_blocked = sum(len(r) for r in blocked.values())
        total_combos = sum(len(r) for r in filter.sharpe_map.values())

        print(f"\n  Total: {total_blocked}/{total_combos} blocked ({100 * total_blocked / total_combos:.1f}%)")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
min_sharpe = 0.0  : Only block losing strategies (negative Sharpe)
                    Conservative - trades most regimes

min_sharpe = 0.3  : Block poor performers
                    Balanced - reasonable quality bar

min_sharpe = 0.5  : Block marginal strategies (recommended)
                    Quality focused - only trade decent strategies

min_sharpe = 0.7  : Only trade good strategies
                    Aggressive - high quality bar

min_sharpe = 1.0  : Only trade excellent strategies
                    Very aggressive - may miss opportunities
    """)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Sharpe filter impact")

    parser.add_argument("--min-sharpe", type=float, help="Show specific threshold impact")

    args = parser.parse_args()

    if args.min_sharpe is not None:
        # Show specific threshold
        filter = SharpeFilter(min_sharpe=args.min_sharpe)
        filter.print_summary()
    else:
        # Show all thresholds
        analyze_thresholds()


if __name__ == "__main__":
    main()
