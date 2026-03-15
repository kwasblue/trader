#!/usr/bin/env python
"""
Optimize All Trading Symbols

Runs the unified optimizer on all symbols in your trade list.
This is a convenience script for the full portfolio optimization workflow.

Usage:
    python tools/optimize_all_symbols.py --save
    python tools/optimize_all_symbols.py --days 1000 --save
    python tools/optimize_all_symbols.py --dry-run  # Preview only
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.symbol_list_manager import get_list_manager
from tools.unified_optimizer import UnifiedOptimizer


async def main():
    """Run unified optimizer on all trading symbols."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Optimize all symbols in your trade list',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize all trading symbols and save
  python tools/optimize_all_symbols.py --save

  # Use more historical data
  python tools/optimize_all_symbols.py --days 1000 --save

  # Preview without saving
  python tools/optimize_all_symbols.py --dry-run

  # Custom timeframes
  python tools/optimize_all_symbols.py --timeframes 5min,15min,30min,1hour --save

  # Specific strategies only
  python tools/optimize_all_symbols.py --strategies rsi,sma,momentum --save
        """
    )

    parser.add_argument(
        '--strategies',
        help='Comma-separated strategies to test (default: all)'
    )
    parser.add_argument(
        '--timeframes',
        default='5min,15min,30min,1hour,day',
        help='Comma-separated timeframes to test'
    )
    parser.add_argument(
        '--days', type=int, default=750,
        help='Days of historical data (default: 750)'
    )
    parser.add_argument(
        '--metric', default='sharpe_ratio',
        choices=['sharpe_ratio', 'total_return', 'win_rate'],
        help='Ranking metric (default: sharpe_ratio)'
    )
    parser.add_argument(
        '--save', action='store_true',
        help='Save to config/strategy_routing.json'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Preview symbols without running optimization'
    )
    parser.add_argument(
        '--output',
        help='Custom output path for config file'
    )

    args = parser.parse_args()

    # Get trade list
    print("\n" + "=" * 80)
    print("OPTIMIZE ALL TRADING SYMBOLS")
    print("=" * 80)

    manager = get_list_manager()
    symbols = manager.get_trade_list()

    if not symbols:
        print("\n❌ ERROR: No symbols in trade list!")
        print("\nAdd symbols to your trade list:")
        print("  amsterdam_ctl.py add AAPL --trade")
        print("  amsterdam_ctl.py add TSLA --trade")
        print("  amsterdam_ctl.py add MSFT --trade")
        return 1

    print(f"\nFound {len(symbols)} symbols in trade list:")
    for i, symbol in enumerate(symbols, 1):
        print(f"  {i:2d}. {symbol}")

    if args.dry_run:
        print("\n🔍 DRY RUN - No optimization will be performed")
        print(f"\nWould optimize {len(symbols)} symbols with:")
        print(f"  Timeframes: {args.timeframes}")
        print(f"  Days: {args.days}")
        print(f"  Metric: {args.metric}")
        if args.strategies:
            print(f"  Strategies: {args.strategies}")
        else:
            print(f"  Strategies: all available")
        return 0

    # Confirm if not saving
    if not args.save:
        print("\n⚠️  WARNING: Results will not be saved (use --save to save)")
        response = input("\nContinue? [y/N]: ")
        if response.lower() not in ('y', 'yes'):
            print("Cancelled.")
            return 0

    # Parse strategies
    strategies = None
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(',')]

    # Parse timeframes
    timeframes = [tf.strip() for tf in args.timeframes.split(',')]

    print("\n" + "=" * 80)
    print(f"STARTING OPTIMIZATION")
    print("=" * 80)
    print(f"Symbols: {len(symbols)}")
    print(f"Timeframes: {timeframes}")
    print(f"Days of data: {args.days}")
    print(f"Metric: {args.metric}")
    if strategies:
        print(f"Strategies: {strategies}")
    else:
        print(f"Strategies: all available")
    print("=" * 80)

    # Create optimizer
    optimizer = UnifiedOptimizer(
        symbols=symbols,
        strategies=strategies,
        timeframes=timeframes
    )

    # Run optimization
    try:
        await optimizer.run_optimization(
            days=args.days,
            metric=args.metric
        )

        # Print summary
        optimizer.print_summary()

        # Save if requested
        if args.save:
            config_path = optimizer.save_config(config_path=args.output)
            print(f"\n{'='*80}")
            print(f"✅ SAVED TO: {config_path}")
            print(f"{'='*80}")
            print("\nOptimization complete! Next steps:")
            print("  1. Review the config file:")
            print(f"     cat {config_path}")
            print("  2. Restart the trader to use new configuration:")
            print("     amsterdam start")
            print("  3. Monitor performance for 1-2 weeks")
            print("  4. Re-optimize periodically (every 3-6 months)")
        else:
            print("\n⚠️  Results not saved (use --save to save to config)")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Optimization cancelled by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
