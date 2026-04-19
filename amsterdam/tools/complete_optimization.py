#!/usr/bin/env python
"""
Complete Optimization Workflow

Runs the full optimization suite:
1. Strategy/timeframe selection (unified_optimizer.py)
2. Strategy parameter optimization (optimize_strategy_params.py)
3. Hybrid sizing test (test_hybrid_sizing.py)

This is a convenience script that runs all optimization steps in sequence.

Usage:
    python tools/complete_optimization.py --save
    python tools/complete_optimization.py --symbols AAPL TSLA --save
    python tools/complete_optimization.py --skip-params  # Skip param optimization
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.symbol_list_manager import get_list_manager
from tools.optimize_strategy_params import StrategyParamOptimizer
from tools.test_hybrid_sizing import HybridSizingTester
from tools.unified_optimizer import UnifiedOptimizer


async def main():
    """Run complete optimization workflow."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Complete optimization workflow - all steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full optimization of all trade list symbols
  python tools/complete_optimization.py --save

  # Optimize specific symbols
  python tools/complete_optimization.py --symbols AAPL TSLA NVDA --save

  # Skip parameter optimization (faster)
  python tools/complete_optimization.py --skip-params --save

  # Skip hybrid sizing test
  python tools/complete_optimization.py --skip-hybrid --save

  # Quick optimization (skip both)
  python tools/complete_optimization.py --skip-params --skip-hybrid --save
        """,
    )

    parser.add_argument("--symbols", nargs="+", help="Symbols to optimize (default: all from trade list)")
    parser.add_argument("--days", type=int, default=750, help="Days of historical data (default: 750)")
    parser.add_argument("--skip-params", action="store_true", help="Skip parameter optimization (faster)")
    parser.add_argument("--skip-hybrid", action="store_true", help="Skip hybrid sizing test")
    parser.add_argument("--save", action="store_true", help="Save results to config files")
    parser.add_argument("--dry-run", action="store_true", help="Preview without running")

    args = parser.parse_args()

    # Get symbols
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        manager = get_list_manager()
        symbols = manager.get_trade_list()

    if not symbols:
        print("❌ ERROR: No symbols to optimize")
        return 1

    print("\n" + "=" * 80)
    print("COMPLETE OPTIMIZATION WORKFLOW")
    print("=" * 80)
    print(f"\nSymbols: {len(symbols)}")
    print("Steps:")
    print("  1. ✓ Strategy/Timeframe Selection")
    if args.skip_params:
        print("  2. ⊘ Parameter Optimization (SKIPPED)")
    else:
        print("  2. ✓ Parameter Optimization")
    if args.skip_hybrid:
        print("  3. ⊘ Hybrid Sizing Test (SKIPPED)")
    else:
        print("  3. ✓ Hybrid Sizing Test")
    print(f"\nDays of data: {args.days}")
    print(f"Save results: {args.save}")
    print("=" * 80)

    if args.dry_run:
        print("\n🔍 DRY RUN - No optimization will be performed")
        return 0

    if not args.save:
        print("\n⚠️  WARNING: Results will not be saved (use --save)")
        response = input("Continue? [y/N]: ")
        if response.lower() not in ("y", "yes"):
            print("Cancelled.")
            return 0

    start_time = datetime.now()

    # ========================================================================
    # STEP 1: Strategy & Timeframe Optimization
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1/3: STRATEGY & TIMEFRAME OPTIMIZATION")
    print("=" * 80)

    try:
        optimizer = UnifiedOptimizer(symbols=symbols, timeframes=["5min", "15min", "30min", "1hour", "day"])

        await optimizer.run_optimization(days=args.days, metric="sharpe_ratio")

        if args.save:
            optimizer.save_config()
            print("\n✅ Step 1 complete: Strategy/timeframe config saved")
        else:
            print("\n✓ Step 1 complete (not saved)")

    except Exception as e:
        print(f"\n❌ Step 1 failed: {e}")
        return 1

    # ========================================================================
    # STEP 2: Parameter Optimization
    # ========================================================================
    if not args.skip_params:
        print("\n" + "=" * 80)
        print("STEP 2/3: STRATEGY PARAMETER OPTIMIZATION")
        print("=" * 80)
        print("\nThis step tests many parameter combinations...")
        print("Estimated time: 30-60 minutes for 19 symbols\n")

        try:
            param_optimizer = StrategyParamOptimizer(symbols=symbols)

            await param_optimizer.optimize_all(days=365, metric="sharpe_ratio")

            if args.save:
                param_optimizer.save_params()
                print("\n✅ Step 2 complete: Parameter config saved")
            else:
                print("\n✓ Step 2 complete (not saved)")

        except Exception as e:
            print(f"\n❌ Step 2 failed: {e}")
            print("Continuing to next step...")
    else:
        print("\n⊘ Step 2 skipped: Parameter optimization")

    # ========================================================================
    # STEP 3: Hybrid Sizing Test
    # ========================================================================
    if not args.skip_hybrid:
        print("\n" + "=" * 80)
        print("STEP 3/3: HYBRID SIZING TEST")
        print("=" * 80)

        try:
            hybrid_tester = HybridSizingTester(symbols=symbols)

            await hybrid_tester.test_all(days=365)

            hybrid_tester.print_summary()

            if args.save:
                hybrid_tester.save_recommendations()
                print("\n✅ Step 3 complete: Hybrid flags updated")
            else:
                print("\n✓ Step 3 complete (not saved)")

        except Exception as e:
            print(f"\n❌ Step 3 failed: {e}")
            print("Optimization incomplete")
    else:
        print("\n⊘ Step 3 skipped: Hybrid sizing test")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("✅ COMPLETE OPTIMIZATION FINISHED")
    print("=" * 80)
    print(f"\nTotal time: {duration}")
    print(f"Symbols optimized: {len(symbols)}")

    if args.save:
        print("\n📁 Updated config files:")
        print("  ✓ config/strategy_routing.json")
        if not args.skip_params:
            print("  ✓ config/strategy_params.json")
        print("\n🚀 Ready to trade! Restart the system:")
        print("  amsterdam start")
        print("\n📊 Monitor performance for 1-2 weeks, then re-optimize if needed")
    else:
        print("\n⚠️  Results not saved - run with --save to update configs")

    print("\n" + "=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
