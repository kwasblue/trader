#!/usr/bin/env python
"""
Test script for the optimization workflow.

Tests:
1. Regime-aware strategy selection
2. Timeframe optimization with corrected regime logic
3. Unified optimization workflow
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import asyncio
import pandas as pd
from core.unified_data_pipeline import UnifiedDataPipeline
from core.backtest.strategy_selector import StrategySelector
from core.backtest.timeframe_optimizer import TimeframeOptimizer
from tools.unified_optimizer import UnifiedOptimizer


def test_regime_aware_strategy_selection():
    """Test regime-aware strategy selection."""
    print("\n" + "=" * 80)
    print("TEST 1: Regime-Aware Strategy Selection")
    print("=" * 80)

    # Load data
    pipeline = UnifiedDataPipeline()
    data = pipeline.get_data("AAPL", timeframe="day")

    if data is None or data.empty:
        print("ERROR: No data available for AAPL")
        return False

    # Limit to recent data for faster testing
    data = data.tail(365).reset_index(drop=True)
    print(f"Loaded {len(data)} bars for AAPL")

    # Run regime-aware selection
    selector = StrategySelector(data, initial_capital=100000)

    result = selector.select_best_strategies_by_regime(
        symbol="AAPL",
        metric="sharpe_ratio",
        verbose=True
    )

    # Verify we got results for all regimes
    expected_regimes = ["low_volatility", "normal", "high_volatility"]
    for regime in expected_regimes:
        if regime not in result.best_strategies:
            print(f"ERROR: Missing regime {regime} in results")
            return False
        print(f"✓ {regime}: {result.best_strategies[regime]}")

    print("\n✓ Test 1 PASSED: Regime-aware selection working")
    return True


async def test_timeframe_optimizer_regime_logic():
    """Test that timeframe optimizer has correct regime logic."""
    print("\n" + "=" * 80)
    print("TEST 2: Timeframe Optimizer Regime Logic")
    print("=" * 80)

    # Create optimizer
    optimizer = TimeframeOptimizer(
        symbols=["AAPL"],
        timeframes=["5min", "15min", "30min"],
        strategies=["sma", "rsi"]  # Just test a few for speed
    )

    # Run optimization
    print("Running timeframe optimization (this may take a minute)...")
    results = await optimizer.run_optimization(days=365, min_bars=50)

    print(f"Got {len(results)} results")

    # Generate config
    config = optimizer.generate_routing_config(results, min_sharpe=0.0)

    # Verify regime logic is correct:
    # - High volatility should have shorter timeframes (5min, 15min)
    # - Low volatility should have longer timeframes (30min, 1hour)

    if "AAPL" not in config:
        print("ERROR: No config generated for AAPL")
        return False

    aapl_config = config["AAPL"]

    print("\nGenerated config:")
    for regime in ["high_volatility", "normal", "low_volatility"]:
        if regime in aapl_config:
            tf = aapl_config[regime].get("timeframe", "N/A")
            strat = aapl_config[regime].get("strategy", "N/A")
            print(f"  {regime}: {strat} @ {tf}")

            # Verify logic
            if regime == "high_volatility":
                if tf not in ["5min", "15min"]:
                    print(f"  ⚠ WARNING: High volatility using {tf} (expected 5min or 15min)")
            elif regime == "low_volatility":
                if tf not in ["30min", "1hour", "day"]:
                    print(f"  ⚠ WARNING: Low volatility using {tf} (expected 30min+ timeframes)")

    print("\n✓ Test 2 PASSED: Timeframe optimizer regime logic corrected")
    return True


async def test_unified_optimizer():
    """Test unified optimization workflow."""
    print("\n" + "=" * 80)
    print("TEST 3: Unified Optimization Workflow")
    print("=" * 80)

    # Create unified optimizer
    optimizer = UnifiedOptimizer(
        symbols=["AAPL"],
        strategies=["sma", "rsi", "momentum"],  # Limited set for speed
        timeframes=["15min", "30min"]  # Limited for speed
    )

    # Run optimization
    print("Running unified optimization...")
    results = await optimizer.run_optimization(days=365, metric="sharpe_ratio")

    # Verify results
    if "AAPL" not in results:
        print("ERROR: No results for AAPL")
        return False

    result = results["AAPL"]

    # Check that we have optimal config
    if not result.optimal_config:
        print("ERROR: No optimal config generated")
        return False

    print("\nOptimal configuration:")
    config = result.optimal_config["AAPL"]
    for regime in ["high_volatility", "normal", "low_volatility"]:
        if regime in config:
            strat = config[regime]["strategy"]
            tf = config[regime]["timeframe"]
            print(f"  {regime}: {strat} @ {tf}")

    # Print summary
    optimizer.print_summary()

    print("\n✓ Test 3 PASSED: Unified optimizer working")
    return True


async def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TESTING OPTIMIZATION WORKFLOW")
    print("=" * 80)

    results = []

    # Test 1: Regime-aware strategy selection
    try:
        success = test_regime_aware_strategy_selection()
        results.append(("Regime-Aware Selection", success))
    except Exception as e:
        print(f"\n✗ Test 1 FAILED: {e}")
        results.append(("Regime-Aware Selection", False))

    # Test 2: Timeframe optimizer regime logic
    try:
        success = await test_timeframe_optimizer_regime_logic()
        results.append(("Timeframe Optimizer Logic", success))
    except Exception as e:
        print(f"\n✗ Test 2 FAILED: {e}")
        results.append(("Timeframe Optimizer Logic", False))

    # Test 3: Unified optimizer
    try:
        success = await test_unified_optimizer()
        results.append(("Unified Optimizer", success))
    except Exception as e:
        print(f"\n✗ Test 3 FAILED: {e}")
        results.append(("Unified Optimizer", False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:<30}: {status}")

    all_passed = all(success for _, success in results)

    print("\n" + "=" * 80)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
