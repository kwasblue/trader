#!/usr/bin/env python3
"""
Daily Strategy Optimization - Update data and optimize routing

This script should run every morning before trading starts (e.g., 9:00 AM ET)
It will:
1. Download/update historical data for all symbols
2. Run regime-based backtesting
3. Update strategy routing configuration
4. Log results

The amsterdam service should be restarted after this runs to pick up new routing.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime

# Ensure project root is in path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from core.unified_data_pipeline import UnifiedDataPipeline
from loggers.logger import Logger


# Setup logger
logger = Logger("daily_optimize.log", "DailyOptimize", propagate=True).get_logger()


async def update_data(symbols, days=90):
    """Download/update historical data for all symbols."""
    logger.info("=" * 80)
    logger.info(f"UPDATING DATA FOR {len(symbols)} SYMBOLS ({days} days)")
    logger.info("=" * 80)

    pipeline = UnifiedDataPipeline()

    try:
        results = await pipeline.update_symbols(
            symbols=symbols,
            days=days,
            force_reprocess=False,
            process_data=True,
        )

        logger.info(f"Data update complete: {results}")
        return True
    except Exception as e:
        logger.error(f"Error updating data: {e}")
        return False


def run_optimization(symbols, days=90, existing_routing=None):
    """Run strategy optimization."""
    logger.info("=" * 80)
    logger.info(f"RUNNING STRATEGY OPTIMIZATION")
    logger.info("=" * 80)

    from core.backtest.regime_backtest import RegimeBacktester, REGIME_TYPES
    from tools.optimize_routing import (
        DEFAULT_STRATEGIES,
        determine_hybrid_config,
    )
    from collections import Counter

    pipeline = UnifiedDataPipeline()

    # Start with existing routing if provided
    combined_routing = existing_routing.copy() if existing_routing else {}
    all_results = {}
    failed = []

    for i, symbol in enumerate(symbols):
        logger.info(f"[{i+1}/{len(symbols)}] {symbol}...")

        try:
            data = pipeline.get_data(symbol)
            if data is None or data.empty:
                logger.warning(f"{symbol}: NO DATA")
                failed.append(symbol)
                continue

            if len(data) > days:
                data = data.tail(days).reset_index(drop=True)

            tester = RegimeBacktester(
                data=data,
                symbol=symbol,
                strategies=DEFAULT_STRATEGIES,
            )

            result = tester.run_regime_analysis(metric="sharpe_ratio", verbose=False)
            all_results[symbol] = result
            combined_routing.update(result.routing_config)

            best = result.best_strategies
            logger.info(f"{symbol}: low_vol={best.get('low_volatility')}, "
                       f"normal={best.get('normal')}, "
                       f"high_vol={best.get('high_volatility')}")

        except Exception as e:
            logger.error(f"{symbol}: ERROR - {e}")
            failed.append(symbol)

    # Add default routing based on most common strategies
    if all_results:
        low_vol_strats = [r.best_strategies.get("low_volatility") for r in all_results.values()]
        normal_strats = [r.best_strategies.get("normal") for r in all_results.values()]
        high_vol_strats = [r.best_strategies.get("high_volatility") for r in all_results.values()]

        combined_routing["default"] = {
            "low_volatility": Counter(low_vol_strats).most_common(1)[0][0] if low_vol_strats else "sma",
            "normal": Counter(normal_strats).most_common(1)[0][0] if normal_strats else "bollinger",
            "high_volatility": Counter(high_vol_strats).most_common(1)[0][0] if high_vol_strats else "rsi",
            "default": "momentum",
            "use_hybrid": True,
        }
    else:
        logger.error("No successful backtests - keeping existing config")
        return None

    # Determine hybrid sizing for each symbol
    hybrid_config = determine_hybrid_config(combined_routing)

    # Add use_hybrid to routing config
    for symbol in combined_routing:
        if symbol != "default" and symbol in hybrid_config:
            combined_routing[symbol]["use_hybrid"] = hybrid_config[symbol]["enabled"]

    # Summary
    logger.info("=" * 80)
    logger.info("OPTIMIZATION RESULTS:")
    logger.info(f"  Successfully optimized: {len(all_results)} symbols")
    logger.info(f"  Failed (preserved existing): {len(failed)} symbols")
    if failed:
        logger.info(f"  Preserved symbols: {', '.join(failed)}")

    # Strategy frequency
    for regime in REGIME_TYPES:
        strats = [all_results[s].best_strategies.get(regime, "unknown") for s in all_results]
        counts = Counter(strats)
        top3 = counts.most_common(3)
        top_str = ", ".join(f"{s}({c})" for s, c in top3)
        logger.info(f"  {regime}: {top_str}")

    # Hybrid summary
    hybrid_enabled = sum(1 for h in hybrid_config.values() if h["enabled"])
    hybrid_disabled = len(hybrid_config) - hybrid_enabled
    logger.info(f"  Hybrid enabled: {hybrid_enabled} symbols")
    logger.info(f"  Hybrid disabled: {hybrid_disabled} symbols")
    logger.info("=" * 80)

    return combined_routing


async def main():
    """Main entry point."""
    start_time = datetime.now()
    logger.info("")
    logger.info("#" * 80)
    logger.info(f"# DAILY STRATEGY OPTIMIZATION - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("#" * 80)
    logger.info("")

    # Load symbols from existing routing config
    routing_path = ROOT / "config" / "strategy_routing.json"
    if not routing_path.exists():
        logger.error(f"Routing config not found: {routing_path}")
        return 1

    with open(routing_path, "r") as f:
        existing_routing = json.load(f)

    symbols = [s for s in existing_routing.keys() if s != "default"]
    logger.info(f"Loaded {len(symbols)} symbols from existing config")
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info("")

    # Step 1: Update data
    data_success = await update_data(symbols, days=90)
    if not data_success:
        logger.error("Data update failed - aborting optimization")
        return 1

    logger.info("")

    # Step 2: Run optimization (pass existing routing to preserve failed symbols)
    new_routing = run_optimization(symbols, days=90, existing_routing=existing_routing)
    if new_routing is None:
        logger.error("Optimization failed - keeping existing config")
        return 1

    logger.info("")

    # Step 3: Save config
    backup_path = routing_path.parent / f"strategy_routing.backup.{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(backup_path, "w") as f:
        json.dump(existing_routing, f, indent=2)
    logger.info(f"Backup saved to: {backup_path}")

    with open(routing_path, "w") as f:
        json.dump(new_routing, f, indent=2)
    logger.info(f"New routing config saved to: {routing_path}")

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info("")
    logger.info("#" * 80)
    logger.info(f"# OPTIMIZATION COMPLETE - Duration: {duration:.1f}s")
    logger.info("#" * 80)
    logger.info("")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
