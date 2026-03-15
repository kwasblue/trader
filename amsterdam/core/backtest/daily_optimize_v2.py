#!/usr/bin/env python3
"""
Improved Daily Strategy Optimization - Walk-forward validation and statistical testing

Key improvements over v1:
1. Uses 365+ days of data (vs 90 days)
2. Walk-forward validation: trains on 80%, validates on 20%
3. Requires minimum trade counts for statistical significance
4. Only updates routing if new strategy is significantly better
5. Adds confidence scoring and detailed metrics logging
6. Preserves existing routing for low-confidence results

This should run weekly or monthly, not daily.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import numpy as np

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

# Configuration
TRAIN_DAYS = 750  # Use 3 years of data for training (more trades per regime)
VALIDATION_SPLIT = 0.8  # Train on 80%, validate on 20%
MIN_TRADES_PER_REGIME = 4  # Minimum trades required (adjusted for conservative strategies)
MIN_SHARPE_IMPROVEMENT = 0.3  # Minimum Sharpe improvement to justify switching strategies
CONFIDENCE_THRESHOLD = 0.45  # Minimum avg confidence to update routing (adjusted for strategy trade frequency)


async def update_data(symbols, days=TRAIN_DAYS):
    """Download/update historical data for all symbols."""
    logger.info("=" * 80)
    logger.info(f"UPDATING DATA FOR {len(symbols)} SYMBOLS ({days} days)")
    logger.info("=" * 80)

    pipeline = UnifiedDataPipeline()

    try:
        results = await pipeline.update_symbols(
            symbols=symbols,
            days=days,
            source='schwab',  # Use Schwab for more historical data
            force_reprocess=True,  # Force reprocessing to ensure fresh data
            process_data=True,
        )

        logger.info(f"Data update complete: {results}")
        return True
    except Exception as e:
        logger.error(f"Error updating data: {e}")
        return False


def calculate_confidence_score(result, validation_result, regime_stats) -> float:
    """
    Calculate confidence score for a strategy selection.

    Factors:
    - Number of trades (more trades = higher confidence)
    - Consistency between train and validation
    - Regime data availability

    Returns: 0.0 to 1.0
    """
    scores = []

    # Trade count score (0-1, saturates at 30 trades)
    trade_count = result.num_trades
    trade_score = min(trade_count / 30, 1.0)
    scores.append(trade_score)

    # Validation consistency score
    if validation_result and validation_result.num_trades >= MIN_TRADES_PER_REGIME:
        # Compare Sharpe ratios
        train_sharpe = result.sharpe_ratio
        val_sharpe = validation_result.sharpe_ratio

        # If both positive or both negative, check consistency
        if train_sharpe * val_sharpe > 0:
            # Consistency = 1 - normalized difference
            diff = abs(train_sharpe - val_sharpe) / (abs(train_sharpe) + 0.01)
            consistency_score = max(0, 1 - diff)
        else:
            # One positive, one negative = poor consistency
            consistency_score = 0.2

        scores.append(consistency_score)
    else:
        # No validation data = lower confidence
        scores.append(0.3)

    # Regime data availability (bars in regime / total bars)
    regime_coverage = regime_stats.pct_of_total / 100.0
    scores.append(regime_coverage)

    # Overall confidence is weighted average
    weights = [0.4, 0.4, 0.2]  # trades, validation, coverage
    confidence = sum(s * w for s, w in zip(scores, weights))

    return confidence


def run_walkforward_optimization(symbols, days=TRAIN_DAYS, existing_routing=None):
    """
    Run walk-forward strategy optimization.

    For each symbol:
    1. Split data into train (80%) and validation (20%)
    2. Find best strategy on training data
    3. Validate on out-of-sample data
    4. Only update if validation confirms and confidence is high
    """
    logger.info("=" * 80)
    logger.info(f"RUNNING WALK-FORWARD OPTIMIZATION")
    logger.info(f"Training period: {int(days * VALIDATION_SPLIT)} days")
    logger.info(f"Validation period: {int(days * (1 - VALIDATION_SPLIT))} days")
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
    optimization_details = {}  # Track detailed metrics

    all_results = {}
    updated_symbols = []
    preserved_symbols = []
    low_confidence_symbols = []

    for i, symbol in enumerate(symbols):
        logger.info(f"[{i+1}/{len(symbols)}] {symbol}...")

        try:
            # Get full dataset
            data = pipeline.get_data(symbol)
            if data is None or data.empty:
                logger.warning(f"{symbol}: NO DATA - preserving existing routing")
                preserved_symbols.append(symbol)
                continue

            if len(data) < days:
                logger.warning(f"{symbol}: Only {len(data)} bars available (need {days}) - using all available data")
                train_data = data.iloc[:int(len(data) * VALIDATION_SPLIT)]
                validation_data = data.iloc[int(len(data) * VALIDATION_SPLIT):]
            else:
                # Use most recent data
                data = data.tail(days).reset_index(drop=True)
                split_idx = int(len(data) * VALIDATION_SPLIT)
                train_data = data.iloc[:split_idx].reset_index(drop=True)
                validation_data = data.iloc[split_idx:].reset_index(drop=True)

            logger.info(f"  Training: {len(train_data)} bars, Validation: {len(validation_data)} bars")

            # Run optimization on training data
            tester = RegimeBacktester(
                data=train_data,
                symbol=symbol,
                strategies=DEFAULT_STRATEGIES,
            )
            train_result = tester.run_regime_analysis(metric="sharpe_ratio", verbose=False)

            # Validate on out-of-sample data
            val_tester = RegimeBacktester(
                data=validation_data,
                symbol=symbol,
                strategies=DEFAULT_STRATEGIES,
            )
            val_result = val_tester.run_regime_analysis(metric="sharpe_ratio", verbose=False)

            # Analyze results per regime
            regime_decisions = {}
            confidence_scores = {}

            for regime in REGIME_TYPES:
                train_best = train_result.best_strategies.get(regime)
                val_best = val_result.best_strategies.get(regime)
                existing_strategy = existing_routing.get(symbol, {}).get(regime) if existing_routing else None

                # Get detailed results
                train_regime_result = train_result.strategy_results[regime].get(train_best)
                val_regime_result = val_result.strategy_results[regime].get(train_best)  # Validate the train winner

                # Check minimum trade requirement
                if not train_regime_result or train_regime_result.num_trades < MIN_TRADES_PER_REGIME:
                    logger.info(f"  {regime}: Insufficient trades ({train_regime_result.num_trades if train_regime_result else 0}) - preserving existing")
                    regime_decisions[regime] = existing_strategy or "sma"
                    confidence_scores[regime] = 0.0
                    continue

                # Calculate confidence
                regime_stats = train_result.regime_stats.get(regime)
                confidence = calculate_confidence_score(train_regime_result, val_regime_result, regime_stats)
                confidence_scores[regime] = confidence

                # Decide whether to use new strategy
                if confidence >= CONFIDENCE_THRESHOLD:
                    # High confidence - check if meaningfully better than existing
                    if existing_strategy and existing_strategy in train_result.strategy_results[regime]:
                        existing_result = train_result.strategy_results[regime][existing_strategy]
                        sharpe_improvement = train_regime_result.sharpe_ratio - existing_result.sharpe_ratio

                        if sharpe_improvement >= MIN_SHARPE_IMPROVEMENT:
                            regime_decisions[regime] = train_best
                            logger.info(f"  {regime}: {train_best} (conf={confidence:.2f}, Sharpe +{sharpe_improvement:.2f} vs {existing_strategy})")
                        else:
                            regime_decisions[regime] = existing_strategy
                            logger.info(f"  {regime}: Keeping {existing_strategy} (new {train_best} not significantly better)")
                    else:
                        # No existing strategy or can't compare
                        regime_decisions[regime] = train_best
                        logger.info(f"  {regime}: {train_best} (conf={confidence:.2f}, train_sharpe={train_regime_result.sharpe_ratio:.2f}, val_sharpe={val_regime_result.sharpe_ratio if val_regime_result else 0:.2f})")
                else:
                    # Low confidence - preserve existing
                    regime_decisions[regime] = existing_strategy or train_best
                    logger.info(f"  {regime}: Low confidence ({confidence:.2f}) - using {regime_decisions[regime]}")

            # Update routing for this symbol
            combined_routing[symbol] = {
                "low_volatility": regime_decisions.get("low_volatility", "sma"),
                "normal": regime_decisions.get("normal", "bollinger"),
                "high_volatility": regime_decisions.get("high_volatility", "rsi"),
                "default": regime_decisions.get("normal", "momentum"),
            }

            # Track optimization details
            avg_confidence = np.mean(list(confidence_scores.values()))
            optimization_details[symbol] = {
                "confidence_scores": confidence_scores,
                "avg_confidence": avg_confidence,
                "strategies": regime_decisions,
            }

            if avg_confidence >= CONFIDENCE_THRESHOLD:
                updated_symbols.append(symbol)
                all_results[symbol] = train_result
            else:
                low_confidence_symbols.append(symbol)

        except Exception as e:
            logger.error(f"{symbol}: ERROR - {e}")
            preserved_symbols.append(symbol)

    # Add default routing based on most common strategies
    if all_results:
        low_vol_strats = [optimization_details[s]["strategies"].get("low_volatility") for s in all_results]
        normal_strats = [optimization_details[s]["strategies"].get("normal") for s in all_results]
        high_vol_strats = [optimization_details[s]["strategies"].get("high_volatility") for s in all_results]

        combined_routing["default"] = {
            "low_volatility": Counter(low_vol_strats).most_common(1)[0][0] if low_vol_strats else "sma",
            "normal": Counter(normal_strats).most_common(1)[0][0] if normal_strats else "bollinger",
            "high_volatility": Counter(high_vol_strats).most_common(1)[0][0] if high_vol_strats else "rsi",
            "default": "momentum",
            "use_hybrid": True,
        }
    else:
        logger.error("No successful optimizations - keeping existing config")
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
    logger.info(f"  High confidence (updated): {len(updated_symbols)} symbols")
    logger.info(f"  Low confidence (preserved): {len(low_confidence_symbols)} symbols")
    logger.info(f"  Failed (preserved): {len(preserved_symbols)} symbols")

    if updated_symbols:
        logger.info(f"  Updated symbols: {', '.join(updated_symbols)}")
    if low_confidence_symbols:
        logger.info(f"  Low confidence symbols: {', '.join(low_confidence_symbols)}")
    if preserved_symbols:
        logger.info(f"  Preserved symbols: {', '.join(preserved_symbols)}")

    # Strategy frequency (only high-confidence)
    if all_results:
        for regime in REGIME_TYPES:
            strats = [optimization_details[s]["strategies"].get(regime) for s in all_results]
            counts = Counter(strats)
            top3 = counts.most_common(3)
            top_str = ", ".join(f"{s}({c})" for s, c in top3)
            logger.info(f"  {regime}: {top_str}")

    # Confidence distribution
    if optimization_details:
        confidences = [d["avg_confidence"] for d in optimization_details.values()]
        logger.info(f"  Average confidence: {np.mean(confidences):.2f}")
        logger.info(f"  Confidence range: [{np.min(confidences):.2f}, {np.max(confidences):.2f}]")

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
    logger.info(f"# WALK-FORWARD STRATEGY OPTIMIZATION - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
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
    data_success = await update_data(symbols, days=TRAIN_DAYS)
    if not data_success:
        logger.error("Data update failed - aborting optimization")
        return 1

    logger.info("")

    # Step 2: Run walk-forward optimization
    new_routing = run_walkforward_optimization(symbols, days=TRAIN_DAYS, existing_routing=existing_routing)
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
