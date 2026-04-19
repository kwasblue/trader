#!/usr/bin/env python3
"""
Strategy Optimization with Chronological Holdout Validation

IMPORTANT: This uses a single train/validation split, NOT true walk-forward optimization.

Key features:
1. Uses 750 days of data (3 years)
2. Chronological split: trains on first 80%, validates on last 20%
3. Tiered minimum trade requirements for statistical reliability
4. Preserves ENTIRE symbol config when average confidence is low
5. Requires positive validation Sharpe as floor
6. Enhanced logging for train vs validation comparison

This should run weekly or monthly, not daily.

True walk-forward (future improvement) would use:
- Rolling windows (e.g., optimize on months 1-18, validate on 19-24)
- Multiple folds with aggregated results
- More robust against overfitting to a single split
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
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from core.unified_data_pipeline import UnifiedDataPipeline
from loggers.logger import Logger


# Setup logger
logger = Logger("strategy_optimize.log", "StrategyOptimize", propagate=True).get_logger()

# Configuration
TRAIN_DAYS = 750  # Use 3 years of data for training
VALIDATION_SPLIT = 0.8  # Train on 80%, validate on 20%

# Tiered minimum trade requirements
MIN_TRADES_REJECT = 5  # Below this = always reject
MIN_TRADES_WEAK = 10   # 5-10 trades = heavy confidence penalty
MIN_TRADES_MODERATE = 20  # 10-20 trades = moderate penalty

# Config validation
assert MIN_TRADES_REJECT < MIN_TRADES_WEAK < MIN_TRADES_MODERATE, \
    "Trade thresholds must be ordered: REJECT < WEAK < MODERATE"

# Minimum practical improvement threshold (NOT statistical significance)
MIN_SHARPE_IMPROVEMENT = 0.3  # Heuristic threshold for meaningful improvement

# Confidence thresholds
CONFIDENCE_THRESHOLD = 0.45  # Minimum avg confidence to update symbol
MIN_REGIME_CONFIDENCE = 0.25  # Floor - no regime can be below this

# Validation quality floors
MIN_VALIDATION_SHARPE = 0.0  # Validation must be non-negative
MIN_VALIDATION_TRADES = 5    # Validation must have some trades


async def update_data(symbols, days=TRAIN_DAYS, force_reprocess=False):
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
            force_reprocess=force_reprocess,
            process_data=True,
        )

        logger.info(f"Data update complete: {results}")
        return True
    except Exception as e:
        logger.error(f"Error updating data: {e}")
        return False


def calculate_trade_score(num_trades: int) -> float:
    """
    Calculate trade count score with tiered penalties.

    Returns 0.0 to 1.0 based on trade count reliability.
    """
    if num_trades < MIN_TRADES_REJECT:
        return 0.0
    elif num_trades < MIN_TRADES_WEAK:
        # Heavy penalty: 5-10 trades = 0.2-0.4
        return 0.2 + (num_trades - MIN_TRADES_REJECT) / (MIN_TRADES_WEAK - MIN_TRADES_REJECT) * 0.2
    elif num_trades < MIN_TRADES_MODERATE:
        # Moderate penalty: 10-20 trades = 0.4-0.7
        return 0.4 + (num_trades - MIN_TRADES_WEAK) / (MIN_TRADES_MODERATE - MIN_TRADES_WEAK) * 0.3
    else:
        # Normal score: 20+ trades, saturates at 30
        return min(0.7 + (num_trades - MIN_TRADES_MODERATE) / 10 * 0.3, 1.0)


def calculate_confidence_score(
    train_result,
    validation_result,
    regime_stats
) -> float:
    """
    Calculate confidence score for a strategy selection.

    Factors:
    - Number of trades (tiered scoring)
    - Train/validation consistency
    - Regime data availability

    Returns: 0.0 to 1.0
    """
    scores = []

    # Trade count score with tiered penalties
    trade_score = calculate_trade_score(train_result.num_trades)
    scores.append(trade_score)

    # Validation consistency score
    if validation_result and validation_result.num_trades >= MIN_VALIDATION_TRADES:
        train_sharpe = train_result.sharpe_ratio
        val_sharpe = validation_result.sharpe_ratio

        # Check if signs match (both positive or both negative)
        if train_sharpe * val_sharpe > 0:
            # Same sign - measure consistency
            diff = abs(train_sharpe - val_sharpe) / (abs(train_sharpe) + 0.01)
            consistency_score = max(0, 1 - diff)
        else:
            # Opposite signs = very poor consistency
            consistency_score = 0.1

        scores.append(consistency_score)
    else:
        # No validation data = low confidence
        scores.append(0.2)

    # Regime data availability
    # NOTE: This can penalize rare but important regimes (e.g., high-volatility crisis periods)
    # We cap the penalty so rare regimes aren't suppressed too much
    regime_coverage = (getattr(regime_stats, "pct_of_total", 0.0) or 0.0) / 100.0
    regime_coverage_score = max(0.3, regime_coverage)  # Floor at 0.3 to limit rare-regime penalty
    scores.append(regime_coverage_score)

    # Weighted average
    weights = [0.4, 0.4, 0.2]  # trades, validation, coverage
    confidence = sum(s * w for s, w in zip(scores, weights))

    return confidence


def run_optimization(symbols, days=TRAIN_DAYS, existing_routing=None):
    """
    Run chronological holdout optimization.

    For each symbol:
    1. Split data chronologically: first 80% train, last 20% validate
    2. Find best strategy on training data per regime
    3. Validate on out-of-sample data
    4. Only update if:
       - All regimes meet minimum confidence floor
       - Average confidence meets threshold
       - Validation quality meets minimum bars
    5. Otherwise preserve ENTIRE existing symbol config
    """
    logger.info("=" * 80)
    logger.info(f"RUNNING CHRONOLOGICAL HOLDOUT OPTIMIZATION")
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
        logger.info(f"\n[{i+1}/{len(symbols)}] {symbol}...")

        try:
            # Get full dataset
            data = pipeline.get_data(symbol)
            if data is None or data.empty:
                logger.warning(f"{symbol}: NO DATA - preserving existing routing")
                preserved_symbols.append(symbol)
                continue

            if len(data) < days:
                logger.warning(f"{symbol}: Only {len(data)} bars available (need {days})")
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
            regime_details = {}  # For detailed logging

            for regime in REGIME_TYPES:
                train_best = train_result.best_strategies.get(regime)
                val_best = val_result.best_strategies.get(regime)
                existing_strategy = existing_routing.get(symbol, {}).get(regime) if existing_routing else None

                # Get detailed results
                train_regime_result = train_result.strategy_results[regime].get(train_best)
                val_regime_result = val_result.strategy_results[regime].get(train_best) if train_best else None

                # Store details for logging
                regime_details[regime] = {
                    "incumbent": existing_strategy,
                    "train_winner": train_best,
                    "val_winner": val_best,
                    "train_sharpe": train_regime_result.sharpe_ratio if train_regime_result else None,
                    "val_sharpe": val_regime_result.sharpe_ratio if val_regime_result else None,
                    "train_trades": train_regime_result.num_trades if train_regime_result else 0,
                    "val_trades": val_regime_result.num_trades if val_regime_result else 0,
                }

                # Check minimum trade requirement
                if not train_regime_result or train_regime_result.num_trades < MIN_TRADES_REJECT:
                    logger.info(f"  {regime}: REJECT - insufficient trades ({train_regime_result.num_trades if train_regime_result else 0} < {MIN_TRADES_REJECT})")
                    regime_decisions[regime] = existing_strategy or "sma"
                    confidence_scores[regime] = 0.0
                    continue

                # Require validation result to exist
                if not val_regime_result:
                    logger.info(f"  {regime}: NO VALIDATION RESULT - preserving incumbent")
                    regime_decisions[regime] = existing_strategy or train_best or "sma"
                    confidence_scores[regime] = 0.0
                    continue

                # Check validation quality floor
                if val_regime_result:
                    if val_regime_result.num_trades < MIN_VALIDATION_TRADES:
                        logger.info(f"  {regime}: LOW VAL TRADES - {val_regime_result.num_trades} < {MIN_VALIDATION_TRADES}")
                        regime_decisions[regime] = existing_strategy or train_best
                        confidence_scores[regime] = 0.15
                        continue

                    if val_regime_result.sharpe_ratio < MIN_VALIDATION_SHARPE:
                        logger.info(f"  {regime}: NEGATIVE VAL SHARPE - {val_regime_result.sharpe_ratio:.2f}")
                        regime_decisions[regime] = existing_strategy or train_best
                        confidence_scores[regime] = 0.2
                        continue

                # Calculate confidence
                regime_stats = train_result.regime_stats.get(regime)
                confidence = calculate_confidence_score(train_regime_result, val_regime_result, regime_stats)
                confidence_scores[regime] = confidence

                # Decide whether to use new strategy
                if confidence >= CONFIDENCE_THRESHOLD:
                    # High confidence - check if meaningfully better than existing
                    if existing_strategy and existing_strategy in train_result.strategy_results[regime]:
                        # Get incumbent's validation performance
                        incumbent_val_result = val_result.strategy_results[regime].get(existing_strategy)

                        # Compare using VALIDATION Sharpe (not training)
                        candidate_val_sharpe = val_regime_result.sharpe_ratio
                        incumbent_val_sharpe = incumbent_val_result.sharpe_ratio if incumbent_val_result else float("-inf")

                        val_sharpe_improvement = candidate_val_sharpe - incumbent_val_sharpe

                        # Also check training improvement for context
                        existing_train_result = train_result.strategy_results[regime][existing_strategy]
                        train_sharpe_improvement = train_regime_result.sharpe_ratio - existing_train_result.sharpe_ratio

                        if val_sharpe_improvement >= MIN_SHARPE_IMPROVEMENT:
                            regime_decisions[regime] = train_best
                            logger.info(f"  {regime}: SWITCH to {train_best} (conf={confidence:.2f}, val_Sharpe +{val_sharpe_improvement:.2f}, train_Sharpe +{train_sharpe_improvement:.2f} vs {existing_strategy})")
                        else:
                            regime_decisions[regime] = existing_strategy
                            logger.info(f"  {regime}: KEEP {existing_strategy} (candidate val_improvement {val_sharpe_improvement:.2f} < {MIN_SHARPE_IMPROVEMENT})")
                    else:
                        # No existing strategy or can't compare
                        regime_decisions[regime] = train_best
                        logger.info(f"  {regime}: SELECT {train_best} (conf={confidence:.2f}, train={train_regime_result.sharpe_ratio:.2f}, val={val_regime_result.sharpe_ratio:.2f})")
                else:
                    # Low confidence - preserve existing
                    regime_decisions[regime] = existing_strategy or train_best
                    logger.info(f"  {regime}: LOW CONF ({confidence:.2f}) - using {regime_decisions[regime]}")

            # Calculate overall metrics
            avg_confidence = np.mean(list(confidence_scores.values()))
            min_confidence = np.min(list(confidence_scores.values()))

            # CRITICAL: Only update if BOTH conditions met:
            # 1. Average confidence meets threshold
            # 2. No regime falls below minimum floor
            if avg_confidence >= CONFIDENCE_THRESHOLD and min_confidence >= MIN_REGIME_CONFIDENCE:
                # High confidence across all regimes - update symbol routing
                combined_routing[symbol] = {
                    "low_volatility": regime_decisions.get("low_volatility", "sma"),
                    "normal": regime_decisions.get("normal", "bollinger"),
                    "high_volatility": regime_decisions.get("high_volatility", "rsi"),
                    # NOTE: default inherits from normal regime (intentional - ambiguous market = treat as normal)
                    "default": regime_decisions.get("normal", "momentum"),
                }
                updated_symbols.append(symbol)
                all_results[symbol] = train_result

                logger.info(f"  ✓ UPDATED (avg_conf={avg_confidence:.2f}, min_conf={min_confidence:.2f})")
            else:
                # Low confidence - preserve ENTIRE existing symbol config
                if symbol in existing_routing:
                    combined_routing[symbol] = existing_routing[symbol].copy()
                    logger.info(f"  ✗ PRESERVED existing config (avg_conf={avg_confidence:.2f}, min_conf={min_confidence:.2f})")
                else:
                    # No existing config - use decisions but mark as low confidence
                    combined_routing[symbol] = {
                        "low_volatility": regime_decisions.get("low_volatility", "sma"),
                        "normal": regime_decisions.get("normal", "bollinger"),
                        "high_volatility": regime_decisions.get("high_volatility", "rsi"),
                        "default": regime_decisions.get("normal", "momentum"),
                    }
                    logger.info(f"  ⚠ NEW SYMBOL with low confidence (avg_conf={avg_confidence:.2f})")

                low_confidence_symbols.append(symbol)

            # Track optimization details for summary
            optimization_details[symbol] = {
                "confidence_scores": confidence_scores,
                "avg_confidence": avg_confidence,
                "min_confidence": min_confidence,
                "strategies": regime_decisions,
                "regime_details": regime_details,
                "updated": avg_confidence >= CONFIDENCE_THRESHOLD and min_confidence >= MIN_REGIME_CONFIDENCE,
            }

        except Exception as e:
            logger.error(f"{symbol}: ERROR - {e}", exc_info=True)
            preserved_symbols.append(symbol)

    # Add default routing based on most common strategies from HIGH-CONFIDENCE symbols only
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
        logger.error("No successful optimizations - using defaults")
        combined_routing["default"] = {
            "low_volatility": "sma",
            "normal": "bollinger",
            "high_volatility": "rsi",
            "default": "momentum",
            "use_hybrid": True,
        }

    # Determine hybrid sizing for each symbol
    hybrid_config = determine_hybrid_config(combined_routing)

    # Add use_hybrid to routing config
    for symbol in combined_routing:
        if symbol != "default" and symbol in hybrid_config:
            combined_routing[symbol]["use_hybrid"] = hybrid_config[symbol]["enabled"]

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION SUMMARY:")
    logger.info(f"  High confidence (updated): {len(updated_symbols)} symbols")
    logger.info(f"  Low confidence (preserved): {len(low_confidence_symbols)} symbols")
    logger.info(f"  Failed (preserved): {len(preserved_symbols)} symbols")

    if updated_symbols:
        logger.info(f"  Updated: {', '.join(updated_symbols)}")
    if low_confidence_symbols:
        logger.info(f"  Low confidence: {', '.join(low_confidence_symbols)}")
    if preserved_symbols:
        logger.info(f"  Errors: {', '.join(preserved_symbols)}")

    # Strategy frequency (only high-confidence symbols)
    if all_results:
        logger.info("\n  Strategy Adoption (high-confidence only):")
        for regime in ["low_volatility", "normal", "high_volatility"]:
            strats = [optimization_details[s]["strategies"].get(regime) for s in all_results]
            counts = Counter(strats)
            top3 = counts.most_common(3)
            top_str = ", ".join(f"{s}({c})" for s, c in top3)
            logger.info(f"    {regime}: {top_str}")

    # Confidence distribution
    if optimization_details:
        confidences = [d["avg_confidence"] for d in optimization_details.values()]
        min_confidences = [d["min_confidence"] for d in optimization_details.values()]
        logger.info(f"\n  Average confidence: {np.mean(confidences):.2f} (range: [{np.min(confidences):.2f}, {np.max(confidences):.2f}])")
        logger.info(f"  Minimum regime confidence: {np.mean(min_confidences):.2f} (range: [{np.min(min_confidences):.2f}, {np.max(min_confidences):.2f}])")

    # Hybrid summary
    hybrid_enabled = sum(1 for h in hybrid_config.values() if h["enabled"])
    hybrid_disabled = len(hybrid_config) - hybrid_enabled
    logger.info(f"\n  Hybrid sizing: {hybrid_enabled} enabled, {hybrid_disabled} disabled")
    logger.info("=" * 80)

    return combined_routing


async def main():
    """Main entry point."""
    start_time = datetime.now()
    logger.info("")
    logger.info("#" * 80)
    logger.info(f"# STRATEGY OPTIMIZATION - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
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

    # Step 1: Update data (only force reprocess if needed)
    data_success = await update_data(symbols, days=TRAIN_DAYS, force_reprocess=False)
    if not data_success:
        logger.error("Data update failed - aborting optimization")
        return 1

    logger.info("")

    # Step 2: Run optimization
    new_routing = run_optimization(symbols, days=TRAIN_DAYS, existing_routing=existing_routing)
    if new_routing is None:
        logger.error("Optimization failed - keeping existing config")
        return 1

    logger.info("")

    # Step 3: Save config with backup
    backup_path = routing_path.parent / f"strategy_routing.backup.{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(backup_path, "w") as f:
        json.dump(existing_routing, f, indent=2)
    logger.info(f"Backup saved to: {backup_path}")

    with open(routing_path, "w") as f:
        json.dump(new_routing, f, indent=2)
    logger.info(f"New routing saved to: {routing_path}")

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info("")
    logger.info("#" * 80)
    logger.info(f"# COMPLETE - Duration: {duration:.1f}s")
    logger.info("#" * 80)
    logger.info("")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
