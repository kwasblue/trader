#!/usr/bin/env python
"""
Unified Optimization Workflow - Complete Strategy & Timeframe Optimization

This tool combines regime-aware strategy selection with timeframe optimization
to generate a complete, production-ready strategy routing config.

Workflow:
1. Run regime-aware strategy selection to find best strategies per regime
2. Run timeframe optimization on the selected strategies
3. Combine results into a unified routing config
4. Save to strategy_routing.json

Usage:
    python tools/unified_optimizer.py AAPL TSLA MSFT --days 750 --save

    # Custom strategies
    python tools/unified_optimizer.py AAPL --strategies rsi,sma,momentum --save

    # Specific timeframes
    python tools/unified_optimizer.py AAPL --timeframes 5min,15min,30min --save
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loggers.logger import Logger
from core.unified_data_pipeline import UnifiedDataPipeline
from core.backtest.strategy_selector import StrategySelector
from core.backtest.timeframe_optimizer import TimeframeOptimizer, BacktestResult
from core.backtest.regime_backtest import RegimeAnalysisResult, REGIME_TYPES


@dataclass
class UnifiedOptimizationResult:
    """Result of unified optimization."""
    symbol: str
    regime_analysis: RegimeAnalysisResult
    timeframe_results: List[BacktestResult]
    optimal_config: Dict[str, Any]


class UnifiedOptimizer:
    """
    Unified optimizer that combines strategy selection and timeframe optimization.

    This ensures consistency between:
    - Strategy selection (which strategy per regime)
    - Timeframe selection (which timeframe per regime)
    """

    def __init__(
        self,
        symbols: List[str],
        strategies: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        data_pipeline: Optional[UnifiedDataPipeline] = None
    ):
        """
        Initialize unified optimizer.

        Args:
            symbols: List of symbols to optimize
            strategies: List of strategies to test (default: all available)
            timeframes: List of timeframes to test (default: ['5min', '15min', '30min', '1hour', 'day'])
            data_pipeline: Optional UnifiedDataPipeline instance
        """
        self.symbols = symbols
        self.strategies = strategies  # None = test all
        self.timeframes = timeframes or ['5min', '15min', '30min', '1hour', 'day']

        self.pipeline = data_pipeline or UnifiedDataPipeline()

        self.logger = Logger(
            "unified_optimizer.log",
            "UnifiedOptimizer",
            propagate=True,
            level=10
        ).get_logger()

        self.results: Dict[str, UnifiedOptimizationResult] = {}

        self.logger.info(
            f"UnifiedOptimizer initialized: {len(symbols)} symbols × {len(self.timeframes)} timeframes"
        )

    async def run_optimization(
        self,
        days: int = 750,
        metric: str = "sharpe_ratio"
    ) -> Dict[str, UnifiedOptimizationResult]:
        """
        Run complete optimization workflow.

        Args:
            days: Days of historical data to use
            metric: Metric for ranking (sharpe_ratio, total_return, win_rate)

        Returns:
            Dict mapping symbol -> UnifiedOptimizationResult
        """
        self.logger.info("=" * 80)
        self.logger.info("UNIFIED OPTIMIZATION WORKFLOW")
        self.logger.info("=" * 80)
        self.logger.info(f"Symbols: {self.symbols}")
        self.logger.info(f"Timeframes: {self.timeframes}")
        self.logger.info(f"Days of data: {days}")
        self.logger.info(f"Ranking metric: {metric}")
        self.logger.info("=" * 80)

        for symbol in self.symbols:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"OPTIMIZING {symbol}")
            self.logger.info(f"{'='*60}")

            try:
                result = await self._optimize_symbol(symbol, days, metric)
                self.results[symbol] = result

            except Exception as e:
                self.logger.exception(f"Error optimizing {symbol}: {e}")

        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"OPTIMIZATION COMPLETE: {len(self.results)}/{len(self.symbols)} symbols")
        self.logger.info("=" * 80)

        return self.results

    async def _optimize_symbol(
        self,
        symbol: str,
        days: int,
        metric: str
    ) -> UnifiedOptimizationResult:
        """
        Optimize a single symbol.

        Steps:
        1. Load historical data
        2. Run regime-aware strategy selection
        3. Run timeframe optimization on selected strategies
        4. Combine into unified config
        """
        # Step 1: Load data
        self.logger.info(f"[{symbol}] Loading historical data...")
        data = self.pipeline.get_data(symbol, timeframe='day')

        if data is None or data.empty:
            self.logger.warning(f"[{symbol}] No data available, skipping...")
            raise ValueError(f"No data for {symbol}")

        if len(data) > days:
            data = data.tail(days).reset_index(drop=True)

        self.logger.info(f"[{symbol}] Loaded {len(data)} bars")

        # Step 2: Regime-aware strategy selection
        self.logger.info(f"[{symbol}] Running regime-aware strategy selection...")

        selector = StrategySelector(
            data=data,
            initial_capital=100000,
            transaction_cost=0.001
        )

        regime_analysis = selector.select_best_strategies_by_regime(
            symbol=symbol,
            metric=metric,
            verbose=False
        )

        # Extract best strategies per regime
        best_strategies = regime_analysis.best_strategies

        self.logger.info(f"[{symbol}] Best strategies per regime:")
        for regime, strategy in best_strategies.items():
            self.logger.info(f"  {regime}: {strategy}")

        # Step 3: Timeframe optimization on selected strategies
        self.logger.info(f"[{symbol}] Running timeframe optimization...")

        # Get unique strategies from all regimes
        unique_strategies = list(set(best_strategies.values()))

        timeframe_optimizer = TimeframeOptimizer(
            symbols=[symbol],
            timeframes=self.timeframes,
            strategies=unique_strategies,
            data_pipeline=self.pipeline
        )

        timeframe_results = await timeframe_optimizer.run_optimization(
            days=days,
            min_bars=100
        )

        # Step 4: Combine into unified config
        optimal_config = self._build_unified_config(
            symbol=symbol,
            regime_analysis=regime_analysis,
            timeframe_results=timeframe_results
        )

        return UnifiedOptimizationResult(
            symbol=symbol,
            regime_analysis=regime_analysis,
            timeframe_results=timeframe_results,
            optimal_config=optimal_config
        )

    def _build_unified_config(
        self,
        symbol: str,
        regime_analysis: RegimeAnalysisResult,
        timeframe_results: List[BacktestResult]
    ) -> Dict[str, Any]:
        """
        Build unified config from regime analysis and timeframe results.

        Logic:
        - Use best strategy per regime from regime analysis
        - Use best timeframe per (regime, strategy) from timeframe results
        - Apply regime-appropriate timeframe heuristics:
          - High volatility: prefer shorter timeframes (5min, 15min)
          - Low volatility: prefer longer timeframes (30min, 1hour, day)
        """
        config = {}

        for regime in REGIME_TYPES:
            best_strategy = regime_analysis.best_strategies.get(regime, "momentum")

            # Find best timeframe for this (regime, strategy) combination
            # Filter results for this strategy
            strategy_results = [
                r for r in timeframe_results
                if r.strategy == best_strategy and r.symbol == symbol
            ]

            if not strategy_results:
                # Fallback to default timeframe
                if regime == "high_volatility":
                    best_timeframe = "15min"
                elif regime == "low_volatility":
                    best_timeframe = "1hour"
                else:
                    best_timeframe = "30min"
            else:
                # Apply regime-appropriate timeframe filtering
                if regime == "high_volatility":
                    # Prefer short timeframes
                    filtered = [r for r in strategy_results if r.timeframe in ['5min', '15min']]
                    candidates = filtered if filtered else strategy_results
                elif regime == "low_volatility":
                    # Prefer long timeframes
                    filtered = [r for r in strategy_results if r.timeframe in ['30min', '1hour', 'day']]
                    candidates = filtered if filtered else strategy_results
                else:
                    # Normal: prefer medium timeframes
                    filtered = [r for r in strategy_results if r.timeframe in ['15min', '30min']]
                    candidates = filtered if filtered else strategy_results

                # Sort by score and take best
                candidates.sort(key=lambda r: r.score, reverse=True)
                best_timeframe = candidates[0].timeframe

            config[regime] = {
                "strategy": best_strategy,
                "timeframe": best_timeframe
            }

            self.logger.info(
                f"  {regime}: {best_strategy} @ {best_timeframe}"
            )

        # Add default (use normal regime)
        if "normal" in config:
            config["default"] = config["normal"]

        # Add use_hybrid flag
        config["use_hybrid"] = False

        return {symbol: config}

    def save_config(
        self,
        config_path: Optional[str] = None,
        merge: bool = True
    ) -> Path:
        """
        Save unified config to strategy_routing.json.

        Args:
            config_path: Path to config file (default: config/strategy_routing.json)
            merge: If True, merge with existing config

        Returns:
            Path to saved config
        """
        if config_path is None:
            config_path = ROOT / "config" / "strategy_routing.json"
        else:
            config_path = Path(config_path)

        # Combine all symbol configs
        combined_config = {}
        for result in self.results.values():
            combined_config.update(result.optimal_config)

        # Load existing config if merging
        existing = {}
        if merge and config_path.exists():
            with open(config_path) as f:
                existing = json.load(f)

        # Merge
        existing.update(combined_config)

        # Save
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(existing, f, indent=2)

        self.logger.info(f"Saved unified config to {config_path}")
        return config_path

    def print_summary(self):
        """Print summary of optimization results."""
        if not self.results:
            print("No results available")
            return

        print("\n" + "=" * 80)
        print("UNIFIED OPTIMIZATION SUMMARY")
        print("=" * 80)

        for symbol, result in self.results.items():
            print(f"\n{symbol}:")
            print("-" * 80)

            # Print regime analysis
            print("\nRegime-Aware Strategy Selection:")
            for regime in REGIME_TYPES:
                strategy = result.regime_analysis.best_strategies.get(regime, "N/A")
                if regime in result.regime_analysis.strategy_results:
                    if strategy in result.regime_analysis.strategy_results[regime]:
                        r = result.regime_analysis.strategy_results[regime][strategy]
                        print(
                            f"  {regime:<18}: {strategy:<15} "
                            f"(Sharpe: {r.sharpe_ratio:>5.2f}, Return: {r.total_return:>+6.1%})"
                        )

            # Print final config
            print("\nFinal Configuration:")
            config = result.optimal_config[symbol]
            for regime in REGIME_TYPES:
                if regime in config:
                    strat = config[regime]['strategy']
                    tf = config[regime]['timeframe']
                    print(f"  {regime:<18}: {strat:<15} @ {tf}")

        print("\n" + "=" * 80)


# ============================================================================
# CLI INTERFACE
# ============================================================================

async def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Unified strategy and timeframe optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize specific symbols
  python tools/unified_optimizer.py AAPL TSLA MSFT --save

  # Custom strategies
  python tools/unified_optimizer.py AAPL --strategies rsi,sma,momentum --save

  # Custom timeframes
  python tools/unified_optimizer.py AAPL --timeframes 5min,15min,30min --save

  # More historical data
  python tools/unified_optimizer.py AAPL TSLA --days 1000 --save
        """
    )

    parser.add_argument(
        'symbols', nargs='+',
        help='Symbols to optimize'
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
        help='Days of historical data'
    )
    parser.add_argument(
        '--metric', default='sharpe_ratio',
        choices=['sharpe_ratio', 'total_return', 'win_rate'],
        help='Ranking metric'
    )
    parser.add_argument(
        '--save', action='store_true',
        help='Save to config/strategy_routing.json'
    )
    parser.add_argument(
        '--output',
        help='Custom output path for config file'
    )

    args = parser.parse_args()

    # Parse strategies
    strategies = None
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(',')]

    # Parse timeframes
    timeframes = [tf.strip() for tf in args.timeframes.split(',')]

    # Create optimizer
    optimizer = UnifiedOptimizer(
        symbols=[s.upper() for s in args.symbols],
        strategies=strategies,
        timeframes=timeframes
    )

    # Run optimization
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
        print(f"SAVED TO: {config_path}")
        print(f"{'='*80}")
        print("\nNext steps:")
        print("  1. Review the config file")
        print("  2. Restart the trader to use the new configuration")
        print("  3. Monitor performance and adjust as needed")


if __name__ == '__main__':
    asyncio.run(main())
