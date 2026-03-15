# core/backtest/timeframe_optimizer.py
"""
Timeframe Optimizer - Find Optimal Timeframes for Strategies

This tool backtests each strategy on multiple timeframes to find which
timeframe performs best for each (symbol, regime, strategy) combination.

Usage:
    from core.backtest.timeframe_optimizer import TimeframeOptimizer

    # Run optimization
    optimizer = TimeframeOptimizer(
        symbols=['AAPL', 'TSLA', 'MSFT'],
        timeframes=['5min', '15min', '30min', '1hour'],
        strategies=['rsi', 'sma', 'meanreversion', 'bollinger'],
        regimes=['low_volatility', 'normal', 'high_volatility']
    )

    results = await optimizer.run_optimization()

    # Generate optimal config
    config = optimizer.generate_routing_config(results)
    optimizer.save_config(config, 'config/strategy_routing_optimized.json')
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pandas as pd

from loggers.logger import Logger
from core.unified_data_pipeline import UnifiedDataPipeline


@dataclass
class BacktestResult:
    """Results from a single backtest run."""
    symbol: str
    strategy: str
    timeframe: str
    regime: Optional[str] = None

    # Performance metrics
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # Trade statistics
    num_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # Other metrics
    calmar_ratio: float = 0.0
    bars_tested: int = 0

    def __post_init__(self):
        """Calculate composite score for ranking."""
        # Weighted score: Sharpe (40%) + Return (30%) + Win Rate (20%) + Calmar (10%)
        self.score = (
            self.sharpe_ratio * 0.4 +
            self.total_return * 0.3 +
            self.win_rate * 0.2 +
            self.calmar_ratio * 0.1
        )


class TimeframeOptimizer:
    """
    Optimize timeframes for strategies across symbols and regimes.

    This runs comprehensive backtests to find the best timeframe for each
    (symbol, strategy, regime) combination based on historical performance.
    """

    def __init__(
        self,
        symbols: List[str],
        timeframes: List[str],
        strategies: List[str],
        regimes: Optional[List[str]] = None,
        data_pipeline: Optional[UnifiedDataPipeline] = None
    ):
        """
        Initialize optimizer.

        Args:
            symbols: List of symbols to optimize
            timeframes: List of timeframes to test (e.g., ['5min', '15min', '30min'])
            strategies: List of strategy names to test
            regimes: Optional list of regimes (if None, optimizes overall)
            data_pipeline: Optional UnifiedDataPipeline instance
        """
        self.symbols = symbols
        self.timeframes = timeframes
        self.strategies = strategies
        self.regimes = regimes or ['overall']

        self.pipeline = data_pipeline or UnifiedDataPipeline()

        self.logger = Logger(
            "timeframe_optimizer.log",
            "TimeframeOptimizer",
            propagate=True,
            level=10
        ).get_logger()

        # Results storage
        self.results: List[BacktestResult] = []

        self.logger.info(
            f"TimeframeOptimizer initialized: {len(symbols)} symbols × "
            f"{len(timeframes)} timeframes × {len(strategies)} strategies"
        )

    async def run_optimization(
        self,
        days: int = 750,
        min_bars: int = 100
    ) -> List[BacktestResult]:
        """
        Run optimization across all combinations.

        Args:
            days: Number of days of historical data to use
            min_bars: Minimum bars required for valid backtest

        Returns:
            List of BacktestResult objects
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING TIMEFRAME OPTIMIZATION")
        self.logger.info("=" * 80)
        self.logger.info(f"Symbols: {self.symbols}")
        self.logger.info(f"Timeframes: {self.timeframes}")
        self.logger.info(f"Strategies: {self.strategies}")
        self.logger.info(f"Regimes: {self.regimes}")
        self.logger.info("=" * 80)

        # Step 1: Ensure we have data for all timeframes
        await self._ensure_data(days)

        # Step 2: Run backtests for all combinations
        total_tests = len(self.symbols) * len(self.timeframes) * len(self.strategies)
        completed = 0

        for symbol in self.symbols:
            for timeframe in self.timeframes:
                for strategy in self.strategies:
                    completed += 1
                    self.logger.info(
                        f"[{completed}/{total_tests}] Testing {symbol} / "
                        f"{strategy} / {timeframe}"
                    )

                    try:
                        result = await self._run_backtest(
                            symbol, timeframe, strategy, min_bars
                        )

                        if result:
                            self.results.append(result)
                            self.logger.info(
                                f"  ✓ Sharpe: {result.sharpe_ratio:.2f}, "
                                f"Return: {result.total_return:.2%}, "
                                f"Score: {result.score:.2f}"
                            )
                        else:
                            self.logger.warning(f"  ✗ Insufficient data")

                    except Exception as e:
                        self.logger.exception(
                            f"  ✗ Error: {e}"
                        )

        self.logger.info("=" * 80)
        self.logger.info(f"Optimization complete: {len(self.results)} results")
        self.logger.info("=" * 80)

        return self.results

    async def _ensure_data(self, days: int):
        """Ensure historical data exists for all timeframes."""
        self.logger.info("Checking historical data availability...")

        # Check which symbols/timeframes need data
        missing = []

        for symbol in self.symbols:
            available_tfs = self.pipeline.list_available_timeframes(symbol)

            for timeframe in self.timeframes:
                if timeframe not in available_tfs:
                    missing.append((symbol, timeframe))

        if missing:
            self.logger.info(f"Need to fetch data for {len(missing)} combinations")

            # Group by timeframe for efficient fetching
            timeframes_to_fetch = {}
            for symbol, tf in missing:
                if tf not in timeframes_to_fetch:
                    timeframes_to_fetch[tf] = []
                timeframes_to_fetch[tf].append(symbol)

            # Fetch missing data
            for tf, symbols in timeframes_to_fetch.items():
                self.logger.info(f"Fetching {tf} data for {len(symbols)} symbols...")
                await self.pipeline.update_symbols(
                    symbols=symbols,
                    timeframes=[tf],
                    days=days
                )
        else:
            self.logger.info("All required data already available")

    async def _run_backtest(
        self,
        symbol: str,
        timeframe: str,
        strategy: str,
        min_bars: int
    ) -> Optional[BacktestResult]:
        """
        Run backtest for a single combination.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe to test
            strategy: Strategy name
            min_bars: Minimum bars required

        Returns:
            BacktestResult or None if insufficient data
        """
        # Load data for this timeframe
        bars = self.pipeline.get_data(symbol, timeframe=timeframe)

        if bars.empty or len(bars) < min_bars:
            return None

        # Run backtest using UnifiedBacktestRunner
        try:
            from core.backtest.unified_backtest_runner import UnifiedBacktestRunner

            runner = UnifiedBacktestRunner(
                strategy_name=strategy,
                data=bars,
                config={
                    'initial_capital': 10000,
                    'position_sizing': 'volatility_scaled',
                    'stop_loss_atr': 2.0,
                    'take_profit_atr': 3.0,
                }
            )

            metrics = runner.run()

            # Create result
            result = BacktestResult(
                symbol=symbol,
                strategy=strategy,
                timeframe=timeframe,
                total_return=metrics.get('total_return', 0.0),
                sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
                sortino_ratio=metrics.get('sortino_ratio', 0.0),
                max_drawdown=metrics.get('max_drawdown', 0.0),
                win_rate=metrics.get('win_rate', 0.0),
                profit_factor=metrics.get('profit_factor', 0.0),
                num_trades=metrics.get('num_trades', 0),
                avg_win=metrics.get('avg_win', 0.0),
                avg_loss=metrics.get('avg_loss', 0.0),
                calmar_ratio=metrics.get('calmar_ratio', 0.0),
                bars_tested=len(bars)
            )

            return result

        except Exception as e:
            self.logger.exception(f"Backtest failed: {e}")
            return None

    def find_best_timeframe(
        self,
        symbol: str,
        strategy: str,
        regime: Optional[str] = None
    ) -> Optional[str]:
        """
        Find best timeframe for a symbol/strategy combination.

        Args:
            symbol: Stock symbol
            strategy: Strategy name
            regime: Optional regime filter

        Returns:
            Best timeframe string or None
        """
        # Filter results for this combination
        filtered = [
            r for r in self.results
            if r.symbol == symbol and r.strategy == strategy
        ]

        if not filtered:
            return None

        # Sort by score (descending)
        filtered.sort(key=lambda r: r.score, reverse=True)

        # Return best timeframe
        return filtered[0].timeframe

    def generate_routing_config(
        self,
        results: Optional[List[BacktestResult]] = None,
        min_sharpe: float = 0.5
    ) -> Dict[str, Any]:
        """
        Generate optimal routing config from backtest results.

        Args:
            results: Optional results (uses self.results if None)
            min_sharpe: Minimum Sharpe ratio to consider strategy valid

        Returns:
            Dictionary in strategy_routing.json format
        """
        results = results or self.results

        if not results:
            raise ValueError("No results available. Run optimization first.")

        self.logger.info("Generating optimal routing config...")

        config = {}

        for symbol in self.symbols:
            config[symbol] = {}

            # Find best strategy/timeframe for each regime
            # For now, we'll use the same regimes for all symbols
            # In the future, could detect regime-specific performance

            # Group results by strategy for this symbol
            symbol_results = [r for r in results if r.symbol == symbol]

            if not symbol_results:
                self.logger.warning(f"No results for {symbol}, skipping")
                continue

            # Sort by score
            symbol_results.sort(key=lambda r: r.score, reverse=True)

            # For each regime, assign strategies based on characteristics
            # This is a heuristic - could be improved with regime-specific backtesting

            # Low volatility: Best overall strategy with shorter timeframe
            low_vol_candidates = [r for r in symbol_results if r.timeframe in ['5min', '15min']]
            if low_vol_candidates:
                best_low_vol = low_vol_candidates[0]
                config[symbol]['low_volatility'] = {
                    'strategy': best_low_vol.strategy,
                    'timeframe': best_low_vol.timeframe
                }

            # Normal: Best overall strategy
            if symbol_results[0].sharpe_ratio >= min_sharpe:
                best_normal = symbol_results[0]
                config[symbol]['normal'] = {
                    'strategy': best_normal.strategy,
                    'timeframe': best_normal.timeframe
                }

                # Use as default too
                config[symbol]['default'] = {
                    'strategy': best_normal.strategy,
                    'timeframe': best_normal.timeframe
                }

            # High volatility: Best strategy with longer timeframe
            high_vol_candidates = [r for r in symbol_results if r.timeframe in ['30min', '1hour']]
            if high_vol_candidates:
                best_high_vol = high_vol_candidates[0]
                config[symbol]['high_volatility'] = {
                    'strategy': best_high_vol.strategy,
                    'timeframe': best_high_vol.timeframe
                }

            # Add use_hybrid flag (could be optimized separately)
            config[symbol]['use_hybrid'] = False

            self.logger.info(
                f"{symbol}: normal={config[symbol].get('normal', {}).get('timeframe', 'N/A')}, "
                f"low_vol={config[symbol].get('low_volatility', {}).get('timeframe', 'N/A')}, "
                f"high_vol={config[symbol].get('high_volatility', {}).get('timeframe', 'N/A')}"
            )

        # Add global default
        config['default'] = {
            'default': {
                'strategy': 'sma',
                'timeframe': '15min'
            },
            'use_hybrid': False
        }

        return config

    def save_config(self, config: Dict[str, Any], path: str):
        """
        Save routing config to JSON file.

        Args:
            config: Routing configuration dictionary
            path: File path to save to
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, 'w') as f:
            json.dump(config, f, indent=2)

        self.logger.info(f"Saved optimal config to {path}")

    def save_results(self, path: str):
        """
        Save detailed results to JSON file.

        Args:
            path: File path to save to
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        results_dict = [asdict(r) for r in self.results]

        with open(path_obj, 'w') as f:
            json.dump(results_dict, f, indent=2)

        self.logger.info(f"Saved detailed results to {path}")

    def print_summary(self):
        """Print summary of optimization results."""
        if not self.results:
            print("No results available")
            return

        print("\n" + "=" * 80)
        print("OPTIMIZATION RESULTS SUMMARY")
        print("=" * 80)

        # Group by symbol
        for symbol in self.symbols:
            symbol_results = [r for r in self.results if r.symbol == symbol]

            if not symbol_results:
                continue

            print(f"\n{symbol}:")
            print("-" * 80)

            # Sort by score
            symbol_results.sort(key=lambda r: r.score, reverse=True)

            # Show top 3
            for i, result in enumerate(symbol_results[:3], 1):
                print(
                    f"  {i}. {result.strategy:15s} @ {result.timeframe:6s} - "
                    f"Sharpe: {result.sharpe_ratio:5.2f}, Return: {result.total_return:6.2%}, "
                    f"Win Rate: {result.win_rate:5.2%}, Score: {result.score:5.2f}"
                )


# ============================================================================
# CLI INTERFACE
# ============================================================================

async def main():
    """Command-line interface for timeframe optimization."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Optimize timeframes for trading strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize for specific symbols
  python -m core.backtest.timeframe_optimizer --symbols AAPL TSLA MSFT

  # Test specific timeframes
  python -m core.backtest.timeframe_optimizer --timeframes 5min 15min 30min

  # Full optimization
  python -m core.backtest.timeframe_optimizer --symbols AAPL TSLA --timeframes 5min 15min 30min 1hour --strategies rsi sma meanreversion
        """
    )

    parser.add_argument(
        '--symbols', nargs='+',
        default=['AAPL', 'TSLA', 'MSFT'],
        help='Symbols to optimize'
    )
    parser.add_argument(
        '--timeframes', nargs='+',
        default=['5min', '15min', '30min'],
        help='Timeframes to test'
    )
    parser.add_argument(
        '--strategies', nargs='+',
        default=['rsi', 'sma', 'meanreversion', 'bollinger'],
        help='Strategies to test'
    )
    parser.add_argument(
        '--days', type=int, default=750,
        help='Days of historical data'
    )
    parser.add_argument(
        '--output', default='config/strategy_routing_optimized.json',
        help='Output config file'
    )
    parser.add_argument(
        '--results', default='results/optimization_results.json',
        help='Detailed results file'
    )

    args = parser.parse_args()

    # Create optimizer
    optimizer = TimeframeOptimizer(
        symbols=args.symbols,
        timeframes=args.timeframes,
        strategies=args.strategies
    )

    # Run optimization
    results = await optimizer.run_optimization(days=args.days)

    # Print summary
    optimizer.print_summary()

    # Generate and save config
    config = optimizer.generate_routing_config(results)
    optimizer.save_config(config, args.output)
    optimizer.save_results(args.results)

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 80)
    print(f"Optimal config saved to: {args.output}")
    print(f"Detailed results saved to: {args.results}")
    print("\nNext steps:")
    print(f"  1. Review the results in {args.results}")
    print(f"  2. Copy {args.output} to config/strategy_routing.json")
    print("  3. Restart trader with new configuration")
    print("=" * 80)


if __name__ == '__main__':
    asyncio.run(main())
