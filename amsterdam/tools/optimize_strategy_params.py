#!/usr/bin/env python
"""
Strategy Parameter Optimizer

Optimizes strategy-specific parameters for each symbol/regime combination.
Reads the current strategy assignments from strategy_routing.json and finds
the best parameters for each strategy using grid search.

Usage:
    python tools/optimize_strategy_params.py --save
    python tools/optimize_strategy_params.py --symbols AAPL TSLA --save
    python tools/optimize_strategy_params.py --dry-run  # Preview only
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.unified_data_pipeline import UnifiedDataPipeline
from core.backtest.regime_backtest import RegimeBacktester, REGIME_TYPES
from core.backtest.optimization import grid_search
from loggers.logger import Logger

# Parameter grids for each strategy (from strategy_selector.py)
STRATEGY_PARAM_GRIDS = {
    "sma": {
        "fast": [5, 10, 15, 20],
        "slow": [20, 30, 50, 100]
    },
    "ema": {
        "fast": [5, 10, 12, 15],
        "slow": [20, 26, 30, 50]
    },
    "momentum": {
        "lookback": [5, 10, 20, 30, 60]
    },
    "meanreversion": {
        "window": [10, 14, 20, 30],
        "threshold": [1.0, 1.5, 2.0, 2.5]
    },
    "rsi": {
        "window": [7, 14, 21],
        "oversold": [20, 25, 30],
        "overbought": [70, 75, 80]
    },
    "bollinger": {
        "window": [10, 20, 30],
        "num_std": [1.5, 2.0, 2.5]
    },
    "macd": {
        "fast": [8, 12, 16],
        "slow": [20, 26, 30],
        "signal": [7, 9, 12]
    },
    "adx": {
        "window": [10, 14, 20],
        "threshold": [20, 25, 30]
    },
    "stochastic": {
        "k_window": [10, 14, 21],
        "d_window": [3, 5, 7],
        "oversold": [15, 20, 25],
        "overbought": [75, 80, 85]
    },
    "donchian": {
        "window": [10, 20, 30, 55]
    },
    "breakout": {
        "window": [10, 20, 30],
        "threshold": [0.01, 0.02, 0.03]
    },
    "vwap": {
        "window": [10, 20, 30]
    },
    "psar": {
        "af_start": [0.01, 0.02, 0.03],
        "af_max": [0.1, 0.2, 0.3]
    },
    "ichimoku": {
        "tenkan": [7, 9, 12],
        "kijun": [22, 26, 30]
    },
}


@dataclass
class ParamOptResult:
    """Result of parameter optimization for one strategy/regime combo."""
    symbol: str
    regime: str
    strategy: str
    best_params: Dict[str, Any]
    sharpe_ratio: float
    total_return: float
    win_rate: float
    num_trades: int


class StrategyParamOptimizer:
    """
    Optimizes strategy parameters for each symbol/regime combination.

    Reads current strategy assignments from strategy_routing.json and
    finds the best parameters for each strategy via grid search.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        data_pipeline: Optional[UnifiedDataPipeline] = None
    ):
        """
        Initialize parameter optimizer.

        Args:
            symbols: List of symbols to optimize (None = all from routing)
            data_pipeline: Optional UnifiedDataPipeline instance
        """
        self.pipeline = data_pipeline or UnifiedDataPipeline()

        self.logger = Logger(
            "param_optimizer.log",
            "ParamOptimizer",
            propagate=True,
            level=10
        ).get_logger()

        # Load current strategy routing
        self.routing_path = ROOT / "config" / "strategy_routing.json"
        self.routing = self._load_routing()

        # Determine symbols to optimize
        if symbols:
            self.symbols = symbols
        else:
            # Get all symbols from routing (excluding 'default')
            self.symbols = [s for s in self.routing.keys() if s != 'default']

        self.results: Dict[str, Dict[str, ParamOptResult]] = {}

        self.logger.info(
            f"StrategyParamOptimizer initialized for {len(self.symbols)} symbols"
        )

    def _load_routing(self) -> Dict:
        """Load strategy routing configuration."""
        if not self.routing_path.exists():
            raise FileNotFoundError(
                f"Strategy routing not found: {self.routing_path}\n"
                "Run strategy optimization first: python tools/optimize_all_symbols.py --save"
            )

        with open(self.routing_path) as f:
            return json.load(f)

    async def optimize_all(
        self,
        days: int = 365,
        metric: str = "sharpe_ratio"
    ) -> Dict[str, Dict[str, ParamOptResult]]:
        """
        Optimize parameters for all symbols/regimes.

        Args:
            days: Days of historical data to use
            metric: Metric for optimization (sharpe_ratio, total_return, win_rate)

        Returns:
            Dict mapping symbol -> regime -> ParamOptResult
        """
        self.logger.info("=" * 80)
        self.logger.info("STRATEGY PARAMETER OPTIMIZATION")
        self.logger.info("=" * 80)
        self.logger.info(f"Symbols: {len(self.symbols)}")
        self.logger.info(f"Days: {days}")
        self.logger.info(f"Metric: {metric}")
        self.logger.info("=" * 80)

        for i, symbol in enumerate(self.symbols, 1):
            self.logger.info(f"\n[{i}/{len(self.symbols)}] Optimizing {symbol}...")

            try:
                symbol_results = await self._optimize_symbol(symbol, days, metric)
                self.results[symbol] = symbol_results

                # Print summary for this symbol
                print(f"\n{symbol}:")
                for regime, result in symbol_results.items():
                    print(
                        f"  {regime:<18}: {result.strategy:<15} "
                        f"params={result.best_params} "
                        f"(Sharpe: {result.sharpe_ratio:.2f})"
                    )

            except Exception as e:
                self.logger.exception(f"Failed to optimize {symbol}: {e}")

        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"OPTIMIZATION COMPLETE: {len(self.results)}/{len(self.symbols)} symbols")
        self.logger.info("=" * 80)

        return self.results

    async def _optimize_symbol(
        self,
        symbol: str,
        days: int,
        metric: str
    ) -> Dict[str, ParamOptResult]:
        """
        Optimize parameters for a single symbol across all regimes.

        Returns:
            Dict mapping regime -> ParamOptResult
        """
        # Load historical data
        data = self.pipeline.get_data(symbol, timeframe='day')

        if data is None or data.empty:
            self.logger.warning(f"No data for {symbol}, skipping...")
            return {}

        if len(data) > days:
            data = data.tail(days).reset_index(drop=True)

        # Get strategy assignments for this symbol
        if symbol not in self.routing:
            self.logger.warning(f"No routing config for {symbol}, skipping...")
            return {}

        symbol_config = self.routing[symbol]
        results = {}

        # Optimize parameters for each regime
        for regime in REGIME_TYPES:
            if regime not in symbol_config:
                continue

            regime_config = symbol_config[regime]

            # Handle both old format (string) and new format (dict)
            if isinstance(regime_config, str):
                strategy_name = regime_config
            elif isinstance(regime_config, dict):
                strategy_name = regime_config.get('strategy')
            else:
                self.logger.warning(f"Invalid config for {symbol}/{regime}")
                continue

            if not strategy_name:
                continue

            self.logger.info(
                f"  Optimizing {regime}: {strategy_name}"
            )

            try:
                result = await self._optimize_strategy_regime(
                    symbol=symbol,
                    regime=regime,
                    strategy_name=strategy_name,
                    data=data,
                    metric=metric
                )

                if result:
                    results[regime] = result

            except Exception as e:
                self.logger.warning(
                    f"  Failed to optimize {strategy_name} for {regime}: {e}"
                )

        return results

    async def _optimize_strategy_regime(
        self,
        symbol: str,
        regime: str,
        strategy_name: str,
        data,
        metric: str
    ) -> Optional[ParamOptResult]:
        """
        Optimize parameters for a specific strategy/regime combination.

        Uses grid search with regime-aware backtesting.
        """
        # Get parameter grid for this strategy
        param_grid = STRATEGY_PARAM_GRIDS.get(strategy_name, {})

        if not param_grid:
            self.logger.debug(
                f"    No parameter grid for {strategy_name}, using defaults"
            )
            # Return result with empty params
            return ParamOptResult(
                symbol=symbol,
                regime=regime,
                strategy=strategy_name,
                best_params={},
                sharpe_ratio=0.0,
                total_return=0.0,
                win_rate=0.0,
                num_trades=0
            )

        # Create regime backtester
        backtester = RegimeBacktester(
            data=data,
            symbol=symbol,
            initial_capital=100000,
            transaction_cost=0.001,
            strategies=[strategy_name]
        )

        # Run grid search for this regime
        # We need to filter the backtest to only this regime
        best_params = {}
        best_score = float('-inf')
        best_metrics = {}

        from itertools import product

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        total_combos = 1
        for values in param_values:
            total_combos *= len(values)

        self.logger.debug(
            f"    Testing {total_combos} parameter combinations..."
        )

        for combo in product(*param_values):
            params = dict(zip(param_names, combo))

            # Run regime-specific backtest
            try:
                # Generate signals with these params
                from strategies.strategy_registry import load_strategy
                strategy = load_strategy(strategy_name, params=params)

                signals = []
                for i in range(len(data)):
                    try:
                        sig = strategy.generate_signal(data.iloc[:i+1])
                        signals.append(int(sig) if sig in (-1, 0, 1) else 0)
                    except:
                        signals.append(0)

                # Backtest only in this regime
                import numpy as np
                signals = np.array(signals)
                result = backtester._backtest_regime_segment(
                    strategy_name=strategy_name,
                    regime=regime,
                    signals=signals
                )

                # Score this combination
                score = getattr(result, metric, 0)

                if score > best_score:
                    best_score = score
                    best_params = params
                    best_metrics = {
                        'sharpe_ratio': result.sharpe_ratio,
                        'total_return': result.total_return,
                        'win_rate': result.win_rate,
                        'num_trades': result.num_trades
                    }

            except Exception as e:
                self.logger.debug(f"      Failed with params {params}: {e}")
                continue

        if not best_params:
            self.logger.warning(f"    No valid parameters found for {strategy_name}/{regime}")
            return None

        self.logger.info(
            f"    Best params: {best_params} "
            f"(Sharpe: {best_metrics.get('sharpe_ratio', 0):.2f})"
        )

        return ParamOptResult(
            symbol=symbol,
            regime=regime,
            strategy=strategy_name,
            best_params=best_params,
            sharpe_ratio=best_metrics.get('sharpe_ratio', 0),
            total_return=best_metrics.get('total_return', 0),
            win_rate=best_metrics.get('win_rate', 0),
            num_trades=best_metrics.get('num_trades', 0)
        )

    def save_params(self, output_path: Optional[str] = None) -> Path:
        """
        Save optimized parameters to strategy_params.json.

        Args:
            output_path: Custom output path (default: config/strategy_params.json)

        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = ROOT / "config" / "strategy_params.json"
        else:
            output_path = Path(output_path)

        # Build params structure
        params_config = {}

        for symbol, regimes in self.results.items():
            params_config[symbol] = {}

            for regime, result in regimes.items():
                params_config[symbol][regime] = {
                    "params": result.best_params,
                    "strategy": result.strategy,  # For reference
                    "_optimized_sharpe": result.sharpe_ratio,  # For reference
                    "_optimized_return": result.total_return   # For reference
                }

        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(params_config, f, indent=2)

        self.logger.info(f"Saved optimized parameters to {output_path}")
        return output_path


async def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Optimize strategy parameters for each symbol/regime',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize all symbols from strategy_routing.json
  python tools/optimize_strategy_params.py --save

  # Optimize specific symbols
  python tools/optimize_strategy_params.py --symbols AAPL TSLA NVDA --save

  # Preview without saving
  python tools/optimize_strategy_params.py --dry-run

  # Use more historical data
  python tools/optimize_strategy_params.py --days 1000 --save
        """
    )

    parser.add_argument(
        '--symbols', nargs='+',
        help='Symbols to optimize (default: all from strategy_routing.json)'
    )
    parser.add_argument(
        '--days', type=int, default=365,
        help='Days of historical data (default: 365)'
    )
    parser.add_argument(
        '--metric', default='sharpe_ratio',
        choices=['sharpe_ratio', 'total_return', 'win_rate'],
        help='Optimization metric (default: sharpe_ratio)'
    )
    parser.add_argument(
        '--save', action='store_true',
        help='Save to config/strategy_params.json'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Preview without saving'
    )
    parser.add_argument(
        '--output',
        help='Custom output path'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("STRATEGY PARAMETER OPTIMIZATION")
    print("=" * 80)

    # Create optimizer
    optimizer = StrategyParamOptimizer(symbols=args.symbols)

    if args.dry_run:
        print(f"\n🔍 DRY RUN - Would optimize parameters for:")
        for symbol in optimizer.symbols:
            print(f"  - {symbol}")
        print(f"\nDays: {args.days}")
        print(f"Metric: {args.metric}")
        return 0

    # Run optimization
    print(f"\nOptimizing {len(optimizer.symbols)} symbols...")
    print(f"This will test many parameter combinations - may take 30-60 minutes\n")

    results = await optimizer.optimize_all(days=args.days, metric=args.metric)

    # Save if requested
    if args.save:
        output_path = optimizer.save_params(output_path=args.output)
        print("\n" + "=" * 80)
        print(f"✅ SAVED TO: {output_path}")
        print("=" * 80)
        print("\nOptimization complete! Next steps:")
        print("  1. Review the parameters:")
        print(f"     cat {output_path}")
        print("  2. Parameters will be used automatically by the system")
        print("  3. Restart trader to use new parameters:")
        print("     amsterdam start")
    else:
        print("\n⚠️  Results not saved (use --save to save)")

    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
