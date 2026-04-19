#!/usr/bin/env python3
"""
Optimize Strategy Routing with Multi-Timeframe Support

Extension of optimize_routing.py that adds timeframe optimization.
Tests all combinations of: symbol × regime × strategy × timeframe

Usage:
    # Optimize with multiple timeframes
    python tools/optimize_routing_multitf.py --timeframes 5min,15min,30min

    # Specific symbols with timeframes
    python tools/optimize_routing_multitf.py -s AAPL,TSLA --timeframes 5min,15min,30min,1hour

    # Don't save (dry run)
    python tools/optimize_routing_multitf.py --dry-run
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Ensure project root is in path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from core.backtest.regime_backtest import REGIME_TYPES, RegimeBacktester
from core.unified_data_pipeline import UnifiedDataPipeline

DEFAULT_STRATEGIES = ["sma", "ema", "macd", "rsi", "bollinger", "momentum", "meanreversion", "stochastic"]

DEFAULT_TIMEFRAMES = ["5min", "15min", "30min"]

# Strategy categories for hybrid sizing decision
TREND_FOLLOWING_STRATEGIES = {"sma", "ema", "macd", "momentum", "adx", "ichimoku", "psar", "donchian", "breakout"}

MEAN_REVERSION_STRATEGIES = {"rsi", "bollinger", "stochastic", "meanreversion", "vwap"}


def should_use_hybrid(strategy: str) -> bool:
    """Determine if hybrid sizing should be used for a strategy."""
    strategy_lower = strategy.lower()
    if strategy_lower in TREND_FOLLOWING_STRATEGIES:
        return True
    elif strategy_lower in MEAN_REVERSION_STRATEGIES:
        return False
    else:
        return True


def determine_hybrid_config(routing: dict) -> dict:
    """Analyze routing config and determine hybrid sizing for each symbol."""
    hybrid_config = {}

    for symbol, regimes in routing.items():
        if symbol == "default":
            continue

        # Count strategy types across regimes
        trend_count = 0
        mr_count = 0

        for regime in ["low_volatility", "normal", "high_volatility"]:
            regime_data = regimes.get(regime, {})
            if isinstance(regime_data, str):
                strategy = regime_data
            elif isinstance(regime_data, dict):
                strategy = regime_data.get("strategy", "")
            else:
                continue

            if strategy.lower() in TREND_FOLLOWING_STRATEGIES:
                trend_count += 1
            elif strategy.lower() in MEAN_REVERSION_STRATEGIES:
                mr_count += 1

        # Enable hybrid if trend-following is dominant
        use_hybrid = trend_count > mr_count

        hybrid_config[symbol] = {
            "enabled": use_hybrid,
            "trend_following_count": trend_count,
            "mean_reversion_count": mr_count,
        }

    return hybrid_config


def optimize_symbol_with_timeframes(
    symbol: str,
    pipeline: UnifiedDataPipeline,
    strategies: list[str],
    timeframes: list[str],
    days: int,
    metric: str,
    verbose: bool = True,
) -> dict[str, Any] | None:
    """
    Optimize a symbol across multiple timeframes.

    Tests all combinations of: regime × strategy × timeframe
    Returns optimal configuration with timeframe per regime.
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  Optimizing {symbol}")
        print(f"  Timeframes: {', '.join(timeframes)}")
        print(f"{'=' * 60}")

    # Results storage: timeframe -> regime -> strategy -> metrics
    all_results: dict[str, dict[str, dict[str, Any]]] = {}

    # Test each timeframe
    for timeframe in timeframes:
        if verbose:
            print(f"\n  Testing {timeframe}...")

        try:
            # Load data at this timeframe
            data = pipeline.get_data(symbol, timeframe=timeframe)

            if data is None or data.empty:
                if verbose:
                    print("    ✗ No data available")
                continue

            # Limit to recent days
            if len(data) > days:
                data = data.tail(days).reset_index(drop=True)

            # Run regime backtest
            tester = RegimeBacktester(
                data=data,
                symbol=symbol,
                strategies=strategies,
            )

            result = tester.run_regime_analysis(metric=metric, verbose=False)

            # Store results for this timeframe
            all_results[timeframe] = {}

            for regime in REGIME_TYPES:
                all_results[timeframe][regime] = {}

                for strategy_name, strategy_result in result.strategy_results[regime].items():
                    all_results[timeframe][regime][strategy_name] = {
                        "sharpe_ratio": strategy_result.sharpe_ratio,
                        "total_return": strategy_result.total_return,
                        "win_rate": strategy_result.win_rate,
                        "num_trades": strategy_result.num_trades,
                        "max_drawdown": strategy_result.max_drawdown,
                        "profit_factor": strategy_result.profit_factor,
                    }

            if verbose:
                print(f"    ✓ Backtests complete ({len(result.strategy_results['normal'])} strategies)")

        except Exception as e:
            if verbose:
                print(f"    ✗ Error: {e}")
            continue

    if not all_results:
        if verbose:
            print(f"  ✗ No valid results for {symbol}")
        return None

    # Find best (strategy, timeframe) for each regime
    best_config = {}

    for regime in REGIME_TYPES:
        # Collect all (timeframe, strategy) combinations for this regime
        candidates = []

        for timeframe, timeframe_results in all_results.items():
            if regime not in timeframe_results:
                continue

            for strategy, metrics in timeframe_results[regime].items():
                # Get metric value
                metric_value = metrics.get(metric, 0)

                candidates.append(
                    {"timeframe": timeframe, "strategy": strategy, "metric_value": metric_value, "metrics": metrics}
                )

        if not candidates:
            continue

        # Find best combination
        best = max(candidates, key=lambda x: x["metric_value"])

        best_config[regime] = {"strategy": best["strategy"], "timeframe": best["timeframe"]}

        if verbose:
            print(
                f"  {regime:<18}: {best['strategy']:<15} @ {best['timeframe']:<6} ({metric}={best['metric_value']:.2f})"
            )

    # Add default (use normal regime as default)
    if "normal" in best_config:
        best_config["default"] = best_config["normal"].copy()

    return {symbol: best_config} if best_config else None


def main():
    parser = argparse.ArgumentParser(
        description="Optimize strategy routing with multi-timeframe support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Optimize with multiple timeframes
    python tools/optimize_routing_multitf.py --timeframes 5min,15min,30min

    # Specific symbols with timeframes
    python tools/optimize_routing_multitf.py -s AAPL,TSLA,MSFT --timeframes 5min,15min,30min,1hour

    # Use different metric
    python tools/optimize_routing_multitf.py --metric total_return --timeframes 15min,30min

    # Dry run (don't save)
    python tools/optimize_routing_multitf.py --dry-run
        """,
    )

    parser.add_argument(
        "-s",
        "--symbols",
        default=None,
        help="Comma-separated symbols (default: all available)",
    )
    parser.add_argument(
        "-d",
        "--days",
        type=int,
        default=750,
        help="Days of historical data (default: 750 = ~2 years)",
    )
    parser.add_argument(
        "--timeframes",
        default=None,
        help="Comma-separated timeframes to test (default: 5min,15min,30min)",
    )
    parser.add_argument(
        "--strategies",
        default=None,
        help="Comma-separated strategies to test",
    )
    parser.add_argument(
        "-m",
        "--metric",
        default="sharpe_ratio",
        choices=["sharpe_ratio", "total_return", "win_rate"],
        help="Metric for ranking (default: sharpe_ratio)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save config (just show results)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file (default: config/strategy_routing.json)",
    )
    parser.add_argument(
        "--fetch-data",
        action="store_true",
        help="Fetch historical data before optimizing",
    )

    args = parser.parse_args()

    pipeline = UnifiedDataPipeline()

    # Get symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = pipeline.list_available_symbols()

        if not symbols:
            routing_path = ROOT / "config" / "strategy_routing.json"
            if routing_path.exists():
                with open(routing_path) as f:
                    existing_routing = json.load(f)
                symbols = [s for s in existing_routing.keys() if s != "default"]
                if symbols:
                    print(f"  (Loaded {len(symbols)} symbols from existing routing config)")

        if not symbols:
            print("\nERROR: No symbols found. Please specify symbols with -s flag.")
            print("Example: python tools/optimize_routing_multitf.py -s AAPL,MSFT,GOOGL")
            sys.exit(1)

    # Get timeframes
    if args.timeframes:
        timeframes = [tf.strip() for tf in args.timeframes.split(",")]
    else:
        timeframes = DEFAULT_TIMEFRAMES

    # Get strategies
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",")]
    else:
        strategies = DEFAULT_STRATEGIES

    print()
    print("=" * 80)
    print("  MULTI-TIMEFRAME STRATEGY ROUTING OPTIMIZATION")
    print("=" * 80)
    print(f"  Symbols: {len(symbols)}")
    print(f"  Timeframes: {', '.join(timeframes)}")
    print(f"  Strategies: {', '.join(strategies)}")
    print(f"  Days: {args.days}")
    print(f"  Metric: {args.metric}")
    print(f"  Dry Run: {args.dry_run}")
    print("=" * 80)

    # Fetch data if requested
    if args.fetch_data:
        print("\nFetching historical data...")
        import asyncio

        async def fetch():
            await pipeline.update_symbols(symbols=symbols, timeframes=timeframes, days=args.days)

        asyncio.run(fetch())
        print("✓ Data fetching complete")

    # Optimize each symbol
    combined_routing = {}
    all_results = {}
    failed = []

    for i, symbol in enumerate(symbols):
        result = optimize_symbol_with_timeframes(
            symbol=symbol,
            pipeline=pipeline,
            strategies=strategies,
            timeframes=timeframes,
            days=args.days,
            metric=args.metric,
            verbose=True,
        )

        if result:
            all_results[symbol] = result[symbol]
            combined_routing.update(result)
        else:
            failed.append(symbol)

    # Add default routing based on most common (strategy, timeframe) combinations
    if all_results:
        default_config = {}

        for regime in REGIME_TYPES:
            # Collect (strategy, timeframe) pairs for this regime
            regime_configs = []

            for symbol_config in all_results.values():
                if regime in symbol_config:
                    regime_configs.append((symbol_config[regime]["strategy"], symbol_config[regime]["timeframe"]))

            if regime_configs:
                # Most common (strategy, timeframe) combination
                most_common = Counter(regime_configs).most_common(1)[0][0]
                default_config[regime] = {"strategy": most_common[0], "timeframe": most_common[1]}

        # Overall default
        if "normal" in default_config:
            default_config["default"] = default_config["normal"].copy()
        else:
            default_config["default"] = {"strategy": "momentum", "timeframe": timeframes[0]}

        combined_routing["default"] = default_config
    else:
        # No results - create basic default
        combined_routing["default"] = {
            "low_volatility": {"strategy": "sma", "timeframe": timeframes[0]},
            "normal": {"strategy": "bollinger", "timeframe": timeframes[0]},
            "high_volatility": {"strategy": "rsi", "timeframe": timeframes[0]},
            "default": {"strategy": "momentum", "timeframe": timeframes[0]},
        }

    # Determine hybrid sizing for each symbol
    hybrid_config = determine_hybrid_config(combined_routing)

    # Add use_hybrid to routing config
    for symbol in combined_routing:
        if symbol != "default" and symbol in hybrid_config:
            combined_routing[symbol]["use_hybrid"] = hybrid_config[symbol]["enabled"]

    # Print summary
    print()
    print("=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Symbol':<8} {'Low Volatility':<20} {'Normal':<20} {'High Volatility':<20} {'Hybrid':<8}")
    print("-" * 80)

    for symbol in sorted(all_results.keys()):
        config = all_results[symbol]

        def format_regime(regime_data):
            if isinstance(regime_data, dict):
                return f"{regime_data.get('strategy', '?'):<10}@{regime_data.get('timeframe', '?')}"
            return str(regime_data)

        low_vol = format_regime(config.get("low_volatility", "?"))
        normal = format_regime(config.get("normal", "?"))
        high_vol = format_regime(config.get("high_volatility", "?"))
        use_hyb = "YES" if hybrid_config.get(symbol, {}).get("enabled", False) else "NO"

        print(f"{symbol:<8} {low_vol:<20} {normal:<20} {high_vol:<20} {use_hyb:<8}")

    print("-" * 80)

    # Strategy+Timeframe frequency
    print()
    print("STRATEGY+TIMEFRAME FREQUENCY BY REGIME:")
    for regime in REGIME_TYPES:
        configs = []
        for symbol_config in all_results.values():
            if regime in symbol_config:
                cfg = symbol_config[regime]
                configs.append(f"{cfg['strategy']}@{cfg['timeframe']}")

        if configs:
            counts = Counter(configs)
            top3 = counts.most_common(3)
            top_str = ", ".join(f"{c}({n})" for c, n in top3)
            print(f"  {regime:<18}: {top_str}")

    # Hybrid summary
    print()
    print("HYBRID SIZING SUMMARY:")
    hybrid_enabled = sum(1 for h in hybrid_config.values() if h["enabled"])
    hybrid_disabled = len(hybrid_config) - hybrid_enabled
    print(f"  Enabled (trend-following dominant):  {hybrid_enabled} symbols")
    print(f"  Disabled (mean-reversion dominant):  {hybrid_disabled} symbols")

    if failed:
        print()
        print(f"FAILED SYMBOLS ({len(failed)}): {', '.join(failed)}")

    # Save config
    if not args.dry_run and combined_routing:
        config_path = Path(args.output) if args.output else ROOT / "config" / "strategy_routing.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(combined_routing, f, indent=2)

        print()
        print(f"✓ Config saved to: {config_path}")
        print(f"  Total symbols configured: {len(combined_routing) - 1}")  # -1 for default

    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
