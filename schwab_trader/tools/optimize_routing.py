#!/usr/bin/env python3
"""
Optimize Strategy Routing - Backtest all symbols and update strategy_routing.json

This tool runs regime-aware backtests on all available symbols and generates
the optimal strategy routing configuration for your live trading system.

Features:
- Tests all strategies across all market regimes
- Determines optimal strategy per (symbol, regime) combination
- Automatically sets hybrid sizing based on strategy type:
  - Trend-following strategies: hybrid sizing ENABLED
  - Mean-reversion strategies: hybrid sizing DISABLED

Usage:
    # Optimize all symbols
    python tools/optimize_routing.py

    # Specific symbols only
    python tools/optimize_routing.py -s AAPL,MSFT,GOOGL

    # Custom strategies and days
    python tools/optimize_routing.py -d 180 --strategies sma,ema,macd,rsi

    # Don't save (dry run)
    python tools/optimize_routing.py --dry-run
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter

# Ensure project root is in path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


DEFAULT_STRATEGIES = [
    "sma", "ema", "macd", "rsi", "bollinger",
    "momentum", "meanreversion", "stochastic"
]

# Strategy categories for hybrid sizing decision
TREND_FOLLOWING_STRATEGIES = {
    "sma", "ema", "macd", "momentum", "adx",
    "ichimoku", "psar", "donchian", "breakout"
}

MEAN_REVERSION_STRATEGIES = {
    "rsi", "bollinger", "stochastic", "meanreversion", "vwap"
}


def should_use_hybrid(strategy: str) -> bool:
    """
    Determine if hybrid sizing should be used for a strategy.

    Hybrid sizing benefits trend-following strategies (trades WITH the trend).
    Mean-reversion strategies trade AGAINST the trend, so hybrid hurts them.
    """
    strategy_lower = strategy.lower()

    if strategy_lower in TREND_FOLLOWING_STRATEGIES:
        return True
    elif strategy_lower in MEAN_REVERSION_STRATEGIES:
        return False
    else:
        # Default to True for unknown strategies
        return True


def determine_hybrid_config(routing: dict) -> dict:
    """
    Analyze routing config and determine hybrid sizing for each symbol.

    Logic:
    - Count trend-following vs mean-reversion strategies per symbol
    - If majority are trend-following, enable hybrid
    - Otherwise disable
    """
    hybrid_config = {}

    for symbol, regimes in routing.items():
        if symbol == "default":
            continue

        # Count strategy types across regimes
        trend_count = 0
        mr_count = 0

        for regime in ["low_volatility", "normal", "high_volatility"]:
            strategy = regimes.get(regime, "")
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


def main():
    parser = argparse.ArgumentParser(
        description="Optimize strategy routing based on regime backtests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Optimize all available symbols
    python tools/optimize_routing.py

    # Specific symbols
    python tools/optimize_routing.py -s AAPL,MSFT,GOOGL,TSLA

    # Use 180 days of data
    python tools/optimize_routing.py -d 180

    # Custom strategies
    python tools/optimize_routing.py --strategies sma,ema,macd,rsi,bollinger

    # Dry run (don't save)
    python tools/optimize_routing.py --dry-run
        """,
    )

    parser.add_argument(
        "-s", "--symbols",
        default=None,
        help="Comma-separated symbols (default: all available)",
    )
    parser.add_argument(
        "-d", "--days",
        type=int,
        default=365,
        help="Days of historical data (default: 365)",
    )
    parser.add_argument(
        "--strategies",
        default=None,
        help="Comma-separated strategies to test",
    )
    parser.add_argument(
        "-m", "--metric",
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
        "-o", "--output",
        default=None,
        help="Output file (default: config/strategy_routing.json)",
    )

    args = parser.parse_args()

    # Import here to avoid slow startup
    from core.unified_data_pipeline import UnifiedDataPipeline
    from core.backtest.regime_backtest import RegimeBacktester, REGIME_TYPES

    pipeline = UnifiedDataPipeline()

    # Get symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = pipeline.list_available_symbols()

    # Get strategies
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",")]
    else:
        strategies = DEFAULT_STRATEGIES

    print()
    print("=" * 80)
    print("  STRATEGY ROUTING OPTIMIZATION")
    print("=" * 80)
    print(f"  Symbols: {len(symbols)}")
    print(f"  Strategies: {', '.join(strategies)}")
    print(f"  Days: {args.days}")
    print(f"  Metric: {args.metric}")
    print(f"  Dry Run: {args.dry_run}")
    print("=" * 80)

    combined_routing = {}
    all_results = {}
    failed = []

    for i, symbol in enumerate(symbols):
        print(f"\n[{i+1}/{len(symbols)}] {symbol}...", end=" ")

        try:
            data = pipeline.get_data(symbol)
            if data is None or data.empty:
                print("NO DATA")
                failed.append(symbol)
                continue

            if len(data) > args.days:
                data = data.tail(args.days).reset_index(drop=True)

            tester = RegimeBacktester(
                data=data,
                symbol=symbol,
                strategies=strategies,
            )

            result = tester.run_regime_analysis(metric=args.metric, verbose=False)
            all_results[symbol] = result
            combined_routing.update(result.routing_config)

            # Print summary
            best = result.best_strategies
            print(f"low_vol={best.get('low_volatility', '?')}, "
                  f"normal={best.get('normal', '?')}, "
                  f"high_vol={best.get('high_volatility', '?')}")

        except Exception as e:
            print(f"ERROR: {e}")
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
        }

    # Determine hybrid sizing for each symbol
    hybrid_config = determine_hybrid_config(combined_routing)

    # Add use_hybrid to routing config
    for symbol in combined_routing:
        if symbol != "default" and symbol in hybrid_config:
            combined_routing[symbol]["use_hybrid"] = hybrid_config[symbol]["enabled"]

    # Default hybrid setting
    combined_routing["default"]["use_hybrid"] = True

    # Print summary
    print()
    print("=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Symbol':<8} {'Low Volatility':<15} {'Normal':<15} {'High Volatility':<15} {'Hybrid':<8}")
    print("-" * 70)

    for symbol in sorted(all_results.keys()):
        r = all_results[symbol].best_strategies
        use_hyb = "YES" if hybrid_config.get(symbol, {}).get("enabled", False) else "NO"
        print(f"{symbol:<8} {r.get('low_volatility', '?'):<15} "
              f"{r.get('normal', '?'):<15} {r.get('high_volatility', '?'):<15} {use_hyb:<8}")

    print("-" * 70)

    # Strategy frequency
    print()
    print("STRATEGY FREQUENCY BY REGIME:")
    for regime in REGIME_TYPES:
        strats = [all_results[s].best_strategies.get(regime, "unknown") for s in all_results]
        counts = Counter(strats)
        top3 = counts.most_common(3)
        top_str = ", ".join(f"{s}({c})" for s, c in top3)
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
        print(f"Config saved to: {config_path}")
        print(f"Total symbols configured: {len(combined_routing) - 1}")  # -1 for default

    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
