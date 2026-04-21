#!/usr/bin/env python3
"""
Trading Speed Benchmark

Measures the performance limits of your trading system:
1. Quote processing throughput
2. Bar aggregation speed
3. Strategy signal generation time
4. Full pipeline latency (quote → signal)

Usage:
    python tools/benchmark_trading_speed.py
"""

import asyncio
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


def benchmark_quote_processing(n_quotes: int = 10000) -> dict:
    """Benchmark raw quote processing speed."""
    from core.alpaca_schwab_hybrid_runner import AlpacaSchwabHybridRunner

    # Simulate quote data
    quotes = []
    base_time = datetime.now(timezone.utc)
    for i in range(n_quotes):
        quotes.append({
            "1": 150.00 + (i % 100) * 0.01,  # bid
            "2": 150.05 + (i % 100) * 0.01,  # ask
            "3": 150.02 + (i % 100) * 0.01,  # last
            "8": 1000 + i,  # volume
            "35": int(base_time.timestamp() * 1000) + i,  # trade time
        })

    # Benchmark canonicalization
    start = time.perf_counter()
    for quote in quotes:
        AlpacaSchwabHybridRunner._canonicalize_schwab_quote(quote, "TEST")
    elapsed = time.perf_counter() - start

    return {
        "operation": "Quote canonicalization",
        "n_ops": n_quotes,
        "total_time_ms": elapsed * 1000,
        "ops_per_sec": n_quotes / elapsed,
        "us_per_op": (elapsed / n_quotes) * 1_000_000,
    }


def benchmark_bar_aggregation(n_quotes: int = 10000) -> dict:
    """Benchmark bar aggregation speed."""
    from collections import defaultdict

    # Simplified aggregation logic (mirrors the runner)
    bar_agg = defaultdict(lambda: {
        "open": None, "high": None, "low": None,
        "close": None, "volume": 0, "bar_id": None
    })

    def aggregate(symbol, price, volume, bar_id):
        agg = bar_agg[symbol]
        if agg["bar_id"] != bar_id:
            agg["open"] = price
            agg["high"] = price
            agg["low"] = price
            agg["bar_id"] = bar_id
        agg["high"] = max(agg["high"], price)
        agg["low"] = min(agg["low"], price)
        agg["close"] = price
        agg["volume"] += volume
        return agg

    # Benchmark
    start = time.perf_counter()
    for i in range(n_quotes):
        price = 150.00 + (i % 100) * 0.01
        bar_id = i // 100  # New bar every 100 quotes
        aggregate("TEST", price, 100, bar_id)
    elapsed = time.perf_counter() - start

    return {
        "operation": "Bar aggregation",
        "n_ops": n_quotes,
        "total_time_ms": elapsed * 1000,
        "ops_per_sec": n_quotes / elapsed,
        "us_per_op": (elapsed / n_quotes) * 1_000_000,
    }


def benchmark_strategy_signal(n_signals: int = 1000) -> dict:
    """Benchmark strategy signal generation."""
    import numpy as np
    import pandas as pd
    from strategies.strategy_registry.strategy_registry import STRATEGY_CLASS_REGISTRY

    def get_strategy(name):
        cls = STRATEGY_CLASS_REGISTRY.get(name)
        if cls is None:
            raise ValueError(f"Strategy '{name}' not found")
        return cls()

    # Create realistic price data (300 bars of history)
    n_bars = 300
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq="1min")
    close = 150 + np.cumsum(np.random.randn(n_bars) * 0.5)
    high = close + np.abs(np.random.randn(n_bars) * 0.3)
    low = close - np.abs(np.random.randn(n_bars) * 0.3)
    open_ = close + np.random.randn(n_bars) * 0.2
    volume = np.random.randint(10000, 100000, n_bars)

    df = pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)

    # Get a few strategies to test
    strategies_to_test = ["sma", "rsi", "macd", "momentum"]
    results = {}

    for strat_name in strategies_to_test:
        try:
            strategy = get_strategy(strat_name)

            # Warm up
            for _ in range(10):
                strategy.generate_signal(df)

            # Benchmark
            times = []
            for _ in range(n_signals):
                start = time.perf_counter()
                strategy.generate_signal(df)
                times.append(time.perf_counter() - start)

            results[strat_name] = {
                "mean_us": mean(times) * 1_000_000,
                "std_us": stdev(times) * 1_000_000 if len(times) > 1 else 0,
                "signals_per_sec": 1 / mean(times),
            }
        except Exception as e:
            results[strat_name] = {"error": str(e)}

    return {
        "operation": "Strategy signal generation",
        "n_signals_per_strategy": n_signals,
        "strategies": results,
    }


def benchmark_indicator_calculation(n_bars: int = 300) -> dict:
    """Benchmark technical indicator calculation."""
    import numpy as np
    import pandas as pd

    # Create price data
    np.random.seed(42)
    close = 150 + np.cumsum(np.random.randn(n_bars) * 0.5)
    high = close + np.abs(np.random.randn(n_bars) * 0.3)
    low = close - np.abs(np.random.randn(n_bars) * 0.3)

    df = pd.DataFrame({"High": high, "Low": low, "Close": close})

    results = {}
    n_iterations = 1000

    # SMA
    start = time.perf_counter()
    for _ in range(n_iterations):
        df["Close"].rolling(20).mean()
    elapsed = time.perf_counter() - start
    results["SMA(20)"] = {"us_per_calc": (elapsed / n_iterations) * 1_000_000}

    # EMA
    start = time.perf_counter()
    for _ in range(n_iterations):
        df["Close"].ewm(span=20).mean()
    elapsed = time.perf_counter() - start
    results["EMA(20)"] = {"us_per_calc": (elapsed / n_iterations) * 1_000_000}

    # RSI (more complex)
    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    start = time.perf_counter()
    for _ in range(n_iterations):
        calc_rsi(df["Close"])
    elapsed = time.perf_counter() - start
    results["RSI(14)"] = {"us_per_calc": (elapsed / n_iterations) * 1_000_000}

    # ATR
    def calc_atr(df, period=14):
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    start = time.perf_counter()
    for _ in range(n_iterations):
        calc_atr(df)
    elapsed = time.perf_counter() - start
    results["ATR(14)"] = {"us_per_calc": (elapsed / n_iterations) * 1_000_000}

    return {
        "operation": "Indicator calculation",
        "n_bars": n_bars,
        "n_iterations": n_iterations,
        "indicators": results,
    }


def benchmark_full_pipeline() -> dict:
    """Benchmark full quote-to-signal pipeline."""
    import numpy as np
    import pandas as pd
    from core.alpaca_schwab_hybrid_runner import AlpacaSchwabHybridRunner
    from strategies.strategy_registry.strategy_registry import STRATEGY_CLASS_REGISTRY

    # Setup
    strategy = STRATEGY_CLASS_REGISTRY.get("sma")()  # Instantiate SMA strategy
    n_bars = 300
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq="1min")
    close = 150 + np.cumsum(np.random.randn(n_bars) * 0.5)

    df = pd.DataFrame({
        "Open": close + np.random.randn(n_bars) * 0.2,
        "High": close + np.abs(np.random.randn(n_bars) * 0.3),
        "Low": close - np.abs(np.random.randn(n_bars) * 0.3),
        "Close": close,
        "Volume": np.random.randint(10000, 100000, n_bars),
    }, index=dates)

    # Simulate full pipeline: quote → canonicalize → aggregate → signal
    n_iterations = 500
    times = []

    for i in range(n_iterations):
        start = time.perf_counter()

        # 1. Canonicalize quote
        quote = {
            "1": 150.00, "2": 150.05, "3": 150.02,
            "8": 1000, "35": int(datetime.now(timezone.utc).timestamp() * 1000),
        }
        bar_data = AlpacaSchwabHybridRunner._canonicalize_schwab_quote(quote, "TEST")

        # 2. Update dataframe (simulate adding new bar)
        # In real system this is numpy buffer append

        # 3. Generate signal
        signal = strategy.generate_signal(df)

        times.append(time.perf_counter() - start)

    return {
        "operation": "Full pipeline (quote → signal)",
        "n_iterations": n_iterations,
        "mean_ms": mean(times) * 1000,
        "std_ms": stdev(times) * 1000,
        "max_signals_per_sec": 1 / mean(times),
        "min_bar_interval_ms": mean(times) * 1000,
    }


def benchmark_order_placement() -> dict:
    """Estimate order placement latency (without actually placing orders)."""
    # This measures network round-trip simulation
    import aiohttp

    async def measure_latency():
        times = []
        # Ping Alpaca API (paper trading endpoint)
        url = "https://paper-api.alpaca.markets/v2/clock"
        headers = {
            "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
            "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY", ""),
        }

        if not headers["APCA-API-KEY-ID"]:
            return {"error": "No Alpaca credentials"}

        async with aiohttp.ClientSession() as session:
            # Warm up
            for _ in range(3):
                try:
                    async with session.get(url, headers=headers) as resp:
                        await resp.json()
                except:
                    pass

            # Measure
            for _ in range(10):
                start = time.perf_counter()
                try:
                    async with session.get(url, headers=headers) as resp:
                        await resp.json()
                    times.append(time.perf_counter() - start)
                except Exception as e:
                    return {"error": str(e)}

        return {
            "mean_ms": mean(times) * 1000,
            "std_ms": stdev(times) * 1000 if len(times) > 1 else 0,
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000,
        }

    return {
        "operation": "Alpaca API round-trip",
        "note": "Measured via /v2/clock endpoint",
        **asyncio.run(measure_latency()),
    }


def main():
    print("=" * 70)
    print("  TRADING SPEED BENCHMARK")
    print("=" * 70)
    print(f"  Time: {datetime.now()}")
    print(f"  Platform: {sys.platform}")
    print("=" * 70)
    print()

    benchmarks = [
        ("Quote Processing", benchmark_quote_processing),
        ("Bar Aggregation", benchmark_bar_aggregation),
        ("Indicator Calculation", benchmark_indicator_calculation),
        ("Strategy Signals", benchmark_strategy_signal),
        ("Full Pipeline", benchmark_full_pipeline),
        ("Network Latency", benchmark_order_placement),
    ]

    results = {}

    for name, func in benchmarks:
        print(f"Running: {name}...")
        try:
            result = func()
            results[name] = result
            print(f"  ✓ Done")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[name] = {"error": str(e)}

    # Print results
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)

    for name, result in results.items():
        print(f"\n{name}:")
        print("-" * 50)

        if "error" in result:
            print(f"  Error: {result['error']}")
            continue

        if "ops_per_sec" in result:
            print(f"  Throughput: {result['ops_per_sec']:,.0f} ops/sec")
            print(f"  Latency: {result['us_per_op']:.1f} μs/op")

        if "strategies" in result:
            for strat, data in result["strategies"].items():
                if "error" in data:
                    print(f"  {strat}: Error - {data['error']}")
                else:
                    print(f"  {strat}: {data['mean_us']:.0f} μs ({data['signals_per_sec']:.0f}/sec)")

        if "indicators" in result:
            for ind, data in result["indicators"].items():
                print(f"  {ind}: {data['us_per_calc']:.0f} μs")

        if "max_signals_per_sec" in result:
            print(f"  Max throughput: {result['max_signals_per_sec']:.0f} signals/sec")
            print(f"  Min bar interval: {result['min_bar_interval_ms']:.1f} ms")

        if "mean_ms" in result and "operation" in result and "API" in result["operation"]:
            print(f"  Round-trip: {result['mean_ms']:.0f} ms (±{result.get('std_ms', 0):.0f})")

    # Summary
    print()
    print("=" * 70)
    print("  SUMMARY: MINIMUM VIABLE BAR INTERVALS")
    print("=" * 70)

    # Calculate constraints
    pipeline_ms = results.get("Full Pipeline", {}).get("mean_ms", 10)
    network_ms = results.get("Network Latency", {}).get("mean_ms", 100)

    print(f"""
  Signal generation:     ~{pipeline_ms:.0f} ms
  Network round-trip:    ~{network_ms:.0f} ms
  Safety margin (2x):    ~{(pipeline_ms + network_ms) * 2:.0f} ms

  RECOMMENDED MINIMUM BAR INTERVALS:

  Conservative:          1000 ms (1 second)   - Very safe
  Moderate:               500 ms              - Good balance
  Aggressive:             250 ms              - May stress Pi

  Note: These are theoretical limits. Real-world performance
  depends on market volatility, number of symbols, and Pi load.
""")

    print("=" * 70)


if __name__ == "__main__":
    main()
