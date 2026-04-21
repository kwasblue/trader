#!/usr/bin/env python3
"""
Stress Test: Can the system handle N symbols at Xms bar intervals?

Simulates realistic trading load:
- Continuous quote stream for N symbols
- Bar aggregation at configurable intervals
- Strategy signal generation on bar close
- Measures throughput, latency, and dropped quotes

Usage:
    python tools/stress_test_trading.py --symbols 50 --interval 500
    python tools/stress_test_trading.py --symbols 30 --interval 250 --duration 60
"""

import argparse
import asyncio
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev, quantiles
from typing import Any

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass
class StressTestMetrics:
    """Metrics collected during stress test."""
    quotes_received: int = 0
    quotes_processed: int = 0
    quotes_dropped: int = 0
    bars_created: int = 0
    signals_generated: int = 0
    signal_latencies_ms: list = field(default_factory=list)
    quote_process_times_us: list = field(default_factory=list)
    bar_process_times_ms: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    start_time: float = 0
    end_time: float = 0


class TradingStressTest:
    """Simulates trading load to find system limits."""

    def __init__(
        self,
        n_symbols: int = 50,
        bar_interval_ms: int = 500,
        quotes_per_sec_per_symbol: int = 10,
        duration_sec: int = 30,
    ):
        self.n_symbols = n_symbols
        self.bar_interval_ms = bar_interval_ms
        self.quotes_per_sec = quotes_per_sec_per_symbol
        self.duration_sec = duration_sec

        # Generate symbol names
        self.symbols = [f"SYM{i:03d}" for i in range(n_symbols)]

        # Bar aggregation state
        self.bar_state: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"open": None, "high": None, "low": None, "close": None, "volume": 0, "bar_id": None}
        )

        # Strategy (use a real one)
        self.strategy = None

        # Price history per symbol (simplified)
        self.price_history: dict[str, list] = defaultdict(list)
        self.history_len = 300

        # Metrics
        self.metrics = StressTestMetrics()

        # Control
        self._running = False
        self._quote_queue: asyncio.Queue = None

    def _load_strategy(self):
        """Load a real strategy for testing."""
        try:
            from strategies.strategy_registry.strategy_registry import STRATEGY_CLASS_REGISTRY
            # Use momentum - it's the fastest
            self.strategy = STRATEGY_CLASS_REGISTRY.get("momentum")()
            return True
        except Exception as e:
            print(f"Warning: Could not load strategy: {e}")
            return False

    def _bar_bucket(self, ts_ms: int) -> int:
        """Calculate bar bucket from timestamp."""
        return ts_ms // self.bar_interval_ms

    def _process_quote(self, symbol: str, price: float, volume: int, ts_ms: int) -> dict | None:
        """Process a quote and return bar if closed."""
        start = time.perf_counter()

        bar_id = self._bar_bucket(ts_ms)
        state = self.bar_state[symbol]
        bar_closed = False

        if state["bar_id"] is None or state["bar_id"] != bar_id:
            # New bar - close previous if exists
            if state["bar_id"] is not None and state["close"] is not None:
                bar_closed = True
                closed_bar = {
                    "symbol": symbol,
                    "Open": state["open"],
                    "High": state["high"],
                    "Low": state["low"],
                    "Close": state["close"],
                    "Volume": state["volume"],
                }

            # Start new bar
            state["open"] = price
            state["high"] = price
            state["low"] = price
            state["close"] = price
            state["volume"] = volume
            state["bar_id"] = bar_id
        else:
            # Update current bar
            state["high"] = max(state["high"], price)
            state["low"] = min(state["low"], price)
            state["close"] = price
            state["volume"] += volume

        elapsed_us = (time.perf_counter() - start) * 1_000_000
        self.metrics.quote_process_times_us.append(elapsed_us)
        self.metrics.quotes_processed += 1

        if bar_closed:
            return closed_bar
        return None

    def _generate_signal(self, symbol: str, bar: dict) -> int:
        """Generate trading signal for a bar."""
        # Add to history
        self.price_history[symbol].append(bar["Close"])
        if len(self.price_history[symbol]) > self.history_len:
            self.price_history[symbol] = self.price_history[symbol][-self.history_len:]

        # Need enough history for strategy
        if len(self.price_history[symbol]) < 50:
            return 0

        # If we have a strategy, use it
        if self.strategy:
            try:
                import pandas as pd
                import numpy as np

                prices = self.price_history[symbol]
                n = len(prices)

                # Create minimal DataFrame for strategy
                df = pd.DataFrame({
                    "Open": prices,
                    "High": [p * 1.001 for p in prices],
                    "Low": [p * 0.999 for p in prices],
                    "Close": prices,
                    "Volume": [10000] * n,
                })

                signal = self.strategy.generate_signal(df)
                return signal
            except Exception as e:
                self.metrics.errors.append(f"Signal error: {e}")
                return 0

        # Fallback: simple momentum
        prices = self.price_history[symbol]
        if len(prices) >= 20:
            sma_fast = mean(prices[-5:])
            sma_slow = mean(prices[-20:])
            if sma_fast > sma_slow * 1.001:
                return 1
            elif sma_fast < sma_slow * 0.999:
                return -1
        return 0

    async def _quote_producer(self):
        """Produce simulated quotes at realistic rate."""
        quote_interval = 1.0 / (self.quotes_per_sec * self.n_symbols)
        base_prices = {s: 100 + i * 0.1 for i, s in enumerate(self.symbols)}

        while self._running:
            for symbol in self.symbols:
                if not self._running:
                    break

                # Simulate price movement
                base_prices[symbol] *= (1 + (hash(f"{symbol}{time.time()}") % 100 - 50) / 100000)
                price = base_prices[symbol]

                quote = {
                    "symbol": symbol,
                    "price": price,
                    "volume": 100,
                    "ts_ms": int(time.time() * 1000),
                }

                self.metrics.quotes_received += 1

                try:
                    self._quote_queue.put_nowait(quote)
                except asyncio.QueueFull:
                    self.metrics.quotes_dropped += 1

            await asyncio.sleep(quote_interval)

    async def _quote_consumer(self):
        """Consume quotes and process them."""
        while self._running:
            try:
                quote = await asyncio.wait_for(
                    self._quote_queue.get(),
                    timeout=0.1
                )

                bar = self._process_quote(
                    quote["symbol"],
                    quote["price"],
                    quote["volume"],
                    quote["ts_ms"]
                )

                if bar:
                    self.metrics.bars_created += 1

                    # Generate signal
                    start = time.perf_counter()
                    signal = self._generate_signal(bar["symbol"], bar)
                    elapsed_ms = (time.perf_counter() - start) * 1000

                    self.metrics.bar_process_times_ms.append(elapsed_ms)

                    if signal != 0:
                        self.metrics.signals_generated += 1
                        self.metrics.signal_latencies_ms.append(elapsed_ms)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.metrics.errors.append(str(e))

    async def run(self) -> StressTestMetrics:
        """Run the stress test."""
        print(f"\nStarting stress test:")
        print(f"  Symbols: {self.n_symbols}")
        print(f"  Bar interval: {self.bar_interval_ms} ms")
        print(f"  Quote rate: {self.quotes_per_sec}/sec/symbol")
        print(f"  Duration: {self.duration_sec} sec")
        print()

        # Load strategy
        self._load_strategy()

        # Initialize
        self._quote_queue = asyncio.Queue(maxsize=10000)
        self._running = True
        self.metrics.start_time = time.time()

        # Start producer and consumer
        producer = asyncio.create_task(self._quote_producer())
        consumer = asyncio.create_task(self._quote_consumer())

        # Progress reporting
        start = time.time()
        while time.time() - start < self.duration_sec:
            await asyncio.sleep(5)
            elapsed = time.time() - start
            qps = self.metrics.quotes_processed / elapsed
            print(f"  [{elapsed:.0f}s] Quotes: {self.metrics.quotes_processed:,} ({qps:.0f}/s) | "
                  f"Bars: {self.metrics.bars_created:,} | "
                  f"Signals: {self.metrics.signals_generated} | "
                  f"Dropped: {self.metrics.quotes_dropped}")

        # Stop
        self._running = False
        producer.cancel()
        consumer.cancel()

        try:
            await producer
        except asyncio.CancelledError:
            pass

        try:
            await consumer
        except asyncio.CancelledError:
            pass

        self.metrics.end_time = time.time()
        return self.metrics


def print_results(metrics: StressTestMetrics, n_symbols: int, bar_interval_ms: int):
    """Print stress test results."""
    duration = metrics.end_time - metrics.start_time

    print()
    print("=" * 70)
    print("  STRESS TEST RESULTS")
    print("=" * 70)

    # Throughput
    print(f"\nTHROUGHPUT:")
    print(f"  Duration: {duration:.1f} sec")
    print(f"  Quotes received: {metrics.quotes_received:,}")
    print(f"  Quotes processed: {metrics.quotes_processed:,}")
    print(f"  Quotes dropped: {metrics.quotes_dropped:,} ({100*metrics.quotes_dropped/max(1,metrics.quotes_received):.1f}%)")
    print(f"  Bars created: {metrics.bars_created:,}")
    print(f"  Signals generated: {metrics.signals_generated:,}")

    # Rates
    print(f"\nRATES:")
    print(f"  Quotes/sec: {metrics.quotes_processed / duration:,.0f}")
    print(f"  Bars/sec: {metrics.bars_created / duration:.1f}")
    print(f"  Expected bars/sec: {n_symbols * 1000 / bar_interval_ms:.1f}")

    # Latencies
    if metrics.quote_process_times_us:
        print(f"\nQUOTE PROCESSING LATENCY:")
        print(f"  Mean: {mean(metrics.quote_process_times_us):.1f} μs")
        if len(metrics.quote_process_times_us) > 1:
            print(f"  Std: {stdev(metrics.quote_process_times_us):.1f} μs")
            p95, p99 = quantiles(metrics.quote_process_times_us, n=100)[94], quantiles(metrics.quote_process_times_us, n=100)[98]
            print(f"  P95: {p95:.1f} μs")
            print(f"  P99: {p99:.1f} μs")

    if metrics.bar_process_times_ms:
        print(f"\nBAR PROCESSING LATENCY (includes signal generation):")
        print(f"  Mean: {mean(metrics.bar_process_times_ms):.2f} ms")
        if len(metrics.bar_process_times_ms) > 1:
            print(f"  Std: {stdev(metrics.bar_process_times_ms):.2f} ms")
            p95, p99 = quantiles(metrics.bar_process_times_ms, n=100)[94], quantiles(metrics.bar_process_times_ms, n=100)[98]
            print(f"  P95: {p95:.2f} ms")
            print(f"  P99: {p99:.2f} ms")

    # Verdict
    print()
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)

    drop_rate = metrics.quotes_dropped / max(1, metrics.quotes_received)
    bar_rate_actual = metrics.bars_created / duration
    bar_rate_expected = n_symbols * 1000 / bar_interval_ms

    issues = []

    if drop_rate > 0.01:
        issues.append(f"High drop rate ({drop_rate*100:.1f}%) - system can't keep up")

    if bar_rate_actual < bar_rate_expected * 0.95:
        issues.append(f"Bar rate below expected ({bar_rate_actual:.1f} vs {bar_rate_expected:.1f})")

    if metrics.bar_process_times_ms and max(metrics.bar_process_times_ms) > bar_interval_ms:
        issues.append(f"Some bars took longer than interval ({max(metrics.bar_process_times_ms):.1f}ms > {bar_interval_ms}ms)")

    if issues:
        print(f"\n  ⚠️  ISSUES DETECTED:")
        for issue in issues:
            print(f"     - {issue}")
        print(f"\n  RECOMMENDATION: Reduce symbols or increase bar interval")
    else:
        print(f"\n  ✅ SYSTEM CAN HANDLE THIS LOAD")
        print(f"\n  {n_symbols} symbols @ {bar_interval_ms}ms = VIABLE")

        # Estimate headroom
        if metrics.bar_process_times_ms:
            avg_bar_time = mean(metrics.bar_process_times_ms)
            headroom = bar_interval_ms / avg_bar_time
            print(f"  Headroom: {headroom:.1f}x (can handle ~{int(n_symbols * headroom * 0.7)} symbols)")

    if metrics.errors:
        print(f"\n  ERRORS ({len(metrics.errors)}):")
        for err in metrics.errors[:5]:
            print(f"     - {err}")

    print()
    print("=" * 70)


async def main():
    parser = argparse.ArgumentParser(description="Stress test trading system")
    parser.add_argument("--symbols", "-n", type=int, default=50, help="Number of symbols")
    parser.add_argument("--interval", "-i", type=int, default=500, help="Bar interval in ms")
    parser.add_argument("--quotes", "-q", type=int, default=10, help="Quotes per second per symbol")
    parser.add_argument("--duration", "-d", type=int, default=30, help="Test duration in seconds")

    args = parser.parse_args()

    test = TradingStressTest(
        n_symbols=args.symbols,
        bar_interval_ms=args.interval,
        quotes_per_sec_per_symbol=args.quotes,
        duration_sec=args.duration,
    )

    metrics = await test.run()
    print_results(metrics, args.symbols, args.interval)


if __name__ == "__main__":
    asyncio.run(main())
