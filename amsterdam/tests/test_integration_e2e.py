#!/usr/bin/env python3
"""
End-to-End Integration Test for Multi-Timeframe System

Tests the complete flow:
1. StrategyRoutingManager loads config
2. Get routing decisions with timeframes
3. BarAggregator configured from routing
4. Bars processed and emitted
5. Regime changes trigger timeframe switches

This test doesn't require external dependencies - it tests the integration
of the components we built.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.bar_aggregator import BarAggregator, Bar
from core.logic.strategy_routing_manager import StrategyRoutingManager


def test_existing_config_backward_compatibility():
    """Test that existing simple config format works."""
    print("\n=== Test 1: Backward Compatibility ===")

    # Load existing config
    config_path = "config/strategy_routing.json"
    router = StrategyRoutingManager(config_path)

    # Test getting routing for existing symbols
    symbols = ['AAPL', 'TSLA', 'MSFT', 'NVDA']
    regimes = ['low_volatility', 'normal', 'high_volatility']

    for symbol in symbols:
        for regime in regimes:
            routing = router.get_routing(symbol, regime)

            # Should have all required fields
            assert 'strategy' in routing, f"Missing strategy for {symbol}/{regime}"
            assert 'timeframe' in routing, f"Missing timeframe for {symbol}/{regime}"
            assert 'use_hybrid' in routing, f"Missing use_hybrid for {symbol}/{regime}"

            # Timeframe should default to '5min'
            assert routing['timeframe'] == '5min', \
                f"Default timeframe should be 5min, got {routing['timeframe']}"

            print(f"  ✓ {symbol:6s} / {regime:15s} → {routing['strategy']:15s} @ {routing['timeframe']}")

    print("  ✅ Backward compatibility: OK")
    return True


def test_routing_to_aggregator_integration():
    """Test integration between routing and aggregator."""
    print("\n=== Test 2: Routing → Aggregator Integration ===")

    # Initialize components
    router = StrategyRoutingManager("config/strategy_routing.json")
    aggregator = BarAggregator()

    # Configure aggregator from routing (simulating startup)
    symbols = ['AAPL', 'TSLA', 'MSFT']
    regime = 'normal'  # Assume normal regime for all

    for symbol in symbols:
        routing = router.get_routing(symbol, regime)
        aggregator.set_timeframe(symbol, routing['timeframe'])
        print(f"  ✓ Configured {symbol}: {routing['strategy']} @ {routing['timeframe']}")

    # Verify aggregator configuration
    stats = aggregator.get_stats()
    assert stats['active_symbols'] == 3, f"Expected 3 symbols, got {stats['active_symbols']}"

    for symbol in symbols:
        tf = aggregator.get_timeframe(symbol)
        assert tf == '5min', f"Expected 5min for {symbol}, got {tf}"

    print("  ✅ Routing → Aggregator: OK")
    return True


def test_end_to_end_bar_flow():
    """Test complete bar flow: routing → aggregator → callback."""
    print("\n=== Test 3: End-to-End Bar Flow ===")

    # Initialize
    router = StrategyRoutingManager("config/strategy_routing.json")
    aggregator = BarAggregator()

    # Configure
    symbols = ['AAPL', 'TSLA']
    for symbol in symbols:
        routing = router.get_routing(symbol, 'normal')
        aggregator.set_timeframe(symbol, routing['timeframe'])

    # Track received bars
    received_bars = []

    def on_bar(bar: Bar):
        received_bars.append(bar)
        print(f"  ✓ Received {bar.symbol} bar @ {bar.timestamp.strftime('%H:%M')} "
              f"[{bar.timeframe}] OHLCV({bar.open:.2f}, {bar.high:.2f}, "
              f"{bar.low:.2f}, {bar.close:.2f}, {bar.volume})")

    aggregator.register_callback(on_bar)

    # Simulate streaming 1-minute bars
    base_time = datetime(2026, 3, 14, 9, 30, 0)

    print("\n  Sending 10 minutes of 1-min bars...")
    for i in range(10):
        for symbol in symbols:
            bar = Bar(
                timestamp=base_time + timedelta(minutes=i),
                open=150.0 + i * 0.1,
                high=151.0 + i * 0.1,
                low=149.0 + i * 0.1,
                close=150.5 + i * 0.1,
                volume=10000,
                symbol=symbol,
                timeframe="1min"
            )
            aggregator.process_bar(bar)

    # Should have 2 completed 5-min bars per symbol (at 5min and 10min marks)
    # But only 1 emitted so far (second window still open)
    aapl_bars = [b for b in received_bars if b.symbol == 'AAPL']
    tsla_bars = [b for b in received_bars if b.symbol == 'TSLA']

    assert len(aapl_bars) == 1, f"Expected 1 AAPL bar, got {len(aapl_bars)}"
    assert len(tsla_bars) == 1, f"Expected 1 TSLA bar, got {len(tsla_bars)}"

    # Verify aggregation
    bar = aapl_bars[0]
    assert bar.timeframe == '5min', f"Expected 5min, got {bar.timeframe}"
    assert bar.volume == 50000, f"Expected 50000 volume (5 bars * 10000), got {bar.volume}"

    print(f"\n  ✅ End-to-End Flow: OK ({len(received_bars)} bars emitted)")
    return True


def test_regime_change_timeframe_switch():
    """Test timeframe switching on regime change."""
    print("\n=== Test 4: Regime Change → Timeframe Switch ===")

    # Create test config with different timeframes per regime
    import tempfile
    import json

    test_config = {
        "AAPL": {
            "low_volatility": {
                "strategy": "meanreversion",
                "timeframe": "5min"
            },
            "normal": {
                "strategy": "rsi",
                "timeframe": "15min"
            },
            "high_volatility": {
                "strategy": "rsi",
                "timeframe": "30min"
            },
            "default": {
                "strategy": "rsi",
                "timeframe": "5min"
            },
            "use_hybrid": False
        }
    }

    # Write test config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f)
        test_config_path = f.name

    try:
        # Initialize with test config
        router = StrategyRoutingManager(test_config_path)
        aggregator = BarAggregator()

        # Start in normal regime
        regime = 'normal'
        routing = router.get_routing('AAPL', regime)
        aggregator.set_timeframe('AAPL', routing['timeframe'])

        print(f"  Initial: regime={regime}, timeframe={routing['timeframe']}")
        assert routing['timeframe'] == '15min'

        # Simulate regime change to high_volatility
        old_regime = regime
        new_regime = 'high_volatility'

        new_routing = router.get_routing('AAPL', new_regime)
        print(f"  Change:  regime={old_regime} → {new_regime}, timeframe={new_routing['timeframe']}")

        # Update aggregator
        aggregator.set_timeframe('AAPL', new_routing['timeframe'])

        # Verify change
        current_tf = aggregator.get_timeframe('AAPL')
        assert current_tf == '30min', f"Expected 30min, got {current_tf}"

        # Check stats
        stats = aggregator.get_stats()
        assert stats['timeframe_changes'] == 1, \
            f"Expected 1 timeframe change, got {stats['timeframe_changes']}"

        print(f"  ✅ Regime Change: OK (timeframe switches recorded)")

    finally:
        # Cleanup
        os.unlink(test_config_path)

    return True


def test_multiple_symbols_different_timeframes():
    """Test multiple symbols with different timeframes simultaneously."""
    print("\n=== Test 5: Multiple Symbols, Different Timeframes ===")

    import tempfile
    import json

    # Create config with different timeframes per symbol
    test_config = {
        "AAPL": {
            "normal": {
                "strategy": "rsi",
                "timeframe": "5min"
            }
        },
        "TSLA": {
            "normal": {
                "strategy": "momentum",
                "timeframe": "15min"
            }
        },
        "MSFT": {
            "normal": {
                "strategy": "sma",
                "timeframe": "30min"
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f)
        test_config_path = f.name

    try:
        router = StrategyRoutingManager(test_config_path)
        aggregator = BarAggregator()

        # Configure each symbol
        symbols = ['AAPL', 'TSLA', 'MSFT']
        expected_tfs = {'AAPL': '5min', 'TSLA': '15min', 'MSFT': '30min'}

        for symbol in symbols:
            routing = router.get_routing(symbol, 'normal')
            aggregator.set_timeframe(symbol, routing['timeframe'])
            print(f"  ✓ {symbol}: {routing['strategy']} @ {routing['timeframe']}")

        # Track received bars
        received_bars = []
        aggregator.register_callback(lambda b: received_bars.append(b))

        # Send 30 minutes of 1-min bars
        base_time = datetime(2026, 3, 14, 9, 30, 0)

        for i in range(30):
            for symbol in symbols:
                bar = Bar(
                    timestamp=base_time + timedelta(minutes=i),
                    open=150.0,
                    high=151.0,
                    low=149.0,
                    close=150.5,
                    volume=1000,
                    symbol=symbol,
                    timeframe="1min"
                )
                aggregator.process_bar(bar)

        # AAPL (5min): Should have 5 bars (5, 10, 15, 20, 25 min marks)
        # TSLA (15min): Should have 1 bar (15 min mark)
        # MSFT (30min): Should have 0 bars (window still open)

        aapl_bars = [b for b in received_bars if b.symbol == 'AAPL']
        tsla_bars = [b for b in received_bars if b.symbol == 'TSLA']
        msft_bars = [b for b in received_bars if b.symbol == 'MSFT']

        print(f"\n  Bars emitted: AAPL={len(aapl_bars)}, TSLA={len(tsla_bars)}, MSFT={len(msft_bars)}")

        assert len(aapl_bars) == 5, f"Expected 5 AAPL bars, got {len(aapl_bars)}"
        assert len(tsla_bars) == 1, f"Expected 1 TSLA bar, got {len(tsla_bars)}"
        assert len(msft_bars) == 0, f"Expected 0 MSFT bars, got {len(msft_bars)}"

        # Verify timeframes
        for bar in aapl_bars:
            assert bar.timeframe == '5min'
        for bar in tsla_bars:
            assert bar.timeframe == '15min'

        print("  ✅ Multiple Symbols/Timeframes: OK")

    finally:
        os.unlink(test_config_path)

    return True


def run_all_tests():
    """Run all integration tests."""
    print("=" * 70)
    print("Multi-Timeframe System - End-to-End Integration Tests")
    print("=" * 70)

    tests = [
        ("Backward Compatibility", test_existing_config_backward_compatibility),
        ("Routing → Aggregator", test_routing_to_aggregator_integration),
        ("End-to-End Bar Flow", test_end_to_end_bar_flow),
        ("Regime Change", test_regime_change_timeframe_switch),
        ("Multiple Symbols/Timeframes", test_multiple_symbols_different_timeframes),
    ]

    results = []

    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  ❌ {name}: FAILED - {e}")

    # Summary
    print("\n" + "=" * 70)
    print("Test Results Summary")
    print("=" * 70)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for name, success, error in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
        if error:
            print(f"       Error: {error}")

    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)

    return passed == total


if __name__ == '__main__':
    # Change to project root
    os.chdir(Path(__file__).parent.parent)

    success = run_all_tests()
    sys.exit(0 if success else 1)
