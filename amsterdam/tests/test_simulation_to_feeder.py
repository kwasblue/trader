#!/usr/bin/env python3
"""
Integration test: Simulation -> EventBus -> Feeder (without Qt GUI)

This test verifies that:
1. SimulationRunner emits PNL_UPDATE events
2. A feeder-style subscriber receives them
3. The complete pipeline works end-to-end

Run with: python3 tests/test_simulation_to_feeder.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asyncio
from datetime import datetime, timezone
from collections import deque

# Reset singleton before imports to ensure clean state
import core.events.eventhandler as eh_module
eh_module.EventHandler._instance = None
eh_module.EventHandler._initialized = False
eh_module._global_event_handler = None

from core.events.eventhandler import get_event_handler
from core.events import events
from core.simulator.simulation import SimConfig, SimulationRunner


class MockFeeder:
    """Simulates what DataFeeder does but without Qt."""

    def __init__(self):
        self.bus = get_event_handler()
        self.pnl_received = deque(maxlen=100)
        self.bar_received = deque(maxlen=100)

        # Subscribe synchronously like real feeder
        self.bus.subscribe_sync(events.EVENT_PNL_UPDATE, self._handle_pnl)
        self.bus.subscribe_sync(events.EVENT_NEW_BAR, self._handle_bar)

        pnl_count = len(self.bus.listeners.get(events.EVENT_PNL_UPDATE, []))
        bar_count = len(self.bus.listeners.get(events.EVENT_NEW_BAR, []))
        print(f"[MockFeeder] Subscribed: PNL={pnl_count}, BAR={bar_count} listeners")

    async def _handle_pnl(self, event):
        data = event.payload if hasattr(event, 'payload') else event
        self.pnl_received.append(data)
        value = data.get('portfolio_value', 0)
        print(f"[MockFeeder] PNL received: ${value:,.2f}")

    async def _handle_bar(self, event):
        data = event.payload if hasattr(event, 'payload') else event
        self.bar_received.append(data)
        symbol = data.get('symbol', '?')
        close = data.get('close', 0)
        # Only print occasionally to avoid spam
        if len(self.bar_received) % 20 == 0:
            print(f"[MockFeeder] BAR #{len(self.bar_received)}: {symbol} close=${close:.2f}")


async def run_test():
    """Run the integration test."""
    print("\n" + "="*70)
    print("  SIMULATION -> FEEDER INTEGRATION TEST")
    print("="*70 + "\n")

    # Create mock feeder (subscribes to events)
    feeder = MockFeeder()

    # Create simulation config (short run for testing)
    config = SimConfig(
        symbols=["AAPL", "MSFT"],
        steps=50,  # Just 50 bars for quick test
        bar_sleep=0.01,  # Fast for testing
        starting_cash=100_000.0,
    )

    # Create and run simulation
    print(f"\n[TEST] Starting simulation with {config.symbols}...")
    print(f"[TEST] EventHandler ID: {id(feeder.bus)}\n")

    runner = SimulationRunner(config)

    # Verify both use same EventHandler
    assert id(runner.events) == id(feeder.bus), "EventHandler mismatch!"
    print(f"[TEST] Simulation EventHandler ID: {id(runner.events)}")
    print(f"[TEST] ✓ Same EventHandler instance confirmed\n")

    # Run simulation
    await runner.run()

    # Check results
    print("\n" + "="*70)
    print("  TEST RESULTS")
    print("="*70)

    pnl_count = len(feeder.pnl_received)
    bar_count = len(feeder.bar_received)

    print(f"\n  PNL events received: {pnl_count}")
    print(f"  BAR events received: {bar_count}")

    if pnl_count > 0:
        last_pnl = feeder.pnl_received[-1]
        print(f"\n  Last PNL:")
        print(f"    Portfolio Value: ${last_pnl.get('portfolio_value', 0):,.2f}")
        print(f"    Unrealized: ${last_pnl.get('unrealized', 0):,.2f}")
        print(f"    Realized: ${last_pnl.get('realized', 0):,.2f}")
        print(f"    Drawdown: {last_pnl.get('drawdown', 0):.2%}")

    # Assertions
    success = True

    if pnl_count == 0:
        print("\n  ❌ FAIL: No PNL events received!")
        success = False
    else:
        print(f"\n  ✅ PASS: Received {pnl_count} PNL events")

    if bar_count == 0:
        print("  ❌ FAIL: No BAR events received!")
        success = False
    else:
        print(f"  ✅ PASS: Received {bar_count} BAR events")

    print("\n" + "="*70)
    if success:
        print("  ✅ ALL TESTS PASSED - Pipeline is working!")
    else:
        print("  ❌ TESTS FAILED - Pipeline has issues")
    print("="*70 + "\n")

    return success


if __name__ == "__main__":
    success = asyncio.run(run_test())
    sys.exit(0 if success else 1)
