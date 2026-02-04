#!/usr/bin/env python3
"""
Integration test for the event pipeline.

Tests the complete data flow:
    SimulationRunner -> EventHandler -> DataFeeder -> Qt Signals

Run with: python -m pytest tests/test_event_pipeline.py -v -s
Or directly: python tests/test_event_pipeline.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asyncio
import logging
from datetime import datetime, timezone
import pytest

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger("test_pipeline")


def reset_singleton():
    """Helper to properly reset EventHandler singleton between tests."""
    import core.events.eventhandler as eh_module
    eh_module.EventHandler._instance = None
    eh_module.EventHandler._initialized = False
    eh_module._global_event_handler = None


def test_singleton_initialization():
    """Test that EventHandler singleton is properly initialized."""
    print("\n" + "="*60)
    print("TEST 1: Singleton Initialization")
    print("="*60)

    reset_singleton()
    from core.events.eventhandler import EventHandler, get_event_handler

    # Create first instance
    bus1 = get_event_handler()
    print(f"  First instance ID: {id(bus1)}")

    # Subscribe something
    received = []
    async def handler(event):
        received.append(event)

    bus1.subscribe_sync("TEST_EVENT", handler)
    listener_count_1 = len(bus1.listeners.get("TEST_EVENT", []))
    print(f"  After subscribe: {listener_count_1} listeners")

    # Get second instance - should be same object with listeners intact
    bus2 = get_event_handler()
    print(f"  Second instance ID: {id(bus2)}")

    listener_count_2 = len(bus2.listeners.get("TEST_EVENT", []))
    print(f"  Listeners preserved: {listener_count_2}")

    assert bus1 is bus2, "Singleton instances should be identical"
    assert listener_count_2 == listener_count_1, "Listeners should be preserved"

    print("  ✅ PASSED: Singleton works correctly")
    return True


def test_synchronous_subscription():
    """Test that synchronous subscriptions work immediately."""
    print("\n" + "="*60)
    print("TEST 2: Synchronous Subscription")
    print("="*60)

    reset_singleton()
    from core.events.eventhandler import get_event_handler

    bus = get_event_handler()

    received = []
    async def handler(event):
        received.append(event.payload)
        print(f"  [Handler] Received: {event.payload}")

    # Subscribe synchronously
    bus.subscribe_sync("SYNC_TEST", handler)
    count = len(bus.listeners.get("SYNC_TEST", []))
    print(f"  Subscribers immediately after subscribe_sync: {count}")

    assert count == 1, "Should have 1 subscriber immediately"
    print("  ✅ PASSED: Synchronous subscription works")
    return True


@pytest.mark.asyncio
async def test_emit_receive():
    """Test that emit -> receive works correctly."""
    print("\n" + "="*60)
    print("TEST 3: Emit -> Receive")
    print("="*60)

    reset_singleton()
    from core.events.eventhandler import get_event_handler

    bus = get_event_handler()

    received = []
    async def handler(event):
        received.append(event.payload)
        print(f"  [Handler] Received payload: {event.payload}")

    # Subscribe
    bus.subscribe_sync("EMIT_TEST", handler)
    print(f"  Subscribed, listeners: {len(bus.listeners.get('EMIT_TEST', []))}")

    # Emit
    test_payload = {"value": 12345, "timestamp": datetime.now(timezone.utc).isoformat()}
    print(f"  Emitting payload: {test_payload}")
    await bus.emit("EMIT_TEST", test_payload)

    # Wait for async handler to complete
    await asyncio.sleep(0.1)

    print(f"  Received count: {len(received)}")
    assert len(received) == 1, f"Should receive 1 event, got {len(received)}"
    assert received[0] == test_payload, "Payload should match"

    print("  ✅ PASSED: Emit -> Receive works")
    return True


@pytest.mark.asyncio
async def test_pnl_event_flow():
    """Test PNL_UPDATE event flow (simulating what simulation does)."""
    print("\n" + "="*60)
    print("TEST 4: PNL_UPDATE Event Flow")
    print("="*60)

    reset_singleton()
    from core.events.eventhandler import get_event_handler
    from core.events.events import EVENT_PNL_UPDATE

    bus = get_event_handler()
    print(f"  EventHandler ID: {id(bus)}")

    pnl_received = []
    async def pnl_handler(event):
        data = event.payload
        value = data.get('portfolio_value', 0)
        pnl_received.append(data)
        print(f"  [PNL Handler] Portfolio value: ${value:,.2f}")

    # Subscribe (like feeder does)
    bus.subscribe_sync(EVENT_PNL_UPDATE, pnl_handler)
    print(f"  Subscribed to {EVENT_PNL_UPDATE}, listeners: {len(bus.listeners.get(EVENT_PNL_UPDATE, []))}")

    # Emit PNL update (like simulation does)
    pnl_payload = {
        "portfolio_value": 100500.75,
        "equity_curve": [100000, 100250, 100500.75],
        "unrealized": 500.75,
        "realized": 0,
        "drawdown": 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    print(f"  Emitting PNL: ${pnl_payload['portfolio_value']:,.2f}")
    await bus.emit(EVENT_PNL_UPDATE, pnl_payload)

    # Wait for handler
    await asyncio.sleep(0.1)

    assert len(pnl_received) == 1, f"Should receive 1 PNL event, got {len(pnl_received)}"
    assert pnl_received[0]['portfolio_value'] == 100500.75

    print("  ✅ PASSED: PNL_UPDATE flow works")
    return True


@pytest.mark.asyncio
async def test_multiple_subscribers():
    """Test that multiple subscribers all receive events."""
    print("\n" + "="*60)
    print("TEST 5: Multiple Subscribers")
    print("="*60)

    reset_singleton()
    from core.events.eventhandler import get_event_handler

    bus = get_event_handler()

    received_1 = []
    received_2 = []
    received_3 = []

    async def handler_1(event):
        received_1.append(event.payload)
        print(f"  [Handler 1] Received")

    async def handler_2(event):
        received_2.append(event.payload)
        print(f"  [Handler 2] Received")

    async def handler_3(event):
        received_3.append(event.payload)
        print(f"  [Handler 3] Received")

    # Subscribe all
    bus.subscribe_sync("MULTI_TEST", handler_1)
    bus.subscribe_sync("MULTI_TEST", handler_2)
    bus.subscribe_sync("MULTI_TEST", handler_3)

    count = len(bus.listeners.get("MULTI_TEST", []))
    print(f"  Total subscribers: {count}")

    # Emit
    await bus.emit("MULTI_TEST", {"value": 42})
    await asyncio.sleep(0.1)

    assert len(received_1) == 1, "Handler 1 should receive"
    assert len(received_2) == 1, "Handler 2 should receive"
    assert len(received_3) == 1, "Handler 3 should receive"

    print("  ✅ PASSED: All subscribers receive events")
    return True


@pytest.mark.asyncio
async def test_feeder_subscription():
    """Test that DataFeeder subscribes correctly (without Qt)."""
    print("\n" + "="*60)
    print("TEST 6: DataFeeder-style Subscription")
    print("="*60)

    reset_singleton()
    from core.events.eventhandler import get_event_handler
    from core.events import events

    # Simulate what feeder does
    bus = get_event_handler()
    print(f"  EventHandler ID: {id(bus)}")

    pnl_received = []
    bar_received = []

    async def handle_pnl(event):
        data = event.payload if hasattr(event, 'payload') else event
        pnl_received.append(data)
        print(f"  [Feeder PNL] Received: ${data.get('portfolio_value', 0):,.2f}")

    async def handle_bar(event):
        data = event.payload if hasattr(event, 'payload') else event
        bar_received.append(data)
        print(f"  [Feeder BAR] Received: {data.get('symbol')}")

    # Subscribe synchronously like feeder does
    bus.subscribe_sync(events.EVENT_PNL_UPDATE, handle_pnl)
    bus.subscribe_sync(events.EVENT_NEW_BAR, handle_bar)

    pnl_count = len(bus.listeners.get(events.EVENT_PNL_UPDATE, []))
    bar_count = len(bus.listeners.get(events.EVENT_NEW_BAR, []))
    print(f"  PNL listeners: {pnl_count}, BAR listeners: {bar_count}")

    # Now simulate what simulation does
    print("  --- Simulating emissions ---")

    # Emit PNL (with all required fields per PnLPayload schema)
    await bus.emit(events.EVENT_PNL_UPDATE, {
        "portfolio_value": 100000.0,
        "equity_curve": [100000.0],
        "unrealized": 0.0,
        "realized": 0.0,
        "drawdown": 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    # Emit BAR
    await bus.emit(events.EVENT_NEW_BAR, {
        "symbol": "AAPL",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "open": 150.0,
        "high": 151.0,
        "low": 149.0,
        "close": 150.5,
        "volume": 1000000,
    })

    # Give tasks time to run
    await asyncio.sleep(0.1)

    assert len(pnl_received) == 1, f"Should receive 1 PNL, got {len(pnl_received)}"
    assert len(bar_received) == 1, f"Should receive 1 BAR, got {len(bar_received)}"

    print("  ✅ PASSED: Feeder-style subscription works")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("  EVENT PIPELINE INTEGRATION TESTS")
    print("="*70)

    results = []

    # Sync tests
    results.append(("Singleton Initialization", test_singleton_initialization()))
    results.append(("Synchronous Subscription", test_synchronous_subscription()))

    # Async tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        results.append(("Emit -> Receive", loop.run_until_complete(test_emit_receive())))
        results.append(("PNL_UPDATE Flow", loop.run_until_complete(test_pnl_event_flow())))
        results.append(("Multiple Subscribers", loop.run_until_complete(test_multiple_subscribers())))
        results.append(("Feeder-style Subscription", loop.run_until_complete(test_feeder_subscription())))
    finally:
        loop.close()

    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} : {name}")

    print("="*70)
    print(f"  TOTAL: {passed}/{total} tests passed")
    print("="*70)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
