"""
Tests for Event System

Coverage:
- Event emission
- Event subscription
- Async handler support
- Sync handler support
- Event validation
- Circuit breaker in event handler
"""

import asyncio

import pytest

from core.events.eventhandler import MAX_CONCURRENT_HANDLERS, EventHandler, get_event_handler
from core.events.events import (
    EVENT_NEW_BAR,
    EVENT_ORDER_STATUS,
    EVENT_PNL_UPDATE,
)


@pytest.fixture
def event_handler():
    """Create a fresh EventHandler for testing."""
    # Reset singleton for testing - must reset BOTH flags!
    EventHandler._instance = None
    EventHandler._initialized = False
    handler = EventHandler()
    return handler


@pytest.fixture
def reset_singleton():
    """Reset the EventHandler singleton after each test."""
    yield
    EventHandler._instance = None
    EventHandler._initialized = False  # Must reset this too!


class TestEventHandlerBasics:
    """Test basic event handler functionality."""

    @pytest.mark.asyncio
    async def test_subscribe_and_emit(self, event_handler, reset_singleton):
        """Test subscribing to and emitting events."""
        received = []

        async def handler(event):
            received.append(event.payload)

        await event_handler.subscribe("test_event", handler)
        await event_handler.emit("test_event", {"message": "hello"})

        # Wait for async processing
        await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0]["message"] == "hello"

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, event_handler, reset_singleton):
        """Test multiple subscribers to same event."""
        received1 = []
        received2 = []

        async def handler1(event):
            received1.append(event.payload)

        async def handler2(event):
            received2.append(event.payload)

        await event_handler.subscribe("test_event", handler1)
        await event_handler.subscribe("test_event", handler2)
        await event_handler.emit("test_event", {"data": "test"})

        await asyncio.sleep(0.1)

        assert len(received1) == 1
        assert len(received2) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_handler, reset_singleton):
        """Test unsubscribing from events."""
        received = []

        async def handler(event):
            received.append(event.payload)

        await event_handler.subscribe("test_event", handler)
        event_handler.unsubscribe("test_event", handler)
        await event_handler.emit("test_event", {"data": "test"})

        await asyncio.sleep(0.1)

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_no_subscribers(self, event_handler, reset_singleton):
        """Test emitting event with no subscribers."""
        # Should not raise an error
        await event_handler.emit("nonexistent_event", {"data": "test"})

    @pytest.mark.asyncio
    async def test_has_subscribers(self, event_handler, reset_singleton):
        """Test checking if event has subscribers."""

        async def handler(event):
            pass

        assert event_handler.has_subscribers("test_event") is False

        await event_handler.subscribe("test_event", handler)

        assert event_handler.has_subscribers("test_event") is True


class TestSyncHandlers:
    """Test sync handler support."""

    @pytest.mark.asyncio
    async def test_sync_handler(self, event_handler, reset_singleton):
        """Test that sync handlers are called correctly."""
        received = []

        def sync_handler(event):
            received.append(event.payload)

        await event_handler.subscribe("test_event", sync_handler)
        await event_handler.emit("test_event", {"data": "sync_test"})

        # Wait for thread pool execution
        await asyncio.sleep(0.2)

        assert len(received) == 1
        assert received[0]["data"] == "sync_test"

    @pytest.mark.asyncio
    async def test_mixed_handlers(self, event_handler, reset_singleton):
        """Test mixed async and sync handlers."""
        async_received = []
        sync_received = []

        async def async_handler(event):
            async_received.append(event.payload)

        def sync_handler(event):
            sync_received.append(event.payload)

        await event_handler.subscribe("test_event", async_handler)
        await event_handler.subscribe("test_event", sync_handler)
        await event_handler.emit("test_event", {"data": "mixed"})

        await asyncio.sleep(0.2)

        assert len(async_received) == 1
        assert len(sync_received) == 1


class TestEventHandlerErrors:
    """Test error handling in event system."""

    @pytest.mark.asyncio
    async def test_handler_exception_doesnt_break_others(self, event_handler, reset_singleton):
        """Test that one handler's exception doesn't affect others."""
        received = []

        async def bad_handler(event):
            raise RuntimeError("Intentional error")

        async def good_handler(event):
            received.append(event.payload)

        await event_handler.subscribe("test_event", bad_handler)
        await event_handler.subscribe("test_event", good_handler)
        await event_handler.emit("test_event", {"data": "test"})

        await asyncio.sleep(0.1)

        # Good handler should still receive the event
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_duplicate_subscription_ignored(self, event_handler, reset_singleton):
        """Test that duplicate subscriptions are ignored."""
        count = 0

        async def handler(event):
            nonlocal count
            count += 1

        await event_handler.subscribe("test_event", handler)
        await event_handler.subscribe("test_event", handler)  # Duplicate

        await event_handler.emit("test_event", {"data": "test"})
        await asyncio.sleep(0.1)

        # Should only be called once
        assert count == 1


class TestEventPayloads:
    """Test event payload handling."""

    @pytest.mark.asyncio
    async def test_bar_event_payload(self, event_handler, reset_singleton):
        """Test bar event payload structure."""
        received = []

        async def handler(event):
            received.append(event)

        await event_handler.subscribe(EVENT_NEW_BAR, handler)
        await event_handler.emit(
            EVENT_NEW_BAR,
            {
                "symbol": "AAPL",
                "open": 150.0,
                "high": 152.0,
                "low": 149.0,
                "close": 151.0,
                "volume": 1000000,
                "timestamp": "2024-01-01T10:00:00Z",
            },
        )

        await asyncio.sleep(0.1)

        assert len(received) == 1
        payload = received[0].payload
        assert payload["symbol"] == "AAPL"
        assert payload["close"] == 151.0

    @pytest.mark.asyncio
    async def test_pnl_event_payload(self, event_handler, reset_singleton):
        """Test P&L event payload structure."""
        received = []

        async def handler(event):
            received.append(event)

        await event_handler.subscribe(EVENT_PNL_UPDATE, handler)
        await event_handler.emit(
            EVENT_PNL_UPDATE,
            {
                "portfolio_value": 100000.0,
                "equity_curve": [100000.0, 100500.0, 100800.0],
                "unrealized": 500.0,
                "realized": 1000.0,
                "drawdown": 0.02,
                "timestamp": "2024-01-01T10:00:00Z",
                "cash": 50000.0,
                "buying_power": 100000.0,
            },
        )

        await asyncio.sleep(0.1)

        assert len(received) == 1
        payload = received[0].payload
        assert payload["portfolio_value"] == 100000.0

    @pytest.mark.asyncio
    async def test_order_status_event(self, event_handler, reset_singleton):
        """Test order status event payload."""
        received = []

        async def handler(event):
            received.append(event)

        await event_handler.subscribe(EVENT_ORDER_STATUS, handler)
        await event_handler.emit(
            EVENT_ORDER_STATUS,
            {
                "order_id": "ORD123",
                "symbol": "AAPL",
                "status": "filled",
                "filled_qty": 10,
                "avg_price": 150.0,
                "timestamp": "2024-01-01T10:00:00Z",
            },
        )

        await asyncio.sleep(0.1)

        assert len(received) == 1
        payload = received[0].payload
        assert payload["order_id"] == "ORD123"
        assert payload["status"] == "filled"


class TestEventHandlerQueue:
    """Test queue-based event handling."""

    @pytest.mark.asyncio
    async def test_publish_queue(self, event_handler, reset_singleton):
        """Test fire-and-forget publish via queue."""
        received = []

        async def handler(event):
            received.append(event.payload)

        await event_handler.start()
        await event_handler.subscribe("test_event", handler)
        await event_handler.publish("test_event", {"data": "queued"})

        # Wait for queue processing
        await asyncio.sleep(0.2)

        assert len(received) == 1
        assert received[0]["data"] == "queued"


class TestGlobalSingleton:
    """Test global singleton accessor."""

    def test_get_event_handler_returns_same_instance(self, reset_singleton):
        """Test that get_event_handler returns same instance."""
        handler1 = get_event_handler()
        handler2 = get_event_handler()

        assert handler1 is handler2


class TestEventHandlerShutdown:
    """Test graceful shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown(self, event_handler, reset_singleton):
        """Test graceful shutdown of event handler."""

        async def slow_handler(event):
            await asyncio.sleep(0.5)

        await event_handler.subscribe("test_event", slow_handler)
        await event_handler.emit("test_event", {"data": "test"})

        # Shutdown should wait for pending tasks
        await event_handler.shutdown()

        # No assertion needed - just verify it doesn't hang or error


class TestEventHandlerConcurrency:
    """Test concurrent event handling."""

    @pytest.mark.asyncio
    async def test_concurrent_emissions(self, event_handler, reset_singleton):
        """Test handling many concurrent emissions."""
        received = []

        async def handler(event):
            received.append(event.payload)

        await event_handler.subscribe("test_event", handler)

        # Emit many events concurrently
        tasks = [event_handler.emit("test_event", {"index": i}) for i in range(100)]
        await asyncio.gather(*tasks)

        await asyncio.sleep(0.5)

        assert len(received) == 100

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self, event_handler, reset_singleton):
        """Test that semaphore limits concurrent handler execution."""
        active_count = 0
        max_active = 0

        async def slow_handler(event):
            nonlocal active_count, max_active
            active_count += 1
            max_active = max(max_active, active_count)
            await asyncio.sleep(0.1)
            active_count -= 1

        await event_handler.subscribe("test_event", slow_handler)

        # Emit many events
        tasks = [event_handler.emit("test_event", {"index": i}) for i in range(100)]
        await asyncio.gather(*tasks)

        await asyncio.sleep(2)

        # Max active should be limited by semaphore
        assert max_active <= MAX_CONCURRENT_HANDLERS
