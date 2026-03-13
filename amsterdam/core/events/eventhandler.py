# core/eventhandler.py
from __future__ import annotations

import asyncio
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable, Dict, List, Set
import weakref

from core.base.event_handler_base import Event, EventHandlerBase
from core.contracts.events import EVENT_SCHEMA_MAP, GuardrailPayload, EVENT_GUARDRAIL_TRIGGERED
from core.events.validation import validate_payload

# Optimization constants - tuned for 100-200 symbol scaling
MAX_CONCURRENT_HANDLERS = 200  # Increased from 50 for multi-symbol scaling
MAX_THREAD_WORKERS = 16  # Increased from 8 for parallel sync callbacks
MAX_QUEUE_SIZE = 10000  # Increased from 2000 for high-frequency events


class EventHandler(EventHandlerBase):
    """
    Optimized async-safe singleton event hub.

    Optimizations:
    - Bounded ThreadPoolExecutor for sync callbacks (prevents thread explosion)
    - Semaphore to limit concurrent async handlers
    - Task tracking for graceful shutdown
    - Early exit when no subscribers (skip validation overhead)
    - Reduced queue size for memory efficiency

    IMPORTANT: This is a true singleton - __init__ is guarded to prevent
    reinitializing the listeners dict on subsequent calls.
    """
    _instance: "EventHandler" | None = None
    _create_lock = threading.Lock()
    _initialized = False  # Guard against __init__ being called multiple times

    def __new__(cls):
        with cls._create_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self):
        # CRITICAL: Only initialize once! Python calls __init__ every time
        # the class is instantiated, even for singletons.
        if EventHandler._initialized:
            return

        # Call parent init (creates listeners dict)
        super().__init__()

        # Initialize our attributes
        self._queue = None
        self._runner = None
        # Optimization: bounded thread pool for sync callbacks
        self._executor = ThreadPoolExecutor(
            max_workers=MAX_THREAD_WORKERS,
            thread_name_prefix="event_handler"
        )
        # Optimization: limit concurrent async handlers
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_HANDLERS)
        # Track active tasks for graceful shutdown
        self._active_tasks: Set[asyncio.Task] = set()

        # Mark as initialized
        EventHandler._initialized = True
        self.logger.info(f"EventHandler singleton initialized, id={id(self)}")
        print(f"[EventHandler] Singleton initialized, id={id(self)}")

    def subscribe_sync(self, event_name: str, callback: Callable[[Event], Any]) -> None:
        """
        SYNCHRONOUS subscribe - use this when you need guaranteed immediate subscription.
        """
        cb_name = getattr(callback, '__name__', repr(callback))
        if callback not in self.listeners[event_name]:
            self.listeners[event_name].append(callback)
            self.logger.debug(f"[EventHandler] Subscribed to '{event_name}' -> {cb_name} (total: {len(self.listeners[event_name])})")
        else:
            self.logger.debug(f"[EventHandler] Duplicate subscription blocked for '{event_name}'")

    async def subscribe(self, event_name: str, callback: Callable[[Event], Any]) -> None:
        """
        Register a callback for an event. Callback can be async or sync.
        For guaranteed immediate subscription, use subscribe_sync() instead.
        """
        self.subscribe_sync(event_name, callback)

    def has_subscribers(self, event_name: str) -> bool:
        """Check if event has any subscribers (optimization for early exit)."""
        return bool(self.listeners.get(event_name))

    async def emit(self, event_name: str, payload: Any) -> None:
        """
        Emit an event asynchronously with optimized dispatch.

        Optimizations:
        - Early exit if no subscribers (skip validation overhead)
        - Bounded thread pool for sync callbacks
        - Semaphore-limited concurrency for async handlers
        - Task tracking for graceful shutdown
        """
        # Optimization: Early exit before validation if no listeners
        callbacks = self.listeners.get(event_name)
        if not callbacks:
            # No listeners for this event - silently skip
            return

        # Debug logging (commented out for production)
        # if event_name in ("PNL_UPDATE", "NEW_BAR"):
        #     self.logger.debug(f"[EventHandler] Emitting {event_name} to {len(callbacks)} listener(s)")

        # Schema validation (only if we have listeners)
        schema = EVENT_SCHEMA_MAP.get(event_name)
        if schema:
            try:
                validate_payload(payload, schema)
            except Exception as e:
                self.logger.error(f"[EventHandler] Invalid payload for {event_name}: {e}")
                return

        event = Event(event_name, payload)
        callbacks = list(callbacks)  # Copy to avoid mutation during iteration

        loop = asyncio.get_running_loop()

        for cb in callbacks:
            try:
                if inspect.iscoroutinefunction(cb):
                    # Optimization: Track task and use semaphore
                    task = loop.create_task(self._safe_async_call(cb, event))
                    self._active_tasks.add(task)
                    task.add_done_callback(self._active_tasks.discard)
                else:
                    # Optimization: Use bounded thread pool instead of default
                    loop.run_in_executor(self._executor, self._safe_sync_call, cb, event)
            except Exception as e:
                self.logger.exception(
                    f"[EventHandler] Failed scheduling callback {getattr(cb, '__name__', repr(cb))}: {e}"
                )

    async def _safe_async_call(self, cb: Callable, event: Event) -> None:
        """Execute async callback with semaphore-limited concurrency."""
        async with self._semaphore:
            try:
                await cb(event)
            except Exception as e:
                self.logger.exception(
                    f"[EventHandler] Error in async callback {getattr(cb, '__name__', repr(cb))}: {e}"
                )

    def _safe_sync_call(self, cb: Callable, event: Event) -> None:
        """Execute sync callback in thread pool (called from executor)."""
        try:
            cb(event)
        except Exception as e:
            self.logger.exception(
                f"[EventHandler] Error in sync callback {getattr(cb, '__name__', repr(cb))}: {e}"
            )

    async def shutdown(self) -> None:
        """Gracefully shutdown: wait for pending tasks and close executor."""
        # Wait for active async tasks
        if self._active_tasks:
            self.logger.info(f"[EventHandler] Waiting for {len(self._active_tasks)} pending tasks...")
            await asyncio.gather(*self._active_tasks, return_exceptions=True)

        # Shutdown thread pool
        self._executor.shutdown(wait=True, cancel_futures=False)
        self.logger.info("[EventHandler] Shutdown complete")


    def unsubscribe(self, event_name: str, callback: Callable[[Event], Any]) -> None:
        if callback in self.listeners[event_name]:
            self.listeners[event_name].remove(callback)
            self.logger.debug(f"[EventHandler] Unsubscribed {getattr(callback, '__name__', repr(callback))} from '{event_name}'")

    def get_event_names(self) -> list[str]:
        return sorted(self.listeners.keys())

    # --- Optional: decoupled publisher loop for backpressure / fire-and-forget ---

    async def start(self) -> None:
        """Start an internal consumer task to drain the publish() queue."""
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)  # Optimized size
        if self._runner is None or self._runner.done():
            self._runner = asyncio.create_task(self._consumer_loop())
            self.logger.info("[EventHandler] Dispatcher loop started")

    async def publish(self, event_name: str, payload: Any) -> None:
        """
        Fire-and-forget enqueue. Pair with start() once at app startup.
        """
        if self._queue is None:
            await self.start()
        await self._queue.put((event_name, payload))

    async def _consumer_loop(self) -> None:
        while True:
            event_name, payload = await self._queue.get()
            try:
                await self.emit(event_name, payload)
            finally:
                self._queue.task_done()

    async def emit_guardrail(self, guard_name: str, triggered: bool, message: str, value: float | None = None):
        """Emit a guardrail event (drawdown, risk limit, etc.)."""
        payload: GuardrailPayload = {
            "guard_name": guard_name,
            "triggered": triggered,
            "message": message,
            "value": value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self.emit(EVENT_GUARDRAIL_TRIGGERED, payload)

# --- Global Singleton Accessor ---
_global_event_handler: EventHandler | None = None

def get_event_handler() -> EventHandler:
    """Return the global EventHandler singleton."""
    global _global_event_handler
    if _global_event_handler is None:
        _global_event_handler = EventHandler()
    return _global_event_handler
