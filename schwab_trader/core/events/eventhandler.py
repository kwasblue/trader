# core/eventhandler.py
from __future__ import annotations

import asyncio
import inspect
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable, Dict, List

from core.base.event_handler_base import Event, EventHandlerBase
from core.events.events import EVENT_SCHEMA_MAP, GuardrailPayload, EVENT_GUARDRAIL_TRIGGERED
from core.events.validation import validate_payload 


class EventHandler(EventHandlerBase):
    """
    Async-safe singleton event hub that accepts both async and sync callbacks.
    - subscribe(): register a callback for an event name
    - emit(): fire an event and await all handlers (run sync callbacks in executor)
    - unsubscribe(): remove a previously registered callback
    - get_event_names(): list all events with subscribers
    - publish()/start(): optional queue-based publisher for decoupled emission
    """
    _instance: "EventHandler" | None = None
    _create_lock = threading.Lock()

    def __new__(cls):
        with cls._create_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                super(EventHandler, cls._instance).__init__()
                cls._instance._queue = None
                cls._instance._runner = None
            return cls._instance

    async def subscribe(self, event_name: str, callback: Callable[[Event], Any]) -> None:
        """
        Register a callback for an event. Callback can be async or sync.
        """
        if callback not in self.listeners[event_name]:
            self.listeners[event_name].append(callback)
            self.logger.debug(f"[EventHandler] Subscribed to '{event_name}' -> {getattr(callback, '__name__', repr(callback))}")
        else:
            self.logger.debug(f"[EventHandler] Callback already subscribed to '{event_name}'")

    # async def emit(self, event_name: str, payload: Any) -> None:
    #     """
    #     Emit an event and await all callbacks.
    #     Sync callbacks are offloaded to the executor.
    #     """
    #     # Schema validation (if defined)
    #     schema = EVENT_SCHEMA_MAP.get(event_name)
    #     if schema:
    #         try:
    #             validate_payload(payload, schema)
    #         except Exception as e:
    #             self.logger.error(f"[EventHandler] Invalid payload for {event_name}: {e}")
    #             return  # or raise if you want hard failure

    #     event = Event(event_name, payload)
    #     callbacks = list(self.listeners.get(event_name, []))  # copy
    #     if not callbacks:
    #         self.logger.debug(f"[EventHandler] Emit '{event_name}' (no listeners)")
    #         return

    #     self.logger.debug(
    #         f"[EventHandler] Emit '{event_name}' to {len(callbacks)} listener(s) | Payload: {payload}"
    #     )
    #     loop = asyncio.get_running_loop()
    #     tasks: List[Awaitable[Any]] = []

    #     for cb in callbacks:
    #         try:
    #             if inspect.iscoroutinefunction(cb):
    #                 tasks.append(asyncio.create_task(cb(event)))
    #             else:
    #                 tasks.append(loop.run_in_executor(None, cb, event))
    #         except Exception as e:
    #             self.logger.exception(
    #                 f"[EventHandler] Failed scheduling callback {getattr(cb, '__name__', repr(cb))}: {e}"
    #             )

    #     if tasks:
    #         results = await asyncio.gather(*tasks, return_exceptions=True)
    #         for cb, res in zip(callbacks, results):
    #             if isinstance(res, Exception):
    #                 self.logger.exception(
    #                     f"[EventHandler] Error in callback {getattr(cb, '__name__', repr(cb))} for '{event_name}': {res}"
    #             )
    
    async def emit(self, event_name: str, payload: Any) -> None:
        """
        Emit an event asynchronously and dispatch all listeners concurrently.
        Sync callbacks are offloaded to a thread executor.
        This version is fully non-blocking — each callback runs as its own task.
        """

        # --- Schema validation (optional but safe) ---
        schema = EVENT_SCHEMA_MAP.get(event_name)
        if schema:
            try:
                validate_payload(payload, schema)
            except Exception as e:
                self.logger.error(f"[EventHandler] Invalid payload for {event_name}: {e}")
                return

        # --- Build event object ---
        event = Event(event_name, payload)
        callbacks = list(self.listeners.get(event_name, []))
        if not callbacks:
            self.logger.debug(f"[EventHandler] Emit '{event_name}' (no listeners)")
            return

        self.logger.debug(f"[EventHandler] Emit '{event_name}' -> {len(callbacks)} listener(s)")

        loop = asyncio.get_running_loop()

        # --- Dispatch each callback without awaiting (non-blocking fan-out) ---
        for cb in callbacks:
            try:
                if inspect.iscoroutinefunction(cb):
                    loop.create_task(self._safe_call(cb, event))
                else:
                    loop.run_in_executor(None, self._safe_call, cb, event)
            except Exception as e:
                self.logger.exception(
                    f"[EventHandler] Failed scheduling callback {getattr(cb, '__name__', repr(cb))}: {e}"
                )

    async def _safe_call(self, cb: Callable, event: Event):
        """Wrapper to isolate callback errors so one bad listener doesn’t crash emit()."""
        try:
            if inspect.iscoroutinefunction(cb):
                await cb(event)
            else:
                cb(event)
        except Exception as e:
            self.logger.exception(
                f"[EventHandler] Error in callback {getattr(cb, '__name__', repr(cb))}: {e}"
            )


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
            self._queue = asyncio.Queue(maxsize=10_000)
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

    async def emit_guardrail(event_handler: EventHandler, guard_name: str, triggered: bool, message: str, value: float | None = None):
        payload: GuardrailPayload = {
            "guard_name": guard_name,
            "triggered": triggered,
            "message": message,
            "value": value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await event_handler.emit(EVENT_GUARDRAIL_TRIGGERED, payload)

# --- Global Singleton Accessor ---
_global_event_handler: EventHandler | None = None

def get_event_handler() -> EventHandler:
    """Return the global EventHandler singleton."""
    global _global_event_handler
    if _global_event_handler is None:
        _global_event_handler = EventHandler()
    return _global_event_handler
