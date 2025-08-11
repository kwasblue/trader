# core/eventhandler.py
from __future__ import annotations

import asyncio
import inspect
import threading
from typing import Any, Callable, Awaitable, Dict, List

from core.base.event_handler_base import Event, EventHandlerBase


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
        # Use a threading lock here (can't await in __new__)
        with cls._create_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                # Initialize the base once (sets listeners, logger, etc.)
                super(EventHandler, cls._instance).__init__()
                # Lazy attrs for publisher loop
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

    async def emit(self, event_name: str, payload: Any) -> None:
        """
        Emit an event and await all callbacks. Sync callbacks are offloaded
        to the default executor so the event loop doesn't block.
        """
        event = Event(event_name, payload)
        callbacks = list(self.listeners.get(event_name, []))  # copy to avoid mutation during iteration
        if not callbacks:
            self.logger.debug(f"[EventHandler] Emit '{event_name}' (no listeners)")
            return

        self.logger.debug(f"[EventHandler] Emit '{event_name}' to {len(callbacks)} listener(s) | Payload: {payload}")
        loop = asyncio.get_running_loop()
        tasks: List[Awaitable[Any]] = []

        for cb in callbacks:
            try:
                if inspect.iscoroutinefunction(cb):
                    tasks.append(asyncio.create_task(cb(event)))
                else:
                    # Run sync handlers in a threadpool
                    tasks.append(loop.run_in_executor(None, cb, event))
            except Exception as e:
                self.logger.exception(f"[EventHandler] Failed scheduling callback {getattr(cb, '__name__', repr(cb))}: {e}")

        if not tasks:
            return

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for cb, res in zip(callbacks, results):
            if isinstance(res, Exception):
                self.logger.exception(f"[EventHandler] Error in callback {getattr(cb, '__name__', repr(cb))} for '{event_name}': {res}")

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


