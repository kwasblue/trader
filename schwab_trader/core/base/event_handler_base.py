"""
Event Handler - Pub/Sub event system for trading components

Provides asynchronous event-driven communication between components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Callable, Coroutine, Any, Dict, List, Optional, Deque
import asyncio
import logging

from loggers.logger import Logger

class Event:
    """
    Event object passed to subscribers.
    
    Contains the event name and payload data.
    """
    
    def __init__(self, name: str, payload: Any):
        """
        Initialize event.
        
        Args:
            name: Event name/type
            payload: Event data (can be dict, object, or any type)
        """
        self.name = name
        self.payload = payload
    
    def __repr__(self) -> str:
        return f"Event(name={self.name}, payload={type(self.payload).__name__})"


class EventHandlerBase(ABC):
    """
    Abstract base class for event handlers.
    
    Provides pub/sub event system for decoupled component communication.
    Components can subscribe to events and emit events without knowing
    about each other.
    
    Features:
    - Asynchronous event emission
    - Multiple subscribers per event
    - Bar window tracking for strategies
    - Thread-safe operations
    
    Example:
        handler = EventHandler()
        
        # Subscribe
        async def on_trade(event):
            print(f"Trade: {event.payload}")
        
        await handler.subscribe("trade_executed", on_trade)
        
        # Emit
        await handler.emit("trade_executed", {"symbol": "AAPL", "qty": 100})
    """
    
    def __init__(self, max_bar_window: int = 100):
        """
        Initialize event handler.
        
        Args:
            max_bar_window: Maximum number of bars to keep per symbol
        """
        # Event name → list of callbacks
        self.listeners: Dict[str, List[Callable[[Event], Coroutine]]] = defaultdict(list)
        
        # Symbol → deque of recent bars
        self.bar_windows: Dict[str, Deque[Any]] = defaultdict(
            lambda: deque(maxlen=max_bar_window)
        )
        
        # Symbol → latest bar
        self.current_day_bar: Dict[str, Optional[Any]] = defaultdict(lambda: None)
        
        # Logger - own file with propagation to app.log
        self.logger = Logger(
            log_file="event_handler.log",
            logger_name="EventHandler",
            propagate=True
        ).get_logger()
    
    # ========================================================================
    # ABSTRACT METHODS
    # ========================================================================
    
    @abstractmethod
    async def subscribe(
        self,
        event_name: str,
        callback: Callable[[Event], Coroutine[Any, Any, None]]
    ) -> None:
        """
        Register a callback for an event.
        
        The callback will be invoked asynchronously whenever the event is emitted.
        Multiple callbacks can be registered for the same event.
        
        Args:
            event_name: Name of event to listen for
            callback: Async function that receives Event object
            
        Example:
            async def handle_order(event: Event):
                order = event.payload
                print(f"Order: {order['symbol']}")
            
            await handler.subscribe("order_filled", handle_order)
        """
        pass
    
    @abstractmethod
    async def emit(self, event_name: str, payload: Any) -> None:
        """
        Emit an event to all subscribers.
        
        Invokes all registered callbacks for this event type asynchronously.
        If a callback raises an exception, it's logged but doesn't affect
        other callbacks.
        
        Args:
            event_name: Name of event to emit
            payload: Event data (dict, object, or any type)
            
        Example:
            await handler.emit("order_filled", {
                "symbol": "AAPL",
                "qty": 100,
                "price": 150.25
            })
        """
        pass
    
    @abstractmethod
    def unsubscribe(
        self,
        event_name: str,
        callback: Callable[[Event], Coroutine[Any, Any, None]]
    ) -> None:
        """
        Remove a callback from an event.
        
        Args:
            event_name: Event name
            callback: Callback function to remove (must be same object)
            
        Example:
            handler.unsubscribe("order_filled", handle_order)
        """
        pass
    
    @abstractmethod
    def get_event_names(self) -> List[str]:
        """
        Get list of events with active subscribers.
        
        Returns:
            List of event names that have registered callbacks
        """
        pass
    
    # ========================================================================
    # BAR TRACKING (Optional utility methods)
    # ========================================================================
    
    def add_bar(self, symbol: str, bar: Any) -> None:
        """
        Add a bar to the symbol's window.
        
        Args:
            symbol: Trading symbol
            bar: Bar data (dict or object)
        """
        self.bar_windows[symbol].append(bar)
        self.current_day_bar[symbol] = bar
    
    def get_bar_window(self, symbol: str, lookback: Optional[int] = None) -> List[Any]:
        """
        Get recent bars for a symbol.
        
        Args:
            symbol: Trading symbol
            lookback: Number of bars to return (None = all)
            
        Returns:
            List of recent bars (oldest to newest)
        """
        bars = list(self.bar_windows[symbol])
        if lookback is not None:
            return bars[-lookback:]
        return bars
    
    def get_current_bar(self, symbol: str) -> Optional[Any]:
        """
        Get most recent bar for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Latest bar or None
        """
        return self.current_day_bar[symbol]
    
    def clear_bar_window(self, symbol: str) -> None:
        """
        Clear bar history for a symbol.
        
        Args:
            symbol: Trading symbol
        """
        self.bar_windows[symbol].clear()
        self.current_day_bar[symbol] = None
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def subscriber_count(self, event_name: str) -> int:
        """
        Get number of subscribers for an event.
        
        Args:
            event_name: Event name
            
        Returns:
            Number of registered callbacks
        """
        return len(self.listeners.get(event_name, []))
    
    def has_subscribers(self, event_name: str) -> bool:
        """
        Check if event has any subscribers.
        
        Args:
            event_name: Event name
            
        Returns:
            True if event has subscribers
        """
        return self.subscriber_count(event_name) > 0
    
    def clear_all_subscribers(self) -> None:
        """Remove all event subscribers."""
        self.listeners.clear()
        self.logger.info("All event subscribers cleared")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(events={len(self.listeners)})"


# ============================================================================
# CONCRETE IMPLEMENTATION
# ============================================================================

class AsyncEventHandler(EventHandlerBase):
    """
    Concrete async event handler implementation.
    
    Provides full async pub/sub event system with error handling.
    """
    
    async def subscribe(
        self,
        event_name: str,
        callback: Callable[[Event], Coroutine[Any, Any, None]]
    ) -> None:
        """Subscribe to an event."""
        if not asyncio.iscoroutinefunction(callback):
            raise TypeError(f"Callback must be async function, got {type(callback)}")
        
        self.listeners[event_name].append(callback)
        self.logger.debug(
            f"Subscribed to '{event_name}' "
            f"(total subscribers: {len(self.listeners[event_name])})"
        )
    
    async def emit(self, event_name: str, payload: Any) -> None:
        """Emit an event to all subscribers."""
        callbacks = self.listeners.get(event_name, [])
        
        if not callbacks:
            self.logger.debug(f"No subscribers for event '{event_name}'")
            return
        
        event = Event(event_name, payload)
        
        # Call all callbacks, catching exceptions
        for callback in callbacks:
            try:
                await callback(event)
            except Exception as e:
                self.logger.error(
                    f"Error in callback for '{event_name}': {e}",
                    exc_info=True
                )
    
    def unsubscribe(
        self,
        event_name: str,
        callback: Callable[[Event], Coroutine[Any, Any, None]]
    ) -> None:
        """Unsubscribe from an event."""
        if event_name in self.listeners:
            try:
                self.listeners[event_name].remove(callback)
                self.logger.debug(f"Unsubscribed from '{event_name}'")
                
                # Clean up empty lists
                if not self.listeners[event_name]:
                    del self.listeners[event_name]
            except ValueError:
                self.logger.warning(
                    f"Callback not found for '{event_name}'"
                )
    
    def get_event_names(self) -> List[str]:
        """Get list of event names with subscribers."""
        return list(self.listeners.keys())