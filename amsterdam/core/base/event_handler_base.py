"""
Event Handler - Pub/Sub event system for trading components

Provides asynchronous event-driven communication between components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Coroutine, Any, Dict, List
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

    def __init__(self):
        """Initialize event handler."""
        # Event name → list of callbacks
        self.listeners: Dict[str, List[Callable[[Event], Coroutine]]] = defaultdict(list)

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