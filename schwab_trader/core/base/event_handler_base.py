from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Callable, Coroutine, Any
from loggers.logger import Logger


class Event:
    def __init__(self, name: str, payload: Any):
        self.name = name
        self.payload = payload
        

class EventHandlerBase(ABC):
    def __init__(self, max_bar_window: int = 100):
        self.listeners = defaultdict(list)  # Event name → list of callbacks
        self.bar_windows = defaultdict(lambda: deque(maxlen=max_bar_window))  # Symbol → deque of bars
        self.current_day_bar = defaultdict(lambda: None)  # Symbol → latest bar
        self.logger = Logger("eventhandler.log", self.__class__.__name__).get_logger()

    @abstractmethod
    async def subscribe(self, event_name: str, callback: Callable[[Event], Coroutine]):
        """
        Register a coroutine callback to be called when an event with the given name is emitted.
        """
        pass

    @abstractmethod
    async def emit(self, event_name: str, payload: Any):
        """
        Emit an event and invoke all subscribed coroutines with the event payload.
        """
        pass

    @abstractmethod
    def unsubscribe(self, event_name: str, callback: Callable):
        """
        Remove a previously registered callback for a specific event.
        """
        pass

    @abstractmethod
    def get_event_names(self) -> list[str]:
        """
        Return a list of currently registered event names.
        """
        pass
