# utils/event_handler.py

from collections import defaultdict
from threading import Lock
from collections import defaultdict, deque

class Event:
    def __init__(self, name, payload):
        self.name = name
        self.payload = payload


class EventHandler:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EventHandler, cls).__new__(cls)
                cls._instance.listeners = defaultdict(list)
                cls._instance.bar_windows = defaultdict(deque)  # ADD THIS
            return cls._instance

    def subscribe(self, event_name, callback):
        print(f"[EventHandler] Subscribed to '{event_name}'")
        self.listeners[event_name].append(callback)

    def emit(self, event_name, payload):
        event = Event(event_name, payload)
        print(f"[EventHandler] Emitting '{event_name}' with payload: {payload}")
        for callback in self.listeners[event_name]:
            callback(event)
