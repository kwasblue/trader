"""
core/event_emitter.py

Dynamic mixin that exposes async emit_* methods which map directly
to events defined in core/events.py.
"""

import datetime
from typing import Any, Dict
from core.events import events

class EventEmitterMixin:
    def __init__(self, bus):
        self.bus = bus

    def _stamp(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def _with_timestamp(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {**payload, "timestamp": payload.get("timestamp", self._stamp())}

    def __getattr__(self, name: str):
        if name.startswith("emit_"):
            event_key = name[len("emit_"):].upper()
            event_const = f"EVENT_{event_key}"
            if hasattr(events, event_const):
                event_name = getattr(events, event_const)

                async def _emitter(payload: Dict[str, Any]):
                    payload = self._with_timestamp(payload)
                    await self.bus.emit(event_name, payload)

                return _emitter

        raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")
