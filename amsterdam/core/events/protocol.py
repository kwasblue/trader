"""
core/event_protocols.py

Dynamic EventEmitterProtocol that provides emit_* methods
for all events in EVENT_SCHEMA_MAP. These methods are not
type-enforced at compile time, but payloads will be validated
at runtime using isinstance() checks.
"""

from typing import Any, Protocol

from core.contracts.events import EVENT_SCHEMA_MAP


def _make_protocol() -> type[Any]:
    namespace: dict[str, Any] = {}

    for event_name in EVENT_SCHEMA_MAP.keys():
        method_name = "emit_" + event_name.lower()

        async def _template(self, payload: dict) -> None: ...

        _template.__name__ = method_name
        namespace[method_name] = _template

    return type("EventEmitterProtocol", (Protocol,), namespace)


EventEmitterProtocol = _make_protocol()
