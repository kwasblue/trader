"""
Core Contracts Module

Single source of truth for data types, event schemas, and contracts
used across the trading system.

Usage:
    # Core types
    from core.contracts import SignalContext, OrderResult, PositionView, BrokerSnapshot

    # Event contracts
    from core.contracts import EVENT_NEW_BAR, EVENT_PNL_UPDATE
    from core.contracts import BarPayload, PnLPayload

    # All event names and schemas
    from core.contracts.events import EVENT_SCHEMA_MAP
"""

# Core types
# Event names (most commonly used)
# Event payloads (most commonly used)
from core.contracts.events import (
    EVENT_ALERT,
    EVENT_HEALTH_UPDATE,
    EVENT_NEW_BAR,
    EVENT_NEW_TRADE,
    EVENT_ORDER_STATUS,
    EVENT_PNL_UPDATE,
    EVENT_POSITION_UPDATE,
    EVENT_STRATEGY_SIGNAL,
    AlertPayload,
    BarPayload,
    HealthPayload,
    OrderStatusPayload,
    PnLPayload,
    PositionPayload,
    StrategySignalPayload,
    TradePayload,
)

# Meta types (for ML training data)
from core.contracts.meta_types import (
    TradeEntryContext,
    TradeExitContext,
)
from core.contracts.types import (
    BrokerSnapshot,
    OrderResult,
    PositionView,
    SignalContext,
)

__all__ = [
    # Core types
    "SignalContext",
    "OrderResult",
    "PositionView",
    "BrokerSnapshot",
    # Meta types
    "TradeEntryContext",
    "TradeExitContext",
    # Event names
    "EVENT_NEW_BAR",
    "EVENT_NEW_TRADE",
    "EVENT_ORDER_STATUS",
    "EVENT_POSITION_UPDATE",
    "EVENT_PNL_UPDATE",
    "EVENT_ALERT",
    "EVENT_HEALTH_UPDATE",
    "EVENT_STRATEGY_SIGNAL",
    # Event payloads
    "BarPayload",
    "TradePayload",
    "OrderStatusPayload",
    "PositionPayload",
    "PnLPayload",
    "AlertPayload",
    "HealthPayload",
    "StrategySignalPayload",
]
