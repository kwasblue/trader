"""
App Types - Backward Compatibility Facade

This module re-exports types from core.contracts for backward compatibility.
New code should import directly from core.contracts.

Usage (legacy):
    from core.app_types import SignalContext, OrderResult

Preferred (new code):
    from core.contracts import SignalContext, OrderResult
"""

# Re-export everything from contracts for backward compatibility
from core.contracts.types import (
    SignalPayload,
    SignalContext,
    OrderResult,
    PositionView,
    BrokerSnapshot,
)
from core.contracts.events import BarPayload

# Legacy type aliases that were defined here
from typing import Literal

BarEventName = Literal["BAR_CREATED"]
SignalEventName = Literal["SIGNAL"]
TradeEventName = Literal["TRADE_EXECUTED"]
PnlEventName = Literal["PNL_UPDATE"]
SystemEventName = Literal["HEARTBEAT", "ERROR", "WARNING", "INFO"]

__all__ = [
    'BarPayload',
    'SignalPayload',
    'SignalContext',
    'OrderResult',
    'PositionView',
    'BrokerSnapshot',
    'BarEventName',
    'SignalEventName',
    'TradeEventName',
    'PnlEventName',
    'SystemEventName',
]
