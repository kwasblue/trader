"""
===============================================================================
DEPRECATED MODULE - DO NOT USE
===============================================================================

This module is DEPRECATED and exists only for backwards compatibility.
It will be removed in a future version.

MIGRATION GUIDE:
----------------
OLD (deprecated):
    from core.events.events import EVENT_NEW_BAR, BarPayload

NEW (recommended):
    from core.contracts.events import EVENT_NEW_BAR, BarPayload
    # Or use the contracts __init__ for common imports:
    from core.contracts import EVENT_NEW_BAR, BarPayload

All event definitions have been moved to core.contracts.events.

===============================================================================
"""

import warnings

# Emit deprecation warning when module is imported
warnings.warn(
    "\n"
    "=" * 70 + "\n"
    "DEPRECATION WARNING: core.events.events is deprecated!\n"
    "=" * 70 + "\n"
    "Use 'from core.contracts.events import ...' instead.\n"
    "This module will be removed in a future version.\n"
    "=" * 70,
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all event names and payloads from contracts
from core.contracts.events import (
    EVENT_ALERT,
    EVENT_BENCHMARK_UPDATE,
    EVENT_CANCEL_ALL,
    EVENT_CONFIG_SNAPSHOT,
    EVENT_COOLDOWN_STATE,
    EVENT_DISTRIBUTION_UPDATE,
    EVENT_FLATTEN_ALL,
    EVENT_FLATTEN_SYMBOL,
    EVENT_GUARDRAIL_TRIGGERED,
    EVENT_HALT_STATE,
    EVENT_HALTED,
    EVENT_HEALTH_UPDATE,
    EVENT_HEATMAP_UPDATE,
    EVENT_HISTORY_UPDATE,
    EVENT_LOG,
    EVENT_MANUAL_ORDER,
    # Event name constants
    EVENT_NEW_BAR,
    EVENT_NEW_TRADE,
    EVENT_NEWS_UPDATE,
    EVENT_ORDER_QUEUE_UPDATE,
    EVENT_ORDER_STATUS,
    EVENT_PERFORMANCE_METRICS,
    EVENT_PLACE_ORDER,
    EVENT_PNL_UPDATE,
    EVENT_POSITION_UPDATE,
    EVENT_PRICE_UPDATE,
    EVENT_REGIME_PERF,
    EVENT_REGIME_UPDATE,
    EVENT_REPLAY_FRAME,
    # Schema map
    EVENT_SCHEMA_MAP,
    EVENT_SESSION_UPDATE,
    EVENT_SET_STRATEGY,
    EVENT_STRATEGY_SIGNAL,
    EVENT_TOGGLE_PANIC,
    EVENT_TRADE_STATS,
    AlertPayload,
    # Payload schemas
    BarPayload,
    BenchmarkPayload,
    CancelAllPayload,
    ConfigSnapshotPayload,
    CooldownStatePayload,
    DistributionPayload,
    # Additional command payloads
    FlattenAllPayload,
    FlattenSymbolPayload,
    GuardrailPayload,
    HaltedPayload,
    HaltStatePayload,
    HealthPayload,
    HeatmapPayload,
    HistoryPayload,
    LogPayload,
    ManualOrderPayload,
    NewsPayload,
    OrderQueueUpdatePayload,
    OrderStatusPayload,
    PerformanceMetricsPayload,
    PlaceOrderPayload,
    PnLPayload,
    PositionPayload,
    PricePayload,
    RegimePayload,
    RegimePerfPayload,
    ReplayFramePayload,
    SessionPayload,
    SetStrategyPayload,
    StrategySignalPayload,
    TogglePanicPayload,
    TradePayload,
    TradeStatsPayload,
)

__all__ = [
    # Event names
    "EVENT_NEW_BAR",
    "EVENT_NEW_TRADE",
    "EVENT_ORDER_STATUS",
    "EVENT_POSITION_UPDATE",
    "EVENT_PNL_UPDATE",
    "EVENT_ALERT",
    "EVENT_HEALTH_UPDATE",
    "EVENT_ORDER_QUEUE_UPDATE",
    "EVENT_HALT_STATE",
    "EVENT_COOLDOWN_STATE",
    "EVENT_PRICE_UPDATE",
    "EVENT_REGIME_UPDATE",
    "EVENT_NEWS_UPDATE",
    "EVENT_STRATEGY_SIGNAL",
    "EVENT_PERFORMANCE_METRICS",
    "EVENT_HEATMAP_UPDATE",
    "EVENT_DISTRIBUTION_UPDATE",
    "EVENT_TRADE_STATS",
    "EVENT_REGIME_PERF",
    "EVENT_HISTORY_UPDATE",
    "EVENT_BENCHMARK_UPDATE",
    "EVENT_REPLAY_FRAME",
    "EVENT_LOG",
    "EVENT_GUARDRAIL_TRIGGERED",
    "EVENT_SESSION_UPDATE",
    "EVENT_CONFIG_SNAPSHOT",
    "EVENT_FLATTEN_ALL",
    "EVENT_CANCEL_ALL",
    "EVENT_FLATTEN_SYMBOL",
    "EVENT_PLACE_ORDER",
    "EVENT_SET_STRATEGY",
    "EVENT_TOGGLE_PANIC",
    "EVENT_HALTED",
    "EVENT_MANUAL_ORDER",
    # Payload schemas
    "BarPayload",
    "PricePayload",
    "RegimePayload",
    "NewsPayload",
    "TradePayload",
    "OrderStatusPayload",
    "PositionPayload",
    "PnLPayload",
    "PerformanceMetricsPayload",
    "HeatmapPayload",
    "DistributionPayload",
    "TradeStatsPayload",
    "RegimePerfPayload",
    "HistoryPayload",
    "BenchmarkPayload",
    "ReplayFramePayload",
    "AlertPayload",
    "LogPayload",
    "HealthPayload",
    "GuardrailPayload",
    "SessionPayload",
    "ConfigSnapshotPayload",
    "StrategySignalPayload",
    "FlattenSymbolPayload",
    "PlaceOrderPayload",
    "SetStrategyPayload",
    # Additional command payloads
    "FlattenAllPayload",
    "CancelAllPayload",
    "TogglePanicPayload",
    "HaltedPayload",
    "HaltStatePayload",
    "CooldownStatePayload",
    "OrderQueueUpdatePayload",
    "ManualOrderPayload",
    # Schema map
    "EVENT_SCHEMA_MAP",
]
