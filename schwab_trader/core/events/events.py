"""
Events Module - Backward Compatibility Facade

This module re-exports event definitions from core.contracts.events for
backward compatibility. New code should import directly from core.contracts.

Usage (legacy):
    from core.events.events import EVENT_NEW_BAR, BarPayload

Preferred (new code):
    from core.contracts import EVENT_NEW_BAR, BarPayload
    from core.contracts.events import EVENT_SCHEMA_MAP
"""

# Re-export all event names and payloads from contracts
from core.contracts.events import (
    # Event name constants
    EVENT_NEW_BAR,
    EVENT_NEW_TRADE,
    EVENT_ORDER_STATUS,
    EVENT_POSITION_UPDATE,
    EVENT_PNL_UPDATE,
    EVENT_ALERT,
    EVENT_HEALTH_UPDATE,
    EVENT_ORDER_QUEUE_UPDATE,
    EVENT_HALT_STATE,
    EVENT_COOLDOWN_STATE,
    EVENT_PRICE_UPDATE,
    EVENT_REGIME_UPDATE,
    EVENT_NEWS_UPDATE,
    EVENT_STRATEGY_SIGNAL,
    EVENT_PERFORMANCE_METRICS,
    EVENT_HEATMAP_UPDATE,
    EVENT_DISTRIBUTION_UPDATE,
    EVENT_TRADE_STATS,
    EVENT_REGIME_PERF,
    EVENT_HISTORY_UPDATE,
    EVENT_BENCHMARK_UPDATE,
    EVENT_REPLAY_FRAME,
    EVENT_LOG,
    EVENT_GUARDRAIL_TRIGGERED,
    EVENT_SESSION_UPDATE,
    EVENT_CONFIG_SNAPSHOT,
    EVENT_FLATTEN_ALL,
    EVENT_CANCEL_ALL,
    EVENT_FLATTEN_SYMBOL,
    EVENT_PLACE_ORDER,
    EVENT_SET_STRATEGY,
    EVENT_TOGGLE_PANIC,
    EVENT_HALTED,
    EVENT_MANUAL_ORDER,
    # Payload schemas
    BarPayload,
    PricePayload,
    RegimePayload,
    NewsPayload,
    TradePayload,
    OrderStatusPayload,
    PositionPayload,
    PnLPayload,
    PerformanceMetricsPayload,
    HeatmapPayload,
    DistributionPayload,
    TradeStatsPayload,
    RegimePerfPayload,
    HistoryPayload,
    BenchmarkPayload,
    ReplayFramePayload,
    AlertPayload,
    LogPayload,
    HealthPayload,
    GuardrailPayload,
    SessionPayload,
    ConfigSnapshotPayload,
    StrategySignalPayload,
    FlattenSymbolPayload,
    PlaceOrderPayload,
    SetStrategyPayload,
    # Schema map
    EVENT_SCHEMA_MAP,
)

__all__ = [
    # Event names
    'EVENT_NEW_BAR', 'EVENT_NEW_TRADE', 'EVENT_ORDER_STATUS',
    'EVENT_POSITION_UPDATE', 'EVENT_PNL_UPDATE', 'EVENT_ALERT',
    'EVENT_HEALTH_UPDATE', 'EVENT_ORDER_QUEUE_UPDATE', 'EVENT_HALT_STATE',
    'EVENT_COOLDOWN_STATE', 'EVENT_PRICE_UPDATE', 'EVENT_REGIME_UPDATE',
    'EVENT_NEWS_UPDATE', 'EVENT_STRATEGY_SIGNAL', 'EVENT_PERFORMANCE_METRICS',
    'EVENT_HEATMAP_UPDATE', 'EVENT_DISTRIBUTION_UPDATE', 'EVENT_TRADE_STATS',
    'EVENT_REGIME_PERF', 'EVENT_HISTORY_UPDATE', 'EVENT_BENCHMARK_UPDATE',
    'EVENT_REPLAY_FRAME', 'EVENT_LOG', 'EVENT_GUARDRAIL_TRIGGERED',
    'EVENT_SESSION_UPDATE', 'EVENT_CONFIG_SNAPSHOT', 'EVENT_FLATTEN_ALL',
    'EVENT_CANCEL_ALL', 'EVENT_FLATTEN_SYMBOL', 'EVENT_PLACE_ORDER',
    'EVENT_SET_STRATEGY', 'EVENT_TOGGLE_PANIC', 'EVENT_HALTED',
    'EVENT_MANUAL_ORDER',
    # Payload schemas
    'BarPayload', 'PricePayload', 'RegimePayload', 'NewsPayload',
    'TradePayload', 'OrderStatusPayload', 'PositionPayload', 'PnLPayload',
    'PerformanceMetricsPayload', 'HeatmapPayload', 'DistributionPayload',
    'TradeStatsPayload', 'RegimePerfPayload', 'HistoryPayload',
    'BenchmarkPayload', 'ReplayFramePayload', 'AlertPayload', 'LogPayload',
    'HealthPayload', 'GuardrailPayload', 'SessionPayload',
    'ConfigSnapshotPayload', 'StrategySignalPayload', 'FlattenSymbolPayload',
    'PlaceOrderPayload', 'SetStrategyPayload',
    # Schema map
    'EVENT_SCHEMA_MAP',
]
