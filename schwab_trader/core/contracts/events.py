"""
Event Contracts Module

Canonical event definitions for the trading system. Contains:
- Event name constants
- TypedDict payload schemas
- Event schema mapping

Usage:
    from core.contracts.events import EVENT_NEW_BAR, BarPayload
    from core.contracts.events import EVENT_SCHEMA_MAP
"""
from typing import TypedDict, Literal, Optional, List, Dict, Type, Any


# =============================================================================
# Event Name Constants
# =============================================================================

# --- Backend -> GUI: Market data & execution ---
EVENT_NEW_BAR = "NEW_BAR"
EVENT_NEW_TRADE = "NEW_TRADE"
EVENT_ORDER_STATUS = "ORDER_STATUS"
EVENT_POSITION_UPDATE = "POSITION_UPDATE"
EVENT_PNL_UPDATE = "PNL_UPDATE"

# --- Backend -> GUI: Monitoring & alerts ---
EVENT_ALERT = "ALERT"
EVENT_HEALTH_UPDATE = "HEALTH_UPDATE"
EVENT_ORDER_QUEUE_UPDATE = "ORDER_QUEUE_UPDATE"
EVENT_HALT_STATE = "HALT_STATE"
EVENT_COOLDOWN_STATE = "COOLDOWN_STATE"

# --- Backend -> GUI: Market context ---
EVENT_PRICE_UPDATE = "PRICE_UPDATE"
EVENT_REGIME_UPDATE = "REGIME_UPDATE"
EVENT_NEWS_UPDATE = "NEWS_UPDATE"

# --- Backend -> GUI: Performance & analytics ---
EVENT_STRATEGY_SIGNAL = "STRATEGY_SIGNAL"
EVENT_PERFORMANCE_METRICS = "PERFORMANCE_METRICS"
EVENT_HEATMAP_UPDATE = "HEATMAP_UPDATE"
EVENT_DISTRIBUTION_UPDATE = "DISTRIBUTION_UPDATE"
EVENT_TRADE_STATS = "TRADE_STATS"
EVENT_REGIME_PERF = "REGIME_PERF"

# --- Backend -> GUI: History & replay ---
EVENT_HISTORY_UPDATE = "HISTORY_UPDATE"
EVENT_BENCHMARK_UPDATE = "BENCHMARK_UPDATE"
EVENT_REPLAY_FRAME = "REPLAY_FRAME"

# --- Backend -> GUI: Logging & ops ---
EVENT_LOG = "LOG"
EVENT_GUARDRAIL_TRIGGERED = "GUARDRAIL_TRIGGERED"
EVENT_SESSION_UPDATE = "SESSION_UPDATE"
EVENT_CONFIG_SNAPSHOT = "CONFIG_SNAPSHOT"

# --- GUI -> Backend: Commands ---
EVENT_FLATTEN_ALL = "FLATTEN_ALL"
EVENT_CANCEL_ALL = "CANCEL_ALL"
EVENT_FLATTEN_SYMBOL = "FLATTEN_SYMBOL"
EVENT_PLACE_ORDER = "PLACE_ORDER"
EVENT_SET_STRATEGY = "SET_STRATEGY"
EVENT_TOGGLE_PANIC = "TOGGLE_PANIC"
EVENT_HALTED = "HALTED"
EVENT_MANUAL_ORDER = "MANUAL_ORDER"


# =============================================================================
# Payload Schemas (TypedDicts)
# =============================================================================

# --- Market data ---
class BarPayload(TypedDict):
    """OHLCV bar data."""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: str  # ISO8601


class PricePayload(TypedDict):
    """Price update with moving averages."""
    symbol: str
    price: float
    ma20: Optional[float]
    ma50: Optional[float]
    timestamp: str


class RegimePayload(TypedDict):
    """Market regime classification."""
    symbol: str
    volatility: str  # low_volatility | normal | high_volatility
    trend: str  # bullish | bearish | sideways
    market: str  # bull | bear | neutral
    timestamp: str


class NewsPayload(TypedDict):
    """News event with sentiment."""
    headline: str
    source: str
    sentiment: Optional[Literal["positive", "negative", "neutral"]]
    timestamp: str


# --- Execution / trades ---
class TradePayload(TypedDict):
    """Trade execution details."""
    symbol: str
    side: Literal["buy", "sell", "long", "short"]
    qty: float
    price: float
    timestamp: str
    pnl: Optional[float]


class OrderStatusPayload(TypedDict):
    """Order status update."""
    order_id: str
    symbol: str
    status: Literal["submitted", "filled", "canceled", "rejected"]
    filled_qty: float
    avg_price: Optional[float]
    timestamp: str


class PositionPayload(TypedDict, total=False):
    """Position update with extended fields."""
    symbol: str
    qty: float
    avg_price: float
    unrealized: float
    realized: float
    timestamp: str
    # Extended fields for GUI display
    side: str  # "long" or "short"
    last: float  # Last/current price
    avg: float  # Alias for avg_price
    unreal: float  # Alias for unrealized
    market_value: float  # Position market value


# --- Performance ---
class PnLPayload(TypedDict, total=False):
    """Portfolio P&L update."""
    portfolio_value: float
    equity_curve: List[float]
    unrealized: float
    realized: float
    drawdown: float
    timestamp: str
    cash: float
    buying_power: float


class PerformanceMetricsPayload(TypedDict):
    """Performance metrics snapshot."""
    sharpe: float
    sortino: float
    kelly: float
    max_dd: float
    hit_rate: float
    avg_win: float
    avg_loss: float
    timestamp: str


class HeatmapPayload(TypedDict):
    """Risk heatmap data."""
    data: List[List[float]]  # 2D matrix for risk vs exposure
    timestamp: str


class DistributionPayload(TypedDict):
    """Return distribution data."""
    bins: List[float]
    counts: List[int]
    timestamp: str


class TradeStatsPayload(TypedDict):
    """Trade statistics."""
    durations: List[float]
    win_streak: int
    loss_streak: int
    timestamp: str


class RegimePerfPayload(TypedDict):
    """Performance by market regime."""
    regimes: Dict[str, float]  # {"low_volatility": equity, ...}
    timestamp: str


# --- History / Replay ---
class HistoryPayload(TypedDict):
    """Historical P&L by day."""
    pnl_by_day: Dict[str, float]  # date -> pnl
    timestamp: str


class BenchmarkPayload(TypedDict):
    """Benchmark comparison."""
    equity_curve: List[float]
    benchmark_curve: List[float]
    timestamp: str


class ReplayFramePayload(TypedDict):
    """Replay frame for backtesting visualization."""
    frame_idx: int
    equity: float
    timestamp: str


# --- Alerts & logs ---
class AlertPayload(TypedDict):
    """Alert notification."""
    level: Literal["info", "warning", "error", "critical"]
    message: str
    symbol: Optional[str]
    timestamp: str


class LogPayload(TypedDict):
    """Log message."""
    message: str
    level: str
    timestamp: str


class HealthPayload(TypedDict):
    """System health status."""
    broker: str
    status: str
    details: Dict
    timestamp: str


# --- Ops ---
class GuardrailPayload(TypedDict, total=False):
    """Risk guardrail trigger event."""
    guard_name: str
    triggered: bool
    message: str
    value: float
    timestamp: str


class SessionPayload(TypedDict):
    """Trading session summary."""
    realized: float
    trade_count: int
    win_rate: float
    timestamp: str


class ConfigSnapshotPayload(TypedDict):
    """Configuration snapshot."""
    risk_pct: float
    routing: str
    active_symbols: List[str]
    timestamp: str


# --- Strategy ---
class StrategySignalPayload(TypedDict):
    """Strategy signal event."""
    symbol: str
    strategy: str
    signal: Literal["buy", "sell", "hold"]
    confidence: Optional[float]
    timestamp: str


# --- GUI -> Backend commands ---
class FlattenSymbolPayload(TypedDict):
    """Flatten symbol command."""
    symbol: str


class PlaceOrderPayload(TypedDict, total=False):
    """Manual order placement command."""
    symbol: str
    side: Literal["BUY", "SELL", "SHORT", "COVER"]
    qty: int
    type: Literal["market", "limit"]
    price: Optional[float]
    tif: Literal["DAY", "IOC", "FOK", "GTC"]
    route: str
    reduce_only: bool
    sl: Optional[float]  # Stop Loss
    tp: Optional[float]  # Take Profit


class SetStrategyPayload(TypedDict):
    """Set strategy command."""
    symbol: str
    strategy_name: str


# --- Additional command payloads ---
class FlattenAllPayload(TypedDict):
    """Flatten all positions command."""
    confirm: bool


class CancelAllPayload(TypedDict):
    """Cancel all orders command."""
    confirm: bool


class TogglePanicPayload(TypedDict):
    """Toggle panic mode command."""
    halted: bool


class HaltedPayload(TypedDict):
    """System halted state."""
    halted: bool
    reason: Optional[str]


class HaltStatePayload(TypedDict):
    """Halt state update."""
    halted: bool
    reason: Optional[str]
    timestamp: str


class CooldownStatePayload(TypedDict):
    """Cooldown state update."""
    symbol: str
    in_cooldown: bool
    remaining_seconds: float
    timestamp: str


class OrderQueueUpdatePayload(TypedDict):
    """Order queue update."""
    pending_count: int
    orders: List[Dict]
    timestamp: str


class ManualOrderPayload(TypedDict, total=False):
    """Manual order command."""
    symbol: str
    side: Literal["BUY", "SELL", "SHORT", "COVER"]
    qty: int
    order_type: Literal["market", "limit"]
    limit_price: Optional[float]
    tif: Literal["DAY", "IOC", "FOK", "GTC"]


# =============================================================================
# Event Schema Map
# =============================================================================

EVENT_SCHEMA_MAP: Dict[str, Type[Any]] = {
    # Market data
    EVENT_NEW_BAR: BarPayload,
    EVENT_PRICE_UPDATE: PricePayload,
    EVENT_REGIME_UPDATE: RegimePayload,
    EVENT_NEWS_UPDATE: NewsPayload,
    # Trades / execution
    EVENT_NEW_TRADE: TradePayload,
    EVENT_ORDER_STATUS: OrderStatusPayload,
    EVENT_POSITION_UPDATE: PositionPayload,
    # Performance
    EVENT_PNL_UPDATE: PnLPayload,
    EVENT_PERFORMANCE_METRICS: PerformanceMetricsPayload,
    EVENT_HEATMAP_UPDATE: HeatmapPayload,
    EVENT_DISTRIBUTION_UPDATE: DistributionPayload,
    EVENT_TRADE_STATS: TradeStatsPayload,
    EVENT_REGIME_PERF: RegimePerfPayload,
    # History & replay
    EVENT_HISTORY_UPDATE: HistoryPayload,
    EVENT_BENCHMARK_UPDATE: BenchmarkPayload,
    EVENT_REPLAY_FRAME: ReplayFramePayload,
    # Alerts & logs
    EVENT_ALERT: AlertPayload,
    EVENT_LOG: LogPayload,
    EVENT_HEALTH_UPDATE: HealthPayload,
    EVENT_GUARDRAIL_TRIGGERED: GuardrailPayload,
    # Ops
    EVENT_SESSION_UPDATE: SessionPayload,
    EVENT_CONFIG_SNAPSHOT: ConfigSnapshotPayload,
    # Strategy
    EVENT_STRATEGY_SIGNAL: StrategySignalPayload,
    # GUI -> Backend commands
    EVENT_FLATTEN_ALL: FlattenAllPayload,
    EVENT_CANCEL_ALL: CancelAllPayload,
    EVENT_FLATTEN_SYMBOL: FlattenSymbolPayload,
    EVENT_PLACE_ORDER: PlaceOrderPayload,
    EVENT_SET_STRATEGY: SetStrategyPayload,
    EVENT_TOGGLE_PANIC: TogglePanicPayload,
    EVENT_HALTED: HaltedPayload,
    EVENT_MANUAL_ORDER: ManualOrderPayload,
    # State updates
    EVENT_ORDER_QUEUE_UPDATE: OrderQueueUpdatePayload,
    EVENT_HALT_STATE: HaltStatePayload,
    EVENT_COOLDOWN_STATE: CooldownStatePayload,
}


# =============================================================================
# Exports
# =============================================================================

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
    # Additional command payloads
    'FlattenAllPayload', 'CancelAllPayload', 'TogglePanicPayload',
    'HaltedPayload', 'HaltStatePayload', 'CooldownStatePayload',
    'OrderQueueUpdatePayload', 'ManualOrderPayload',
    # Schema map
    'EVENT_SCHEMA_MAP',
]
