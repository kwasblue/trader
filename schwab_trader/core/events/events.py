"""
core/events.py

Central event name definitions and payload schemas (TypedDicts).
Ensures backend + GUI use consistent event names and payloads.
"""

from typing import TypedDict, Literal, Optional, List, Dict, Type, Any

# =====================================================
# === Event name constants
# =====================================================

# --- Backend → GUI ---
# Market data & execution
EVENT_NEW_BAR            = "NEW_BAR"
EVENT_NEW_TRADE          = "NEW_TRADE"
EVENT_ORDER_STATUS       = "ORDER_STATUS"
EVENT_POSITION_UPDATE    = "POSITION_UPDATE"
EVENT_PNL_UPDATE         = "PNL_UPDATE"


# Monitoring & alerts
EVENT_ALERT              = "ALERT"
EVENT_HEALTH_UPDATE      = "HEALTH_UPDATE"
EVENT_ORDER_QUEUE_UPDATE = "ORDER_QUEUE_UPDATE"
EVENT_HALT_STATE         = "HALT_STATE"
EVENT_COOLDOWN_STATE     = "COOLDOWN_STATE"

# Market context
EVENT_PRICE_UPDATE       = "PRICE_UPDATE"
EVENT_REGIME_UPDATE      = "REGIME_UPDATE"
EVENT_NEWS_UPDATE        = "NEWS_UPDATE"

# Performance & analytics
EVENT_STRATEGY_SIGNAL    = "STRATEGY_SIGNAL"
EVENT_PERFORMANCE_METRICS= "PERFORMANCE_METRICS"
EVENT_HEATMAP_UPDATE     = "HEATMAP_UPDATE"
EVENT_DISTRIBUTION_UPDATE= "DISTRIBUTION_UPDATE"
EVENT_TRADE_STATS        = "TRADE_STATS"
EVENT_REGIME_PERF        = "REGIME_PERF"

# History & replay
EVENT_HISTORY_UPDATE     = "HISTORY_UPDATE"
EVENT_BENCHMARK_UPDATE   = "BENCHMARK_UPDATE"
EVENT_REPLAY_FRAME       = "REPLAY_FRAME"

# Logging & ops
EVENT_LOG                = "LOG"
EVENT_GUARDRAIL_TRIGGERED= "GUARDRAIL_TRIGGERED"
EVENT_SESSION_UPDATE     = "SESSION_UPDATE"
EVENT_CONFIG_SNAPSHOT    = "CONFIG_SNAPSHOT"

# --- GUI → Backend ---
EVENT_FLATTEN_ALL        = "FLATTEN_ALL"
EVENT_CANCEL_ALL         = "CANCEL_ALL"
EVENT_FLATTEN_SYMBOL     = "FLATTEN_SYMBOL"
EVENT_PLACE_ORDER        = "PLACE_ORDER"
EVENT_SET_STRATEGY       = "SET_STRATEGY"
EVENT_TOGGLE_PANIC       = "TOGGLE_PANIC"
EVENT_HALTED             = "HALTED"
EVENT_MANUAL_ORDER       = "MANUAL_ORDER"

# =====================================================
# === Payload schemas (TypedDicts)
# =====================================================

# --- Market data ---
class BarPayload(TypedDict):
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: str  # ISO8601

class PricePayload(TypedDict):
    symbol: str
    price: float
    ma20: Optional[float]
    ma50: Optional[float]
    timestamp: str

class RegimePayload(TypedDict):
    symbol: str
    volatility: str   # low_volatility | normal | high_volatility
    trend: str        # bullish | bearish | sideways
    market: str       # bull | bear | neutral
    timestamp: str

class NewsPayload(TypedDict):
    headline: str
    source: str
    sentiment: Optional[Literal["positive","negative","neutral"]]
    timestamp: str

# --- Execution / trades ---
class TradePayload(TypedDict):
    symbol: str
    side: Literal["buy", "sell", "long", "short"]
    qty: float
    price: float
    timestamp: str
    pnl: Optional[float]

class OrderStatusPayload(TypedDict):
    order_id: str
    symbol: str
    status: Literal["submitted", "filled", "canceled", "rejected"]
    filled_qty: float
    avg_price: Optional[float]
    timestamp: str

class PositionPayload(TypedDict, total=False):
    symbol: str
    qty: float
    avg_price: float
    unrealized: float
    realized: float
    timestamp: str
    # Extended fields for GUI display
    side: str  # "long" or "short"
    last: float  # Last/current price
    avg: float  # Alias for avg_price (GUI model uses this)
    unreal: float  # Alias for unrealized (GUI model uses this)
    market_value: float  # Position market value

# --- Performance ---
class PnLPayload(TypedDict, total=False):
    portfolio_value: float
    equity_curve: List[float]
    unrealized: float
    realized: float
    drawdown: float
    timestamp: str
    cash: float  # Available cash
    buying_power: float  # Buying power from broker

class PerformanceMetricsPayload(TypedDict):
    sharpe: float
    sortino: float
    kelly: float
    max_dd: float
    hit_rate: float
    avg_win: float
    avg_loss: float
    timestamp: str

class HeatmapPayload(TypedDict):
    data: List[List[float]]   # 2D matrix for risk vs exposure
    timestamp: str

class DistributionPayload(TypedDict):
    bins: List[float]
    counts: List[int]
    timestamp: str

class TradeStatsPayload(TypedDict):
    durations: List[float]
    win_streak: int
    loss_streak: int
    timestamp: str

class RegimePerfPayload(TypedDict):
    regimes: Dict[str, float]  # {"low_volatility": equity, "high_volatility": equity}
    timestamp: str

# --- History / Replay ---
class HistoryPayload(TypedDict):
    pnl_by_day: Dict[str, float]  # date -> pnl
    timestamp: str

class BenchmarkPayload(TypedDict):
    equity_curve: List[float]
    benchmark_curve: List[float]
    timestamp: str

class ReplayFramePayload(TypedDict):
    frame_idx: int
    equity: float
    timestamp: str

# --- Alerts & logs ---
class AlertPayload(TypedDict):
    level: Literal["info", "warning", "error", "critical"]
    message: str
    symbol: Optional[str]
    timestamp: str

class LogPayload(TypedDict):
    message: str
    level: str
    timestamp: str

class HealthPayload(TypedDict):
    broker: str
    status: str
    details: Dict
    timestamp: str

# --- Ops ---
class GuardrailPayload(TypedDict, total=False):
    guard_name: str
    triggered: bool
    message: str
    value: float  # Generic value (drawdown, cooldown time, etc.)
    timestamp: str

class SessionPayload(TypedDict):
    realized: float
    trade_count: int
    win_rate: float
    timestamp: str

class ConfigSnapshotPayload(TypedDict):
    risk_pct: float
    routing: str
    active_symbols: List[str]
    timestamp: str

# --- Strategy ---
class StrategySignalPayload(TypedDict):
    symbol: str
    strategy: str
    signal: Literal["buy", "sell", "hold"]
    confidence: Optional[float]
    timestamp: str

# --- GUI → Backend commands ---
class FlattenSymbolPayload(TypedDict):
    symbol: str

class PlaceOrderPayload(TypedDict, total=False):
    symbol: str
    side: Literal["BUY", "SELL", "SHORT", "COVER"]
    qty: int
    type: Literal["market", "limit"]
    price: Optional[float]
    tif: Literal["DAY", "IOC", "FOK", "GTC"]
    route: str
    reduce_only: bool
    sl: Optional[float]   # Stop Loss
    tp: Optional[float]   # Take Profit

class SetStrategyPayload(TypedDict):
    symbol: str
    strategy_name: str

# =====================================================
# === Central Schema Map
# =====================================================

EVENT_SCHEMA_MAP: Dict[str, Type[Any]]  = {
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
}
