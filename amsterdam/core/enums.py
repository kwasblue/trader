"""
Enums for Trading System

This module defines all enums used throughout the trading system for
type-safe constants and improved code readability.
"""

from enum import Enum, IntEnum

# ============================================================================
# ORDER ENUMS
# ============================================================================


class OrderSide(str, Enum):
    """Order side (buy or sell)."""

    BUY = "buy"
    SELL = "sell"

    def __str__(self) -> str:
        return self.value

    @property
    def sign(self) -> int:
        """Return +1 for buy, -1 for sell."""
        return 1 if self == OrderSide.BUY else -1

    def opposite(self) -> "OrderSide":
        """Return the opposite side."""
        return OrderSide.SELL if self == OrderSide.BUY else OrderSide.BUY


class OrderType(str, Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

    def __str__(self) -> str:
        return self.value

    @property
    def requires_limit_price(self) -> bool:
        """Check if this order type requires a limit price."""
        return self in (OrderType.LIMIT, OrderType.STOP_LIMIT)

    @property
    def requires_stop_price(self) -> bool:
        """Check if this order type requires a stop price."""
        return self in (OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP)


class OrderStatus(str, Enum):
    """Order status."""

    PENDING = "pending"  # Order submitted but not yet accepted
    ACCEPTED = "accepted"  # Order accepted by broker
    PARTIAL = "partial"  # Partially filled
    FILLED = "filled"  # Completely filled
    CANCELLED = "cancelled"  # Cancelled by user or system
    REJECTED = "rejected"  # Rejected by broker
    EXPIRED = "expired"  # Expired (for day orders)

    def __str__(self) -> str:
        return self.value

    @property
    def is_open(self) -> bool:
        """Check if order is still open."""
        return self in (OrderStatus.PENDING, OrderStatus.ACCEPTED, OrderStatus.PARTIAL)

    @property
    def is_closed(self) -> bool:
        """Check if order is closed."""
        return self in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED)


class TimeInForce(str, Enum):
    """Time in force for orders."""

    GTC = "gtc"  # Good till cancelled
    DAY = "day"  # Day order (expires at market close)
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    OPG = "opg"  # At the open
    CLS = "cls"  # At the close

    def __str__(self) -> str:
        return self.value


# ============================================================================
# POSITION ENUMS
# ============================================================================


class PositionSide(str, Enum):
    """Position side (long or short)."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"  # No position

    def __str__(self) -> str:
        return self.value

    @property
    def sign(self) -> int:
        """Return +1 for long, -1 for short, 0 for flat."""
        if self == PositionSide.LONG:
            return 1
        elif self == PositionSide.SHORT:
            return -1
        return 0

    @classmethod
    def from_quantity(cls, qty: float) -> "PositionSide":
        """Determine position side from quantity."""
        if qty > 0:
            return cls.LONG
        elif qty < 0:
            return cls.SHORT
        return cls.FLAT


class PositionState(str, Enum):
    """
    Position lifecycle state for concurrency control.

    Tracks the state of a position through order placement and fills
    to prevent race conditions and duplicate orders.
    """

    NONE = "none"  # No position, no pending orders
    PENDING_ENTRY = "pending_entry"  # Entry order placed, awaiting fill
    OPEN = "open"  # Position active
    PENDING_EXIT = "pending_exit"  # Exit order placed
    PENDING_ADD = "pending_add"  # Adding to existing position

    def __str__(self) -> str:
        return self.value

    @property
    def is_pending(self) -> bool:
        """Check if state represents a pending order."""
        return self in (PositionState.PENDING_ENTRY, PositionState.PENDING_EXIT, PositionState.PENDING_ADD)

    @property
    def allows_new_orders(self) -> bool:
        """Check if this state allows placing new orders."""
        return self in (PositionState.NONE, PositionState.OPEN)


# ============================================================================
# MARKET ENUMS
# ============================================================================


class MarketStatus(str, Enum):
    """Market status."""

    OPEN = "open"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    POST_MARKET = "post_market"
    EARLY_CLOSE = "early_close"
    HOLIDAY = "holiday"

    def __str__(self) -> str:
        return self.value

    @property
    def is_tradeable(self) -> bool:
        """Check if market is tradeable."""
        return self in (MarketStatus.OPEN, MarketStatus.PRE_MARKET, MarketStatus.POST_MARKET)


class AssetClass(str, Enum):
    """Asset class."""

    EQUITY = "equity"
    OPTION = "option"
    FUTURE = "future"
    FOREX = "forex"
    CRYPTO = "crypto"
    BOND = "bond"

    def __str__(self) -> str:
        return self.value


# ============================================================================
# STRATEGY ENUMS
# ============================================================================


class SignalType(str, Enum):
    """Trading signal type."""

    ENTRY_LONG = "entry_long"
    ENTRY_SHORT = "entry_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    HOLD = "hold"
    CLOSE_ALL = "close_all"

    def __str__(self) -> str:
        return self.value

    @property
    def is_entry(self) -> bool:
        """Check if signal is an entry."""
        return self in (SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT)

    @property
    def is_exit(self) -> bool:
        """Check if signal is an exit."""
        return self in (SignalType.EXIT_LONG, SignalType.EXIT_SHORT, SignalType.CLOSE_ALL)

    @property
    def side(self) -> OrderSide:
        """Get the order side for this signal."""
        if self in (SignalType.ENTRY_LONG, SignalType.EXIT_SHORT):
            return OrderSide.BUY
        elif self in (SignalType.ENTRY_SHORT, SignalType.EXIT_LONG):
            return OrderSide.SELL
        raise ValueError(f"Signal {self} has no clear order side")


class StrategyState(str, Enum):
    """Strategy execution state."""

    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

    def __str__(self) -> str:
        return self.value


# ============================================================================
# EXECUTION ENUMS
# ============================================================================


class ExecutionType(str, Enum):
    """Execution type."""

    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

    def __str__(self) -> str:
        return self.value

    @property
    def is_live(self) -> bool:
        """Check if execution is live trading."""
        return self == ExecutionType.LIVE


class FillType(str, Enum):
    """Fill simulation type for backtesting."""

    CLOSE = "close"  # Fill at bar close
    OPEN = "open"  # Fill at next bar open
    BID_ASK = "bid_ask"  # Fill using bid/ask spread
    SLIPPAGE = "slippage"  # Fill with slippage model

    def __str__(self) -> str:
        return self.value


# ============================================================================
# RISK MANAGEMENT ENUMS
# ============================================================================


class RiskLevel(IntEnum):
    """Risk level (integer for easy comparison)."""

    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5

    def __str__(self) -> str:
        return self.name.replace("_", " ").title()


class PositionSizingMethod(str, Enum):
    """Position sizing method."""

    FIXED = "fixed"  # Fixed dollar amount or shares
    PERCENT_EQUITY = "percent_equity"  # Percentage of total equity
    PERCENT_RISK = "percent_risk"  # Risk-based sizing
    KELLY = "kelly"  # Kelly criterion
    VOLATILITY = "volatility"  # Volatility-adjusted

    def __str__(self) -> str:
        return self.value


# ============================================================================
# EVENT ENUMS
# ============================================================================


class EventPriority(IntEnum):
    """Event priority (lower number = higher priority)."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3

    def __str__(self) -> str:
        return self.name.title()


# ============================================================================
# DATA ENUMS
# ============================================================================


class BarInterval(str, Enum):
    """Bar/candle interval."""

    TICK = "tick"
    SEC_1 = "1s"
    SEC_5 = "5s"
    SEC_15 = "15s"
    SEC_30 = "30s"
    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"

    def __str__(self) -> str:
        return self.value

    @property
    def seconds(self) -> int:
        """Get interval in seconds."""
        mapping = {
            BarInterval.TICK: 0,
            BarInterval.SEC_1: 1,
            BarInterval.SEC_5: 5,
            BarInterval.SEC_15: 15,
            BarInterval.SEC_30: 30,
            BarInterval.MIN_1: 60,
            BarInterval.MIN_5: 300,
            BarInterval.MIN_15: 900,
            BarInterval.MIN_30: 1800,
            BarInterval.HOUR_1: 3600,
            BarInterval.HOUR_4: 14400,
            BarInterval.DAY_1: 86400,
            BarInterval.WEEK_1: 604800,
            BarInterval.MONTH_1: 2592000,  # Approximate
        }
        return mapping[self]


class DataFeed(str, Enum):
    """Data feed source."""

    ALPACA = "alpaca"
    INTERACTIVE_BROKERS = "interactive_brokers"
    POLYGON = "polygon"
    YAHOO = "yahoo"
    BINANCE = "binance"
    COINBASE = "coinbase"
    CUSTOM = "custom"

    def __str__(self) -> str:
        return self.value


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def parse_order_side(value: str) -> OrderSide:
    """
    Parse string to OrderSide enum.

    Args:
        value: String value (case-insensitive)

    Returns:
        OrderSide enum

    Raises:
        ValueError: If value is invalid
    """
    try:
        return OrderSide(value.lower())
    except ValueError:
        raise ValueError(f"Invalid order side: {value}. Must be one of: {[s.value for s in OrderSide]}")


def parse_order_type(value: str) -> OrderType:
    """
    Parse string to OrderType enum.

    Args:
        value: String value (case-insensitive)

    Returns:
        OrderType enum

    Raises:
        ValueError: If value is invalid
    """
    try:
        return OrderType(value.lower())
    except ValueError:
        raise ValueError(f"Invalid order type: {value}. Must be one of: {[t.value for t in OrderType]}")


def parse_time_in_force(value: str) -> TimeInForce:
    """
    Parse string to TimeInForce enum.

    Args:
        value: String value (case-insensitive)

    Returns:
        TimeInForce enum

    Raises:
        ValueError: If value is invalid
    """
    try:
        return TimeInForce(value.lower())
    except ValueError:
        raise ValueError(f"Invalid time in force: {value}. Must be one of: {[t.value for t in TimeInForce]}")
