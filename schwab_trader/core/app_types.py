from typing import TypedDict, Literal, Optional, Dict, Any
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field


BarEventName = Literal["BAR_CREATED"]
SignalEventName = Literal["SIGNAL"]
TradeEventName = Literal["TRADE_EXECUTED"]
PnlEventName = Literal["PNL_UPDATE"]
SystemEventName = Literal["HEARTBEAT","ERROR","WARNING","INFO"]

class BarPayload(TypedDict):
    symbol: str
    timestamp: datetime
    open: float; high: float; low: float; close: float
    volume: int

class SignalPayload(TypedDict):
    symbol: str
    timestamp: datetime
    signal: int  # -1,0,1
    strategy: str
    atr: float | None
    regime: str | None


@dataclass
class SignalContext:
    """
    Unified context for signal processing.

    Consolidates all signal-related data into a single object for cleaner
    API signatures and easier extension.

    Example:
        context = SignalContext(
            symbol="AAPL",
            signal=1,
            price=150.25,
            atr=2.5,
            regime="trending",
            timestamp=datetime.now(timezone.utc),
            strategy_name="momentum",
            confidence=0.85
        )

        # Or from kwargs for backward compatibility
        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,
            price=150.25,
            atr=2.5,
            regime="normal"
        )
    """
    # Core signal data
    symbol: str
    signal: int  # -1 (short/sell), 0 (hold), 1 (long/buy)
    price: float
    atr: float
    regime: str
    timestamp: datetime

    # Strategy context
    strategy_name: Optional[str] = None
    confidence: float = 1.0  # Signal strength 0-1

    # Market context
    df: Optional[Any] = None  # DataFrame with price history
    market_open: bool = True

    # Extensible metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_kwargs(
        cls,
        symbol: str,
        signal: int,
        price: float = 0.0,
        atr: float = 0.0,
        regime: str = "normal",
        timestamp: Optional[datetime] = None,
        strategy_name: Optional[str] = None,
        confidence: float = 1.0,
        df: Optional[Any] = None,
        market_open: bool = True,
        **kwargs
    ) -> "SignalContext":
        """
        Factory for backward compatibility with scattered kwargs.

        Accepts any additional kwargs and stores them in metadata.

        Args:
            symbol: Trading symbol
            signal: Signal value (-1, 0, 1)
            price: Current market price
            atr: Average True Range
            regime: Market regime classification
            timestamp: Signal timestamp (defaults to now)
            strategy_name: Name of generating strategy
            confidence: Signal confidence 0-1
            df: Price history DataFrame
            market_open: Whether market is open
            **kwargs: Additional data stored in metadata

        Returns:
            SignalContext instance
        """
        return cls(
            symbol=symbol,
            signal=signal,
            price=price,
            atr=atr,
            regime=regime,
            timestamp=timestamp or datetime.now(timezone.utc),
            strategy_name=strategy_name,
            confidence=confidence,
            df=df,
            market_open=market_open,
            metadata=kwargs
        )

    def is_buy(self) -> bool:
        """Check if signal is a buy/long signal."""
        return self.signal == 1

    def is_sell(self) -> bool:
        """Check if signal is a sell/short signal."""
        return self.signal == -1

    def is_hold(self) -> bool:
        """Check if signal is a hold/no-action signal."""
        return self.signal == 0


@dataclass
class OrderResult:
    """
    Result of an order placement or cancellation.

    Supports both simple success/failure check and detailed order info.

    Note:
        - avg_price is a deprecated alias for avg_fill_price
        - Use avg_fill_price for new code
    """
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None  # Broker's client order ID
    symbol: Optional[str] = None
    side: Optional[str] = None
    qty: Optional[float] = None
    type: Optional[str] = None  # market, limit, stop, stop_limit
    time_in_force: Optional[str] = None
    status: Optional[str] = None  # submitted, filled, cancelled, rejected, etc.
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_qty: Optional[float] = None
    avg_fill_price: Optional[float] = None
    raw: Optional[Dict] = None  # broker-native response payload
    message: Optional[str] = None
    commission: Optional[float] = None  # Commission/fees paid for this order

    # Legacy field for backwards compatibility
    success: Optional[bool] = None
    avg_price: Optional[float] = None  # DEPRECATED: alias for avg_fill_price

    def __bool__(self) -> bool:
        """Allow: if result: ... to check for success."""
        if self.success is not None:
            return self.success
        # Infer success from status
        return self.status in ('filled', 'submitted', 'accepted', 'working', 'pending')

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == 'filled'

    @property
    def is_rejected(self) -> bool:
        """Check if order was rejected."""
        return self.status in ('rejected', 'error', 'failed')

    @property
    def filled_value(self) -> float:
        """
        Calculate the total filled value (filled_qty * avg_fill_price).

        Returns:
            Total value of filled portion, or 0.0 if not available
        """
        qty = self.filled_qty or 0.0
        # Use avg_fill_price, falling back to avg_price for legacy compatibility
        price = self.avg_fill_price or self.avg_price or 0.0
        return qty * price


@dataclass
class PositionView:
    """View of a position in a symbol."""
    symbol: Optional[str] = None
    qty: float = 0
    avg_entry_price: float = 0.0
    market_price: Optional[float] = None
    side: Optional[str] = None  # 'long', 'short', 'flat'
    last_price: Optional[float] = None  # alias for market_price
    unrealized_pl: Optional[float] = None  # P&L from broker
    unrealized_plpc: Optional[float] = None  # P&L percentage from broker

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def avg_price(self) -> float:
        """Alias for avg_entry_price for backwards compatibility."""
        return self.avg_entry_price

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L if market price is available."""
        # Use broker-provided P&L if available
        if self.unrealized_pl is not None:
            return self.unrealized_pl
        if self.market_price is None or self.qty == 0:
            return 0.0
        return (self.market_price - self.avg_entry_price) * self.qty


@dataclass
class BrokerSnapshot:
    """Snapshot of broker account state."""
    account_number: Optional[str] = None
    status: Optional[str] = None
    cash: float = 0.0
    buying_power: float = 0.0
    equity: float = 0.0
    portfolio_value: float = 0.0
    positions: Optional[Dict[str, PositionView]] = None

    def to_dict(self) -> Dict:
        result = {
            "account_number": self.account_number,
            "status": self.status,
            "cash": self.cash,
            "buying_power": self.buying_power,
            "equity": self.equity,
            "portfolio_value": self.portfolio_value,
        }
        if self.positions:
            result["positions"] = {k: v.to_dict() for k, v in self.positions.items()}
        return result