from typing import TypedDict, Literal, Optional, Dict
from datetime import datetime
from dataclasses import dataclass, asdict


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
class OrderResult:
    """
    Result of an order placement or cancellation.

    Supports both simple success/failure check and detailed order info.
    """
    order_id: Optional[str] = None
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

    # Legacy field for backwards compatibility
    success: Optional[bool] = None
    avg_price: Optional[float] = None  # alias for avg_fill_price

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


@dataclass
class PositionView:
    """View of a position in a symbol."""
    symbol: Optional[str] = None
    qty: float = 0
    avg_entry_price: float = 0.0
    market_price: Optional[float] = None
    side: Optional[str] = None  # 'long', 'short', 'flat'
    last_price: Optional[float] = None  # alias for market_price

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L if market price is available."""
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