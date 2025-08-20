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
    success: bool
    order_id: Optional[str] = None
    filled_qty: int = 0
    avg_price: Optional[float] = None
    message: Optional[str] = None
    raw: Optional[Dict] = None  # optional broker-native payload

    def __bool__(self) -> bool:
        # lets you do: if result: ...
        return self.success

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PositionView:
    symbol: str
    qty: int
    avg_price: float
    last_price: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BrokerSnapshot:
    cash: float
    equity: float
    # map "AAPL" -> PositionView
    positions: Dict[str, PositionView]

    def to_dict(self) -> Dict:
        return {
            "cash": self.cash,
            "equity": self.equity,
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
        }