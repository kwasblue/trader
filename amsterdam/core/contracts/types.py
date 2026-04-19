"""
Core Data Types Module

Canonical type definitions for the trading system. This is the single source
of truth for dataclasses and types used across the codebase.

Usage:
    from core.contracts import SignalContext, OrderResult, PositionView, BrokerSnapshot
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, TypedDict

# =============================================================================
# TypedDict Payloads (lightweight, no validation overhead)
# =============================================================================


class SignalPayload(TypedDict):
    """Signal payload for events."""

    symbol: str
    timestamp: datetime
    signal: int  # -1, 0, 1
    strategy: str
    atr: float | None
    regime: str | None


# =============================================================================
# Core Dataclasses
# =============================================================================


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
    strategy_name: str | None = None
    confidence: float = 1.0  # Signal strength 0-1

    # Market context
    df: Any | None = None  # DataFrame with price history
    market_open: bool = True

    # Order parameters (configurable per signal)
    order_type: str = "market"  # market, limit, stop, stop_limit
    time_in_force: str = "day"  # day, gtc, ioc, fok
    limit_price: float | None = None  # For limit orders
    stop_price: float | None = None  # For stop orders

    # Extensible metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_kwargs(
        cls,
        symbol: str,
        signal: int,
        price: float = 0.0,
        atr: float = 0.0,
        regime: str = "normal",
        timestamp: datetime | None = None,
        strategy_name: str | None = None,
        confidence: float = 1.0,
        df: Any | None = None,
        market_open: bool = True,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: float | None = None,
        stop_price: float | None = None,
        **kwargs,
    ) -> "SignalContext":
        """
        Factory for backward compatibility with scattered kwargs.

        Accepts any additional kwargs and stores them in metadata.
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
            order_type=order_type,
            time_in_force=time_in_force,
            limit_price=limit_price,
            stop_price=stop_price,
            metadata=kwargs,
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

    order_id: str | None = None
    client_order_id: str | None = None  # Broker's client order ID
    symbol: str | None = None
    side: str | None = None
    qty: float | None = None
    type: str | None = None  # market, limit, stop, stop_limit
    time_in_force: str | None = None
    status: str | None = None  # submitted, filled, cancelled, rejected, etc.
    limit_price: float | None = None
    stop_price: float | None = None
    filled_qty: float | None = None
    avg_fill_price: float | None = None
    raw: dict | None = None  # broker-native response payload
    message: str | None = None
    commission: float | None = None  # Commission/fees paid for this order

    # Legacy field for backwards compatibility
    success: bool | None = None
    avg_price: float | None = None  # DEPRECATED: alias for avg_fill_price

    def __bool__(self) -> bool:
        """Allow: if result: ... to check for success."""
        if self.success is not None:
            return self.success
        # Infer success from status
        return self.status in ("filled", "submitted", "accepted", "working", "pending")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == "filled"

    @property
    def is_rejected(self) -> bool:
        """Check if order was rejected."""
        return self.status in ("rejected", "error", "failed")

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

    symbol: str | None = None
    qty: float = 0
    avg_entry_price: float = 0.0
    market_price: float | None = None
    side: str | None = None  # 'long', 'short', 'flat'
    last_price: float | None = None  # alias for market_price
    unrealized_pl: float | None = None  # P&L from broker
    unrealized_plpc: float | None = None  # P&L percentage from broker

    def to_dict(self) -> dict:
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

    account_number: str | None = None
    status: str | None = None
    cash: float = 0.0
    buying_power: float = 0.0
    equity: float = 0.0
    portfolio_value: float = 0.0
    positions: dict[str, PositionView] | None = None

    def to_dict(self) -> dict:
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


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # TypedDicts
    "SignalPayload",
    # Dataclasses
    "SignalContext",
    "OrderResult",
    "PositionView",
    "BrokerSnapshot",
]
