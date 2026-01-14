"""
Event Models - Pydantic models for event payloads

This module defines strongly-typed event models for all system events,
providing validation, serialization, and documentation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime

from core.enums import OrderSide, OrderStatus


# ============================================================================
# ORDER EVENTS
# ============================================================================

class OrderStatusEvent(BaseModel):
    """
    Event emitted when order status changes.
    
    Emitted when:
    - Order is submitted
    - Order is accepted by broker
    - Order is partially filled
    - Order is completely filled
    - Order is cancelled
    - Order is rejected
    """
    
    order_id: Optional[str] = Field(None, description="Unique order identifier")
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="Order side (buy/sell)")
    qty: float = Field(..., gt=0, description="Order quantity")
    filled_qty: float = Field(0.0, ge=0, description="Quantity filled so far")
    avg_price: Optional[float] = Field(None, gt=0, description="Average fill price")
    status: str = Field(..., description="Order status")
    reason: Optional[str] = Field(None, description="Rejection/cancellation reason")
    timestamp: datetime = Field(..., description="Event timestamp (UTC)")
    
    @field_validator('side')
    @classmethod
    def validate_side(cls, v: str) -> str:
        """Validate order side."""
        if v not in [s.value for s in OrderSide]:
            raise ValueError(f"Invalid side: {v}")
        return v
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate order status."""
        if v not in [s.value for s in OrderStatus]:
            raise ValueError(f"Invalid status: {v}")
        return v
    
    @field_validator('filled_qty')
    @classmethod
    def validate_filled_qty(cls, v: float, info) -> float:
        """Validate filled quantity doesn't exceed order quantity."""
        if 'qty' in info.data and v > info.data['qty']:
            raise ValueError(f"Filled qty ({v}) cannot exceed order qty ({info.data['qty']})")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "order_id": "123e4567-e89b-12d3-a456-426614174000",
                "symbol": "AAPL",
                "side": "buy",
                "qty": 100.0,
                "filled_qty": 100.0,
                "avg_price": 150.25,
                "status": "filled",
                "reason": None,
                "timestamp": "2024-01-15T14:30:00Z"
            }
        }


class TradeEvent(BaseModel):
    """
    Event emitted when trade is executed.
    
    Emitted when:
    - Order fill occurs (partial or complete)
    """
    
    order_id: Optional[str] = Field(None, description="Associated order ID")
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="Trade side (buy/sell)")
    qty: float = Field(..., gt=0, description="Trade quantity")
    price: float = Field(..., gt=0, description="Execution price")
    pnl: Optional[float] = Field(None, description="Realized P&L (if closing)")
    commission: float = Field(0.0, ge=0, description="Commission/fees paid")
    timestamp: datetime = Field(..., description="Event timestamp (UTC)")
    
    @field_validator('side')
    @classmethod
    def validate_side(cls, v: str) -> str:
        """Validate trade side."""
        if v not in [s.value for s in OrderSide]:
            raise ValueError(f"Invalid side: {v}")
        return v
    
    @property
    def trade_value(self) -> float:
        """Calculate total trade value (excluding commission)."""
        return self.qty * self.price
    
    @property
    def net_value(self) -> float:
        """Calculate net trade value (including commission)."""
        return self.trade_value - self.commission
    
    class Config:
        json_schema_extra = {
            "example": {
                "order_id": "123e4567-e89b-12d3-a456-426614174000",
                "symbol": "AAPL",
                "side": "buy",
                "qty": 100.0,
                "price": 150.25,
                "pnl": None,
                "commission": 1.50,
                "timestamp": "2024-01-15T14:30:00Z"
            }
        }


# ============================================================================
# P&L EVENTS
# ============================================================================

class PnLUpdateEvent(BaseModel):
    """
    Event emitted when portfolio P&L is updated.
    
    Emitted when:
    - Trade execution changes realized P&L
    - Position mark-to-market changes unrealized P&L
    - Regular periodic updates
    """
    
    portfolio_value: float = Field(..., description="Total portfolio value")
    equity_curve: List[float] = Field(..., description="Historical equity values")
    unrealized: float = Field(..., description="Current unrealized P&L")
    realized: float = Field(..., description="Cumulative realized P&L")
    drawdown: float = Field(..., le=0, description="Current drawdown from peak")
    cash: Optional[float] = Field(None, ge=0, description="Available cash")
    timestamp: datetime = Field(..., description="Event timestamp (UTC)")
    
    @property
    def total_pnl(self) -> float:
        """Calculate total P&L (realized + unrealized)."""
        return self.realized + self.unrealized
    
    @property
    def drawdown_pct(self) -> float:
        """Calculate drawdown as percentage."""
        if not self.equity_curve:
            return 0.0
        peak = max(self.equity_curve)
        if peak == 0:
            return 0.0
        return self.drawdown / peak
    
    class Config:
        json_schema_extra = {
            "example": {
                "portfolio_value": 105000.0,
                "equity_curve": [100000.0, 102000.0, 105000.0],
                "unrealized": 2000.0,
                "realized": 3000.0,
                "drawdown": -1000.0,
                "cash": 50000.0,
                "timestamp": "2024-01-15T14:30:00Z"
            }
        }


# ============================================================================
# POSITION EVENTS
# ============================================================================

class PositionUpdateEvent(BaseModel):
    """
    Event emitted when position is updated.
    
    Emitted when:
    - Position is opened
    - Position size changes
    - Position is closed
    """
    
    symbol: str = Field(..., description="Trading symbol")
    qty: float = Field(..., description="Current position quantity (+ long, - short)")
    avg_price: float = Field(..., gt=0, description="Average entry price")
    current_price: Optional[float] = Field(None, gt=0, description="Current market price")
    unrealized_pnl: float = Field(..., description="Unrealized P&L")
    realized_pnl: float = Field(0.0, description="Realized P&L from this position")
    timestamp: datetime = Field(..., description="Event timestamp (UTC)")
    
    @property
    def market_value(self) -> float:
        """Calculate current market value of position."""
        if self.current_price is None:
            return 0.0
        return abs(self.qty) * self.current_price
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.qty > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.qty < 0
    
    @property
    def is_flat(self) -> bool:
        """Check if position is flat (no position)."""
        return self.qty == 0
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "qty": 100.0,
                "avg_price": 148.50,
                "current_price": 150.25,
                "unrealized_pnl": 175.0,
                "realized_pnl": 0.0,
                "timestamp": "2024-01-15T14:30:00Z"
            }
        }


# ============================================================================
# SIGNAL EVENTS
# ============================================================================

class SignalEvent(BaseModel):
    """
    Event emitted when trading signal is generated.
    
    Emitted when:
    - Strategy generates entry signal
    - Strategy generates exit signal
    """
    
    symbol: str = Field(..., description="Trading symbol")
    signal_type: str = Field(..., description="Signal type")
    side: str = Field(..., description="Order side for this signal")
    strength: float = Field(..., ge=0, le=1, description="Signal strength (0-1)")
    target_price: Optional[float] = Field(None, gt=0, description="Target entry/exit price")
    stop_price: Optional[float] = Field(None, gt=0, description="Stop loss price")
    qty: Optional[float] = Field(None, gt=0, description="Suggested quantity")
    metadata: dict = Field(default_factory=dict, description="Additional signal metadata")
    timestamp: datetime = Field(..., description="Event timestamp (UTC)")
    
    @field_validator('side')
    @classmethod
    def validate_side(cls, v: str) -> str:
        """Validate signal side."""
        if v not in [s.value for s in OrderSide]:
            raise ValueError(f"Invalid side: {v}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "signal_type": "entry_long",
                "side": "buy",
                "strength": 0.85,
                "target_price": 150.00,
                "stop_price": 145.00,
                "qty": 100.0,
                "metadata": {"indicator": "sma_cross", "timeframe": "1h"},
                "timestamp": "2024-01-15T14:30:00Z"
            }
        }


# ============================================================================
# RISK EVENTS
# ============================================================================

class RiskAlertEvent(BaseModel):
    """
    Event emitted when risk threshold is breached.
    
    Emitted when:
    - Position size exceeds limit
    - Daily loss exceeds threshold
    - Drawdown exceeds maximum
    - Margin requirements not met
    """
    
    alert_type: str = Field(..., description="Type of risk alert")
    severity: str = Field(..., description="Alert severity (low/medium/high/critical)")
    message: str = Field(..., description="Alert message")
    symbol: Optional[str] = Field(None, description="Related symbol (if applicable)")
    current_value: float = Field(..., description="Current value that triggered alert")
    threshold: float = Field(..., description="Threshold that was breached")
    action_taken: Optional[str] = Field(None, description="Automated action taken")
    timestamp: datetime = Field(..., description="Event timestamp (UTC)")
    
    @property
    def breach_amount(self) -> float:
        """Calculate amount by which threshold was breached."""
        return self.current_value - self.threshold
    
    @property
    def breach_pct(self) -> float:
        """Calculate percentage breach."""
        if self.threshold == 0:
            return 0.0
        return (self.current_value - self.threshold) / abs(self.threshold)
    
    class Config:
        json_schema_extra = {
            "example": {
                "alert_type": "daily_loss_limit",
                "severity": "high",
                "message": "Daily loss limit exceeded",
                "symbol": None,
                "current_value": -5500.0,
                "threshold": -5000.0,
                "action_taken": "All orders cancelled, new orders blocked",
                "timestamp": "2024-01-15T14:30:00Z"
            }
        }


# ============================================================================
# SYSTEM EVENTS
# ============================================================================

class SystemEvent(BaseModel):
    """
    Event emitted for system-level notifications.
    
    Emitted when:
    - System starts/stops
    - Strategy initializes/errors
    - Connection status changes
    - Critical errors occur
    """
    
    event_type: str = Field(..., description="System event type")
    severity: str = Field(..., description="Event severity")
    message: str = Field(..., description="Event message")
    component: Optional[str] = Field(None, description="Component that emitted event")
    details: dict = Field(default_factory=dict, description="Additional event details")
    timestamp: datetime = Field(..., description="Event timestamp (UTC)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_type": "strategy_started",
                "severity": "info",
                "message": "MomentumStrategy initialized successfully",
                "component": "MomentumStrategy",
                "details": {"symbols": ["AAPL", "MSFT"], "timeframe": "1h"},
                "timestamp": "2024-01-15T14:30:00Z"
            }
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_event(event_cls: type[BaseModel], data: dict) -> BaseModel:
    """
    Validate and parse event data.
    
    Args:
        event_cls: Pydantic model class
        data: Event data dictionary
        
    Returns:
        Validated event model instance
        
    Raises:
        ValidationError: If data is invalid
    """
    return event_cls(**data)
