"""
Base Executor - Abstract base class for trade execution logic

This module defines the contract for executor implementations that handle
signal-based trade execution, order placement, position management, and trade logic.

The executor is completely decoupled from strategies - it receives simple numeric
signals (1 = buy, -1 = sell, 0 = hold) and executes accordingly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime
import pandas as pd

from core.events.eventhandler import EventHandler
from core.events.events import (
    EVENT_ORDER_STATUS,
    EVENT_NEW_TRADE,
    EVENT_PNL_UPDATE,
    EVENT_ALERT,
)
from core.events.event_models import (
    OrderStatusEvent,
    TradeEvent,
    PnLUpdateEvent,
    RiskAlertEvent,
)
from core.enums import OrderSide, OrderType, OrderStatus
from core.exceptions import (
    OrderError,
    OrderNotFoundError,
    InsufficientFundsError,
    InvalidOrderError,
    RiskViolationError,
)
from core.app_types import OrderResult, PositionView


class BaseExecutor(ABC):
    """
    Abstract base class for trade execution logic.
    
    Executors are completely decoupled from strategies:
    - Strategies generate simple signals: 1 (buy), -1 (sell), 0 (hold)
    - Executors receive signals and decide how to execute them
    - No knowledge of strategy internals required
    
    Responsibilities:
    - Interpret numeric signals
    - Apply risk management rules
    - Calculate position sizes
    - Place orders via broker
    - Track positions and P&L
    - Emit events for monitoring
    
    Design Philosophy:
    - Strategy → generates signal (what to do conceptually)
    - Executor → interprets signal and applies rules (what to do practically)
    - Broker → executes orders (how to do it technically)
    
    Example:
        class MyExecutor(BaseExecutor):
            def execute(self, symbol, df, signal, price, atr_value):
                if signal == 1:  # Buy signal
                    qty = self.calculate_position_size(symbol, price, atr_value)
                    if self.check_risk_limits(symbol, OrderSide.BUY, qty, price):
                        return self.buy(symbol, qty)
                elif signal == -1:  # Sell signal
                    return self.close_position(symbol)
                return None  # Hold
    """
    
    def __init__(self):
        """Initialize executor with event bus."""
        self.bus = EventHandler()
    
    # ========================================================================
    # ABSTRACT METHODS - SIGNAL EXECUTION
    # ========================================================================
    
    @abstractmethod
    def execute(
        self,
        symbol: str,
        df: pd.DataFrame,
        signal: int,
        price: float,
        atr_value: float,
    ) -> None:
        """
        Execute trading logic based on signal.
        
        This is the main entry point. The signal comes from a strategy as a
        simple integer value, and the executor decides how to act on it.
        
        Signal Values:
        - 1: Buy/Long signal
        - -1: Sell/Short signal
        - 0: Hold/No action
        
        Args:
            symbol: Trading symbol (e.g., "AAPL")
            df: Historical + real-time price data
            signal: Signal from strategy (1, -1, or 0)
            price: Current price
            atr_value: Average True Range for volatility-based sizing
            
        Raises:
            RiskViolationError: If trade violates risk rules
            InvalidOrderError: If order parameters invalid
            OrderError: If order placement fails
        """
        pass
    
    # ========================================================================
    # ABSTRACT METHODS - ORDER PLACEMENT
    # ========================================================================
    
    @abstractmethod
    def buy(
        self,
        symbol: str,
        qty: int,
        order_type: OrderType = OrderType.MARKET,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Place a buy order.
        
        Args:
            symbol: Trading symbol
            qty: Quantity to buy (integer shares)
            order_type: Order type (MARKET, LIMIT, etc.)
            **kwargs: Additional broker-specific parameters
                - limit_price: For LIMIT orders
                - stop_price: For STOP orders
            
        Returns:
            Order response dictionary with order_id, status, etc.
            
        Raises:
            InsufficientFundsError: If insufficient account balance
            InvalidOrderError: If order parameters invalid
            OrderError: If order placement fails
        """
        pass
    
    @abstractmethod
    def sell(
        self,
        symbol: str,
        qty: int,
        order_type: OrderType = OrderType.MARKET,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Place a sell order.
        
        Args:
            symbol: Trading symbol
            qty: Quantity to sell (integer shares)
            order_type: Order type (MARKET, LIMIT, etc.)
            **kwargs: Additional broker-specific parameters
                - limit_price: For LIMIT orders
                - stop_price: For STOP orders
            
        Returns:
            Order response dictionary with order_id, status, etc.
            
        Raises:
            InsufficientPositionError: If trying to sell more than held
            InvalidOrderError: If order parameters invalid
            OrderError: If order placement fails
        """
        pass
    
    @abstractmethod
    def place_oco_order(
        self,
        symbol: str,
        qty: int,
        stop_price: float,
        limit_price: float,
    ) -> Dict[str, Any]:
        """
        Place a One-Cancels-Other (OCO) bracket order.
        
        Creates two orders simultaneously:
        - Stop-loss order at stop_price (protects downside)
        - Take-profit order at limit_price (captures upside)
        
        When one executes, the other is automatically cancelled.
        
        Args:
            symbol: Trading symbol
            qty: Quantity for bracket (integer shares)
            stop_price: Stop-loss trigger price
            limit_price: Take-profit limit price
            
        Returns:
            Order response dictionary for OCO order group
            
        Raises:
            InvalidOrderError: If prices invalid (e.g., stop > limit for long)
            OrderError: If order placement fails
            
        Note:
            Not all brokers support OCO natively. Implementation may
            simulate with multiple orders and manual cancellation logic.
        """
        pass
    
    # ========================================================================
    # ABSTRACT METHODS - ORDER MANAGEMENT
    # ========================================================================
    
    @abstractmethod
    def get_open_orders(self) -> Any:
        """
        Get all open orders.
        
        Returns:
            List of open orders or broker-specific order object
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> Any:
        """
        Cancel a specific order by ID.
        
        Args:
            order_id: Unique order identifier
            
        Returns:
            Cancellation response from broker
            
        Raises:
            OrderNotFoundError: If order_id doesn't exist
            OrderError: If cancellation fails
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Any:
        """
        Get status of a specific order.
        
        Args:
            order_id: Unique order identifier
            
        Returns:
            Order status from broker
            
        Raises:
            OrderNotFoundError: If order_id doesn't exist
        """
        pass
    
    # ========================================================================
    # ABSTRACT METHODS - LOGGING & RETRY
    # ========================================================================
    
    @abstractmethod
    def log_order_response(self, response: dict) -> None:
        """
        Log order response or error.
        
        Args:
            response: Order response dictionary from broker
                Should contain: order_id, status, symbol, qty, price, etc.
        """
        pass
    
    @abstractmethod
    def retry_failed_order(
        self,
        *args,
        max_retries: int = 3,
        delay: float = 2.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Retry order placement on failure with exponential backoff.
        
        Args:
            *args: Positional arguments for order function
            max_retries: Maximum number of retry attempts
            delay: Initial delay in seconds (doubles each retry)
            **kwargs: Keyword arguments for order function
            
        Returns:
            Order response from successful execution
            
        Raises:
            OrderError: If all retries fail
            
        Example:
            result = self.retry_failed_order(
                self.buy,
                symbol="AAPL",
                qty=100,
                max_retries=3
            )
        """
        pass
    
    # ========================================================================
    # EVENT EMISSION HELPERS (IMPLEMENTED)
    # ========================================================================
    
    async def emit_order_status(
        self,
        status: OrderStatus,
        symbol: str,
        side: OrderSide,
        qty: float,
        order_id: Optional[str] = None,
        filled_qty: float = 0.0,
        avg_price: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> None:
        """
        Emit order status event.
        
        Args:
            status: Order status (PENDING, FILLED, CANCELLED, etc.)
            symbol: Trading symbol
            side: Order side (BUY or SELL)
            qty: Order quantity
            order_id: Order identifier
            filled_qty: Quantity filled so far
            avg_price: Average fill price
            reason: Reason for rejection/cancellation (if applicable)
        """
        event = OrderStatusEvent(
            order_id=order_id,
            symbol=symbol,
            side=side.value,
            qty=qty,
            filled_qty=filled_qty,
            avg_price=avg_price,
            status=status.value,
            reason=reason,
            timestamp=datetime.utcnow(),
        )
        await self.bus.emit(EVENT_ORDER_STATUS, event.model_dump())
    
    async def emit_new_trade(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        price: float,
        pnl: Optional[float] = None,
        commission: float = 0.0,
        order_id: Optional[str] = None,
    ) -> None:
        """
        Emit trade execution event.
        
        Args:
            symbol: Trading symbol
            side: Trade side (BUY or SELL)
            qty: Trade quantity
            price: Execution price
            pnl: Realized P&L from this trade (if closing position)
            commission: Commission/fees paid
            order_id: Associated order ID
        """
        event = TradeEvent(
            order_id=order_id,
            symbol=symbol,
            side=side.value,
            qty=qty,
            price=price,
            pnl=pnl,
            commission=commission,
            timestamp=datetime.utcnow(),
        )
        await self.bus.emit(EVENT_NEW_TRADE, event.model_dump())
    
    async def emit_pnl_update(
        self,
        portfolio_value: float,
        equity_curve: List[float],
        unrealized: float,
        realized: float,
        drawdown: float,
        cash: Optional[float] = None,
    ) -> None:
        """
        Emit P&L update event.
        
        Args:
            portfolio_value: Total portfolio value (cash + positions)
            equity_curve: Historical equity values
            unrealized: Current unrealized P&L
            realized: Cumulative realized P&L
            drawdown: Current drawdown from peak (negative value)
            cash: Available cash balance
        """
        event = PnLUpdateEvent(
            portfolio_value=portfolio_value,
            equity_curve=equity_curve,
            unrealized=unrealized,
            realized=realized,
            drawdown=drawdown,
            cash=cash,
            timestamp=datetime.utcnow(),
        )
        await self.bus.emit(EVENT_PNL_UPDATE, event.model_dump())
    
    async def emit_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        symbol: Optional[str] = None,
        current_value: float = 0.0,
        threshold: float = 0.0,
        action_taken: Optional[str] = None,
    ) -> None:
        """
        Emit risk/system alert event.
        
        Args:
            alert_type: Type of alert (risk_limit, system_error, etc.)
            severity: Alert severity (low/medium/high/critical)
            message: Human-readable alert message
            symbol: Related symbol (if applicable)
            current_value: Value that triggered alert
            threshold: Threshold that was breached
            action_taken: Automated action taken (e.g., "orders cancelled")
        """
        event = RiskAlertEvent(
            alert_type=alert_type,
            severity=severity,
            message=message,
            symbol=symbol,
            current_value=current_value,
            threshold=threshold,
            action_taken=action_taken,
            timestamp=datetime.utcnow(),
        )
        await self.bus.emit(EVENT_ALERT, event.model_dump())
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def __repr__(self) -> str:
        """String representation of executor."""
        return f"{self.__class__.__name__}()"