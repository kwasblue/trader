"""
Base Broker Interface - Abstract interface for broker implementations

This module defines the contract that all broker adapters (live, simulation, paper)
must implement to ensure consistent behavior across different execution environments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from core.app_types import OrderResult, PositionView, BrokerSnapshot
from core.contracts.events import EVENT_ORDER_STATUS, EVENT_NEW_TRADE, EVENT_PNL_UPDATE
from core.events.eventhandler import EventHandler
from core.enums import OrderSide, OrderType, OrderStatus, TimeInForce
from core.exceptions import (
    BrokerError,
    OrderNotFoundError,
    SymbolNotFoundError,
    InsufficientFundsError,
    InvalidOrderError,
)


class BaseBrokerInterface(ABC):
    """
    Abstract base class for broker adapters.
    
    Provides a unified interface for:
    - Order execution (market, limit, stop, OCO)
    - Position and account management
    - Market status queries
    - Event emission for order status, trades, and P&L updates
    
    All broker implementations (SimulatedBroker, AlpacaBroker, IBKRBroker, etc.)
    must inherit from this class and implement all abstract methods.
    
    Architecture:
    - Async methods for I/O operations (network calls to live brokers)
    - Sync methods for in-memory operations (simulation, local state)
    - Event bus for pub/sub communication with strategies and monitoring
    
    Example:
        class MyBroker(BaseBrokerInterface):
            async def place_order(self, symbol, qty, side, **kwargs):
                # Validate
                self._validate_order(symbol, qty, side, **kwargs)
                
                # Execute
                result = await self._execute_order(...)
                
                # Emit event
                await self.emit_order_status(
                    status=OrderStatus.FILLED,
                    symbol=symbol,
                    side=side,
                    ...
                )
                
                return result
    """
    
    def __init__(self):
        """Initialize the broker interface with an event bus."""
        self.bus = EventHandler()
    
    # ========================================================================
    # ABSTRACT METHODS - ORDER EXECUTION
    # ========================================================================
    
    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        **kwargs
    ) -> OrderResult:
        """
        Place an order (async).
        
        Args:
            symbol: Trading symbol (e.g., "AAPL", "BTCUSD")
            qty: Quantity to trade (positive number)
            side: Order side (OrderSide.BUY or OrderSide.SELL)
            order_type: Order type (OrderType enum)
            limit_price: Limit price (required for limit/stop_limit orders)
            stop_price: Stop price (required for stop/stop_limit orders)
            time_in_force: Time in force (TimeInForce enum)
            **kwargs: Additional broker-specific parameters
            
        Returns:
            OrderResult with order_id, status, filled_qty, avg_price, etc.
            
        Raises:
            InvalidOrderError: If invalid parameters
            InsufficientFundsError: If insufficient account balance
            SymbolNotFoundError: If symbol is invalid
            BrokerError: If order placement fails
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> OrderResult:
        """
        Cancel an open order (async).
        
        Args:
            order_id: Unique order identifier
            
        Returns:
            OrderResult with cancellation status
            
        Raises:
            OrderNotFoundError: If order_id doesn't exist
            BrokerError: If cancellation fails
        """
        pass
    
    # ========================================================================
    # ABSTRACT METHODS - ASYNC ORDER HELPERS
    # ========================================================================

    @abstractmethod
    async def place_market_order(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        price: Optional[float] = None
    ) -> OrderResult:
        """
        Place a market order (async).

        Convenience method for strategies that need immediate execution.
        For simulation, this executes instantly. For live brokers, this
        awaits until order is accepted.

        Args:
            symbol: Trading symbol
            qty: Quantity (positive integer)
            side: OrderSide.BUY or OrderSide.SELL
            price: Optional price hint for simulation (ignored in live)

        Returns:
            OrderResult with execution details

        Raises:
            InvalidOrderError: If parameters invalid
            InsufficientFundsError: If insufficient funds
            BrokerError: If execution fails
        """
        pass

    @abstractmethod
    async def place_oco_order(
        self,
        symbol: str,
        qty: int,
        stop_price: float,
        limit_price: float
    ) -> OrderResult:
        """
        Place a One-Cancels-Other (OCO) bracket order (async).

        Creates two orders: a stop-loss and a take-profit. When one
        executes, the other is automatically cancelled.

        Args:
            symbol: Trading symbol
            qty: Quantity (positive integer)
            stop_price: Stop-loss trigger price
            limit_price: Take-profit limit price

        Returns:
            OrderResult for the OCO order group

        Raises:
            InvalidOrderError: If parameters invalid
            BrokerError: If execution fails

        Note:
            Not all brokers support OCO natively. Implementation may
            simulate this behavior with multiple orders.
        """
        pass

    # ========================================================================
    # ABSTRACT METHODS - ACCOUNT & POSITION QUERIES
    # ========================================================================
    
    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[PositionView]:
        """
        Get current position for a symbol (async).
        
        Args:
            symbol: Trading symbol
            
        Returns:
            PositionView with qty, avg_price, unrealized_pnl, etc.
            None if no position exists
        """
        pass
    
    @abstractmethod
    async def get_account_info(self) -> BrokerSnapshot:
        """
        Get current account state (async).
        
        Returns:
            BrokerSnapshot with cash, equity, buying_power, positions, etc.
        """
        pass
    
    @abstractmethod
    def get_available_funds(self) -> float:
        """
        Get available cash for trading (sync).
        
        Returns:
            Available cash balance
        """
        pass
    
    @abstractmethod
    def get_default_account(self) -> str:
        """
        Get default account identifier (sync).
        
        Returns:
            Account ID or name (e.g., "paper", "live", account number)
        """
        pass
    
    # ========================================================================
    # ABSTRACT METHODS - MARKET DATA & STATUS
    # ========================================================================
    
    @abstractmethod
    async def is_market_open(self) -> bool:
        """
        Check if market is currently open (async).
        
        Returns:
            True if market is open for trading, False otherwise
        """
        pass
    
    @abstractmethod
    def get_quote(self, symbol: str) -> float:
        """
        Get current market price for a symbol (sync).
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price (last trade, mid, or bid/ask depending on implementation)
            
        Raises:
            SymbolNotFoundError: If symbol is invalid
        """
        pass
    
    @abstractmethod
    def mark_price(self, symbol: str, price: float) -> None:
        """
        Update mark price for a symbol (sync).
        
        Used for mark-to-market equity calculations and P&L tracking.
        
        Args:
            symbol: Trading symbol
            price: New mark price
        """
        pass
    
    # ========================================================================
    # ABSTRACT METHODS - ORDER STATUS & MANAGEMENT
    # ========================================================================
    
    @abstractmethod
    async def get_open_orders(self) -> List[OrderResult]:
        """
        Get all open orders (async).
        
        Returns:
            List of OrderResult for all pending orders
        """
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderResult:
        """
        Get status of a specific order (async).
        
        Args:
            order_id: Unique order identifier
            
        Returns:
            OrderResult with current order status
            
        Raises:
            OrderNotFoundError: If order_id doesn't exist
        """
        pass
    
    # ========================================================================
    # CONNECTION MANAGEMENT (OPTIONAL)
    # ========================================================================
    
    async def connect(self) -> None:
        """
        Establish connection to broker (optional).
        
        Override this method if your broker requires explicit connection.
        Default implementation does nothing.
        """
        pass
    
    async def disconnect(self) -> None:
        """
        Close connection to broker (optional).
        
        Override this method if your broker requires explicit disconnection.
        Default implementation does nothing.
        """
        pass
    
    async def __aenter__(self):
        """Context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.disconnect()
    
    # ========================================================================
    # VALIDATION HELPERS
    # ========================================================================
    
    def _validate_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        Validate order parameters before submission.
        
        Args:
            symbol: Trading symbol
            qty: Quantity
            side: Order side
            order_type: Order type
            limit_price: Limit price (if applicable)
            stop_price: Stop price (if applicable)
            **kwargs: Additional parameters
            
        Raises:
            InvalidOrderError: If validation fails
        """
        # Validate quantity
        if qty <= 0:
            raise InvalidOrderError(f"Quantity must be positive, got {qty}")
        
        # Validate symbol
        if not symbol or not isinstance(symbol, str):
            raise InvalidOrderError(f"Invalid symbol: {symbol}")
        
        # Validate limit price for limit orders
        if order_type.requires_limit_price and limit_price is None:
            raise InvalidOrderError(
                f"{order_type.value} orders require limit_price"
            )
        
        # Validate stop price for stop orders
        if order_type.requires_stop_price and stop_price is None:
            raise InvalidOrderError(
                f"{order_type.value} orders require stop_price"
            )
        
        # Validate prices are positive
        if limit_price is not None and limit_price <= 0:
            raise InvalidOrderError(f"Limit price must be positive: {limit_price}")
        
        if stop_price is not None and stop_price <= 0:
            raise InvalidOrderError(f"Stop price must be positive: {stop_price}")
    
    # ========================================================================
    # EVENT EMISSION - PROTECTED METHODS
    # ========================================================================
    
    async def _emit_order_status(self, payload: Dict[str, Any]) -> None:
        """
        Emit raw order status event (protected).
        
        Args:
            payload: Event payload dictionary
        """
        await self.bus.emit(EVENT_ORDER_STATUS, payload)
    
    async def _emit_new_trade(self, payload: Dict[str, Any]) -> None:
        """
        Emit raw trade event (protected).
        
        Args:
            payload: Event payload dictionary
        """
        await self.bus.emit(EVENT_NEW_TRADE, payload)
    
    async def _emit_pnl_update(self, payload: Dict[str, Any]) -> None:
        """
        Emit raw P&L update event (protected).
        
        Args:
            payload: Event payload dictionary
        """
        await self.bus.emit(EVENT_PNL_UPDATE, payload)
    
    # ========================================================================
    # EVENT EMISSION - PUBLIC HELPERS
    # ========================================================================
    
    async def emit_order_status(
        self,
        status: OrderStatus,
        symbol: str,
        side: OrderSide,
        qty: float = 0.0,
        order_id: Optional[str] = None,
        filled_qty: float = 0.0,
        avg_price: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> None:
        """
        Emit normalized order status event.
        
        Args:
            status: Order status (OrderStatus enum)
            symbol: Trading symbol
            side: OrderSide.BUY or OrderSide.SELL
            qty: Order quantity
            order_id: Unique order identifier
            filled_qty: Quantity filled so far
            avg_price: Average fill price
            reason: Reason for rejection/cancellation (if applicable)
        """
        payload = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side.value,
            "qty": qty,
            "filled_qty": filled_qty,
            "avg_price": avg_price,
            "status": status.value,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self._emit_order_status(payload)
    
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
        Emit normalized trade execution event.
        
        Args:
            symbol: Trading symbol
            side: OrderSide.BUY or OrderSide.SELL
            qty: Trade quantity
            price: Execution price
            pnl: Realized P&L from this trade (if closing position)
            commission: Commission/fees paid
            order_id: Associated order ID
        """
        payload = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side.value,
            "qty": qty,
            "price": price,
            "pnl": pnl,
            "commission": commission,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self._emit_new_trade(payload)
    
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
        Emit normalized P&L update event.
        
        Args:
            portfolio_value: Total portfolio value (cash + positions)
            equity_curve: Historical equity curve values
            unrealized: Current unrealized P&L
            realized: Cumulative realized P&L
            drawdown: Current drawdown from peak
            cash: Available cash (optional, can be derived from portfolio_value)
        """
        payload = {
            "portfolio_value": portfolio_value,
            "equity_curve": equity_curve,
            "unrealized": unrealized,
            "realized": realized,
            "drawdown": drawdown,
            "cash": cash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self._emit_pnl_update(payload)
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def __repr__(self) -> str:
        """String representation of the broker."""
        return f"{self.__class__.__name__}(account={self.get_default_account()})"
