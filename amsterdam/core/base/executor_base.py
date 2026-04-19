"""
Base Executor - Abstract base class for broker order operations.

This module defines the contract for executor implementations that handle
order placement and management. Executors are thin adapters for broker operations.

NOTE: Signal processing and trade logic belong in ExecutionEngine, not Executor.
The executor is responsible only for:
- Placing orders via broker
- Cancelling orders
- Querying order status
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.enums import OrderType


class BaseExecutor(ABC):
    """
    Abstract base class for broker order operations.

    Executors are thin adapters for broker operations:
    - Place orders (market, limit, stop)
    - Cancel orders
    - Query order status

    NOTE: Signal processing, position tracking, and event emission
    belong in ExecutionEngine, not here. The executor is just a
    broker adapter.

    Design Philosophy:
    - ExecutionEngine → handles signals, sizing, events (orchestrator)
    - Executor → places/cancels orders via broker (adapter)
    - Broker → executes orders (implementation)

    Example:
        class MyExecutor(BaseExecutor):
            def buy(self, symbol, qty, **kwargs):
                return self.broker.place_market_order(symbol, qty, "buy")

            def sell(self, symbol, qty, **kwargs):
                return self.broker.place_market_order(symbol, qty, "sell")
    """

    def __init__(self):
        """Initialize executor."""
        pass

    # ========================================================================
    # ABSTRACT METHODS - ORDER PLACEMENT
    # ========================================================================

    @abstractmethod
    def buy(self, symbol: str, qty: int, order_type: OrderType = OrderType.MARKET, **kwargs) -> dict[str, Any]:
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
    def sell(self, symbol: str, qty: int, order_type: OrderType = OrderType.MARKET, **kwargs) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
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
    def retry_failed_order(self, *args, max_retries: int = 3, delay: float = 2.0, **kwargs) -> dict[str, Any]:
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
    # UTILITY METHODS
    # ========================================================================

    def __repr__(self) -> str:
        """String representation of executor."""
        return f"{self.__class__.__name__}()"
