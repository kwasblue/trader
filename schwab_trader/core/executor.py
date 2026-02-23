"""
Live Executor - Thin wrapper for broker order operations.

This is a minimal adapter between the execution engine and the broker.
All business logic (sizing, position tracking, event emission) belongs
in LiveExecutionEngine.

Responsibilities:
- Place orders via broker
- Cancel orders
- Query order status
- Query open orders
"""

from __future__ import annotations

import uuid
from typing import Optional, List, Any, Dict
from core.base.executor_base import BaseExecutor
from core.base.base_broker_interface import BaseBrokerInterface
from core.enums import OrderSide, OrderType, OrderStatus
from core.app_types import OrderResult
from loggers.logger import Logger


class LiveExecutor(BaseExecutor):
    """
    Thin wrapper for broker order operations.

    This executor is intentionally minimal - it's just an adapter
    for broker operations. All trade logic, position tracking, and
    event emission belongs in LiveExecutionEngine.

    Example:
        executor = LiveExecutor(broker=alpaca_broker, dry_run=False)

        # Place order
        result = executor.place_order("AAPL", 100, OrderSide.BUY)

        # Cancel order
        success = executor.cancel_order("order_123")

        # Query orders
        orders = executor.get_open_orders("AAPL")
    """

    def __init__(
        self,
        broker: BaseBrokerInterface,
        dry_run: bool = False
    ):
        """
        Initialize live executor.

        Args:
            broker: Broker interface for order execution
            dry_run: If True, log orders but don't execute
        """
        super().__init__()

        self.broker = broker
        self.dry_run = dry_run

        # Setup logging
        self.logger = Logger("app.log", "LiveExecutor").get_logger()

        self.logger.info(f"LiveExecutor initialized (dry_run={dry_run})")

    # ========================================================================
    # ORDER PLACEMENT
    # ========================================================================

    def place_order(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        **kwargs
    ) -> Optional[OrderResult]:
        """
        Place an order via broker.

        Args:
            symbol: Trading symbol
            qty: Quantity
            side: Order side (BUY/SELL)
            order_type: Order type (MARKET, LIMIT, etc.)
            **kwargs: Additional order parameters
                - limit_price: For LIMIT orders
                - stop_price: For STOP orders
                - time_in_force: DAY, GTC, etc.

        Returns:
            OrderResult if successful, None if failed
        """
        if self.dry_run:
            price = kwargs.get('price', 0.0)
            self.logger.info(
                f"[DRY RUN] {side.value} {qty} {symbol} @ ${price:.2f}"
            )
            return OrderResult(
                order_id=f"dry_run_{uuid.uuid4()}",
                symbol=symbol,
                side=side,
                filled_qty=qty,
                avg_price=price,
                status="filled",  # Mark as filled for bool check
            )

        try:
            self.logger.info(
                f"[LIVE] Placing {order_type.value} {side.value} order: "
                f"{symbol} {qty} shares"
            )

            # Build order kwargs
            order_kwargs = {
                "symbol": symbol,
                "qty": qty,
                "side": side.value.lower(),
            }

            # Add order type specific params
            if order_type == OrderType.LIMIT:
                order_kwargs["limit_price"] = kwargs.get("limit_price")
            elif order_type == OrderType.STOP:
                order_kwargs["stop_price"] = kwargs.get("stop_price")
            elif order_type == OrderType.STOP_LIMIT:
                order_kwargs["limit_price"] = kwargs.get("limit_price")
                order_kwargs["stop_price"] = kwargs.get("stop_price")

            if "time_in_force" in kwargs:
                order_kwargs["time_in_force"] = kwargs["time_in_force"]

            # Place via broker
            if order_type == OrderType.MARKET:
                response = self.broker.place_market_order(
                    symbol=symbol,
                    qty=qty,
                    side=side.value.lower()
                )
            else:
                response = self.broker.place_order(**order_kwargs)

            # Extract order info from response
            order_id = self._extract_order_id(response)
            avg_price = kwargs.get('price', 0.0)

            self.logger.info(f"[{symbol}] Order placed: {order_id}")

            return OrderResult(
                order_id=order_id,
                symbol=symbol,
                side=side,
                filled_qty=qty,
                avg_price=avg_price,
                status="filled",  # Mark as filled for bool check
            )

        except Exception as e:
            self.logger.error(f"[{symbol}] Order failed: {e}")
            return None

    def buy(self, symbol: str, qty: int, **kwargs) -> bool:
        """Buy (open long position)."""
        result = self.place_order(symbol, qty, OrderSide.BUY, **kwargs)
        return result is not None

    def sell(self, symbol: str, qty: int, **kwargs) -> bool:
        """Sell (close long position)."""
        result = self.place_order(symbol, qty, OrderSide.SELL, **kwargs)
        return result is not None

    def place_oco_order(
        self,
        symbol: str,
        qty: int,
        stop_price: float,
        limit_price: float
    ) -> bool:
        """
        Place OCO (one-cancels-other) order.

        Not all brokers support this natively.
        """
        try:
            if hasattr(self.broker, 'place_oco_order'):
                self.broker.place_oco_order(
                    symbol=symbol,
                    qty=qty,
                    stop_price=stop_price,
                    limit_price=limit_price
                )
                return True
            else:
                self.logger.warning(
                    f"[{symbol}] OCO orders not supported by this broker"
                )
                return False
        except Exception as e:
            self.logger.error(f"[{symbol}] OCO order failed: {e}")
            return False

    # ========================================================================
    # ORDER MANAGEMENT
    # ========================================================================

    def cancel_order(self, order_id: str) -> bool:
        """Cancel order by ID."""
        try:
            self.broker.cancel_order(order_id)
            self.logger.info(f"Cancelled order: {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Any]:
        """Get open orders from broker."""
        try:
            return self.broker.get_open_orders(symbol)
        except Exception as e:
            self.logger.error(f"Failed to get open orders: {e}")
            return []

    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get status of order."""
        try:
            order = self.broker.get_order(order_id)
            if hasattr(order, 'status'):
                return OrderStatus(order.status)
            return None
        except Exception as e:
            self.logger.error(f"Failed to get order status: {e}")
            return None

    # ========================================================================
    # HELPERS
    # ========================================================================

    def _extract_order_id(self, response: Any) -> str:
        """Extract order ID from broker response."""
        if hasattr(response, "order_id"):
            return str(response.order_id)
        elif hasattr(response, "id"):
            return str(response.id)
        elif isinstance(response, dict):
            return str(response.get("id", response.get("order_id", f"order_{uuid.uuid4()}")))
        else:
            return f"order_{uuid.uuid4()}"

    def log_order_response(self, response: Any) -> None:
        """Log order response."""
        self.logger.info(f"Order response: {response}")

    def retry_failed_order(self, symbol: str, side: OrderSide, qty: int) -> bool:
        """Retry failed order (not implemented)."""
        self.logger.warning("Order retry not implemented")
        return False

