"""
Mock Executor - Simulated executor for backtesting and paper trading

Handles:
- Simulated trade execution
- Position tracking
- Event emission (trades, orders, P&L)
- Instant fills at market price
- No network latency
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from collections import defaultdict
from typing import Optional, Dict, Any
import pandas as pd

from core.base.executor_base import BaseExecutor
from core.base.base_broker_interface import BaseBrokerInterface
from core.enums import OrderSide, OrderType, OrderStatus
from core.app_types import OrderResult
from loggers.logger import Logger
from core.events.eventhandler import EventHandler, get_event_handler
from core.contracts.events import (
    EVENT_NEW_TRADE, EVENT_ORDER_STATUS, EVENT_PNL_UPDATE,
    EVENT_ALERT, EVENT_POSITION_UPDATE,
    TradePayload, OrderStatusPayload, PnLPayload,
    AlertPayload, PositionPayload
)


class MockExecutor(BaseExecutor):
    """
    Mock executor for simulation and testing.
    
    Features:
    - Simulated order execution
    - Instant fills (no latency)
    - Position tracking
    - Event emission for monitoring
    - Compatible with MockBroker

    Example:
        broker = MockBroker(starting_cash=100000)
        executor = MockExecutor(broker=broker)

        # Place orders via buy/sell
        executor.buy(symbol="AAPL", qty=10, price=150.25)
        executor.sell(symbol="AAPL", qty=10, price=155.00)
    """
    
    def __init__(
        self,
        broker: BaseBrokerInterface,
        event_handler: Optional[EventHandler] = None
    ):
        """
        Initialize mock executor.

        Args:
            broker: Broker interface (typically MockBroker)
            event_handler: Event bus for emissions

        Note: Position sizing is handled by the ExecutionEngine, not the Executor.
              The Executor is a thin adapter that just places orders with given qty.
        """
        super().__init__()

        self.broker = broker

        # Event bus
        self.bus = event_handler or get_event_handler()
        
        # Position tracking (local cache)
        self.positions: Dict[str, int] = defaultdict(int)
        self.entry_prices: Dict[str, float] = {}
        
        # Setup logging - own file with propagation to app.log
        self.logger = Logger(
            log_file="mock_executor.log",
            logger_name="MockExecutor",
            propagate=True
        ).get_logger()
        
        self.logger.info("MockExecutor initialized")
    
    # ========================================================================
    # ========================================================================
    # ORDER PLACEMENT (BaseExecutor Implementation)
    # ========================================================================

    def buy(self, symbol: str, qty: int, **kwargs) -> bool:
        """Buy (open long position)."""
        price = kwargs.get('price', 0.0)
        return self._place_order(symbol, qty, OrderSide.BUY, price)
    
    def sell(self, symbol: str, qty: int, **kwargs) -> bool:
        """Sell (close long position)."""
        price = kwargs.get('price', 0.0)
        return self._place_order(symbol, qty, OrderSide.SELL, price)
    
    def place_oco_order(
        self,
        symbol: str,
        qty: int,
        stop_price: float,
        limit_price: float
    ) -> bool:
        """
        Place OCO (one-cancels-other) order.
        
        Not implemented for mock.
        """
        self.logger.warning(
            f"[{symbol}] OCO orders not implemented in MockExecutor"
        )
        return False
    
    def get_open_orders(self, symbol: Optional[str] = None) -> list:
        """Get open orders (always empty for mock - instant fills)."""
        return []
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order (no-op for mock - instant fills)."""
        self.logger.info(f"Mock cancel order: {order_id}")
        return True
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status (always filled for mock)."""
        return OrderStatus.FILLED
    
    def log_order_response(self, response: Any) -> None:
        """Log order response."""
        self.logger.debug(f"Order response: {response}")
    
    def retry_failed_order(self, symbol: str, side: OrderSide, qty: int) -> bool:
        """Retry failed order (not needed for mock)."""
        return False
    
    # ========================================================================
    # ORDER PLACEMENT
    # ========================================================================
    
    def _place_order(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        price: float
    ) -> bool:
        """
        Place mock order with instant fill.
        
        Args:
            symbol: Trading symbol
            qty: Quantity
            side: Order side (BUY/SELL)
            price: Market price
            
        Returns:
            True if successful
        """
        now = datetime.now(timezone.utc)
        
        try:
            self.logger.info(
                f"[MOCK] Placing {side.value} order: {symbol} {qty}@${price:.2f}"
            )
            
            # Place order via broker (instant fill)
            result = self.broker.place_market_order(
                symbol=symbol,
                qty=qty,
                side=side,
                price=price
            )
            
            if not result.success:
                self.logger.error(
                    f"[{symbol}] Order failed: {result.message}"
                )
                self._emit_alert(
                    "error",
                    f"Order failed for {symbol}: {result.message}",
                    symbol
                )
                return False
            
            self.logger.info(
                f"[MOCK] Order filled: {side.value} {qty}@${price:.2f}"
            )
            
            # Update local position tracking
            self._update_position(symbol, side, qty, price)
            
            # Events are emitted by broker, but we can emit additional ones if needed
            
            return True
            
        except Exception as e:
            self.logger.error(f"[{symbol}] Order failed: {e}")
            self._emit_alert(
                "error",
                f"Order failed for {symbol}: {str(e)}",
                symbol
            )
            return False
    
    # ========================================================================
    # POSITION MANAGEMENT
    # ========================================================================
    
    def _update_position(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        price: float
    ) -> None:
        """Update local position tracking."""
        if side == OrderSide.BUY:
            old_qty = self.positions[symbol]
            new_qty = old_qty + qty
            
            # Update average entry price
            if old_qty > 0:
                total_cost = (self.entry_prices[symbol] * old_qty) + (price * qty)
                self.entry_prices[symbol] = total_cost / new_qty
            else:
                self.entry_prices[symbol] = price
            
            self.positions[symbol] = new_qty
            
        elif side == OrderSide.SELL:
            self.positions[symbol] -= qty
            
            if self.positions[symbol] <= 0:
                self.positions[symbol] = 0
                self.entry_prices[symbol] = 0.0
        
        self.logger.debug(
            f"[{symbol}] Position updated: {self.positions[symbol]} "
            f"@ ${self.entry_prices.get(symbol, 0.0):.2f}"
        )
    
    # ========================================================================
    # EVENT EMISSION
    # ========================================================================
    
    def _emit_alert(
        self,
        level: str,
        message: str,
        symbol: Optional[str] = None
    ) -> None:
        """Emit alert event."""
        payload: AlertPayload = {
            "level": level,
            "message": message,
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        asyncio.create_task(self.bus.emit(EVENT_ALERT, payload))