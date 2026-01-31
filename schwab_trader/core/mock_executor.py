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
from core.base.position_sizer_base import PositionSizerBase
from core.enums import OrderSide, OrderType, OrderStatus
from core.app_types import OrderResult
from loggers.logger import Logger
from core.events.eventhandler import EventHandler, get_event_handler
from core.events.events import (
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
        
        # Execute signal
        executor.execute(
            symbol="AAPL",
            df=historical_data,
            signal=1,
            price=150.25,
            atr_value=2.5
        )
    """
    
    def __init__(
        self,
        broker: BaseBrokerInterface,
        sizer: Optional[PositionSizerBase] = None,
        event_handler: Optional[EventHandler] = None
    ):
        """
        Initialize mock executor.
        
        Args:
            broker: Broker interface (typically MockBroker)
            sizer: Optional position sizer
            event_handler: Event bus for emissions
        """
        super().__init__()
        
        self.broker = broker
        self.sizer = sizer
        
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
    # CORE EXECUTION (BaseExecutor Implementation)
    # ========================================================================
    
    def execute(
        self,
        symbol: str,
        df: Optional[pd.DataFrame],
        signal: int,
        price: float,
        atr_value: float
    ) -> None:
        """
        Execute trade signal (mock/simulation).
        
        Args:
            symbol: Trading symbol
            df: Historical data (for context)
            signal: Signal (1=buy, -1=sell, 0=hold)
            price: Current price
            atr_value: ATR value
        """
        # Validate ATR
        if pd.isna(atr_value) or atr_value <= 0:
            self.logger.warning(
                f"[{symbol}] Invalid ATR: {atr_value}, skipping"
            )
            return
        
        # Skip hold signals
        if signal == 0:
            self.logger.debug(f"[{symbol}] HOLD signal - no action")
            return
        
        # Calculate position size
        qty = self._calculate_quantity(symbol, price, atr_value, signal)
        
        if qty <= 0:
            self.logger.warning(
                f"[{symbol}] Position size too small: {qty}"
            )
            return
        
        # Determine action based on signal and current position
        current_pos = self.positions.get(symbol, 0)
        
        if signal == 1 and current_pos == 0:
            # Open long
            self._place_order(symbol, qty, OrderSide.BUY, price)
            
        elif signal == -1 and current_pos > 0:
            # Close long
            self._place_order(symbol, current_pos, OrderSide.SELL, price)
        
        else:
            self.logger.debug(
                f"[{symbol}] No action for signal={signal}, pos={current_pos}"
            )
    
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
    
    def _calculate_quantity(
        self,
        symbol: str,
        price: float,
        atr: float,
        signal: int
    ) -> int:
        """Calculate position size."""
        if self.sizer is None:
            # Default: $10k position size
            return int(10000 / price)
        
        # Use position sizer
        stop_loss_price = price - (atr * 2) if signal == 1 else price + (atr * 2)
        
        try:
            cash = self.broker.get_available_funds()
        except Exception:
            cash = 100000  # Fallback
        
        qty = int(self.sizer.calculate_position_size(
            symbol=symbol,
            price=price,
            account_value=cash,
            atr=atr,
            stop_loss_price=stop_loss_price
        ))
        
        return max(qty, 0)
    
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