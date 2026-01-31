"""
Mock Broker - Simulated broker for backtesting and paper trading

Features:
- Instant fills at market price
- Position tracking
- Event emission for all actions
- Portfolio state synchronization
- No slippage/commission (configurable)
"""

from __future__ import annotations

import uuid
import asyncio
from datetime import datetime, timezone
from typing import Dict, Optional, List
import logging

from core.base.base_broker_interface import BaseBrokerInterface
from core.logic.portfolio_state import PortfolioState, SymbolPosition
from core.app_types import OrderResult, PositionView, BrokerSnapshot
from core.enums import OrderSide, OrderStatus, OrderType, TimeInForce
from loggers.logger import Logger
from core.events.eventhandler import EventHandler, get_event_handler
from core.events.events import (
    EVENT_ORDER_STATUS, OrderStatusPayload,
    EVENT_NEW_TRADE, TradePayload,
    EVENT_POSITION_UPDATE, PositionPayload,
    EVENT_PNL_UPDATE, PnLPayload,
    EVENT_PRICE_UPDATE, PricePayload
)

class MockBroker(BaseBrokerInterface):
    """
    Mock broker for simulation and testing.
    
    Provides:
    - Instant order fills (no latency)
    - Perfect execution (no slippage by default)
    - Event emission for monitoring
    - Position tracking
    - Portfolio state integration
    
    Example:
        broker = MockBroker(
            starting_cash=100000,
            slippage=0.001,  # 0.1% slippage
            commission=1.0    # $1 per trade
        )
        
        # Place order
        result = broker.place_market_order("AAPL", 100, "buy", price=150.0)
        
        # Check position
        position = await broker.get_position("AAPL")
        
        # Get account info
        account = await broker.get_account_info()
    """
    
    def __init__(
        self,
        starting_cash: float = 100_000.0,
        slippage: float = 0.0,
        commission: float = 0.0,
        event_handler: Optional[EventHandler] = None
    ):
        """
        Initialize mock broker.
        
        Args:
            starting_cash: Initial cash balance
            slippage: Slippage as decimal (0.001 = 0.1%)
            commission: Commission per trade in dollars
            event_handler: Event bus for emissions
        """
        super().__init__()
        
        # Portfolio state (single source of truth)
        self.portfolio = PortfolioState(cash=starting_cash)
        
        # Trading costs
        self.slippage = slippage
        self.commission = commission
        
        # Event handler
        self.event_handler = event_handler or get_event_handler()

        # Equity history for PnL events
        from collections import deque
        self.equity_history: deque = deque(maxlen=1000)
        self.equity_history.append(starting_cash)

        # Order tracking
        self.orders: Dict[str, OrderResult] = {}
        
        # Setup logging - own file with propagation to app.log
        self.logger = Logger(
            log_file="mock_broker.log",
            logger_name="MockBroker",
            propagate=True
        ).get_logger()
        
        self.logger.info(
            f"MockBroker initialized: cash=${starting_cash:,.2f}, "
            f"slippage={slippage:.4f}, commission=${commission:.2f}"
        )
    
    # ========================================================================
    # ORDER PLACEMENT (Sync - for backward compatibility)
    # ========================================================================
    
    def place_market_order(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        price: Optional[float] = None
    ) -> OrderResult:
        """
        Place market order with instant fill.
        
        Args:
            symbol: Trading symbol
            qty: Quantity (positive)
            side: OrderSide.BUY or OrderSide.SELL
            price: Market price (required for mock)
            
        Returns:
            OrderResult with fill details
        """
        if price is None:
            return OrderResult(
                success=False,
                message="MockBroker requires price parameter"
            )
        
        if qty <= 0:
            return OrderResult(
                success=False,
                message=f"Invalid quantity: {qty}"
            )
        
        # Convert OrderSide enum to string for internal use
        if side == OrderSide.BUY:
            side_str = "buy"
        elif side == OrderSide.SELL:
            side_str = "sell"
        else:
            return OrderResult(
                success=False,
                message=f"Invalid side: {side}"
            )
        
        # Apply slippage
        if self.slippage > 0:
            if side == OrderSide.BUY:
                # Slippage increases buy price
                price = price * (1 + self.slippage)
            else:
                # Slippage decreases sell price
                price = price * (1 - self.slippage)
        
        # Generate order ID
        order_id = f"mock_{uuid.uuid4().hex[:8]}"
        
        # Check affordability
        if side == OrderSide.BUY:
            cost = (qty * price) + self.commission
            if cost > self.portfolio.cash:
                return OrderResult(
                    success=False,
                    order_id=order_id,
                    message=f"Insufficient funds: need ${cost:.2f}, have ${self.portfolio.cash:.2f}"
                )
        else:
            # Check position exists for sell
            pos = self.portfolio.get_position(symbol)
            if not pos or pos.qty < qty:
                have = pos.qty if pos else 0
                return OrderResult(
                    success=False,
                    order_id=order_id,
                    message=f"Insufficient position: need {qty}, have {have}"
                )
        
        # Apply fill to portfolio (use string for portfolio)
        try:
            self.portfolio.apply_fill(symbol, side_str, qty, price)
            
            # Deduct commission
            self.portfolio.cash -= self.commission
            
        except Exception as e:
            return OrderResult(
                success=False,
                order_id=order_id,
                message=f"Fill failed: {e}"
            )
        
        # Create result
        result = OrderResult(
            success=True,
            order_id=order_id,
            symbol=symbol,
            side=side,
            filled_qty=qty,
            avg_price=price,
            status=OrderStatus.FILLED,
            timestamp=datetime.now(timezone.utc),
            commission=self.commission
        )
        
        # Store order
        self.orders[order_id] = result
        
        # Emit events (fire and forget)
        if self.event_handler:
            asyncio.create_task(self._emit_order_events(result))
        
        self.logger.info(
            f"[{symbol}] {side.value.upper()} {qty}@${price:.2f} "
            f"(commission=${self.commission:.2f})"
        )
        
        return result
    
    def place_oco_order(
        self,
        symbol: str,
        qty: int,
        stop_price: float,
        limit_price: float
    ) -> OrderResult:
        """
        Place OCO order (one-cancels-other).
        
        Not implemented in mock - just returns success.
        """
        order_id = f"mock_oco_{uuid.uuid4().hex[:8]}"
        
        self.logger.warning(
            f"[{symbol}] OCO order placed but not tracked "
            f"(stop=${stop_price:.2f}, limit=${limit_price:.2f})"
        )
        
        return OrderResult(
            success=True,
            order_id=order_id,
            message="OCO accepted but not tracked in mock"
        )
    
    # ========================================================================
    # ORDER PLACEMENT (Async)
    # ========================================================================
    
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
        Async order placement.
        
        Only market orders fully supported in mock.
        """
        if order_type == OrderType.MARKET:
            price = limit_price or kwargs.get('price')
            return self.place_market_order(symbol, int(qty), side, price)
        
        # Other order types not implemented
        return OrderResult(
            success=False,
            message=f"Order type '{order_type.value}' not supported in mock"
        )
    
    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel order by ID."""
        if order_id not in self.orders:
            return OrderResult(
                success=False,
                order_id=order_id,
                message="Order not found"
            )
        
        # Update order status
        order = self.orders[order_id]
        order.status = OrderStatus.CANCELLED
        
        # Emit cancellation event
        if self.event_handler:
            payload: OrderStatusPayload = {
                "order_id": order_id,
                "symbol": order.symbol or "N/A",
                "status": "cancelled",
                "filled_qty": 0.0,
                "avg_price": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await self.event_handler.emit(EVENT_ORDER_STATUS, payload)
        
        self.logger.info(f"Order cancelled: {order_id}")
        
        return OrderResult(
            success=True,
            order_id=order_id,
            status=OrderStatus.CANCELLED
        )
    
    # ========================================================================
    # POSITION & ACCOUNT INFO
    # ========================================================================
    
    async def get_position(self, symbol: str) -> Optional[PositionView]:
        """Get current position for symbol."""
        pos = self.portfolio.get_position(symbol)
        
        if not pos or pos.is_flat:
            return None
        
        return PositionView(
            symbol=symbol,
            qty=pos.qty,
            avg_entry_price=pos.avg_price,
            market_price=pos.last_price,
            last_price=pos.last_price
        )
    
    async def get_account_info(self) -> BrokerSnapshot:
        """Get account snapshot."""
        # Build position views
        position_views = {}
        for symbol, pos in self.portfolio.positions.items():
            if not pos.is_flat:
                position_views[symbol] = PositionView(
                    symbol=symbol,
                    qty=pos.qty,
                    avg_entry_price=pos.avg_price,
                    market_price=pos.last_price,
                    last_price=pos.last_price
                )
        
        return BrokerSnapshot(
            cash=self.portfolio.cash,
            equity=self.portfolio.total_equity(),
            positions=position_views,
            portfolio_value=self.portfolio.total_equity()
        )
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        """
        Get open orders.
        
        Mock broker doesn't track open orders (instant fills).
        """
        return []
    
    async def get_order_status(self, order_id: str) -> OrderResult:
        """Get order status by ID."""
        if order_id not in self.orders:
            return OrderResult(
                success=False,
                order_id=order_id,
                message="Order not found"
            )
        
        return self.orders[order_id]
    
    # ========================================================================
    # MARKET DATA & STATUS
    # ========================================================================
    
    async def is_market_open(self) -> bool:
        """Check if market is open (always true for mock)."""
        return True
    
    def get_quote(self, symbol: str) -> float:
        """Get last known price for symbol."""
        pos = self.portfolio.get_position(symbol)
        if pos:
            return pos.last_price
        return 0.0
    
    def get_available_funds(self) -> float:
        """Get available cash."""
        return self.portfolio.cash
    
    def has_sufficient_funds(self, symbol: str, qty: int) -> bool:
        """Check if enough cash for order."""
        # Simplified check using last known price
        pos = self.portfolio.get_position(symbol)
        price = pos.last_price if pos else 100.0  # Default estimate
        
        cost = (qty * price) + self.commission
        return self.portfolio.cash >= cost
    
    def get_default_account(self) -> str:
        """Get default account ID."""
        return "MOCK-ACCOUNT"
    
    # ========================================================================
    # MARK-TO-MARKET
    # ========================================================================
    
    def mark_price(self, symbol: str, price: float) -> None:
        """
        Update market price for position (mark-to-market).
        
        Updates portfolio and emits events.
        
        Args:
            symbol: Trading symbol
            price: Current market price
        """
        # Update portfolio
        self.portfolio.update_price(symbol, price)
        
        # Emit price update event (async task)
        if self.event_handler:
            price_payload: PricePayload = {
                "symbol": symbol,
                "price": float(price),
                "ma20": None,
                "ma50": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            asyncio.create_task(self.event_handler.emit(EVENT_PRICE_UPDATE, price_payload))
            
            # If we have a position, emit position and P&L updates
            pos = self.portfolio.get_position(symbol)
            if pos and not pos.is_flat:
                asyncio.create_task(self._emit_position_update(symbol, pos))
                asyncio.create_task(self._emit_pnl_update())
    
    # ========================================================================
    # EVENT EMISSION
    # ========================================================================
    
    async def _emit_order_events(self, result: OrderResult) -> None:
        """Emit all events for an order fill."""
        now = datetime.now(timezone.utc).isoformat()
        
        try:
            # Order status event
            order_status: OrderStatusPayload = {
                "order_id": result.order_id,
                "symbol": result.symbol or "N/A",
                "status": result.status.value if result.status else "filled",
                "filled_qty": float(result.filled_qty),
                "avg_price": float(result.avg_price),
                "timestamp": now,
            }
            await self.event_handler.emit(EVENT_ORDER_STATUS, order_status)
            
            # Trade event
            trade: TradePayload = {
                "symbol": result.symbol,
                "side": result.side.value.lower() if result.side else "unknown",
                "qty": float(result.filled_qty),
                "price": float(result.avg_price),
                "timestamp": now,
                "pnl": None,
            }
            await self.event_handler.emit(EVENT_NEW_TRADE, trade)
            
            # Position update
            pos = self.portfolio.get_position(result.symbol)
            if pos:
                await self._emit_position_update(result.symbol, pos)
            
            # P&L update
            await self._emit_pnl_update()
            
        except Exception as e:
            self.logger.error(f"Failed to emit order events: {e}")
    
    async def _emit_position_update(
        self,
        symbol: str,
        pos: SymbolPosition
    ) -> None:
        """Emit position update event."""
        payload: PositionPayload = {
            "symbol": symbol,
            "qty": pos.qty,
            "avg_price": pos.avg_price,
            "unrealized": pos.unrealized_pnl,
            "realized": pos.realized_pnl,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self.event_handler.emit(EVENT_POSITION_UPDATE, payload)
    
    async def _emit_pnl_update(self) -> None:
        """Emit P&L update event."""
        equity = self.portfolio.total_equity()
        self.equity_history.append(equity)
        payload: PnLPayload = {
            "portfolio_value": equity,
            "equity_curve": list(self.equity_history),
            "unrealized": self.portfolio.total_unrealized(),
            "realized": self.portfolio.total_realized(),
            "drawdown": self.portfolio.current_drawdown(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self.event_handler.emit(EVENT_PNL_UPDATE, payload)
    
    # ========================================================================
    # UTILITY
    # ========================================================================
    
    def reset(self, starting_cash: Optional[float] = None) -> None:
        """
        Reset broker to initial state.
        
        Useful for running multiple simulations.
        """
        cash = starting_cash or 100_000.0
        self.portfolio = PortfolioState(cash=cash)
        self.orders.clear()
        
        self.logger.info(f"Broker reset with ${cash:,.2f}")
    
    def __repr__(self) -> str:
        return (
            f"MockBroker("
            f"cash=${self.portfolio.cash:,.2f}, "
            f"equity=${self.portfolio.total_equity():,.2f}, "
            f"positions={self.portfolio.num_positions})"
        )
