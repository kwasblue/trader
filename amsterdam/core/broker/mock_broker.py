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

import asyncio
import uuid
from datetime import datetime, timezone
from datetime import time as dt_time
from zoneinfo import ZoneInfo

from core.app_types import BrokerSnapshot, OrderResult, PositionView
from core.base.base_broker_interface import BaseBrokerInterface
from core.contracts.events import (
    EVENT_NEW_TRADE,
    EVENT_ORDER_STATUS,
    EVENT_PNL_UPDATE,
    EVENT_POSITION_UPDATE,
    EVENT_PRICE_UPDATE,
    OrderStatusPayload,
    PnLPayload,
    PositionPayload,
    PricePayload,
    TradePayload,
)
from core.enums import OrderSide, OrderStatus, OrderType, TimeInForce
from core.events.eventhandler import EventHandler, get_event_handler
from core.logic.portfolio_state import PortfolioState, SymbolPosition
from loggers.logger import Logger


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

    # US Eastern timezone for market hours
    ET = ZoneInfo("America/New_York")

    # Regular market hours (9:30 AM - 4:00 PM ET)
    MARKET_OPEN = dt_time(9, 30)
    MARKET_CLOSE = dt_time(16, 0)

    def __init__(
        self,
        starting_cash: float = 100_000.0,
        slippage: float = 0.0,
        commission: float = 0.0,
        event_handler: EventHandler | None = None,
        respect_market_hours: bool = False,
    ):
        """
        Initialize mock broker.

        Args:
            starting_cash: Initial cash balance
            slippage: Slippage as decimal (0.001 = 0.1%)
            commission: Commission per trade in dollars
            event_handler: Event bus for emissions
            respect_market_hours: If True, check actual market hours (ET timezone).
                                  If False, always return market open (default for testing).
        """
        super().__init__()
        self.respect_market_hours = respect_market_hours

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

        # Order tracking (all orders, historical)
        self.orders: dict[str, OrderResult] = {}

        # Open/pending orders (not yet filled)
        self._open_orders: dict[str, OrderResult] = {}

        # Setup logging - own file with propagation to app.log
        self.logger = Logger(log_file="mock_broker.log", logger_name="MockBroker", propagate=True).get_logger()

        self.logger.info(
            f"MockBroker initialized: cash=${starting_cash:,.2f}, "
            f"slippage={slippage:.4f}, commission=${commission:.2f}, "
            f"respect_market_hours={respect_market_hours}"
        )

    # ========================================================================
    # ORDER PLACEMENT (Async)
    # ========================================================================

    async def place_market_order(
        self, symbol: str, qty: int, side: OrderSide, price: float | None = None
    ) -> OrderResult:
        """
        Place market order with instant fill (async).

        Args:
            symbol: Trading symbol
            qty: Quantity (positive)
            side: OrderSide.BUY or OrderSide.SELL
            price: Market price (required for mock)

        Returns:
            OrderResult with fill details
        """
        if price is None:
            return OrderResult(success=False, message="MockBroker requires price parameter")

        if qty <= 0:
            return OrderResult(success=False, message=f"Invalid quantity: {qty}")

        # Convert OrderSide enum to string for internal use
        if side == OrderSide.BUY:
            side_str = "buy"
        elif side == OrderSide.SELL:
            side_str = "sell"
        else:
            return OrderResult(success=False, message=f"Invalid side: {side}")

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
                    message=f"Insufficient funds: need ${cost:.2f}, have ${self.portfolio.cash:.2f}",
                )
        else:
            # Check position exists for sell
            pos = self.portfolio.get_position(symbol)
            if not pos or pos.qty < qty:
                have = pos.qty if pos else 0
                return OrderResult(
                    success=False, order_id=order_id, message=f"Insufficient position: need {qty}, have {have}"
                )

        # Apply fill to portfolio (use string for portfolio)
        try:
            self.portfolio.apply_fill(symbol, side_str, qty, price)

            # Deduct commission
            self.portfolio.cash -= self.commission

        except Exception as e:
            return OrderResult(success=False, order_id=order_id, message=f"Fill failed: {e}")

        # Create result
        result = OrderResult(
            success=True,
            order_id=order_id,
            symbol=symbol,
            side=side,
            filled_qty=qty,
            avg_price=price,
            status=OrderStatus.FILLED,
            commission=self.commission,
        )

        # Store order
        self.orders[order_id] = result

        # Emit events (await instead of create_task)
        if self.event_handler:
            await self._emit_order_events(result)

        self.logger.info(f"[{symbol}] {side.value.upper()} {qty}@${price:.2f} (commission=${self.commission:.2f})")

        return result

    async def place_oco_order(self, symbol: str, qty: int, stop_price: float, limit_price: float) -> OrderResult:
        """
        Place OCO order (one-cancels-other).

        Creates two linked orders:
        - A stop order at stop_price
        - A limit order at limit_price

        When one fills, the other is automatically cancelled.

        Args:
            symbol: Trading symbol
            qty: Number of shares
            stop_price: Stop price for stop order leg
            limit_price: Limit price for limit order leg

        Returns:
            OrderResult with OCO order ID containing both legs
        """
        oco_id = f"mock_oco_{uuid.uuid4().hex[:8]}"
        stop_id = f"{oco_id}_stop"
        limit_id = f"{oco_id}_limit"

        # Create stop order leg
        stop_order = OrderResult(
            success=True,
            order_id=stop_id,
            symbol=symbol,
            side=OrderSide.SELL,
            filled_qty=0,
            avg_price=None,
            status=OrderStatus.PENDING,
            message="OCO stop leg",
        )
        stop_order.order_type = OrderType.STOP
        stop_order.stop_price = stop_price
        stop_order.limit_price = None
        stop_order.qty = qty
        stop_order.oco_pair_id = limit_id  # Link to partner

        # Create limit order leg
        limit_order = OrderResult(
            success=True,
            order_id=limit_id,
            symbol=symbol,
            side=OrderSide.SELL,
            filled_qty=0,
            avg_price=None,
            status=OrderStatus.PENDING,
            message="OCO limit leg",
        )
        limit_order.order_type = OrderType.LIMIT
        limit_order.limit_price = limit_price
        limit_order.stop_price = None
        limit_order.qty = qty
        limit_order.oco_pair_id = stop_id  # Link to partner

        # Track both orders
        self._open_orders[stop_id] = stop_order
        self._open_orders[limit_id] = limit_order
        self.orders[stop_id] = stop_order
        self.orders[limit_id] = limit_order

        # Emit order status events
        if self.event_handler:
            for order in [stop_order, limit_order]:
                payload: OrderStatusPayload = {
                    "order_id": order.order_id,
                    "symbol": symbol,
                    "status": "pending",
                    "filled_qty": 0.0,
                    "avg_price": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await self.event_handler.emit(EVENT_ORDER_STATUS, payload)

        self.logger.info(
            f"[{symbol}] OCO order placed: stop=${stop_price:.2f} ({stop_id}), limit=${limit_price:.2f} ({limit_id})"
        )

        # Return parent OCO result
        return OrderResult(
            success=True, order_id=oco_id, symbol=symbol, message=f"OCO order: stop={stop_id}, limit={limit_id}"
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
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        **kwargs,
    ) -> OrderResult:
        """
        Place order with support for market, limit, stop, and stop-limit orders.

        Args:
            symbol: Trading symbol
            qty: Quantity
            side: BUY or SELL
            order_type: MARKET, LIMIT, STOP, or STOP_LIMIT
            limit_price: Limit price (required for LIMIT and STOP_LIMIT)
            stop_price: Stop trigger price (required for STOP and STOP_LIMIT)
            time_in_force: GTC, DAY, etc.

        Returns:
            OrderResult with order details
        """
        price = kwargs.get("price")

        if order_type == OrderType.MARKET:
            # Market orders fill immediately
            fill_price = limit_price or price
            return await self.place_market_order(symbol, int(qty), side, fill_price)

        # Generate order ID for pending orders
        order_id = f"mock_{uuid.uuid4().hex[:8]}"

        # Validate required prices for order types
        if order_type == OrderType.LIMIT and limit_price is None:
            return OrderResult(success=False, order_id=order_id, message="Limit price required for LIMIT orders")

        if order_type == OrderType.STOP and stop_price is None:
            return OrderResult(success=False, order_id=order_id, message="Stop price required for STOP orders")

        if order_type == OrderType.STOP_LIMIT and (stop_price is None or limit_price is None):
            return OrderResult(
                success=False, order_id=order_id, message="Both stop and limit prices required for STOP_LIMIT orders"
            )

        # Create pending order
        pending_order = OrderResult(
            success=True,
            order_id=order_id,
            symbol=symbol,
            side=side,
            filled_qty=0,
            avg_price=None,
            status=OrderStatus.PENDING,
            message=f"Order pending: {order_type.value}",
        )

        # Store additional order info for later processing
        pending_order.order_type = order_type
        pending_order.limit_price = limit_price
        pending_order.stop_price = stop_price
        pending_order.qty = int(qty)
        pending_order.time_in_force = time_in_force
        pending_order.stop_triggered = False  # For STOP_LIMIT tracking

        # Track as open order
        self._open_orders[order_id] = pending_order
        self.orders[order_id] = pending_order

        # Emit order status event
        if self.event_handler:
            payload: OrderStatusPayload = {
                "order_id": order_id,
                "symbol": symbol,
                "status": "pending",
                "filled_qty": 0.0,
                "avg_price": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await self.event_handler.emit(EVENT_ORDER_STATUS, payload)

        self.logger.info(
            f"[{symbol}] {order_type.value} {side.value} {qty} limit={limit_price} stop={stop_price} -> {order_id}"
        )

        return pending_order

    async def check_pending_orders(self, symbol: str, current_price: float) -> list[OrderResult]:
        """
        Check pending orders against current price and fill if conditions are met.

        Args:
            symbol: Symbol to check
            current_price: Current market price

        Returns:
            List of orders that were filled
        """
        filled_orders = []

        # Get pending orders for this symbol
        pending = [
            (oid, order)
            for oid, order in self._open_orders.items()
            if order.symbol == symbol and order.status == OrderStatus.PENDING
        ]

        for order_id, order in pending:
            should_fill = False
            fill_price = current_price
            order_type = getattr(order, "order_type", OrderType.MARKET)
            limit_price = getattr(order, "limit_price", None)
            stop_price = getattr(order, "stop_price", None)
            stop_triggered = getattr(order, "stop_triggered", False)

            if order_type == OrderType.LIMIT:
                # LIMIT: Fill when price reaches limit
                if order.side == OrderSide.BUY and current_price <= limit_price:
                    should_fill = True
                    fill_price = limit_price  # Fill at limit price
                elif order.side == OrderSide.SELL and current_price >= limit_price:
                    should_fill = True
                    fill_price = limit_price

            elif order_type == OrderType.STOP:
                # STOP: Convert to market when stop price hit
                if order.side == OrderSide.BUY and current_price >= stop_price:
                    should_fill = True
                    fill_price = current_price  # Fill at market
                elif order.side == OrderSide.SELL and current_price <= stop_price:
                    should_fill = True
                    fill_price = current_price

            elif order_type == OrderType.STOP_LIMIT:
                # STOP_LIMIT: Two-phase - first stop triggers, then limit
                if not stop_triggered:
                    # Check if stop should trigger
                    if order.side == OrderSide.BUY and current_price >= stop_price:
                        order.stop_triggered = True
                        self.logger.info(f"[{symbol}] Stop triggered for {order_id}")
                    elif order.side == OrderSide.SELL and current_price <= stop_price:
                        order.stop_triggered = True
                        self.logger.info(f"[{symbol}] Stop triggered for {order_id}")

                if order.stop_triggered:
                    # Stop has triggered, now act like a limit order
                    if order.side == OrderSide.BUY and current_price <= limit_price:
                        should_fill = True
                        fill_price = limit_price
                    elif order.side == OrderSide.SELL and current_price >= limit_price:
                        should_fill = True
                        fill_price = limit_price

            # Execute fill if conditions met
            if should_fill:
                qty = getattr(order, "qty", order.filled_qty)
                fill_result = await self.place_market_order(symbol, qty, order.side, fill_price)

                if fill_result.success:
                    # Update original order with fill details
                    order.status = OrderStatus.FILLED
                    order.filled_qty = qty
                    order.avg_price = fill_price
                    order.message = f"Filled at ${fill_price:.2f}"

                    # Remove from open orders
                    del self._open_orders[order_id]
                    filled_orders.append(order)

                    self.logger.info(f"[{symbol}] {order_type.value} filled: {order_id} @ ${fill_price:.2f}")

                    # Handle OCO: Cancel the partner order
                    oco_pair_id = getattr(order, "oco_pair_id", None)
                    if oco_pair_id and oco_pair_id in self._open_orders:
                        partner_order = self._open_orders[oco_pair_id]
                        partner_order.status = OrderStatus.CANCELLED
                        partner_order.message = "Cancelled: OCO partner filled"
                        del self._open_orders[oco_pair_id]

                        self.logger.info(f"[{symbol}] OCO partner cancelled: {oco_pair_id}")

                        # Emit cancellation event
                        if self.event_handler:
                            cancel_payload: OrderStatusPayload = {
                                "order_id": oco_pair_id,
                                "symbol": symbol,
                                "status": "cancelled",
                                "filled_qty": 0.0,
                                "avg_price": None,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                            await self.event_handler.emit(EVENT_ORDER_STATUS, cancel_payload)

        return filled_orders

    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel order by ID."""
        # Check both open orders and historical orders
        order = self._open_orders.get(order_id) or self.orders.get(order_id)

        if order is None:
            return OrderResult(success=False, order_id=order_id, message="Order not found")

        # Update order status
        order.status = OrderStatus.CANCELLED

        # Remove from open orders if present
        if order_id in self._open_orders:
            del self._open_orders[order_id]

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

        return OrderResult(success=True, order_id=order_id, status=OrderStatus.CANCELLED)

    # ========================================================================
    # POSITION & ACCOUNT INFO
    # ========================================================================

    async def get_position(self, symbol: str) -> PositionView | None:
        """Get current position for symbol."""
        pos = self.portfolio.get_position(symbol)

        if not pos or pos.is_flat:
            return None

        return PositionView(
            symbol=symbol,
            qty=pos.qty,
            avg_entry_price=pos.avg_price,
            market_price=pos.last_price,
            last_price=pos.last_price,
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
                    last_price=pos.last_price,
                )

        return BrokerSnapshot(
            cash=self.portfolio.cash,
            equity=self.portfolio.total_equity(),
            positions=position_views,
            portfolio_value=self.portfolio.total_equity(),
        )

    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        """
        Get open orders (pending, not yet filled).

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of open orders
        """
        if symbol:
            return [order for order in self._open_orders.values() if order.symbol == symbol]
        return list(self._open_orders.values())

    async def get_order_status(self, order_id: str) -> OrderResult:
        """Get order status by ID."""
        if order_id not in self.orders:
            return OrderResult(success=False, order_id=order_id, message="Order not found")

        return self.orders[order_id]

    # ========================================================================
    # MARKET DATA & STATUS
    # ========================================================================

    async def is_market_open(self) -> bool:
        """
        Check if market is open.

        When respect_market_hours is False (default), always returns True.
        When respect_market_hours is True, uses ET timezone logic:
        - Market hours: 9:30 AM - 4:00 PM ET, Mon-Fri
        - Returns False on weekends and outside hours
        """
        if not self.respect_market_hours:
            return True

        now_et = datetime.now(self.ET)

        # Check if weekend (Saturday=5, Sunday=6)
        if now_et.weekday() >= 5:
            return False

        # Check market hours
        current_time = now_et.time()
        return self.MARKET_OPEN <= current_time < self.MARKET_CLOSE

    def get_quote(self, symbol: str) -> float:
        """Get last known price for symbol."""
        pos = self.portfolio.get_position(symbol)
        if pos:
            return pos.last_price
        return 0.0

    def get_available_funds(self) -> float:
        """Get available cash."""
        return self.portfolio.cash

    def get_buying_power(self) -> float:
        """
        Get buying power for trading.

        For mock broker, returns cash (simulates no leverage/margin).
        """
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

        Also checks pending orders to see if any should be filled.

        Args:
            symbol: Trading symbol
            price: Current market price
        """
        # Update portfolio
        self.portfolio.update_price(symbol, price)

        # Check pending orders for this symbol
        asyncio.create_task(self.check_pending_orders(symbol, price))

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

    async def _emit_position_update(self, symbol: str, pos: SymbolPosition) -> None:
        """Emit position update event."""
        side = "long" if pos.qty > 0 else "short" if pos.qty < 0 else "flat"
        payload: PositionPayload = {
            "symbol": symbol,
            "qty": pos.qty,
            "avg_price": pos.avg_price,
            "avg": pos.avg_price,  # GUI model uses 'avg'
            "last": pos.last_price,  # GUI model uses 'last'
            "unrealized": pos.unrealized_pnl,
            "unreal": pos.unrealized_pnl,  # GUI model uses 'unreal'
            "realized": pos.realized_pnl,
            "side": side,
            "market_value": pos.qty * pos.last_price,
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

    def reset(self, starting_cash: float | None = None) -> None:
        """
        Reset broker to initial state.

        Useful for running multiple simulations.
        """
        cash = starting_cash or 100_000.0
        self.portfolio = PortfolioState(cash=cash)
        self.orders.clear()
        self._open_orders.clear()

        self.logger.info(f"Broker reset with ${cash:,.2f}")

    def __repr__(self) -> str:
        return (
            f"MockBroker("
            f"cash=${self.portfolio.cash:,.2f}, "
            f"equity=${self.portfolio.total_equity():,.2f}, "
            f"positions={self.portfolio.num_positions})"
        )
