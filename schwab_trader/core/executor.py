"""
Live Executor - Production trade executor with event handling

Handles:
- Trade execution via broker
- Position tracking
- Event emission (trades, orders, P&L)
- GUI event handling (manual orders, flatten, cancel)
- Risk management integration
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
    EVENT_ALERT, EVENT_MANUAL_ORDER, EVENT_FLATTEN_ALL,
    EVENT_FLATTEN_SYMBOL, EVENT_CANCEL_ALL, EVENT_POSITION_UPDATE,
    TradePayload, OrderStatusPayload, PnLPayload,
    AlertPayload, PositionPayload
)

class LiveExecutor(BaseExecutor):
    """
    Production executor for live trading.
    
    Features:
    - Broker order placement
    - Position sizing integration
    - Event-driven architecture
    - GUI command handling
    - Dry-run mode support
    - Comprehensive event emission
    
    Example:
        executor = LiveExecutor(
            broker=alpaca_broker,
            sizer=atr_sizer,
            dry_run=False
        )
        
        # Subscribe to GUI events
        await executor.subscribe_to_gui_events()
        
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
        dry_run: bool = False,
        event_handler: Optional[EventHandler] = None
    ):
        """
        Initialize live executor.
        
        Args:
            broker: Broker interface for order execution
            sizer: Optional position sizer (for sizing in executor)
            dry_run: If True, log orders but don't execute
            event_handler: Event bus for emissions
        """
        super().__init__()
        
        self.broker = broker
        self.sizer = sizer
        self.dry_run = dry_run
        
        # Event bus
        self.bus = event_handler or get_event_handler()
        
        # Position tracking (local cache)
        self.positions: Dict[str, int] = defaultdict(int)
        self.entry_prices: Dict[str, float] = {}
        self.stops: Dict[str, float] = {}
        self.targets: Dict[str, float] = {}

        # P&L tracking
        self._realized_pnl: float = 0.0
        self._last_prices: Dict[str, float] = {}
        self._peak_equity: float = 0.0
        
        # Setup logging
        self.logger = Logger("app.log", "LiveExecutor").get_logger()
        
        self.logger.info(
            f"LiveExecutor initialized (dry_run={dry_run})"
        )
        
        # Subscribe to GUI events
        asyncio.create_task(self._subscribe_to_gui_events())
    
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
        Execute trade signal.
        
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
        
        # Determine market conditions (if we have df)
        market_conditions = self._get_market_conditions(df, atr_value) if df is not None else "normal"
        
        # Calculate position size (if sizer available)
        qty = self._calculate_quantity(
            symbol, price, atr_value, signal, market_conditions
        )
        
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
        
        Not all brokers support this natively.
        """
        self.logger.warning(
            f"[{symbol}] OCO orders not implemented in this executor"
        )
        return False
    
    def get_open_orders(self, symbol: Optional[str] = None) -> list:
        """Get open orders from broker."""
        try:
            return self.broker.get_open_orders(symbol)
        except Exception as e:
            self.logger.error(f"Failed to get open orders: {e}")
            return []
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order by ID."""
        try:
            self.broker.cancel_order(order_id)
            self.logger.info(f"Cancelled order: {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
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
    
    def log_order_response(self, response: Any) -> None:
        """Log order response."""
        self.logger.info(f"Order response: {response}")
    
    def retry_failed_order(self, symbol: str, side: OrderSide, qty: int) -> bool:
        """Retry failed order (not implemented)."""
        self.logger.warning("Order retry not implemented")
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
        Place order and emit events.
        
        Args:
            symbol: Trading symbol
            qty: Quantity
            side: Order side (BUY/SELL)
            price: Price (for event emission)
            
        Returns:
            True if successful
        """
        now = datetime.now(timezone.utc)
        
        # Dry run mode
        if self.dry_run:
            self.logger.info(
                f"[DRY RUN] {side.value} {qty} {symbol} @ ${price:.2f}"
            )
            self._emit_alert(
                "info",
                f"[DRY RUN] {side.value} {qty} {symbol} @ ${price:.2f}",
                symbol
            )
            return True
        
        # Execute order via broker
        try:
            self.logger.info(
                f"[LIVE] Placing {side.value} order: {symbol} {qty}@${price:.2f}"
            )
            
            response = self.broker.place_market_order(
                symbol=symbol,
                qty=qty,
                side=side.value.lower()
            )
            
            self.logger.info(
                f"[{symbol}] Order placed: {response}"
            )
            
            # Parse response
            order_id = self._get_order_id(response)
            status = self._get_order_status_str(response)
            
            # Emit order status event
            self._emit_order_status(
                order_id=order_id,
                symbol=symbol,
                status=status,
                filled_qty=qty,
                avg_price=price
            )
            
            # Emit trade event
            self._emit_trade(
                symbol=symbol,
                side=side,
                qty=qty,
                price=price
            )
            
            # Update local position tracking
            self._update_position(symbol, side, qty, price)
            
            # Emit position update
            self._emit_position_update(symbol, price)
            
            # Emit P&L update
            self._emit_pnl_update()
            
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
        # Update last known price
        self._last_prices[symbol] = price

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
            # Calculate realized P&L for this sale
            if symbol in self.entry_prices and self.entry_prices[symbol] > 0:
                entry_price = self.entry_prices[symbol]
                realized = qty * (price - entry_price)
                self._realized_pnl += realized
                self.logger.debug(
                    f"[{symbol}] Realized P&L: ${realized:.2f} "
                    f"(entry=${entry_price:.2f}, exit=${price:.2f})"
                )

            self.positions[symbol] -= qty

            if self.positions[symbol] <= 0:
                self.positions[symbol] = 0
                self.entry_prices[symbol] = 0.0
        
        self.logger.debug(
            f"[{symbol}] Position updated: {self.positions[symbol]} "
            f"@ ${self.entry_prices.get(symbol, 0.0):.2f}"
        )
    
    def _calculate_quantity(self, symbol, state, action_type, price, atr, regime, trade_logic, **kwargs):
            """Calculate position size."""
            if action_type in ("exit", "reversal"):
                position = self.portfolio.positions.get(symbol)
                return abs(position.qty) if position else 0
            
            if action_type == "partial_exit":
                position = self.portfolio.positions.get(symbol)
                if not position:
                    return 0
                if hasattr(trade_logic, 'get_exit_quantity'):
                    return trade_logic.get_exit_quantity(position.qty, is_partial=True)
                exit_fraction = trade_logic.get_param('exit_fraction', 0.25)
                return max(int(abs(position.qty) * exit_fraction), 1)
            
            # Get signal from kwargs
            signal = kwargs.get('signal', 0)
            
            sl_mults = getattr(trade_logic, 'sl_mults', {"normal": 1.5})
            sl_mult = sl_mults.get(regime, 1.5)
            stop_loss_price = (price - (atr * sl_mult) if signal > 0
                            else price + (atr * sl_mult))
            
            qty = self.sizer.calculate_position_size(
                symbol=symbol,
                price=price,
                account_value=self.portfolio.total_value,
                signal_strength=1.0,
                atr=atr,
                stop_loss_price=stop_loss_price,
                signal=signal,
                portfolio=self.portfolio,
                market_conditions=regime
            )

            self.logger.debug(f"Sizer: {symbol} price={price:.2f} sl={stop_loss_price:.2f} qty={qty}")
            
            return qty
    
    def _can_afford(self, symbol: str, qty: int, price: float) -> bool:
        """Check if order is affordable."""
        try:
            return self.broker.has_sufficient_funds(symbol, qty)
        except (AttributeError, NotImplementedError):
            # Fallback: simple check
            try:
                cash = self.broker.get_available_funds()
                return qty * price <= cash
            except Exception:
                return True  # Assume yes if can't check
    
    def _get_market_conditions(
        self,
        df: pd.DataFrame,
        atr_value: float
    ) -> str:
        """Determine market conditions from ATR."""
        if df is None or 'ATR' not in df.columns:
            return "normal"
        
        atr_25 = df['ATR'].quantile(0.25)
        atr_75 = df['ATR'].quantile(0.75)
        
        if atr_value < atr_25:
            return "low_volatility"
        elif atr_value > atr_75:
            return "high_volatility"
        else:
            return "normal"
    
    # ========================================================================
    # EVENT EMISSION
    # ========================================================================
    
    def _emit_order_status(
        self,
        order_id: str,
        symbol: str,
        status: str,
        filled_qty: float,
        avg_price: float
    ) -> None:
        """Emit order status event."""
        payload: OrderStatusPayload = {
            "order_id": order_id,
            "symbol": symbol,
            "status": status,
            "filled_qty": float(filled_qty),
            "avg_price": float(avg_price),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        asyncio.create_task(self.bus.emit(EVENT_ORDER_STATUS, payload))
    
    def _emit_trade(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        price: float
    ) -> None:
        """Emit trade event."""
        payload: TradePayload = {
            "symbol": symbol,
            "side": side.value.lower(),
            "qty": float(qty),
            "price": float(price),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pnl": None,
        }
        asyncio.create_task(self.bus.emit(EVENT_NEW_TRADE, payload))
    
    def _emit_position_update(self, symbol: str, last_price: float) -> None:
        """Emit position update event."""
        qty = self.positions.get(symbol, 0)
        avg_price = self.entry_prices.get(symbol, 0.0)
        # Calculate unrealized P&L for this position
        unrealized = qty * (last_price - avg_price) if qty and avg_price else 0.0
        side = "long" if qty > 0 else "short" if qty < 0 else "flat"
        payload: PositionPayload = {
            "symbol": symbol,
            "qty": qty,
            "avg_price": avg_price,
            "avg": avg_price,  # GUI model uses 'avg'
            "last": last_price,  # GUI model uses 'last'
            "unrealized": unrealized,
            "unreal": unrealized,  # GUI model uses 'unreal'
            "realized": 0.0,  # Per-symbol realized not tracked; global in _realized_pnl
            "side": side,
            "market_value": qty * last_price,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        asyncio.create_task(self.bus.emit(EVENT_POSITION_UPDATE, payload))
    
    def _get_unrealized_pnl(self) -> float:
        """Calculate unrealized P&L from open positions."""
        total = 0.0
        for symbol, qty in self.positions.items():
            if qty > 0 and symbol in self.entry_prices:
                entry_price = self.entry_prices[symbol]
                current_price = self._last_prices.get(symbol, entry_price)
                total += qty * (current_price - entry_price)
        return total

    def _get_drawdown(self, current_equity: float) -> float:
        """Calculate current drawdown from peak equity."""
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - current_equity) / self._peak_equity

    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        # Try to get from broker
        try:
            quote = self.broker.get_quote(symbol)
            if quote and hasattr(quote, 'last_price'):
                price = float(quote.last_price)
                self._last_prices[symbol] = price
                return price
        except Exception:
            pass
        # Fall back to last known price
        return self._last_prices.get(symbol, self.entry_prices.get(symbol, 0.0))

    def _emit_pnl_update(self) -> None:
        """Emit P&L update event."""
        try:
            cash = self.broker.get_available_funds()
        except Exception:
            cash = 0.0

        # Calculate portfolio value using current prices
        portfolio_value = cash
        for symbol, qty in self.positions.items():
            if qty > 0:
                current_price = self._last_prices.get(symbol, self.entry_prices.get(symbol, 0.0))
                portfolio_value += qty * current_price

        unrealized = self._get_unrealized_pnl()
        drawdown = self._get_drawdown(portfolio_value)

        payload: PnLPayload = {
            "portfolio_value": portfolio_value,
            "unrealized": unrealized,
            "realized": self._realized_pnl,
            "drawdown": drawdown,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        asyncio.create_task(self.bus.emit(EVENT_PNL_UPDATE, payload))
    
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
    
    # ========================================================================
    # GUI EVENT HANDLERS
    # ========================================================================
    
    async def _subscribe_to_gui_events(self) -> None:
        """Subscribe to GUI command events."""
        await self.bus.subscribe(EVENT_MANUAL_ORDER, self._handle_manual_order)
        await self.bus.subscribe(EVENT_FLATTEN_ALL, self._handle_flatten_all)
        await self.bus.subscribe(EVENT_FLATTEN_SYMBOL, self._handle_flatten_symbol)
        await self.bus.subscribe(EVENT_CANCEL_ALL, self._handle_cancel_all)
        
        self.logger.info("Subscribed to GUI events")
    
    async def _handle_manual_order(self, event) -> None:
        """Handle manual order from GUI."""
        payload = event.payload
        symbol = payload["symbol"]
        qty = int(payload["qty"])
        side = payload["side"].upper()
        price = float(payload.get("price", 0.0))
        order_type = payload.get("type", "market").lower()
        
        self.logger.info(
            f"[MANUAL ORDER] {side} {qty} {symbol} ({order_type})"
        )
        
        try:
            if order_type == "limit":
                # Limit order
                self.broker.place_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    order_type="limit",
                    limit_price=price,
                    time_in_force=payload.get("tif", "DAY")
                )
            else:
                # Market order
                self._place_order(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                    price=price
                )
                
        except Exception as e:
            self.logger.error(f"Manual order failed: {e}")
            self._emit_alert(
                "error",
                f"Manual order failed for {symbol}: {str(e)}",
                symbol
            )
    
    async def _handle_flatten_all(self, event) -> None:
        """Flatten all positions."""
        self.logger.info("[GUI] Flatten ALL positions")
        
        for symbol, qty in list(self.positions.items()):
            if qty > 0:
                try:
                    # Get current price (simplified - use 0 for now)
                    price = 0.0
                    self._place_order(symbol, qty, OrderSide.SELL, price)
                except Exception as e:
                    self.logger.error(f"Failed to flatten {symbol}: {e}")
                    self._emit_alert(
                        "error",
                        f"Flatten failed for {symbol}: {str(e)}",
                        symbol
                    )
    
    async def _handle_flatten_symbol(self, event) -> None:
        """Flatten specific symbol."""
        symbol = event.payload["symbol"]
        qty = self.positions.get(symbol, 0)
        
        self.logger.info(f"[GUI] Flatten {symbol} (qty={qty})")
        
        if qty > 0:
            try:
                price = self._get_current_price(symbol)
                self._place_order(symbol, qty, OrderSide.SELL, price)
            except Exception as e:
                self.logger.error(f"Failed to flatten {symbol}: {e}")
                self._emit_alert(
                    "error",
                    f"Flatten failed for {symbol}: {str(e)}",
                    symbol
                )
    
    async def _handle_cancel_all(self, event) -> None:
        """Cancel all open orders."""
        self.logger.info("[GUI] Cancel all open orders")
        
        try:
            open_orders = self.get_open_orders()
            
            for order in open_orders:
                order_id = self._get_order_id(order)
                if order_id:
                    self.cancel_order(order_id)
                    
        except Exception as e:
            self.logger.error(f"Failed to cancel all orders: {e}")
            self._emit_alert(
                "error",
                f"Cancel all failed: {str(e)}",
                None
            )
    
    # ========================================================================
    # HELPERS
    # ========================================================================
    
    def _get_order_id(self, response: Any) -> str:
        """Extract order ID from response."""
        if hasattr(response, "order_id"):
            return str(response.order_id)
        elif isinstance(response, dict) and "id" in response:
            return str(response["id"])
        else:
            return f"mock_{uuid.uuid4()}"
    
    def _get_order_status_str(self, response: Any) -> str:
        """Extract order status from response."""
        if hasattr(response, "status"):
            return str(response.status)
        elif isinstance(response, dict) and "status" in response:
            return str(response["status"])
        else:
            return "filled"