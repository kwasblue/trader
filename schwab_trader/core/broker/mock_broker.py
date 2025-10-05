from __future__ import annotations
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Dict, Optional, List

from core.base.base_broker_interface import BaseBrokerInterface
from core.logic.portfolio_state import PortfolioState
from core.app_types import OrderResult, PositionView, BrokerSnapshot

# Events
from core.events.events import (
    EVENT_ORDER_STATUS, OrderStatusPayload,
    EVENT_NEW_TRADE, TradePayload,
    EVENT_POSITION_UPDATE, PositionPayload,
    EVENT_PNL_UPDATE, PnLPayload,
    EVENT_PRICE_UPDATE, PricePayload
)
from core.events.eventhandler import EventHandler, get_event_handler


class MockBroker(BaseBrokerInterface):
    def __init__(self, starting_cash: float = 100_000.0, event_handler: EventHandler | None = None):
        self._cash = starting_cash
        self._positions: Dict[str, PositionView] = {}
        self._portfolio = PortfolioState(cash=starting_cash)
        self.event_handler =  get_event_handler()

        # (optional) Sanity log for debug
        self.event_handler.logger.debug(
            f"[MockBroker] Initialized with EventHandler id={id(self.event_handler)} | cash={self._cash}"
        )

    # --- sync used by your strategies ---
    def place_market_order(self, symbol: str, qty: int, side: str, price: Optional[float] = None) -> OrderResult:
        if price is None:
            return OrderResult(False, message="MockBroker requires a price for market orders in sim")

        order_id = f"mock_{uuid.uuid4().hex[:8]}"
        filled_qty = 0

        if side == "buy":
            cost = qty * price
            if cost > self._cash:
                return OrderResult(False, order_id=order_id, message="Insufficient funds")
            self._cash -= cost
            pos = self._positions.get(symbol)
            if pos:
                new_qty = pos.qty + qty
                new_avg = (pos.avg_entry_price * pos.qty + qty * price) / new_qty
                self._positions[symbol] = PositionView(symbol, new_qty, new_avg, price, price)
            else:
                self._positions[symbol] = PositionView(symbol, qty, price, price, price)
            filled_qty = qty

        else:  # sell
            pos = self._positions.get(symbol)
            have = pos.qty if pos else 0
            if qty > have:
                return OrderResult(False, order_id=order_id, message="Insufficient position")
            proceeds = qty * price
            self._cash += proceeds
            new_qty = have - qty
            if new_qty == 0:
                self._positions.pop(symbol, None)
            else:
                self._positions[symbol] = PositionView(symbol, new_qty, pos.avg_entry_price, price, price)
            filled_qty = qty

        # Update portfolio state
        self._portfolio.apply_fill(symbol, side, qty, price)

        # Create result
        res = OrderResult(True, order_id=order_id, filled_qty=filled_qty, avg_price=price, symbol=symbol, side=side)

        # Emit events
        if self.event_handler:
            now = datetime.now(timezone.utc).isoformat()

            # Order status
            order_status: OrderStatusPayload = {
                "order_id": res.order_id,
                "symbol": symbol,
                "status": "filled",
                "filled_qty": filled_qty,
                "avg_price": price,
                "timestamp": now,
            }
            asyncio.create_task(self.event_handler.emit(EVENT_ORDER_STATUS, order_status))

            # New trade
            trade: TradePayload = {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "timestamp": now,
                "pnl": None,  # could add realized pnl if tracked
            }
            asyncio.create_task(self.event_handler.emit(EVENT_NEW_TRADE, trade))

            # Position update
            pos = self._positions.get(symbol)
            if pos:
                position: PositionPayload = {
                    "symbol": symbol,
                    "qty": pos.qty,
                    "avg_price": pos.avg_entry_price,
                    "unrealized": (pos.market_price - pos.avg_entry_price) * pos.qty if pos.qty else 0.0,
                    "realized": 0.0,  # not tracked in this mock
                    "timestamp": now,
                }
                asyncio.create_task(self.event_handler.emit(EVENT_POSITION_UPDATE, position))

            # PnL update
            pnl: PnLPayload = {
                "portfolio_value": self._portfolio.total_equity(),
                "equity_curve": [],  # optional: maintain rolling equity list in PortfolioState
                "unrealized": self._portfolio.total_unrealized(),
                "realized": 0.0,  # can add realized tracking
                "drawdown": 0.0,  # mock could leave at 0 or compute max
                "timestamp": now,
            }
            asyncio.create_task(self.event_handler.emit(EVENT_PNL_UPDATE, pnl))

        return res

    def place_oco_order(self, symbol: str, qty: int, stop_price: float, limit_price: float) -> OrderResult:
        return OrderResult(True, order_id=f"mock_oco_{symbol}", message="Accepted (no live routing in sim)")

    # --- async wrappers ---
    async def place_order(self, *args, **kwargs) -> OrderResult:
        if kwargs.get("order_type", "market") == "market":
            return self.place_market_order(kwargs["symbol"], int(kwargs["qty"]), kwargs["side"], kwargs.get("limit_price"))
        return OrderResult(False, message="Limit/stop not simulated here")

    async def cancel_order(self, order_id: str) -> OrderResult:
        res = OrderResult(True, order_id=order_id, status="canceled", message="Canceled (mock)")
        if self.event_handler:
            payload: OrderStatusPayload = {
                "order_id": order_id,
                "symbol": "N/A",
                "status": "canceled",
                "filled_qty": 0.0,
                "avg_price": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await self.event_handler.emit(EVENT_ORDER_STATUS, payload)
        return res

    async def get_position(self, symbol: str) -> Optional[PositionView]:
        return self._positions.get(symbol)

    async def get_account_info(self) -> BrokerSnapshot:
        equity = self._cash + sum(p.qty * p.market_price for p in self._positions.values())
        return BrokerSnapshot(cash=self._cash, equity=equity, positions=self._positions.copy())

    async def is_market_open(self) -> bool:
        return True

    def get_default_account(self) -> str:
        return "MOCK-ACCOUNT"

    def get_quote(self, symbol: str) -> float:
        p = self._positions.get(symbol)
        return p.market_price if p else 0.0

    def get_available_funds(self) -> float:
        return self._cash

    async def get_open_orders(self) -> List[OrderResult]:
        return []

    async def get_order_status(self, order_id: str) -> OrderResult:
        return OrderResult(True, order_id=order_id, status="filled", message="Filled (mock)")

    # mark-to-market so equity is correct
    async def mark_price(self, symbol: str, price: float) -> None:
            p = self._positions.get(symbol)
            if p:
                p.market_price = float(price)
                self._portfolio.update_price(symbol, price)
            
                # --- emit price update ---
                if self.event_handler:
                    payload: PricePayload = {
                        "symbol": symbol,
                        "price": float(price),
                        "ma20": None,
                        "ma50": None,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    await self.event_handler.emit(EVENT_PRICE_UPDATE, payload)

                # --- if position exists, emit position + pnl updates ---
                if p:
                    pos_payload: PositionPayload = {
                        "symbol": symbol,
                        "qty": p.qty,
                        "avg_price": p.avg_entry_price,
                        "unrealized": (price - p.avg_entry_price) * p.qty,
                        "realized": 0.0,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    await self.event_handler.emit(EVENT_POSITION_UPDATE, pos_payload)

                pnl_payload: PnLPayload = {
                    "portfolio_value": self._portfolio.total_equity(),
                    "equity_curve": [],
                    "unrealized": self._portfolio.total_unrealized(),
                    "realized": 0.0,
                    "drawdown": 0.0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await self.event_handler.emit(EVENT_PNL_UPDATE, pnl_payload)
