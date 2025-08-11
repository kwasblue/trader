from __future__ import annotations
import uuid
from datetime import datetime
from typing import Dict, Optional, List

from core.base.base_broker_interface import BaseBrokerInterface
from core.logic.portfolio_state import PortfolioState
from core.types import OrderResult, PositionView, BrokerSnapshot


class MockBroker(BaseBrokerInterface):
    def __init__(self, starting_cash: float = 100_000.0):
        self._cash = starting_cash
        self._positions: Dict[str, PositionView] = {}

    # --- sync used by your strategies ---
    def place_market_order(self, symbol: str, qty: int, side: str, price: Optional[float] = None) -> OrderResult:
        # use provided price (sim) as the fill
        if price is None:
            return OrderResult(False, message="MockBroker requires a price for market orders in sim")

        if side == "buy":
            cost = qty * price
            if cost > self._cash:
                return OrderResult(False, message="Insufficient funds")
            self._cash -= cost
            pos = self._positions.get(symbol)
            if pos:
                new_qty = pos.qty + qty
                new_avg = (pos.avg_price * pos.qty + qty * price) / new_qty
                self._positions[symbol] = PositionView(symbol, new_qty, new_avg, price)
            else:
                self._positions[symbol] = PositionView(symbol, qty, price, price)
        else:  # sell
            pos = self._positions.get(symbol)
            have = pos.qty if pos else 0
            if qty > have:
                return OrderResult(False, message="Insufficient position")
            proceeds = qty * price
            self._cash += proceeds
            new_qty = have - qty
            if new_qty == 0:
                self._positions.pop(symbol, None)
            else:
                self._positions[symbol] = PositionView(symbol, new_qty, pos.avg_price, price)

        return OrderResult(True, order_id="mock_"+symbol, filled_qty=qty, avg_price=price)

    def place_oco_order(self, symbol: str, qty: int, stop_price: float, limit_price: float) -> OrderResult:
        # Sim doesn’t simulate pending orders here; acknowledge only
        return OrderResult(True, order_id=f"mock_oco_{symbol}", message="Accepted (no live routing in sim)")

    # --- async generic methods (simple wrappers or stubs for sim) ---
    async def place_order(self, *args, **kwargs) -> OrderResult:
        # route market orders to the sync helper for consistency
        if kwargs.get("order_type", "market") == "market":
            return self.place_market_order(kwargs["symbol"], int(kwargs["qty"]), kwargs["side"], kwargs.get("limit_price"))
        return OrderResult(False, message="Limit/stop not simulated here")

    async def cancel_order(self, order_id: str) -> OrderResult:
        return OrderResult(True, order_id=order_id, message="Canceled (mock)")

    async def get_position(self, symbol: str) -> Optional[PositionView]:
        return self._positions.get(symbol)

    async def get_account_info(self) -> BrokerSnapshot:
        equity = self._cash + sum(p.qty * p.last_price for p in self._positions.values())
        return BrokerSnapshot(cash=self._cash, equity=equity, positions=self._positions.copy())

    async def is_market_open(self) -> bool:
        return True

    def get_default_account(self) -> str:
        return "MOCK-ACCOUNT"

    def get_quote(self, symbol: str) -> float:
        p = self._positions.get(symbol)
        return p.last_price if p else 0.0

    def get_available_funds(self) -> float:
        return self._cash

    async def get_open_orders(self) -> List[OrderResult]:
        return []

    async def get_order_status(self, order_id: str) -> OrderResult:
        return OrderResult(True, order_id=order_id, message="Filled (mock)")
    
    # mark-to-market so equity is correct
    def mark_price(self, symbol: str, price: float) -> None:
        p = self._positions.get(symbol)
        if p:
            p.last = float(price)