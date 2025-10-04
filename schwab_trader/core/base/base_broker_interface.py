
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List
from core.app_types import OrderResult, PositionView, BrokerSnapshot
from datetime import datetime, timezone
from core.events.events import EVENT_ORDER_STATUS, EVENT_NEW_TRADE, EVENT_PNL_UPDATE
from core.events.eventhandler import EventHandler

class BaseBrokerInterface(ABC):
    """
    Minimal cross-broker interface:
      - order execution
      - account info
      - position management
      - market status
    """
    def __init__(self):
        self.bus = EventHandler()

    # --- Orders (generic async) ---
    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,                      # "buy" | "sell"
        order_type: str = "market",     # "market" | "limit"
        limit_price: float = None,
        stop_price: float = None,
        time_in_force: str = "gtc",
        **kwargs
    ) -> OrderResult:
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> OrderResult:
        pass

    # --- Sync helpers used by your sim/live code today ---
    @abstractmethod
    def place_market_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        price: Optional[float] = None
    ) -> OrderResult:
        pass

    @abstractmethod
    def place_oco_order(
        self,
        symbol: str,
        qty: int,
        stop_price: float,
        limit_price: float
    ) -> OrderResult:
        pass

    # --- State / info ---
    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[PositionView]:
        pass

    @abstractmethod
    async def get_account_info(self) -> BrokerSnapshot:
        pass

    @abstractmethod
    async def is_market_open(self) -> bool:
        pass

    @abstractmethod
    def get_default_account(self) -> str:
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> float:
        pass

    # You can keep this for convenience even though snapshot has cash
    @abstractmethod
    def get_available_funds(self) -> float:
        pass

    @abstractmethod
    async def get_open_orders(self) -> List[OrderResult]:
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderResult:
        pass

    @abstractmethod
    def mark_price(self, symbol: str, price: float) -> None:
        """Update last price for MTM equity calc."""
        pass

        # --- Emit helpers (subclasses call when relevant) ---
    async def _emit_order_status(self, payload: dict):
        await self.bus.emit(EVENT_ORDER_STATUS, payload)

    async def _emit_new_trade(self, payload: dict):
        await self.bus.emit(EVENT_NEW_TRADE, payload)

    async def _emit_pnl_update(self, payload: dict):
        await self.bus.emit(EVENT_PNL_UPDATE, payload)

    
    # high-level helpers (normalize schema)
    async def emit_order_status(
        self, status: str, symbol: str, side: str,
        qty: float = 0.0, order_id: str | None = None,
        filled_qty: float = 0.0, avg_price: float | None = None,
        reason: str | None = None,
    ):
        payload = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "filled_qty": filled_qty,
            "avg_price": avg_price,
            "status": status,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self._emit_order_status(payload)

    async def emit_new_trade(self, symbol: str, side: str, qty: float, price: float, pnl: float | None = None):
        payload = {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "pnl": pnl,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self._emit_new_trade(payload)

    async def emit_pnl_update(self, portfolio_value: float, equity_curve: list[float],
                              unrealized: float, realized: float, drawdown: float):
        payload = {
            "portfolio_value": portfolio_value,
            "equity_curve": equity_curve,
            "unrealized": unrealized,
            "realized": realized,
            "drawdown": drawdown,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self._emit_pnl_update(payload)