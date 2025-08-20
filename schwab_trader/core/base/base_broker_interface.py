
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List
from core.app_types import OrderResult, PositionView, BrokerSnapshot

class BaseBrokerInterface(ABC):
    """
    Minimal cross-broker interface:
      - order execution
      - account info
      - position management
      - market status
    """

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