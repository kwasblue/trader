from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

from core.events.eventhandler import EventHandler
from core.events.events import (
    EVENT_ORDER_STATUS,
    EVENT_NEW_TRADE,
    EVENT_PNL_UPDATE,
    EVENT_ALERT,
    OrderStatusPayload,
    TradePayload,
    PnLPayload,
    AlertPayload,
)

class BaseExecutor(ABC):
    """
    Abstract base class for trade executors.

    Executors are responsible for submitting orders,
    tracking positions, logging, and enforcing trade logic.
    """
    def __init__(self):
        self.bus = EventHandler()

    # --- Emit helpers (subclasses call these) ---
    async def _emit_order_status(self, payload: OrderStatusPayload):
        await self.bus.emit(EVENT_ORDER_STATUS, payload)

    async def _emit_new_trade(self, payload: TradePayload):
        await self.bus.emit(EVENT_NEW_TRADE, payload)

    async def _emit_pnl_update(self, payload: PnLPayload):
        await self.bus.emit(EVENT_PNL_UPDATE, payload)

    async def _emit_alert(self, payload: AlertPayload):
        await self.bus.emit(EVENT_ALERT, payload)

    @abstractmethod
    def execute(
        self,
        symbol: str,
        df: pd.DataFrame,
        signal: int,
        price: float,
        atr_value: float
    ) -> None:
        """
        Execute logic based on signal and market conditions.

        Args:
            symbol (str): The ticker symbol.
            df (pd.DataFrame): Historical + real-time price data.
            signal (int): -1 for sell, 0 for hold, 1 for buy.
            price (float): Current price.
            atr_value (float): Latest ATR value.
        """
        pass

    @abstractmethod
    def buy(self, symbol: str, qty: int, order_type: str = "MARKET", **kwargs) -> Dict[str, Any]:
        """
        Submit a buy order.
        """
        pass

    @abstractmethod
    def sell(self, symbol: str, qty: int, order_type: str = "MARKET", **kwargs) -> Dict[str, Any]:
        """
        Submit a sell order.
        """
        pass

    @abstractmethod
    def place_oco_order(self, symbol: str, qty: int, stop_price: float, limit_price: float) -> Dict[str, Any]:
        """
        Submit an OCO (One Cancels Other) order.
        """
        pass

    @abstractmethod
    def get_open_orders(self) -> Any:
        """
        Retrieve all open orders.
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> Any:
        """
        Cancel a specific order by ID.
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Any:
        """
        Get the status of a specific order.
        """
        pass

    @abstractmethod
    def log_order_response(self, response: dict) -> None:
        """
        Log the order response or error.
        """
        pass

    @abstractmethod
    def retry_failed_order(self, *args, max_retries=3, delay=2, **kwargs) -> Dict[str, Any]:
        """
        Retry order placement on failure.
        """
        pass
