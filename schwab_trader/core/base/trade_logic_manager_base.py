from abc import ABC, abstractmethod
from typing import Any


class TradeLogicManagerBase(ABC):
    """
    Abstract base class for managing trade logic retrieval by symbol and strategy.
    """

    @abstractmethod
    def get(self, symbol: str, strategy_name: str = None) -> Any:
        """
        Retrieve the trade logic object for a given symbol and optional strategy.

        Args:
            symbol (str): The stock symbol (e.g., "AAPL").
            strategy_name (str, optional): Specific strategy identifier (e.g., "momentum").

        Returns:
            Any: The trade logic instance implementing an `execute()` method.
        """
        pass
