from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any


class TradeLoggerBase(ABC):
    """
    Abstract base class for logging trade events, executions, errors, and performance summaries.
    Intended to be extended by custom loggers (file-based, DB-based, Discord-based, etc.)
    """

    @abstractmethod
    def log_trade(
        self,
        timestamp: datetime,
        symbol: str,
        action: str,  # 'BUY', 'SELL', 'SHORT', 'COVER', 'HOLD'
        price: float,
        quantity: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        strategy: Optional[str] = None,
        regime: Optional[str] = None,
        cash_before: Optional[float] = None,
        cash_after: Optional[float] = None,
        position_before: Optional[int] = None,
        position_after: Optional[int] = None,
        notes: Optional[str] = None
    ) -> None:
        """
        Log a trade execution.
        """
        pass

    @abstractmethod
    def log_error(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an error or exception.
        """
        pass

    @abstractmethod
    def log_info(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        General info logs (entry/exit signals, regime changes, sizing decisions).
        """
        pass

    @abstractmethod
    def log_summary(self, summary_stats: Dict[str, Any]) -> None:
        """
        End-of-day or end-of-session portfolio summary.
        """
        pass

    @abstractmethod
    def flush(self) -> None:
        """
        Finalize and persist logs if needed (e.g., write buffered logs to file).
        """
        pass
