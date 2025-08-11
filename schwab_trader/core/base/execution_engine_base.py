from abc import ABC, abstractmethod
from typing import Any

from core.base.base_broker_interface import BaseBrokerInterface
from core.base.position_sizer_base import PositionSizerBase
from core.base.trade_logger_base import TradeLoggerBase
from core.base.trade_logic_manager_base import TradeLogicManagerBase


class ExecutionEngineBase(ABC):
    def __init__(
        self,
        broker: BaseBrokerInterface,
        sizer: PositionSizerBase,
        performance_tracker: TradeLoggerBase,
        trade_logic_manager: TradeLogicManagerBase
    ):
        self.broker = broker
        self.sizer = sizer
        self.performance_tracker = performance_tracker
        self.trade_logic_manager = trade_logic_manager

    @abstractmethod
    def handle_signal(
        self,
        symbol: str,
        state: Any,
        signal: int,
        price: float,
        atr: float,
        regime: str,
        strategy_name: str = None
    ) -> None:
        """
        Execute trade logic based on a new signal.

        Parameters:
            symbol: Stock ticker
            state: SymbolState or similar object holding current context
            signal: Trade signal (+1, -1, 0)
            price: Current price
            atr: Average True Range for volatility context
            regime: Market regime classification
            strategy_name: Optional strategy identifier
        """
        pass
 