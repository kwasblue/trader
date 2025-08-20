# core/execution/live_execution_engine.py

from core.base.execution_engine_base import ExecutionEngineBase
from core.base.base_broker_interface import BaseBrokerInterface
from core.base.position_sizer_base import PositionSizerBase
from core.base.trade_logger_base import TradeLoggerBase
from core.logic.trade_logic_manager import DynamicTradeLogicManager
from loggers.logger import Logger
from typing import Any


class LiveExecutionEngine(ExecutionEngineBase):
    """
    Executes real trades via broker interface (e.g., Alpaca, Schwab).
    Routes logic through strategy router, sizes position, and logs results.
    """

    def __init__(
        self,
        broker: BaseBrokerInterface,
        sizer: PositionSizerBase,
        performance_tracker: TradeLoggerBase,
        trade_logic_manager: DynamicTradeLogicManager
    ):
        super().__init__(broker, sizer, performance_tracker, trade_logic_manager)
        self.logger = Logger("live_execution.log", self.__class__.__name__).get_logger()
        self.logger.info("Initialized LiveExecutionEngine")

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
        Handle a trading signal and execute a real trade via broker.

        Args:
            symbol (str): The stock ticker symbol.
            state (Any): SymbolState or similar object holding position and status.
            signal (int): +1 = Buy, -1 = Sell, 0 = Hold.
            price (float): Current market price.
            atr (float): Average True Range value.
            regime (str): Market condition label.
            strategy_name (str): Strategy name for dynamic routing.
        """
        self.logger.debug(
            f"[{symbol}] Live handling signal: {signal} | Price: {price} | "
            f"ATR: {atr} | Regime: {regime} | Strategy: {strategy_name}"
        )

        try:
            state.strategy_name = strategy_name
            trade_logic = self.trade_logic_manager.get(symbol, strategy_name)

            trade_logic.execute(
                symbol=symbol,
                state=state,
                signal=signal,
                price=price,
                atr=atr,
                regime=regime,
                broker=self.broker,
                sizer=self.sizer,
                performance_tracker=self.performance_tracker,
                portfolio = self.portfolio
            )

            self.logger.info(f"[{symbol}] Executed live trade signal: {signal} at ${price:.2f}")

        except Exception as e:
            self.logger.exception(f"[{symbol}] Error in live execution: {e}")
