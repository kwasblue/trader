from core.base.execution_engine_base import ExecutionEngineBase
from core.base.base_broker_interface import BaseBrokerInterface
from core.base.position_sizer_base import PositionSizerBase
from core.base.trade_logger_base import TradeLoggerBase
from core.logic.trade_logic_manager import DynamicTradeLogicManager
from core.logic.symbol_state import SymbolState
from core.drawdown_monitor import DrawdownMonitor
from core.logic.default_trade_logic import DefaultTradeLogic
from loggers.logger import Logger
from typing import Any


class MockExecutionEngine(ExecutionEngineBase):
    def __init__(
        self,
        broker: BaseBrokerInterface,
        sizer: PositionSizerBase,
        performance_tracker: TradeLoggerBase,
        trade_logic_manager: DynamicTradeLogicManager,
        portfolio,
        drawdown_monitor: DrawdownMonitor | None = None
        
    ):
        super().__init__(broker, sizer, performance_tracker, trade_logic_manager)
        self.drawdown_monitor = drawdown_monitor
        self.portfolio = portfolio
        self.logger = Logger("mock_execution.log", self.__class__.__name__).get_logger()
        self.logger.info("Initialized MockExecutionEngine")

    def handle_signal(
        self,
        symbol: str,
        state: SymbolState,
        signal: int,
        price: float,
        atr: float,
        regime: str,
        strategy_name: str = None
    ) -> None:
        """
        Simulate execution of a trade signal.
        """
        if self.drawdown_monitor and not self.drawdown_monitor.can_trade(symbol):
            self.logger.debug(f"[{symbol}] Skipping trade: drawdown lock/cooldown active.")
            return
        
        self.logger.debug(
            f"[{symbol}] Mock handling signal: {signal} | Price: {price} | "
            f"ATR: {atr} | Regime: {regime} | Strategy: {strategy_name}"
        )

        try:
            state.strategy_name = strategy_name
            trade_logic = self.trade_logic_manager.get(symbol, regime)

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
                portfolio=self.portfolio
            )

            self.logger.info(f"[{symbol}] Executed mock trade signal: {signal} at ${price:.2f}")

        except Exception as e:
            self.logger.exception(f"[{symbol}] Error in mock execution: {e}")
 