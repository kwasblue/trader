from core.base.execution_engine_base import ExecutionEngineBase
from loggers.logger import Logger

class GenericExecutionEngine(ExecutionEngineBase):
    def __init__(self, broker, sizer, performance_tracker, trade_logic_manager):
        super().__init__(broker, sizer, performance_tracker, trade_logic_manager)
        self.logger = Logger("generic_execution.log", self.__class__.__name__).get_logger()
        self.logger.info("Initialized GenericExecutionEngine")

    def handle_signal(self, symbol, state, signal, price, atr, regime, strategy_name=None):
        if signal == 0:
            self.logger.debug(f"[{symbol}] HOLD signal — skipping execution.")
            return

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
                tracker=self.performance_tracker
            )
            self.logger.info(f"[{symbol}] Executed trade signal: {signal} at ${price:.2f}")
        except Exception as e:
            self.logger.exception(f"[{symbol}] Error in execution: {e}")
