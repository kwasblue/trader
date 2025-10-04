from core.base.execution_engine_base import ExecutionEngineBase
from core.base.base_broker_interface import BaseBrokerInterface
from core.base.position_sizer_base import PositionSizerBase
from core.base.trade_logger_base import TradeLoggerBase
from core.logic.trade_logic_manager import DynamicTradeLogicManager
from core.logic.symbol_state import SymbolState
from core.drawdown_monitor import DrawdownMonitor
from loggers.logger import Logger
from datetime import datetime, UTC
from core.events.events import TradePayload, EVENT_NEW_TRADE, PositionPayload, EVENT_POSITION_UPDATE, PnLPayload, EVENT_PNL_UPDATE, EVENT_STRATEGY_SIGNAL

import asyncio


class MockExecutionEngine(ExecutionEngineBase):
    def __init__(
        self,
        broker: BaseBrokerInterface,
        sizer: PositionSizerBase,
        performance_tracker: TradeLoggerBase,
        trade_logic_manager: DynamicTradeLogicManager,
        portfolio,
        drawdown_monitor: DrawdownMonitor | None = None,
        event_handler = None
        
    ):
        super().__init__(broker, sizer, performance_tracker, trade_logic_manager)
        self.drawdown_monitor = drawdown_monitor
        self.portfolio = portfolio
        self.logger = Logger("mock_execution.log", self.__class__.__name__).get_logger()
        self.logger.info("Initialized MockExecutionEngine")
        self.event_handler = event_handler

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

            action = trade_logic.execute(
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
            if action and action["side"] in ("buy", "sell"):
                qty = action["qty"]
                side = action["side"]
                self.portfolio.apply_fill(symbol, side, qty, price)  # update PortfolioState

                # --- emit trade event ---
                trade: TradePayload = {
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "price": price,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "pnl": self.portfolio.unrealized_pnl(symbol),
                }
                asyncio.create_task(self.event_handler.emit(EVENT_NEW_TRADE, trade))

                # --- emit position update ---
                pos: PositionPayload = {
                    "symbol": symbol,
                    "qty": self.portfolio.positions[symbol].qty,
                    "avg_price": self.portfolio.positions[symbol].avg_price,
                    "cash": self.portfolio.cash,
                    "last_price": price,
                }
                asyncio.create_task(self.event_handler.emit(EVENT_POSITION_UPDATE, pos))

                # --- emit PnL update ---
                pnl: PnLPayload = {
                    "portfolio_value": self.portfolio.total_equity(),
                    "unrealized": self.portfolio.total_unrealized(),
                    "realized": self.portfolio.total_realized(),
                    "drawdown": self.portfolio.drawdown(),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                asyncio.create_task(self.event_handler.emit(EVENT_PNL_UPDATE, pnl))

            self.logger.info(f"[{symbol}] Executed mock trade signal: {signal} at ${price:.2f}")

        

        except Exception as e:
            self.logger.exception(f"[{symbol}] Error in mock execution: {e}")
    
    async def subscribe_signals(self):
        async def on_signal(event):
            payload = event.payload
            sig_val = 1 if payload["signal"] in (1, "buy") else -1 if payload["signal"] in (-1, "sell") else 0
            self.handle_signal(
                symbol=payload["symbol"],
                state=self.symbol_states[payload["symbol"]],
                signal=sig_val,
                price=payload.get("price", 0.0),
                atr=payload.get("atr", 0.0),
                regime=payload.get("regime", "normal"),
                strategy_name=payload.get("strategy")
            )
        await self.event_handler.subscribe(EVENT_STRATEGY_SIGNAL, on_signal)