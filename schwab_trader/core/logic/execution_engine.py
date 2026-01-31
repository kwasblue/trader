"""
Generic Execution Engine - Orchestrates trading with flexible logic routing

Supports:
- Multiple trade logic managers (per strategy, symbol, or regime)
- Logic routing based on context
- Fallback to default logic
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Union
from datetime import datetime, timezone
import logging

from core.base.execution_engine_base import ExecutionEngineBase
from core.base.executor_base import BaseExecutor
from core.base.base_broker_interface import BaseBrokerInterface
from core.base.position_sizer_base import PositionSizerBase
from core.base.trade_logger_base import TradeLoggerBase
from core.base.trade_logic_manager_base import TradeLogicManagerBase
from core.logic.portfolio_state import PortfolioState
from core.logic.symbol_state import SymbolState
from core.app_types import OrderResult
from core.enums import OrderSide
from loggers.logger import Logger
from core.logic.trade_logic_router import TradeLogicRouter


class GenericExecutionEngine(ExecutionEngineBase):
    """
    Generic execution engine with flexible logic routing.
    
    Features:
    - Multiple trade logic managers
    - Logic routing by symbol/strategy/regime
    - Standard orchestration flow
    """
    
    def __init__(
        self,
        broker: BaseBrokerInterface,
        executor: BaseExecutor,
        sizer: PositionSizerBase,
        performance_tracker: TradeLoggerBase,
        trade_logic_manager: Union[TradeLogicManagerBase, TradeLogicRouter],
        portfolio: PortfolioState,
    ):
        """
        Initialize execution engine.
        
        Args:
            broker: Broker interface
            executor: Trade executor
            sizer: Position sizer
            performance_tracker: Trade logger
            trade_logic_manager: Single manager OR router
            portfolio: Portfolio state
        """
        super().__init__(
            broker=broker,
            executor=executor,
            sizer=sizer,
            performance_tracker=performance_tracker,
            trade_logic_manager=trade_logic_manager,
            portfolio=portfolio
        )
        
        # Wrap single manager in router if needed
        if isinstance(trade_logic_manager, TradeLogicRouter):
            self.logic_router = trade_logic_manager
        else:
            self.logic_router = TradeLogicRouter(trade_logic_manager)
        
        # Logger - own file with propagation to app.log
        self.logger = Logger(
            log_file="execution_engine.log",
            logger_name="GenericExecutionEngine",
            propagate=True
        ).get_logger()
        
        self.logger.info("GenericExecutionEngine initialized")
    
    # ========================================================================
    # MAIN SIGNAL HANDLING
    # ========================================================================
    
    def handle_signal(
        self,
        symbol: str,
        state: SymbolState,
        signal: int,
        price: float,
        atr: float,
        regime: str,
        strategy_name: Optional[str] = None,
        **kwargs
    ) -> Optional[OrderResult]:
        """
        Handle signal from strategy.
        
        Routes to appropriate trade logic and orchestrates execution.
        """
        # Skip hold signals
        if signal == 0:
            self.logger.debug(f"[{symbol}] HOLD signal - no action")
            return None
        
        # Set strategy name on state
        state.strategy_name = strategy_name
        
        self.logger.info(
            f"[{symbol}] Processing signal: {signal} @ ${price:.2f} "
            f"(regime={regime}, strategy={strategy_name})"
        )
        
        try:
            # Get appropriate trade logic for this context
            trade_logic = self.logic_router.get_logic(
                symbol=symbol,
                strategy=strategy_name,
                regime=regime
            )
            
            self.logger.debug(
                f"[{symbol}] Using logic: {trade_logic.__class__.__name__}"
            )
            
            # 1. Check trade logic approval
            should_trade, reason = self._check_trade_approval(
                trade_logic, symbol, state, signal, price, atr, regime, **kwargs
            )
            
            if not should_trade:
                self.logger.info(f"[{symbol}] Trade blocked: {reason}")
                return None
            
            # 2. Determine action type
            action_type, side = self._determine_action(symbol, state, signal, reason)
            
            self.logger.info(f"[{symbol}] Action approved: {action_type} {side.value}")
            
            # 3. Calculate quantity
            qty = self._calculate_quantity(
                symbol, state, action_type, price, atr, regime, trade_logic, **kwargs
            )
            
            if qty <= 0:
                self.logger.warning(f"[{symbol}] Position size too small: {qty}")
                return None
            
            # 4. Execute trade
            result = self._execute_trade(
                symbol, state, side, qty, price, atr, action_type, **kwargs
            )
            
            if result:
                # 5. Post-execution tasks
                self._post_execution(
                    symbol, state, result, action_type, regime, strategy_name
                )
            
            return result
            
        except Exception as e:
            self.logger.exception(f"[{symbol}] Error in execution: {e}")
            return None
    
    # ========================================================================
    # ORCHESTRATION STEPS
    # ========================================================================
    
    def _check_trade_approval(
        self,
        trade_logic: TradeLogicManagerBase,
        symbol: str,
        state: SymbolState,
        signal: int,
        price: float,
        atr: float,
        regime: str,
        **kwargs
    ) -> tuple[bool, Optional[str]]:
        """Check if trade should be executed using appropriate logic."""
        # Get current position from portfolio
        position = self.portfolio.positions.get(symbol)
        current_qty = 0 if not position else position.qty
        avg_price = None if not position else position.avg_price
        
        # Update state with current position
        state.current_position = current_qty
        if current_qty != 0:
            state.side = "long" if current_qty > 0 else "short"
        else:
            state.side = None
        
        # Get market status
        market_open = kwargs.get('market_open', True)
        
        # Check with the routed trade logic
        should_trade, reason = trade_logic.should_trade(
            symbol=symbol,
            state=state,
            signal=signal,
            regime=regime,
            price=price,
            atr=atr,
            avg_price=avg_price,
            market_open=market_open,
            account_positions=len(self.portfolio.positions)
        )
        
        return should_trade, reason
    
    def _determine_action(
        self,
        symbol: str,
        state: SymbolState,
        signal: int,
        reason: Optional[str]
    ) -> tuple[str, OrderSide]:
        """Determine what type of action to take."""
        in_position = state.side is not None
        
        if not in_position:
            action_type = "entry"
            side = OrderSide.BUY if signal == 1 else OrderSide.SELL
        elif reason and "partial" in reason.lower():
            action_type = "partial_exit"
            side = OrderSide.SELL if state.side == "long" else OrderSide.BUY
        elif reason and "reversal" in reason.lower():
            action_type = "reversal"
            side = OrderSide.SELL if state.side == "long" else OrderSide.BUY
        else:
            action_type = "exit"
            side = OrderSide.SELL if state.side == "long" else OrderSide.BUY
        
        return action_type, side
    
    def _calculate_quantity(
        self,
        symbol: str,
        state: SymbolState,
        action_type: str,
        price: float,
        atr: float,
        regime: str,
        trade_logic: TradeLogicManagerBase,
        **kwargs
    ) -> int:
        """Calculate position size."""
        # For exits, use current position size
        if action_type in ("exit", "reversal"):
            position = self.portfolio.positions.get(symbol)
            if not position:
                return 0
            return abs(position.qty)
        
        # For partial exits, get fraction from logic
        if action_type == "partial_exit":
            position = self.portfolio.positions.get(symbol)
            if not position:
                return 0
            
            # Use logic's exit quantity method if available
            if hasattr(trade_logic, 'get_exit_quantity'):
                return trade_logic.get_exit_quantity(position.qty, is_partial=True)
            else:
                exit_fraction = trade_logic.get_param('exit_fraction', 0.25)
                return max(int(abs(position.qty) * exit_fraction), 1)
        
        # For entries, use position sizer
        sl_mults = getattr(trade_logic, 'sl_mults', {"normal": 1.5})
        sl_mult = sl_mults.get(regime, 1.5)
        
        stop_loss_price = (price - (atr * sl_mult) if state.side != "short"
                          else price + (atr * sl_mult))
        
        qty = self.sizer.calculate_position_size(
            symbol=symbol,
            price=price,
            account_value=self.portfolio.total_value,
            signal_strength=1.0,
            atr=atr,
            stop_loss_price=stop_loss_price
        )
        
        return qty
    
    def _execute_trade(
        self,
        symbol: str,
        state: SymbolState,
        side: OrderSide,
        qty: int,
        price: float,
        atr: float,
        action_type: str,
        **kwargs
    ) -> Optional[OrderResult]:
        """Execute trade via executor."""
        df = kwargs.get('df')
        signal = 1 if side == OrderSide.BUY else -1
        
        try:
            self.executor.execute(
                symbol=symbol,
                df=df,
                signal=signal,
                price=price,
                atr_value=atr
            )
            
            result = OrderResult(
                order_id=f"{symbol}_{datetime.now(timezone.utc).timestamp()}",
                symbol=symbol,
                side=side,
                filled_qty=qty,
                avg_price=price,
                timestamp=datetime.now(timezone.utc)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"[{symbol}] Execution failed: {e}")
            return None
    
    def _post_execution(
        self,
        symbol: str,
        state: SymbolState,
        result: OrderResult,
        action_type: str,
        regime: str,
        strategy_name: Optional[str]
    ) -> None:
        """Post-execution tasks."""
        self.portfolio.update_position(symbol, result)
        state.last_trade_time = datetime.now(timezone.utc)
        
        self.performance_tracker.log_trade(
            symbol=symbol,
            action=result.side,
            price=result.avg_price,
            quantity=result.filled_qty,
            order_id=result.order_id,
            strategy=strategy_name,
            regime=regime,
            sl=getattr(state, 'stop_loss', None),
            tp=getattr(state, 'take_profit', None),
            notes=f"action={action_type}, bars_held={getattr(state, 'bars_held', 0)}"
        )
        
        if action_type in ("exit", "reversal"):
            state.reset()
        
        self.logger.info(
            f"[{symbol}] Trade completed: {action_type} "
            f"{result.side.value} {result.filled_qty}@${result.avg_price:.2f}"
        )
    
    # ========================================================================
    # LOGIC REGISTRATION (Convenience methods)
    # ========================================================================
    
    def register_symbol_logic(self, symbol: str, logic: TradeLogicManagerBase) -> None:
        """Register symbol-specific logic."""
        self.logic_router.register_symbol_logic(symbol, logic)
        self.logger.info(f"Registered logic for {symbol}: {logic.__class__.__name__}")
    
    def register_strategy_logic(self, strategy: str, logic: TradeLogicManagerBase) -> None:
        """Register strategy-specific logic."""
        self.logic_router.register_strategy_logic(strategy, logic)
        self.logger.info(f"Registered logic for strategy '{strategy}': {logic.__class__.__name__}")
    
    def register_regime_logic(self, regime: str, logic: TradeLogicManagerBase) -> None:
        """Register regime-specific logic."""
        self.logic_router.register_regime_logic(regime, logic)
        self.logger.info(f"Registered logic for regime '{regime}': {logic.__class__.__name__}")