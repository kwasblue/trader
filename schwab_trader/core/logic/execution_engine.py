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
from core.base.trade_logic_manager_base import TradeApprover, TradeLogicManagerBase
from core.logic.portfolio_state import PortfolioState
from core.logic.symbol_state import SymbolState
from core.app_types import OrderResult, SignalContext
from core.enums import OrderSide
from loggers.logger import Logger
from core.logic.trade_logic_router import TradeApproverRouter


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
        trade_logic_manager: Union[TradeApprover, TradeLogicManagerBase, TradeApproverRouter],
        portfolio: PortfolioState,
        position_manager: Optional["PositionManager"] = None,
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
            position_manager: PositionManager for exit/SL/TP logic
        """
        # Import here to avoid circular imports
        from core.logic.position_manager import PositionManager as PM

        super().__init__(
            broker=broker,
            executor=executor,
            sizer=sizer,
            performance_tracker=performance_tracker,
            trade_logic_manager=trade_logic_manager,
            portfolio=portfolio,
            position_manager=position_manager or PM(),
        )

        # Wrap single approver in router if needed
        if isinstance(trade_logic_manager, TradeApproverRouter):
            self.approver_router = trade_logic_manager
        else:
            self.approver_router = TradeApproverRouter(trade_logic_manager)

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
        context: SignalContext,
        state: SymbolState,
    ) -> Optional[OrderResult]:
        """
        Handle signal from strategy.

        This is the primary entry point. Takes a unified SignalContext
        instead of scattered parameters.

        Args:
            context: SignalContext containing all signal data
            state: SymbolState for this symbol

        Returns:
            OrderResult if trade executed, None otherwise
        """
        # Skip hold signals
        if context.is_hold():
            self.logger.debug(f"[{context.symbol}] HOLD signal - no action")
            return None

        # Set strategy name on state
        state.strategy_name = context.strategy_name

        self.logger.info(
            f"[{context.symbol}] Processing signal: {context.signal} @ ${context.price:.2f} "
            f"(regime={context.regime}, strategy={context.strategy_name})"
        )

        try:
            # Get appropriate approver for this context
            trade_approver = self.approver_router.get_approver(
                symbol=context.symbol,
                strategy=context.strategy_name,
                regime=context.regime
            )

            self.logger.debug(
                f"[{context.symbol}] Using approver: {trade_approver.__class__.__name__}"
            )

            # 1. Check trade approval
            should_trade, reason = self._check_trade_approval(
                trade_approver, context, state
            )

            if not should_trade:
                self.logger.info(f"[{context.symbol}] Trade blocked: {reason}")
                return None

            # 2. Determine action type
            action_type, side = self._determine_action(
                context.symbol, state, context.signal, reason
            )

            self.logger.info(
                f"[{context.symbol}] Action approved: {action_type} {side.value}"
            )

            # 3. Calculate quantity
            qty = self._calculate_quantity(
                context.symbol, state, action_type, context.price, context.atr,
                context.regime, trade_approver, signal=context.signal
            )

            if qty <= 0:
                self.logger.warning(f"[{context.symbol}] Position size too small: {qty}")
                return None

            # 4. Execute trade
            result = self._execute_trade(
                context.symbol, state, side, qty, context.price, context.atr, action_type
            )

            if result:
                # 5. Post-execution tasks
                self._post_execution(
                    context.symbol, state, result, action_type,
                    context.regime, context.strategy_name
                )

            return result

        except Exception as e:
            self.logger.exception(f"[{context.symbol}] Error in execution: {e}")
            return None

    # handle_signal_legacy inherited from ExecutionEngineBase

    # ========================================================================
    # ORCHESTRATION STEPS
    # ========================================================================

    def _check_trade_approval(
        self,
        trade_approver: TradeApprover,
        context: SignalContext,
        state: SymbolState,
    ) -> tuple[bool, Optional[str]]:
        """Check if trade should be executed using approver."""
        # Use base class helper to sync state with portfolio
        self._setup_approval_state(context.symbol, state)

        # Check with the approver (pass context directly)
        return trade_approver.should_trade(
            context=context,
            state=state,
            account_positions=len(self.portfolio.positions)
        )

    # _determine_action inherited from ExecutionEngineBase

    def _calculate_quantity(
        self,
        symbol: str,
        state: SymbolState,
        action_type: str,
        price: float,
        atr: float,
        regime: str,
        trade_approver: TradeApprover,
        **kwargs
    ) -> int:
        """Calculate position size using PositionManager for exits."""
        # Handle exits via PositionManager (consistent with LiveExecutionEngine)
        if action_type in ("exit", "reversal", "partial_exit"):
            position = self.portfolio.positions.get(symbol)
            if not position:
                return 0
            is_partial = action_type == "partial_exit"
            return self.position_manager.get_exit_quantity(position.qty, is_partial=is_partial)

        # For entries, use position sizer
        sl_mults = getattr(trade_approver, 'sl_mults', {"normal": 1.5})
        sl_mult = sl_mults.get(regime, 1.5)

        stop_loss_price = (price - (atr * sl_mult) if state.side != "short"
                          else price + (atr * sl_mult))

        return self.sizer.calculate_position_size(
            symbol=symbol,
            price=price,
            account_value=self.portfolio.total_value,
            signal_strength=1.0,
            atr=atr,
            stop_loss_price=stop_loss_price
        )
    
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
        """Execute trade via executor buy/sell methods."""
        try:
            # Use buy/sell methods directly (executor is a thin adapter)
            if side == OrderSide.BUY:
                self.executor.buy(symbol=symbol, qty=qty, price=price)
            else:
                self.executor.sell(symbol=symbol, qty=qty, price=price)

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
    
