"""
Execution Engine - Orchestrates strategy signals into executed trades

This module defines the execution engine that coordinates between strategies,
executors, risk management, position sizing, and brokers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Tuple
from datetime import datetime, timezone
import logging

from core.base.base_broker_interface import BaseBrokerInterface
from core.base.executor_base import BaseExecutor
from core.base.position_sizer_base import PositionSizerBase
from core.base.trade_logger_base import TradeLoggerBase
from core.base.trade_logic_manager_base import TradeLogicManagerBase
from core.logic.portfolio_state import PortfolioState
from core.app_types import OrderResult
from core.enums import OrderSide

logger = logging.getLogger(__name__)


class ExecutionEngineBase(ABC):
    """
    Abstract base class for execution engines.
    
    The execution engine is the orchestration layer that coordinates:
    - Strategy signals → Execution decisions
    - Position sizing → Order quantities
    - Risk management → Trade approval
    - Broker interface → Order execution
    - Performance tracking → Trade logging
    
    Architecture Flow:
        Strategy generates signal (1, -1, 0)
              ↓
        Engine.handle_signal() orchestrates:
              ↓
        1. Trade Logic Manager → Should we trade?
        2. Position Sizer → How much to trade?
        3. Executor → Execute the trade
        4. Broker → Route the order
        5. Trade Logger → Record the result
    
    Design Philosophy:
    - Engine is the conductor, not the performer
    - Delegates to specialized components
    - Maintains no trading logic itself
    - Coordinates state updates across components
    
    Example:
        class MyExecutionEngine(ExecutionEngineBase):
            def handle_signal(self, symbol, state, signal, price, atr, regime, **kwargs):
                # 1. Check if we should trade
                if not self.trade_logic_manager.should_trade(state, signal):
                    return None
                
                # 2. Calculate position size
                qty = self.sizer.calculate_size(symbol, price, atr)
                
                # 3. Execute trade
                result = self.executor.execute(symbol, df, signal, price, atr)
                
                # 4. Log performance
                if result:
                    self.performance_tracker.log_trade(result)
                
                return result
    """
    
    def __init__(
        self,
        broker: BaseBrokerInterface,
        executor: BaseExecutor,
        sizer: PositionSizerBase,
        performance_tracker: TradeLoggerBase,
        trade_logic_manager: TradeLogicManagerBase,
        portfolio: PortfolioState,
    ):
        """
        Initialize execution engine with all required components.
        
        Args:
            broker: Broker interface for market access
            executor: Executor for signal interpretation and order placement
            sizer: Position sizer for calculating trade quantities
            performance_tracker: Trade logger for performance tracking
            trade_logic_manager: Logic manager for trade approval
            portfolio: Portfolio state tracker
        """
        self.broker = broker
        self.executor = executor
        self.sizer = sizer
        self.performance_tracker = performance_tracker
        self.trade_logic_manager = trade_logic_manager
        self.portfolio = portfolio
        
        logger.info("ExecutionEngine initialized")
    
    # ========================================================================
    # ABSTRACT METHODS
    # ========================================================================
    
    @abstractmethod
    def handle_signal(
        self,
        symbol: str,
        state: Any,
        signal: int,
        price: float,
        atr: float,
        regime: str,
        strategy_name: Optional[str] = None,
        **kwargs
    ) -> Optional[OrderResult]:
        """
        Execute trade logic based on a new signal from strategy.
        
        This is the main entry point for processing signals. The engine should:
        1. Validate signal and market conditions
        2. Check trade approval via trade_logic_manager and trade gate
        3. Calculate position size via sizer
        4. Execute trade via executor
        5. Update portfolio state
        6. Log trade via performance_tracker
        
        Args:
            symbol: Trading symbol (e.g., "AAPL")
            state: Symbol-specific state object (SymbolState or similar)
                   Contains position info, last trade time, P&L, etc.
            signal: Strategy signal
                   - +1: Buy/Long signal
                   - -1: Sell/Short signal
                   - 0: Hold/No action
            price: Current market price
            atr: Average True Range (for volatility-based sizing)
            regime: Market regime classification (e.g., "trending", "ranging")
            strategy_name: Optional identifier for which strategy generated signal
            **kwargs: Additional context (indicators, metadata, etc.)
        
        Returns:
            OrderResult if trade was executed, None if skipped
            
        Example Implementation:
            def handle_signal(self, symbol, state, signal, price, atr, regime, **kwargs):
                # Skip if signal is hold
                if signal == 0:
                    return None
                
                # Check if we should trade
                if not self.trade_logic_manager.should_trade(
                    symbol, state, signal, regime
                ):
                    logger.info(f"Trade logic blocked signal for {symbol}")
                    return None
                
                # Calculate position size
                qty = self.sizer.calculate_size(
                    symbol=symbol,
                    price=price,
                    atr=atr,
                    account_value=self.portfolio.total_value,
                    signal_strength=1.0
                )
                
                # Get market data for executor
                df = kwargs.get('df')
                
                # Execute trade
                result = self.executor.execute(
                    symbol=symbol,
                    df=df,
                    signal=signal,
                    price=price,
                    atr_value=atr
                )
                
                # Update portfolio
                if result:
                    self.portfolio.update_position(symbol, result)
                    self.performance_tracker.log_trade(result)
                
                return result
        """
        pass

    # ========================================================================
    # SHARED IMPLEMENTATIONS
    # ========================================================================

    def _determine_action(
        self,
        symbol: str,
        state: Any,
        signal: int,
        reason: Optional[str]
    ) -> Tuple[str, OrderSide]:
        """
        Determine action type and order side from signal.

        This is a shared implementation used by both Live and Mock engines.

        Args:
            symbol: Trading symbol
            state: Symbol state with position info
            signal: Strategy signal (1 for long, -1 for short)
            reason: Reason string (may contain "partial" or "reversal")

        Returns:
            Tuple of (action_type, order_side)
            - action_type: "entry", "exit", "partial_exit", or "reversal"
            - order_side: OrderSide.BUY or OrderSide.SELL
        """
        in_position = state.side is not None

        if not in_position:
            return "entry", OrderSide.BUY if signal == 1 else OrderSide.SELL
        elif reason and "partial" in reason.lower():
            return "partial_exit", (OrderSide.SELL if state.side == "long"
                                   else OrderSide.BUY)
        elif reason and "reversal" in reason.lower():
            return "reversal", (OrderSide.SELL if state.side == "long"
                               else OrderSide.BUY)
        else:
            return "exit", (OrderSide.SELL if state.side == "long"
                           else OrderSide.BUY)

    def _setup_approval_state(
        self,
        symbol: str,
        state: Any
    ) -> Tuple[int, Optional[float]]:
        """
        Set up state for trade approval check.

        Syncs the state object with the current portfolio position.

        Args:
            symbol: Trading symbol
            state: Symbol state to update

        Returns:
            Tuple of (current_qty, avg_price)
        """
        position = self.portfolio.positions.get(symbol)
        current_qty = 0 if not position else position.qty
        avg_price = None if not position else position.avg_price

        state.current_position = current_qty
        state.side = ("long" if current_qty > 0 else
                     "short" if current_qty < 0 else None)

        return current_qty, avg_price

    def _get_exit_quantity(
        self,
        symbol: str,
        action_type: str,
        trade_logic: Any
    ) -> int:
        """
        Calculate quantity for exit/partial_exit actions.

        Args:
            symbol: Trading symbol
            action_type: Type of action ("exit", "reversal", "partial_exit")
            trade_logic: Trade logic manager for partial exit params

        Returns:
            Exit quantity (positive integer)
        """
        position = self.portfolio.positions.get(symbol)

        if action_type in ("exit", "reversal"):
            return abs(position.qty) if position else 0

        if action_type == "partial_exit":
            if not position:
                return 0
            if hasattr(trade_logic, 'get_exit_quantity'):
                return trade_logic.get_exit_quantity(position.qty, is_partial=True)
            exit_fraction = trade_logic.get_param('exit_fraction', 0.25)
            return max(int(abs(position.qty) * exit_fraction), 1)

        return 0  # Not an exit action

    def _post_execution(
        self,
        symbol: str,
        state: Any,
        result: OrderResult,
        action_type: str,
        regime: str,
        strategy_name: Optional[str]
    ) -> None:
        """
        Post-execution logging and state updates.

        Shared implementation that:
        1. Calls _update_portfolio_after_execution hook
        2. Updates state.last_trade_time
        3. Logs trade to performance tracker
        4. Resets state on exit/reversal

        Args:
            symbol: Trading symbol
            state: Symbol state to update
            result: Order execution result
            action_type: Type of action taken
            regime: Market regime
            strategy_name: Strategy that generated signal
        """
        # Hook for subclasses to handle portfolio updates
        self._update_portfolio_after_execution(symbol, result)

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

    def _update_portfolio_after_execution(
        self,
        symbol: str,
        result: OrderResult
    ) -> None:
        """
        Hook for portfolio updates after execution.

        Default implementation updates portfolio from order result.
        Override in subclasses that handle portfolio updates differently.

        Args:
            symbol: Trading symbol
            result: Order execution result
        """
        self.portfolio.update_position(symbol, result)

    # ========================================================================
    # OPTIONAL METHODS (Can be overridden)
    # ========================================================================
    
    def on_market_open(self) -> None:
        """
        Called when market opens.
        
        Override to implement open-of-day logic like:
        - Cancel all pending orders
        - Reset daily counters
        - Apply gap adjustments
        """
        pass
    
    def on_market_close(self) -> None:
        """
        Called when market closes.
        
        Override to implement close-of-day logic like:
        - Close all positions (if day trading)
        - Cancel open orders
        - Calculate daily P&L
        - Save state
        """
        pass
    
    def on_new_bar(self, symbol: str, bar: Dict[str, Any]) -> None:
        """
        Called when new bar data arrives.
        
        Override to implement bar-level logic like:
        - Update indicators
        - Check stop losses
        - Adjust trailing stops
        
        Args:
            symbol: Trading symbol
            bar: New bar data with OHLCV fields
        """
        pass
    
    def emergency_stop(self) -> None:
        """
        Emergency stop - cancel all orders and close all positions.
        
        Override to implement custom emergency shutdown logic.
        Default implementation closes everything immediately.
        """
        logger.critical("EMERGENCY STOP TRIGGERED")
        
        try:
            # Cancel all open orders
            open_orders = self.broker.get_open_orders()
            for order in open_orders:
                try:
                    self.broker.cancel_order(order.order_id)
                except Exception as e:
                    logger.error(f"Failed to cancel order {order.order_id}: {e}")
            
            # Close all positions
            for symbol, position in self.portfolio.positions.items():
                if position.qty != 0:
                    try:
                        self.executor.close_position(symbol)
                    except Exception as e:
                        logger.error(f"Failed to close position {symbol}: {e}")
            
            logger.info("Emergency stop completed")
            
        except Exception as e:
            logger.critical(f"Emergency stop failed: {e}")
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get current portfolio summary.
        
        Returns:
            Dictionary with portfolio metrics:
            - total_value: Total portfolio value
            - cash: Available cash
            - positions: Open positions
            - unrealized_pnl: Unrealized P&L
            - realized_pnl: Realized P&L
            - daily_pnl: Today's P&L
        """
        return {
            'total_value': self.portfolio.total_value,
            'cash': self.portfolio.cash,
            'positions': self.portfolio.get_positions(),
            'unrealized_pnl': self.portfolio.unrealized_pnl,
            'realized_pnl': self.portfolio.realized_pnl,
            'daily_pnl': self.portfolio.get_daily_pnl(),
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics from tracker.
        
        Returns:
            Dictionary with performance metrics:
            - total_trades: Number of trades
            - win_rate: Percentage of winning trades
            - profit_factor: Gross profit / gross loss
            - sharpe_ratio: Risk-adjusted return
            - max_drawdown: Maximum drawdown
            - ... other metrics from tracker
        """
        return self.performance_tracker.get_metrics()
    
    def validate_state(self) -> bool:
        """
        Validate internal state consistency.
        
        Returns:
            True if state is valid
            
        Raises:
            ValueError: If state is inconsistent
        """
        # Check broker connection
        if not self.broker:
            raise ValueError("Broker not initialized")
        
        # Check portfolio state
        if self.portfolio.total_value < 0:
            raise ValueError("Portfolio value is negative")
        
        # Check position sizes match broker
        for symbol, position in self.portfolio.positions.items():
            broker_position = self.broker.get_position(symbol)
            if broker_position and abs(broker_position.qty - position.qty) > 0.01:
                logger.warning(
                    f"Position mismatch for {symbol}: "
                    f"portfolio={position.qty}, broker={broker_position.qty}"
                )
        
        return True
    
    # ========================================================================
    # SPECIAL METHODS
    # ========================================================================
    
    def __repr__(self) -> str:
        """String representation of execution engine."""
        return (
            f"{self.__class__.__name__}("
            f"broker={self.broker.__class__.__name__}, "
            f"portfolio_value=${self.portfolio.total_value:,.2f})"
        )