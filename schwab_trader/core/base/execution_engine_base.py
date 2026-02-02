"""
Execution Engine - Orchestrates strategy signals into executed trades

This module defines the execution engine that coordinates between strategies,
executors, risk management, position sizing, and brokers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List
import logging

from core.base.base_broker_interface import BaseBrokerInterface
from core.base.executor_base import BaseExecutor
from core.base.position_sizer_base import PositionSizerBase
from core.base.trade_logger_base import TradeLoggerBase
from core.base.trade_logic_manager_base import TradeLogicManagerBase
from core.logic.portfolio_state import PortfolioState
from core.app_types import OrderResult

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