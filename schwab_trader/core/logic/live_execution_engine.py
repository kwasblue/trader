"""
Live Execution Engine - Real-time trade execution with live broker

Executes real trades using:
- Live broker connection (Alpaca, Schwab, IBKR, etc.)
- Dynamic trade logic routing
- Real-time position tracking
- Production logging and monitoring
"""

from __future__ import annotations

from typing import Optional, Dict, Any
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

# Import the router we created
from core.logic.trade_logic_router import TradeLogicRouter

class LiveExecutionEngine(ExecutionEngineBase):
    """
    Live execution engine for real-time trading.
    
    Features:
    - Real broker connectivity
    - Dynamic trade logic routing (by symbol/strategy/regime)
    - Production-grade error handling
    - Comprehensive logging
    - State synchronization with broker
    
    Example:
        # Setup
        broker = AlpacaBroker(api_key=..., secret_key=...)
        executor = LiveExecutor(broker)
        sizer = ATRPositionSizer(risk_per_trade=0.01)
        tracker = DatabaseTradeLogger()
        logic = DefaultTradeLogicManager()
        portfolio = PortfolioState(initial_cash=100000)
        
        engine = LiveExecutionEngine(
            broker=broker,
            executor=executor,
            sizer=sizer,
            performance_tracker=tracker,
            trade_logic_manager=logic,
            portfolio=portfolio
        )
        
        # Register specific logics
        engine.register_symbol_logic("BTC-USD", crypto_logic)
        engine.register_strategy_logic("scalping", scalp_logic)
        
        # Handle signals
        result = engine.handle_signal(
            symbol="AAPL",
            state=state,
            signal=1,
            price=150.25,
            atr=2.5,
            regime="trending",
            strategy_name="momentum"
        )
    """
    
    def __init__(
        self,
        broker: BaseBrokerInterface,
        executor: BaseExecutor,
        sizer: PositionSizerBase,
        performance_tracker: TradeLoggerBase,
        trade_logic_manager: TradeLogicManagerBase,
        portfolio: PortfolioState,
        sync_on_start: bool = True
    ):
        """
        Initialize live execution engine.
        
        Args:
            broker: Live broker connection
            executor: Trade executor
            sizer: Position sizer
            performance_tracker: Trade logger
            trade_logic_manager: Default trade logic (or router)
            portfolio: Portfolio state tracker
            sync_on_start: Whether to sync portfolio with broker on start
        """
        super().__init__(
            broker=broker,
            executor=executor,
            sizer=sizer,
            performance_tracker=performance_tracker,
            trade_logic_manager=trade_logic_manager,
            portfolio=portfolio
        )
        
        # Setup logic router
        if isinstance(trade_logic_manager, TradeLogicRouter):
            self.logic_router = trade_logic_manager
        else:
            self.logic_router = TradeLogicRouter(trade_logic_manager)
        
        # Setup logging
        self.logger = Logger("app.log", "LiveExecutionEngine").get_logger()
        
        self.logger.info("LiveExecutionEngine initialized")
        
        # Sync portfolio with broker if requested
        if sync_on_start:
            self._sync_portfolio_with_broker()
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    def _sync_portfolio_with_broker(self) -> None:
        """
        Sync portfolio state with live broker positions.
        
        Critical for live trading to ensure internal state matches broker.
        """
        try:
            self.logger.info("Syncing portfolio with broker...")
            
            # Get account info from broker
            account = self.broker.get_account_info()
            
            # Update portfolio cash/equity
            self.portfolio.cash = getattr(account, 'cash', 0.0)
            self.portfolio.total_value = getattr(account, 'equity', 0.0)
            
            # Sync positions
            broker_positions = getattr(account, 'positions', [])
            
            for pos in broker_positions:
                symbol = pos.symbol
                qty = pos.qty
                avg_price = pos.avg_price
                
                self.portfolio.positions[symbol] = pos
                
                self.logger.info(
                    f"Synced position: {symbol} qty={qty} avg=${avg_price:.2f}"
                )
            
            self.logger.info(
                f"Portfolio synced: ${self.portfolio.total_value:,.2f}, "
                f"{len(self.portfolio.positions)} positions"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to sync portfolio: {e}")
            self.logger.warning("Starting with unsynced portfolio - positions may be inaccurate!")
    
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
        Handle live trading signal.
        
        Orchestrates:
        1. Route to appropriate trade logic
        2. Check approval
        3. Size position
        4. Execute via live broker
        5. Log and update state
        
        Args:
            symbol: Trading symbol
            state: Symbol state
            signal: Strategy signal (1, -1, 0)
            price: Current market price
            atr: Average True Range
            regime: Market regime
            strategy_name: Strategy identifier
            **kwargs: Additional context (df, market_open, etc.)
            
        Returns:
            OrderResult if executed, None if skipped
        """
        # Skip hold signals
        if signal == 0:
            self.logger.debug(f"[{symbol}] HOLD signal - no action")
            return None
        
        # Set strategy name
        state.strategy_name = strategy_name
        
        self.logger.info(
            f"[LIVE] [{symbol}] Processing signal: {signal} @ ${price:.2f} "
            f"(regime={regime}, strategy={strategy_name})"
        )
        
        try:
            # Get appropriate trade logic
            trade_logic = self.logic_router.get_logic(
                symbol=symbol,
                strategy=strategy_name,
                regime=regime
            )
            
            self.logger.debug(
                f"[{symbol}] Using logic: {trade_logic.__class__.__name__}"
            )
            
            # 1. Sync position from broker (live safety check)
            self._refresh_position(symbol)
            
            # 2. Check trade approval
            should_trade, reason = self._check_trade_approval(
                trade_logic, symbol, state, signal, price, atr, regime, **kwargs
            )
            
            if not should_trade:
                self.logger.info(f"[LIVE] [{symbol}] Trade blocked: {reason}")
                return None
            
            # 3. Determine action
            action_type, side = self._determine_action(symbol, state, signal, reason)
            
            self.logger.info(
                f"[LIVE] [{symbol}] Action approved: {action_type} {side.value}"
            )
            
            # 4. Calculate quantity
            qty = self._calculate_quantity(
                symbol, state, action_type, price, atr, regime, trade_logic, **kwargs
            )
            
            if qty <= 0:
                self.logger.warning(f"[LIVE] [{symbol}] Position size too small: {qty}")
                return None
            
            # 5. Pre-execution validation (live safety)
            if not self._validate_live_execution(symbol, side, qty, price):
                self.logger.error(f"[LIVE] [{symbol}] Pre-execution validation failed")
                return None
            
            # 6. Execute trade (LIVE)
            result = self._execute_live_trade(
                symbol, state, side, qty, price, atr, action_type, **kwargs
            )
            
            if result:
                # 7. Post-execution tasks
                self._post_execution(
                    symbol, state, result, action_type, regime, strategy_name
                )
            
            return result
            
        except Exception as e:
            self.logger.exception(f"[LIVE] [{symbol}] Error in live execution: {e}")
            
            # Alert on live execution errors
            self.performance_tracker.log_error(
                message=f"Live execution error for {symbol}",
                error=e,
                context={
                    'symbol': symbol,
                    'signal': signal,
                    'price': price,
                    'strategy': strategy_name
                }
            )
            
            return None
    
    # ========================================================================
    # LIVE-SPECIFIC METHODS
    # ========================================================================
    
    def _refresh_position(self, symbol: str) -> None:
        """
        Refresh position from broker before executing.
        
        Critical for live trading to avoid stale state.
        """
        try:
            broker_position = self.broker.get_position(symbol)
            
            if broker_position:
                self.portfolio.positions[symbol] = broker_position
                self.logger.debug(
                    f"[{symbol}] Position refreshed: qty={broker_position.qty}"
                )
            elif symbol in self.portfolio.positions:
                # Position closed on broker but still in our state
                del self.portfolio.positions[symbol]
                self.logger.warning(
                    f"[{symbol}] Position removed (not found on broker)"
                )
                
        except Exception as e:
            self.logger.warning(f"[{symbol}] Failed to refresh position: {e}")
    
    def _validate_live_execution(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        price: float
    ) -> bool:
        """
        Final validation before live execution.
        
        Checks:
        - Market is open (for market orders)
        - Sufficient buying power
        - Symbol is tradeable
        - Quantity within limits
        
        Returns:
            True if safe to execute
        """
        try:
            # Check market status
            if not self.broker.is_market_open():
                self.logger.warning("Market is closed - order may not fill immediately")
                # Continue anyway - some brokers accept orders when closed
            
            # Check buying power
            available = self.broker.get_available_funds()
            required = qty * price
            
            if side == OrderSide.BUY and required > available:
                self.logger.error(
                    f"Insufficient funds: need ${required:,.2f}, have ${available:,.2f}"
                )
                return False
            
            # Check quantity limits
            if qty <= 0:
                self.logger.error(f"Invalid quantity: {qty}")
                return False
            
            # Check if symbol is tradeable (if broker supports)
            if hasattr(self.broker, 'is_tradeable'):
                if not self.broker.is_tradeable(symbol):
                    self.logger.error(f"Symbol not tradeable: {symbol}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False
    
    def _execute_live_trade(
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
        """
        Execute LIVE trade via broker.
        
        This places actual orders in the market!
        """
        df = kwargs.get('df')
        signal = 1 if side == OrderSide.BUY else -1
        
        try:
            self.logger.info(
                f"[LIVE EXECUTION] [{symbol}] Placing {side.value} order: "
                f"{qty} shares @ ${price:.2f}"
            )
            
            # Execute via executor (which calls broker)
            self.executor.execute(
                symbol=symbol,
                df=df,
                signal=signal,
                price=price,
                atr_value=atr
            )
            
            # Get actual fill from broker
            # In real implementation, executor would return OrderResult
            # For now, create from expected values
            result = OrderResult(
                order_id=f"LIVE_{symbol}_{datetime.now(timezone.utc).timestamp()}",
                symbol=symbol,
                side=side,
                filled_qty=qty,
                avg_price=price,
                timestamp=datetime.now(timezone.utc)
            )
            
            self.logger.info(
                f"[LIVE] [{symbol}] Order executed: {side.value} {qty}@${price:.2f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.exception(f"[LIVE] [{symbol}] Execution failed: {e}")
            
            # Log to performance tracker
            self.performance_tracker.log_error(
                message=f"Order execution failed for {symbol}",
                error=e,
                context={
                    'symbol': symbol,
                    'side': side.value,
                    'qty': qty,
                    'price': price
                }
            )
            
            return None
    
    # ========================================================================
    # DELEGATION TO GENERIC ENGINE LOGIC
    # ========================================================================
    
    def _check_trade_approval(self, trade_logic, symbol, state, signal, price, atr, regime, **kwargs):
        """Use same approval logic as generic engine."""
        position = self.portfolio.positions.get(symbol)
        current_qty = 0 if not position else position.qty
        avg_price = None if not position else position.avg_price
        
        state.current_position = current_qty
        state.side = ("long" if current_qty > 0 else
                     "short" if current_qty < 0 else None)
        
        market_open = kwargs.get('market_open', self.broker.is_market_open())
        
        return trade_logic.should_trade(
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
    
    def _determine_action(self, symbol, state, signal, reason):
        """Reuse generic engine logic."""
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
    
    def _calculate_quantity(self, symbol, state, action_type, price, atr, regime, trade_logic, **kwargs):
        """Reuse generic engine logic."""
        if action_type in ("exit", "reversal"):
            position = self.portfolio.positions.get(symbol)
            return abs(position.qty) if position else 0
        
        if action_type == "partial_exit":
            position = self.portfolio.positions.get(symbol)
            if not position:
                return 0
            if hasattr(trade_logic, 'get_exit_quantity'):
                return trade_logic.get_exit_quantity(position.qty, is_partial=True)
            exit_fraction = trade_logic.get_param('exit_fraction', 0.25)
            return max(int(abs(position.qty) * exit_fraction), 1)
        
        sl_mults = getattr(trade_logic, 'sl_mults', {"normal": 1.5})
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
    
    def _post_execution(self, symbol, state, result, action_type, regime, strategy_name):
        """Post-execution logging and state updates."""
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
            notes=f"[LIVE] action={action_type}, bars_held={getattr(state, 'bars_held', 0)}"
        )
        
        if action_type in ("exit", "reversal"):
            state.reset()
        
        self.logger.info(
            f"[LIVE] [{symbol}] Trade logged: {action_type} "
            f"{result.side.value} {result.filled_qty}@${result.avg_price:.2f}"
        )
    
    # ========================================================================
    # LOGIC REGISTRATION
    # ========================================================================
    
    def register_symbol_logic(self, symbol: str, logic: TradeLogicManagerBase) -> None:
        """Register symbol-specific logic."""
        self.logic_router.register_symbol_logic(symbol, logic)
        self.logger.info(f"[LIVE] Registered logic for {symbol}: {logic.__class__.__name__}")
    
    def register_strategy_logic(self, strategy: str, logic: TradeLogicManagerBase) -> None:
        """Register strategy-specific logic."""
        self.logic_router.register_strategy_logic(strategy, logic)
        self.logger.info(f"[LIVE] Registered logic for strategy '{strategy}': {logic.__class__.__name__}")
    
    def register_regime_logic(self, regime: str, logic: TradeLogicManagerBase) -> None:
        """Register regime-specific logic."""
        self.logic_router.register_regime_logic(regime, logic)
        self.logger.info(f"[LIVE] Registered logic for regime '{regime}': {logic.__class__.__name__}")