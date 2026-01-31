"""
Mock Execution Engine - Simulated trade execution for backtesting/paper trading

Simulates trade execution without real broker:
- Instant fills at market price
- No slippage (unless configured)
- Drawdown monitoring
- Event-driven signal subscription
- Portfolio state tracking
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from datetime import datetime, timezone
import asyncio
import logging

from core.base.execution_engine_base import ExecutionEngineBase
from core.base.executor_base import BaseExecutor
from core.base.base_broker_interface import BaseBrokerInterface
from core.base.position_sizer_base import PositionSizerBase
from core.base.trade_logger_base import TradeLoggerBase
from core.base.trade_logic_manager_base import TradeLogicManagerBase
from core.logic.default_trade_logic import DefaultTradeLogicManager
from core.logic.portfolio_state import PortfolioState
from core.logic.symbol_state import SymbolState
from core.drawdown_monitor import DrawdownMonitor
from core.app_types import OrderResult
from core.enums import OrderSide
from loggers.logger import Logger
from core.events.eventhandler import EventHandler
from core.events.events import (
    EVENT_NEW_TRADE, EVENT_POSITION_UPDATE, EVENT_PNL_UPDATE, EVENT_STRATEGY_SIGNAL,
    TradePayload, PositionPayload, PnLPayload
)
from core.logic.trade_logic_router import TradeLogicRouter


class MockExecutionEngine(ExecutionEngineBase):
    """
    Mock execution engine for simulation and testing.
    
    Features:
    - Simulated fills (no real broker)
    - Drawdown monitoring with trade locks
    - Event-driven signal subscription
    - Portfolio state tracking
    - Event emission for monitoring
    
    Use Cases:
    - Backtesting
    - Paper trading
    - Strategy development
    - Testing without risk
    
    Example:
        # Setup
        broker = MockBroker()
        executor = MockExecutor(broker)
        sizer = FixedSizer(shares=100)
        tracker = FileTradeLogger("trades.csv")
        logic = DefaultTradeLogicManager()
        portfolio = PortfolioState(initial_cash=100000)
        drawdown_monitor = DrawdownMonitor(max_drawdown=0.10)
        event_handler = AsyncEventHandler()
        
        engine = MockExecutionEngine(
            broker=broker,
            executor=executor,
            sizer=sizer,
            performance_tracker=tracker,
            trade_logic_manager=logic,
            portfolio=portfolio,
            drawdown_monitor=drawdown_monitor,
            event_handler=event_handler
        )
        
        # Subscribe to signals
        await engine.subscribe_signals()
        
        # Signals will be handled automatically via events
    """
    
    def __init__(
        self,
        broker: BaseBrokerInterface,
        executor: BaseExecutor,
        sizer: PositionSizerBase,
        performance_tracker: TradeLoggerBase,
        trade_logic_manager: TradeLogicManagerBase,
        portfolio: PortfolioState,
        drawdown_monitor: Optional[DrawdownMonitor] = None,
        event_handler: Optional[EventHandler] = None,
    ):
        """
        Initialize mock execution engine.
        
        Args:
            broker: Mock broker
            executor: Executor (can be mock)
            sizer: Position sizer
            performance_tracker: Trade logger
            trade_logic_manager: Trade logic (or router)
            portfolio: Portfolio state
            drawdown_monitor: Optional drawdown monitor for risk control
            event_handler: Optional event bus for signal subscription
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
        
        # Setup drawdown monitor
        self.drawdown_monitor = drawdown_monitor
        
        # Setup event handler
        self.event_handler = event_handler
        
        # Track symbol states
        self.symbol_states: Dict[str, SymbolState] = {}

        # Equity history for PnL events (equity_curve field)
        from collections import deque
        self.equity_history: deque = deque(maxlen=1000)
        self.equity_history.append(portfolio.cash)

        # Setup logging - own file with propagation to app.log
        self.logger = Logger(
            log_file="mock_engine.log",
            logger_name="MockExecutionEngine",
            propagate=True
        ).get_logger()
        
        self.logger.info("MockExecutionEngine initialized")
    
    # ========================================================================
    # MAIN SIGNAL HANDLING
    # ========================================================================
    
    async def handle_signal(
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
        Handle signal with mock execution.
        
        Simulates trade execution with instant fills.
        """
        # Skip hold signals
        if signal == 0:
            self.logger.debug(f"[{symbol}] HOLD signal - no action")
            return None
        
        # Check drawdown monitor
        if self.drawdown_monitor and not self.drawdown_monitor.can_trade(symbol):
            self.logger.warning(
                f"[{symbol}] Trade blocked by drawdown monitor "
                f"(drawdown lock or cooldown active)"
            )
            return None
        
        # Set strategy name
        state.strategy_name = strategy_name
        
        self.logger.info(
            f"[MOCK] [{symbol}] Processing signal: {signal} @ ${price:.2f} "
            f"(regime={regime}, strategy={strategy_name})"
        )
        
        try:
            # Get appropriate trade logic
            trade_logic  = self.trade_logic_manager.get(symbol, regime)

            
            self.logger.debug(
                f"[{symbol}] Using logic: {trade_logic.__class__.__name__}"
            )
            
            # 1. Check trade approval
            should_trade, reason = self._check_trade_approval(
                trade_logic, symbol, state, signal, price, atr, regime, **kwargs
            )
            
            if not should_trade:
                self.logger.info(f"[MOCK] [{symbol}] Trade blocked: {reason}")
                return None
            
            # 2. Determine action
            action_type, side = self._determine_action(symbol, state, signal, reason)
            
            self.logger.info(
                f"[MOCK] [{symbol}] Action approved: {action_type} {side.value}"
            )
            
            # 3. Calculate quantity - PASS SIGNAL IN KWARGS
            qty = self._calculate_quantity(
                symbol, state, action_type, price, atr, regime, trade_logic, 
                signal=signal, **kwargs
            )
            
            if qty <= 0:
                self.logger.warning(f"[MOCK] [{symbol}] Position size too small: {qty}")
                return None
            
            # 4. Execute mock trade (instant fill)
            result = self._execute_mock_trade(
                symbol, state, side, qty, price, action_type
            )
            
            if result:
                # 5. Update portfolio
                self._update_mock_portfolio(symbol, side, qty, price)
                
                # 6. Post-execution tasks
                self._post_execution(
                    symbol, state, result, action_type, regime, strategy_name
                )
                
                # 7. Emit events
                if self.event_handler:
                    await self._emit_trade_events(symbol, side, qty, price)
            
            return result
            
        except Exception as e:
            self.logger.exception(f"[MOCK] [{symbol}] Error in mock execution: {e}")
            return None
        
    # ========================================================================
    # MOCK-SPECIFIC METHODS
    # ========================================================================
    
    def _execute_mock_trade(
        self,
        symbol: str,
        state: SymbolState,
        side: OrderSide,
        qty: int,
        price: float,
        action_type: str
    ) -> OrderResult:
        """
        Execute mock trade with instant fill.
        
        Simulates:
        - Instant execution
        - Fill at market price (no slippage by default)
        - Order ID generation
        
        Returns:
            OrderResult with mock fill details
        """
        self.logger.info(
            f"[MOCK EXECUTION] [{symbol}] Simulating {side.value} order: "
            f"{qty} shares @ ${price:.2f}"
        )
        
        # Create mock order result (instant fill)
        result = OrderResult(
            success=True,
            order_id=f"MOCK_{symbol}_{datetime.now(timezone.utc).timestamp()}",
            symbol=symbol,
            side=side,
            filled_qty=qty,
            avg_price=price,  # Perfect fill at market price
        )
        
        self.logger.info(
            f"[MOCK] [{symbol}] Order filled: {side.value} {qty}@${price:.2f}"
        )
        
        return result
    
    def _update_mock_portfolio(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        price: float
    ) -> None:
        """
        Update portfolio state with mock fill.
        
        Args:
            symbol: Trading symbol
            side: Order side
            qty: Quantity filled
            price: Fill price
        """
        # Convert OrderSide to string for portfolio
        side_str = "buy" if side == OrderSide.BUY else "sell"
        
        # Apply fill to portfolio
        self.portfolio.apply_fill(symbol, side_str, qty, price)
        
        self.logger.debug(
            f"[{symbol}] Portfolio updated: "
            f"cash=${self.portfolio.cash:,.2f}, "
            f"equity=${self.portfolio.total_equity():,.2f}"
        )
        
        # Update drawdown monitor if present
        if self.drawdown_monitor:
            current_drawdown = self.portfolio.drawdown()
            #self.drawdown_monitor.update(symbol, current_drawdown)
            symbol_equity = self.portfolio.unrealized_pnl(symbol)
            self.drawdown_monitor.update_symbol(symbol, symbol_equity)
            
    
    async def _emit_trade_events(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        price: float
    ) -> None:
        """
        Emit trade-related events.
        
        Emits:
        - Trade event
        - Position update event
        - P&L update event
        """
        if not self.event_handler:
            return
        
        try:
            # Trade event
            trade_payload: TradePayload = {
                "symbol": symbol,
                "side": side.value.lower(),
                "qty": qty,
                "price": price,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pnl": self.portfolio.unrealized_pnl(symbol),
            }
            asyncio.create_task(
                self.event_handler.emit(EVENT_NEW_TRADE, trade_payload)
            )
            
            # Position update event
            position = self.portfolio.positions.get(symbol)
            if position:
                position_payload: PositionPayload = {
                    "symbol": symbol,
                    "qty": position.qty,
                    "avg_price": position.avg_price,
                    "unrealized": position.unrealized_pnl,
                    "realized": position.realized_pnl,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                asyncio.create_task(
                    self.event_handler.emit(EVENT_POSITION_UPDATE, position_payload)
                )
            
            # P&L update event
            equity = self.portfolio.total_equity()
            self.equity_history.append(equity)
            pnl_payload: PnLPayload = {
                "portfolio_value": equity,
                "equity_curve": list(self.equity_history),
                "unrealized": self.portfolio.total_unrealized(),
                "realized": self.portfolio.total_realized(),
                "drawdown": self.portfolio.drawdown(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            asyncio.create_task(
                self.event_handler.emit(EVENT_PNL_UPDATE, pnl_payload)
            )
            
            self.logger.debug(f"[{symbol}] Trade events emitted")
            
        except Exception as e:
            self.logger.error(f"[{symbol}] Failed to emit events: {e}")
    
    # ========================================================================
    # DELEGATION (Reuse logic from base)
    # ========================================================================

    def _check_trade_approval(self, trade_logic, symbol, state, signal, price, atr, regime, **kwargs):
        """Check if trade should execute."""
        position = self.portfolio.positions.get(symbol)
        current_qty = 0 if not position else position.qty
        avg_price = None if not position else position.avg_price
        
        state.current_position = current_qty
        state.side = ("long" if current_qty > 0 else
                    "short" if current_qty < 0 else None)
        
        market_open = kwargs.get('market_open', True)
        
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
        """Determine action type."""
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
        """Calculate position size."""
        if action_type in ("exit", "reversal"):
            position = self.portfolio.positions.get(symbol)
            return abs(position.qty) if position else 5
        
        if action_type == "partial_exit":
            position = self.portfolio.positions.get(symbol)
            if not position:
                return 5
            if hasattr(trade_logic, 'get_exit_quantity'):
                return trade_logic.get_exit_quantity(position.qty, is_partial=True)
            exit_fraction = trade_logic.get_param('exit_fraction', 0.25)
            return max(int(abs(position.qty) * exit_fraction), 1)
        
        # Get signal from kwargs
        signal = kwargs.get('signal', 0)
        
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
            stop_loss_price=stop_loss_price,
            signal=signal,
            portfolio=self.portfolio,
            market_conditions=regime
        )
    
    def _post_execution(self, symbol, state, result, action_type, regime, strategy_name):
        """Post-execution tasks."""
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
            notes=f"[MOCK] action={action_type}, bars_held={getattr(state, 'bars_held', 0)}"
        )
        
        if action_type in ("exit", "reversal"):
            state.reset()
        
        self.logger.info(
            f"[MOCK] [{symbol}] Trade logged: {action_type} "
            f"{result.side.value} {result.filled_qty}@${result.avg_price:.2f}"
        )
    
    # ========================================================================
    # EVENT SUBSCRIPTION
    # ========================================================================
    
    async def subscribe_signals(self) -> None:
        """
        Subscribe to strategy signal events.
        
        Automatically handles signals emitted to EVENT_STRATEGY_SIGNAL.
        Creates/manages SymbolState for each symbol.
        """
        if not self.event_handler:
            self.logger.warning("No event handler - cannot subscribe to signals")
            return
        
        async def on_signal(event):
            """Handle incoming signal event."""
            payload = event.payload
            symbol = payload["symbol"]
            
            # Parse signal
            signal_raw = payload["signal"]
            if signal_raw in (1, "buy"):
                sig_val = 1
            elif signal_raw in (-1, "sell"):
                sig_val = -1
            else:
                sig_val = 0
            
            # Ensure symbol state exists
            if symbol not in self.symbol_states:
                self.symbol_states[symbol] = SymbolState(symbol=symbol)
                self.logger.debug(f"[{symbol}] Created new SymbolState")
            
            state = self.symbol_states[symbol]
            
            # Handle signal
            self.handle_signal(
                symbol=symbol,
                state=state,
                signal=sig_val,
                price=payload.get("price", 0.0),
                atr=payload.get("atr", 0.0),
                regime=payload.get("regime", "normal"),
                strategy_name=payload.get("strategy")
            )
        
        await self.event_handler.subscribe(EVENT_STRATEGY_SIGNAL, on_signal)
        self.logger.info("Subscribed to strategy signals")
    
    # ========================================================================
    # LOGIC REGISTRATION
    # ========================================================================
    
    def register_symbol_logic(self, symbol: str, logic: TradeLogicManagerBase) -> None:
        """Register symbol-specific logic."""
        self.logic_router.register_symbol_logic(symbol, logic)
        self.logger.info(f"[MOCK] Registered logic for {symbol}: {logic.__class__.__name__}")
    
    def register_strategy_logic(self, strategy: str, logic: TradeLogicManagerBase) -> None:
        """Register strategy-specific logic."""
        self.logic_router.register_strategy_logic(strategy, logic)
        self.logger.info(f"[MOCK] Registered logic for strategy '{strategy}': {logic.__class__.__name__}")
    
    def register_regime_logic(self, regime: str, logic: TradeLogicManagerBase) -> None:
        """Register regime-specific logic."""
        self.logic_router.register_regime_logic(regime, logic)
        self.logger.info(f"[MOCK] Registered logic for regime '{regime}': {logic.__class__.__name__}")