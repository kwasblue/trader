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
from core.app_types import OrderResult, SignalContext
from core.enums import OrderSide
from loggers.logger import Logger
from core.events.eventhandler import EventHandler
from core.events.events import (
    EVENT_NEW_TRADE, EVENT_POSITION_UPDATE, EVENT_PNL_UPDATE, EVENT_STRATEGY_SIGNAL,
    EVENT_FLATTEN_ALL, EVENT_CANCEL_ALL, EVENT_FLATTEN_SYMBOL, EVENT_MANUAL_ORDER,
    EVENT_HALTED, EVENT_ALERT, EVENT_ORDER_STATUS,
    TradePayload, PositionPayload, PnLPayload, AlertPayload, OrderStatusPayload
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

        # Halt state for GUI control
        self._halted = False

        self.logger.info("MockExecutionEngine initialized")
    
    # ========================================================================
    # MAIN SIGNAL HANDLING
    # ========================================================================

    # Minimum confidence threshold for signal processing
    min_signal_confidence: float = 0.0

    async def handle_signal_context(
        self,
        context: SignalContext
    ) -> Optional[OrderResult]:
        """
        Handle signal with mock execution using unified context.

        Simulates trade execution with instant fills.

        Args:
            context: SignalContext containing all signal data

        Returns:
            OrderResult if executed, None if skipped
        """
        # Filter low confidence signals
        if context.confidence < self.min_signal_confidence:
            self.logger.debug(
                f"[{context.symbol}] Signal confidence {context.confidence:.2f} "
                f"below threshold {self.min_signal_confidence:.2f}"
            )
            return None

        # Check if trading is halted
        if getattr(self, '_halted', False):
            self.logger.debug(f"[{context.symbol}] Trading halted - ignoring signal")
            return None

        # Get or create state from metadata or symbol_states
        state = context.metadata.get('state')
        if state is None:
            if context.symbol not in self.symbol_states:
                self.symbol_states[context.symbol] = SymbolState(symbol=context.symbol)
            state = self.symbol_states[context.symbol]

        # Track bar for bar-based cooldown (call on every signal = every bar)
        if hasattr(self, 'trade_logic_manager') and self.trade_logic_manager:
            # Get the trade logic and notify of new bar
            trade_logic = self.trade_logic_manager.get(context.symbol, context.regime)
            if hasattr(trade_logic, 'on_bar'):
                trade_logic.on_bar(context.symbol)

        # Skip hold signals
        if context.is_hold():
            self.logger.debug(f"[{context.symbol}] HOLD signal - no action")
            return None

        # Drawdown monitor check - only if drawdown_monitor is set
        if self.drawdown_monitor and not self.drawdown_monitor.can_trade(context.symbol):
            self.logger.warning(
                f"[{context.symbol}] Trade blocked by drawdown monitor "
                f"(drawdown lock or cooldown active)"
            )
            return None

        # Set strategy name
        state.strategy_name = context.strategy_name

        self.logger.info(
            f"[MOCK] [{context.symbol}] Processing signal: {context.signal} @ ${context.price:.2f} "
            f"(regime={context.regime}, strategy={context.strategy_name}, confidence={context.confidence:.2f})"
        )

        try:
            # Get appropriate trade logic
            trade_logic = self.trade_logic_manager.get(context.symbol, context.regime)

            self.logger.debug(
                f"[{context.symbol}] Using logic: {trade_logic.__class__.__name__}"
            )

            # 1. Check trade approval
            # Filter out 'state' from metadata to avoid duplicate argument
            extra_kwargs = {k: v for k, v in context.metadata.items() if k != 'state'}
            should_trade, reason = self._check_trade_approval(
                trade_logic, context.symbol, state, context.signal,
                context.price, context.atr, context.regime,
                market_open=context.market_open, df=context.df, **extra_kwargs
            )

            if not should_trade:
                self.logger.info(f"[MOCK] [{context.symbol}] Trade blocked: {reason}")
                return None

            # 2. Determine action
            action_type, side = self._determine_action(
                context.symbol, state, context.signal, reason
            )

            self.logger.info(
                f"[MOCK] [{context.symbol}] Action approved: {action_type} {side.value}"
            )

            # 3. Calculate quantity - PASS SIGNAL IN KWARGS
            qty = self._calculate_quantity(
                context.symbol, state, action_type, context.price, context.atr,
                context.regime, trade_logic, signal=context.signal,
                df=context.df, **extra_kwargs
            )

            if qty <= 0:
                self.logger.warning(f"[MOCK] [{context.symbol}] Position size too small: {qty}")
                return None

            # 4. Execute mock trade (instant fill)
            result = self._execute_mock_trade(
                context.symbol, state, side, qty, context.price, action_type
            )

            if result:
                # 5. Update portfolio
                self._update_mock_portfolio(context.symbol, side, qty, context.price)

                # 6. Post-execution tasks
                self._post_execution(
                    context.symbol, state, result, action_type, context.regime, context.strategy_name
                )

                # 7. Reset bar-based cooldown after trade
                if hasattr(trade_logic, 'on_trade'):
                    trade_logic.on_trade(context.symbol)

                # 8. Emit events
                if self.event_handler:
                    await self._emit_trade_events(context.symbol, side, qty, context.price)

            return result

        except Exception as e:
            self.logger.exception(f"[MOCK] [{context.symbol}] Error in mock execution: {e}")
            return None

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
        DEPRECATED: Use handle_signal_context() instead.

        Handle signal with mock execution (backward compatible wrapper).

        Simulates trade execution with instant fills.
        """
        from datetime import datetime, timezone

        # Create SignalContext from parameters
        context = SignalContext.from_kwargs(
            symbol=symbol,
            signal=signal,
            price=price,
            atr=atr,
            regime=regime,
            timestamp=datetime.now(timezone.utc),
            strategy_name=strategy_name,
            df=kwargs.get('df'),
            market_open=kwargs.get('market_open', True),
            state=state,
            **{k: v for k, v in kwargs.items() if k not in ('df', 'market_open')}
        )

        return await self.handle_signal_context(context)
        
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
            # Update portfolio-level drawdown first
            portfolio_equity = self.portfolio.total_equity()
            self.drawdown_monitor.update_portfolio(portfolio_equity)

            # Update symbol-level drawdown using position market value (not unrealized P&L)
            position = self.portfolio.positions.get(symbol)
            if position and position.qty != 0:
                # Use position market value (qty * last_price) as symbol equity
                symbol_equity = abs(position.qty) * position.last_price
            else:
                # No position - skip symbol update to avoid 0-value issues
                symbol_equity = None

            if symbol_equity is not None and symbol_equity > 0:
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
        - Order status event
        - Trade event
        - Position update event
        - P&L update event
        """
        if not self.event_handler:
            return

        try:
            timestamp = datetime.now(timezone.utc).isoformat()

            # Order status event (for Execution tab)
            order_payload: OrderStatusPayload = {
                "order_id": f"MOCK_{symbol}_{timestamp}",
                "symbol": symbol,
                "status": "filled",
                "filled_qty": float(qty),
                "avg_price": float(price),
                "timestamp": timestamp,
            }
            asyncio.create_task(
                self.event_handler.emit(EVENT_ORDER_STATUS, order_payload)
            )

            # Trade event
            trade_payload: TradePayload = {
                "symbol": symbol,
                "side": side.value.lower(),
                "qty": qty,
                "price": price,
                "timestamp": timestamp,
                "pnl": self.portfolio.unrealized_pnl(symbol),
            }
            asyncio.create_task(
                self.event_handler.emit(EVENT_NEW_TRADE, trade_payload)
            )
            
            # Position update event
            position = self.portfolio.positions.get(symbol)
            if position:
                side = "long" if position.qty > 0 else "short" if position.qty < 0 else "flat"
                position_payload: PositionPayload = {
                    "symbol": symbol,
                    "qty": position.qty,
                    "avg_price": position.avg_price,
                    "avg": position.avg_price,  # GUI model uses 'avg'
                    "last": position.last_price,  # GUI model uses 'last'
                    "unrealized": position.unrealized_pnl,
                    "unreal": position.unrealized_pnl,  # GUI model uses 'unreal'
                    "realized": position.realized_pnl,
                    "side": side,
                    "market_value": position.qty * position.last_price,
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
        """Check trade approval (sync, no broker call)."""
        current_qty, avg_price = self._setup_approval_state(symbol, state)
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

    # _determine_action() is inherited from ExecutionEngineBase

    def _calculate_quantity(self, symbol, state, action_type, price, atr, regime, trade_logic, **kwargs):
        """Calculate position size, use base class for exits with mock fallback."""
        # Handle exits via base class helper
        if action_type in ("exit", "reversal", "partial_exit"):
            qty = self._get_exit_quantity(symbol, action_type, trade_logic)
            return qty if qty > 0 else 5  # Mock fallback for testing

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

    def _update_portfolio_after_execution(self, symbol: str, result) -> None:
        """Skip portfolio update - already handled in _update_mock_portfolio()."""
        pass

    def _post_execution(self, symbol, state, result, action_type, regime, strategy_name):
        """Post-execution tasks with Mock-specific logging."""
        # Use base class implementation for common logic
        super()._post_execution(symbol, state, result, action_type, regime, strategy_name)

        # Mock-specific logging
        self.logger.info(
            f"[MOCK] [{symbol}] Trade logged: {action_type} "
            f"{result.side.value} {result.filled_qty}@${result.avg_price:.2f}"
        )
    
    # ========================================================================
    # EVENT SUBSCRIPTION
    # ========================================================================
    
    async def subscribe_signals(self) -> None:
        """
        Subscribe to strategy signal events and GUI commands.

        Automatically handles signals emitted to EVENT_STRATEGY_SIGNAL.
        Creates/manages SymbolState for each symbol.
        Also handles GUI commands (flatten, cancel, halt, manual orders).
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
            await self.handle_signal(
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

        # Subscribe to GUI commands
        await self._subscribe_gui_events()

    async def _subscribe_gui_events(self) -> None:
        """Subscribe to GUI command events (flatten, cancel, halt, manual orders)."""
        if not self.event_handler:
            return

        await self.event_handler.subscribe(EVENT_FLATTEN_ALL, self._handle_flatten_all)
        await self.event_handler.subscribe(EVENT_FLATTEN_SYMBOL, self._handle_flatten_symbol)
        await self.event_handler.subscribe(EVENT_CANCEL_ALL, self._handle_cancel_all)
        await self.event_handler.subscribe(EVENT_MANUAL_ORDER, self._handle_manual_order)
        await self.event_handler.subscribe(EVENT_HALTED, self._handle_halt)

        self.logger.info("Subscribed to GUI command events")

    async def _handle_flatten_all(self, event) -> None:
        """Flatten all positions in the mock portfolio."""
        self.logger.info("[GUI] Flatten ALL positions (mock)")

        for symbol, position in list(self.portfolio.positions.items()):
            if position.qty != 0:
                try:
                    qty = abs(position.qty)
                    side = OrderSide.SELL if position.qty > 0 else OrderSide.BUY
                    price = position.last_price

                    # Execute mock close
                    self._update_mock_portfolio(symbol, side, qty, price)

                    # Emit events
                    if self.event_handler:
                        await self._emit_trade_events(symbol, side, qty, price)

                    self.logger.info(f"[FLATTEN] Closed {symbol}: {qty} @ ${price:.2f}")
                except Exception as e:
                    self.logger.error(f"[FLATTEN] Failed for {symbol}: {e}")

        self._emit_alert("info", "All positions flattened")

    async def _handle_flatten_symbol(self, event) -> None:
        """Flatten a specific symbol."""
        symbol = event.payload.get("symbol")
        if not symbol:
            return

        position = self.portfolio.positions.get(symbol)
        if not position or position.qty == 0:
            self.logger.info(f"[FLATTEN] No position in {symbol}")
            return

        try:
            qty = abs(position.qty)
            side = OrderSide.SELL if position.qty > 0 else OrderSide.BUY
            price = position.last_price

            self._update_mock_portfolio(symbol, side, qty, price)

            if self.event_handler:
                await self._emit_trade_events(symbol, side, qty, price)

            self.logger.info(f"[FLATTEN] Closed {symbol}: {qty} @ ${price:.2f}")
            self._emit_alert("info", f"Position {symbol} flattened")
        except Exception as e:
            self.logger.error(f"[FLATTEN] Failed for {symbol}: {e}")

    async def _handle_cancel_all(self, event) -> None:
        """Cancel all orders (no-op in mock since fills are instant)."""
        self.logger.info("[GUI] Cancel all orders (mock - no pending orders)")
        self._emit_alert("info", "All orders cancelled (mock)")

    async def _handle_manual_order(self, event) -> None:
        """Handle manual order from GUI."""
        payload = event.payload
        symbol = payload.get("symbol", "")
        qty = int(payload.get("qty", 0))
        side_str = payload.get("side", "BUY").upper()
        price = float(payload.get("price", 0))

        if not symbol or qty <= 0:
            self.logger.warning(f"[MANUAL] Invalid order: {payload}")
            return

        side = OrderSide.BUY if side_str == "BUY" else OrderSide.SELL

        self.logger.info(f"[MANUAL] {side_str} {qty} {symbol} @ ${price:.2f}")

        # Execute mock trade
        self._update_mock_portfolio(symbol, side, qty, price)

        if self.event_handler:
            await self._emit_trade_events(symbol, side, qty, price)

        self._emit_alert("info", f"Manual order executed: {side_str} {qty} {symbol}")

    async def _handle_halt(self, event) -> None:
        """Handle halt/resume trading."""
        halted = event.payload.get("halted", False)
        self._halted = halted
        state = "HALTED" if halted else "RESUMED"
        self.logger.info(f"[GUI] Trading {state}")
        self._emit_alert("warning" if halted else "info", f"Trading {state}")

    def _emit_alert(self, level: str, message: str, symbol: str = None) -> None:
        """Emit an alert event."""
        if not self.event_handler:
            return

        payload: AlertPayload = {
            "level": level,
            "message": message,
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        asyncio.create_task(self.event_handler.emit(EVENT_ALERT, payload))
    
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