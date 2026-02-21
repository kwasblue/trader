"""
Live Execution Engine - Real-time trade execution with live broker

Executes real trades using:
- Live broker connection (Alpaca, Schwab, IBKR, etc.)
- Dynamic trade logic routing
- Real-time position tracking
- Production logging and monitoring
"""

from __future__ import annotations

from typing import Optional, Dict, Any, TYPE_CHECKING, Union
import asyncio
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
from core.logic.position_manager import PositionManager
from core.app_types import OrderResult, SignalContext
from core.enums import OrderSide, PositionState
from core.order_registry import OrderRegistry
from core.trade_validator import TradeValidator
from core.logging_config import (
    get_component_logger,
    generate_correlation_id,
    set_correlation_id,
    format_log_message,
)
from loggers.logger import Logger
from core.contracts.events import (
    EVENT_ALERT, AlertPayload, EVENT_NEW_TRADE, TradePayload,
    EVENT_MANUAL_ORDER, EVENT_FLATTEN_ALL, EVENT_FLATTEN_SYMBOL, EVENT_CANCEL_ALL,
)
from core.exceptions import DailyLossLimitExceededError

# Import the router we created
from core.logic.trade_logic_router import TradeApproverRouter

if TYPE_CHECKING:
    from core.state_reconciler import StateReconciler
    from core.drawdown_monitor import DrawdownMonitor
    from core.state_sync import StateSynchronizer

# Default PositionManager instance
_default_position_manager: Optional[PositionManager] = None


def get_default_position_manager() -> PositionManager:
    """Get or create the default PositionManager instance."""
    global _default_position_manager
    if _default_position_manager is None:
        _default_position_manager = PositionManager()
    return _default_position_manager

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
        
        # Create router with default logic and register overrides
        router = TradeApproverRouter(logic)
        router.register_symbol_approver("BTC-USD", crypto_approver)
        router.register_strategy_approver("scalping", scalp_approver)

        engine = LiveExecutionEngine(
            broker=broker,
            executor=executor,
            sizer=sizer,
            performance_tracker=tracker,
            trade_logic_manager=router,  # Pass router, not raw logic
            portfolio=portfolio
        )

        # Handle signals (router resolves approver automatically)
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
        sync_on_start: bool = True,
        reconciler: Optional["StateReconciler"] = None,
        event_handler: Optional[Any] = None,
        daily_loss_limit: Optional[float] = None,
        drawdown_monitor: Optional["DrawdownMonitor"] = None,
        position_manager: Optional[PositionManager] = None,
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
            reconciler: State reconciler for halt checking
            event_handler: Event handler for emitting trade decision events
            daily_loss_limit: Maximum allowed daily loss (absolute $). None = no limit.
            drawdown_monitor: DrawdownMonitor for per-symbol and portfolio drawdown control.
                              If provided, trades are blocked when drawdown limits are breached.
            position_manager: PositionManager for position lifecycle (SL/TP/exits).
                              If not provided, a default instance is created.
        """
        super().__init__(
            broker=broker,
            executor=executor,
            sizer=sizer,
            performance_tracker=performance_tracker,
            trade_logic_manager=trade_logic_manager,
            portfolio=portfolio,
            position_manager=position_manager or get_default_position_manager(),
        )

        # Store reconciler reference for halt checking
        self.reconciler = reconciler

        # DrawdownMonitor for per-symbol and portfolio-level risk control
        self.drawdown_monitor = drawdown_monitor

        # Event handler for GUI notifications
        self.event_handler = event_handler

        # Setup approver router (single authority for approver selection)
        if isinstance(trade_logic_manager, TradeApproverRouter):
            self.approver_router = trade_logic_manager
        else:
            self.approver_router = TradeApproverRouter(trade_logic_manager)

        # Initialize order registry for local order tracking
        self.order_registry = OrderRegistry()

        # Initialize trade validator
        self.validator = TradeValidator(
            portfolio=portfolio,
            order_registry=self.order_registry,
            reconciler=reconciler,
            broker=broker,
        )

        # Setup logging with structured support
        self.logger = Logger("execution_engine.log", "LiveExecutionEngine", propagate=True).get_logger()

        # Daily loss limit tracking
        self.daily_loss_limit = daily_loss_limit
        self._daily_realized_pnl: float = 0.0
        self._daily_loss_limit_breached: bool = False
        self._trading_day_start: Optional[datetime] = None

        # Initialize symbol states dict for state synchronization
        if not hasattr(self, 'symbol_states'):
            self.symbol_states: Dict[str, SymbolState] = {}

        # State synchronizer for Portfolio <-> Symbol consistency
        from core.state_sync import StateSynchronizer
        self._state_sync = StateSynchronizer(
            portfolio=self.portfolio,
            symbol_states=self.symbol_states,
        )

        self.logger.info(
            f"LiveExecutionEngine initialized with order registry and validator"
            f"{f', daily_loss_limit=${daily_loss_limit:,.2f}' if daily_loss_limit else ''}"
            f"{', drawdown_monitor=enabled' if drawdown_monitor else ''}"
        )

        # Sync portfolio with broker if requested
        # NOTE: sync_on_start should be False for live trading -
        # the AlpacaRunner/SchwabRunner handles sync via StateReconciler
        if sync_on_start:
            # Can't await in __init__, use asyncio.run for one-time sync
            import asyncio
            asyncio.run(self._sync_portfolio_with_broker())
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    async def _sync_portfolio_with_broker(self) -> None:
        """
        Sync portfolio state with live broker positions.

        Critical for live trading to ensure internal state matches broker.
        """
        try:
            self.logger.info("Syncing portfolio with broker...")

            # Get account info from broker
            account = await self.broker.get_account_info()

            # Update portfolio cash/equity
            self.portfolio.cash = getattr(account, 'cash', 0.0)
            self.portfolio.total_value = getattr(account, 'equity', 0.0)

            # Sync positions
            broker_positions = getattr(account, 'positions', {})

            for symbol, pos in broker_positions.items():
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
    # DAILY LOSS LIMIT MANAGEMENT
    # ========================================================================

    def start_new_trading_day(self) -> None:
        """
        Reset daily P&L tracking for a new trading day.

        Call this at market open or session start.
        """
        self._daily_realized_pnl = 0.0
        self._daily_loss_limit_breached = False
        self._trading_day_start = datetime.now(timezone.utc)
        self.logger.info("New trading day started - daily P&L reset")

    def record_trade_pnl(self, realized_pnl: float) -> None:
        """
        Record realized P&L from a completed trade.

        Args:
            realized_pnl: Realized profit/loss from the trade (positive = profit)
        """
        self._daily_realized_pnl += realized_pnl
        self.logger.debug(
            f"Trade P&L recorded: ${realized_pnl:+,.2f}, "
            f"daily total: ${self._daily_realized_pnl:+,.2f}"
        )

        # Check if limit breached
        if self._check_daily_loss_limit_breached():
            self.logger.warning(
                f"DAILY LOSS LIMIT BREACHED: ${abs(self._daily_realized_pnl):,.2f} lost, "
                f"limit ${self.daily_loss_limit:,.2f}"
            )
            # Emit alert
            if self.event_handler:
                asyncio.create_task(
                    self.event_handler.emit(EVENT_ALERT, AlertPayload(
                        level="critical",
                        message=f"Daily loss limit breached! Lost ${abs(self._daily_realized_pnl):,.2f}",
                        symbol=None,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ))
                )

    def _check_daily_loss_limit_breached(self) -> bool:
        """
        Check if daily loss limit has been breached.

        Returns:
            True if limit breached, False otherwise
        """
        if self._daily_loss_limit_breached:
            return True

        if self.daily_loss_limit is None:
            return False

        if self._daily_realized_pnl < -abs(self.daily_loss_limit):
            self._daily_loss_limit_breached = True
            return True

        return False

    def get_daily_pnl(self) -> float:
        """Get current daily realized P&L."""
        return self._daily_realized_pnl

    def get_daily_loss_remaining(self) -> Optional[float]:
        """
        Get remaining loss budget before hitting daily limit.

        Returns:
            Remaining budget in dollars, or None if no limit set
        """
        if self.daily_loss_limit is None:
            return None
        return abs(self.daily_loss_limit) + self._daily_realized_pnl

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
        Handle live trading signal using unified context.

        Orchestrates:
        1. Check halt state (enforced)
        2. Pre-trade validation
        3. Route to appropriate trade logic
        4. Check approval
        5. Size position
        6. Set pending state
        7. Register order
        8. Execute via live broker
        9. Update state on result
        10. Log and update state

        Args:
            context: SignalContext containing all signal data

        Returns:
            OrderResult if executed, None if skipped
        """
        # Generate correlation ID for tracing this operation
        correlation_id = generate_correlation_id()
        set_correlation_id(correlation_id)

        # 1. HALT CHECK (enforced!)
        if self.reconciler and self.reconciler.is_halted:
            self.logger.warning(
                format_log_message(
                    "Trading halted by reconciler - skipping signal",
                    correlation_id=correlation_id,
                    symbol=context.symbol
                )
            )
            return None

        # 2. DAILY LOSS LIMIT CHECK (enforced!)
        if self._check_daily_loss_limit_breached():
            self.logger.warning(
                format_log_message(
                    f"Daily loss limit breached (${abs(self._daily_realized_pnl):,.2f} lost, "
                    f"limit ${self.daily_loss_limit:,.2f}) - trading halted",
                    correlation_id=correlation_id,
                    symbol=context.symbol
                )
            )
            return None

        # Filter low confidence signals
        if context.confidence < self.min_signal_confidence:
            self.logger.debug(
                format_log_message(
                    f"Signal confidence {context.confidence:.2f} below threshold",
                    correlation_id=correlation_id,
                    symbol=context.symbol
                )
            )
            return None

        # 3. DRAWDOWN CHECK (enforced!)
        if self.drawdown_monitor:
            can_trade = await self.drawdown_monitor.can_trade_async(context.symbol)
            if not can_trade:
                self.logger.warning(
                    format_log_message(
                        f"Drawdown limit blocking trade for {context.symbol}",
                        correlation_id=correlation_id,
                        symbol=context.symbol
                    )
                )
                return None
            self.logger.debug(
                format_log_message(
                    f"Drawdown check passed for {context.symbol}",
                    correlation_id=correlation_id,
                    symbol=context.symbol
                )
            )

        # Skip hold signals
        if context.is_hold():
            self.logger.debug(
                format_log_message("HOLD signal - no action", correlation_id=correlation_id, symbol=context.symbol)
            )
            return None

        # Get or create state from metadata or create new
        state = context.metadata.get('state')
        if state is None:
            if context.symbol not in self.symbol_states:
                new_state = SymbolState(symbol=context.symbol)
                self.symbol_states[context.symbol] = new_state
                # Register with state synchronizer
                self._state_sync.register_symbol(context.symbol, new_state)
            state = self.symbol_states[context.symbol]

        # Set strategy name
        state.strategy_name = context.strategy_name

        self.logger.info(
            format_log_message(
                f"Processing signal: {context.signal} @ ${context.price:.2f} "
                f"(regime={context.regime}, strategy={context.strategy_name})",
                correlation_id=correlation_id,
                symbol=context.symbol
            )
        )

        try:
            # Get appropriate trade approver
            trade_approver = self.approver_router.get_approver(
                symbol=context.symbol,
                strategy=context.strategy_name,
                regime=context.regime
            )
            # Alias for backwards compatibility in this method
            trade_logic = trade_approver

            self.logger.debug(
                format_log_message(
                    f"Using approver: {trade_approver.__class__.__name__}",
                    correlation_id=correlation_id,
                    symbol=context.symbol
                )
            )

            # 1. Sync position from broker (live safety check)
            await self._refresh_position(context.symbol)

            # Filter out 'state' from metadata to avoid duplicate argument
            # 2. Check trade approval (pure gating - pass context directly)
            # Approver answers: "Are we ALLOWED to trade?"
            # PositionManager answers: "SHOULD we exit?" (for in-position)
            should_trade, reason = await self._check_trade_approval(
                trade_logic, context, state
            )

            if not should_trade:
                self.logger.info(
                    format_log_message(
                        f"Trade blocked: {reason}",
                        correlation_id=correlation_id,
                        symbol=context.symbol
                    )
                )
                # Emit alert for GUI visibility
                if self.event_handler:
                    signal_text = {-1: "SELL", 0: "HOLD", 1: "BUY"}.get(context.signal, "?")
                    await self.event_handler.emit(EVENT_ALERT, AlertPayload(
                        level="info",
                        message=f"[{context.symbol}] {signal_text} blocked: {reason}",
                        symbol=context.symbol,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ))
                return None

            # 3. Determine action using PositionManager for exit decisions
            in_position = state.side is not None

            if not in_position:
                # Entry - approver already gated it
                action_type = "entry"
                side = OrderSide.BUY if context.signal == 1 else OrderSide.SELL
            else:
                # In position - use PositionManager for exit logic
                # Exit timing (min_bars_to_hold, swing_mode) owned by PositionManager
                should_exit, exit_reason = self.position_manager.check_exit_conditions(
                    state=state,
                    price=context.price,
                    signal=context.signal,
                )

                if not should_exit:
                    # Update trailing stop and excursions even if not exiting
                    self.position_manager.update_trailing_stop(
                        state, context.price, context.atr, context.regime
                    )
                    position = self.portfolio.positions.get(context.symbol)
                    if position:
                        self.position_manager.update_excursions(
                            state, context.price, position.avg_price
                        )
                    state.bars_held = getattr(state, 'bars_held', 0) + 1
                    self.logger.debug(
                        format_log_message(
                            f"Holding position (bars_held={state.bars_held})",
                            correlation_id=correlation_id,
                            symbol=context.symbol
                        )
                    )
                    return None

                # Determine exit type from reason
                reason = exit_reason
                if "partial" in exit_reason.lower():
                    action_type = "partial_exit"
                elif "reversal" in exit_reason.lower():
                    action_type = "reversal"
                else:
                    action_type = "exit"

                side = OrderSide.SELL if state.side == "long" else OrderSide.BUY

            # 4. Calculate quantity
            qty = self._calculate_quantity(
                context.symbol, state, action_type, context.price, context.atr,
                context.regime, trade_logic, signal=context.signal,
                df=context.df, **extra_kwargs
            )

            if qty <= 0:
                self.logger.warning(
                    format_log_message(
                        f"Position size too small: {qty}",
                        correlation_id=correlation_id,
                        symbol=context.symbol
                    )
                )
                return None

            # 5. PRE-TRADE VALIDATION via TradeValidator
            position_state = self.portfolio.get_position_state(context.symbol)
            validation = await self.validator.validate(
                symbol=context.symbol,
                side="buy" if side == OrderSide.BUY else "sell",
                qty=qty,
                price=context.price,
                action_type=action_type,
                position_state=position_state,
            )

            if not validation.valid:
                self.logger.warning(
                    format_log_message(
                        f"Validation failed: {validation.errors}",
                        correlation_id=correlation_id,
                        symbol=context.symbol
                    )
                )
                return None

            if validation.warnings:
                self.logger.info(
                    format_log_message(
                        f"Validation warnings: {validation.warnings}",
                        correlation_id=correlation_id,
                        symbol=context.symbol
                    )
                )

            # Legacy validation (kept for additional safety)
            if not await self._validate_live_execution(context.symbol, side, qty, context.price):
                self.logger.error(
                    format_log_message(
                        "Pre-execution validation failed",
                        correlation_id=correlation_id,
                        symbol=context.symbol
                    )
                )
                return None

            # 6. SET PENDING STATE
            pending_state = (
                PositionState.PENDING_ENTRY if action_type == "entry"
                else PositionState.PENDING_EXIT if action_type in ("exit", "reversal")
                else PositionState.PENDING_ADD
            )
            await self.portfolio.set_position_state(context.symbol, pending_state)
            state.position_state = pending_state

            self.logger.info(
                format_log_message(
                    f"Action approved: {action_type} {side.value} {qty} @ ${context.price:.2f}",
                    correlation_id=correlation_id,
                    symbol=context.symbol,
                    state=pending_state.value
                )
            )

            # 7. Execute trade (LIVE) - order registration happens inside
            result = await self._execute_live_trade(
                context.symbol, state, side, qty, context.price, context.atr, action_type,
                correlation_id=correlation_id,
                order_type=context.order_type,
                time_in_force=context.time_in_force,
                limit_price=context.limit_price,
                stop_price=context.stop_price,
                df=context.df, **extra_kwargs
            )

            # 8. UPDATE STATE ON RESULT
            if result:
                # Order filled - update to OPEN or NONE
                new_state = PositionState.OPEN if action_type == "entry" else PositionState.NONE
                await self.portfolio.set_position_state(context.symbol, new_state)
                state.position_state = new_state

                # 9. For entries, calculate SL/TP levels via PositionManager
                if action_type == "entry":
                    self.position_manager.calculate_levels(
                        state=state,
                        price=result.avg_price,
                        atr=context.atr,
                        condition=context.regime,
                        side=side
                    )
                    self.logger.debug(
                        format_log_message(
                            f"Entry levels set: SL=${state.stop_loss:.2f}, TP=${state.take_profit:.2f}",
                            correlation_id=correlation_id,
                            symbol=context.symbol
                        )
                    )

                # 10. Post-execution tasks
                self._post_execution(
                    context.symbol, state, result, action_type, context.regime, context.strategy_name
                )

                self.logger.info(
                    format_log_message(
                        f"Execution complete: {side.value} {qty} filled",
                        correlation_id=correlation_id,
                        symbol=context.symbol,
                        state=new_state.value
                    )
                )
            else:
                # Order failed - revert state
                await self.portfolio.set_position_state(
                    context.symbol,
                    PositionState.OPEN if action_type in ("exit", "partial_exit") else PositionState.NONE
                )
                state.position_state = PositionState.NONE

            return result

        except Exception as e:
            self.logger.exception(
                format_log_message(
                    f"Error in live execution: {e}",
                    correlation_id=correlation_id,
                    symbol=context.symbol
                )
            )

            # Revert position state on error
            try:
                await self.portfolio.set_position_state(context.symbol, PositionState.NONE)
            except Exception:
                pass

            # Alert on live execution errors
            self.performance_tracker.log_error(
                message=f"Live execution error for {context.symbol}",
                error=e,
                context={
                    'symbol': context.symbol,
                    'signal': context.signal,
                    'price': context.price,
                    'strategy': context.strategy_name,
                    'correlation_id': correlation_id,
                }
            )

            return None

    async def handle_signal(
        self,
        context: SignalContext,
        state: SymbolState,
    ) -> Optional[OrderResult]:
        """
        Handle live trading signal.

        This is the primary entry point. Takes unified SignalContext
        and explicit SymbolState.

        Args:
            context: SignalContext containing all signal data
            state: SymbolState for this symbol

        Returns:
            OrderResult if executed, None if skipped
        """
        # Store state in metadata so handle_signal_context can use it
        context.metadata['state'] = state
        return await self.handle_signal_context(context)

    async def handle_signal_legacy(
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
        DEPRECATED: Use handle_signal(context, state) instead.

        Backward-compatible wrapper that creates SignalContext from loose params.
        """
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
            **{k: v for k, v in kwargs.items() if k not in ('df', 'market_open')}
        )
        return await self.handle_signal(context, state)
    
    # ========================================================================
    # LIVE-SPECIFIC METHODS
    # ========================================================================
    
    async def _has_pending_orders(self, symbol: str) -> bool:
        """
        Check if there are any pending orders for a symbol.

        Uses local OrderRegistry first (fast), falls back to broker query.

        Args:
            symbol: Trading symbol to check

        Returns:
            True if there are open/pending orders for this symbol
        """
        # First check local registry (fast, no network call)
        if await self.order_registry.has_pending_orders(symbol):
            self.logger.debug(f"[{symbol}] Found pending order in local registry")
            return True

        # Fall back to broker query for orders we might have missed
        try:
            open_orders = await self.broker.get_open_orders()
            for order in open_orders:
                if order.symbol == symbol:
                    self.logger.debug(
                        f"[{symbol}] Found pending order on broker: {order.order_id} "
                        f"({order.side} {order.qty} @ status={order.status})"
                    )
                    return True
            return False
        except Exception as e:
            self.logger.warning(f"[{symbol}] Failed to check pending orders: {e}")
            # Conservative: assume pending orders exist on error
            return True

    async def _refresh_position(self, symbol: str) -> None:
        """
        Refresh position from broker before executing.

        Critical for live trading to avoid stale state.
        Uses local OrderRegistry for fast pending order checks.
        """
        try:
            broker_position = await self.broker.get_position(symbol)

            if broker_position:
                # Convert PositionView to SymbolPosition for portfolio
                from core.logic.portfolio_state import SymbolPosition
                self.portfolio.positions[symbol] = SymbolPosition(
                    qty=int(broker_position.qty),
                    avg_price=float(broker_position.avg_price),
                    last_price=float(broker_position.last_price or broker_position.avg_price),
                )
                self.logger.debug(
                    f"[{symbol}] Position refreshed: qty={broker_position.qty}"
                )
            elif symbol in self.portfolio.positions:
                # Position not on broker - use local registry for fast check
                if await self.order_registry.has_pending_orders(symbol):
                    self.logger.debug(
                        f"[{symbol}] Has pending orders in registry, keeping local state"
                    )
                    return

                # Double-check with broker (in case registry missed something)
                if await self._has_pending_orders(symbol):
                    self.logger.debug(
                        f"[{symbol}] Has pending orders on broker, keeping local state"
                    )
                    return

                # No pending orders, safe to remove position
                del self.portfolio.positions[symbol]
                await self.portfolio.clear_position_state(symbol)
                self.logger.warning(
                    f"[{symbol}] Position removed (not found on broker)"
                )

        except Exception as e:
            self.logger.warning(f"[{symbol}] Failed to refresh position: {e}")
    
    async def _cancel_conflicting_orders(self, symbol: str, side: OrderSide) -> None:
        """
        Cancel any open orders that would conflict with a new order.

        Uses local OrderRegistry for fast lookup, then cancels via broker.
        Alpaca rejects orders that would create a "wash trade" (opposite-side
        order while another is pending).

        Args:
            symbol: Trading symbol
            side: Side of the new order (buy/sell)
        """
        new_side = "buy" if side == OrderSide.BUY else "sell"

        try:
            # Use local registry for fast lookup
            conflicting = await self.order_registry.get_conflicting_orders(symbol, new_side)

            if conflicting:
                self.logger.info(
                    f"[{symbol}] Found {len(conflicting)} conflicting orders in registry"
                )

                for order in conflicting:
                    try:
                        await self.broker.cancel_order(order.order_id)
                        await self.order_registry.update_status(order.order_id, "cancelled")
                        self.logger.info(
                            f"[{symbol}] Cancelled conflicting order {order.order_id}"
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"[{symbol}] Failed to cancel order {order.order_id}: {e}"
                        )

            # Also check broker for orders not in our registry
            open_orders = await self.broker.get_open_orders()

            for order in open_orders:
                if order.symbol != symbol:
                    continue

                # Skip if already in registry (handled above)
                if await self.order_registry.get(order.order_id):
                    continue

                order_side = order.side.lower() if order.side else ""
                if order_side != new_side:
                    try:
                        await self.broker.cancel_order(order.order_id)
                        self.logger.info(
                            f"[{symbol}] Cancelled conflicting broker order {order.order_id}"
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"[{symbol}] Failed to cancel order {order.order_id}: {e}"
                        )

        except Exception as e:
            self.logger.warning(f"[{symbol}] Failed to check/cancel conflicting orders: {e}")

    async def _validate_live_execution(
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
            if not await self.broker.is_market_open():
                self.logger.warning("Market is closed - order may not fill immediately")
                # Continue anyway - some brokers accept orders when closed

            # Check buying power
            if hasattr(self.broker, 'get_buying_power'):
                available = self.broker.get_buying_power()
            else:
                available = self.broker.get_available_funds()

            required = qty * price

            if side == OrderSide.BUY and required > available:
                self.logger.error(
                    f"[{symbol}] Insufficient buying power: need ${required:,.2f}, "
                    f"have ${available:,.2f} (qty={qty} @ ${price:.2f})"
                )
                return False

            # Check quantity limits
            if qty <= 0:
                self.logger.error(f"[{symbol}] Invalid quantity: {qty}")
                return False

            # Check if symbol is tradeable (if broker supports)
            if hasattr(self.broker, 'is_tradeable'):
                if not self.broker.is_tradeable(symbol):
                    self.logger.error(f"Symbol not tradeable: {symbol}")
                    return False

            self.logger.debug(
                f"[{symbol}] Validation passed: {qty} shares @ ${price:.2f} = "
                f"${required:,.2f} (buying_power=${available:,.2f})"
            )
            return True

        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False
    
    async def _execute_live_trade(
        self,
        symbol: str,
        state: SymbolState,
        side: OrderSide,
        qty: int,
        price: float,
        atr: float,
        action_type: str,
        correlation_id: Optional[str] = None,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        **kwargs
    ) -> Optional[OrderResult]:
        """
        Execute LIVE trade via broker.

        This places actual orders in the market!
        All logic (sizing, validation) already done upstream.

        Args:
            symbol: Trading symbol
            state: Symbol state
            side: Order side (BUY/SELL)
            qty: Quantity to trade
            price: Expected price (for position sizing reference)
            atr: ATR value
            action_type: Type of action (entry, exit, etc.)
            correlation_id: Correlation ID for tracing
            order_type: Order type (market, limit, stop, stop_limit)
            time_in_force: Time in force (day, gtc, ioc, fok)
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders

        After order placement:
        1. Registers order in local OrderRegistry
        2. Applies optimistic update to portfolio
        The reconciler will correct if the actual fill differs.
        """
        correlation_id = correlation_id or generate_correlation_id()
        side_str = "buy" if side == OrderSide.BUY else "sell"

        try:
            self.logger.info(
                format_log_message(
                    f"Placing {order_type} {side.value} order: {qty} shares @ ~${price:.2f} "
                    f"(TIF={time_in_force})",
                    correlation_id=correlation_id,
                    symbol=symbol
                )
            )

            # Cancel any conflicting orders before placing new one
            # This prevents wash trade rejections from Alpaca
            await self._cancel_conflicting_orders(symbol, side)

            # Build order kwargs based on order type
            order_kwargs = {
                "symbol": symbol,
                "qty": qty,
                "side": side_str,
                "order_type": order_type,
                "time_in_force": time_in_force,
            }

            # Add price parameters based on order type
            if order_type in ("limit", "stop_limit") and limit_price is not None:
                order_kwargs["limit_price"] = limit_price
            if order_type in ("stop", "stop_limit") and stop_price is not None:
                order_kwargs["stop_price"] = stop_price

            # Place order directly via broker (qty already calculated)
            order_response = await self.broker.place_order(**order_kwargs)

            # Extract order ID from response if available
            order_id = (
                getattr(order_response, 'order_id', None) or
                getattr(order_response, 'id', None) or
                f"LIVE_{symbol}_{datetime.now(timezone.utc).timestamp()}"
            )

            # Register order in local registry for tracking
            tracked_order = await self.order_registry.register(
                order_id=str(order_id),
                symbol=symbol,
                side=side_str,
                qty=qty,
                correlation_id=correlation_id,
                status="pending"
            )

            # Update symbol state with pending order
            state.pending_order_id = str(order_id)

            self.logger.debug(
                format_log_message(
                    f"Order registered in local registry: {order_id}",
                    correlation_id=correlation_id,
                    symbol=symbol
                )
            )

            # Create result from expected values
            # Note: Actual fill price/qty will be tracked via order events
            result = OrderResult(
                order_id=str(order_id),
                symbol=symbol,
                side=side,
                filled_qty=qty,
                avg_price=price,
            )

            # Optimistic portfolio update with state sync
            # Apply expected fill immediately so UI/logic sees updated state
            # The reconciler will correct if actual fill differs
            # Use StateSynchronizer to keep Portfolio and Symbol states in sync
            await self._state_sync.apply_fill_and_sync(symbol, side_str, qty, price)
            self.portfolio.mark_updated()

            # Update order status to filled (optimistic)
            await self.order_registry.update_status(str(order_id), "filled", qty)

            self.logger.info(
                format_log_message(
                    f"Order submitted and optimistic fill applied: {side.value} {qty}@${price:.2f}",
                    correlation_id=correlation_id,
                    symbol=symbol
                )
            )

            # Emit trade event for GUI
            if self.event_handler:
                trade_payload: TradePayload = {
                    "symbol": symbol,
                    "side": side_str,
                    "qty": qty,
                    "price": price,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "pnl": None,  # P&L calculated separately
                }
                await self.event_handler.emit(EVENT_NEW_TRADE, trade_payload)

            return result

        except Exception as e:
            self.logger.exception(
                format_log_message(
                    f"Execution failed: {e}",
                    correlation_id=correlation_id,
                    symbol=symbol
                )
            )

            # Log to performance tracker
            self.performance_tracker.log_error(
                message=f"Order execution failed for {symbol}",
                error=e,
                context={
                    'symbol': symbol,
                    'side': side.value,
                    'qty': qty,
                    'price': price,
                    'correlation_id': correlation_id,
                }
            )

            return None
    
    # ========================================================================
    # DELEGATION TO GENERIC ENGINE LOGIC
    # ========================================================================
    
    async def _check_trade_approval(
        self,
        trade_logic,
        context: SignalContext,
        state: SymbolState,
    ):
        """Check trade approval (pass context directly)."""
        # Sync state with portfolio
        self._setup_approval_state(context.symbol, state)

        # Pass context directly to approver
        return trade_logic.should_trade(
            context=context,
            state=state,
            account_positions=len(self.portfolio.positions)
        )

    def _calculate_quantity(self, symbol, state, action_type, price, atr, regime, trade_logic, **kwargs):
        """Calculate position size for entries, use PositionManager for exits."""
        # Handle exits via PositionManager
        if action_type in ("exit", "reversal", "partial_exit"):
            position = self.portfolio.positions.get(symbol)
            if not position:
                return 0
            is_partial = action_type == "partial_exit"
            return self.position_manager.get_exit_quantity(position.qty, is_partial=is_partial)

        # Get signal from kwargs (passed from handle_signal)
        signal = kwargs.get('signal', 1)

        sl_mults = getattr(trade_logic, 'sl_mults', {"normal": 1.5})
        sl_mult = sl_mults.get(regime, 1.5)

        # ATR sanity check: if ATR is 0 or suspiciously small (< 0.5% of price),
        # it's likely from minute bars or missing data. Use a floor.
        min_atr = price * 0.005  # 0.5% of price
        if atr <= 0 or atr < min_atr:
            old_atr = atr
            atr = max(min_atr, price * 0.01)  # Use 1% of price as default
            self.logger.warning(
                f"[{symbol}] ATR too small (${old_atr:.2f}), using ${atr:.2f} "
                f"(1% of ${price:.2f})"
            )

        # Calculate stop loss based on signal direction (not current position)
        # Long (signal > 0): stop below price
        # Short (signal < 0): stop above price
        if signal > 0:
            stop_loss_price = price - (atr * sl_mult)
        else:
            stop_loss_price = price + (atr * sl_mult)

        # Get actual buying power from broker
        buying_power = self.portfolio.cash  # Default fallback
        try:
            if hasattr(self.broker, 'get_buying_power'):
                buying_power = self.broker.get_buying_power()
                self.logger.debug(f"[{symbol}] Broker buying power: ${buying_power:,.2f}")
            else:
                buying_power = self.broker.get_available_funds()
        except Exception as e:
            self.logger.warning(f"[{symbol}] Failed to get buying power: {e}, using cash")
            buying_power = self.portfolio.cash

        # Safety: If buying power is 0 or very low, skip sizing
        if buying_power < 100:
            self.logger.warning(f"[{symbol}] Insufficient buying power: ${buying_power:.2f}")
            return 0

        return self.sizer.calculate_position_size(
            symbol=symbol,
            price=price,
            account_value=self.portfolio.total_value,
            signal_strength=1.0,
            atr=atr,
            stop_loss_price=stop_loss_price,
            signal=signal,
            portfolio=self.portfolio,
            market_conditions=regime,
            current_cash=buying_power  # Use actual buying power, not cash!
        )
    
    def _update_portfolio_after_execution(self, symbol: str, result) -> None:
        """
        Skip portfolio update - already handled in _execute_live_trade().

        LiveExecutionEngine applies optimistic updates immediately after
        order placement, so we skip the base class update to avoid
        double-counting.
        """
        pass  # Already done in _execute_live_trade()

    def _post_execution(self, symbol, state, result, action_type, regime, strategy_name):
        """Post-execution logging and state updates with Live-specific logging."""
        # Use base class implementation for common logic
        super()._post_execution(symbol, state, result, action_type, regime, strategy_name)

        # Record realized P&L for daily loss tracking (on exits/closes)
        if action_type in ("exit", "partial_exit", "reversal"):
            # Get realized P&L from the trade
            position = self.portfolio.positions.get(symbol)
            if position:
                realized_pnl = position.realized_pnl
                if realized_pnl != 0:
                    self.record_trade_pnl(realized_pnl)
                    # Reset position's realized_pnl after recording
                    position.realized_pnl = 0.0

        # Update drawdown monitor with new portfolio equity after trade
        if self.drawdown_monitor:
            equity = self.portfolio.total_equity()
            # Use sync version since _post_execution is synchronous
            self.drawdown_monitor.update_portfolio(equity)
            self.logger.debug(
                f"[{symbol}] Drawdown monitor updated: equity=${equity:,.2f}"
            )

        # Live-specific logging
        self.logger.info(
            f"[LIVE] [{symbol}] Trade logged: {action_type} "
            f"{result.side.value} {result.filled_qty}@${result.avg_price:.2f}"
        )
    

    # ========================================================================
    # GUI EVENT HANDLERS
    # ========================================================================

    async def subscribe_to_gui_events(self) -> None:
        """Subscribe to GUI command events (manual orders, flatten, cancel)."""
        if not self.event_handler:
            self.logger.warning("No event handler - GUI events disabled")
            return

        await self.event_handler.subscribe(EVENT_MANUAL_ORDER, self._handle_manual_order)
        await self.event_handler.subscribe(EVENT_FLATTEN_ALL, self._handle_flatten_all)
        await self.event_handler.subscribe(EVENT_FLATTEN_SYMBOL, self._handle_flatten_symbol)
        await self.event_handler.subscribe(EVENT_CANCEL_ALL, self._handle_cancel_all)

        self.logger.info("Subscribed to GUI events")

    async def _handle_manual_order(self, event) -> None:
        """Handle manual order from GUI."""
        payload = event.payload
        symbol = payload["symbol"]
        qty = int(payload["qty"])
        side_str = payload["side"].upper()
        price = float(payload.get("price", 0.0))
        order_type = payload.get("type", "market").lower()

        self.logger.info(f"[MANUAL ORDER] {side_str} {qty} {symbol} ({order_type})")

        try:
            side = OrderSide.BUY if side_str == "BUY" else OrderSide.SELL

            if order_type == "limit":
                await self.broker.place_order(
                    symbol=symbol,
                    qty=qty,
                    side=side_str.lower(),
                    order_type="limit",
                    limit_price=price,
                    time_in_force=payload.get("tif", "day")
                )
            else:
                await self.broker.place_market_order(
                    symbol=symbol,
                    qty=qty,
                    side=side_str.lower()
                )

            # Update portfolio optimistically
            await self._state_sync.apply_fill_and_sync(symbol, side_str.lower(), qty, price)
            self.portfolio.mark_updated()

            # Emit trade event
            if self.event_handler:
                trade_payload: TradePayload = {
                    "symbol": symbol,
                    "side": side_str.lower(),
                    "qty": qty,
                    "price": price,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "pnl": None,
                }
                await self.event_handler.emit(EVENT_NEW_TRADE, trade_payload)

        except Exception as e:
            self.logger.error(f"Manual order failed: {e}")
            if self.event_handler:
                await self.event_handler.emit(EVENT_ALERT, AlertPayload(
                    level="error",
                    message=f"Manual order failed for {symbol}: {str(e)}",
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))

    async def _handle_flatten_all(self, event) -> None:
        """Flatten all positions."""
        self.logger.info("[GUI] Flatten ALL positions")

        for symbol, position in list(self.portfolio.positions.items()):
            if position.qty != 0:
                try:
                    qty = abs(position.qty)
                    side = "sell" if position.qty > 0 else "buy"
                    await self.broker.place_market_order(
                        symbol=symbol,
                        qty=qty,
                        side=side
                    )
                    self.logger.info(f"Flattened {symbol}: {side} {qty}")
                except Exception as e:
                    self.logger.error(f"Failed to flatten {symbol}: {e}")
                    if self.event_handler:
                        await self.event_handler.emit(EVENT_ALERT, AlertPayload(
                            level="error",
                            message=f"Flatten failed for {symbol}: {str(e)}",
                            symbol=symbol,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                        ))

    async def _handle_flatten_symbol(self, event) -> None:
        """Flatten specific symbol."""
        symbol = event.payload["symbol"]
        position = self.portfolio.positions.get(symbol)

        if not position or position.qty == 0:
            self.logger.info(f"[GUI] No position to flatten for {symbol}")
            return

        qty = abs(position.qty)
        side = "sell" if position.qty > 0 else "buy"

        self.logger.info(f"[GUI] Flatten {symbol} ({side} {qty})")

        try:
            await self.broker.place_market_order(
                symbol=symbol,
                qty=qty,
                side=side
            )
            self.logger.info(f"Flattened {symbol}: {side} {qty}")
        except Exception as e:
            self.logger.error(f"Failed to flatten {symbol}: {e}")
            if self.event_handler:
                await self.event_handler.emit(EVENT_ALERT, AlertPayload(
                    level="error",
                    message=f"Flatten failed for {symbol}: {str(e)}",
                    symbol=symbol,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))

    async def _handle_cancel_all(self, event) -> None:
        """Cancel all open orders."""
        self.logger.info("[GUI] Cancel all open orders")

        try:
            open_orders = await self.broker.get_open_orders()

            for order in open_orders:
                order_id = getattr(order, 'order_id', None) or getattr(order, 'id', None)
                if order_id:
                    try:
                        await self.broker.cancel_order(str(order_id))
                        await self.order_registry.update_status(str(order_id), "cancelled")
                        self.logger.info(f"Cancelled order: {order_id}")
                    except Exception as e:
                        self.logger.warning(f"Failed to cancel order {order_id}: {e}")

        except Exception as e:
            self.logger.error(f"Failed to cancel all orders: {e}")
            if self.event_handler:
                await self.event_handler.emit(EVENT_ALERT, AlertPayload(
                    level="error",
                    message=f"Cancel all failed: {str(e)}",
                    symbol=None,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))