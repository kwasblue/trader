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
from datetime import datetime, timezone, time
from zoneinfo import ZoneInfo
import logging

# US Eastern timezone for market hours calculations
ET = ZoneInfo("America/New_York")

from core.base.execution_engine_base import ExecutionEngineBase
from core.base.executor_base import BaseExecutor
from core.base.base_broker_interface import BaseBrokerInterface
from core.base.position_sizer_base import PositionSizerBase
from core.base.trade_logger_base import TradeLoggerBase
from core.base.trade_logic_manager_base import TradeLogicManagerBase
from core.logic.portfolio_state import PortfolioState
from core.logic.symbol_state import SymbolState
from core.logic.position_manager import PositionManager
from core.logic.hybrid_position_sizer import HybridPositionSizer
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
from core.contracts.meta_types import TradeEntryContext, TradeExitContext
from loggers.meta_trade_logger import MetaTradeLogger, generate_trade_id

# Import the router we created
from core.logic.trade_logic_router import TradeApproverRouter
from core.tracing import trace

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

        # Position refresh cache to reduce broker API calls
        self._position_cache: Dict[str, datetime] = {}  # symbol -> last_refresh_time
        self._position_cache_ttl: float = 30.0  # Cache TTL in seconds (increased for scaling)

        # Meta trade logger for ML training data
        # Load config for meta logging settings
        try:
            from core.config_loader import get_config
            cfg = get_config()
            ml_config = getattr(cfg, 'ml_training', None)
            if ml_config and getattr(ml_config, 'meta_logging_enabled', True):
                meta_log_file = getattr(ml_config, 'meta_log_file', 'meta_trades_live.jsonl')
                self.meta_logger = MetaTradeLogger(log_file=meta_log_file)
            else:
                self.meta_logger = MetaTradeLogger(log_file='meta_trades_live.jsonl')
        except Exception:
            self.meta_logger = MetaTradeLogger(log_file='meta_trades_live.jsonl')

        # Hybrid position sizer - set by BaseLiveRunner if enabled
        # Uses confidence + trend alignment to adjust position sizes
        self.hybrid_sizer: Optional[HybridPositionSizer] = None

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
        Uses sync_from_snapshot() to maintain proper state ownership.
        """
        try:
            self.logger.info("Syncing portfolio with broker...")

            # Get account info from broker
            account = await self.broker.get_account_info()

            # Use sync_from_snapshot() - the canonical way to sync state
            # This respects STATE_OWNERSHIP_MATRIX boundaries
            self.portfolio.sync_from_snapshot(account)

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

    @trace
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

        # Track bar for bar-based cooldown (call on every signal = every bar)
        if hasattr(self, 'trade_logic_manager') and self.trade_logic_manager:
            trade_logic = self.trade_logic_manager.get(context.symbol, context.regime)
            if hasattr(trade_logic, 'on_bar'):
                trade_logic.on_bar(context.symbol)

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
                exit_reason_lower = exit_reason.lower()
                if "partial" in exit_reason_lower:
                    action_type = "partial_exit"
                elif "reversal" in exit_reason_lower:
                    action_type = "reversal"
                elif "stop" in exit_reason_lower or "stop loss" in exit_reason_lower:
                    action_type = "stop_loss"
                elif "take profit" in exit_reason_lower or "profit" in exit_reason_lower:
                    action_type = "take_profit"
                else:
                    action_type = "exit"

                side = OrderSide.SELL if state.side == "long" else OrderSide.BUY

            # 4. Calculate quantity (pass context for hybrid sizing)
            qty = self._calculate_quantity(
                context.symbol, state, action_type, context.price, context.atr,
                context.regime, trade_logic, signal=context.signal,
                df=context.df, context=context
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
            state_set = await self.portfolio.set_position_state(context.symbol, pending_state)
            if not state_set:
                # Invalid state transition - likely a pending order already exists
                current_state = self.portfolio.get_position_state(context.symbol)
                self.logger.warning(
                    format_log_message(
                        f"Cannot set {pending_state.value}: current state is {current_state.value}",
                        correlation_id=correlation_id,
                        symbol=context.symbol
                    )
                )
                return None
            state.position_state = pending_state

            self.logger.info(
                format_log_message(
                    f"Action approved: {action_type} {side.value} {qty} @ ${context.price:.2f}",
                    correlation_id=correlation_id,
                    symbol=context.symbol,
                    state=pending_state.value
                )
            )

            # 6.5 CAPTURE PRE-TRADE STATE (before portfolio is updated)
            # This is critical for accurate trade logging - portfolio will be
            # updated inside _execute_live_trade before _post_execution runs
            position_before = self.portfolio.positions.get(context.symbol)
            pre_state = {
                'cash': self.portfolio.cash,
                'position_qty': position_before.qty if position_before else 0,
                'avg_price': position_before.avg_price if position_before else None,
            }

            # 7. Execute trade (LIVE) - order registration happens inside
            result = await self._execute_live_trade(
                context.symbol, state, side, qty, context.price, context.atr, action_type,
                correlation_id=correlation_id,
                order_type=context.order_type,
                time_in_force=context.time_in_force,
                limit_price=context.limit_price,
                stop_price=context.stop_price,
                df=context.df
            )

            # 8. UPDATE STATE ON RESULT
            if result:
                # Order filled - update to OPEN or NONE
                new_state = PositionState.OPEN if action_type == "entry" else PositionState.NONE
                await self.portfolio.set_position_state(context.symbol, new_state)
                state.position_state = new_state

                # 9. For entries, calculate SL/TP levels via PositionManager
                if action_type == "entry":
                    # Track entry regime for strategy switching logic
                    state.entry_regime = context.regime

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

                    # Log entry for meta-model training
                    self._log_meta_entry(context, state, result, qty)

                # 10. Post-execution tasks (pass pre_state for accurate logging)
                self._post_execution(
                    context.symbol, state, result, action_type, context.regime, context.strategy_name,
                    pre_state=pre_state
                )

                # 11. Reset bar-based cooldown after trade
                if hasattr(trade_logic, 'on_trade'):
                    trade_logic.on_trade(context.symbol)

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

    async def _refresh_position(self, symbol: str, force: bool = False) -> None:
        """
        Refresh position from broker before executing.

        Uses caching to reduce broker API calls. Positions are cached for
        `_position_cache_ttl` seconds (default 5s).

        Args:
            symbol: Trading symbol
            force: If True, bypass cache and always refresh from broker
        """
        # Check cache unless forced refresh
        if not force:
            last_refresh = self._position_cache.get(symbol)
            if last_refresh:
                age = (datetime.now(timezone.utc) - last_refresh).total_seconds()
                if age < self._position_cache_ttl:
                    self.logger.debug(
                        f"[{symbol}] Using cached position (age={age:.1f}s)"
                    )
                    return

        try:
            broker_position = await self.broker.get_position(symbol)

            # Update cache timestamp
            self._position_cache[symbol] = datetime.now(timezone.utc)

            if broker_position:
                # Use proper method to sync position - maintains state ownership
                self.portfolio.sync_position_from_broker(
                    symbol=symbol,
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

                # No pending orders, safe to remove position via proper method
                self.portfolio.remove_position(symbol)
                self.logger.warning(
                    f"[{symbol}] Position removed (not found on broker)"
                )

        except Exception as e:
            self.logger.warning(f"[{symbol}] Failed to refresh position: {e}")

    def invalidate_position_cache(self, symbol: Optional[str] = None) -> None:
        """
        Invalidate position cache.

        Call this after trades complete to ensure fresh data on next check.

        Args:
            symbol: Symbol to invalidate, or None to clear all
        """
        if symbol:
            self._position_cache.pop(symbol, None)
        else:
            self._position_cache.clear()
    
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

    @trace
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

            # Safety check for exit orders: verify qty against broker's actual position
            # This prevents "insufficient qty" errors when local state is out of sync
            if action_type in ("exit", "reversal", "partial_exit") and side_str == "sell":
                try:
                    broker_positions = await self.broker.get_positions()
                    broker_pos = next((p for p in broker_positions if p.symbol == symbol), None)
                    broker_qty = int(broker_pos.qty) if broker_pos else 0
                    if qty > broker_qty:
                        self.logger.warning(
                            format_log_message(
                                f"Exit qty {qty} exceeds broker position {broker_qty}, capping to {broker_qty}",
                                correlation_id=correlation_id,
                                symbol=symbol
                            )
                        )
                        qty = broker_qty
                        if qty <= 0:
                            self.logger.warning(
                                format_log_message(
                                    "No position at broker to exit, skipping order",
                                    correlation_id=correlation_id,
                                    symbol=symbol
                                )
                            )
                            return None
                except Exception as e:
                    self.logger.warning(
                        format_log_message(
                            f"Could not verify broker position: {e}, proceeding with local qty",
                            correlation_id=correlation_id,
                            symbol=symbol
                        )
                    )

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

            # Verify order fill before applying to portfolio
            # This prevents state drift from rejected/partial/cancelled orders
            actual_filled_qty = 0
            actual_fill_price = price  # Default to expected price

            if self.reconciler:
                verified = await self.reconciler.verify_order(
                    order_id=str(order_id),
                    symbol=symbol,
                    expected_qty=qty,
                    expected_side=side_str,
                    correlation_id=correlation_id,
                )

                # Always get actual fill from broker, regardless of verification result
                # This handles partial fills correctly
                try:
                    order_status = await self.broker.get_order_status(str(order_id))
                    actual_filled_qty = int(order_status.filled_qty or 0)
                    actual_fill_price = float(order_status.avg_fill_price or price)
                    order_final_status = (order_status.status or "").lower()
                except Exception as e:
                    self.logger.warning(
                        format_log_message(
                            f"Could not get fill details: {e}",
                            correlation_id=correlation_id,
                            symbol=symbol
                        )
                    )
                    if verified:
                        actual_filled_qty = qty  # Assume full fill if verified
                    order_final_status = "unknown"

                if actual_filled_qty == 0:
                    self.logger.warning(
                        format_log_message(
                            f"Order not filled (status={order_final_status}) - not applying to portfolio",
                            correlation_id=correlation_id,
                            symbol=symbol
                        )
                    )
                    await self.order_registry.update_status(str(order_id), "rejected", 0)
                    return None

                if actual_filled_qty < qty:
                    self.logger.warning(
                        format_log_message(
                            f"Partial fill: {actual_filled_qty}/{qty} shares",
                            correlation_id=correlation_id,
                            symbol=symbol
                        )
                    )
            else:
                # No reconciler - fall back to optimistic update (less safe)
                self.logger.warning(
                    format_log_message(
                        "No reconciler available - using optimistic fill (unsafe)",
                        correlation_id=correlation_id,
                        symbol=symbol
                    )
                )
                actual_filled_qty = qty

            # Create result with actual fill values
            result = OrderResult(
                order_id=str(order_id),
                symbol=symbol,
                side=side,
                filled_qty=actual_filled_qty,
                avg_price=actual_fill_price,
                status="filled",
            )

            # Apply verified fill to portfolio
            await self._state_sync.apply_fill_and_sync(symbol, side_str, actual_filled_qty, actual_fill_price)
            self.portfolio.mark_updated()

            # Update order status with actual fill
            await self.order_registry.update_status(str(order_id), "filled", actual_filled_qty)

            # Invalidate position cache to ensure fresh data on next refresh
            self.invalidate_position_cache(symbol)

            self.logger.info(
                format_log_message(
                    f"Order verified and fill applied: {side.value} {actual_filled_qty}@${actual_fill_price:.2f}",
                    correlation_id=correlation_id,
                    symbol=symbol
                )
            )

            # Emit trade event for GUI
            if self.event_handler:
                trade_payload: TradePayload = {
                    "symbol": symbol,
                    "side": side_str,
                    "qty": actual_filled_qty,
                    "price": actual_fill_price,
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
        """Async wrapper for base class _check_trade_approval."""
        # Delegate to synchronous base implementation
        return super()._check_trade_approval(trade_logic, context, state)

    def _calculate_quantity(self, symbol, state, action_type, price, atr, regime, trade_logic, **kwargs):
        """
        Calculate position size for entries, use PositionManager for exits.

        For entries, applies hybrid sizing if enabled:
        - Gets base quantity from standard sizer
        - Applies confidence + trend multiplier from hybrid sizer
        - Can block trades (return 0) if low confidence against trend
        """
        # Handle exits via PositionManager
        if action_type in ("exit", "reversal", "partial_exit"):
            position = self.portfolio.positions.get(symbol)
            if not position:
                return 0
            is_partial = action_type == "partial_exit"
            return self.position_manager.get_exit_quantity(position.qty, is_partial=is_partial)

        # Get signal and context from kwargs
        signal = kwargs.get('signal', 1)
        context = kwargs.get('context')

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

        # Calculate base position size
        base_qty = self.sizer.calculate_position_size(
            symbol=symbol,
            price=price,
            account_value=self.portfolio.total_value,
            signal_strength=1.0,
            atr=atr,
            stop_loss_price=stop_loss_price,
            signal=signal,
            portfolio=self.portfolio,
            market_conditions=regime,
            current_cash=buying_power
        )

        # Apply hybrid sizing if enabled
        if self.hybrid_sizer and self.hybrid_sizer.enabled:
            # Get daily context and confidence from context metadata
            daily_ctx = {}
            confidence = 1.0  # Default to full confidence if not provided

            if context:
                daily_ctx = context.metadata.get('daily_context', {})
                confidence = getattr(context, 'confidence', 1.0)

            # Calculate hybrid sizing result
            sizing_result = self.hybrid_sizer.calculate(
                signal=signal,
                confidence=confidence,
                daily_context=daily_ctx,
            )

            # Apply multiplier
            adjusted_qty = int(base_qty * sizing_result.base_multiplier)

            # Log sizing decision
            if sizing_result.base_multiplier == 0:
                self.logger.info(
                    f"[{symbol}] Trade BLOCKED by hybrid sizer: {sizing_result.reason}"
                )
                return 0
            elif sizing_result.base_multiplier < 1.0:
                self.logger.info(
                    f"[{symbol}] Position reduced to {sizing_result.base_multiplier:.0%}: "
                    f"{sizing_result.reason} (base_qty={base_qty} -> {adjusted_qty})"
                )
            else:
                self.logger.debug(
                    f"[{symbol}] Hybrid sizing: {sizing_result.reason}"
                )

            return adjusted_qty

        return base_qty
    
    def _update_portfolio_after_execution(self, symbol: str, result) -> None:
        """
        Skip portfolio update - already handled in _execute_live_trade().

        LiveExecutionEngine applies optimistic updates immediately after
        order placement, so we skip the base class update to avoid
        double-counting.
        """
        pass  # Already done in _execute_live_trade()

    # ========================================================================
    # META TRADE LOGGING
    # ========================================================================

    def _log_meta_entry(
        self,
        context: SignalContext,
        state: SymbolState,
        result: OrderResult,
        qty: int,
    ) -> None:
        """
        Log trade entry for meta-model training.

        Captures all relevant features at entry time.

        Args:
            context: SignalContext with signal data
            state: SymbolState for the symbol
            result: OrderResult from execution
            qty: Quantity traded
        """
        try:
            # Generate and store trade_id
            trade_id = generate_trade_id(context.symbol, context.timestamp)
            state.trade_id = trade_id

            # Compute ATR percentile if we have access to atr_hist
            atr_percentile = 0.5  # Default
            if hasattr(self, '_atr_hist_ref') and self._atr_hist_ref:
                hist = list(self._atr_hist_ref.get(context.symbol, []))
                if len(hist) >= 10:
                    atr_percentile = sum(1 for h in hist if h <= context.atr) / len(hist)

            # Get drawdown info
            portfolio_dd = 0.0
            symbol_dd = 0.0
            if self.drawdown_monitor:
                portfolio_dd = self.drawdown_monitor.get_portfolio_drawdown()
                symbol_dd = self.drawdown_monitor.get_symbol_drawdown(context.symbol)

            # Calculate position size as percentage of portfolio
            position_value = qty * result.avg_price
            position_size_pct = position_value / self.portfolio.total_value if self.portfolio.total_value > 0 else 0.0

            # Calculate hours since last trade
            hours_since_last_trade = 999.0
            if state.last_trade_time:
                delta = context.timestamp - state.last_trade_time
                hours_since_last_trade = delta.total_seconds() / 3600

            # Convert timestamp to ET for market-aware calculations
            ts_utc = context.timestamp if context.timestamp.tzinfo else context.timestamp.replace(tzinfo=timezone.utc)
            ts_et = ts_utc.astimezone(ET)

            # Calculate minutes since market open (9:30 ET)
            market_open_et = ts_et.replace(hour=9, minute=30, second=0, microsecond=0)
            if ts_et >= market_open_et:
                minutes_since_open = int((ts_et - market_open_et).total_seconds() / 60)
            else:
                minutes_since_open = 0

            # Get bars in regime from trade gate if available
            bars_in_regime = getattr(state, 'regime_persist', 1)

            # Build entry context (use ET hour/day for market-aware features)
            entry_context = TradeEntryContext(
                trade_id=trade_id,
                timestamp=context.timestamp,
                symbol=context.symbol,
                side="buy" if context.signal > 0 else "sell",
                qty=qty,
                price=result.avg_price,
                strategy=context.strategy_name or "unknown",
                regime=context.regime,
                atr=context.atr,
                atr_percentile=atr_percentile,
                drawdown_portfolio_pct=portfolio_dd,
                drawdown_symbol_pct=symbol_dd,
                position_size_pct=position_size_pct,
                hour_of_day=ts_et.hour,  # ET hour for US market relevance
                day_of_week=ts_et.weekday(),
                minutes_since_open=minutes_since_open,
                bars_in_regime=bars_in_regime,
                hours_since_last_trade=hours_since_last_trade,
                signal_strength=context.signal,
            )

            self.meta_logger.log_entry(entry_context)

        except Exception as e:
            self.logger.warning(f"[{context.symbol}] Failed to log meta entry: {e}")

    def _log_meta_exit(
        self,
        symbol: str,
        state: SymbolState,
        result: OrderResult,
        action_type: str,
    ) -> None:
        """
        Log trade exit for meta-model training.

        Captures outcome metrics for correlation with entry features.

        Args:
            symbol: Trading symbol
            state: SymbolState with trade data
            result: OrderResult from execution
            action_type: Type of exit (exit, partial_exit, reversal)
        """
        try:
            # Skip if no trade_id (wasn't logged on entry)
            if not state.trade_id:
                return

            # Get entry price for P&L calculation
            entry_price = state.entry_price or result.avg_price
            exit_price = result.avg_price

            # Calculate P&L
            pnl_dollars = 0.0
            pnl_percent = 0.0
            filled_qty = result.filled_qty or 0

            if filled_qty > 0 and entry_price > 0:
                if state.side == "long":
                    pnl_dollars = (exit_price - entry_price) * filled_qty
                else:  # short
                    pnl_dollars = (entry_price - exit_price) * filled_qty

                pnl_percent = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
                if state.side == "short":
                    pnl_percent = -pnl_percent

            # Get excursion metrics
            mae_percent = 0.0
            mfe_percent = 0.0

            if state.max_adverse_excursion is not None and entry_price > 0:
                mae_percent = state.max_adverse_excursion / entry_price
            if state.max_favorable_excursion is not None and entry_price > 0:
                mfe_percent = state.max_favorable_excursion / entry_price

            # Map action_type to exit_reason
            exit_reason_map = {
                "exit": "signal_exit",
                "partial_exit": "partial_profit",
                "reversal": "signal_reversal",
                "stop_loss": "stop_loss",
                "take_profit": "take_profit",
            }
            exit_reason = exit_reason_map.get(action_type, action_type)

            # Build exit context
            exit_context = TradeExitContext(
                trade_id=state.trade_id,
                timestamp=datetime.now(timezone.utc),
                price=exit_price,
                pnl_dollars=pnl_dollars,
                pnl_percent=pnl_percent,
                hold_bars=state.bars_held,
                mae_percent=mae_percent,
                mfe_percent=mfe_percent,
                exit_reason=exit_reason,
            )

            self.meta_logger.log_exit(exit_context)

            # Clear trade_id after logging
            state.trade_id = None

        except Exception as e:
            self.logger.warning(f"[{symbol}] Failed to log meta exit: {e}")

    def set_atr_hist_reference(self, atr_hist: Dict[str, Any]) -> None:
        """
        Set reference to ATR history for ATR percentile calculation.

        Call this from the runner to provide access to atr_hist.

        Args:
            atr_hist: Dict mapping symbol to deque of ATR values
        """
        self._atr_hist_ref = atr_hist

    def _post_execution(self, symbol, state, result, action_type, regime, strategy_name, pre_state=None):
        """Post-execution logging and state updates with Live-specific logging."""
        # Log exit for meta-model training (before base class resets state)
        if action_type in ("exit", "partial_exit", "reversal"):
            self._log_meta_exit(symbol, state, result, action_type)

        # Use base class implementation for common logic
        # Pass pre_state for accurate before/after logging
        super()._post_execution(symbol, state, result, action_type, regime, strategy_name, pre_state=pre_state)

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
        """Flatten all positions using broker's actual positions."""
        self.logger.info("[GUI] Flatten ALL positions")

        # Use broker's actual positions to avoid state mismatch errors
        try:
            broker_positions = await self.broker.get_positions()
        except Exception as e:
            self.logger.error(f"Failed to get broker positions: {e}")
            return

        for pos in broker_positions:
            if pos.qty != 0:
                try:
                    qty = abs(int(pos.qty))
                    side = "sell" if pos.qty > 0 else "buy"
                    await self.broker.place_market_order(
                        symbol=pos.symbol,
                        qty=qty,
                        side=side
                    )
                    self.logger.info(f"Flattened {pos.symbol}: {side} {qty}")
                except Exception as e:
                    self.logger.error(f"Failed to flatten {pos.symbol}: {e}")
                    if self.event_handler:
                        await self.event_handler.emit(EVENT_ALERT, AlertPayload(
                            level="error",
                            message=f"Flatten failed for {symbol}: {str(e)}",
                            symbol=symbol,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                        ))

    async def _handle_flatten_symbol(self, event) -> None:
        """Flatten specific symbol using broker's actual position."""
        symbol = event.payload["symbol"]

        # Get actual position from broker to avoid state mismatch errors
        try:
            broker_positions = await self.broker.get_positions()
            broker_pos = next((p for p in broker_positions if p.symbol == symbol), None)
        except Exception as e:
            self.logger.error(f"[GUI] Failed to get broker positions: {e}")
            return

        if not broker_pos or broker_pos.qty == 0:
            self.logger.info(f"[GUI] No position at broker to flatten for {symbol}")
            return

        qty = abs(int(broker_pos.qty))
        side = "sell" if broker_pos.qty > 0 else "buy"

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