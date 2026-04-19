"""
Standard Trade Approver - Pure gating for trade approval

Implements trade gating logic:
- Cooldown enforcement
- Position limits
- Market hours checks
- Basic validation

Position management (SL/TP/trailing/exits) is handled by PositionManager.
The Engine orchestrates both: Approver gates, PositionManager manages.

NAMING HISTORY:
- Previously called "DefaultTradeLogicManager" (deprecated)
- Renamed to "StandardTradeApprover" for clarity

DEPRECATED NAMES (backwards compatibility):
- DefaultTradeLogicManager → Use StandardTradeApprover instead
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from core.base.trade_logic_manager_base import TradeApprover
from core.contracts.events import EVENT_STRATEGY_SIGNAL, StrategySignalPayload
from core.enums import OrderSide
from core.events.eventhandler import EventHandler
from core.logic.symbol_state import SymbolState
from core.tracing import trace
from loggers.logger import Logger

if TYPE_CHECKING:
    from core.contracts.types import SignalContext

# Logger - own file with propagation to app.log
_logger_instance = Logger(log_file="trade_logic.log", logger_name="TradeApprover", propagate=True)
logger = _logger_instance.get_logger()


class StandardTradeApprover(TradeApprover):
    """
    Pure gating approver for trade decisions.

    This approver answers ONE question: "Are we ALLOWED to trade?"
    It does NOT manage positions - that's PositionManager's job.

    Gating checks:
    - Signal validity (not hold)
    - Price/ATR validity
    - Cooldown period elapsed
    - Position limit not exceeded
    - Market hours appropriate

    The Engine orchestrates:
    1. Approver.should_trade() → Are we allowed?
    2. PositionManager.check_exit_conditions() → Should we exit?
    3. PositionManager.calculate_levels() → Set up entry levels
    """

    def __init__(
        self,
        # Stop/Target multipliers (passed to PositionManager via Engine)
        tp_mult_low: float = 1.5,
        tp_mult_normal: float = 2.0,
        tp_mult_high: float = 3.0,
        sl_mult_low: float = 1.0,
        sl_mult_normal: float = 1.5,
        sl_mult_high: float = 2.0,
        # Position management params (used by Engine)
        exit_fraction: float = 0.25,
        trailing_stop: bool = True,
        min_bars_to_hold: int = 3,
        # Swing mode
        swing_mode: bool = False,
        min_hold_days: int = 1,
        # Gating settings
        cooldown_seconds: int = 300,
        cooldown_bars: int = 5,
        cooldown_mode: str = "bars",
        max_positions: int = 10,
        event_handler: EventHandler | None = None,
        **kwargs,
    ):
        """
        Initialize trade approver.

        Args:
            tp_mult_*: Take profit multipliers by regime (stored for Engine access)
            sl_mult_*: Stop loss multipliers by regime (stored for Engine access)
            exit_fraction: Fraction for partial exits
            trailing_stop: Whether trailing stops are enabled
            min_bars_to_hold: Minimum bars before exits
            swing_mode: Enforce multi-day holds
            min_hold_days: Days to hold in swing mode
            cooldown_*: Cooldown settings
            max_positions: Maximum concurrent positions
            event_handler: Event bus for signal emission
        """
        super().__init__(
            cooldown_seconds=cooldown_seconds,
            cooldown_bars=cooldown_bars,
            cooldown_mode=cooldown_mode,
            max_positions=max_positions,
            **kwargs,
        )

        # Store multipliers for Engine/PositionManager access
        self.tp_mults = {"low_volatility": tp_mult_low, "normal": tp_mult_normal, "high_volatility": tp_mult_high}
        self.sl_mults = {"low_volatility": sl_mult_low, "normal": sl_mult_normal, "high_volatility": sl_mult_high}

        # Position management params (Engine reads these)
        self.exit_fraction = exit_fraction
        self.trailing_stop = trailing_stop
        self.min_bars_to_hold = min_bars_to_hold
        self.swing_mode = swing_mode
        self.min_hold_days = min_hold_days
        self.event_handler = event_handler

        cooldown_info = (
            f"{cooldown_bars} bars"
            if cooldown_mode == "bars"
            else f"{cooldown_seconds}s"
            if cooldown_mode == "time"
            else f"{cooldown_bars} bars/{cooldown_seconds}s"
        )
        swing_info = f", swing_mode=True (min {min_hold_days} day(s))" if swing_mode else ""
        logger.info(
            f"StandardTradeApprover initialized: cooldown={cooldown_info} "
            f"(mode={cooldown_mode}), max_positions={max_positions}{swing_info}"
        )

    # ========================================================================
    # MAIN GATING LOGIC
    # ========================================================================

    @trace
    def should_trade(
        self,
        context: SignalContext,
        state: SymbolState,
        account_positions: int = 0,
    ) -> tuple[bool, str | None]:
        """
        Pure gating check - are we ALLOWED to trade?

        Does NOT decide what kind of trade or manage positions.
        Just answers: "Is trading allowed right now?"

        For flat positions: checks entry gating (cooldown, limits, hours)
        For in-position: always returns True (Engine uses PositionManager for exit logic)

        Args:
            context: SignalContext containing signal, price, atr, regime, etc.
            state: Symbol state
            account_positions: Number of current positions in account

        Returns:
            (True, None) if allowed
            (False, reason) if gated
        """
        # Extract from context
        symbol = context.symbol
        signal = context.signal
        price = context.price
        atr = context.atr
        regime = context.regime
        market_open = context.market_open

        # Emit signal event regardless of outcome
        if self.event_handler:
            self._emit_signal_event(symbol, state, signal, price, regime)

        # Basic validation
        if signal == 0:
            return False, "Hold signal"

        if price is None or price <= 0:
            return False, "Invalid price"

        if atr is None or atr <= 0:
            return False, "Invalid ATR"

        # Check if we're in a position
        in_position = self.is_in_position(state)

        if not in_position:
            # Entry gating
            side = OrderSide.BUY if signal == 1 else OrderSide.SELL
            return self._can_enter_position_impl(symbol, state, side, account_positions, market_open)
        else:
            # In position - always allow (Engine uses PositionManager for exit logic)
            return True, None

    def can_enter_position(self, symbol: str, state: SymbolState, side: OrderSide, **kwargs) -> tuple[bool, str | None]:
        """
        Check if new position can be opened (entry gating).

        Args:
            symbol: Trading symbol
            state: Symbol state
            side: Order side
            **kwargs: Additional context (account_positions, market_open)

        Returns:
            (True, None) if allowed
            (False, reason) if gated
        """
        account_positions = kwargs.get("account_positions", 0)
        market_open = kwargs.get("market_open", True)
        return self._can_enter_position_impl(symbol, state, side, account_positions, market_open)

    def _can_enter_position_impl(
        self,
        symbol: str,
        state: SymbolState,
        side: OrderSide,
        account_positions: int,
        market_open: bool,
    ) -> tuple[bool, str | None]:
        """
        Implementation of entry gating logic.

        Args:
            symbol: Trading symbol
            state: Symbol state
            side: Order side
            account_positions: Number of current positions
            market_open: Whether market is open

        Returns:
            (True, None) if allowed
            (False, reason) if gated
        """
        # Must be flat
        if self.is_in_position(state):
            return False, f"Already in position ({self.get_position_side(state)})"

        # Check cooldown
        is_allowed, reason = self.check_cooldown(symbol, state)
        if not is_allowed:
            return False, reason

        # Check position limit
        is_allowed, reason = self.check_position_limit(account_positions)
        if not is_allowed:
            return False, reason

        # Check market hours
        is_allowed, reason = self.check_market_hours(market_open)
        if not is_allowed:
            return False, reason

        return True, None

    def can_exit_position(self, symbol: str, state: SymbolState, side: OrderSide, **kwargs) -> tuple[bool, str | None]:
        """
        Check if position can be closed (exit gating).

        This is a basic check - PositionManager handles exit logic.

        Args:
            symbol: Trading symbol
            state: Symbol state
            side: Order side (SELL to close long, BUY to close short)
            **kwargs: Additional context

        Returns:
            (True, None) if allowed
            (False, reason) if gated
        """
        # Must be in position
        if not self.is_in_position(state):
            return False, "Not in position"

        # Verify side matches position
        if kwargs.get("is_full_exit", True):
            if self.is_long(state) and side != OrderSide.SELL:
                return False, "Cannot buy to exit long (need sell)"
            if self.is_short(state) and side != OrderSide.BUY:
                return False, "Cannot sell to exit short (need buy)"

        return True, None

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _emit_signal_event(
        self, symbol: str, state: SymbolState, signal: int, price: float | None, regime: str | None
    ) -> None:
        """Emit signal event to event bus."""
        if not self.event_handler:
            return

        payload: StrategySignalPayload = {
            "symbol": symbol,
            "strategy": state.strategy_name or "default",
            "signal": signal,
            "price": price,
            "regime": regime,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            asyncio.create_task(self.event_handler.emit(EVENT_STRATEGY_SIGNAL, payload))
        except Exception as e:
            logger.error(f"Failed to emit signal event: {e}")

    def normalize_regime(self, regime: str | None) -> str:
        """Normalize regime to known conditions."""
        if regime in ("low_volatility", "normal", "high_volatility"):
            return regime
        return "normal"


# ============================================================================
# DEPRECATED ALIAS - For backwards compatibility only
# ============================================================================
import warnings


class DefaultTradeLogicManager(StandardTradeApprover):
    """
    DEPRECATED: Use StandardTradeApprover instead.

    This is a backwards-compatibility alias that will be removed in a future version.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "DefaultTradeLogicManager is deprecated. Use StandardTradeApprover instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = ["StandardTradeApprover", "DefaultTradeLogicManager"]
