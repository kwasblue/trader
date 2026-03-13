"""
Position Manager - Manages position lifecycle: entries, stops, targets, exits.

This module separates position management concerns from trade approval logic:
- TradeApprover GATES trades (should we trade?)
- PositionManager MANAGES positions (how do we manage open positions?)

Responsibilities:
- Calculate entry levels (SL/TP/partial targets)
- Update trailing stops
- Track excursions (MFE/MAE)
- Check partial exit conditions
- Calculate exit quantities
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict
from datetime import datetime, timezone

from core.logic.symbol_state import SymbolState
from core.enums import OrderSide
from loggers.logger import Logger

# Logger - own file with propagation to app.log
_logger_instance = Logger(
    log_file="position_manager.log",
    logger_name="PositionManager",
    propagate=True
)
logger = _logger_instance.get_logger()


class PositionManager:
    """
    Manages position lifecycle: entry levels, stops, targets, exits.

    TradeApprover GATES trades (should we trade?)
    PositionManager MANAGES positions (how do we manage open positions?)

    Features:
    - Entry level calculation (stop loss, take profit, partial targets)
    - Trailing stop updates
    - Excursion tracking (MFE/MAE)
    - Partial exit management

    Example:
        manager = PositionManager(
            tp_mults={"normal": 2.0, "low_volatility": 1.5, "high_volatility": 3.0},
            sl_mults={"normal": 1.5, "low_volatility": 1.0, "high_volatility": 2.0},
            exit_fraction=0.25
        )

        # Calculate entry levels
        manager.calculate_levels(state, price=150.0, atr=2.5, condition="normal", side=OrderSide.BUY)

        # Update trailing stop on each bar
        manager.update_trailing_stop(state, price=152.0, atr=2.5, condition="normal")

        # Check for partial exit
        should_exit, reason = manager.check_partial_exit(state, price=154.0)
    """

    def __init__(
        self,
        tp_mults: Optional[Dict[str, float]] = None,
        sl_mults: Optional[Dict[str, float]] = None,
        exit_fraction: float = 0.25,
        min_bars_to_hold: int = 3,
        swing_mode: bool = False,
        min_hold_days: int = 1,
        signal_based_exits: bool = True,
    ):
        """
        Initialize position manager.

        Args:
            tp_mults: Take profit ATR multipliers by market condition
            sl_mults: Stop loss ATR multipliers by market condition
            exit_fraction: Fraction to exit on partial targets
            min_bars_to_hold: Minimum bars before allowing TP/reversal exits
            swing_mode: Whether to enforce multi-day holds
            min_hold_days: Days to hold in swing mode before allowing exit
            signal_based_exits: If False, only exit on SL/TP (no signal reversal exits)
        """
        self.tp_mults = tp_mults or {
            "low_volatility": 1.5,
            "normal": 2.0,
            "high_volatility": 3.0
        }
        self.sl_mults = sl_mults or {
            "low_volatility": 1.0,
            "normal": 1.5,
            "high_volatility": 2.0
        }
        self.exit_fraction = exit_fraction

        # Exit timing rules (owned by PositionManager)
        self.min_bars_to_hold = min_bars_to_hold
        self.swing_mode = swing_mode
        self.min_hold_days = min_hold_days
        self.signal_based_exits = signal_based_exits

        swing_info = f", swing_mode={swing_mode}" if swing_mode else ""
        signal_exit_info = "" if signal_based_exits else ", signal_exits=DISABLED"
        logger.info(
            f"PositionManager initialized: exit_fraction={exit_fraction}, "
            f"min_bars_to_hold={min_bars_to_hold}{swing_info}{signal_exit_info}"
        )

    # ========================================================================
    # ENTRY LEVEL CALCULATION
    # ========================================================================

    def calculate_levels(
        self,
        state: SymbolState,
        price: float,
        atr: float,
        condition: str,
        side: OrderSide
    ) -> None:
        """
        Calculate and set SL/TP/partial targets on state.

        Args:
            state: Symbol state to update
            price: Entry price
            atr: ATR value for volatility-based levels
            condition: Market condition (low_volatility, normal, high_volatility)
            side: Order side (BUY for long, SELL for short)
        """
        sl_mult = self.sl_mults.get(condition, 1.5)
        tp_mult = self.tp_mults.get(condition, 2.0)

        if side == OrderSide.BUY:
            # Long position
            state.stop_loss = price - (atr * sl_mult)
            state.take_profit = price + (atr * tp_mult)
            state.partial_exit_targets = [
                price + (atr * 1.0),
                price + (atr * 2.0)
            ]
        else:
            # Short position
            state.stop_loss = price + (atr * sl_mult)
            state.take_profit = price - (atr * tp_mult)
            state.partial_exit_targets = [
                price - (atr * 1.0),
                price - (atr * 2.0)
            ]

        # Reset position tracking
        state.pyramid_layer = 1
        state.bars_held = 0
        state.max_favorable_excursion = None
        state.max_adverse_excursion = None
        state.entry_date = datetime.now(timezone.utc)

        logger.debug(
            f"[{state.symbol}] Levels set: SL=${state.stop_loss:.2f}, "
            f"TP=${state.take_profit:.2f}, condition={condition}"
        )

    # ========================================================================
    # TRAILING STOP
    # ========================================================================

    def update_trailing_stop(
        self,
        state: SymbolState,
        price: float,
        atr: float,
        condition: str
    ) -> None:
        """
        Update trailing stop based on current price.

        For longs: trails stop up as price rises
        For shorts: trails stop down as price falls

        Args:
            state: Symbol state to update
            price: Current market price
            atr: ATR value
            condition: Market condition
        """
        sl_mult = self.sl_mults.get(condition, 1.5)

        if state.is_long:
            # Trail stop up as price rises
            new_stop = price - (atr * sl_mult)
            if state.stop_loss is None:
                state.stop_loss = new_stop
            else:
                state.stop_loss = max(state.stop_loss, new_stop)
        elif state.is_short:
            # Trail stop down as price falls
            new_stop = price + (atr * sl_mult)
            if state.stop_loss is None:
                state.stop_loss = new_stop
            else:
                state.stop_loss = min(state.stop_loss, new_stop)

    # ========================================================================
    # EXCURSION TRACKING
    # ========================================================================

    def update_excursions(
        self,
        state: SymbolState,
        current_price: float,
        avg_price: float
    ) -> None:
        """
        Track MFE/MAE for trade analytics.

        Args:
            state: Symbol state to update
            current_price: Current market price
            avg_price: Average entry price
        """
        if state.is_long:
            excursion = current_price - avg_price
            if state.max_favorable_excursion is None or excursion > state.max_favorable_excursion:
                state.max_favorable_excursion = excursion
            if state.max_adverse_excursion is None or excursion < state.max_adverse_excursion:
                state.max_adverse_excursion = excursion
        elif state.is_short:
            excursion = avg_price - current_price
            if state.max_favorable_excursion is None or excursion > state.max_favorable_excursion:
                state.max_favorable_excursion = excursion
            if state.max_adverse_excursion is None or excursion < state.max_adverse_excursion:
                state.max_adverse_excursion = excursion

    # ========================================================================
    # EXIT CONDITION CHECKS
    # ========================================================================

    def check_exit_conditions(
        self,
        state: SymbolState,
        price: float,
        signal: int,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check all exit conditions for a position.

        Uses instance settings for exit timing (min_bars_to_hold, swing_mode, etc.)
        to ensure consistent behavior. These are owned by PositionManager, not
        passed from TradeApprover.

        Checks in order:
        1. Swing mode hold period (if enabled)
        2. Stop loss (always triggers for protection)
        3. Signal reversal (respects min hold)
        4. Take profit (respects min hold)
        5. Partial exit targets (respects min hold)

        Args:
            state: Symbol state with position info
            price: Current market price
            signal: Current signal (-1, 0, 1)

        Returns:
            Tuple of (should_exit, reason)
        """
        if not state.is_in_position:
            return False, None

        # Swing mode check - prevent same-day exits (except stop loss)
        if self.swing_mode and state.entry_date:
            now = datetime.now(timezone.utc)
            days_held = (now.date() - state.entry_date.date()).days
            if days_held < self.min_hold_days:
                # Check if it's a stop loss hit (always allowed for protection)
                hit_sl = self._check_stop_loss_hit(state, price)
                if hit_sl:
                    logger.warning(
                        f"[{state.symbol}] SWING MODE: Stop loss hit on same day - "
                        f"allowing protective exit"
                    )
                    return True, "Stop loss hit (swing mode override)"

                logger.debug(
                    f"[{state.symbol}] SWING MODE: Holding position "
                    f"(entered {days_held} day(s) ago, need {self.min_hold_days})"
                )
                return False, f"Swing mode: must hold {self.min_hold_days} day(s) before exit"

        # Stop loss always triggers (protection)
        if self._check_stop_loss_hit(state, price):
            return True, "Stop loss hit"

        # Signal reversal - but respect minimum hold (and config)
        if self.signal_based_exits and state.bars_held >= self.min_bars_to_hold:
            if state.is_long and signal == -1:
                return True, "Signal reversal: exit long"
            if state.is_short and signal == 1:
                return True, "Signal reversal: exit short"

        # Take profit - requires minimum hold period
        if state.bars_held >= self.min_bars_to_hold:
            if self._check_take_profit_hit(state, price):
                return True, "Take profit hit"

        # Partial exit targets - only after minimum hold
        if state.bars_held >= self.min_bars_to_hold:
            should_partial, reason = self.check_partial_exit(state, price)
            if should_partial:
                return True, reason

        return False, None

    def _check_stop_loss_hit(self, state: SymbolState, price: float) -> bool:
        """Check if stop loss was hit."""
        if state.stop_loss is None:
            return False

        if state.is_long:
            return price <= state.stop_loss
        elif state.is_short:
            return price >= state.stop_loss
        return False

    def _check_take_profit_hit(self, state: SymbolState, price: float) -> bool:
        """Check if take profit was hit."""
        if state.take_profit is None:
            return False

        if state.is_long:
            return price >= state.take_profit
        elif state.is_short:
            return price <= state.take_profit
        return False

    def check_partial_exit(
        self,
        state: SymbolState,
        price: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if partial exit target was hit.

        Args:
            state: Symbol state
            price: Current market price

        Returns:
            Tuple of (should_exit, reason)
        """
        if not state.partial_exit_targets:
            return False, None

        target = state.partial_exit_targets[0]

        if state.is_long and price >= target:
            state.partial_exit_targets.pop(0)
            return True, f"Partial exit target hit: ${target:.2f}"

        if state.is_short and price <= target:
            state.partial_exit_targets.pop(0)
            return True, f"Partial exit target hit: ${target:.2f}"

        return False, None

    # ========================================================================
    # EXIT QUANTITY
    # ========================================================================

    def get_exit_quantity(
        self,
        current_position: float,
        is_partial: bool = False
    ) -> int:
        """
        Calculate exit quantity.

        Args:
            current_position: Current position size
            is_partial: Whether this is a partial exit

        Returns:
            Quantity to exit
        """
        if not is_partial:
            return abs(int(current_position))

        # Partial exit
        qty = max(int(abs(current_position) * self.exit_fraction), 1)
        return qty


# Module-level default instance for convenience
_default_manager: Optional[PositionManager] = None


def get_position_manager() -> PositionManager:
    """Get or create the default PositionManager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = PositionManager()
    return _default_manager
