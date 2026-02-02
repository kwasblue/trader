"""
Default Trade Logic Manager - Business rules for trade approval

Implements standard trade approval logic with:
- Position management (entries, exits, partials)
- Stop loss and take profit logic
- Trailing stops
- Market condition adjustments
- Cooldown periods
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
from datetime import datetime, timezone

from core.base.trade_logic_manager_base import TradeLogicManagerBase
from core.enums import OrderSide
from core.logic.symbol_state import SymbolState
from core.events.eventhandler import EventHandler
from core.events.events import EVENT_STRATEGY_SIGNAL, StrategySignalPayload
from loggers.logger import Logger
import asyncio

# Logger - own file with propagation to app.log
_logger_instance = Logger(
    log_file="trade_logic.log",
    logger_name="DefaultTradeLogic",
    propagate=True
)
logger = _logger_instance.get_logger()


class DefaultTradeLogicManager(TradeLogicManagerBase):
    """
    Default trade logic manager with standard rules.
    
    Features:
    - Market condition-based stops/targets
    - Partial exit logic
    - Trailing stops
    - Signal reversal handling
    - Cooldown enforcement
    """
    
    def __init__(
        self,
        # Stop/Target multipliers by market condition
        tp_mult_low: float = 1.5,
        tp_mult_normal: float = 2.0,
        tp_mult_high: float = 3.0,
        sl_mult_low: float = 1.0,
        sl_mult_normal: float = 1.5,
        sl_mult_high: float = 2.0,
        # Position management
        exit_fraction: float = 0.25,
        trailing_stop: bool = True,
        min_bars_to_hold: int = 3,  # Minimum bars before allowing TP/reversal exits
        # Swing mode - prevents same-day exits (uses regular buying power instead of day trading)
        swing_mode: bool = False,
        min_hold_days: int = 1,  # Minimum days to hold before exit (for swing mode)
        # General settings - cooldown can be time-based, bar-based, or both
        cooldown_seconds: int = 300,
        cooldown_bars: int = 5,
        cooldown_mode: str = 'bars',  # 'time', 'bars', or 'both'
        max_positions: int = 10,
        event_handler: Optional[EventHandler] = None,
        **kwargs
    ):
        """
        Initialize trade logic manager.
        
        Args:
            tp_mult_low: Take profit ATR multiplier (low volatility)
            tp_mult_normal: Take profit ATR multiplier (normal)
            tp_mult_high: Take profit ATR multiplier (high volatility)
            sl_mult_low: Stop loss ATR multiplier (low volatility)
            sl_mult_normal: Stop loss ATR multiplier (normal)
            sl_mult_high: Stop loss ATR multiplier (high volatility)
            exit_fraction: Fraction to exit on partial targets
            trailing_stop: Whether to use trailing stops
            cooldown_seconds: Minimum time between trades
            max_positions: Maximum concurrent positions
            event_handler: Event bus for signal emission
        """
        super().__init__(
            cooldown_seconds=cooldown_seconds,
            cooldown_bars=cooldown_bars,
            cooldown_mode=cooldown_mode,
            max_positions=max_positions,
            **kwargs
        )
        
        # Stop/target multipliers by market condition
        self.tp_mults = {
            "low_volatility": tp_mult_low,
            "normal": tp_mult_normal,
            "high_volatility": tp_mult_high
        }
        self.sl_mults = {
            "low_volatility": sl_mult_low,
            "normal": sl_mult_normal,
            "high_volatility": sl_mult_high
        }
        
        self.exit_fraction = exit_fraction
        self.trailing_stop = trailing_stop
        self.min_bars_to_hold = min_bars_to_hold
        self.swing_mode = swing_mode
        self.min_hold_days = min_hold_days
        self.event_handler = event_handler

        cooldown_info = f"{cooldown_bars} bars" if cooldown_mode == 'bars' else f"{cooldown_seconds}s" if cooldown_mode == 'time' else f"{cooldown_bars} bars/{cooldown_seconds}s"
        swing_info = f", swing_mode=True (min {min_hold_days} day(s))" if swing_mode else ""
        logger.info(f"DefaultTradeLogicManager initialized: cooldown={cooldown_info} (mode={cooldown_mode}), max_positions={max_positions}, trailing_stop={trailing_stop}, min_hold={min_bars_to_hold} bars{swing_info}")
    
    # ========================================================================
    # MAIN DECISION LOGIC
    # ========================================================================
    
    def should_trade(
        self,
        symbol: str,
        state: SymbolState,
        signal: int,
        regime: str,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """
        Main approval logic - decides if trade should execute.
        
        Checks:
        1. Signal validity
        2. Market conditions
        3. Position state
        4. Cooldown
        5. Entry/exit conditions
        """
        # Emit signal event regardless of whether we execute
        if self.event_handler:
            self._emit_signal_event(symbol, state, signal, kwargs.get('price'), regime)
        
        # Basic validation
        if signal == 0:
            return False, "Hold signal"
        
        price = kwargs.get('price')
        atr = kwargs.get('atr')
        
        if price is None or price <= 0:
            return False, "Invalid price"
        
        if atr is None or atr <= 0:
            return False, "Invalid ATR"
        
        # Normalize regime
        condition = self._normalize_regime(regime)
        
        # Check if we're in a position
        in_position = self.is_in_position(state)
        
        # Remove price and atr from kwargs since we're passing them as positional args
        kwargs_clean = {k: v for k, v in kwargs.items() if k not in ['price', 'atr']}
        
        if not in_position:
            # Trying to enter new position
            return self._check_entry(symbol, state, signal, price, atr, condition, **kwargs_clean)
        else:
            # Managing existing position
            return self._check_exit_or_manage(symbol, state, signal, price, atr, condition, **kwargs_clean)
        
    def can_enter_position(
        self,
        symbol: str,
        state: SymbolState,
        side: OrderSide,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Check if new position can be opened."""
        # Must be flat
        if self.is_in_position(state):
            return False, f"Already in position ({self.get_position_side(state)})"
        
        # Check cooldown
        is_allowed, reason = self.check_cooldown(symbol, state)
        if not is_allowed:
            return False, reason
        
        # Check position limit
        account_positions = kwargs.get('account_positions', 0)
        is_allowed, reason = self.check_position_limit(account_positions)
        if not is_allowed:
            return False, reason
        
        # Check market hours
        market_open = kwargs.get('market_open', True)
        is_allowed, reason = self.check_market_hours(market_open)
        if not is_allowed:
            return False, reason
        
        return True, None
    
    def can_exit_position(
        self,
        symbol: str,
        state: SymbolState,
        side: OrderSide,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Check if position can be closed."""
        # Must be in position
        if not self.is_in_position(state):
            return False, "Not in position"
        
        # For full exit, verify side matches position
        if kwargs.get('is_full_exit', True):
            if self.is_long(state) and side != OrderSide.SELL:
                return False, "Cannot buy to exit long (need sell)"
            if self.is_short(state) and side != OrderSide.BUY:
                return False, "Cannot sell to exit short (need buy)"
        
        return True, None
    
    # ========================================================================
    # ENTRY LOGIC
    # ========================================================================
    
    def _check_entry(
        self,
        symbol: str,
        state: SymbolState,
        signal: int,
        price: float,
        atr: float,
        condition: str,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Check if entry should be taken."""
        # Determine side from signal
        side = OrderSide.BUY if signal == 1 else OrderSide.SELL
        
        # Check entry conditions
        can_enter, reason = self.can_enter_position(
            symbol, state, side, **kwargs
        )
        if not can_enter:
            return False, reason
        
        # Calculate stops and targets for this entry
        self._calculate_entry_levels(state, price, atr, condition, side)
        
        return True, None
    
    def _calculate_entry_levels(
        self,
        state: SymbolState,
        price: float,
        atr: float,
        condition: str,
        side: OrderSide
    ) -> None:
        """Calculate stop loss and take profit levels for entry."""
        sl_mult = self.sl_mults[condition]
        tp_mult = self.tp_mults[condition]
        
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
        state.entry_date = datetime.now(timezone.utc)  # Track entry date for swing mode
    
    # ========================================================================
    # EXIT/MANAGEMENT LOGIC
    # ========================================================================
    
    def _check_exit_or_manage(
        self,
        symbol: str,
        state: SymbolState,
        signal: int,
        price: float,
        atr: float,
        condition: str,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Check exit conditions or position management."""
        # Update position tracking
        state.bars_held += 1
        avg_price = kwargs.get('avg_price', price)
        self._update_excursions(state, price, avg_price)

        # Swing mode check - prevent same-day exits (except stop loss)
        if self.swing_mode and state.entry_date:
            now = datetime.now(timezone.utc)
            days_held = (now.date() - state.entry_date.date()).days
            if days_held < self.min_hold_days:
                # Check if it's a stop loss hit (always allowed for protection)
                if self.is_long(state):
                    hit_sl = state.stop_loss is not None and price <= state.stop_loss
                else:
                    hit_sl = state.stop_loss is not None and price >= state.stop_loss

                if hit_sl:
                    logger.warning(f"[{symbol}] SWING MODE: Stop loss hit on same day - allowing protective exit")
                    return True, "Stop loss hit (swing mode override)"

                logger.debug(f"[{symbol}] SWING MODE: Holding position (entered {days_held} day(s) ago, need {self.min_hold_days})")
                return False, f"Swing mode: must hold {self.min_hold_days} day(s) before exit"

        # Minimum hold period - don't exit within first few bars (except stop loss)
        min_bars_to_hold = getattr(self, 'min_bars_to_hold', 3)

        # Check for signal reversal (opposite direction) - but respect minimum hold
        if state.bars_held >= min_bars_to_hold:
            if self.is_long(state) and signal == -1:
                return True, "Signal reversal: exit long"
            if self.is_short(state) and signal == 1:
                return True, "Signal reversal: exit short"

        # Check stop loss / take profit
        if self.is_long(state):
            hit_tp = state.take_profit is not None and price >= state.take_profit
            hit_sl = state.stop_loss is not None and price <= state.stop_loss
        else:
            hit_tp = state.take_profit is not None and price <= state.take_profit
            hit_sl = state.stop_loss is not None and price >= state.stop_loss

        # Stop loss always triggers (protection)
        if hit_sl:
            return True, "Stop loss hit"

        # Take profit requires minimum hold period
        if hit_tp and state.bars_held >= min_bars_to_hold:
            return True, "Take profit hit"

        # Check partial exit targets (only after minimum hold)
        if state.bars_held >= min_bars_to_hold:
            should_partial, reason = self._check_partial_exit(state, price)
            if should_partial:
                return True, reason

        # Update trailing stop if enabled
        if self.trailing_stop:
            self._update_trailing_stop(state, price, atr, condition)

        # No exit condition met, continue holding
        return False, "Continue holding position"
    
    def _check_partial_exit(
        self,
        state: SymbolState,
        price: float
    ) -> Tuple[bool, Optional[str]]:
        """Check if partial exit target hit."""
        if not state.partial_exit_targets:
            return False, None
        
        target = state.partial_exit_targets[0]
        
        if self.is_long(state) and price >= target:
            state.partial_exit_targets.pop(0)
            return True, f"Partial exit target hit: ${target:.2f}"
        
        if self.is_short(state) and price <= target:
            state.partial_exit_targets.pop(0)
            return True, f"Partial exit target hit: ${target:.2f}"
        
        return False, None
    
    def _update_trailing_stop(
        self,
        state: SymbolState,
        price: float,
        atr: float,
        condition: str
    ) -> None:
        """Update trailing stop loss."""
        sl_mult = self.sl_mults[condition]
        
        if self.is_long(state):
            # Trail stop up as price rises
            new_stop = price - (atr * sl_mult)
            state.stop_loss = max(state.stop_loss, new_stop)
        else:
            # Trail stop down as price falls
            new_stop = price + (atr * sl_mult)
            state.stop_loss = min(state.stop_loss, new_stop)
    
    def _update_excursions(
        self,
        state: SymbolState,
        current_price: float,
        avg_price: float
    ) -> None:
        """Update max favorable and adverse excursions."""
        if self.is_long(state):
            excursion = current_price - avg_price
            if state.max_favorable_excursion is None or excursion > state.max_favorable_excursion:
                state.max_favorable_excursion = excursion
            if state.max_adverse_excursion is None or excursion < state.max_adverse_excursion:
                state.max_adverse_excursion = excursion
        else:
            excursion = avg_price - current_price
            if state.max_favorable_excursion is None or excursion > state.max_favorable_excursion:
                state.max_favorable_excursion = excursion
            if state.max_adverse_excursion is None or excursion < state.max_adverse_excursion:
                state.max_adverse_excursion = excursion
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _normalize_regime(self, regime: Optional[str]) -> str:
        """Normalize regime to known conditions."""
        if regime in ("low_volatility", "normal", "high_volatility"):
            return regime
        return "normal"
    
    def _emit_signal_event(
        self,
        symbol: str,
        state: SymbolState,
        signal: int,
        price: Optional[float],
        regime: Optional[str]
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
            asyncio.create_task(
                self.event_handler.emit(EVENT_STRATEGY_SIGNAL, payload)
            )
        except Exception as e:
            logger.error(f"Failed to emit signal event: {e}")
    
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