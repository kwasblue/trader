"""
Position Sizer Module

Provides risk-based position sizing for trading systems.

Recommended Usage:
    from core.position_sizer import PositionSizer  # Alias for production sizer

    sizer = PositionSizer(
        risk_percentage=0.02,
        max_trade_pct=0.10,
        max_holding_pct=0.20,
    )

Available Classes:
    PositionSizer (alias)     - Recommended. Points to KellyPositionSizer.
    KellyPositionSizer        - Production sizer with full feature set:
                                * Requires PortfolioState in kwargs
                                * Per-symbol reservation tracking
                                * Per-trade and per-holding caps
                                * Fee rate and lot size awareness
    SimplePositionSizer       - Legacy sizer for simple use cases:
                                * Uses account_value and current_cash directly
                                * No portfolio awareness
                                * Good for standalone backtests

DEPRECATED NAMES (backwards compatibility):
    DynamicPositionSizer2     - Use KellyPositionSizer instead
    DynamicPositionSizer      - Use SimplePositionSizer instead
    LegacyPositionSizer       - Use SimplePositionSizer instead

For live trading and simulation, use PositionSizer (KellyPositionSizer).
"""
import logging
import threading
from core.base.position_sizer_base import PositionSizerBase
from core.logic.portfolio_state import PortfolioState
from loggers.logger import Logger
from core.tracing import trace
from typing import Optional
import math

# Logger - own file with propagation to app.log
_logger_instance = Logger(
    log_file="position_sizer.log",
    logger_name="PositionSizer",
    propagate=True
)
logger = _logger_instance.get_logger()

class SimplePositionSizer(PositionSizerBase):
    """
    A simple ATR-based position sizer that adjusts risk exposure based on market conditions.

    Supports dynamic adjustment of the risk percentage and calculates position
    size based on stop-loss and available capital.

    Use this for:
    - Standalone backtests
    - Simple use cases without portfolio awareness
    - When you don't need per-symbol reservation tracking

    For production trading, use KellyPositionSizer instead.
    """

    def __init__(self, risk_percentage: float): 
        if not (0 < risk_percentage < 1):
            raise ValueError("risk_percentage must be between 0 and 1 (non-inclusive).")
        self.risk_per_trade = risk_percentage
        self.min_risk_percentage = self.risk_per_trade * 0.5
        self.max_risk_percentage = self.risk_per_trade * 3

    def adjust_risk_percentage(self, market_conditions: str) -> float:
        """
        Adjust risk percentage based on market volatility.

        Args:
            market_conditions (str): 'high_volatility', 'low_volatility', or 'normal'

        Returns:
            float: Adjusted risk percentage
        """
        if market_conditions == "high_volatility":
            return max(self.min_risk_percentage, self.risk_per_trade * 0.5)
        elif market_conditions == "low_volatility":
            return min(self.max_risk_percentage, self.risk_per_trade * 1.25)
        return self.risk_per_trade

    @trace
    def calculate_position_size(
        self,
        symbol: str = None,
        price: float = None,
        account_value: float = None,
        signal_strength: float = 1.0,
        atr: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        **kwargs
    ) -> int:
        """
        Calculates how many shares to buy/sell based on capital and volatility.

        Args:
            symbol: Trading symbol (for logging)
            price: Entry price of the asset
            account_value: Total account value
            signal_strength: Signal confidence (0-1)
            atr: Average True Range (used to calculate stop if stop_loss_price not given)
            stop_loss_price: Stop-loss price for the trade
            **kwargs: Additional parameters:
                - signal: +1 for long, -1 for short, 0 for no trade
                - market_conditions: 'low_volatility', 'high_volatility', or 'normal'
                - current_cash: Cash available (uses account_value if not provided)

        Returns:
            int: Number of shares to trade
        """
        # Extract kwargs with defaults
        signal = kwargs.get('signal', 1)
        market_conditions = kwargs.get('market_conditions', 'normal')
        current_cash = kwargs.get('current_cash', account_value)

        if signal == 0:
            return 0

        # ATR sanity check: if ATR is too small relative to price, it's likely
        # from minute bars instead of daily bars. Use a floor of 0.5% of price.
        min_atr = price * 0.005  # 0.5% of price as minimum ATR
        if atr is not None and atr > 0:
            if atr < min_atr:
                logger.warning(
                    f"[{symbol}] ATR too small (${atr:.2f}), likely minute data. "
                    f"Using floor of ${min_atr:.2f} (0.5% of price)"
                )
                atr = min_atr
        else:
            # No ATR provided, use 1% of price as default
            logger.warning(f"[{symbol}] No ATR provided, using 1% of price as default")
            atr = price * 0.01

        # Calculate stop loss from ATR if not provided
        if stop_loss_price is None:
            sl_mult = 1.5  # default stop loss multiplier
            if signal > 0:
                stop_loss_price = price - (atr * sl_mult)
            else:
                stop_loss_price = price + (atr * sl_mult)

        risk_pct = self.adjust_risk_percentage(market_conditions)
        risk_per_trade = current_cash * risk_pct

        if risk_per_trade < 5:
            logger.debug(f"[{symbol}] Risk per trade too small: ${risk_per_trade:.2f}")
            return 0

        # Directional risk per share
        if signal > 0:
            risk_per_share = price - stop_loss_price
        else:  # short
            risk_per_share = stop_loss_price - price

        if risk_per_share <= 0:
            logger.warning(f"[{symbol}] Invalid stop-loss positioning, using fallback")
            risk_per_share = price * 0.02  # 2% fallback

        position_size = risk_per_trade / risk_per_share

        # Multiple safety caps to prevent over-sized positions:
        # 1. Can't afford more than cash allows
        max_affordable = int(current_cash // price) if price > 0 else 0

        # 2. Hard cap: Never risk more than 10% of account on a single position
        max_position_value = current_cash * 0.10  # 10% max position
        max_by_position_cap = int(max_position_value // price) if price > 0 else 0

        # 3. Minimum risk per share floor to prevent tiny risk = huge position
        min_risk_per_share = price * 0.005  # At least 0.5% of price
        if risk_per_share < min_risk_per_share:
            logger.warning(
                f"[{symbol}] Risk/share too small (${risk_per_share:.2f}), "
                f"using floor ${min_risk_per_share:.2f}"
            )
            risk_per_share = min_risk_per_share
            position_size = risk_per_trade / risk_per_share

        final_size = max(0, min(int(position_size), max_affordable, max_by_position_cap))

        logger.info(
            f"[{symbol}] Position size: {final_size} shares @ ${price:.2f} "
            f"(notional=${final_size * price:,.2f}, risk=${risk_per_trade:.2f}, "
            f"risk/share=${risk_per_share:.2f}, cash=${current_cash:,.2f})"
        )

        return final_size
        

    def update_capital(self, new_capital: float) -> None:
        self.capital = new_capital
        logger.debug(f"Capital updated to: {new_capital}")

    def reset_risk(self, new_risk: float) -> None:
        if not (0 < new_risk < 1):
            raise ValueError("new_risk must be between 0 and 1 (non-inclusive).")
        self.risk_per_trade = new_risk
        self.min_risk_percentage = new_risk * 0.5
        self.max_risk_percentage = new_risk * 3
        logger.debug(f"Risk percentage reset to: {new_risk}")


class KellyPositionSizer(PositionSizerBase):
    """
    Production position sizer with full risk management features.

    Features:
    - Uses PortfolioState directly for accurate capital tracking
    - Enforces per-trade notional cap AND per-holding notional cap
    - Caps by buying power, risk budget, and regime adjustment
    - Reserves notional *per-symbol* intra-bar so parallel signals don't double-spend
    - Pyramiding always allowed, constrained by max_holding_pct
    - Thread-safe reservation tracking

    This is the recommended sizer for:
    - Live trading
    - Simulation
    - Any scenario requiring portfolio-aware sizing

    Note: Despite the name, this sizer uses ATR-based risk sizing, not pure Kelly criterion.
    The Kelly criterion sizer is in core/kelly_sizer.py for probability-based sizing.
    """

    def __init__(
        self,
        risk_percentage: float,
        *,
        fee_rate: float = 0.001,
        max_trade_pct: Optional[float] = None,
        max_holding_pct: Optional[float] = None,
        allow_fractional: bool = False,
        lot_size: int = 1,
    ):
        super().__init__()
        if not (0 < risk_percentage < 1):
            raise ValueError("risk_percentage must be between 0 and 1 (non-inclusive).")
        self.risk_per_trade = float(risk_percentage)
        self.min_risk_percentage = self.risk_per_trade * 0.5
        self.max_risk_percentage = self.risk_per_trade * 3.0

        self.fee_rate = float(fee_rate)
        self.max_trade_pct = max_trade_pct
        self.max_holding_pct = max_holding_pct
        self.allow_fractional = bool(allow_fractional)
        self.lot_size = max(1, int(lot_size))

        # per-symbol reserved notional with thread-safe lock
        self._reserved_notional: dict[str, float] = {}
        self._reservation_lock = threading.RLock()

        logger.info(f"KellyPositionSizer initialized: risk={risk_percentage:.1%}, max_trade={max_trade_pct}, max_holding={max_holding_pct}")

    # ---- risk adaptation ----
    def adjust_risk_percentage(self, market_conditions: str) -> float:
        if market_conditions == "high_volatility":
            return max(self.min_risk_percentage, self.risk_per_trade * 0.5)
        elif market_conditions == "low_volatility":
            return min(self.max_risk_percentage, self.risk_per_trade * 1.25)
        return self.risk_per_trade

    @trace
    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        account_value: float,
        signal_strength: float = 1.0,
        atr: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        **kwargs
    ) -> int:
        """
        Calculate position size based on risk management rules.
        
        Args:
            symbol: Trading symbol
            price: Current price
            account_value: Total account value (not used, we use portfolio)
            signal_strength: Signal confidence (not currently used)
            atr: Average True Range
            stop_loss_price: Stop loss price
            **kwargs: Additional parameters:
                - signal: Trade signal (1, -1)
                - portfolio: PortfolioState instance
                - market_conditions: Market regime
                
        Returns:
            Position size in shares
        """
        signal = kwargs.get('signal', 0)
        portfolio = kwargs.get('portfolio')
        market_conditions = kwargs.get('market_conditions', 'normal')
        
        if signal == 0:
            return 0
        
        if portfolio is None:
            raise ValueError("portfolio must be provided in kwargs")
        
        if stop_loss_price is None:
            raise ValueError("stop_loss_price must be provided")

        price = float(price)
        stop_loss_price = float(stop_loss_price)
        equity = float(portfolio.equity)
        cash = float(portfolio.cash)

        # Guard against NaN/inf values (can happen when ATR not yet calculated)
        if math.isnan(price) or math.isnan(stop_loss_price) or math.isinf(stop_loss_price):
            logger.debug(f"[Sizer] {symbol} => No position (invalid price or stop_loss)")
            return 0

        px_gross = price * (1.0 + self.fee_rate)
        if px_gross <= 0 or equity <= 0:
            return 0

        with self._reservation_lock:
            reserved = self._reserved_notional.get(symbol, 0.0)
            total_reserved = sum(self._reserved_notional.values())
        avail_bp = max(0.0, cash - total_reserved)

        logger.debug(f"[Sizer] {symbol}: Cash=${cash:.2f}, Reserved=${total_reserved:.2f}, Available=${avail_bp:.2f}")

        # risk budget
        risk_pct = self.adjust_risk_percentage(market_conditions)
        risk_dollars = equity * risk_pct
        if risk_dollars < 5.0:
            return 0

        # per-share risk
        if signal > 0:  # long
            risk_per_share = price - stop_loss_price
        else:           # short
            risk_per_share = stop_loss_price - price
        if risk_per_share <= 0:
            return 0

        qty_risk = risk_dollars / risk_per_share
        qty_cap_bp = avail_bp / px_gross

        # per-trade cap
        if self.max_trade_pct and self.max_trade_pct > 0:
            max_trade_notional = equity * self.max_trade_pct
            qty_cap_trade = max_trade_notional / px_gross
        else:
            qty_cap_trade = float("inf")

        # per-holding cap
        existing_pos = portfolio.positions.get(symbol)
        existing_notional = abs(existing_pos.qty) * price if existing_pos else 0.0
        if self.max_holding_pct and self.max_holding_pct > 0:
            max_holding_notional = equity * self.max_holding_pct
            remaining_notional = max(0.0, max_holding_notional - existing_notional)
            qty_cap_holding = remaining_notional / px_gross
        else:
            qty_cap_holding = float("inf")

        # final qty
        qty_float = min(qty_risk, qty_cap_bp, qty_cap_trade, qty_cap_holding)
        if self.allow_fractional:
            qty_final = qty_float
        else:
            qty_final = math.floor(qty_float / self.lot_size) * self.lot_size

        qty_int = int(qty_final) if not self.allow_fractional else qty_final
        if qty_int <= 0:
            logger.debug(f"[Sizer] {symbol} => No position (qty<=0). "
                  f"Risk={qty_risk:.2f}, BP={qty_cap_bp:.2f}, TradeCap={qty_cap_trade:.2f}, HoldCap={qty_cap_holding:.2f}")
            return 0
        
        if self.max_holding_pct and self.max_holding_pct > 0:
            max_holding_notional = equity * self.max_holding_pct
            total_notional = existing_notional + (qty_int * px_gross)
            if total_notional > max_holding_notional:
                # shrink position to fit
                max_additional = max_holding_notional - existing_notional
                qty_int = math.floor(max(0.0, max_additional) / px_gross)

        # safety: don't allow trades that wipe out nearly all cash
        if qty_int * px_gross > cash:
            qty_int = math.floor(cash / px_gross)

        if qty_int <= 0:
            logger.debug(f"[Sizer] {symbol} => No position (after cap check).")
            return 0

        # reserve per-symbol notional (tracks this symbol's cumulative reservation)
        new_trade_notional = qty_int * px_gross
        with self._reservation_lock:
            self._reserved_notional[symbol] = self._reserved_notional.get(symbol, 0.0) + new_trade_notional

        logger.info(f"[Sizer] {symbol} => Qty={qty_int} "
              f"(Risk={qty_risk:.2f}, BP={qty_cap_bp:.2f}, TradeCap={qty_cap_trade:.2f}, HoldCap={qty_cap_holding:.2f})")
        return qty_int

    def reset_reserved(self, symbol: Optional[str] = None):
        """Reset reserved notional (per symbol or all)."""
        with self._reservation_lock:
            if symbol is None:
                if self._reserved_notional:
                    logger.debug(f"[Sizer] Resetting all reservations: {self._reserved_notional}")
                self._reserved_notional.clear()
            else:
                if symbol in self._reserved_notional:
                    logger.debug(f"[Sizer] Resetting reservation for {symbol}: ${self._reserved_notional[symbol]:.2f}")
                self._reserved_notional.pop(symbol, None)

    # 🔑 Alias for engine compatibility
    reset_bar_reservations = reset_reserved


# ============================================================================
# Module-level Aliases
# ============================================================================

# Canonical name - use this for new code
PositionSizer = KellyPositionSizer

# ============================================================================
# DEPRECATED ALIASES - For backwards compatibility only
# ============================================================================
import warnings

def _deprecated_alias(name: str, new_name: str, cls):
    """Create a deprecated alias that warns on use."""
    class DeprecatedAlias(cls):
        def __init__(self, *args, **kwargs):
            warnings.warn(
                f"{name} is deprecated. Use {new_name} instead.",
                DeprecationWarning,
                stacklevel=2
            )
            super().__init__(*args, **kwargs)
    DeprecatedAlias.__name__ = name
    DeprecatedAlias.__qualname__ = name
    return DeprecatedAlias

# Deprecated names - emit warning when instantiated
DynamicPositionSizer2 = _deprecated_alias("DynamicPositionSizer2", "KellyPositionSizer", KellyPositionSizer)
DynamicPositionSizer = _deprecated_alias("DynamicPositionSizer", "SimplePositionSizer", SimplePositionSizer)
LegacyPositionSizer = _deprecated_alias("LegacyPositionSizer", "SimplePositionSizer", SimplePositionSizer)

__all__ = [
    # Recommended names
    'PositionSizer',           # Alias for KellyPositionSizer
    'KellyPositionSizer',      # Production sizer with portfolio awareness
    'SimplePositionSizer',     # Simple sizer for backtests
    # Deprecated names (backwards compatibility)
    'DynamicPositionSizer2',   # DEPRECATED: Use KellyPositionSizer
    'DynamicPositionSizer',    # DEPRECATED: Use SimplePositionSizer
    'LegacyPositionSizer',     # DEPRECATED: Use SimplePositionSizer
]