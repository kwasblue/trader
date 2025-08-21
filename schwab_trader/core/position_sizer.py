# core/base/position_sizer.py (or wherever your DynamicPositionSizer lives)
from core.base.position_sizer_base import PositionSizerBase
from typing import Optional
import math

class DynamicPositionSizer(PositionSizerBase):
    """
    A dynamic position sizer that adjusts risk exposure based on market conditions.

    Supports dynamic adjustment of the risk percentage and calculates position
    size based on stop-loss and available capital.
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

    def calculate_position_size(
        self,
        price: float,
        stop_loss_price: float,
        current_cash: float,
        market_conditions: str,
        signal: int
    ) -> int:
        """
        Calculates how many shares to buy/sell based on capital and volatility.

        Args:
            price (float): Entry price of the asset
            stop_loss_price (float): Stop-loss price for the trade
            current_cash (float): Cash available for the trade
            market_conditions (str): 'low_volatility', 'high_volatility', or 'normal'
            signal (int): +1 for long, -1 for short, 0 for no trade

        Returns:
            int: Number of shares to trade
        """
        if signal == 0:
            return 0

        risk_pct = self.adjust_risk_percentage(market_conditions)
        risk_per_trade = current_cash * risk_pct

        if risk_per_trade < 5:
            return 0

        # Directional risk per share
        if signal > 0:
            risk_per_share = price - stop_loss_price
        else:  # short
            risk_per_share = stop_loss_price + price

        if risk_per_share <= 0:
            raise ValueError("Invalid stop-loss: must be logically beyond the entry price for signal direction.")

        position_size = risk_per_trade / risk_per_share
        max_affordable = int(current_cash // price)   # or price
        return max(0, min(int(position_size), max_affordable))
        

    def update_capital(self, new_capital: float) -> None:
        self.capital = new_capital
        print(f"[PositionSizer] Capital updated to: {new_capital}")

    def reset_risk(self, new_risk: float) -> None:
        if not (0 < new_risk < 1):
            raise ValueError("new_risk must be between 0 and 1 (non-inclusive).")
        self.risk_per_trade = new_risk
        self.min_risk_percentage = new_risk * 0.5
        self.max_risk_percentage = new_risk * 3
        print(f"[PositionSizer] Risk percentage reset to: {new_risk}")


class DynamicPositionSizer2(PositionSizerBase):
    """
    Risk- and capital-aware position sizer.

    - Use `update_capital(equity)` to set equity basis for risk sizing.
    - Pass AVAILABLE BUYING POWER as `current_cash` to `calculate_position_size(...)`.
    - Caps by buying power and optional notional % of equity.
    - Applies a fee/slippage buffer.
    - Reserves notional intra-bar so parallel signals don't double-spend.
    """

    def __init__(
        self,
        risk_percentage: float,
        *,
        fee_rate: float = 0.001,                 # 10 bps buffer for fees/slippage
        max_notional_pct: Optional[float] = None,# e.g., 0.25 caps any single order to 25% of equity
        allow_fractional: bool = False,          # keep False if you place share-qty orders
        lot_size: int = 1
    ):
        if not (0 < risk_percentage < 1):
            raise ValueError("risk_percentage must be between 0 and 1 (non-inclusive).")
        self.risk_per_trade = float(risk_percentage)
        self.min_risk_percentage = self.risk_per_trade * 0.5
        self.max_risk_percentage = self.risk_per_trade * 3.0

        self.fee_rate = float(fee_rate)
        self.max_notional_pct = max_notional_pct
        self.allow_fractional = bool(allow_fractional)
        self.lot_size = max(1, int(lot_size))

        # live state
        self.capital: float = 0.0              # treat as equity basis for risk sizing
        self._reserved_notional: float = 0.0   # reserved this bar (BP-aware sizing)

    # ---- risk adaptation by regime ----
    def adjust_risk_percentage(self, market_conditions: str) -> float:
        if market_conditions == "high_volatility":
            return max(self.min_risk_percentage, self.risk_per_trade * 0.5)
        elif market_conditions == "low_volatility":
            return min(self.max_risk_percentage, self.risk_per_trade * 1.25)
        return self.risk_per_trade

    # ---- main API ----
    def calculate_position_size(
        self,
        price: float,
        stop_loss_price: float,
        current_cash: float,     # pass AVAILABLE BUYING POWER here
        market_conditions: str,
        signal: int
    ) -> int:
        """
        Returns integer shares to trade, bounded by risk, buying power, and optional notional cap.
        Also reserves notional for the remainder of the bar.
        """
        if signal == 0:
            return 0

        price = float(price)
        stop_loss_price = float(stop_loss_price)
        bp_live = float(current_cash)

        # gross price with fee/slippage buffer
        px_gross = price * (1.0 + self.fee_rate)
        if px_gross <= 0:
            return 0

        # available BP after prior reservations in this bar
        avail_bp = max(0.0, bp_live - self._reserved_notional)

        # equity basis for risk sizing
        equity = self.capital if self.capital > 0 else bp_live

        # risk dollars by regime-adjusted risk %
        risk_pct = self.adjust_risk_percentage(market_conditions)
        risk_dollars = equity * risk_pct
        if risk_dollars < 5.0:  # too tiny to bother
            return 0

        # directional risk per share
        # long: stop below entry → risk = price - stop
        # short: stop above entry → risk = stop - price
        if signal > 0:
            risk_per_share = price - stop_loss_price
        else:
            risk_per_share = stop_loss_price - price

        if risk_per_share <= 0:
            # invalid stop; don't trade
            return 0

        # risk-based qty
        qty_risk = risk_dollars / risk_per_share

        # capital caps
        qty_cap_bp = avail_bp / px_gross
        if self.max_notional_pct and self.max_notional_pct > 0:
            qty_cap_notional = (equity * self.max_notional_pct) / px_gross
        else:
            qty_cap_notional = float("inf")

        qty_float = min(qty_risk, qty_cap_bp, qty_cap_notional)

        # lotting
        if self.allow_fractional:
            # if your broker supports fractional qty, you could round to, say, 3 decimals
            qty_final = math.floor(qty_float / self.lot_size) * self.lot_size if self.lot_size > 1 else qty_float
        else:
            qty_final = math.floor(qty_float / self.lot_size) * self.lot_size

        qty_int = int(qty_final) if not self.allow_fractional else int(qty_final)  # keep int API for now
        if qty_int <= 0:
            return 0

        # reserve notional so subsequent symbols in the same bar see reduced BP
        self._reserved_notional += qty_int * px_gross
        return qty_int

    # ---- lifecycle helpers ----
    def update_capital(self, new_capital: float) -> None:
        """Set equity basis used for risk sizing."""
        self.capital = float(new_capital)
        print(f"[PositionSizer] Equity updated to: {self.capital:.2f}")

    def reset_risk(self, new_risk: float) -> None:
        if not (0 < new_risk < 1):
            raise ValueError("new_risk must be between 0 and 1 (non-inclusive).")
        self.risk_per_trade = float(new_risk)
        self.min_risk_percentage = self.risk_per_trade * 0.5
        self.max_risk_percentage = self.risk_per_trade * 3.0
        print(f"[PositionSizer] Risk percentage reset to: {new_risk}")

    def reset_bar_reservations(self) -> None:
        """Call once per bar close so the next bar starts with fresh availability."""
        self._reserved_notional = 0.0

    def release_reserved(self, notional: float) -> None:
        """Call if an order is rejected/canceled before fill."""
        self._reserved_notional = max(0.0, self._reserved_notional - max(0.0, float(notional)))

    # optional tuners
    def set_fee_rate(self, fee_rate: float) -> None:
        self.fee_rate = max(0.0, float(fee_rate))

    def set_max_notional_pct(self, pct: Optional[float]) -> None:
        self.max_notional_pct = pct if (pct is None or pct > 0) else None
