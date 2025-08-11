from core.base.position_sizer_base import PositionSizerBase

class KellyPositionSizer(PositionSizerBase):
    """
    Position sizer using the Kelly Criterion.
    Requires estimated win probability and reward-to-risk ratio.
    """

    def __init__(self, kelly_fraction: float = 0.5):
        """
        Args:
            kelly_fraction (float): Fraction of full Kelly allocation to use (e.g. 0.5 = half-Kelly)
        """
        if not (0 < kelly_fraction <= 1):
            raise ValueError("kelly_fraction must be in (0, 1]")
        self.kelly_fraction = kelly_fraction

    def calculate_position_size(
        self,
        price: float,
        stop_loss_price: float,
        current_cash: float,
        market_conditions: str,
        signal: int,
        win_probability: float = 0.55,
        reward_to_risk: float = 2.0,
    ) -> int:
        """
        Calculate position size using Kelly formula.

        Args:
            price (float): Entry price
            stop_loss_price (float): Stop-loss price
            current_cash (float): Available capital
            market_conditions (str): Not used in this sizer
            signal (int): Trade direction (+1, -1, or 0)
            win_probability (float): Estimated probability of winning
            reward_to_risk (float): Reward-to-risk ratio (R)

        Returns:
            int: Position size
        """
        if signal == 0 or price <= 0 or stop_loss_price <= 0:
            return 0

        p = win_probability
        b = reward_to_risk
        q = 1 - p

        kelly_fraction = ((b * p - q) / b) * self.kelly_fraction

        if kelly_fraction <= 0:
            return 0  # No trade if expected edge is negative

        risk_per_trade = current_cash * kelly_fraction

        if signal > 0:
            risk_per_share = price - stop_loss_price
        else:
            risk_per_share = stop_loss_price + price

        if risk_per_share <= 0:
            raise ValueError("Invalid stop-loss relative to price")

        position_size = risk_per_trade / risk_per_share
        return max(1, int(position_size))
