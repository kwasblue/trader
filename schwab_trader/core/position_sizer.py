from core.base.position_sizer_base import PositionSizerBase


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

