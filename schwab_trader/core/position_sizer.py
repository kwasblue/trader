class DynamicPositionSizer:
    def __init__(self, risk_percentage: float): 
        if not (0 < risk_percentage < 1):
            raise ValueError("risk_percentage must be between 0 and 1 (non-inclusive).")
        self.risk_per_trade = risk_percentage  # e.g., 0.02 for 2% risk per trade
        self.min_risk_percentage = self.risk_per_trade * 0.5
        self.max_risk_percentage = self.risk_per_trade * 3
    
    def adjust_risk_percentage(self, market_conditions: str) -> float:
        """
        Dynamically adjust the risk percentage based on market conditions.
        In high volatility, reduce the risk; in low volatility, increase the risk.
        
        Args:
            market_conditions (str): Expected values: 'high_volatility' or 'low_volatility'
            
        Returns:
            float: The updated risk percentage.
        """
        
        if market_conditions == "high_volatility":
            return max(self.min_risk_percentage, self.risk_per_trade * 0.5)
        elif market_conditions == "low_volatility":
            return min(self.max_risk_percentage, self.risk_per_trade * 1.25)
        return self.risk_per_trade
    
    def calculate_position_size(
        self,
        stock_price: float,
        stop_loss_price: float,
        current_cash: float,
        market_conditions: str,
        signal: int
    ) -> int:
        """
        Calculates position size based on entry, stop loss, and risk, supporting long and short trades.

        Args:
            stock_price (float): Entry price of the trade.
            stop_loss_price (float): Stop-loss price.
            current_cash (float): Capital available.
            market_conditions (str): Market volatility classification.
            signal (int): Trading signal. +1 = long, -1 = short, 0 = flat/hold.

        Returns:
            int: Position size (at least 1 if signal is active).
        """
        if signal == 0:
            return 0

        risk_per_trade = current_cash * self.adjust_risk_percentage(market_conditions)
        if risk_per_trade < 5:  # or another small value
            #print(f"[Sizer] Skipping trade: risk ${risk_per_trade:.2f} too small.")
            return 0

        # Determine direction: long (buy) or short (sell)
        if signal > 0:
            risk_per_share = stock_price - stop_loss_price
        elif signal < 0:
            risk_per_share = stop_loss_price + stock_price

        if risk_per_share <= 0:
            raise ValueError("Invalid stop-loss: must be logically beyond the entry price for signal direction.")

        position_size = risk_per_trade / risk_per_share
        return max(1, int(position_size))

    
    def update_capital(self, new_capital: float) -> None:
        self.capital = new_capital
        print(f"Capital updated to: {new_capital}")
    
    def reset_risk(self, new_risk: float) -> None:
        if not (0 < new_risk < 1):
            raise ValueError("new_risk must be between 0 and 1 (non-inclusive).")
        self.risk_per_trade = new_risk
        self.min_risk_percentage = new_risk - 0.02
        self.max_risk_percentage = new_risk + 0.05
        print(f"Risk percentage reset to: {new_risk}")
