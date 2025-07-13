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
            # Reduce risk in high volatility
            self.risk_per_trade = min(self.min_risk_percentage, self.risk_per_trade - 0.01)
        elif market_conditions == "low_volatility":
            # Increase risk in low volatility
            self.risk_per_trade = max(self.max_risk_percentage, self.risk_per_trade + 0.01)
        # Log the change if you have a logger (or use print for testing)
        return self.risk_per_trade
    
    def calculate_position_size(self, stock_price: float, stop_loss_price: float, current_cash: float, market_conditions: str) -> int:
        """
        Calculate the number of shares to buy based on risk per trade and stop-loss distance.
        
        Args:
            stock_price (float): Current stock price.
            stop_loss_price (float): Stop-loss price.
            current_cash (float): Available cash.
            market_conditions (str): Market condition string.
        
        Returns:
            int: Number of shares (at least 1).
        """
        risk_per_trade = current_cash * self.adjust_risk_percentage(market_conditions)
        risk_per_share = stock_price - stop_loss_price
        if risk_per_share <= 0:
            raise ValueError("Invalid stop loss: Stop loss must be lower than the current stock price.")
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
