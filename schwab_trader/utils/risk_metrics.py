import numpy as np
import pandas as pd

class RiskQuantifier:
    def __init__(self):
        self.returns = None
        self.risk_free_rate = 0.0425
        self.return_column = None

    def set_returns(self, returns_df, return_column: str = 'return'):
        if not isinstance(returns_df, pd.DataFrame):
            raise ValueError("Input must be a Pandas DataFrame.")
        if return_column not in returns_df.columns:
            raise ValueError(f"'{return_column}' column not found in the DataFrame.")
        
        self.returns = pd.to_numeric(returns_df[return_column], errors='coerce').dropna().values
        self.return_column = return_column

    def set_risk_free_rate(self, risk_free_rate):
        self.risk_free_rate = risk_free_rate

    def calculate_standard_deviation(self):
        if self.returns is None:
            raise ValueError("Returns data not set.")
        return np.std(self.returns, ddof=1)

    def calculate_sharpe_ratio(self, periods_per_year=252):
        if self.returns is None or self.risk_free_rate is None:
            raise ValueError("Returns data or risk-free rate not set.")

        # Convert risk-free rate to match the return frequency
        adjusted_risk_free_rate = self.risk_free_rate / periods_per_year  

        # Compute excess returns
        excess_returns = self.returns - adjusted_risk_free_rate

        # Mean excess return
        avg_excess_return = np.mean(excess_returns)

        # Standard deviation of returns
        std_dev = np.std(self.returns, ddof=1)  # ddof=1 ensures sample standard deviation

        if std_dev == 0:
            return np.inf if avg_excess_return > 0 else -np.inf

        # Annualized Sharpe Ratio
        return (avg_excess_return / std_dev) * np.sqrt(periods_per_year)

    
    def calculate_sortino_ratio(self):
        if self.returns is None:
            raise ValueError("Returns data not set.")
        downside_returns = self.returns[self.returns < self.risk_free_rate] - self.risk_free_rate
        downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 0 else np.nan
        return np.mean(self.returns - self.risk_free_rate) / downside_std if downside_std != 0 else np.nan
    
    def calculate_max_drawdown(self):
        if self.returns is None:
            raise ValueError("Returns data not set.")
        cumulative_returns = (1 + pd.Series(self.returns)).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()
    
    def calculate_kelly_criterion(self):
        if self.returns is None:
            raise ValueError("Returns data not set.")
        mean_return = np.mean(self.returns)
        variance = np.var(self.returns)
        return mean_return / variance if variance != 0 else np.nan
    
    def calculate_beta(self, market_returns_df):
        if self.returns is None or not isinstance(market_returns_df, pd.DataFrame):
            raise ValueError("Returns data or market returns not set or not a Pandas DataFrame.")
        if self.return_column not in market_returns_df.columns:
            raise ValueError(f"Market DataFrame must contain a '{self.return_column}' column.")
        market_returns = pd.to_numeric(market_returns_df[self.return_column], errors='coerce').dropna().values
        covariance_matrix = np.cov(self.returns, market_returns)
        return covariance_matrix[0, 1] / np.var(market_returns)
    
    def calculate_alpha(self, market_returns_df):
        if self.returns is None or not isinstance(market_returns_df, pd.DataFrame):
            raise ValueError("Returns data or market returns not set or not a Pandas DataFrame.")
        if self.return_column not in market_returns_df.columns:
            raise ValueError(f"Market DataFrame must contain a '{self.return_column}' column.")
        market_returns = pd.to_numeric(market_returns_df[self.return_column], errors='coerce').dropna().values
        beta = self.calculate_beta(market_returns_df)
        avg_return = np.mean(self.returns)
        market_avg_return = np.mean(market_returns)
        return avg_return - (self.risk_free_rate + beta * (market_avg_return - self.risk_free_rate))
    
    def calculate_treynor_ratio(self, market_returns_df):
        beta = self.calculate_beta(market_returns_df)
        return (np.mean(self.returns) - self.risk_free_rate) / beta if beta != 0 else np.nan


    def calculate_var(self, confidence_level=0.05):
        """Calculates Value at Risk (VaR) at a given confidence level."""
        if self.returns is None:
            raise ValueError("Returns data not set.")
        
        sorted_returns = np.sort(self.returns)
        index = int(confidence_level * len(sorted_returns))
        return sorted_returns[index]

    def risk_profile(self, df: pd.DataFrame, strategy_name: str):
        """Calculates key risk performance metrics."""
        if 'Strategy_Return' not in df.columns:
            raise ValueError("'Strategy_Return' column not found in the DataFrame.")

        annualized_return = df['Strategy_Return'].mean() * 252  
        standard_deviation = df['Strategy_Return'].std(ddof=1) * np.sqrt(252)  
        sharpe_ratio = (annualized_return - self.risk_free_rate * 252) / standard_deviation if standard_deviation != 0 else np.nan

        cumulative_returns = (1 + df['Strategy_Return']).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()

        beta = 1.0  # Default assumption
        alpha = annualized_return - (self.risk_free_rate * 252 + beta * (annualized_return - self.risk_free_rate * 252))
        treynor_ratio = (annualized_return - self.risk_free_rate * 252) / beta if beta != 0 else np.nan
        var = df['Strategy_Return'].quantile(0.05)

        return {
            'Standard Deviation': standard_deviation,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Beta': beta,
            'Alpha': alpha,
            'Treynor Ratio': treynor_ratio,
            'VaR': var,
            'Strategy': strategy_name
        }
