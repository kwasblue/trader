import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from risk.basic_risk_model import BasicRiskModel

class AdvancedRiskModel(BasicRiskModel):
    """
    Advanced risk model with expanded analytics for drawdown, tail risk, and higher moments.
    maybe add regime aware metrics later on
    """

    def compute_metrics(self, returns: pd.Series, benchmark: pd.Series = None, risk_free_rate: float = 0.0) -> dict:
        metrics = {
            "sharpe": self.sharpe_ratio(returns, risk_free_rate),
            "sortino": self.sortino_ratio(returns, risk_free_rate),
            "max_drawdown": self.max_drawdown(returns),
            "kelly_fraction": self.kelly_criterion(returns),
            "value_at_risk": self.value_at_risk(returns),
            "conditional_var": self.conditional_var(returns),
            "drawdown_duration": self.drawdown_duration(returns),
            "skewness": skew(returns),
            "kurtosis": kurtosis(returns),
            "omega_ratio": self.omega_ratio(returns, threshold=0.0),
            "rolling_sharpe": self.rolling_sharpe(returns)
        }

        if benchmark is not None:
            alpha, beta = self.alpha_beta(returns, benchmark, risk_free_rate)
            metrics["alpha"] = alpha
            metrics["beta"] = beta

        return metrics

    # === New Metrics ===

    def conditional_var(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        var_threshold = returns.quantile(confidence_level)
        return returns[returns <= var_threshold].mean()

    def drawdown_duration(self, returns: pd.Series) -> int:
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative < peak).astype(int)

        # Count longest drawdown period
        durations = (drawdown * (drawdown.groupby((drawdown != drawdown.shift()).cumsum()).cumcount() + 1))
        return durations.max()
    
    def omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]
        return gains.sum() / losses.sum() if losses.sum() != 0 else np.inf

    def rolling_sharpe(self, returns: pd.Series, window: int = 30, risk_free_rate: float = 0.0) -> float:
        """
        Returns the latest rolling Sharpe over the specified window.
        """
        excess = returns - (risk_free_rate / 252)
        rolling_mean = excess.rolling(window).mean()
        rolling_std = excess.rolling(window).std()
        rolling_sharpe = (np.sqrt(252) * rolling_mean / rolling_std).dropna()
        return rolling_sharpe.iloc[-1] if not rolling_sharpe.empty else 0.0
