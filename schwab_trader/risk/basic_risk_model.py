# core/risk/basic_risk_model.py 
#can use the 5 year tresury yield as risk free rate

import numpy as np
import pandas as pd
from core.base.risk_model_base import RiskModelBase


class BasicRiskModel(RiskModelBase):
    """
    Implements common risk and performance metrics for trading returns.
    """

    def compute_metrics(self, returns: pd.Series, benchmark: pd.Series = None, risk_free_rate: float = 0.0) -> dict:
        """
        Compute a set of core risk metrics from return series.

        Args:
            returns (pd.Series): Periodic returns (e.g., daily) in decimal form (e.g., 0.01 for 1%).
            benchmark (pd.Series): Optional benchmark return series for alpha/beta.
            risk_free_rate (float): Annualized risk-free rate (e.g., 0.02 = 2%).

        Returns:
            dict: All computed risk metrics.
        """
        metrics = {
            "sharpe": self.sharpe_ratio(returns, risk_free_rate),
            "sortino": self.sortino_ratio(returns, risk_free_rate),
            "max_drawdown": self.max_drawdown(returns),
            "kelly_fraction": self.kelly_criterion(returns),
            "value_at_risk": self.value_at_risk(returns),
        }

        if benchmark is not None:
            alpha, beta = self.alpha_beta(returns, benchmark, risk_free_rate)
            metrics["alpha"] = alpha
            metrics["beta"] = beta

        return metrics

    def compute_metric(self, returns: pd.Series, metric_name: str, **kwargs) -> float:
        """
        Compute an individual risk metric.

        Args:
            returns (pd.Series): Periodic return series.
            metric_name (str): One of ["sharpe", "sortino", "max_drawdown", "kelly", "var", "alpha", "beta"]
            kwargs: Optional args like benchmark or risk_free_rate

        Returns:
            float
        """
        metric_name = metric_name.lower()

        if metric_name == "sharpe":
            return self.sharpe_ratio(returns, kwargs.get("risk_free_rate", 0.0))
        elif metric_name == "sortino":
            return self.sortino_ratio(returns, kwargs.get("risk_free_rate", 0.0))
        elif metric_name == "max_drawdown":
            return self.max_drawdown(returns)
        elif metric_name == "kelly":
            return self.kelly_criterion(returns)
        elif metric_name == "var":
            return self.value_at_risk(returns)
        elif metric_name in ["alpha", "beta"]:
            alpha, beta = self.alpha_beta(returns, kwargs["benchmark"], kwargs.get("risk_free_rate", 0.0))
            return alpha if metric_name == "alpha" else beta
        else:
            raise ValueError(f"Unknown metric '{metric_name}'")

    # --- Risk Metric Implementations ---

    def sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        excess = returns - (risk_free_rate / 252)
        return np.sqrt(252) * excess.mean() / excess.std(ddof=1) if excess.std(ddof=1) else 0.0

    def sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        excess = returns - (risk_free_rate / 252)
        downside_std = np.std(excess[excess < 0])
        return np.sqrt(252) * excess.mean() / downside_std if downside_std else 0.0

    def max_drawdown(self, returns: pd.Series) -> float:
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    def kelly_criterion(self, returns: pd.Series) -> float:
        mean_return = returns.mean()
        variance = returns.var()
        return mean_return / variance if variance > 0 else 0.0

    def value_at_risk(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        return returns.quantile(confidence_level)

    def alpha_beta(self, returns: pd.Series, benchmark: pd.Series, risk_free_rate: float = 0.0):
        """
        CAPM-based alpha and beta calculation.
        """
        excess_returns = returns - (risk_free_rate / 252)
        excess_benchmark = benchmark - (risk_free_rate / 252)
        covariance_matrix = np.cov(excess_returns, excess_benchmark)
        beta = covariance_matrix[0, 1] / covariance_matrix[1, 1] if covariance_matrix[1, 1] != 0 else 0.0
        alpha = excess_returns.mean() - beta * excess_benchmark.mean()
        return alpha * 252, beta  # annualize alpha
