"""
Risk Metrics Module

Provides risk and performance metric calculations for trading systems.

Usage:
    from core.risk import StandardRiskModel

    model = StandardRiskModel(risk_free_rate=0.04)
    metrics = model.compute_metrics(returns)
    print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.base.risk_model_base import RiskModelBase


class StandardRiskModel(RiskModelBase):
    """
    Standard implementation of risk metrics calculation.

    Computes common risk and performance metrics including:
    - Sharpe, Sortino, Calmar ratios
    - Max drawdown, current drawdown
    - VaR, CVaR
    - Win rate, profit factor

    Example:
        model = StandardRiskModel(risk_free_rate=0.04)

        # Compute all metrics
        metrics = model.compute_metrics(returns)

        # Compute single metric
        sharpe = model.compute_metric(returns, "sharpe_ratio")
    """

    def compute_metrics(
        self, returns: pd.Series, prices: pd.Series | None = None, benchmark_returns: pd.Series | None = None, **kwargs
    ) -> dict[str, float]:
        """
        Compute comprehensive set of risk metrics.

        Args:
            returns: Time series of returns
            prices: Optional price series (not used currently)
            benchmark_returns: Optional benchmark returns for beta/alpha
            **kwargs: Additional parameters

        Returns:
            Dictionary with all computed metrics
        """
        if len(returns) < 2:
            return self._empty_metrics()

        metrics = {
            # Returns
            "total_return": self.compute_total_return(returns),
            "annualized_return": self.compute_annualized_return(returns),
            # Volatility
            "volatility": self.compute_volatility(returns),
            "volatility_daily": self.compute_volatility(returns, annualized=False),
            # Risk-adjusted returns
            "sharpe_ratio": self.compute_sharpe_ratio(returns),
            "sortino_ratio": self.compute_sortino_ratio(returns),
            "calmar_ratio": self.compute_calmar_ratio(returns),
            # Drawdown
            "max_drawdown": self.compute_max_drawdown(returns),
            "current_drawdown": self.compute_current_drawdown(returns),
            # Value at Risk
            "var_95": self.compute_var(returns, 0.95),
            "var_99": self.compute_var(returns, 0.99),
            "cvar_95": self.compute_cvar(returns, 0.95),
            # Trade statistics
            "win_rate": self.compute_win_rate(returns),
            "profit_factor": self.compute_profit_factor(returns),
            # Period info
            "n_periods": len(returns),
        }

        # Add beta/alpha if benchmark provided
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            beta, alpha = self._compute_beta_alpha(returns, benchmark_returns)
            metrics["beta"] = beta
            metrics["alpha"] = alpha

        return metrics

    def compute_metric(self, returns: pd.Series, metric_name: str, **kwargs) -> float:
        """
        Compute a single risk metric.

        Args:
            returns: Time series of returns
            metric_name: Name of metric to compute
            **kwargs: Metric-specific parameters

        Returns:
            Computed metric value

        Raises:
            ValueError: If metric_name is not recognized
        """
        method_map = {
            "total_return": self.compute_total_return,
            "annualized_return": self.compute_annualized_return,
            "volatility": self.compute_volatility,
            "sharpe_ratio": self.compute_sharpe_ratio,
            "sortino_ratio": self.compute_sortino_ratio,
            "calmar_ratio": self.compute_calmar_ratio,
            "max_drawdown": self.compute_max_drawdown,
            "current_drawdown": self.compute_current_drawdown,
            "win_rate": self.compute_win_rate,
            "profit_factor": self.compute_profit_factor,
        }

        # Handle VaR/CVaR with confidence levels
        if metric_name.startswith("var_"):
            conf = int(metric_name.split("_")[1]) / 100
            return self.compute_var(returns, conf)
        elif metric_name.startswith("cvar_"):
            conf = int(metric_name.split("_")[1]) / 100
            return self.compute_cvar(returns, conf)

        if metric_name not in method_map:
            raise ValueError(f"Unknown metric: {metric_name}")

        return method_map[metric_name](returns)

    def _compute_beta_alpha(self, returns: pd.Series, benchmark_returns: pd.Series) -> tuple[float, float]:
        """Compute beta and alpha relative to benchmark."""
        # Align series
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 10:
            return 0.0, 0.0

        strat = aligned.iloc[:, 0]
        bench = aligned.iloc[:, 1]

        # Beta = Cov(r, m) / Var(m)
        cov = np.cov(strat, bench)[0, 1]
        var_m = bench.var()

        if var_m == 0:
            return 0.0, 0.0

        beta = cov / var_m

        # Alpha = r - rf - beta * (m - rf)
        rf_period = self.risk_free_rate / self.periods_per_year
        alpha = strat.mean() - rf_period - beta * (bench.mean() - rf_period)
        alpha_annual = alpha * self.periods_per_year

        return beta, alpha_annual

    def _empty_metrics(self) -> dict[str, float]:
        """Return empty metrics dictionary."""
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "volatility_daily": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "current_drawdown": 0.0,
            "var_95": 0.0,
            "var_99": 0.0,
            "cvar_95": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "n_periods": 0,
        }


class RiskQuantifier:
    """
    Legacy risk quantifier for backward compatibility.

    Prefer StandardRiskModel for new code.
    """

    def __init__(self):
        self.returns = None
        self.risk_free_rate = 0.0425
        self.return_column = None

    def set_returns(self, returns_df: pd.DataFrame, return_column: str = "return"):
        if not isinstance(returns_df, pd.DataFrame):
            raise ValueError("Input must be a Pandas DataFrame.")
        if return_column not in returns_df.columns:
            raise ValueError(f"'{return_column}' column not found in the DataFrame.")

        self.returns = pd.to_numeric(returns_df[return_column], errors="coerce").dropna().values
        self.return_column = return_column

    def set_risk_free_rate(self, risk_free_rate: float):
        self.risk_free_rate = risk_free_rate

    def calculate_standard_deviation(self) -> float:
        if self.returns is None:
            raise ValueError("Returns data not set.")
        return np.std(self.returns, ddof=1)

    def calculate_sharpe_ratio(self, periods_per_year: int = 252) -> float:
        if self.returns is None:
            raise ValueError("Returns data not set.")

        adjusted_rf = self.risk_free_rate / periods_per_year
        excess_returns = self.returns - adjusted_rf
        avg_excess = np.mean(excess_returns)
        std_dev = np.std(self.returns, ddof=1)

        if std_dev == 0:
            return np.inf if avg_excess > 0 else -np.inf

        return (avg_excess / std_dev) * np.sqrt(periods_per_year)

    def calculate_sortino_ratio(self) -> float:
        if self.returns is None:
            raise ValueError("Returns data not set.")
        downside = self.returns[self.returns < self.risk_free_rate] - self.risk_free_rate
        downside_std = np.std(downside, ddof=1) if len(downside) > 0 else np.nan
        return np.mean(self.returns - self.risk_free_rate) / downside_std if downside_std != 0 else np.nan

    def calculate_max_drawdown(self) -> float:
        if self.returns is None:
            raise ValueError("Returns data not set.")
        cumulative = (1 + pd.Series(self.returns)).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    def calculate_kelly_criterion(self) -> float:
        if self.returns is None:
            raise ValueError("Returns data not set.")
        mean_return = np.mean(self.returns)
        variance = np.var(self.returns)
        return mean_return / variance if variance != 0 else np.nan

    def calculate_beta(self, market_returns_df: pd.DataFrame) -> float:
        if self.returns is None or not isinstance(market_returns_df, pd.DataFrame):
            raise ValueError("Returns data or market returns not set.")
        if self.return_column not in market_returns_df.columns:
            raise ValueError(f"Market DataFrame must contain '{self.return_column}' column.")
        market = pd.to_numeric(market_returns_df[self.return_column], errors="coerce").dropna().values
        cov = np.cov(self.returns, market)
        return cov[0, 1] / np.var(market)

    def calculate_alpha(self, market_returns_df: pd.DataFrame) -> float:
        if self.returns is None:
            raise ValueError("Returns data not set.")
        market = pd.to_numeric(market_returns_df[self.return_column], errors="coerce").dropna().values
        beta = self.calculate_beta(market_returns_df)
        return np.mean(self.returns) - (self.risk_free_rate + beta * (np.mean(market) - self.risk_free_rate))

    def calculate_treynor_ratio(self, market_returns_df: pd.DataFrame) -> float:
        beta = self.calculate_beta(market_returns_df)
        return (np.mean(self.returns) - self.risk_free_rate) / beta if beta != 0 else np.nan

    def calculate_var(self, confidence_level: float = 0.05) -> float:
        if self.returns is None:
            raise ValueError("Returns data not set.")
        sorted_returns = np.sort(self.returns)
        index = int(confidence_level * len(sorted_returns))
        return sorted_returns[index]

    def risk_profile(self, df: pd.DataFrame, strategy_name: str) -> dict[str, Any]:
        """Calculate key risk performance metrics."""
        if "Strategy_Return" not in df.columns:
            raise ValueError("'Strategy_Return' column not found.")

        annual_return = df["Strategy_Return"].mean() * 252
        std_dev = df["Strategy_Return"].std(ddof=1) * np.sqrt(252)
        sharpe = (annual_return - self.risk_free_rate * 252) / std_dev if std_dev != 0 else np.nan

        cumulative = (1 + df["Strategy_Return"]).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()

        beta = 1.0
        alpha = annual_return - (self.risk_free_rate * 252 + beta * (annual_return - self.risk_free_rate * 252))
        treynor = (annual_return - self.risk_free_rate * 252) / beta if beta != 0 else np.nan
        var = df["Strategy_Return"].quantile(0.05)

        return {
            "Standard Deviation": std_dev,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd,
            "Beta": beta,
            "Alpha": alpha,
            "Treynor Ratio": treynor,
            "VaR": var,
            "Strategy": strategy_name,
        }


__all__ = [
    "StandardRiskModel",
    "RiskQuantifier",
]
