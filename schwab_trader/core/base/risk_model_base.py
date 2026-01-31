"""
Risk Model - Calculate risk and performance metrics

Provides framework for computing various risk metrics from returns data:
- Volatility (standard deviation, ATR)
- Drawdown (max drawdown, current drawdown)
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Value at Risk (VaR, CVaR)
- Other metrics (beta, correlation, etc.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RiskModelBase(ABC):
    """
    Abstract base class for risk metric calculation.
    
    Risk models analyze return series to compute risk and performance metrics.
    These metrics are used for:
    - Portfolio monitoring
    - Strategy evaluation
    - Risk management decisions
    - Performance reporting
    
    Design Philosophy:
    - Risk model only CALCULATES metrics
    - Does NOT enforce limits (that's risk manager's job)
    - Does NOT make trading decisions
    - Provides both individual metrics and metric sets
    
    Example:
        risk_model = StandardRiskModel()
        
        # Compute all metrics
        metrics = risk_model.compute_metrics(returns)
        print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
        print(f"Max DD: {metrics['max_drawdown']:.2%}")
        
        # Compute single metric
        sharpe = risk_model.compute_metric(returns, "sharpe_ratio")
    """
    
    def __init__(self, **kwargs):
        """
        Initialize risk model with configuration.
        
        Args:
            **kwargs: Model-specific parameters like:
                - risk_free_rate: Risk-free rate for Sharpe ratio
                - confidence_level: Confidence level for VaR
                - periods_per_year: Trading periods (252 for daily, etc.)
        """
        self.params = dict(kwargs)
        self.risk_free_rate = kwargs.get('risk_free_rate', 0.0)
        self.periods_per_year = kwargs.get('periods_per_year', 252)
        self.confidence_level = kwargs.get('confidence_level', 0.95)
    
    # ========================================================================
    # ABSTRACT METHODS
    # ========================================================================
    
    @abstractmethod
    def compute_metrics(
        self,
        returns: pd.Series,
        prices: Optional[pd.Series] = None,
        benchmark_returns: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute comprehensive set of risk metrics.
        
        Calculates all available risk and performance metrics from
        return series. This is the main method for getting a complete
        risk profile.
        
        Args:
            returns: Time series of returns (daily, hourly, etc.)
            prices: Optional price series (for some calculations)
            benchmark_returns: Optional benchmark returns (for beta, alpha)
            **kwargs: Additional parameters:
                - start_date: Filter returns from this date
                - end_date: Filter returns to this date
                - min_periods: Minimum periods required
                
        Returns:
            Dictionary with risk metrics:
                - 'total_return': Cumulative return
                - 'annualized_return': Annualized return
                - 'volatility': Standard deviation of returns
                - 'sharpe_ratio': Risk-adjusted return
                - 'sortino_ratio': Downside risk-adjusted return
                - 'max_drawdown': Maximum peak-to-trough decline
                - 'current_drawdown': Current drawdown from peak
                - 'calmar_ratio': Return / max drawdown
                - 'win_rate': Percentage of positive returns
                - 'profit_factor': Gross profit / gross loss
                - 'var_95': Value at Risk (95% confidence)
                - 'cvar_95': Conditional VaR (expected shortfall)
                - ... and more
                
        Example:
            metrics = risk_model.compute_metrics(returns)
            print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
            print(f"Max DD: {metrics['max_drawdown']:.2%}")
        """
        pass
    
    @abstractmethod
    def compute_metric(
        self,
        returns: pd.Series,
        metric_name: str,
        **kwargs
    ) -> float:
        """
        Compute a single risk metric.
        
        More efficient when you only need one specific metric instead
        of computing all metrics.
        
        Args:
            returns: Time series of returns
            metric_name: Name of metric to compute:
                - 'sharpe_ratio'
                - 'max_drawdown'
                - 'volatility'
                - 'sortino_ratio'
                - 'calmar_ratio'
                - 'var_95' or 'var_99'
                - 'cvar_95' or 'cvar_99'
                - etc.
            **kwargs: Metric-specific parameters
            
        Returns:
            Computed metric value
            
        Raises:
            ValueError: If metric_name is not recognized
            
        Example:
            sharpe = risk_model.compute_metric(returns, "sharpe_ratio")
            max_dd = risk_model.compute_metric(returns, "max_drawdown")
        """
        pass
    
    # ========================================================================
    # HELPER METHODS (Can be used by implementations)
    # ========================================================================
    
    def compute_total_return(self, returns: pd.Series) -> float:
        """
        Compute cumulative return.
        
        Args:
            returns: Return series
            
        Returns:
            Total cumulative return
        """
        if len(returns) == 0:
            return 0.0
        return (1 + returns).prod() - 1
    
    def compute_annualized_return(self, returns: pd.Series) -> float:
        """
        Compute annualized return.
        
        Args:
            returns: Return series
            
        Returns:
            Annualized return
        """
        if len(returns) == 0:
            return 0.0
        
        total_return = self.compute_total_return(returns)
        n_periods = len(returns)
        
        if n_periods == 0:
            return 0.0
        
        return (1 + total_return) ** (self.periods_per_year / n_periods) - 1
    
    def compute_volatility(
        self,
        returns: pd.Series,
        annualized: bool = True
    ) -> float:
        """
        Compute volatility (standard deviation).
        
        Args:
            returns: Return series
            annualized: Whether to annualize volatility
            
        Returns:
            Volatility
        """
        if len(returns) < 2:
            return 0.0
        
        vol = returns.std()
        
        if annualized:
            vol *= np.sqrt(self.periods_per_year)
        
        return vol
    
    def compute_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Compute Sharpe ratio.
        
        Args:
            returns: Return series
            
        Returns:
            Sharpe ratio (excess return / volatility)
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / self.periods_per_year
        
        if excess_returns.std() == 0:
            return 0.0
        
        return np.sqrt(self.periods_per_year) * excess_returns.mean() / excess_returns.std()
    
    def compute_sortino_ratio(self, returns: pd.Series) -> float:
        """
        Compute Sortino ratio (uses downside deviation).
        
        Args:
            returns: Return series
            
        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / self.periods_per_year
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return 0.0
        
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return 0.0
        
        return np.sqrt(self.periods_per_year) * excess_returns.mean() / downside_std
    
    def compute_max_drawdown(self, returns: pd.Series) -> float:
        """
        Compute maximum drawdown.
        
        Args:
            returns: Return series
            
        Returns:
            Maximum drawdown (negative value)
        """
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()
    
    def compute_current_drawdown(self, returns: pd.Series) -> float:
        """
        Compute current drawdown from peak.
        
        Args:
            returns: Return series
            
        Returns:
            Current drawdown (negative value or 0)
        """
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.max()
        current_value = cumulative.iloc[-1]
        
        return (current_value - running_max) / running_max
    
    def compute_calmar_ratio(self, returns: pd.Series) -> float:
        """
        Compute Calmar ratio (return / max drawdown).
        
        Args:
            returns: Return series
            
        Returns:
            Calmar ratio
        """
        annual_return = self.compute_annualized_return(returns)
        max_dd = self.compute_max_drawdown(returns)
        
        if max_dd == 0:
            return 0.0
        
        return annual_return / abs(max_dd)
    
    def compute_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Compute Value at Risk.
        
        Args:
            returns: Return series
            confidence: Confidence level (0.95 = 95%)
            
        Returns:
            VaR (negative value representing loss)
        """
        if len(returns) == 0:
            return 0.0
        
        return returns.quantile(1 - confidence)
    
    def compute_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Compute Conditional VaR (Expected Shortfall).
        
        Average of losses beyond VaR threshold.
        
        Args:
            returns: Return series
            confidence: Confidence level (0.95 = 95%)
            
        Returns:
            CVaR (negative value representing expected loss)
        """
        if len(returns) == 0:
            return 0.0
        
        var = self.compute_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def compute_win_rate(self, returns: pd.Series) -> float:
        """
        Compute win rate (percentage of positive returns).
        
        Args:
            returns: Return series
            
        Returns:
            Win rate (0.0 to 1.0)
        """
        if len(returns) == 0:
            return 0.0
        
        return (returns > 0).sum() / len(returns)
    
    def compute_profit_factor(self, returns: pd.Series) -> float:
        """
        Compute profit factor (gross profit / gross loss).
        
        Args:
            returns: Return series
            
        Returns:
            Profit factor (>1 is good)
        """
        if len(returns) == 0:
            return 0.0
        
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def validate_returns(self, returns: pd.Series, min_periods: int = 2) -> bool:
        """
        Validate return series has sufficient data.
        
        Args:
            returns: Return series
            min_periods: Minimum required periods
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If returns are invalid
        """
        if not isinstance(returns, pd.Series):
            raise ValueError("Returns must be a pandas Series")
        
        if len(returns) < min_periods:
            raise ValueError(f"Insufficient data: {len(returns)} < {min_periods}")
        
        if returns.isna().any():
            logger.warning("Returns contain NaN values")
        
        return True
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """Get configuration parameter."""
        return self.params.get(key, default)
    
    def set_param(self, key: str, value: Any) -> None:
        """Set configuration parameter."""
        self.params[key] = value
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(risk_free_rate={self.risk_free_rate:.4f})"