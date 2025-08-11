# core/base/risk_model_base.py

from abc import ABC, abstractmethod
import pandas as pd

class RiskModelBase(ABC):
    """
    Abstract base class for calculating risk metrics.
    All risk models should inherit from this and implement required methods.
    """

    @abstractmethod
    def compute_metrics(self, returns: pd.Series, **kwargs) -> dict:
        """
        Compute a dictionary of risk metrics from a return series.

        Args:
            returns (pd.Series): Time series of daily (or periodic) returns.
            kwargs: Optional parameters for metric-specific adjustments.

        Returns:
            dict: Dictionary of computed risk metrics.
        """
        pass

    @abstractmethod
    def compute_metric(self, returns: pd.Series, metric_name: str, **kwargs) -> float:
        """
        Compute a single risk metric.

        Args:
            returns (pd.Series): Time series of returns.
            metric_name (str): Name of the metric to compute.
            kwargs: Additional parameters if needed.

        Returns:
            float: Computed value for the specified risk metric.
        """
        pass
