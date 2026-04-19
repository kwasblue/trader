"""
Risk Module

Centralized risk management and metrics for the trading system.

Usage:
    from core.risk import StandardRiskModel, RiskQuantifier

    # New approach
    model = StandardRiskModel(risk_free_rate=0.04)
    metrics = model.compute_metrics(returns)

    # Legacy approach (still supported)
    quantifier = RiskQuantifier()
    quantifier.set_returns(df, 'return')
    sharpe = quantifier.calculate_sharpe_ratio()
"""

from core.risk.metrics import StandardRiskModel, RiskQuantifier
from core.base.risk_model_base import RiskModelBase

__all__ = [
    'RiskModelBase',
    'StandardRiskModel',
    'RiskQuantifier',
]
