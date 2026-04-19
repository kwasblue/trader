"""
Risk Metrics - Backward Compatibility Facade

This module re-exports RiskQuantifier from core.risk for backward compatibility.
New code should import directly from core.risk.

Usage (legacy):
    from utils.risk_metrics import RiskQuantifier

Preferred (new code):
    from core.risk import StandardRiskModel, RiskQuantifier
"""

from core.risk.metrics import RiskQuantifier

__all__ = ["RiskQuantifier"]
