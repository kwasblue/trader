"""
ML Module - Machine learning components for trade quality prediction.

Components:
- TradeQualityFilter: Filters trades based on ML model predictions
"""

from core.ml.trade_quality_filter import TradeQualityFilter, get_trade_filter

__all__ = [
    'TradeQualityFilter',
    'get_trade_filter',
]
