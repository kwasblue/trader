"""
Trade Quality Filter - Uses ML model to filter trade signals.

Integrates with the trading pipeline to score potential trades
and optionally reject low-quality signals.

Usage:
    from core.ml.trade_quality_filter import TradeQualityFilter

    # Initialize with model
    filter = TradeQualityFilter(model_path="models/trade_quality_model.joblib")

    # Score a potential trade
    features = {
        'feat_atr_percentile': 0.72,
        'feat_drawdown_portfolio_pct': 0.02,
        'feat_hour_of_day': 10,
        ...
    }
    should_trade, score = filter.evaluate(features)

    if should_trade:
        # Execute trade
        pass
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import joblib

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

from loggers.logger import Logger


class TradeQualityFilter:
    """
    ML-based trade quality filter.

    Uses a trained model to predict whether a trade will be profitable
    and optionally filters out low-confidence signals.
    """

    DEFAULT_FEATURES = [
        "feat_atr_percentile",
        "feat_drawdown_portfolio_pct",
        "feat_drawdown_symbol_pct",
        "feat_position_size_pct",
        "feat_hour_of_day",
        "feat_day_of_week",
        "feat_minutes_since_open",
        "feat_bars_in_regime",
        "feat_hours_since_last_trade",
        "feat_signal_strength",
    ]

    def __init__(
        self,
        model_path: str | None = None,
        min_confidence: float = 0.5,
        enabled: bool = True,
        features: list[str] | None = None,
    ):
        """
        Initialize trade quality filter.

        Args:
            model_path: Path to trained model (.joblib)
            min_confidence: Minimum probability to approve trade (0-1)
            enabled: Whether filtering is active
            features: Feature names in order (auto-detected if None)
        """
        self.model_path = model_path
        self.min_confidence = min_confidence
        self.enabled = enabled
        self.features = features or self.DEFAULT_FEATURES

        self.model = None
        self.scaler = None
        self._loaded = False

        # Logger
        self.logger = Logger(
            log_file="trade_quality_filter.log", logger_name="TradeQualityFilter", propagate=True
        ).get_logger()

        # Stats
        self.stats = {
            "total_evaluated": 0,
            "approved": 0,
            "rejected": 0,
            "avg_score": 0.0,
        }

        if model_path and enabled:
            self._load_model()

    def _load_model(self) -> bool:
        """Load the trained model and scaler."""
        if not HAS_JOBLIB:
            self.logger.warning("joblib not installed. ML filtering disabled.")
            self.enabled = False
            return False

        model_path = Path(self.model_path)
        scaler_path = model_path.parent / f"{model_path.stem}_scaler.joblib"

        if not model_path.exists():
            self.logger.warning(f"Model not found: {model_path}. ML filtering disabled.")
            self.enabled = False
            return False

        try:
            self.model = joblib.load(model_path)
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)

            # Load feature names from metrics if available
            metrics_path = model_path.parent / f"{model_path.stem}_metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                    if "features" in metrics:
                        self.features = metrics["features"]

            self._loaded = True
            self.logger.info(f"Loaded trade quality model: {model_path} (min_confidence={self.min_confidence})")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.enabled = False
            return False

    def evaluate(
        self,
        features: dict[str, Any],
        symbol: str = "",
    ) -> tuple[bool, float]:
        """
        Evaluate a potential trade.

        Args:
            features: Dictionary of feature values
            symbol: Symbol for logging

        Returns:
            (should_trade, confidence_score)
        """
        self.stats["total_evaluated"] += 1

        # If disabled or no model, approve all
        if not self.enabled or not self._loaded:
            return True, 1.0

        try:
            # Build feature vector
            X = np.array([[features.get(f, 0) for f in self.features]])

            # Scale if scaler available
            if self.scaler is not None:
                X = self.scaler.transform(X)

            # Predict
            prob = self.model.predict_proba(X)[0, 1]  # P(profitable)

            # Update running average
            n = self.stats["total_evaluated"]
            self.stats["avg_score"] = (self.stats["avg_score"] * (n - 1) + prob) / n

            # Decision
            should_trade = prob >= self.min_confidence

            if should_trade:
                self.stats["approved"] += 1
                self.logger.debug(f"[{symbol}] APPROVED (score={prob:.3f} >= {self.min_confidence})")
            else:
                self.stats["rejected"] += 1
                self.logger.info(f"[{symbol}] REJECTED (score={prob:.3f} < {self.min_confidence})")

            return should_trade, float(prob)

        except Exception as e:
            self.logger.error(f"Evaluation error: {e}")
            return True, 0.5  # Approve on error

    def build_features(
        self,
        atr: float,
        atr_history: list[float],
        portfolio_drawdown: float,
        symbol_drawdown: float,
        position_size_pct: float,
        hour: int,
        day_of_week: int,
        minutes_since_open: int,
        bars_in_regime: int,
        hours_since_last_trade: float,
        signal_strength: int,
    ) -> dict[str, float]:
        """
        Build feature dictionary from raw values.

        Convenience method to construct features dict.
        """
        # Calculate ATR percentile
        if atr_history and len(atr_history) >= 10:
            atr_percentile = sum(1 for h in atr_history if h <= atr) / len(atr_history)
        else:
            atr_percentile = 0.5

        return {
            "feat_atr_percentile": atr_percentile,
            "feat_drawdown_portfolio_pct": portfolio_drawdown,
            "feat_drawdown_symbol_pct": symbol_drawdown,
            "feat_position_size_pct": position_size_pct,
            "feat_hour_of_day": hour,
            "feat_day_of_week": day_of_week,
            "feat_minutes_since_open": minutes_since_open,
            "feat_bars_in_regime": bars_in_regime,
            "feat_hours_since_last_trade": hours_since_last_trade,
            "feat_signal_strength": signal_strength,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get filtering statistics."""
        total = self.stats["total_evaluated"]
        return {
            **self.stats,
            "approval_rate": self.stats["approved"] / total if total > 0 else 0,
            "rejection_rate": self.stats["rejected"] / total if total > 0 else 0,
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = {
            "total_evaluated": 0,
            "approved": 0,
            "rejected": 0,
            "avg_score": 0.0,
        }

    def set_min_confidence(self, value: float) -> None:
        """Update minimum confidence threshold."""
        self.min_confidence = max(0.0, min(1.0, value))
        self.logger.info(f"Min confidence updated to {self.min_confidence}")


# Singleton instance for easy access
_default_filter: TradeQualityFilter | None = None


def get_trade_filter(model_path: str = "models/trade_quality_model.joblib", **kwargs) -> TradeQualityFilter:
    """
    Get or create the default trade quality filter.

    Args:
        model_path: Path to model
        **kwargs: Additional arguments for TradeQualityFilter

    Returns:
        TradeQualityFilter instance
    """
    global _default_filter

    if _default_filter is None:
        _default_filter = TradeQualityFilter(model_path=model_path, **kwargs)

    return _default_filter


__all__ = [
    "TradeQualityFilter",
    "get_trade_filter",
]
