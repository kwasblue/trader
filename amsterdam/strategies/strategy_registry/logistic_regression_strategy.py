"""
Logistic Regression Strategy - ML-based trading signal generation.

Uses a trained sklearn pipeline to predict price direction.
Thresholds are configurable via params or config file.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.base.base_strategy import BaseStrategy

# Default thresholds (can be overridden via config or params)
DEFAULT_BUY_THRESHOLD = 0.52
DEFAULT_SELL_THRESHOLD = 0.48


def _load_ml_config() -> dict:
    """Load ML config from config/ml_config.json if available."""
    config_path = Path(__file__).resolve().parents[2] / "config" / "ml_config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


class LogisticRegressionStrategy(BaseStrategy):
    """
    ML-based strategy using logistic regression (or any sklearn pipeline).

    Params:
        model: Trained sklearn pipeline with 'preprocessor' and 'model' steps
        buy_threshold: Probability threshold for buy signal (default: 0.52)
        sell_threshold: Probability threshold for sell signal (default: 0.48)
    """

    def __init__(self, **params):
        super().__init__(**params)

        # Load thresholds from config or params
        ml_config = _load_ml_config()
        prediction_config = ml_config.get("prediction", {})

        self.buy_threshold = params.get("buy_threshold", prediction_config.get("buy_threshold", DEFAULT_BUY_THRESHOLD))
        self.sell_threshold = params.get(
            "sell_threshold", prediction_config.get("sell_threshold", DEFAULT_SELL_THRESHOLD)
        )

    def generate_signal(self, data) -> int:
        """
        Generate trading signal using the ML model.

        Returns:
            1 for buy, -1 for sell, 0 for hold
        """
        model = self.params.get("model")
        if model is None:
            raise ValueError("Model must be provided in params for LogisticRegressionStrategy.")

        try:
            preprocessor = model.named_steps["preprocessor"]
            raw_features = preprocessor.feature_names_in_

            # Get features from data
            missing_features = [f for f in raw_features if f not in data.columns]
            if missing_features:
                # Log warning but continue with available features
                print(f"Warning: Missing features: {missing_features[:5]}...")

            available_features = [f for f in raw_features if f in data.columns]
            if not available_features:
                return 0

            X = data[available_features].copy()
            X_transformed = preprocessor.transform(X)

            # Get prediction probabilities
            preds_proba = model.named_steps["model"].predict_proba(X_transformed)[:, 1]

            # Generate signals based on thresholds
            signals = np.where(preds_proba > self.buy_threshold, 1, np.where(preds_proba < self.sell_threshold, -1, 0))

            if len(signals) == 0:
                return 0

            return int(signals[-1])

        except Exception as e:
            print(f"Prediction failed: {e}")
            return 0

    def generate_signals_vectorized(self, data: pd.DataFrame) -> list[int] | None:
        """Vectorized ML signal generation for fast backtesting."""
        model = self.params.get("model")
        if model is None:
            return None

        try:
            preprocessor = model.named_steps["preprocessor"]
            raw_features = preprocessor.feature_names_in_

            # Get available features
            available_features = [f for f in raw_features if f in data.columns]
            if not available_features:
                return None

            X = data[available_features].copy()
            X_transformed = preprocessor.transform(X)

            # Get prediction probabilities for all rows at once
            preds_proba = model.named_steps["model"].predict_proba(X_transformed)[:, 1]

            # Generate signals based on thresholds (vectorized)
            signals = np.where(preds_proba > self.buy_threshold, 1, np.where(preds_proba < self.sell_threshold, -1, 0))

            return signals.tolist()

        except Exception as e:
            print(f"Vectorized prediction failed: {e}")
            return None
