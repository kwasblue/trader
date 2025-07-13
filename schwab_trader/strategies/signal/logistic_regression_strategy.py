import numpy as np
from strategies.base_strategy import BaseStrategy

class LogisticRegressionStrategy(BaseStrategy):
    def generate_signal(self, data):
        model = self.params.get("model")
        if model is None:
            raise ValueError("Model must be provided in params for LogisticRegressionStrategy.")

        try:
            preprocessor = model.named_steps["preprocessor"]
            raw_features = preprocessor.feature_names_in_
            X = data[raw_features].copy()
            X_transformed = preprocessor.transform(X)
            preds_proba = model.named_steps["model"].predict_proba(X_transformed)[:, 1]
            data["Signal"] = np.where(preds_proba > 0.52, 1, np.where(preds_proba < 0.48, -1, 0))
            return data
        except Exception as e:
            print(f"Prediction failed: {e}")
            data["Signal"] = 0
            return data