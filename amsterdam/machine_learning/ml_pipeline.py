"""
ML Pipeline - Training and Model Selection for Trading Strategies

This module provides:
- Model training with hyperparameter tuning
- Model persistence and loading
- Leaderboard tracking for model comparison
- Best model selection

Usage:
    # Train models
    from machine_learning.ml_pipeline import MLPipeline
    pipeline = MLPipeline()
    results = pipeline.train(symbols=["AAPL", "MSFT"])

    # Load best model
    model, info = pipeline.load_best_model()
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, precision_recall_curve,
    auc,
)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Optional LightGBM
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

logger = logging.getLogger(__name__)


@dataclass
class MLConfig:
    """Configuration for ML pipeline."""

    # Training settings
    seed: int = 42
    test_size: float = 0.2
    cv_splits: int = 5
    n_iter: int = 20
    scoring: str = "roc_auc"

    # Prediction thresholds
    buy_threshold: float = 0.52
    sell_threshold: float = 0.48

    # Paths
    experiments_dir: str = "experiments"
    leaderboard_file: str = "leaderboard.csv"

    # Models to train
    enabled_models: List[str] = field(default_factory=lambda: [
        "logistic_regression", "random_forest", "lightgbm"
    ])

    @classmethod
    def from_file(cls, path: Optional[str] = None) -> "MLConfig":
        """Load config from JSON file."""
        if path is None:
            path = PROJECT_ROOT / "config" / "ml_config.json"
        else:
            path = Path(path)

        if not path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return cls()

        try:
            with open(path) as f:
                data = json.load(f)

            training = data.get("training", {})
            prediction = data.get("prediction", {})
            paths = data.get("paths", {})

            return cls(
                seed=training.get("seed", 42),
                test_size=training.get("test_size", 0.2),
                cv_splits=training.get("cv_splits", 5),
                n_iter=training.get("n_iter", 20),
                scoring=training.get("scoring", "roc_auc"),
                buy_threshold=prediction.get("buy_threshold", 0.52),
                sell_threshold=prediction.get("sell_threshold", 0.48),
                experiments_dir=paths.get("experiments_dir", "experiments"),
                leaderboard_file=paths.get("leaderboard_file", "leaderboard.csv"),
            )
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return cls()


class MLPipeline:
    """
    Machine Learning Pipeline for Trading Strategy Development.

    Handles:
    - Data preprocessing
    - Model training with hyperparameter tuning
    - Model persistence
    - Model selection from leaderboard
    """

    def __init__(self, config: Optional[MLConfig] = None):
        self.config = config or MLConfig.from_file()
        self.run_id: Optional[str] = None
        self.run_dir: Optional[Path] = None

    def _create_run_dir(self, tag: str) -> Path:
        """Create directory for this training run."""
        self.run_id = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = PROJECT_ROOT / self.config.experiments_dir / f"{tag}_{self.run_id}"

        for subdir in ['models', 'search', 'metrics', 'reports']:
            (self.run_dir / subdir).mkdir(parents=True, exist_ok=True)

        return self.run_dir

    def _get_model_configs(self) -> Dict[str, dict]:
        """Get model configurations."""
        configs = {}

        if "logistic_regression" in self.config.enabled_models:
            configs['Logistic_Regression'] = {
                'model': LogisticRegression(
                    class_weight='balanced',
                    max_iter=2000,
                    solver='saga',
                    random_state=self.config.seed
                ),
                'params': {
                    'model__C': np.logspace(-3, 2, 20),
                    'model__penalty': ['l1', 'l2'],
                }
            }

        if "random_forest" in self.config.enabled_models:
            configs['Random_Forest'] = {
                'model': RandomForestClassifier(
                    class_weight='balanced',
                    n_jobs=-1,
                    random_state=self.config.seed
                ),
                'params': {
                    'model__n_estimators': np.arange(200, 801, 100),
                    'model__max_depth': [None] + list(np.arange(4, 21, 4)),
                    'model__min_samples_split': [2, 5, 10],
                    'model__max_features': ['sqrt', 'log2', None],
                }
            }

        if "lightgbm" in self.config.enabled_models and HAS_LGBM:
            configs['LightGBM'] = {
                'model': LGBMClassifier(
                    n_estimators=600,
                    objective='binary',
                    random_state=self.config.seed,
                    verbose=-1,
                ),
                'params': {
                    'model__learning_rate': np.logspace(-3, -1, 10),
                    'model__num_leaves': np.arange(16, 64),
                    'model__max_depth': [-1, 4, 6, 8, 10],
                    'model__subsample': np.linspace(0.6, 1.0, 5),
                    'model__colsample_bytree': np.linspace(0.6, 1.0, 5),
                }
            }

        return configs

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess data for training."""
        df = df.sort_values(by='Date').copy()
        # Handle both epoch milliseconds and string date formats
        if df['Date'].dtype in ('int64', 'float64'):
            df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        else:
            df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Drop all-NaN columns
        df.dropna(axis=1, how='all', inplace=True)

        # Binary next-day up label
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)

        X = df.drop(columns=['Target'])
        y = df['Target']
        return X, y

    def _create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Create sklearn preprocessor."""
        num_features = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_features = X.select_dtypes(include=['object']).columns.tolist()
        if 'Ticker' not in cat_features and 'Ticker' in X.columns:
            cat_features.append('Ticker')

        num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
        ])

        cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        return ColumnTransformer([
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ])

    def _clean_params(self, d: dict) -> dict:
        """Convert numpy types to Python types."""
        out = {}
        for k, v in d.items():
            if hasattr(v, 'item'):
                try:
                    out[k] = v.item()
                    continue
                except Exception:
                    pass
            if hasattr(v, 'tolist'):
                try:
                    out[k] = v.tolist()
                    continue
                except Exception:
                    pass
            out[k] = v
        return out

    def _save_model(self, model, model_name: str) -> str:
        """Save trained model."""
        path = self.run_dir / 'models' / f"{model_name.lower()}_best.joblib"
        joblib.dump(model, path)
        return str(path)

    def _save_metrics(self, metrics: dict, model_name: str) -> str:
        """Save model metrics."""
        path = self.run_dir / 'metrics' / f'{model_name}_metrics.json'

        def json_default(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.ndarray,)):
                return o.tolist()
            return str(o)

        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2, default=json_default)
        return str(path)

    def _update_leaderboard(self, results: Dict[str, dict], tag: str):
        """Update global leaderboard."""
        lb_path = PROJECT_ROOT / self.config.experiments_dir / self.config.leaderboard_file

        rows = []
        for model_name, vals in results.items():
            rows.append({
                'run_id': self.run_id,
                'tag': tag,
                'model': model_name,
                'roc_auc': vals.get('ROC-AUC'),
                'pr_auc': vals.get('PR-AUC'),
                'accuracy': vals.get('Accuracy'),
                'f1': vals.get('F1-Score'),
                'run_dir': str(self.run_dir),
            })

        df = pd.DataFrame(rows)
        if lb_path.exists():
            old = pd.read_csv(lb_path)
            df = pd.concat([old, df], ignore_index=True)
        df.to_csv(lb_path, index=False)

    def train(
        self,
        data: Optional[pd.DataFrame] = None,
        symbols: Optional[List[str]] = None,
        tag: str = "training"
    ) -> Dict[str, dict]:
        """
        Train all configured models.

        Args:
            data: Pre-loaded DataFrame with stock data
            symbols: List of symbols to load data for (if data not provided)
            tag: Run tag for identification

        Returns:
            Dictionary of results per model
        """
        # Load data if not provided
        if data is None:
            if symbols is None:
                symbols = ["AAPL"]
            data = self._load_stock_data(symbols)

        # Preprocess
        X, y = self.preprocess_data(data)

        # Split data (chronological, no shuffle)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, shuffle=False
        )

        # Create run directory
        self._create_run_dir(tag)

        # Save metadata
        meta = {
            'run_id': self.run_id,
            'tag': tag,
            'seed': self.config.seed,
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
        }
        with open(self.run_dir / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        # Create preprocessor
        preprocessor = self._create_preprocessor(X)

        # Train models
        results = {}
        cv = TimeSeriesSplit(n_splits=self.config.cv_splits)
        model_configs = self._get_model_configs()

        for model_name, config in model_configs.items():
            logger.info(f"Training {model_name}...")

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', config['model'])
            ])

            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=config['params'],
                n_iter=self.config.n_iter,
                scoring=self.config.scoring,
                cv=cv,
                verbose=1,
                n_jobs=-1,
                random_state=self.config.seed,
                refit=True,
            )

            search.fit(X_train, y_train)

            # Evaluate
            best = search.best_estimator_
            y_pred = best.predict(X_test)
            y_prob = best.predict_proba(X_test)[:, 1] if hasattr(best, 'predict_proba') else None

            # Calculate metrics
            pr_auc = roc = None
            if y_prob is not None:
                prec, rec, _ = precision_recall_curve(y_test, y_prob)
                pr_auc = auc(rec, prec)
                roc = roc_auc_score(y_test, y_prob)

            result = {
                'Best Params': self._clean_params(search.best_params_),
                'Accuracy': float(accuracy_score(y_test, y_pred)),
                'Precision': float(precision_score(y_test, y_pred, zero_division=0)),
                'Recall': float(recall_score(y_test, y_pred, zero_division=0)),
                'F1-Score': float(f1_score(y_test, y_pred, zero_division=0)),
                'ROC-AUC': None if roc is None else float(roc),
                'PR-AUC': None if pr_auc is None else float(pr_auc),
            }

            # Save artifacts
            model_path = self._save_model(best, model_name)
            metrics_path = self._save_metrics(result, model_name)
            result['Artifacts'] = {'model': model_path, 'metrics': metrics_path}

            results[model_name] = result
            logger.info(f"{model_name}: ROC-AUC={result['ROC-AUC']:.4f}")

        # Save results and update leaderboard
        with open(self.run_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        self._update_leaderboard(results, tag)

        logger.info(f"Training complete. Artifacts saved to: {self.run_dir}")
        return results

    def _load_stock_data(self, symbols: List[str]) -> pd.DataFrame:
        """Load stock data for training."""
        try:
            from data.datautils import load_stock_Data
            store = load_stock_Data(symbols)
            frames = []
            for ticker in symbols:
                try:
                    tmp = store.get_dataframe(ticker)
                    if tmp is not None and not tmp.empty:
                        tmp = tmp.copy()
                        tmp['Ticker'] = ticker
                        frames.append(tmp)
                except KeyError:
                    logger.warning(f"No data for {ticker}")
            return pd.concat(frames, ignore_index=True)
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def load_best_model(
        self,
        model: Optional[str] = None,
        primary: str = 'roc_auc',
    ) -> Tuple[Any, Dict]:
        """
        Load the best model from the leaderboard.

        Args:
            model: Specific model type (e.g., 'LightGBM')
            primary: Primary metric for ranking

        Returns:
            Tuple of (sklearn pipeline, info dict)
        """
        lb_path = PROJECT_ROOT / self.config.experiments_dir / self.config.leaderboard_file

        if not lb_path.exists():
            raise FileNotFoundError(f"Leaderboard not found: {lb_path}")

        df = pd.read_csv(lb_path)

        # Filter by model if specified
        if model is not None:
            df = df[df['model'] == model]

        if df.empty:
            raise ValueError("No models found in leaderboard")

        # Sort by primary metric
        df = df.sort_values(by=primary, ascending=False)
        best_row = df.iloc[0]

        # Load the model
        run_dir = Path(best_row['run_dir'])
        model_name = best_row['model']

        # Try to find model path from results.json
        results_path = run_dir / 'results.json'
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            if model_name in results and 'Artifacts' in results[model_name]:
                model_path = results[model_name]['Artifacts'].get('model')
                if model_path and Path(model_path).exists():
                    pipe = joblib.load(model_path)
                    info = {
                        'model_name': model_name,
                        'run_dir': str(run_dir),
                        'metrics': {
                            'roc_auc': best_row.get('roc_auc'),
                            'accuracy': best_row.get('accuracy'),
                            'f1': best_row.get('f1'),
                        }
                    }
                    return pipe, info

        # Fallback to conventional path
        model_path = run_dir / 'models' / f"{model_name.lower()}_best.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        pipe = joblib.load(model_path)
        info = {
            'model_name': model_name,
            'run_dir': str(run_dir),
            'metrics': {
                'roc_auc': best_row.get('roc_auc'),
                'accuracy': best_row.get('accuracy'),
                'f1': best_row.get('f1'),
            }
        }
        return pipe, info


# CLI entry point
def main():
    """Train models from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Train ML models for trading")
    parser.add_argument("--symbols", "-s", type=str, default="AAPL",
                        help="Comma-separated symbols")
    parser.add_argument("--tag", "-t", type=str, default="training",
                        help="Run tag")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    pipeline = MLPipeline()
    results = pipeline.train(symbols=symbols, tag=args.tag)

    print("\nResults:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for k, v in metrics.items():
            if k != 'Artifacts' and isinstance(v, float):
                print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
