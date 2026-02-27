#!/usr/bin/env python3
"""
ML Training Pipeline - Train trade quality prediction model.

This script trains a model to predict whether a trade will be profitable
based on entry conditions (features logged by MetaTradeLogger).

Usage:
    # Train from all available data
    python tools/train_trade_model.py

    # Train from specific parquet file
    python tools/train_trade_model.py --input data/trades_full.parquet

    # Train with custom parameters
    python tools/train_trade_model.py --min-trades 200 --test-size 0.3

    # Evaluate existing model
    python tools/train_trade_model.py --evaluate-only

Output:
    - models/trade_quality_model.joblib (trained model)
    - models/model_metrics.json (performance metrics)
    - models/feature_importance.json (feature rankings)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import joblib

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TradeQualityTrainer:
    """
    Trains a model to predict trade quality (profitable vs unprofitable).

    Features used:
    - ATR percentile (volatility context)
    - Portfolio/symbol drawdown
    - Position size as % of portfolio
    - Time features (hour, day of week)
    - Regime persistence
    - Signal strength
    """

    # Default feature columns
    DEFAULT_FEATURES = [
        'feat_atr_percentile',
        'feat_drawdown_portfolio_pct',
        'feat_drawdown_symbol_pct',
        'feat_position_size_pct',
        'feat_hour_of_day',
        'feat_day_of_week',
        'feat_minutes_since_open',
        'feat_bars_in_regime',
        'feat_hours_since_last_trade',
        'feat_signal_strength',
    ]

    def __init__(
        self,
        model_dir: str = "models",
        features: Optional[List[str]] = None,
    ):
        """
        Initialize trainer.

        Args:
            model_dir: Directory to save models
            features: Feature columns to use (defaults to DEFAULT_FEATURES)
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.features = features or self.DEFAULT_FEATURES
        self.model = None
        self.scaler = StandardScaler()
        self.metrics: Dict[str, Any] = {}

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load training data from Parquet or JSONL.

        Args:
            data_path: Path to data file

        Returns:
            DataFrame with trade data
        """
        path = Path(data_path)

        if path.suffix == '.parquet':
            df = pd.read_parquet(path)
        elif path.suffix == '.jsonl':
            # Convert JSONL to DataFrame
            df = self._load_jsonl(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        print(f"Loaded {len(df)} trades from {path}")
        return df

    def _load_jsonl(self, path: Path) -> pd.DataFrame:
        """Load and merge entry/exit events from JSONL."""
        entries = {}
        exits = {}

        with open(path, 'r') as f:
            for line in f:
                event = json.loads(line)
                trade_id = event['trade_id']

                if event['event'] == 'entry':
                    entries[trade_id] = event
                elif event['event'] == 'exit':
                    exits[trade_id] = event

        # Merge entries and exits
        rows = []
        for trade_id, entry in entries.items():
            if trade_id in exits:
                exit_event = exits[trade_id]
                row = {
                    'trade_id': trade_id,
                    'symbol': entry['symbol'],
                    'side': entry['side'],
                    'entry_price': entry['price'],
                    'exit_price': exit_event['price'],
                    'qty': entry['qty'],
                }

                # Add features
                for key, value in entry.get('features', {}).items():
                    row[f'feat_{key}'] = value

                # Add outcomes
                for key, value in exit_event.get('outcome', {}).items():
                    row[f'out_{key}'] = value

                rows.append(row)

        return pd.DataFrame(rows)

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'out_pnl_dollars',
        min_trades: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and target for training.

        Args:
            df: Trade DataFrame
            target_col: Column to use for target (P&L)
            min_trades: Minimum trades required

        Returns:
            X (features), y (target), feature_names
        """
        if len(df) < min_trades:
            raise ValueError(f"Need at least {min_trades} trades, got {len(df)}")

        # Filter to available features
        available_features = [f for f in self.features if f in df.columns]

        if not available_features:
            raise ValueError(f"No features found. Available columns: {list(df.columns)}")

        print(f"Using {len(available_features)} features: {available_features}")

        # Extract features
        X = df[available_features].copy()

        # Handle missing values
        X = X.fillna(0)

        # Create binary target (profitable = 1, unprofitable = 0)
        y = (df[target_col] > 0).astype(int)

        print(f"Target distribution: {y.value_counts().to_dict()}")
        print(f"Win rate: {y.mean():.1%}")

        return X.values, y.values, available_features

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model_type: str = 'random_forest',
        test_size: float = 0.2,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Names of features
            model_type: 'random_forest', 'gradient_boosting', or 'logistic'
            test_size: Fraction for test set
            cv_folds: Cross-validation folds

        Returns:
            Dictionary of metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Select model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
        elif model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        print(f"\nTraining {model_type} model...")

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=cv_folds, scoring='accuracy'
        )
        print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

        # Train final model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate on test set
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        self.metrics = {
            'model_type': model_type,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'test_accuracy': float(accuracy_score(y_test, y_pred)),
            'test_precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'test_recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'test_f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'test_roc_auc': float(roc_auc_score(y_test, y_prob)),
            'baseline_accuracy': float(max(y.mean(), 1 - y.mean())),
            'trained_at': datetime.now().isoformat(),
            'features': feature_names,
        }

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(feature_names, self.model.feature_importances_))
            self.metrics['feature_importance'] = dict(
                sorted(importance.items(), key=lambda x: x[1], reverse=True)
            )

        # Print report
        print("\n" + "="*60)
        print("MODEL PERFORMANCE")
        print("="*60)
        print(f"Test Accuracy:  {self.metrics['test_accuracy']:.3f}")
        print(f"Test Precision: {self.metrics['test_precision']:.3f}")
        print(f"Test Recall:    {self.metrics['test_recall']:.3f}")
        print(f"Test F1:        {self.metrics['test_f1']:.3f}")
        print(f"Test ROC-AUC:   {self.metrics['test_roc_auc']:.3f}")
        print(f"Baseline:       {self.metrics['baseline_accuracy']:.3f}")
        print()
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Loss', 'Profit']))

        if 'feature_importance' in self.metrics:
            print("\nFeature Importance:")
            for feat, imp in list(self.metrics['feature_importance'].items())[:10]:
                print(f"  {feat}: {imp:.4f}")

        return self.metrics

    def save_model(self, model_name: str = "trade_quality_model") -> None:
        """Save the trained model and scaler."""
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        # Save model
        model_path = self.model_dir / f"{model_name}.joblib"
        joblib.dump(self.model, model_path)
        print(f"\nModel saved to: {model_path}")

        # Save scaler
        scaler_path = self.model_dir / f"{model_name}_scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")

        # Save metrics
        metrics_path = self.model_dir / f"{model_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")

        # Save feature importance separately for easy access
        if 'feature_importance' in self.metrics:
            importance_path = self.model_dir / f"{model_name}_feature_importance.json"
            with open(importance_path, 'w') as f:
                json.dump(self.metrics['feature_importance'], f, indent=2)
            print(f"Feature importance saved to: {importance_path}")

    def load_model(self, model_name: str = "trade_quality_model") -> None:
        """Load a trained model."""
        model_path = self.model_dir / f"{model_name}.joblib"
        scaler_path = self.model_dir / f"{model_name}_scaler.joblib"

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        print(f"Model loaded from: {model_path}")

    def predict(self, features: Dict[str, float]) -> Tuple[int, float]:
        """
        Predict trade quality.

        Args:
            features: Dictionary of feature values

        Returns:
            (prediction, probability) - 1=profitable, 0=unprofitable
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load first.")

        # Build feature vector
        X = np.array([[features.get(f, 0) for f in self.features]])
        X_scaled = self.scaler.transform(X)

        pred = self.model.predict(X_scaled)[0]
        prob = self.model.predict_proba(X_scaled)[0, 1]

        return int(pred), float(prob)


def find_data_files(data_dir: str = "data") -> List[Path]:
    """Find all parquet files with trade data."""
    data_path = Path(data_dir)
    files = list(data_path.glob("trades*.parquet"))
    return sorted(files, key=lambda x: x.stat().st_size, reverse=True)


def main():
    parser = argparse.ArgumentParser(
        description="Train trade quality prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--input', '-i',
        help='Input data file (parquet or jsonl). Auto-detects if not specified.'
    )
    parser.add_argument(
        '--model-type', '-m',
        choices=['random_forest', 'gradient_boosting', 'logistic'],
        default='random_forest',
        help='Model type to train (default: random_forest)'
    )
    parser.add_argument(
        '--model-name',
        default='trade_quality_model',
        help='Name for saved model files'
    )
    parser.add_argument(
        '--min-trades',
        type=int,
        default=50,
        help='Minimum trades required for training'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set fraction (default: 0.2)'
    )
    parser.add_argument(
        '--evaluate-only',
        action='store_true',
        help='Only evaluate existing model, do not train'
    )
    parser.add_argument(
        '--combine-all',
        action='store_true',
        help='Combine all available parquet files for training'
    )

    args = parser.parse_args()

    trainer = TradeQualityTrainer()

    if args.evaluate_only:
        # Load and evaluate existing model
        trainer.load_model(args.model_name)
        print("Model loaded. Use trainer.predict() to make predictions.")
        return

    # Find or load data
    if args.input:
        df = trainer.load_data(args.input)
    elif args.combine_all:
        # Combine all parquet files
        files = find_data_files()
        if not files:
            print("No parquet files found in data/")
            print("Run: python tools/convert_meta_trades.py first")
            sys.exit(1)

        print(f"Combining {len(files)} parquet files...")
        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset=['trade_id'])
        print(f"Combined {len(df)} unique trades")
    else:
        # Auto-detect largest file
        files = find_data_files()
        if not files:
            # Try logs directory for JSONL
            jsonl_files = list(Path("logs").glob("meta_trades*.jsonl"))
            if jsonl_files:
                largest = max(jsonl_files, key=lambda x: x.stat().st_size)
                print(f"No parquet files found. Using JSONL: {largest}")
                df = trainer.load_data(str(largest))
            else:
                print("No training data found!")
                print("Generate data with: python tools/run_historical_sim.py")
                print("Then convert with: python tools/convert_meta_trades.py")
                sys.exit(1)
        else:
            df = trainer.load_data(str(files[0]))

    # Prepare and train
    try:
        X, y, features = trainer.prepare_data(df, min_trades=args.min_trades)
        trainer.train(
            X, y, features,
            model_type=args.model_type,
            test_size=args.test_size,
        )
        trainer.save_model(args.model_name)

        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Model saved to: models/{args.model_name}.joblib")
        print(f"\nTo use in trading:")
        print("  from tools.train_trade_model import TradeQualityTrainer")
        print("  trainer = TradeQualityTrainer()")
        print(f"  trainer.load_model('{args.model_name}')")
        print("  pred, prob = trainer.predict(features_dict)")

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
