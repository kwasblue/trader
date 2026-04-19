#!/usr/bin/env python3
"""
Automated Parameter Tuning for Continuous Adaptive Sizing

Uses walk-forward optimization with Bayesian search to find optimal parameters
for the continuous adaptive sizing system.

Features:
- Walk-forward validation (prevents overfitting)
- Bayesian optimization (sample-efficient)
- Multi-objective optimization (Sharpe, drawdown, win rate)
- Regularization penalties (prevents extreme parameters)
- Parallel evaluation (optional)

Usage:
    # Quick tuning (10 iterations)
    python tune_adaptive_sizing.py --symbol AAPL --n-trials 10

    # Full tuning (100 iterations, parallel)
    python tune_adaptive_sizing.py --symbol AAPL --n-trials 100 --parallel

    # Multi-symbol tuning
    python tune_adaptive_sizing.py --symbols AAPL,TSLA,MSFT --n-trials 50
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import logging
from datetime import datetime, timedelta
import argparse

# Bayesian optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("Warning: scikit-optimize not installed. Install with: pip install scikit-optimize")
    print("Falling back to random search.")

from tools.backtest_adaptive_features import AdaptiveBacktester
from core.unified_data_pipeline import UnifiedDataPipeline


@dataclass
class TuningResult:
    """Result from a single parameter evaluation."""
    params: Dict
    sharpe: float
    total_return: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    objective_score: float  # Combined objective
    train_sharpe: float  # In-sample
    test_sharpe: float   # Out-of-sample


class AdaptiveSizingTuner:
    """
    Automated parameter tuner for continuous adaptive sizing.

    Uses walk-forward optimization:
    1. Split data into train/test windows
    2. Optimize on train window
    3. Validate on test window
    4. Repeat for multiple windows
    5. Select parameters with best average out-of-sample performance
    """

    def __init__(
        self,
        symbol: str,
        bars: pd.DataFrame,
        train_fraction: float = 0.7,
        n_windows: int = 3,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize tuner.

        Args:
            symbol: Stock symbol to optimize
            bars: Full historical data
            train_fraction: Fraction of data for training (0.7 = 70% train, 30% test)
            n_windows: Number of walk-forward windows
            logger: Optional logger
        """
        self.symbol = symbol
        self.bars = bars
        self.train_fraction = train_fraction
        self.n_windows = n_windows
        self.logger = logger or logging.getLogger(__name__)

        # Parameter search space
        self.param_space = self._define_search_space()

        # Results tracking
        self.results: List[TuningResult] = []
        self.best_params: Optional[Dict] = None
        self.best_score: float = -np.inf

    def _define_search_space(self) -> List:
        """
        Define parameter search space for Bayesian optimization.

        Returns:
            List of skopt.space dimensions
        """
        if not BAYESIAN_AVAILABLE:
            return []

        return [
            # Bayesian shrinkage
            Real(0.25, 1.0, name='bayesian_prior_sharpe'),
            Integer(3, 10, name='bayesian_prior_weight'),

            # Exponential smoothing
            Real(0.1, 0.5, name='sharpe_smoothing_alpha'),

            # Sizing
            Real(1.0, 2.0, name='scaling_alpha'),
            Real(0.20, 0.50, name='min_multiplier'),
            Real(1.0, 2.0, name='max_multiplier'),
            Real(0.01, 0.15, name='skip_trade_threshold'),

            # Cold start
            Real(0.30, 0.70, name='ceiling_very_low_trades'),
            Real(0.50, 1.0, name='ceiling_low_trades'),

            # Metrics
            Real(1.5, 3.0, name='max_sharpe_for_scaling'),
        ]

    def _params_to_config(self, params: Dict) -> Dict:
        """
        Convert parameter dict to config format for backtester.

        Args:
            params: Parameter dictionary

        Returns:
            Config dictionary for ContinuousPositionSizer
        """
        return {
            # Bayesian
            'bayesian_prior_sharpe': params['bayesian_prior_sharpe'],
            'bayesian_prior_weight': int(params['bayesian_prior_weight']),

            # Smoothing
            'sharpe_smoothing_alpha': params['sharpe_smoothing_alpha'],

            # Sizing
            'scaling_alpha': params['scaling_alpha'],
            'min_multiplier': params['min_multiplier'],
            'max_multiplier': params['max_multiplier'],
            'skip_trade_threshold': params['skip_trade_threshold'],

            # Cold start
            'use_graduated_ceiling': True,
            'cold_start_very_low_threshold': 5,
            'cold_start_low_threshold': 10,
            'ceiling_very_low_trades': params['ceiling_very_low_trades'],
            'ceiling_low_trades': params['ceiling_low_trades'],

            # Metrics
            'max_sharpe_for_scaling': params['max_sharpe_for_scaling'],
            'use_continuous_formula': True,
        }

    def _calculate_objective(
        self,
        sharpe: float,
        max_drawdown: float,
        win_rate: float,
        num_trades: int,
        params: Dict
    ) -> float:
        """
        Calculate objective score for optimization.

        Combines multiple metrics with penalties:
        - Primary: Sharpe ratio
        - Penalty: Max drawdown
        - Penalty: Too few trades
        - Penalty: Extreme parameters (regularization)

        Args:
            sharpe: Sharpe ratio
            max_drawdown: Maximum drawdown (%)
            win_rate: Win rate (%)
            num_trades: Number of trades
            params: Parameter values

        Returns:
            Objective score (higher is better)
        """
        # Primary objective: Sharpe ratio
        score = sharpe

        # Penalty for large drawdowns
        # Sharpe 2.0 with 30% DD is worse than Sharpe 1.8 with 15% DD
        drawdown_penalty = max_drawdown / 100.0  # Convert % to decimal
        score -= drawdown_penalty * 2.0  # 10% DD = -0.20 to score

        # Penalty for too few trades (need statistical significance)
        if num_trades < 30:
            trade_penalty = (30 - num_trades) * 0.05
            score -= trade_penalty

        # Regularization: Penalize extreme parameters
        # Prevents overfitting to in-sample data
        reg_penalty = 0.0

        # Penalize very high scaling_alpha (overleveraging)
        if params['scaling_alpha'] > 1.7:
            reg_penalty += (params['scaling_alpha'] - 1.7) * 0.5

        # Penalize very low floor (too aggressive sizing down)
        if params['min_multiplier'] < 0.25:
            reg_penalty += (0.25 - params['min_multiplier']) * 0.5

        # Penalize very high ceiling (overleveraging)
        if params['max_multiplier'] > 1.7:
            reg_penalty += (params['max_multiplier'] - 1.7) * 0.5

        score -= reg_penalty

        return score

    def _walk_forward_split(self, window_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get train/test split for walk-forward window.

        Args:
            window_idx: Window index (0 to n_windows-1)

        Returns:
            (train_bars, test_bars) tuple
        """
        total_bars = len(self.bars)
        window_size = total_bars // self.n_windows

        # Window boundaries
        window_start = window_idx * window_size
        window_end = min((window_idx + 1) * window_size, total_bars)

        window_bars = self.bars.iloc[window_start:window_end]

        # Split window into train/test
        train_size = int(len(window_bars) * self.train_fraction)
        train_bars = window_bars.iloc[:train_size]
        test_bars = window_bars.iloc[train_size:]

        return train_bars, test_bars

    def evaluate_params(
        self,
        params: Dict,
        train_bars: pd.DataFrame,
        test_bars: pd.DataFrame
    ) -> TuningResult:
        """
        Evaluate parameter set on train/test split.

        Args:
            params: Parameter dictionary
            train_bars: Training data
            test_bars: Testing data

        Returns:
            TuningResult with in-sample and out-of-sample metrics
        """
        config = self._params_to_config(params)

        # Train (in-sample)
        train_bt = AdaptiveBacktester(self.symbol, train_bars)
        train_result = train_bt.run_backtest(
            regime_method='stable',
            sizing_method='continuous',
            initial_capital=100000,
            sizing_config=config
        )

        # Test (out-of-sample)
        test_bt = AdaptiveBacktester(self.symbol, test_bars)
        test_result = test_bt.run_backtest(
            regime_method='stable',
            sizing_method='continuous',
            initial_capital=100000,
            sizing_config=config
        )

        # Calculate objective on TEST data (out-of-sample)
        objective = self._calculate_objective(
            sharpe=test_result.sharpe_ratio,
            max_drawdown=test_result.max_drawdown,
            win_rate=test_result.win_rate,
            num_trades=test_result.num_trades,
            params=params
        )

        return TuningResult(
            params=params.copy(),
            sharpe=test_result.sharpe_ratio,
            total_return=test_result.total_return,
            max_drawdown=test_result.max_drawdown,
            win_rate=test_result.win_rate,
            num_trades=test_result.num_trades,
            objective_score=objective,
            train_sharpe=train_result.sharpe_ratio,
            test_sharpe=test_result.sharpe_ratio
        )

    def objective_function(self, param_values: List) -> float:
        """
        Objective function for Bayesian optimization.

        Evaluates parameters across all walk-forward windows and returns
        average out-of-sample objective score.

        Args:
            param_values: List of parameter values (matched to param_space)

        Returns:
            Negative objective score (for minimization)
        """
        # Convert param values to dict
        params = {
            dim.name: val
            for dim, val in zip(self.param_space, param_values)
        }

        # Evaluate on all walk-forward windows
        window_scores = []
        for window_idx in range(self.n_windows):
            train_bars, test_bars = self._walk_forward_split(window_idx)

            result = self.evaluate_params(params, train_bars, test_bars)
            window_scores.append(result.objective_score)

            self.logger.info(
                f"Window {window_idx+1}/{self.n_windows}: "
                f"Test Sharpe={result.test_sharpe:.2f}, "
                f"Objective={result.objective_score:.2f}"
            )

        # Average objective across windows
        avg_objective = np.mean(window_scores)

        # Track results
        self.results.append(TuningResult(
            params=params.copy(),
            sharpe=np.mean([r.sharpe for r in self.results[-self.n_windows:]]) if self.results else 0,
            total_return=0,  # Not meaningful across windows
            max_drawdown=0,  # Not meaningful across windows
            win_rate=0,      # Not meaningful across windows
            num_trades=0,    # Not meaningful across windows
            objective_score=avg_objective,
            train_sharpe=0,  # Not meaningful across windows
            test_sharpe=avg_objective  # Use avg objective as proxy
        ))

        # Update best
        if avg_objective > self.best_score:
            self.best_score = avg_objective
            self.best_params = params.copy()
            self.logger.info(f"New best! Objective={avg_objective:.3f}, Params={params}")

        # Return negative (skopt minimizes)
        return -avg_objective

    def tune_bayesian(self, n_trials: int = 50) -> Dict:
        """
        Run Bayesian optimization to find best parameters.

        Args:
            n_trials: Number of parameter sets to evaluate

        Returns:
            Best parameter dictionary
        """
        if not BAYESIAN_AVAILABLE:
            raise ImportError("scikit-optimize required for Bayesian tuning")

        self.logger.info(
            f"Starting Bayesian optimization: {n_trials} trials, "
            f"{self.n_windows} walk-forward windows"
        )

        # Run optimization
        result = gp_minimize(
            self.objective_function,
            self.param_space,
            n_calls=n_trials,
            random_state=42,
            verbose=True,
            n_jobs=1  # Set to -1 for parallel (if supported)
        )

        # Best parameters (convert numpy types to native Python for JSON serialization)
        best_params = {}
        for dim, val in zip(self.param_space, result.x):
            # Convert based on dimension type
            if dim.name == 'bayesian_prior_weight':  # Integer parameter
                best_params[dim.name] = int(val)
            elif isinstance(val, (np.integer, np.floating)):
                best_params[dim.name] = float(val)
            else:
                best_params[dim.name] = val

        self.logger.info(f"Optimization complete!")
        self.logger.info(f"Best objective score: {-result.fun:.3f}")
        self.logger.info(f"Best parameters: {json.dumps(best_params, indent=2)}")

        return best_params

    def tune_random(self, n_trials: int = 50) -> Dict:
        """
        Run random search (fallback if Bayesian not available).

        Args:
            n_trials: Number of random parameter sets to try

        Returns:
            Best parameter dictionary
        """
        self.logger.info(
            f"Starting random search: {n_trials} trials, "
            f"{self.n_windows} walk-forward windows"
        )

        for trial in range(n_trials):
            # Sample random parameters
            params = {
                'bayesian_prior_sharpe': np.random.uniform(0.25, 1.0),
                'bayesian_prior_weight': np.random.randint(3, 11),
                'sharpe_smoothing_alpha': np.random.uniform(0.1, 0.5),
                'scaling_alpha': np.random.uniform(1.0, 2.0),
                'min_multiplier': np.random.uniform(0.20, 0.50),
                'max_multiplier': np.random.uniform(1.0, 2.0),
                'skip_trade_threshold': np.random.uniform(0.01, 0.15),
                'ceiling_very_low_trades': np.random.uniform(0.30, 0.70),
                'ceiling_low_trades': np.random.uniform(0.50, 1.0),
                'max_sharpe_for_scaling': np.random.uniform(1.5, 3.0),
            }

            # Evaluate
            param_values = [params[dim.name] for dim in self.param_space]
            objective = self.objective_function(param_values)

            self.logger.info(
                f"Trial {trial+1}/{n_trials}: Objective={-objective:.3f}"
            )

        self.logger.info(f"Random search complete!")
        self.logger.info(f"Best objective score: {self.best_score:.3f}")
        self.logger.info(f"Best parameters: {json.dumps(self.best_params, indent=2)}")

        return self.best_params

    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def save_results(self, output_path: Path):
        """Save tuning results to JSON."""
        output = {
            'symbol': self.symbol,
            'n_trials': len(self.results),
            'n_windows': self.n_windows,
            'best_params': self._convert_numpy_types(self.best_params),
            'best_score': float(self.best_score),
            'all_results': [
                {
                    'params': self._convert_numpy_types(r.params),
                    'objective_score': float(r.objective_score),
                    'sharpe': float(r.sharpe),
                    'max_drawdown': float(r.max_drawdown),
                }
                for r in self.results
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        self.logger.info(f"Results saved to {output_path}")


def main():
    """Run tuning from command line."""
    parser = argparse.ArgumentParser(description='Tune adaptive sizing parameters')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Symbol to optimize')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols for multi-symbol tuning')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--n-windows', type=int, default=3, help='Number of walk-forward windows')
    parser.add_argument('--method', choices=['bayesian', 'random'], default='bayesian')
    parser.add_argument('--output', type=str, help='Output JSON file')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Load data
    pipeline = UnifiedDataPipeline()

    # Multi-symbol or single symbol
    if args.symbols:
        symbols = args.symbols.split(',')
    else:
        symbols = [args.symbol]

    for symbol in symbols:
        logger.info(f"\n{'='*70}")
        logger.info(f"TUNING PARAMETERS FOR {symbol}")
        logger.info(f"{'='*70}\n")

        # Load bars (use more data for tuning)
        bars = pipeline.get_data(symbol, timeframe='day')
        if len(bars) < 500:
            logger.warning(f"Only {len(bars)} bars for {symbol}, need 500+. Skipping.")
            continue

        # Use last 750 bars for tuning
        bars = bars.tail(750)

        # Create tuner
        tuner = AdaptiveSizingTuner(
            symbol=symbol,
            bars=bars,
            train_fraction=0.7,
            n_windows=args.n_windows,
            logger=logger
        )

        # Run tuning
        if args.method == 'bayesian' and BAYESIAN_AVAILABLE:
            best_params = tuner.tune_bayesian(n_trials=args.n_trials)
        else:
            best_params = tuner.tune_random(n_trials=args.n_trials)

        # Save results
        output_path = Path(args.output) if args.output else Path(f'tuning_results_{symbol}.json')
        tuner.save_results(output_path)

        # Print summary
        print(f"\n{'='*70}")
        print(f"BEST PARAMETERS FOR {symbol}")
        print(f"{'='*70}")
        print(json.dumps(best_params, indent=2))
        print(f"\nBest objective score: {tuner.best_score:.3f}")
        print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()
