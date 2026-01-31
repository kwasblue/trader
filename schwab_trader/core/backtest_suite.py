"""
Advanced Backtesting Suite

Provides:
- Data validation and cleaning
- Parameter optimization (grid search)
- Walk-forward analysis
- Monte Carlo simulation
- Advanced slippage models
- Benchmark comparison
- Performance attribution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from datetime import datetime, timedelta
import logging

from strategies.strategy_registry import load_strategy
from core.position_sizer import DynamicPositionSizer

logger = logging.getLogger(__name__)


# ============================================================================
# DATA VALIDATION
# ============================================================================

@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    cleaned_data: Optional[pd.DataFrame] = None


def validate_ohlcv_data(
    data: pd.DataFrame,
    required_columns: List[str] = None,
    fix_issues: bool = True
) -> ValidationResult:
    """
    Validate OHLCV data for backtesting.

    Checks:
    - Required columns exist
    - No NaN/inf values in critical columns
    - Prices are positive
    - High >= Low, High >= Open/Close, Low <= Open/Close
    - Dates are sorted and unique
    - No unrealistic price moves (> 50% in one bar)

    Args:
        data: DataFrame with OHLCV data
        required_columns: Columns that must exist
        fix_issues: If True, attempt to fix issues

    Returns:
        ValidationResult with status and optionally cleaned data
    """
    if required_columns is None:
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close']

    errors = []
    warnings_list = []
    df = data.copy()

    # Check required columns
    missing = set(required_columns) - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return ValidationResult(False, errors, warnings_list)

    # Normalize column names
    col_map = {}
    for col in df.columns:
        if col.lower() == 'open':
            col_map[col] = 'Open'
        elif col.lower() == 'high':
            col_map[col] = 'High'
        elif col.lower() == 'low':
            col_map[col] = 'Low'
        elif col.lower() == 'close':
            col_map[col] = 'Close'
        elif col.lower() == 'volume':
            col_map[col] = 'Volume'
    if col_map:
        df = df.rename(columns=col_map)

    # Check for NaN values
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                if fix_issues:
                    df[col] = df[col].ffill().bfill()
                    warnings_list.append(f"Fixed {nan_count} NaN values in {col}")
                else:
                    errors.append(f"Found {nan_count} NaN values in {col}")

    # Check for inf values
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                errors.append(f"Found {inf_count} infinite values in {col}")

    # Check prices are positive
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            neg_count = (df[col] <= 0).sum()
            if neg_count > 0:
                errors.append(f"Found {neg_count} non-positive values in {col}")

    # Check OHLC consistency
    if all(c in df.columns for c in ['Open', 'High', 'Low', 'Close']):
        # High should be >= all others
        high_violations = ((df['High'] < df['Open']) |
                          (df['High'] < df['Close']) |
                          (df['High'] < df['Low'])).sum()
        if high_violations > 0:
            if fix_issues:
                df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
                warnings_list.append(f"Fixed {high_violations} High violations")
            else:
                errors.append(f"Found {high_violations} rows where High < other prices")

        # Low should be <= all others
        low_violations = ((df['Low'] > df['Open']) |
                         (df['Low'] > df['Close']) |
                         (df['Low'] > df['High'])).sum()
        if low_violations > 0:
            if fix_issues:
                df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)
                warnings_list.append(f"Fixed {low_violations} Low violations")
            else:
                errors.append(f"Found {low_violations} rows where Low > other prices")

    # Check for unrealistic price moves (> 50% in one bar)
    if 'Close' in df.columns:
        pct_change = df['Close'].pct_change().abs()
        extreme_moves = (pct_change > 0.5).sum()
        if extreme_moves > 0:
            warnings_list.append(f"Found {extreme_moves} bars with >50% price change")

    # Check date sorting
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        if not df['Date'].is_monotonic_increasing:
            if fix_issues:
                df = df.sort_values('Date').reset_index(drop=True)
                warnings_list.append("Sorted data by Date")
            else:
                errors.append("Dates are not sorted in ascending order")

        # Check for duplicates
        dup_count = df['Date'].duplicated().sum()
        if dup_count > 0:
            if fix_issues:
                df = df.drop_duplicates(subset='Date', keep='last')
                warnings_list.append(f"Removed {dup_count} duplicate dates")
            else:
                errors.append(f"Found {dup_count} duplicate dates")

    is_valid = len(errors) == 0
    return ValidationResult(is_valid, errors, warnings_list, df if fix_issues else None)


# ============================================================================
# SLIPPAGE MODELS
# ============================================================================

class SlippageModel:
    """Base class for slippage models."""

    def calculate_slippage(
        self,
        price: float,
        quantity: int,
        side: str,  # 'buy' or 'sell'
        volume: float = None,
        volatility: float = None
    ) -> float:
        """Return the execution price after slippage."""
        raise NotImplementedError


class FixedSlippage(SlippageModel):
    """Fixed percentage slippage."""

    def __init__(self, slippage_pct: float = 0.001):
        self.slippage_pct = slippage_pct

    def calculate_slippage(self, price, quantity, side, volume=None, volatility=None):
        if side == 'buy':
            return price * (1 + self.slippage_pct)
        return price * (1 - self.slippage_pct)


class RandomSlippage(SlippageModel):
    """Random slippage within a range."""

    def __init__(self, min_pct: float = -0.001, max_pct: float = 0.001):
        self.min_pct = min_pct
        self.max_pct = max_pct

    def calculate_slippage(self, price, quantity, side, volume=None, volatility=None):
        slippage = np.random.uniform(self.min_pct, self.max_pct)
        if side == 'buy':
            slippage = abs(slippage)  # Always adverse for buys
        else:
            slippage = -abs(slippage)  # Always adverse for sells
        return price * (1 + slippage)


class VolumeBasedSlippage(SlippageModel):
    """
    Slippage based on order size relative to volume.

    Larger orders relative to volume have more slippage.
    """

    def __init__(
        self,
        base_slippage: float = 0.0001,
        volume_impact: float = 0.1,
        max_slippage: float = 0.02
    ):
        self.base_slippage = base_slippage
        self.volume_impact = volume_impact
        self.max_slippage = max_slippage

    def calculate_slippage(self, price, quantity, side, volume=None, volatility=None):
        if volume is None or volume <= 0:
            volume = quantity * 100  # Assume we're 1% of volume

        # Order size as fraction of volume
        participation = quantity / volume

        # Slippage increases with participation rate
        slippage = self.base_slippage + (participation * self.volume_impact)
        slippage = min(slippage, self.max_slippage)

        if side == 'buy':
            return price * (1 + slippage)
        return price * (1 - slippage)


class VolatilityAdjustedSlippage(SlippageModel):
    """
    Slippage adjusted for volatility.

    Higher volatility = more slippage.
    """

    def __init__(
        self,
        base_slippage: float = 0.0005,
        volatility_multiplier: float = 2.0,
        max_slippage: float = 0.03
    ):
        self.base_slippage = base_slippage
        self.volatility_multiplier = volatility_multiplier
        self.max_slippage = max_slippage

    def calculate_slippage(self, price, quantity, side, volume=None, volatility=None):
        if volatility is None:
            volatility = 0.02  # Default 2% daily volatility

        # Slippage scales with volatility
        slippage = self.base_slippage + (volatility * self.volatility_multiplier / 100)
        slippage = min(slippage, self.max_slippage)

        if side == 'buy':
            return price * (1 + slippage)
        return price * (1 - slippage)


# ============================================================================
# PARAMETER OPTIMIZATION
# ============================================================================

@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    best_params: Dict[str, Any]
    best_metric: float
    all_results: List[Dict[str, Any]]
    metric_name: str


def grid_search(
    data: pd.DataFrame,
    strategy_name: str,
    param_grid: Dict[str, List[Any]],
    metric: str = 'sharpe_ratio',
    initial_capital: float = 10000,
    transaction_cost: float = 0.001,
    n_jobs: int = 1,
    verbose: bool = True
) -> OptimizationResult:
    """
    Grid search over strategy parameters.

    Args:
        data: OHLCV DataFrame
        strategy_name: Name of strategy to optimize
        param_grid: Dict mapping param names to lists of values
        metric: Metric to optimize ('sharpe_ratio', 'total_return', 'sortino_ratio', 'max_drawdown')
        initial_capital: Starting capital
        transaction_cost: Transaction cost fraction
        n_jobs: Number of parallel jobs (1 = sequential)
        verbose: Print progress

    Returns:
        OptimizationResult with best params and all results
    """
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    if verbose:
        logger.info(f"Testing {len(combinations)} parameter combinations...")

    results = []

    def evaluate_params(params_tuple):
        params = dict(zip(param_names, params_tuple))
        try:
            bt = VectorizedBacktester(data.copy(), initial_capital, transaction_cost)
            portfolio_df = bt.run(strategy_name, params)

            if portfolio_df.empty:
                return None

            metrics = bt.get_metrics(portfolio_df)

            # Calculate the target metric
            if metric == 'sharpe_ratio':
                metric_value = metrics.get('sharpe_ratio', -999)
            elif metric == 'total_return':
                metric_value = metrics.get('total_return', -999)
            elif metric == 'sortino_ratio':
                metric_value = metrics.get('sortino_ratio', -999)
            elif metric == 'max_drawdown':
                # Negate so higher is better
                metric_value = -metrics.get('max_drawdown', 1)
            else:
                metric_value = metrics.get(metric, -999)

            # Convert metrics to performance dict format for compatibility
            performance = {
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Sortino Ratio': metrics.get('sortino_ratio', 0),
                'Max Drawdown': metrics.get('max_drawdown', 0),
                'Total Return': metrics.get('total_return', 0),
                'Win Rate': metrics.get('win_rate', 0),
                'Profit Factor': metrics.get('profit_factor', 0)
            }

            return {
                'params': params,
                'metric_value': metric_value,
                'performance': performance
            }
        except Exception as e:
            if verbose:
                logger.warning(f"Error with params {params}: {e}")
            return None

    # Run evaluations
    if n_jobs == 1:
        for i, combo in enumerate(combinations):
            result = evaluate_params(combo)
            if result:
                results.append(result)
            if verbose and (i + 1) % 10 == 0:
                logger.info(f"  Completed {i + 1}/{len(combinations)}")
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(evaluate_params, combo): combo for combo in combinations}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

    if not results:
        raise ValueError("No valid results from optimization")

    # Find best
    best = max(results, key=lambda x: x['metric_value'])

    if verbose:
        logger.info(f"Best {metric}: {best['metric_value']:.4f}")
        logger.info(f"Best params: {best['params']}")

    return OptimizationResult(
        best_params=best['params'],
        best_metric=best['metric_value'],
        all_results=results,
        metric_name=metric
    )


# ============================================================================
# WALK-FORWARD ANALYSIS
# ============================================================================

@dataclass
class WalkForwardResult:
    """Result of walk-forward analysis."""
    windows: List[Dict[str, Any]]
    overall_return: float
    overall_sharpe: float
    out_of_sample_returns: List[float]
    in_sample_params: List[Dict[str, Any]]


def walk_forward_analysis(
    data: pd.DataFrame,
    strategy_name: str,
    param_grid: Dict[str, List[Any]],
    train_size: int = 252,  # ~1 year of daily data
    test_size: int = 63,    # ~3 months
    step_size: int = 63,    # Roll forward by 3 months
    metric: str = 'sharpe_ratio',
    initial_capital: float = 10000,
    verbose: bool = True
) -> WalkForwardResult:
    """
    Walk-forward analysis with rolling train/test windows.

    1. Train on [0, train_size] → find best params
    2. Test on [train_size, train_size + test_size]
    3. Roll forward by step_size and repeat

    Args:
        data: OHLCV DataFrame
        strategy_name: Strategy to test
        param_grid: Parameters to optimize
        train_size: Training window size (bars)
        test_size: Testing window size (bars)
        step_size: How much to roll forward each iteration
        metric: Metric to optimize
        initial_capital: Starting capital
        verbose: Print progress

    Returns:
        WalkForwardResult with per-window and overall results
    """
    windows = []
    oos_returns = []
    is_params = []
    cumulative_capital = initial_capital

    n_bars = len(data)
    start_idx = 0
    window_num = 0

    while start_idx + train_size + test_size <= n_bars:
        window_num += 1
        train_end = start_idx + train_size
        test_end = train_end + test_size

        train_data = data.iloc[start_idx:train_end].copy()
        test_data = data.iloc[train_end:test_end].copy()

        if verbose:
            logger.info(f"Window {window_num}: Train [{start_idx}:{train_end}], Test [{train_end}:{test_end}]")

        # Optimize on training data
        try:
            opt_result = grid_search(
                train_data, strategy_name, param_grid,
                metric=metric, initial_capital=cumulative_capital,
                verbose=False
            )
            best_params = opt_result.best_params
            is_metric = opt_result.best_metric
        except Exception as e:
            if verbose:
                logger.warning(f"  Optimization failed: {e}")
            start_idx += step_size
            continue

        if verbose:
            logger.info(f"  Best in-sample params: {best_params}")
            logger.info(f"  In-sample {metric}: {is_metric:.4f}")

        # Test on out-of-sample data
        try:
            bt = VectorizedBacktester(test_data, cumulative_capital, 0.001)
            portfolio_df = bt.run(strategy_name, best_params)

            if not portfolio_df.empty:
                final_value = portfolio_df['Portfolio_Value'].iloc[-1]
                oos_return = (final_value - cumulative_capital) / cumulative_capital
                cumulative_capital = final_value

                metrics = bt.get_metrics(portfolio_df)
                oos_sharpe = metrics.get('sharpe_ratio', 0)
            else:
                oos_return = 0
                oos_sharpe = 0
        except Exception as e:
            if verbose:
                logger.warning(f"  Test failed: {e}")
            oos_return = 0
            oos_sharpe = 0

        if verbose:
            logger.info(f"  Out-of-sample return: {oos_return:.2%}")

        windows.append({
            'window': window_num,
            'train_start': start_idx,
            'train_end': train_end,
            'test_start': train_end,
            'test_end': test_end,
            'best_params': best_params,
            'is_metric': is_metric,
            'oos_return': oos_return,
            'oos_sharpe': oos_sharpe
        })

        oos_returns.append(oos_return)
        is_params.append(best_params)

        start_idx += step_size

    # Calculate overall metrics
    overall_return = (cumulative_capital - initial_capital) / initial_capital
    avg_oos_sharpe = np.mean([w['oos_sharpe'] for w in windows]) if windows else 0

    if verbose:
        logger.info("=== Walk-Forward Summary ===")
        logger.info(f"Windows tested: {len(windows)}")
        logger.info(f"Overall return: {overall_return:.2%}")
        logger.info(f"Average OOS Sharpe: {avg_oos_sharpe:.4f}")

    return WalkForwardResult(
        windows=windows,
        overall_return=overall_return,
        overall_sharpe=avg_oos_sharpe,
        out_of_sample_returns=oos_returns,
        in_sample_params=is_params
    )


# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

@dataclass
class MonteCarloResult:
    """Result of Monte Carlo simulation."""
    mean_return: float
    median_return: float
    std_return: float
    percentiles: Dict[int, float]
    max_drawdowns: List[float]
    sharpe_ratios: List[float]
    final_values: List[float]
    confidence_interval_95: Tuple[float, float]


def monte_carlo_simulation(
    trades: List[Dict],
    initial_capital: float = 10000,
    n_simulations: int = 1000,
    seed: int = None
) -> MonteCarloResult:
    """
    Monte Carlo simulation by randomizing trade order.

    Takes a list of historical trades and shuffles their order
    to estimate the distribution of possible outcomes.

    Args:
        trades: List of trade dicts with 'pnl' key
        initial_capital: Starting capital
        n_simulations: Number of simulations to run
        seed: Random seed for reproducibility

    Returns:
        MonteCarloResult with distribution statistics
    """
    if seed is not None:
        np.random.seed(seed)

    if not trades:
        raise ValueError("No trades provided for Monte Carlo simulation")

    # Extract P&L from trades
    trade_pnls = [t.get('pnl', 0) for t in trades if 'pnl' in t]
    if not trade_pnls:
        # Try to calculate from price and quantity
        trade_pnls = []
        for t in trades:
            action = t.get('Action', '').upper()
            price = t.get('Price', 0)
            qty = t.get('Quantity', 0)
            if action == 'SELL':
                # Approximate P&L (would need entry price for accuracy)
                trade_pnls.append(price * qty * 0.01)  # Assume 1% profit
            elif action == 'BUY':
                trade_pnls.append(-price * qty * 0.01)  # Buying is cost

    if not trade_pnls:
        raise ValueError("Could not extract P&L from trades")

    final_values = []
    max_drawdowns = []
    sharpe_ratios = []

    for _ in range(n_simulations):
        # Shuffle trade order
        shuffled = np.random.permutation(trade_pnls)

        # Calculate equity curve
        equity = [initial_capital]
        for pnl in shuffled:
            equity.append(equity[-1] + pnl)

        equity = np.array(equity)
        final_values.append(equity[-1])

        # Calculate max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdowns.append(np.max(drawdown))

        # Calculate Sharpe (simplified)
        returns = np.diff(equity) / equity[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0
        sharpe_ratios.append(sharpe)

    final_values = np.array(final_values)
    returns = (final_values - initial_capital) / initial_capital

    return MonteCarloResult(
        mean_return=np.mean(returns),
        median_return=np.median(returns),
        std_return=np.std(returns),
        percentiles={
            5: np.percentile(returns, 5),
            25: np.percentile(returns, 25),
            50: np.percentile(returns, 50),
            75: np.percentile(returns, 75),
            95: np.percentile(returns, 95)
        },
        max_drawdowns=max_drawdowns,
        sharpe_ratios=sharpe_ratios,
        final_values=final_values.tolist(),
        confidence_interval_95=(np.percentile(returns, 2.5), np.percentile(returns, 97.5))
    )


# ============================================================================
# BENCHMARK COMPARISON
# ============================================================================

@dataclass
class BenchmarkComparison:
    """Comparison of strategy vs benchmark."""
    strategy_return: float
    benchmark_return: float
    excess_return: float
    strategy_sharpe: float
    benchmark_sharpe: float
    beta: float
    alpha: float
    information_ratio: float
    tracking_error: float
    up_capture: float
    down_capture: float


def compare_to_benchmark(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02
) -> BenchmarkComparison:
    """
    Compare strategy performance to a benchmark.

    Args:
        strategy_returns: Series of strategy returns
        benchmark_returns: Series of benchmark returns
        risk_free_rate: Annualized risk-free rate

    Returns:
        BenchmarkComparison with various metrics
    """
    # Align the series
    aligned = pd.DataFrame({
        'strategy': strategy_returns,
        'benchmark': benchmark_returns
    }).dropna()

    strat = aligned['strategy']
    bench = aligned['benchmark']

    # Basic returns
    strategy_total = (1 + strat).prod() - 1
    benchmark_total = (1 + bench).prod() - 1
    excess_return = strategy_total - benchmark_total

    # Sharpe ratios (annualized)
    daily_rf = risk_free_rate / 252
    strat_sharpe = (strat.mean() - daily_rf) / strat.std() * np.sqrt(252) if strat.std() > 0 else 0
    bench_sharpe = (bench.mean() - daily_rf) / bench.std() * np.sqrt(252) if bench.std() > 0 else 0

    # Beta and Alpha
    cov = np.cov(strat, bench)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0
    alpha = (strat.mean() - daily_rf - beta * (bench.mean() - daily_rf)) * 252

    # Information Ratio
    active_returns = strat - bench
    tracking_error = active_returns.std() * np.sqrt(252)
    information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() > 0 else 0

    # Up/Down capture
    up_periods = bench > 0
    down_periods = bench < 0

    if up_periods.sum() > 0:
        up_capture = strat[up_periods].mean() / bench[up_periods].mean()
    else:
        up_capture = 0

    if down_periods.sum() > 0:
        down_capture = strat[down_periods].mean() / bench[down_periods].mean()
    else:
        down_capture = 0

    return BenchmarkComparison(
        strategy_return=strategy_total,
        benchmark_return=benchmark_total,
        excess_return=excess_return,
        strategy_sharpe=strat_sharpe,
        benchmark_sharpe=bench_sharpe,
        beta=beta,
        alpha=alpha,
        information_ratio=information_ratio,
        tracking_error=tracking_error,
        up_capture=up_capture,
        down_capture=down_capture
    )


# ============================================================================
# VECTORIZED BACKTESTER
# ============================================================================

class VectorizedBacktester:
    """
    High-performance vectorized backtester.

    Uses vectorized signal generation for 10-100x speedup on large datasets.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 10000,
        transaction_cost: float = 0.001,
        slippage_model: SlippageModel = None
    ):
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage_model or FixedSlippage(0.0005)

        # Validate and prepare data
        result = validate_ohlcv_data(self.data, fix_issues=True)
        if not result.is_valid:
            raise ValueError(f"Invalid data: {result.errors}")
        if result.cleaned_data is not None:
            self.data = result.cleaned_data

    def run(
        self,
        strategy_name: str,
        strategy_params: Dict = None,
        position_sizing: str = 'fixed',  # 'fixed', 'risk_parity', 'volatility_scaled'
        position_size: float = 0.1,  # Fraction of capital per trade
        stop_loss_atr: float = 2.0,
        take_profit_atr: float = 3.0
    ) -> pd.DataFrame:
        """
        Run vectorized backtest.

        Args:
            strategy_name: Name of strategy
            strategy_params: Strategy parameters
            position_sizing: Position sizing method
            position_size: Base position size as fraction of capital
            stop_loss_atr: Stop loss in ATR multiples
            take_profit_atr: Take profit in ATR multiples

        Returns:
            DataFrame with backtest results
        """
        if strategy_params is None:
            strategy_params = {}

        # Load strategy and generate signals
        strategy = load_strategy(strategy_name, params=strategy_params)

        # Use vectorized signal generation if available
        if hasattr(strategy, 'generate_signals_vectorized'):
            signals = strategy.generate_signals_vectorized(self.data)
            if signals is not None:
                self.data['Signal'] = signals

        if 'Signal' not in self.data.columns:
            # Fall back to row-by-row
            result = strategy.generate_signal(self.data.copy())
            if isinstance(result, int):
                # Single signal - need to iterate
                signals = []
                for i in range(len(self.data)):
                    sig = strategy.generate_signal(self.data.iloc[:i+1])
                    signals.append(sig)
                self.data['Signal'] = signals
            elif isinstance(result, pd.DataFrame) and 'Signal' in result.columns:
                self.data['Signal'] = result['Signal'].values

        # Calculate ATR for position sizing
        if 'ATR' not in self.data.columns:
            high = self.data['High'].values
            low = self.data['Low'].values
            close = self.data['Close'].values

            tr = np.maximum(high - low,
                           np.maximum(np.abs(high - np.roll(close, 1)),
                                     np.abs(low - np.roll(close, 1))))
            tr[0] = high[0] - low[0]
            self.data['ATR'] = pd.Series(tr).rolling(14).mean().values

        # Vectorized position simulation
        close = self.data['Close'].values
        signals = self.data['Signal'].values
        atr = self.data['ATR'].values
        volume = self.data['Volume'].values if 'Volume' in self.data.columns else np.ones(len(close)) * 1000000

        n = len(close)
        position = np.zeros(n)
        cash = np.zeros(n)
        portfolio_value = np.zeros(n)
        trades = []

        cash[0] = self.initial_capital

        for i in range(1, n):
            cash[i] = cash[i-1]
            position[i] = position[i-1]

            current_price = close[i]
            current_atr = atr[i] if not np.isnan(atr[i]) else close[i] * 0.02
            signal = signals[i]

            # Position sizing
            if position_sizing == 'volatility_scaled':
                target_vol = 0.15  # 15% annual vol target
                daily_vol = current_atr / current_price
                size_multiplier = target_vol / (daily_vol * np.sqrt(252)) if daily_vol > 0 else 1
                size_multiplier = np.clip(size_multiplier, 0.1, 3.0)
            else:
                size_multiplier = 1.0

            trade_value = self.initial_capital * position_size * size_multiplier
            quantity = int(trade_value / current_price)

            # Execute trades
            if signal == 1 and position[i] <= 0 and quantity > 0:
                # Close short if any
                if position[i] < 0:
                    exec_price = self.slippage.calculate_slippage(
                        current_price, abs(int(position[i])), 'buy', volume[i], current_atr/current_price*100
                    )
                    cost = abs(position[i]) * exec_price * (1 + self.transaction_cost)
                    cash[i] -= cost
                    position[i] = 0

                # Open long
                exec_price = self.slippage.calculate_slippage(
                    current_price, quantity, 'buy', volume[i], current_atr/current_price*100
                )
                cost = quantity * exec_price * (1 + self.transaction_cost)
                if cost <= cash[i]:
                    cash[i] -= cost
                    position[i] = quantity
                    trades.append({
                        'idx': i,
                        'action': 'BUY',
                        'price': exec_price,
                        'quantity': quantity,
                        'stop_loss': current_price - stop_loss_atr * current_atr
                    })

            elif signal == -1 and position[i] >= 0 and quantity > 0:
                # Close long if any
                if position[i] > 0:
                    exec_price = self.slippage.calculate_slippage(
                        current_price, int(position[i]), 'sell', volume[i], current_atr/current_price*100
                    )
                    proceeds = position[i] * exec_price * (1 - self.transaction_cost)
                    cash[i] += proceeds
                    position[i] = 0
                    trades.append({
                        'idx': i,
                        'action': 'SELL',
                        'price': exec_price,
                        'quantity': int(position[i-1])
                    })

            # Check stop loss
            if len(trades) > 0 and position[i] > 0:
                last_trade = trades[-1]
                if 'stop_loss' in last_trade and current_price <= last_trade['stop_loss']:
                    exec_price = current_price * (1 - 0.001)  # Extra slippage on stop
                    proceeds = position[i] * exec_price * (1 - self.transaction_cost)
                    cash[i] += proceeds
                    position[i] = 0
                    trades.append({
                        'idx': i,
                        'action': 'STOP_LOSS',
                        'price': exec_price,
                        'quantity': int(position[i-1])
                    })

            portfolio_value[i] = cash[i] + position[i] * current_price

        # Handle initial value
        portfolio_value[0] = self.initial_capital

        # Build result DataFrame
        result = self.data.copy()
        result['Position'] = position
        result['Cash'] = cash
        result['Portfolio_Value'] = portfolio_value
        result['Strategy_Return'] = pd.Series(portfolio_value).pct_change().fillna(0)

        # Calculate drawdown
        peak = np.maximum.accumulate(portfolio_value)
        result['Drawdown'] = (peak - portfolio_value) / peak

        return result

    def get_metrics(self, result: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics from backtest result."""
        returns = result['Strategy_Return'].values
        portfolio_value = result['Portfolio_Value'].values

        total_return = (portfolio_value[-1] - self.initial_capital) / self.initial_capital

        # Sharpe ratio
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0

        # Sortino ratio
        downside = returns[returns < 0]
        if len(downside) > 0 and np.std(downside) > 0:
            sortino = np.mean(returns) / np.std(downside) * np.sqrt(252)
        else:
            sortino = sharpe

        # Max drawdown
        peak = np.maximum.accumulate(portfolio_value)
        drawdown = (peak - portfolio_value) / peak
        max_drawdown = np.max(drawdown)

        # Win rate
        trade_returns = returns[returns != 0]
        if len(trade_returns) > 0:
            win_rate = np.sum(trade_returns > 0) / len(trade_returns)
        else:
            win_rate = 0

        # Profit factor
        gains = np.sum(returns[returns > 0])
        losses = abs(np.sum(returns[returns < 0]))
        profit_factor = gains / losses if losses > 0 else float('inf')

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': len(trade_returns),
            'final_value': portfolio_value[-1]
        }
