"""
Strategy Selector Module

Evaluates all available strategies on a given symbol and selects the best performers.
Supports walk-forward validation to prevent overfitting.

Usage:
    # As a module
    from core.backtest.strategy_selector import StrategySelector

    selector = StrategySelector(data)
    results = selector.select_best_strategies(symbol="AAPL", top_n=3)

    # From CLI
    python -m core.backtest.strategy_selector AAPL --days 365 --top 3
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from core.backtest.backtester import VectorizedBacktester
from core.backtest.optimization import grid_search
from core.backtest.regime_backtest import REGIME_TYPES, RegimeAnalysisResult, RegimeBacktester
from core.backtest.walk_forward import walk_forward_analysis
from strategies.strategy_registry import list_strategies

logger = logging.getLogger(__name__)


# Default parameter grids for each strategy
# Keys match the registry format (lowercase, no underscores)
STRATEGY_PARAM_GRIDS = {
    "sma": {"fast": [5, 10, 15, 20], "slow": [20, 30, 50, 100]},
    "ema": {"fast": [5, 10, 12, 15], "slow": [20, 26, 30, 50]},
    "momentum": {"lookback": [5, 10, 20, 30, 60]},
    "meanreversion": {"window": [10, 14, 20, 30], "threshold": [1.0, 1.5, 2.0, 2.5]},
    "rsi": {"window": [7, 14, 21], "oversold": [20, 25, 30], "overbought": [70, 75, 80]},
    "bollinger": {"window": [10, 20, 30], "num_std": [1.5, 2.0, 2.5]},
    "macd": {"fast": [8, 12, 16], "slow": [20, 26, 30], "signal": [7, 9, 12]},
    "adx": {"window": [10, 14, 20], "threshold": [20, 25, 30]},
    "stochastic": {
        "k_window": [10, 14, 21],
        "d_window": [3, 5, 7],
        "oversold": [15, 20, 25],
        "overbought": [75, 80, 85],
    },
    "donchian": {"window": [10, 20, 30, 55]},
    "breakout": {"window": [10, 20, 30], "threshold": [0.01, 0.02, 0.03]},
    "vwap": {"window": [10, 20, 30]},
    "psar": {"af_start": [0.01, 0.02, 0.03], "af_max": [0.1, 0.2, 0.3]},
    "ichimoku": {"tenkan": [7, 9, 12], "kijun": [22, 26, 30]},
}


@dataclass
class StrategyEvaluation:
    """Result of evaluating a single strategy."""

    strategy_name: str
    best_params: dict[str, Any]
    sharpe_ratio: float
    sortino_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    composite_score: float
    walk_forward_sharpe: float | None = None
    is_overfit: bool = False
    all_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionResult:
    """Result of strategy selection."""

    symbol: str
    top_strategies: list[StrategyEvaluation]
    all_evaluations: list[StrategyEvaluation]
    data_period: str
    num_bars: int
    selection_config: dict[str, Any] = field(default_factory=dict)


class StrategySelector:
    """
    Evaluates and ranks trading strategies for a given symbol.

    Features:
    - Tests all registered strategies with parameter optimization
    - Composite scoring based on multiple metrics
    - Walk-forward validation to detect overfitting
    - Saves results to config files for production use
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
    ):
        """
        Initialize the strategy selector.

        Args:
            data: OHLCV DataFrame with historical data
            initial_capital: Starting capital for backtests
            transaction_cost: Transaction cost as fraction
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

        # Get available strategies (exclude meta-strategies)
        self._available_strategies = self._get_available_strategies()

        logger.info(f"StrategySelector initialized with {len(self.data)} bars")
        logger.info(f"Available strategies: {len(self._available_strategies)}")

    def _get_available_strategies(self) -> list[str]:
        """Get list of available strategy names."""
        # Get strategies from registry, excluding meta-strategies
        strategies = []

        for name in list_strategies():
            # Skip meta/ML strategies that need special handling
            if name in ("combined", "logisticregression"):
                continue
            strategies.append(name)

        return sorted(set(strategies))

    def select_best_strategies(
        self,
        symbol: str,
        top_n: int = 3,
        metric: str = "composite",
        use_walk_forward: bool = True,
        walk_forward_train_size: int = 252,
        walk_forward_test_size: int = 63,
        parallel: bool = False,
        n_jobs: int = 4,
        verbose: bool = True,
    ) -> SelectionResult:
        """
        Evaluate all strategies and select the top performers.

        Args:
            symbol: Stock symbol (for logging/config)
            top_n: Number of top strategies to return
            metric: Ranking metric ('composite', 'sharpe_ratio', 'sortino_ratio', 'total_return')
            use_walk_forward: Whether to use walk-forward validation
            walk_forward_train_size: Training window for walk-forward
            walk_forward_test_size: Test window for walk-forward
            parallel: Whether to run evaluations in parallel
            n_jobs: Number of parallel jobs
            verbose: Print progress

        Returns:
            SelectionResult with top strategies and all evaluations
        """
        if verbose:
            logger.info(f"Evaluating {len(self._available_strategies)} strategies for {symbol}")

        evaluations = []

        for i, strategy_name in enumerate(self._available_strategies):
            if verbose:
                logger.info(f"[{i + 1}/{len(self._available_strategies)}] Evaluating {strategy_name}...")

            try:
                evaluation = self._evaluate_strategy(
                    strategy_name=strategy_name,
                    use_walk_forward=use_walk_forward,
                    walk_forward_train_size=walk_forward_train_size,
                    walk_forward_test_size=walk_forward_test_size,
                    verbose=verbose,
                )

                if evaluation is not None:
                    evaluations.append(evaluation)
                    if verbose:
                        logger.info(
                            f"  -> Sharpe: {evaluation.sharpe_ratio:.2f}, "
                            f"Return: {evaluation.total_return:.1%}, "
                            f"Score: {evaluation.composite_score:.2f}"
                        )

            except Exception as e:
                logger.warning(f"Error evaluating {strategy_name}: {e}")
                continue

        if not evaluations:
            raise ValueError("No strategies could be evaluated successfully")

        # Sort by metric
        if metric == "composite":
            evaluations.sort(key=lambda x: x.composite_score, reverse=True)
        elif metric == "sharpe_ratio":
            evaluations.sort(key=lambda x: x.sharpe_ratio, reverse=True)
        elif metric == "sortino_ratio":
            evaluations.sort(key=lambda x: x.sortino_ratio, reverse=True)
        elif metric == "total_return":
            evaluations.sort(key=lambda x: x.total_return, reverse=True)
        else:
            evaluations.sort(key=lambda x: x.composite_score, reverse=True)

        # Get date range
        data_period = f"{len(self.data)} bars"
        try:
            if "Date" in self.data.columns:
                dates = self.data["Date"]
                # Check if dates are Unix timestamps (milliseconds)
                if pd.api.types.is_numeric_dtype(dates):
                    # Convert from milliseconds to datetime
                    dates = pd.to_datetime(dates, unit="ms")
                else:
                    dates = pd.to_datetime(dates)
                data_period = f"{dates.min().date()} to {dates.max().date()}"
            elif isinstance(self.data.index, pd.DatetimeIndex):
                data_period = f"{self.data.index.min().date()} to {self.data.index.max().date()}"
        except Exception:
            pass  # Keep default "{N} bars" format

        result = SelectionResult(
            symbol=symbol,
            top_strategies=evaluations[:top_n],
            all_evaluations=evaluations,
            data_period=data_period,
            num_bars=len(self.data),
            selection_config={
                "metric": metric,
                "use_walk_forward": use_walk_forward,
                "initial_capital": self.initial_capital,
                "transaction_cost": self.transaction_cost,
            },
        )

        if verbose:
            self._print_results(result)

        return result

    def select_best_strategies_by_regime(
        self,
        symbol: str,
        top_n: int = 3,
        metric: str = "sharpe_ratio",
        verbose: bool = True,
    ) -> RegimeAnalysisResult:
        """
        Evaluate all strategies with regime-aware backtesting.

        This uses ATR-based regime detection to test strategies separately
        during low/normal/high volatility periods. This better matches live
        trading where strategies are selected based on current regime.

        Args:
            symbol: Stock symbol (for logging/config)
            top_n: Number of top strategies to return (not used, all regimes tested)
            metric: Ranking metric (sharpe_ratio, total_return, win_rate)
            verbose: Print progress

        Returns:
            RegimeAnalysisResult with best strategies per regime
        """
        if verbose:
            logger.info(f"Running regime-aware strategy selection for {symbol}")
            logger.info(f"Testing {len(self._available_strategies)} strategies across {len(REGIME_TYPES)} regimes")

        # Use RegimeBacktester for regime-aware evaluation
        tester = RegimeBacktester(
            data=self.data,
            symbol=symbol,
            initial_capital=self.initial_capital,
            transaction_cost=self.transaction_cost,
            strategies=self._available_strategies,
        )

        # Run regime analysis
        result = tester.run_regime_analysis(
            metric=metric,
            verbose=verbose,
        )

        if verbose:
            tester.print_results(result)

        return result

    def _evaluate_strategy(
        self,
        strategy_name: str,
        use_walk_forward: bool = True,
        walk_forward_train_size: int = 252,
        walk_forward_test_size: int = 63,
        verbose: bool = False,
    ) -> StrategyEvaluation | None:
        """Evaluate a single strategy with optional walk-forward validation."""

        # Get parameter grid for this strategy
        param_grid = STRATEGY_PARAM_GRIDS.get(strategy_name, {})

        if not param_grid:
            # Use default parameters
            param_grid = {}

        # Run grid search optimization
        try:
            if param_grid:
                opt_result = grid_search(
                    data=self.data,
                    strategy_name=strategy_name,
                    param_grid=param_grid,
                    metric="sharpe_ratio",
                    initial_capital=self.initial_capital,
                    transaction_cost=self.transaction_cost,
                    verbose=False,
                )
                best_params = opt_result.best_params
            else:
                best_params = {}

            # Run final backtest with best params
            bt = VectorizedBacktester(self.data.copy(), self.initial_capital, self.transaction_cost)
            portfolio_df = bt.run(strategy_name, best_params)

            if portfolio_df.empty:
                return None

            metrics = bt.get_metrics(portfolio_df)

            # Calculate composite score
            composite = self._calculate_composite_score(metrics)

            # Walk-forward validation (optional)
            wf_sharpe = None
            is_overfit = False

            if (
                use_walk_forward
                and param_grid
                and len(self.data) > (walk_forward_train_size + walk_forward_test_size * 2)
            ):
                try:
                    wf_result = walk_forward_analysis(
                        data=self.data,
                        strategy_name=strategy_name,
                        param_grid=param_grid,
                        train_size=walk_forward_train_size,
                        test_size=walk_forward_test_size,
                        metric="sharpe_ratio",
                        initial_capital=self.initial_capital,
                        verbose=False,
                    )

                    if wf_result and "overall" in wf_result:
                        wf_sharpe = wf_result["overall"].get("oos_sharpe", 0)
                        is_sharpe = metrics.get("sharpe_ratio", 0)
                        # Overfit if walk-forward is significantly worse
                        is_overfit = wf_sharpe < (is_sharpe * 0.5) if is_sharpe > 0 else False

                except Exception as e:
                    logger.debug(f"Walk-forward failed for {strategy_name}: {e}")

            return StrategyEvaluation(
                strategy_name=strategy_name,
                best_params=best_params,
                sharpe_ratio=metrics.get("sharpe_ratio", 0),
                sortino_ratio=metrics.get("sortino_ratio", 0),
                total_return=metrics.get("total_return", 0),
                max_drawdown=metrics.get("max_drawdown", 0),
                win_rate=metrics.get("win_rate", 0),
                profit_factor=metrics.get("profit_factor", 0),
                num_trades=metrics.get("num_trades", 0),
                composite_score=composite,
                walk_forward_sharpe=wf_sharpe,
                is_overfit=is_overfit,
                all_metrics=metrics,
            )

        except Exception as e:
            logger.debug(f"Strategy {strategy_name} evaluation failed: {e}")
            return None

    def _calculate_composite_score(self, metrics: dict[str, Any], weights: dict[str, float] = None) -> float:
        """
        Calculate composite score from multiple metrics.

        Default weights:
        - Sharpe Ratio: 35%
        - Sortino Ratio: 25%
        - Total Return: 20%
        - Max Drawdown: 15% (inverted - lower is better)
        - Win Rate: 5%
        """
        if weights is None:
            weights = {
                "sharpe_ratio": 0.35,
                "sortino_ratio": 0.25,
                "total_return": 0.20,
                "max_drawdown": 0.15,
                "win_rate": 0.05,
            }

        # Normalize metrics to 0-100 scale
        sharpe = min(max(metrics.get("sharpe_ratio", 0), -3), 3)  # Clip to [-3, 3]
        sharpe_norm = (sharpe + 3) / 6 * 100  # 0-100

        sortino = min(max(metrics.get("sortino_ratio", 0), -3), 5)  # Clip
        sortino_norm = (sortino + 3) / 8 * 100  # 0-100

        total_return = metrics.get("total_return", 0)
        return_norm = min(max((total_return + 0.5) / 1.5 * 100, 0), 100)  # -50% to +100%

        max_dd = abs(metrics.get("max_drawdown", 0))
        dd_norm = max(100 - (max_dd * 200), 0)  # Lower DD = higher score

        win_rate = metrics.get("win_rate", 0)
        wr_norm = win_rate * 100  # Already 0-1

        score = (
            weights["sharpe_ratio"] * sharpe_norm
            + weights["sortino_ratio"] * sortino_norm
            + weights["total_return"] * return_norm
            + weights["max_drawdown"] * dd_norm
            + weights["win_rate"] * wr_norm
        )

        return score

    def _print_results(self, result: SelectionResult) -> None:
        """Print results summary."""
        print("\n" + "=" * 70)
        print(f"  STRATEGY SELECTION RESULTS - {result.symbol}")
        print("=" * 70)
        print(f"  Data Period: {result.data_period}")
        print(f"  Bars Analyzed: {result.num_bars}")
        print(f"  Strategies Evaluated: {len(result.all_evaluations)}")
        print("=" * 70)

        print("\n  TOP STRATEGIES:\n")

        for i, eval in enumerate(result.top_strategies, 1):
            overfit_warning = " [!OVERFIT]" if eval.is_overfit else ""
            print(f"  #{i} {eval.strategy_name}{overfit_warning}")
            print(f"      Parameters: {eval.best_params}")
            print(f"      Sharpe: {eval.sharpe_ratio:.2f} | Sortino: {eval.sortino_ratio:.2f}")
            print(f"      Return: {eval.total_return:.1%} | Max DD: {eval.max_drawdown:.1%}")
            print(f"      Win Rate: {eval.win_rate:.1%} | Profit Factor: {eval.profit_factor:.2f}")
            print(f"      Composite Score: {eval.composite_score:.1f}")
            if eval.walk_forward_sharpe is not None:
                print(f"      Walk-Forward Sharpe: {eval.walk_forward_sharpe:.2f}")
            print()

        print("=" * 70)

    def save_to_config(
        self, result: SelectionResult, config_dir: str = None, regime: str = "normal"
    ) -> tuple[Path, Path]:
        """
        Save selection results to strategy routing config files.

        Args:
            result: Selection result to save
            config_dir: Config directory (default: config/)
            regime: Market regime to save for ('low_volatility', 'normal', 'high_volatility')

        Returns:
            Tuple of (routing_path, params_path)
        """
        if config_dir is None:
            config_dir = Path(__file__).resolve().parents[2] / "config"
        else:
            config_dir = Path(config_dir)

        routing_path = config_dir / "strategy_routing.json"
        params_path = config_dir / "strategy_params.json"

        # Load existing configs
        routing = {}
        params = {}

        if routing_path.exists():
            with open(routing_path) as f:
                routing = json.load(f)

        if params_path.exists():
            with open(params_path) as f:
                params = json.load(f)

        symbol = result.symbol

        # Initialize symbol entries if needed
        if symbol not in routing:
            routing[symbol] = {}
        if symbol not in params:
            params[symbol] = {}

        # Save top strategy for the given regime
        if result.top_strategies:
            top = result.top_strategies[0]
            routing[symbol][regime] = top.strategy_name
            params[symbol][regime] = {"params": top.best_params}

            # Also save ranked list
            routing[symbol][f"{regime}_ranked"] = [e.strategy_name for e in result.top_strategies]

        # Write configs
        with open(routing_path, "w") as f:
            json.dump(routing, f, indent=2)

        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)

        logger.info(f"Saved routing config to {routing_path}")
        logger.info(f"Saved params config to {params_path}")

        return routing_path, params_path

    def save_regime_result_to_config(
        self,
        result: RegimeAnalysisResult,
        config_dir: str = None,
        add_timeframes: bool = False,
        default_timeframe: str = "day",
    ) -> Path:
        """
        Save regime analysis result to strategy routing config.

        Args:
            result: Regime analysis result
            config_dir: Config directory (default: config/)
            add_timeframes: Whether to add timeframe specifications
            default_timeframe: Default timeframe for all regimes

        Returns:
            Path to routing config file
        """
        if config_dir is None:
            config_dir = Path(__file__).resolve().parents[2] / "config"
        else:
            config_dir = Path(config_dir)

        routing_path = config_dir / "strategy_routing.json"

        # Load existing config
        routing = {}
        if routing_path.exists():
            with open(routing_path) as f:
                routing = json.load(f)

        symbol = result.symbol

        # Initialize symbol entry if needed
        if symbol not in routing:
            routing[symbol] = {}

        # Save best strategy for each regime
        for regime in REGIME_TYPES:
            best_strategy = result.best_strategies.get(regime, "momentum")

            if add_timeframes:
                routing[symbol][regime] = {"strategy": best_strategy, "timeframe": default_timeframe}
            else:
                routing[symbol][regime] = best_strategy

        # Add default (use normal regime)
        if "normal" in result.best_strategies:
            if add_timeframes:
                routing[symbol]["default"] = {
                    "strategy": result.best_strategies["normal"],
                    "timeframe": default_timeframe,
                }
            else:
                routing[symbol]["default"] = result.best_strategies["normal"]

        # Add use_hybrid flag if it doesn't exist
        if "use_hybrid" not in routing[symbol]:
            routing[symbol]["use_hybrid"] = False

        # Write config
        with open(routing_path, "w") as f:
            json.dump(routing, f, indent=2)

        logger.info(f"Saved regime-aware routing config to {routing_path}")

        return routing_path


def select_best_strategies(symbol: str, data: pd.DataFrame, top_n: int = 3, **kwargs) -> SelectionResult:
    """
    Convenience function to select best strategies for a symbol.

    Args:
        symbol: Stock symbol
        data: OHLCV DataFrame
        top_n: Number of top strategies to return
        **kwargs: Additional arguments passed to StrategySelector.select_best_strategies

    Returns:
        SelectionResult with top strategies
    """
    selector = StrategySelector(data)
    return selector.select_best_strategies(symbol=symbol, top_n=top_n, **kwargs)


# CLI Support
if __name__ == "__main__":
    import argparse
    import sys

    # Add project root to path
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    from core.unified_data_pipeline import UnifiedDataPipeline

    parser = argparse.ArgumentParser(description="Select best trading strategies for a given symbol")
    parser.add_argument("symbol", type=str, help="Stock symbol (e.g., AAPL)")
    parser.add_argument("--days", type=int, default=365, help="Days of historical data")
    parser.add_argument("--top", type=int, default=3, help="Number of top strategies")
    parser.add_argument(
        "--metric",
        type=str,
        default="composite",
        choices=["composite", "sharpe_ratio", "sortino_ratio", "total_return"],
        help="Ranking metric",
    )
    parser.add_argument("--no-walk-forward", action="store_true", help="Disable walk-forward validation")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument("--save", action="store_true", help="Save results to config files")
    parser.add_argument(
        "--regime",
        type=str,
        default="normal",
        choices=["low_volatility", "normal", "high_volatility"],
        help="Market regime to save for (legacy mode only)",
    )
    parser.add_argument(
        "--regime-aware", action="store_true", help="Use regime-aware backtesting (tests strategies per regime)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(message)s")

    print(f"\nLoading {args.days} days of data for {args.symbol}...")

    # Load data using the pipeline
    try:
        pipeline = UnifiedDataPipeline()
        data = pipeline.load_symbol_data(args.symbol)

        if data is None or data.empty:
            # Try to fetch if not available
            print("No cached data, fetching from source...")
            pipeline.update_symbols([args.symbol], days=args.days)
            data = pipeline.load_symbol_data(args.symbol)

        if data is None or data.empty:
            print(f"Error: Could not load data for {args.symbol}")
            sys.exit(1)

        # Limit to requested days
        if len(data) > args.days:
            data = data.tail(args.days).reset_index(drop=True)

        print(f"Loaded {len(data)} bars")

    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nTrying to load from raw data files...")

        # Fallback to raw data
        from core.config_loader import get_raw_data_path

        data_dir = get_raw_data_path()

        data_file = data_dir / f"{args.symbol}.csv"
        if data_file.exists():
            data = pd.read_csv(data_file)
            if len(data) > args.days:
                data = data.tail(args.days).reset_index(drop=True)
            print(f"Loaded {len(data)} bars from {data_file}")
        else:
            print(f"Error: No data file found at {data_file}")
            sys.exit(1)

    # Run strategy selection
    selector = StrategySelector(data, initial_capital=args.capital)

    if args.regime_aware:
        # Use regime-aware backtesting
        result = selector.select_best_strategies_by_regime(
            symbol=args.symbol, top_n=args.top, metric=args.metric, verbose=True
        )

        # Save to config if requested
        if args.save:
            routing_path = selector.save_regime_result_to_config(result, add_timeframes=True, default_timeframe="day")
            print("\nSaved to:")
            print(f"  {routing_path}")
    else:
        # Use legacy overall backtesting
        result = selector.select_best_strategies(
            symbol=args.symbol,
            top_n=args.top,
            metric=args.metric,
            use_walk_forward=not args.no_walk_forward,
            verbose=True,
        )

        # Save to config if requested
        if args.save:
            routing_path, params_path = selector.save_to_config(result, regime=args.regime)
            print("\nSaved to:")
            print(f"  {routing_path}")
            print(f"  {params_path}")

    print("\nDone!")
