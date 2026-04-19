"""
Regime-Aware Backtester - Test strategies segmented by market regime.

This backtester mirrors the live system's regime detection and strategy routing:
1. Detects regimes using ATR-based classification (same as live system)
2. Tests each strategy's performance within each regime
3. Outputs optimal strategy per (symbol, regime) for strategy_routing.json

Usage:
    from core.backtest.regime_backtest import RegimeBacktester

    tester = RegimeBacktester(data, symbol="AAPL")
    results = tester.run_regime_analysis()

    # Get optimal routing config
    routing = tester.generate_routing_config()
    # {
    #     "AAPL": {
    #         "low_volatility": "sma",
    #         "normal": "momentum",
    #         "high_volatility": "meanreversion"
    #     }
    # }
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.backtest.validation import validate_ohlcv_data
from strategies.strategy_registry import load_strategy

logger = logging.getLogger(__name__)

# Regime types matching live system
REGIME_TYPES = ["low_volatility", "normal", "high_volatility"]

# Default strategies to test
DEFAULT_STRATEGIES = [
    "sma",
    "ema",
    "macd",
    "rsi",
    "bollinger",
    "momentum",
    "meanreversion",
    "stochastic",
    "adx",
    "donchian",
]


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate ATR series."""
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]

    atr = np.zeros(len(tr))
    atr[:period] = np.nan

    for i in range(period, len(tr)):
        atr[i] = np.mean(tr[i - period + 1 : i + 1])

    return atr


def classify_regime(atr: float, atr_history: list[float]) -> str:
    """
    Classify market regime based on ATR relative to historical values.
    Matches the live system's classification logic.
    """
    if atr is None or len(atr_history) < 20:
        return "normal"

    recent_atr = atr_history[-50:] if len(atr_history) >= 50 else atr_history
    atr_mean = np.mean(recent_atr)
    atr_std = np.std(recent_atr) if len(recent_atr) > 1 else 0.0

    high_threshold = atr_mean + atr_std
    low_threshold = atr_mean - 0.5 * atr_std

    if atr > high_threshold:
        return "high_volatility"
    elif atr < low_threshold:
        return "low_volatility"
    else:
        return "normal"


@dataclass
class RegimeStats:
    """Statistics for a single regime."""

    regime: str
    bar_count: int
    pct_of_total: float
    avg_atr: float
    avg_return: float  # Buy-and-hold return during this regime


@dataclass
class StrategyRegimeResult:
    """Result of testing a strategy within a specific regime."""

    strategy: str
    regime: str
    total_return: float
    sharpe_ratio: float
    win_rate: float
    num_trades: int
    avg_trade_pnl: float
    max_drawdown: float
    profit_factor: float


@dataclass
class RegimeAnalysisResult:
    """Complete result of regime analysis."""

    symbol: str
    data_period: str
    total_bars: int

    # Regime distribution
    regime_stats: dict[str, RegimeStats]

    # Strategy performance by regime
    strategy_results: dict[str, dict[str, StrategyRegimeResult]]  # regime -> strategy -> result

    # Best strategy per regime
    best_strategies: dict[str, str]  # regime -> best strategy name

    # Optimal routing config (ready for strategy_routing.json)
    routing_config: dict[str, Any]


class RegimeBacktester:
    """
    Backtest strategies segmented by market regime.

    This replicates the live system's behavior:
    - Detect regime at each bar using ATR classification
    - Track which strategy would perform best in each regime
    - Generate optimal routing configuration
    """

    def __init__(
        self,
        data: pd.DataFrame,
        symbol: str = "SYMBOL",
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,
        strategies: list[str] | None = None,
    ):
        """
        Initialize the regime backtester.

        Args:
            data: OHLCV DataFrame
            symbol: Symbol name
            initial_capital: Starting capital
            transaction_cost: Transaction cost fraction
            strategies: List of strategies to test (default: common strategies)
        """
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.strategies = strategies or DEFAULT_STRATEGIES

        # Validate and prepare data
        self.data = data.copy()
        result = validate_ohlcv_data(self.data, fix_issues=True)
        if result.cleaned_data is not None:
            self.data = result.cleaned_data

        self._normalize_columns()

        # Calculate ATR and regimes
        self._calculate_regimes()

        logger.info(f"RegimeBacktester initialized: {symbol}, {len(self.data)} bars")

    def _normalize_columns(self) -> None:
        """Normalize column names."""
        column_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        for old, new in column_map.items():
            if old in self.data.columns and new not in self.data.columns:
                self.data = self.data.rename(columns={old: new})

    def _calculate_regimes(self) -> None:
        """Calculate ATR and classify regime for each bar."""
        high = self.data["High"].values
        low = self.data["Low"].values
        close = self.data["Close"].values

        # Calculate ATR
        atr = calculate_atr(high, low, close, period=14)
        self.data["ATR"] = atr

        # Classify regime for each bar (matching live system logic)
        regimes = []
        atr_history = []

        for i in range(len(self.data)):
            current_atr = atr[i]
            if not np.isnan(current_atr):
                atr_history.append(current_atr)

            regime = classify_regime(current_atr if not np.isnan(current_atr) else None, atr_history)
            regimes.append(regime)

        self.data["Regime"] = regimes

    def get_regime_distribution(self) -> dict[str, RegimeStats]:
        """Get statistics about regime distribution in the data."""
        stats = {}
        total = len(self.data)

        for regime in REGIME_TYPES:
            mask = self.data["Regime"] == regime
            regime_data = self.data[mask]
            count = len(regime_data)

            if count > 0:
                avg_atr = regime_data["ATR"].mean()
                # Calculate buy-and-hold return during this regime
                returns = regime_data["Close"].pct_change().fillna(0)
                avg_return = returns.mean() * 252  # Annualized
            else:
                avg_atr = 0
                avg_return = 0

            stats[regime] = RegimeStats(
                regime=regime,
                bar_count=count,
                pct_of_total=count / total * 100 if total > 0 else 0,
                avg_atr=avg_atr,
                avg_return=avg_return,
            )

        return stats

    def _generate_signals(self, strategy_name: str) -> np.ndarray:
        """Generate signals for a strategy."""
        n = len(self.data)
        signals = np.zeros(n, dtype=int)

        try:
            strategy = load_strategy(strategy_name)

            # Try vectorized first
            if hasattr(strategy, "generate_signals_vectorized"):
                vec_signals = strategy.generate_signals_vectorized(self.data)
                if vec_signals is not None:
                    return np.array(vec_signals)

            # Fall back to row-by-row
            for i in range(n):
                try:
                    sig = strategy.generate_signal(self.data.iloc[: i + 1])
                    signals[i] = int(sig) if sig in (-1, 0, 1) else 0
                except Exception:
                    signals[i] = 0

        except Exception as e:
            logger.warning(f"Failed to generate signals for {strategy_name}: {e}")

        return signals

    def _backtest_regime_segment(
        self,
        strategy_name: str,
        regime: str,
        signals: np.ndarray,
    ) -> StrategyRegimeResult:
        """
        Backtest a strategy only during bars matching the specified regime.
        """
        # Get indices where regime matches
        regime_mask = (self.data["Regime"] == regime).values

        close = self.data["Close"].values
        atr = self.data["ATR"].values

        # Simulation state
        cash = self.initial_capital
        position = 0
        entry_price = 0.0

        trades = []
        equity = [self.initial_capital]

        for i in range(1, len(self.data)):
            current_price = close[i]
            current_atr = atr[i] if not np.isnan(atr[i]) else current_price * 0.02
            signal = signals[i]
            in_regime = regime_mask[i]

            # Only trade when in the target regime
            if in_regime:
                # Entry
                if signal == 1 and position == 0:
                    quantity = int((cash * 0.1) / current_price)
                    if quantity > 0:
                        cost = quantity * current_price * (1 + self.transaction_cost)
                        if cost <= cash:
                            cash -= cost
                            position = quantity
                            entry_price = current_price

                # Exit
                elif signal == -1 and position > 0:
                    proceeds = position * current_price * (1 - self.transaction_cost)
                    pnl = proceeds - (position * entry_price)
                    trades.append(pnl)
                    cash += proceeds
                    position = 0
                    entry_price = 0.0

            # Stop loss check (applies even outside regime)
            if position > 0:
                stop_price = entry_price - 2.0 * current_atr
                if close[i] <= stop_price:
                    proceeds = position * stop_price * (1 - self.transaction_cost)
                    pnl = proceeds - (position * entry_price)
                    trades.append(pnl)
                    cash += proceeds
                    position = 0
                    entry_price = 0.0

            portfolio_value = cash + position * current_price
            equity.append(portfolio_value)

        # Close any open position
        if position > 0:
            proceeds = position * close[-1] * (1 - self.transaction_cost)
            pnl = proceeds - (position * entry_price)
            trades.append(pnl)
            cash += proceeds

        final_value = cash
        equity = np.array(equity)

        # Calculate metrics
        total_return = (final_value - self.initial_capital) / self.initial_capital

        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns)]

        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown)

        num_trades = len(trades)
        if num_trades > 0:
            wins = [t for t in trades if t > 0]
            losses = [t for t in trades if t < 0]
            win_rate = len(wins) / num_trades
            avg_trade_pnl = np.mean(trades)
            total_wins = sum(wins) if wins else 0
            total_losses = abs(sum(losses)) if losses else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")
        else:
            win_rate = 0.0
            avg_trade_pnl = 0.0
            profit_factor = 0.0

        return StrategyRegimeResult(
            strategy=strategy_name,
            regime=regime,
            total_return=total_return,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            num_trades=num_trades,
            avg_trade_pnl=avg_trade_pnl,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
        )

    def run_regime_analysis(
        self,
        metric: str = "sharpe_ratio",
        verbose: bool = True,
    ) -> RegimeAnalysisResult:
        """
        Run full regime analysis - test all strategies in all regimes.

        Args:
            metric: Metric for ranking (sharpe_ratio, total_return, win_rate)
            verbose: Print progress

        Returns:
            RegimeAnalysisResult with best strategies per regime
        """
        if verbose:
            print(f"\nAnalyzing {self.symbol} with {len(self.data)} bars...")
            print(f"Strategies to test: {len(self.strategies)}")
            print(f"Regimes: {REGIME_TYPES}")

        # Get regime distribution
        regime_stats = self.get_regime_distribution()

        if verbose:
            print("\nRegime Distribution:")
            for regime, stats in regime_stats.items():
                print(f"  {regime}: {stats.bar_count} bars ({stats.pct_of_total:.1f}%)")

        # Test each strategy
        strategy_results: dict[str, dict[str, StrategyRegimeResult]] = {regime: {} for regime in REGIME_TYPES}

        for i, strategy_name in enumerate(self.strategies):
            if verbose:
                print(f"\n[{i + 1}/{len(self.strategies)}] Testing {strategy_name}...")

            # Generate signals once
            signals = self._generate_signals(strategy_name)

            # Test in each regime
            for regime in REGIME_TYPES:
                result = self._backtest_regime_segment(strategy_name, regime, signals)
                strategy_results[regime][strategy_name] = result

                if verbose:
                    print(
                        f"  {regime}: return={result.total_return:+.1%}, "
                        f"sharpe={result.sharpe_ratio:.2f}, trades={result.num_trades}"
                    )

        # Find best strategy per regime
        best_strategies = {}
        for regime in REGIME_TYPES:
            regime_results = strategy_results[regime]
            if regime_results:
                best = max(regime_results.keys(), key=lambda s: getattr(regime_results[s], metric, 0))
                best_strategies[regime] = best

        # Build routing config
        routing_config = {self.symbol: {regime: best_strategies.get(regime, "momentum") for regime in REGIME_TYPES}}
        routing_config[self.symbol]["default"] = best_strategies.get("normal", "momentum")

        # Get data period
        try:
            if "Date" in self.data.columns:
                dates = pd.to_datetime(self.data["Date"])
                data_period = f"{dates.min().date()} to {dates.max().date()}"
            else:
                data_period = f"{len(self.data)} bars"
        except Exception:
            data_period = f"{len(self.data)} bars"

        return RegimeAnalysisResult(
            symbol=self.symbol,
            data_period=data_period,
            total_bars=len(self.data),
            regime_stats=regime_stats,
            strategy_results=strategy_results,
            best_strategies=best_strategies,
            routing_config=routing_config,
        )

    def print_results(self, result: RegimeAnalysisResult) -> None:
        """Print formatted results."""
        print()
        print("=" * 80)
        print(f"  REGIME-AWARE STRATEGY ANALYSIS - {result.symbol}")
        print("=" * 80)
        print(f"  Data Period: {result.data_period}")
        print(f"  Total Bars: {result.total_bars}")
        print()

        # Regime distribution
        print("  REGIME DISTRIBUTION:")
        print("  " + "-" * 60)
        for regime in REGIME_TYPES:
            stats = result.regime_stats[regime]
            print(f"  {regime:<18} {stats.bar_count:>6} bars ({stats.pct_of_total:>5.1f}%)")
        print()

        # Best strategies per regime
        print("  OPTIMAL STRATEGY PER REGIME:")
        print("  " + "-" * 60)
        for regime in REGIME_TYPES:
            best = result.best_strategies.get(regime, "N/A")
            if best in result.strategy_results[regime]:
                r = result.strategy_results[regime][best]
                print(f"  {regime:<18} -> {best:<15} (Sharpe: {r.sharpe_ratio:>5.2f}, Return: {r.total_return:>+6.1%})")
            else:
                print(f"  {regime:<18} -> {best:<15}")
        print()

        # Detailed results table
        print("  FULL RESULTS BY REGIME:")
        print()

        for regime in REGIME_TYPES:
            print(f"  [{regime.upper()}]")
            print(f"  {'Strategy':<15} {'Return':>10} {'Sharpe':>8} {'Win Rate':>10} {'Trades':>8}")
            print("  " + "-" * 55)

            # Sort by Sharpe
            sorted_strats = sorted(
                result.strategy_results[regime].keys(),
                key=lambda s: result.strategy_results[regime][s].sharpe_ratio,
                reverse=True,
            )

            for strat in sorted_strats[:5]:  # Top 5
                r = result.strategy_results[regime][strat]
                marker = " *" if strat == result.best_strategies.get(regime) else "  "
                print(
                    f"{marker}{strat:<14} {r.total_return:>+9.1%} {r.sharpe_ratio:>8.2f} "
                    f"{r.win_rate:>9.1%} {r.num_trades:>8}"
                )
            print()

        # Routing config
        print("  ROUTING CONFIG (for strategy_routing.json):")
        print("  " + "-" * 60)
        print(f"  {json.dumps(result.routing_config, indent=4)}")
        print()
        print("=" * 80)

    def save_routing_config(
        self,
        result: RegimeAnalysisResult,
        config_path: str | None = None,
        merge: bool = True,
    ) -> Path:
        """
        Save routing config to strategy_routing.json.

        Args:
            result: Analysis result
            config_path: Path to config file (default: config/strategy_routing.json)
            merge: If True, merge with existing config

        Returns:
            Path to saved config
        """
        if config_path is None:
            config_path = Path(__file__).resolve().parents[2] / "config" / "strategy_routing.json"
        else:
            config_path = Path(config_path)

        # Load existing config if merging
        existing = {}
        if merge and config_path.exists():
            with open(config_path) as f:
                existing = json.load(f)

        # Merge
        existing.update(result.routing_config)

        # Save
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(existing, f, indent=2)

        logger.info(f"Saved routing config to {config_path}")
        return config_path


def run_multi_symbol_regime_analysis(
    symbols: list[str],
    days: int = 365,
    strategies: list[str] | None = None,
    metric: str = "sharpe_ratio",
    save_config: bool = False,
    verbose: bool = True,
) -> dict[str, RegimeAnalysisResult]:
    """
    Run regime analysis across multiple symbols.

    Args:
        symbols: List of symbols
        days: Days of data to use
        strategies: Strategies to test
        metric: Ranking metric
        save_config: Save to strategy_routing.json
        verbose: Print progress

    Returns:
        Dict of symbol -> RegimeAnalysisResult
    """
    from core.unified_data_pipeline import UnifiedDataPipeline

    pipeline = UnifiedDataPipeline()
    results = {}
    combined_routing = {}

    for symbol in symbols:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  Processing {symbol}")
            print(f"{'=' * 60}")

        try:
            data = pipeline.get_data(symbol)
            if data is None or data.empty:
                print(f"  No data for {symbol}, skipping...")
                continue

            if len(data) > days:
                data = data.tail(days).reset_index(drop=True)

            tester = RegimeBacktester(
                data=data,
                symbol=symbol,
                strategies=strategies,
            )

            result = tester.run_regime_analysis(metric=metric, verbose=verbose)
            results[symbol] = result

            if verbose:
                tester.print_results(result)

            # Collect routing
            combined_routing.update(result.routing_config)

        except Exception as e:
            print(f"  Error processing {symbol}: {e}")

    # Save combined config
    if save_config and combined_routing:
        config_path = Path(__file__).resolve().parents[2] / "config" / "strategy_routing.json"

        # Load existing
        existing = {}
        if config_path.exists():
            with open(config_path) as f:
                existing = json.load(f)

        existing.update(combined_routing)

        with open(config_path, "w") as f:
            json.dump(existing, f, indent=2)

        print(f"\nSaved routing config to {config_path}")

    return results


# CLI support
if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    parser = argparse.ArgumentParser(description="Regime-aware strategy backtesting")
    parser.add_argument("symbols", nargs="+", help="Symbols to analyze")
    parser.add_argument("-d", "--days", type=int, default=365, help="Days of data")
    parser.add_argument("-s", "--strategies", default=None, help="Comma-separated strategies to test")
    parser.add_argument("-m", "--metric", default="sharpe_ratio", choices=["sharpe_ratio", "total_return", "win_rate"])
    parser.add_argument("--save", action="store_true", help="Save to strategy_routing.json")

    args = parser.parse_args()

    strategies = None
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",")]

    run_multi_symbol_regime_analysis(
        symbols=[s.upper() for s in args.symbols],
        days=args.days,
        strategies=strategies,
        metric=args.metric,
        save_config=args.save,
        verbose=True,
    )
