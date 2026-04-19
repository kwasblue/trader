#!/usr/bin/env python3
"""
Backtest Adaptive Features

Tests and compares:
1. Unstable vs Stable Regime Detection
2. Fixed vs Continuous Position Sizing
3. Combined system performance

Usage:
    # Test single symbol
    python tools/backtest_adaptive_features.py --symbol AAPL --days 750

    # Test all symbols
    python tools/backtest_adaptive_features.py --all --days 750

    # Compare configurations
    python tools/backtest_adaptive_features.py --symbol AAPL --compare
"""

import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

# Add project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.continuous_metrics import ContinuousMetrics
from core.continuous_position_sizer import ContinuousPositionSizer
from core.stable_regime_detector import StableRegimeDetector
from core.unified_data_pipeline import UnifiedDataPipeline
from strategies.strategy_registry import load_strategy, list_strategies


@dataclass
class BacktestMetrics:
    """Backtest performance metrics."""
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    avg_trade_return: float
    volatility: float
    calmar_ratio: float

    # Regime-specific
    regime_switches: int
    avg_bars_per_regime: float

    # Position sizing specific
    avg_position_multiplier: float
    trades_skipped: int

    # Risk-adjusted
    risk_adjusted_return: float


@dataclass
class Trade:
    """Single trade record."""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    strategy: str
    regime: str
    side: str  # BUY or SELL
    entry_price: float
    exit_price: float
    shares: int
    position_size: float
    position_multiplier: float
    pnl: float
    return_pct: float
    held_bars: int


class AdaptiveBacktester:
    """
    Backtester for adaptive features.

    Tests multiple configurations:
    - Unstable regime detection (no stability features)
    - Stable regime detection (with hysteresis, persistence, cooldown, etc.)
    - Fixed position sizing (constant 10%)
    - Continuous adaptive position sizing
    """

    def __init__(self, symbol: str, bars: pd.DataFrame, logger=None):
        """
        Initialize backtester.

        Args:
            symbol: Stock symbol
            bars: Historical OHLCV data
            logger: Optional logger
        """
        self.symbol = symbol
        self.logger = logger or logging.getLogger(__name__)

        # Keep original data with capitalized columns for strategies
        self.bars_with_caps = bars.copy()

        # Add lowercase column aliases for strategies that expect them
        # This ensures compatibility with both capitalized and lowercase expectations
        if 'Open' in self.bars_with_caps.columns:
            self.bars_with_caps['open'] = self.bars_with_caps['Open']
        if 'High' in self.bars_with_caps.columns:
            self.bars_with_caps['high'] = self.bars_with_caps['High']
        if 'Low' in self.bars_with_caps.columns:
            self.bars_with_caps['low'] = self.bars_with_caps['Low']
        if 'Close' in self.bars_with_caps.columns:
            self.bars_with_caps['close'] = self.bars_with_caps['Close']
        if 'Volume' in self.bars_with_caps.columns:
            self.bars_with_caps['volume'] = self.bars_with_caps['Volume']

        # DEBUG: Log column status
        self.logger.info(f"AdaptiveBacktester init: has_Close={'Close' in self.bars_with_caps.columns}, has_close={'close' in self.bars_with_caps.columns}, shape={self.bars_with_caps.shape}")

        # Normalize column names for internal use (handle both formats)
        column_map = {
            'Date': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        bars_normalized = bars.rename(columns=column_map)

        # Ensure we have the required columns
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        self.bars = bars_normalized[[c for c in required if c in bars_normalized.columns]].copy()

        # Load strategy routing
        routing_path = ROOT / "config" / "strategy_routing.json"
        with open(routing_path) as f:
            self.strategy_routing = json.load(f)

        # Load continuous adaptation config
        continuous_config_path = ROOT / "config" / "continuous_adaptation.json"
        try:
            with open(continuous_config_path) as f:
                continuous_config_full = json.load(f)
                # Extract just the position_sizing.limits section for the sizer
                self.continuous_sizing_config = continuous_config_full.get('position_sizing', {}).get('limits', {})
                self.logger.info(f"Loaded continuous config: skip_threshold={self.continuous_sizing_config.get('skip_trade_threshold', 'NOT SET')}")
        except FileNotFoundError:
            self.logger.warning("continuous_adaptation.json not found, using defaults")
            self.continuous_sizing_config = {}

        # Metrics calculator
        self.metrics_calc = ContinuousMetrics()

        # Initialize strategy instances (strategies expect 'Close' not 'close')
        # Load ALL available strategies
        self.strategies = {}
        strategy_names = list_strategies()  # Get all available strategies
        self.logger.info(f"Loading {len(strategy_names)} strategies: {strategy_names}")
        for strategy_name in strategy_names:
            try:
                self.strategies[strategy_name] = load_strategy(strategy_name)
                self.logger.debug(f"Loaded strategy: {strategy_name}")
            except Exception as e:
                self.logger.warning(f"Could not load strategy {strategy_name}: {e}")

        # Track trades
        self.trades: List[Trade] = []

    def _detect_regime_unstable(self, bar_idx: int) -> str:
        """
        Detect regime using unstable method (raw thresholds, no stability).

        This is the baseline to compare against.
        """
        window = self.bars.iloc[:bar_idx+1]
        atr_pct = self.metrics_calc.calculate_atr_percentile(
            self.symbol, window
        )

        # Raw classification (no hysteresis, no smoothing)
        if atr_pct < 33:
            return "low_volatility"
        elif atr_pct < 67:
            return "normal"
        else:
            return "high_volatility"

    def _detect_regime_stable(
        self,
        detector: StableRegimeDetector,
        bar_idx: int
    ) -> str:
        """
        Detect regime using stable detector (with all stability features).
        """
        window = self.bars.iloc[:bar_idx+1]
        regime = detector.update_regime(self.symbol, window)
        return regime

    def _get_strategy_for_regime(self, regime: str) -> str:
        """Get strategy from routing config."""
        routing = self.strategy_routing.get(self.symbol, {})
        regime_config = routing.get(regime, routing.get('default', {}))

        if isinstance(regime_config, dict):
            return regime_config.get('strategy', 'sma')
        return regime_config or 'sma'

    def _precompute_signals_vectorized(self):
        """
        Pre-compute signals for all strategies using vectorized methods.

        Returns:
            Dict mapping strategy name to list of signals
        """
        signals_cache = {}

        for strategy_name, strategy_instance in self.strategies.items():
            try:
                # Use vectorized signal generation (much faster)
                signals = strategy_instance.generate_signals_vectorized(self.bars_with_caps)

                if signals is not None:
                    # Convert to BUY/SELL/HOLD strings
                    signals_str = []
                    for sig in signals:
                        if sig == 1:
                            signals_str.append("BUY")
                        elif sig == -1:
                            signals_str.append("SELL")
                        else:
                            signals_str.append("HOLD")
                    signals_cache[strategy_name] = signals_str
                    self.logger.debug(f"Pre-computed {len(signals_str)} signals for {strategy_name}")
                else:
                    # Fallback: generate signals row by row (slower but works)
                    signals_str = []
                    for i in range(len(self.bars_with_caps)):
                        if i < 50:
                            signals_str.append("HOLD")
                        else:
                            window = self.bars_with_caps.iloc[:i+1]
                            sig = strategy_instance.generate_signal(window)
                            if sig == 1:
                                signals_str.append("BUY")
                            elif sig == -1:
                                signals_str.append("SELL")
                            else:
                                signals_str.append("HOLD")
                    signals_cache[strategy_name] = signals_str
                    self.logger.debug(f"Generated {len(signals_str)} signals for {strategy_name} (fallback)")

            except Exception as e:
                self.logger.warning(f"Could not generate signals for {strategy_name}: {e}")
                # Create all HOLD signals as fallback
                signals_cache[strategy_name] = ["HOLD"] * len(self.bars_with_caps)

        return signals_cache

    def _generate_signal(self, strategy: str, bar_idx: int, signals_cache: dict = None) -> str:
        """
        Generate trading signal using cached signals if available.

        Returns:
            "BUY", "SELL", or "HOLD"
        """
        # Use cached signals if available
        if signals_cache and strategy in signals_cache:
            if bar_idx < len(signals_cache[strategy]):
                return signals_cache[strategy][bar_idx]

        # Fallback to direct generation (slower)
        # Need minimum bars for strategy warmup
        if bar_idx < 50:
            return "HOLD"

        # Get strategy instance
        if strategy not in self.strategies:
            self.logger.warning(f"Strategy {strategy} not loaded, using HOLD")
            return "HOLD"

        strategy_instance = self.strategies[strategy]

        # Get data window up to current bar (with capitalized columns for strategies)
        window = self.bars_with_caps.iloc[:bar_idx+1]

        try:
            # Generate signal using strategy
            signal = strategy_instance.generate_signal(window)

            # Convert signal to BUY/SELL/HOLD
            if signal == 1:
                return "BUY"
            elif signal == -1:
                return "SELL"
            else:
                return "HOLD"

        except Exception as e:
            self.logger.warning(f"Error generating signal for {strategy} at bar {bar_idx}: {e}")
            return "HOLD"

    def _calculate_position_size_fixed(
        self,
        capital: float,
        max_pct: float = 0.10
    ) -> Tuple[float, float]:
        """
        Fixed position sizing (baseline).

        Returns:
            (position_size, multiplier)
        """
        return (capital * max_pct, 1.0)

    def _calculate_position_size_continuous(
        self,
        sizer: ContinuousPositionSizer,
        strategy: str,
        capital: float,
        max_pct: float,
        bar_idx: int
    ) -> Tuple[float, float, bool]:
        """
        Continuous adaptive position sizing.

        Returns:
            (position_size, multiplier, should_skip)
        """
        window = self.bars.iloc[:bar_idx+1]

        # Get recent trades for Sharpe calculation
        recent_trades = [
            {
                'pnl': t.pnl,
                'entry_value': t.position_size,
                'return_pct': t.return_pct,
                'timestamp': t.entry_time
            }
            for t in self.trades[-30:]  # Last 30 trades
        ]

        result = sizer.calculate_position_size(
            symbol=self.symbol,
            strategy=strategy,
            base_capital=capital,
            max_position_pct=max_pct,
            bars=window,
            recent_trades=recent_trades,
            atr_lookback=100,  # Reduced for backtesting (vs 250 default)
            sharpe_lookback_days=30
        )

        should_skip = sizer.should_skip_trade(result)

        return (result.final_size, result.multiplier, should_skip)

    def run_backtest(
        self,
        regime_method: str = 'stable',  # 'stable' or 'unstable'
        sizing_method: str = 'continuous',  # 'continuous' or 'fixed'
        initial_capital: float = 100000,
        max_position_pct: float = 0.10,
        regime_config: Optional[Dict] = None,
        sizing_config: Optional[Dict] = None
    ) -> BacktestMetrics:
        """
        Run backtest with specified configuration.

        Args:
            regime_method: 'stable' or 'unstable'
            sizing_method: 'continuous' or 'fixed'
            initial_capital: Starting capital
            max_position_pct: Maximum position size
            regime_config: Config for regime detection
            sizing_config: Config for position sizing

        Returns:
            BacktestMetrics with results
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Backtesting {self.symbol}")
        self.logger.info(f"  Regime: {regime_method}")
        self.logger.info(f"  Sizing: {sizing_method}")
        self.logger.info(f"  Bars: {len(self.bars)}")
        self.logger.info(f"{'='*60}\n")

        # Initialize components
        if regime_method == 'stable':
            regime_detector = StableRegimeDetector(
                config=regime_config or {},
                logger=self.logger
            )

        if sizing_method == 'continuous':
            # Merge provided config with loaded config (provided config takes precedence)
            final_sizing_config = {**self.continuous_sizing_config, **(sizing_config or {})}
            position_sizer = ContinuousPositionSizer(
                config=final_sizing_config,
                logger=self.logger
            )
            self.logger.info(f"Continuous sizing config: {final_sizing_config}")

        # Pre-compute all signals (vectorized for speed)
        self.logger.info("Pre-computing signals...")
        signals_cache = self._precompute_signals_vectorized()
        self.logger.info(f"Signals pre-computed for {len(signals_cache)} strategies")

        # DEBUG: Log signal counts
        for strat_name, signals in signals_cache.items():
            buy_count = sum(1 for s in signals if s == "BUY")
            sell_count = sum(1 for s in signals if s == "SELL")
            if buy_count > 0 or sell_count > 0:
                self.logger.info(f"  {strat_name}: {buy_count} BUY, {sell_count} SELL")

        # Reset state
        self.trades = []
        capital = initial_capital
        position = None  # Current open position
        equity_curve = [initial_capital]
        regime_history = []

        # Run through bars (start after warmup for ATR percentile calculation)
        # Need 264 bars minimum (14 for ATR + 250 for lookback)
        # Adaptive warmup: use min of 300 or 40% of data
        warmup_bars = min(300, int(len(self.bars) * 0.4))
        self.logger.info(f"Using warmup_bars={warmup_bars} for {len(self.bars)} total bars")
        for i in range(warmup_bars, len(self.bars)):
            bar = self.bars.iloc[i]

            # Detect regime
            if regime_method == 'stable':
                regime = self._detect_regime_stable(regime_detector, i)
            else:
                regime = self._detect_regime_unstable(i)

            regime_history.append(regime)

            # Get strategy
            strategy = self._get_strategy_for_regime(regime)

            # Generate signal (use cached signals)
            signal = self._generate_signal(strategy, i, signals_cache)

            # Close existing position if needed
            if position and signal == "SELL":
                # Exit trade
                exit_price = bar['close']
                pnl = (exit_price - position['entry_price']) * position['shares']
                return_pct = (exit_price / position['entry_price'] - 1) * 100

                capital += pnl

                trade = Trade(
                    entry_time=position['entry_time'],
                    exit_time=bar['timestamp'],
                    symbol=self.symbol,
                    strategy=strategy,
                    regime=regime,
                    side='LONG',
                    entry_price=position['entry_price'],
                    exit_price=exit_price,
                    shares=position['shares'],
                    position_size=position['position_size'],
                    position_multiplier=position['multiplier'],
                    pnl=pnl,
                    return_pct=return_pct,
                    held_bars=i - position['entry_bar']
                )

                self.trades.append(trade)
                position = None

            # Open new position if signal
            if position is None and signal == "BUY":
                # Calculate position size
                if sizing_method == 'continuous':
                    size, multiplier, should_skip = self._calculate_position_size_continuous(
                        position_sizer, strategy, capital, max_position_pct, i
                    )

                    if should_skip:
                        continue
                else:
                    size, multiplier = self._calculate_position_size_fixed(
                        capital, max_position_pct
                    )

                # Enter trade
                entry_price = bar['close']
                shares = int(size / entry_price)

                if shares > 0:
                    position = {
                        'entry_time': bar['timestamp'],
                        'entry_bar': i,
                        'entry_price': entry_price,
                        'shares': shares,
                        'position_size': shares * entry_price,
                        'multiplier': multiplier,
                        'strategy': strategy,
                        'regime': regime
                    }

            # Update equity curve
            if position:
                current_value = capital + (bar['close'] - position['entry_price']) * position['shares']
            else:
                current_value = capital

            equity_curve.append(current_value)

        # Close any remaining position
        if position:
            final_bar = self.bars.iloc[-1]
            exit_price = final_bar['close']
            pnl = (exit_price - position['entry_price']) * position['shares']
            return_pct = (exit_price / position['entry_price'] - 1) * 100

            capital += pnl

            trade = Trade(
                entry_time=position['entry_time'],
                exit_time=final_bar['timestamp'],
                symbol=self.symbol,
                strategy=position['strategy'],
                regime=position['regime'],
                side='LONG',
                entry_price=position['entry_price'],
                exit_price=exit_price,
                shares=position['shares'],
                position_size=position['position_size'],
                position_multiplier=position['multiplier'],
                pnl=pnl,
                return_pct=return_pct,
                held_bars=len(self.bars) - position['entry_bar']
            )

            self.trades.append(trade)

        # Calculate metrics
        metrics = self._calculate_metrics(
            equity_curve,
            regime_history,
            regime_method,
            sizing_method
        )

        return metrics

    def _calculate_metrics(
        self,
        equity_curve: List[float],
        regime_history: List[str],
        regime_method: str,
        sizing_method: str
    ) -> BacktestMetrics:
        """Calculate performance metrics from backtest."""

        if len(self.trades) == 0:
            return BacktestMetrics(
                strategy_name=f"{regime_method}_{sizing_method}",
                total_return=0, sharpe_ratio=0, sortino_ratio=0,
                max_drawdown=0, win_rate=0, num_trades=0,
                avg_trade_return=0, volatility=0, calmar_ratio=0,
                regime_switches=0, avg_bars_per_regime=0,
                avg_position_multiplier=0, trades_skipped=0,
                risk_adjusted_return=0
            )

        # Total return
        total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100

        # Returns
        returns = pd.Series(equity_curve).pct_change().dropna()

        # Sharpe ratio
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = (returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino = 0

        # Max drawdown
        cummax = pd.Series(equity_curve).cummax()
        drawdown = (pd.Series(equity_curve) - cummax) / cummax
        max_dd = abs(drawdown.min()) * 100

        # Win rate
        wins = [t for t in self.trades if t.pnl > 0]
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0

        # Avg trade return
        avg_trade_ret = np.mean([t.return_pct for t in self.trades])

        # Volatility
        volatility = returns.std() * np.sqrt(252) * 100

        # Calmar ratio
        calmar = total_return / max_dd if max_dd > 0 else 0

        # Regime switches
        regime_switches = sum(1 for i in range(1, len(regime_history))
                             if regime_history[i] != regime_history[i-1])

        avg_bars_per_regime = len(regime_history) / (regime_switches + 1)

        # Position sizing stats
        avg_multiplier = np.mean([t.position_multiplier for t in self.trades])

        # Risk-adjusted return
        risk_adj_return = total_return / volatility if volatility > 0 else 0

        return BacktestMetrics(
            strategy_name=f"{regime_method}_{sizing_method}",
            total_return=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            win_rate=win_rate,
            num_trades=len(self.trades),
            avg_trade_return=avg_trade_ret,
            volatility=volatility,
            calmar_ratio=calmar,
            regime_switches=regime_switches,
            avg_bars_per_regime=avg_bars_per_regime,
            avg_position_multiplier=avg_multiplier,
            trades_skipped=0,  # TODO: track separately
            risk_adjusted_return=risk_adj_return
        )


def compare_configurations(symbol: str, bars: pd.DataFrame) -> pd.DataFrame:
    """
    Compare all configuration combinations.

    Tests:
    1. Unstable regime + Fixed sizing (baseline)
    2. Stable regime + Fixed sizing
    3. Unstable regime + Continuous sizing
    4. Stable regime + Continuous sizing (full system)
    """
    backtester = AdaptiveBacktester(symbol, bars)

    configurations = [
        ('unstable', 'fixed', "Baseline"),
        ('stable', 'fixed', "Stable Regimes Only"),
        ('unstable', 'continuous', "Continuous Sizing Only"),
        ('stable', 'continuous', "Full Adaptive System"),
    ]

    results = []

    for regime_method, sizing_method, name in configurations:
        print(f"\nTesting: {name}")
        print(f"  Regime: {regime_method}, Sizing: {sizing_method}")

        metrics = backtester.run_backtest(
            regime_method=regime_method,
            sizing_method=sizing_method
        )

        result = {
            'Configuration': name,
            'Regime': regime_method,
            'Sizing': sizing_method,
            **asdict(metrics)
        }

        results.append(result)

    df = pd.DataFrame(results)
    return df


def main():
    """Run backtests."""
    parser = argparse.ArgumentParser(description='Backtest adaptive features')
    parser.add_argument('--symbol', '-s', default='AAPL', help='Symbol to test')
    parser.add_argument('--all', action='store_true', help='Test all symbols')
    parser.add_argument('--days', '-d', type=int, default=750, help='Days of data')
    parser.add_argument('--compare', action='store_true', help='Compare all configs')
    parser.add_argument('--output', '-o', help='Output file (.csv)')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    # Load data
    pipeline = UnifiedDataPipeline()

    symbols = [args.symbol]
    if args.all:
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META', 'AMZN']

    all_results = []

    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"Testing {symbol}")
        print(f"{'='*80}")

        # Load data
        bars = pipeline.get_data(symbol, timeframe='day')

        if bars is None or bars.empty:
            print(f"No data for {symbol}, skipping")
            continue

        # Limit to requested days
        if len(bars) > args.days:
            bars = bars.tail(args.days)

        # Run comparison
        results_df = compare_configurations(symbol, bars)
        results_df['Symbol'] = symbol

        all_results.append(results_df)

        # Print results
        print(f"\n{symbol} Results:")
        print("="*80)
        print(results_df[['Configuration', 'total_return', 'sharpe_ratio',
                         'max_drawdown', 'win_rate', 'num_trades',
                         'regime_switches']].to_string(index=False))

    # Combine all results
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)

        # Save if requested
        if args.output:
            final_df.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")

    print("\n" + "="*80)
    print("Backtest Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
