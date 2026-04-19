#!/usr/bin/env python3
"""
Continuous Metrics Calculator

Calculates real-time metrics for continuous adaptation:
- ATR percentile (volatility measurement)
- Rolling Sharpe ratio (recent strategy performance)
- Other continuous market metrics

Usage:
    metrics = ContinuousMetrics()

    # Get current volatility percentile
    vol_pct = metrics.calculate_atr_percentile('AAPL', bars)

    # Get recent performance
    sharpe = metrics.calculate_rolling_sharpe('AAPL', 'rsi', recent_trades)

    # Combined risk score
    risk_score = metrics.get_risk_score('AAPL', bars, 'rsi', recent_trades)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


@dataclass
class RiskMetrics:
    """Container for continuous risk metrics."""

    atr_percentile: float  # 0-100
    rolling_sharpe: float  # typically -2 to +3
    volatility_score: float  # 0-1 (normalized)
    performance_score: float  # 0-1 (normalized)
    combined_score: float  # 0-1 (overall confidence)


class ContinuousMetrics:
    """
    Calculate continuous metrics for adaptive position sizing.

    Provides real-time measurements of:
    1. Market volatility (ATR percentile)
    2. Strategy performance (rolling Sharpe)
    3. Combined risk/confidence scores
    """

    def __init__(self, logger=None, config=None):
        self.logger = logger or logging.getLogger(__name__)

        # Configuration
        self.atr_period = 14  # ATR calculation period
        self.atr_lookback = 250  # Historical lookback for percentile
        self.sharpe_lookback_days = 30  # Rolling Sharpe window

        # Minimum trades for Sharpe calculation
        # NOTE: This is a RELIABILITY THRESHOLD, not a hard execution wall.
        # Below this threshold, Bayesian shrinkage dominates (heavy prior weight).
        # Above it, sample data dominates (light prior weight).
        # Trading is still allowed with N < min_trades via prior + shrinkage.
        self.min_trades_for_sharpe = 10

        # Bayesian shrinkage parameters
        self.bayesian_prior_sharpe = 0.5  # Conservative prior (vs 1.0)
        self.bayesian_prior_weight = 5  # Equivalent to 5 trades of prior belief

        # Exponential smoothing parameter
        self.sharpe_smoothing_alpha = 0.3  # EMA alpha (0.3 = 30% new, 70% old)

        # Smoothed Sharpe history: {(symbol, strategy): smoothed_value}
        self.smoothed_sharpe_history = {}

        # Override from config if provided
        if config:
            self.bayesian_prior_sharpe = config.get("bayesian_prior_sharpe", 0.5)
            self.bayesian_prior_weight = config.get("bayesian_prior_weight", 5)
            self.sharpe_smoothing_alpha = config.get("sharpe_smoothing_alpha", 0.3)

    def calculate_atr(self, bars: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range.

        Args:
            bars: DataFrame with 'high', 'low', 'close' columns
            period: ATR period (default 14)

        Returns:
            Current ATR value
        """
        if len(bars) < period + 1:
            return 0.0

        # Calculate True Range for each bar
        true_ranges = []
        for i in range(1, len(bars)):
            high = bars.iloc[i]["high"]
            low = bars.iloc[i]["low"]
            prev_close = bars.iloc[i - 1]["close"]

            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)

        # Average of last 'period' true ranges
        if len(true_ranges) >= period:
            atr = sum(true_ranges[-period:]) / period
            return atr

        return 0.0

    def calculate_atr_percentile(
        self, symbol: str, bars: pd.DataFrame, current_period: int = 14, lookback: int = 250
    ) -> float:
        """
        Calculate where current ATR ranks in historical distribution.

        Args:
            symbol: Stock symbol
            bars: Historical bars (need at least lookback + current_period)
            current_period: Period for current ATR calculation
            lookback: How many bars to use for historical comparison

        Returns:
            Percentile rank (0-100)
        """
        # Adjust lookback if not enough bars available
        required_bars = lookback + current_period
        if len(bars) < required_bars:
            # Use what we have, or minimum 50 bars
            lookback = max(50, len(bars) - current_period)
            if lookback < 50:
                self.logger.debug(
                    f"{symbol}: Not enough bars for ATR percentile "
                    f"(need {required_bars}, have {len(bars)}), returning neutral"
                )
                return 50.0  # Return neutral value

        # Calculate current ATR
        current_atr = self.calculate_atr(bars.tail(current_period + 1), current_period)

        if current_atr == 0:
            return 50.0

        # Calculate historical ATR values (rolling window)
        historical_atrs = []
        for i in range(current_period, lookback):
            if i >= len(bars):
                break
            window = bars.iloc[i - current_period : i + 1]
            atr = self.calculate_atr(window, current_period)
            if atr > 0:
                historical_atrs.append(atr)

        if not historical_atrs:
            return 50.0

        # Calculate percentile rank
        # (how many historical values are below current)
        below_current = sum(1 for atr in historical_atrs if atr < current_atr)
        percentile = (below_current / len(historical_atrs)) * 100

        self.logger.debug(f"{symbol} ATR: current={current_atr:.2f}, percentile={percentile:.1f}%")

        return percentile

    def calculate_rolling_sharpe(
        self,
        symbol: str,
        strategy: str,
        recent_trades: list[dict],
        lookback_days: int = 30,
        use_bayesian_shrinkage: bool = True,
        use_smoothing: bool = True,
    ) -> float:
        """
        Calculate rolling Sharpe ratio from recent trades with Bayesian shrinkage and smoothing.

        Uses Bayesian shrinkage to blend sample Sharpe with prior belief, reducing noise
        when trade count is low. Also applies exponential smoothing to prevent twitchiness.

        Args:
            symbol: Stock symbol
            strategy: Strategy name
            recent_trades: List of recent trade dicts with 'pnl', 'timestamp'
            lookback_days: How many days to look back
            use_bayesian_shrinkage: Apply Bayesian shrinkage (default True)
            use_smoothing: Apply exponential smoothing (default True)

        Returns:
            Sharpe ratio (annualized, shrunk, and smoothed)
        """
        key = (symbol, strategy)

        if not recent_trades:
            self.logger.debug(f"{symbol}/{strategy}: No trades for Sharpe calculation")
            # Return smoothed value if available, else prior
            if use_smoothing and key in self.smoothed_sharpe_history:
                return self.smoothed_sharpe_history[key]
            return self.bayesian_prior_sharpe if use_bayesian_shrinkage else 0.0

        # Filter trades by timeframe
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        filtered_trades = []
        for t in recent_trades:
            ts = t.get("timestamp", datetime.now())
            # Handle both string and datetime timestamps
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except (ValueError, AttributeError):
                    continue
            if ts > cutoff_date:
                filtered_trades.append(t)

        if len(filtered_trades) < self.min_trades_for_sharpe:
            self.logger.debug(
                f"{symbol}/{strategy}: Only {len(filtered_trades)} trades "
                f"(below reliability threshold {self.min_trades_for_sharpe}), "
                f"using prior/history (not skipping trade)"
            )
            # Return smoothed value if available, else prior
            # NOTE: This allows trading with conservative estimate, not a skip condition
            if use_smoothing and key in self.smoothed_sharpe_history:
                return self.smoothed_sharpe_history[key]
            return self.bayesian_prior_sharpe if use_bayesian_shrinkage else 0.0

        # Extract returns
        returns = []
        for trade in filtered_trades:
            # Get return as percentage
            if "return_pct" in trade:
                ret = trade["return_pct"]
            elif "pnl" in trade and "entry_value" in trade:
                ret = (trade["pnl"] / trade["entry_value"]) * 100
            else:
                continue
            returns.append(ret)

        if len(returns) < self.min_trades_for_sharpe:
            if use_smoothing and key in self.smoothed_sharpe_history:
                return self.smoothed_sharpe_history[key]
            return self.bayesian_prior_sharpe if use_bayesian_shrinkage else 0.0

        # Calculate sample Sharpe ratio
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            sample_sharpe = 0.0
        else:
            # Annualize assuming 252 trading days
            sample_sharpe = (mean_return / std_return) * np.sqrt(252)

        # Apply Bayesian shrinkage
        if use_bayesian_shrinkage:
            n = len(returns)
            prior_n = self.bayesian_prior_weight
            prior_sharpe = self.bayesian_prior_sharpe

            # Weighted average: (n * sample + prior_n * prior) / (n + prior_n)
            shrunk_sharpe = (n * sample_sharpe + prior_n * prior_sharpe) / (n + prior_n)

            self.logger.debug(
                f"{symbol}/{strategy} Sharpe: sample={sample_sharpe:.2f}, "
                f"shrunk={shrunk_sharpe:.2f} (n={n}, prior={prior_sharpe:.2f}, "
                f"prior_n={prior_n})"
            )
            current_sharpe = shrunk_sharpe
        else:
            current_sharpe = sample_sharpe

        # Apply exponential smoothing
        if use_smoothing:
            if key in self.smoothed_sharpe_history:
                prev_smoothed = self.smoothed_sharpe_history[key]
                smoothed_sharpe = (
                    self.sharpe_smoothing_alpha * current_sharpe + (1 - self.sharpe_smoothing_alpha) * prev_smoothed
                )
                self.logger.debug(
                    f"{symbol}/{strategy} Smoothing: current={current_sharpe:.2f}, "
                    f"prev={prev_smoothed:.2f}, smoothed={smoothed_sharpe:.2f}"
                )
            else:
                # First value - no smoothing needed
                smoothed_sharpe = current_sharpe

            # Update history
            self.smoothed_sharpe_history[key] = smoothed_sharpe
            final_sharpe = smoothed_sharpe
        else:
            final_sharpe = current_sharpe

        self.logger.debug(
            f"{symbol}/{strategy} Rolling Sharpe: {final_sharpe:.2f} "
            f"(mean={mean_return:.2f}%, std={std_return:.2f}%, n={len(returns)})"
        )

        return final_sharpe

    def normalize_atr_percentile(self, atr_percentile: float) -> float:
        """
        Convert ATR percentile to 0-1 score (inverted - lower vol = higher score).

        Args:
            atr_percentile: ATR percentile (0-100)

        Returns:
            Normalized score (0-1), where 1 = lowest volatility
        """
        # Invert: low volatility = high score
        # 0th percentile (lowest vol) = 1.0
        # 100th percentile (highest vol) = 0.0
        return max(0.0, min(1.0, (100 - atr_percentile) / 100))

    def normalize_sharpe(self, sharpe: float, max_sharpe: float = 2.0) -> float:
        """
        Convert Sharpe ratio to 0-1 score.

        Args:
            sharpe: Sharpe ratio (typically -2 to +3)
            max_sharpe: Sharpe value that maps to 1.0 (default 2.0)

        Returns:
            Normalized score (0-1), where 1 = excellent performance
        """
        # Negative Sharpe = 0
        # Sharpe >= max_sharpe = 1.0
        if sharpe <= 0:
            return 0.0

        return min(1.0, sharpe / max_sharpe)

    def get_risk_metrics(
        self,
        symbol: str,
        bars: pd.DataFrame,
        strategy: str,
        recent_trades: list[dict],
        atr_lookback: int = 250,
        sharpe_lookback_days: int = 30,
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.

        Args:
            symbol: Stock symbol
            bars: Historical price bars
            strategy: Strategy name
            recent_trades: Recent trade history
            atr_lookback: Lookback period for ATR percentile
            sharpe_lookback_days: Days for rolling Sharpe

        Returns:
            RiskMetrics object with all calculated metrics
        """
        # Calculate raw metrics
        atr_pct = self.calculate_atr_percentile(symbol, bars, lookback=atr_lookback)

        sharpe = self.calculate_rolling_sharpe(symbol, strategy, recent_trades, lookback_days=sharpe_lookback_days)

        # Normalize to 0-1 scores
        vol_score = self.normalize_atr_percentile(atr_pct)
        perf_score = self.normalize_sharpe(sharpe)

        # Combined score (geometric mean - both must be good)
        combined = np.sqrt(vol_score * perf_score)

        metrics = RiskMetrics(
            atr_percentile=atr_pct,
            rolling_sharpe=sharpe,
            volatility_score=vol_score,
            performance_score=perf_score,
            combined_score=combined,
        )

        self.logger.info(
            f"{symbol}/{strategy} Risk Metrics: "
            f"ATR={atr_pct:.1f}% (score={vol_score:.2f}), "
            f"Sharpe={sharpe:.2f} (score={perf_score:.2f}), "
            f"Combined={combined:.2f}"
        )

        return metrics


def main():
    """Example usage."""
    import json
    from pathlib import Path

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Load some test data
    data_path = Path(__file__).parents[1] / "data" / "data_storage" / "proc_data"
    test_file = data_path / "proc_AAPL_day.json"

    if test_file.exists():
        with open(test_file) as f:
            data = json.load(f)

        bars = pd.DataFrame(data)

        # Test metrics
        metrics_calc = ContinuousMetrics()

        # ATR percentile
        atr_pct = metrics_calc.calculate_atr_percentile("AAPL", bars)
        print(f"AAPL ATR Percentile: {atr_pct:.1f}%")

        # Mock some trades for Sharpe calculation
        mock_trades = [
            {"pnl": 100, "entry_value": 10000, "timestamp": datetime.now() - timedelta(days=i)} for i in range(20)
        ]

        sharpe = metrics_calc.calculate_rolling_sharpe("AAPL", "rsi", mock_trades)
        print(f"AAPL/RSI Rolling Sharpe: {sharpe:.2f}")

        # Full metrics
        metrics = metrics_calc.get_risk_metrics("AAPL", bars, "rsi", mock_trades)
        print(f"Combined Score: {metrics.combined_score:.2f}")


if __name__ == "__main__":
    main()
