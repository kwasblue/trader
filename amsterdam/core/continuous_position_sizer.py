#!/usr/bin/env python3
"""
Continuous Adaptive Position Sizer

Uses real-time metrics to dynamically adjust position sizes based on:
- Current market volatility (ATR percentile)
- Recent strategy performance (rolling Sharpe)
- Custom risk parameters

Integrates with existing position sizing but adds continuous adaptation layer.

Usage:
    sizer = ContinuousPositionSizer()

    # Calculate position size with continuous adaptation
    position_size = sizer.calculate_position_size(
        symbol='AAPL',
        strategy='rsi',
        base_capital=100000,
        bars=recent_bars,
        recent_trades=trade_history
    )
"""

import logging
from dataclasses import dataclass

import pandas as pd

from core.continuous_metrics import ContinuousMetrics, RiskMetrics


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""

    symbol: str
    strategy: str
    base_size: float  # Base position size (dollars)
    raw_score: float  # Raw combined score before scaling/bounds (0-1)
    multiplier: float  # Final multiplier after scaling and bounds
    final_size: float  # Adjusted position size (dollars)
    metrics: RiskMetrics  # The metrics used for calculation
    rationale: str  # Human-readable explanation


class ContinuousPositionSizer:
    """
    Adaptive position sizer using continuous metrics.

    Adjusts position size based on:
    1. Volatility conditions (lower vol = larger positions)
    2. Recent performance (better Sharpe = larger positions)
    3. Configurable risk thresholds
    """

    def __init__(self, config: dict | None = None, logger: logging.Logger | None = None):
        """
        Initialize continuous position sizer.

        Args:
            config: Configuration dict with thresholds
            logger: Optional logger
        """
        self.logger = logger or logging.getLogger(__name__)

        # Default configuration
        self.config = {
            # Volatility thresholds (percentile)
            "vol_low_threshold": 30,
            "vol_medium_threshold": 50,
            "vol_high_threshold": 70,
            # Sharpe thresholds
            "sharpe_excellent": 1.5,
            "sharpe_good": 1.0,
            "sharpe_acceptable": 0.5,
            # Position multipliers for different scenarios
            "multiplier_max": 1.0,  # Best conditions
            "multiplier_good": 0.75,  # Good conditions
            "multiplier_medium": 0.50,  # Medium conditions
            "multiplier_low": 0.25,  # Poor conditions
            "multiplier_min": 0.10,  # Worst conditions (or skip)
            # Continuous formula parameters
            "use_continuous_formula": True,  # vs discrete buckets
            "max_sharpe_for_scaling": 2.0,  # Sharpe >= this = 1.0 multiplier
            "scaling_alpha": 1.5,  # Scaling constant for leverage (C_final = α * √(V×P))
            # Safety limits (applied AFTER scaling)
            "min_multiplier": 0.30,  # Floor
            "max_multiplier": 1.5,  # Ceiling
            "skip_trade_threshold": 0.05,  # Check raw score BEFORE flooring
            # Bayesian shrinkage parameters (passed to metrics calculator)
            "bayesian_prior_sharpe": 0.5,  # Conservative prior
            "bayesian_prior_weight": 5,  # Equivalent to 5 trades
            "sharpe_smoothing_alpha": 0.3,  # EMA alpha for smoothing
            # Cold-start graduated ceiling (reduces ceiling when trade count is low)
            "use_graduated_ceiling": True,
            "cold_start_very_low_threshold": 5,  # N < 5: very cold
            "cold_start_low_threshold": 10,  # N < 10: warming up
            "ceiling_very_low_trades": 0.50,  # Max multiplier when N < 5
            "ceiling_low_trades": 0.75,  # Max multiplier when 5 <= N < 10
        }

        # Override with provided config
        if config:
            self.config.update(config)

        # Initialize metrics calculator with config
        self.metrics_calc = ContinuousMetrics(logger=self.logger, config=self.config)

    def calculate_discrete_multiplier(self, metrics: RiskMetrics) -> tuple[float, str]:
        """
        Calculate position multiplier using discrete buckets.

        Args:
            metrics: Risk metrics

        Returns:
            (multiplier, rationale) tuple
        """
        vol = metrics.atr_percentile
        sharpe = metrics.rolling_sharpe

        # Excellent conditions: Low vol + High Sharpe
        if sharpe >= self.config["sharpe_excellent"] and vol < self.config["vol_low_threshold"]:
            return (self.config["multiplier_max"], f"Excellent: Low vol ({vol:.0f}%) + High Sharpe ({sharpe:.2f})")

        # Good conditions: Low/Medium vol + Good Sharpe
        elif sharpe >= self.config["sharpe_excellent"] and vol < self.config["vol_medium_threshold"]:
            return (0.9, f"Very Good: Medium vol ({vol:.0f}%) + High Sharpe ({sharpe:.2f})")

        # Good Sharpe but high volatility
        elif sharpe >= self.config["sharpe_excellent"]:
            return (
                self.config["multiplier_good"],
                f"Good but volatile: High vol ({vol:.0f}%) + High Sharpe ({sharpe:.2f})",
            )

        # Decent Sharpe, low vol
        elif sharpe >= self.config["sharpe_good"] and vol < self.config["vol_low_threshold"]:
            return (0.85, f"Good: Low vol ({vol:.0f}%) + Decent Sharpe ({sharpe:.2f})")

        # Decent Sharpe, medium vol
        elif sharpe >= self.config["sharpe_good"] and vol < self.config["vol_medium_threshold"]:
            return (self.config["multiplier_good"], f"Decent: Medium vol ({vol:.0f}%) + Good Sharpe ({sharpe:.2f})")

        # Decent Sharpe, high vol
        elif sharpe >= self.config["sharpe_good"]:
            return (self.config["multiplier_medium"], f"Cautious: High vol ({vol:.0f}%) + Decent Sharpe ({sharpe:.2f})")

        # Marginal performance
        elif sharpe >= self.config["sharpe_acceptable"]:
            return (self.config["multiplier_low"], f"Marginal: Sharpe {sharpe:.2f}, Vol {vol:.0f}%")

        # Poor performance
        else:
            return (self.config["multiplier_min"], f"Poor: Sharpe {sharpe:.2f}, Vol {vol:.0f}% - Consider skipping")

    def get_effective_ceiling(self, trade_count: int) -> float:
        """
        Calculate effective ceiling based on trade count (graduated ceiling for cold start).

        Implements a warm-up policy:
        - N < 5: very cold start, cap at 50%
        - 5 <= N < 10: warming up, cap at 75%
        - N >= 10: mature, use full ceiling

        This allows trading during cold start (via prior + shrinkage) while
        limiting exposure until sufficient trade history exists.

        Args:
            trade_count: Number of recent trades

        Returns:
            Effective ceiling (may be lower than config max_multiplier)
        """
        if not self.config.get("use_graduated_ceiling", True):
            return self.config["max_multiplier"]

        very_low_thresh = self.config.get("cold_start_very_low_threshold", 5)
        low_thresh = self.config.get("cold_start_low_threshold", 10)

        if trade_count < very_low_thresh:
            # Very cold start
            ceiling = self.config.get("ceiling_very_low_trades", 0.50)
            self.logger.debug(f"Cold start (N={trade_count} < {very_low_thresh}): using reduced ceiling {ceiling:.2f}")
            return ceiling
        elif trade_count < low_thresh:
            # Warming up
            ceiling = self.config.get("ceiling_low_trades", 0.75)
            self.logger.debug(f"Warming up (N={trade_count} < {low_thresh}): using reduced ceiling {ceiling:.2f}")
            return ceiling
        else:
            # Mature - use full ceiling
            return self.config["max_multiplier"]

    def calculate_continuous_multiplier(self, metrics: RiskMetrics, trade_count: int) -> tuple[float, float, str]:
        """
        Calculate position multiplier using continuous formula with scaling.

        Formula:
            C_raw = √(V × P)           (geometric mean, range [0, 1])
            C_scaled = α × C_raw        (apply scaling constant)
            C_final = clamp(C_scaled, floor, ceiling)

        Where:
            V = volatility_score (0-1, inverted ATR percentile)
            P = performance_score (0-1, normalized Sharpe)
            α = scaling_alpha (allows leverage > 1.0)

        Args:
            metrics: Risk metrics (with pre-normalized scores)

        Returns:
            (raw_score, final_multiplier, rationale) tuple
        """
        # Step 1: Calculate raw combined score (geometric mean)
        raw_score = metrics.combined_score  # Already √(V × P)

        # Step 2: Apply scaling constant
        scaling_alpha = self.config.get("scaling_alpha", 1.0)
        scaled_score = scaling_alpha * raw_score

        # Step 3: Apply floor and effective ceiling (graduated for cold start)
        floor = self.config["min_multiplier"]
        effective_ceiling = self.get_effective_ceiling(trade_count)
        final_multiplier = max(floor, min(effective_ceiling, scaled_score))

        # Build rationale
        rationale_parts = [
            f"Vol={metrics.atr_percentile:.0f}% (V={metrics.volatility_score:.2f})",
            f"Sharpe={metrics.rolling_sharpe:.2f} (P={metrics.performance_score:.2f})",
            f"Raw=√(V×P)={raw_score:.2f}",
        ]

        if scaling_alpha != 1.0:
            rationale_parts.append(f"Scaled={scaled_score:.2f} (α={scaling_alpha:.2f})")

        if final_multiplier != scaled_score:
            if final_multiplier == floor:
                bound_type = "floor"
            elif final_multiplier == effective_ceiling and effective_ceiling < self.config["max_multiplier"]:
                bound_type = f"grad-ceiling (N={trade_count})"
            else:
                bound_type = "ceiling"
            rationale_parts.append(f"Final={final_multiplier:.2f} ({bound_type})")
        else:
            rationale_parts.append(f"Final={final_multiplier:.2f}")

        rationale = "Continuous: " + ", ".join(rationale_parts)

        return raw_score, final_multiplier, rationale

    def calculate_position_size(
        self,
        symbol: str,
        strategy: str,
        base_capital: float,
        max_position_pct: float,
        bars: pd.DataFrame,
        recent_trades: list[dict],
        atr_lookback: int = 250,
        sharpe_lookback_days: int = 30,
    ) -> PositionSizeResult:
        """
        Calculate position size with continuous adaptation.

        Args:
            symbol: Stock symbol
            strategy: Strategy name
            base_capital: Total capital
            max_position_pct: Maximum position size as decimal (e.g., 0.10 = 10%)
            bars: Recent price bars for volatility calculation
            recent_trades: Recent trade history for performance calculation
            atr_lookback: Lookback period for ATR percentile
            sharpe_lookback_days: Days for rolling Sharpe calculation

        Returns:
            PositionSizeResult with calculated size and metrics
        """
        # Calculate base position size (before adjustment)
        base_size = base_capital * max_position_pct

        # Count recent trades for graduated ceiling
        from datetime import datetime, timedelta

        if recent_trades:
            cutoff = datetime.now() - timedelta(days=sharpe_lookback_days)
            recent_count = sum(
                1
                for t in recent_trades
                if (
                    t.get("timestamp", datetime.now())
                    if not isinstance(t.get("timestamp"), str)
                    else datetime.fromisoformat(t["timestamp"])
                )
                > cutoff
            )
        else:
            recent_count = 0

        # Get risk metrics
        metrics = self.metrics_calc.get_risk_metrics(
            symbol=symbol,
            bars=bars,
            strategy=strategy,
            recent_trades=recent_trades,
            atr_lookback=atr_lookback,
            sharpe_lookback_days=sharpe_lookback_days,
        )

        # Calculate multiplier
        if self.config["use_continuous_formula"]:
            raw_score, multiplier, rationale = self.calculate_continuous_multiplier(metrics, trade_count=recent_count)
        else:
            # Discrete method doesn't have raw_score concept
            multiplier, rationale = self.calculate_discrete_multiplier(metrics)
            raw_score = metrics.combined_score  # Use combined score as proxy

        # Calculate final position size
        final_size = base_size * multiplier

        result = PositionSizeResult(
            symbol=symbol,
            strategy=strategy,
            base_size=base_size,
            raw_score=raw_score,
            multiplier=multiplier,
            final_size=final_size,
            metrics=metrics,
            rationale=rationale,
        )

        self.logger.info(
            f"{symbol}/{strategy} Position Size: "
            f"Base=${base_size:.0f} × {multiplier:.2f} = ${final_size:.0f} "
            f"({rationale})"
        )

        return result

    def should_skip_trade(self, result: PositionSizeResult, min_size_threshold: float | None = None) -> bool:
        """
        Determine if trade should be skipped based on RAW score (before flooring).

        This fixes the skip logic bug where skip_threshold < floor makes skip unreachable.
        We check the raw combined score BEFORE applying scaling and bounds.

        Args:
            result: Position size result (contains raw_score)
            min_size_threshold: Minimum raw score to trade (if None, uses config value)

        Returns:
            True if trade should be skipped
        """
        # Use config value if not specified
        if min_size_threshold is None:
            min_size_threshold = self.config.get("skip_trade_threshold", 0.15)

        # Check raw score BEFORE flooring (this is the key fix)
        if result.raw_score < min_size_threshold:
            self.logger.warning(
                f"{result.symbol}/{result.strategy} SKIP TRADE: "
                f"Raw score {result.raw_score:.2f} < {min_size_threshold:.2f} "
                f"(final multiplier would be {result.multiplier:.2f}) "
                f"({result.rationale})"
            )
            return True

        return False


def main():
    """Example usage."""
    import json
    from datetime import datetime, timedelta
    from pathlib import Path

    # Setup
    logging.basicConfig(level=logging.INFO)

    # Load test data
    data_path = Path(__file__).parents[1] / "data" / "data_storage" / "proc_data"
    test_file = data_path / "proc_AAPL_day.json"

    if test_file.exists():
        with open(test_file) as f:
            data = json.load(f)

        bars = pd.DataFrame(data)

        # Mock some trades
        mock_trades = [
            {
                "pnl": 100 * (1 if i % 3 != 0 else -1),  # 67% win rate
                "entry_value": 10000,
                "timestamp": datetime.now() - timedelta(days=i),
                "return_pct": 1.0 if i % 3 != 0 else -1.0,
            }
            for i in range(20)
        ]

        # Test position sizer
        sizer = ContinuousPositionSizer()

        result = sizer.calculate_position_size(
            symbol="AAPL",
            strategy="rsi",
            base_capital=100000,
            max_position_pct=0.10,
            bars=bars,
            recent_trades=mock_trades,
        )

        print(f"\n{'=' * 60}")
        print(f"Position Size Calculation: {result.symbol}/{result.strategy}")
        print(f"{'=' * 60}")
        print(f"Base Capital: ${result.base_size:,.0f}")
        print(f"Multiplier: {result.multiplier:.2%}")
        print(f"Final Size: ${result.final_size:,.0f}")
        print("\nMetrics:")
        print(f"  ATR Percentile: {result.metrics.atr_percentile:.1f}%")
        print(f"  Rolling Sharpe: {result.metrics.rolling_sharpe:.2f}")
        print(f"  Volatility Score: {result.metrics.volatility_score:.2f}")
        print(f"  Performance Score: {result.metrics.performance_score:.2f}")
        print(f"\nRationale: {result.rationale}")
        print(f"Skip Trade? {sizer.should_skip_trade(result)}")


if __name__ == "__main__":
    main()
