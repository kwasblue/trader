#!/usr/bin/env python3
"""
Stable Regime Detector with Noise Filtering

Implements multiple stability mechanisms:
1. Hysteresis bands - Different thresholds for entering vs exiting
2. Regime persistence - Require N bars in new regime before switching
3. Cooldown periods - Minimum time between switches
4. Multi-timeframe confirmation - Use longer timeframes for regime
5. Smoothed metrics - EMA instead of raw values

Usage:
    detector = StableRegimeDetector()

    # Update with each new bar
    regime = detector.update_regime(symbol, new_bar, all_bars)

    # Check if regime switched
    if detector.regime_changed(symbol):
        print(f"Regime switched to {regime}")
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd


@dataclass
class RegimeState:
    """Current regime state for a symbol."""

    current_regime: str
    previous_regime: str
    regime_start_time: datetime
    regime_bar_count: int  # How many bars in current regime
    last_switch_time: datetime
    atr_percentile: float
    smoothed_atr_percentile: float  # EMA smoothed
    regime_strength: float  # How "strong" is current regime (0-1)


class StableRegimeDetector:
    """
    Regime detector with stability features to prevent noise-induced switching.

    Stability mechanisms:
    1. Hysteresis - Different thresholds for entering vs staying in regime
    2. Persistence - Require N consecutive bars in new regime
    3. Cooldown - Minimum time between regime switches
    4. Smoothing - EMA of volatility metrics
    5. Strength measurement - Only switch if new regime is strong
    """

    def __init__(self, config: dict | None = None, logger: logging.Logger | None = None):
        """
        Initialize stable regime detector.

        Args:
            config: Configuration dict
            logger: Optional logger
        """
        self.logger = logger or logging.getLogger(__name__)

        # Default configuration
        self.config = {
            # Base thresholds (percentile)
            "low_volatility_threshold": 33,
            "high_volatility_threshold": 67,
            # Hysteresis bands (prevent rapid switching)
            "hysteresis_band": 5,  # ±5% buffer around thresholds
            # Persistence requirements
            "min_bars_in_regime": 5,  # Require 5 bars before switching
            # Cooldown (prevent thrashing)
            "min_minutes_between_switches": 30,  # 30 minutes minimum
            # Smoothing
            "use_smoothed_atr": True,
            "atr_ema_alpha": 0.2,  # 0.2 = slower smoothing, more stable
            # Strength requirements
            "min_regime_strength": 0.6,  # Need 60% confidence to switch
            # Multi-timeframe (use longer bars for regime detection)
            "regime_detection_timeframe": "day",  # Detect regime on daily bars
            "execution_timeframe": None,  # Execute on any timeframe
        }

        # Override with provided config
        if config:
            self.config.update(config)

        # State tracking per symbol
        self.regime_states: dict[str, RegimeState] = {}

        # History for persistence checking
        self.regime_history: dict[str, deque] = {}

    def _calculate_hysteresis_thresholds(self, current_regime: str) -> tuple[float, float]:
        """
        Calculate hysteresis thresholds based on current regime.

        Different thresholds for staying in vs leaving a regime.

        Args:
            current_regime: Current regime

        Returns:
            (low_threshold, high_threshold) with hysteresis applied
        """
        base_low = self.config["low_volatility_threshold"]
        base_high = self.config["high_volatility_threshold"]
        band = self.config["hysteresis_band"]

        if current_regime == "low_volatility":
            # Staying in low vol: easier to stay, harder to leave
            # Need to go ABOVE threshold + band to leave
            return (0, base_low + band)

        elif current_regime == "normal":
            # Staying in normal: symmetric bands
            # Need to go outside threshold ± band to leave
            return (base_low - band, base_high + band)

        elif current_regime == "high_volatility":
            # Staying in high vol: easier to stay, harder to leave
            # Need to go BELOW threshold - band to leave
            return (base_high - band, 100)

        else:
            # No hysteresis for initial classification
            return (base_low, base_high)

    def _classify_regime(self, atr_percentile: float, current_regime: str | None = None) -> str:
        """
        Classify regime with hysteresis if currently in a regime.

        Args:
            atr_percentile: Current ATR percentile (0-100)
            current_regime: Current regime (None if first classification)

        Returns:
            Regime classification
        """
        if current_regime:
            # Use hysteresis thresholds
            low_thresh, high_thresh = self._calculate_hysteresis_thresholds(current_regime)
        else:
            # First time - use base thresholds
            low_thresh = self.config["low_volatility_threshold"]
            high_thresh = self.config["high_volatility_threshold"]

        if atr_percentile < low_thresh:
            return "low_volatility"
        elif atr_percentile < high_thresh:
            return "normal"
        else:
            return "high_volatility"

    def _smooth_atr_percentile(self, symbol: str, new_percentile: float) -> float:
        """
        Apply exponential moving average to ATR percentile.

        Args:
            symbol: Stock symbol
            new_percentile: New raw ATR percentile

        Returns:
            Smoothed ATR percentile
        """
        if not self.config["use_smoothed_atr"]:
            return new_percentile

        state = self.regime_states.get(symbol)

        if state is None:
            # First time - no smoothing
            return new_percentile

        # EMA formula: smoothed = alpha * new + (1 - alpha) * previous
        alpha = self.config["atr_ema_alpha"]
        smoothed = alpha * new_percentile + (1 - alpha) * state.smoothed_atr_percentile

        return smoothed

    def _calculate_regime_strength(self, atr_percentile: float, regime: str) -> float:
        """
        Calculate how "strong" a regime is (0-1).

        Strong regime = far from boundaries
        Weak regime = close to boundaries

        Args:
            atr_percentile: ATR percentile
            regime: Regime classification

        Returns:
            Strength score (0-1)
        """
        low_thresh = self.config["low_volatility_threshold"]
        high_thresh = self.config["high_volatility_threshold"]

        if regime == "low_volatility":
            # Strength = how far below low threshold
            # 0th percentile = 1.0, low_thresh = 0.0
            if atr_percentile >= low_thresh:
                return 0.0
            return (low_thresh - atr_percentile) / low_thresh

        elif regime == "normal":
            # Strength = how far from both boundaries
            # Middle = 1.0, edges = 0.0
            mid_point = (low_thresh + high_thresh) / 2
            distance_from_mid = abs(atr_percentile - mid_point)
            max_distance = (high_thresh - low_thresh) / 2
            return 1.0 - (distance_from_mid / max_distance)

        elif regime == "high_volatility":
            # Strength = how far above high threshold
            # 100th percentile = 1.0, high_thresh = 0.0
            if atr_percentile <= high_thresh:
                return 0.0
            return (atr_percentile - high_thresh) / (100 - high_thresh)

        return 0.0

    def _check_cooldown(self, symbol: str) -> bool:
        """
        Check if symbol is in cooldown period.

        Args:
            symbol: Stock symbol

        Returns:
            True if in cooldown (cannot switch)
        """
        state = self.regime_states.get(symbol)
        if state is None:
            return False

        min_minutes = self.config["min_minutes_between_switches"]
        time_since_switch = datetime.now() - state.last_switch_time

        if time_since_switch < timedelta(minutes=min_minutes):
            remaining = timedelta(minutes=min_minutes) - time_since_switch
            self.logger.debug(f"{symbol} in cooldown: {remaining.seconds // 60}min remaining")
            return True

        return False

    def _check_persistence(self, symbol: str, new_regime: str) -> bool:
        """
        Check if new regime has persisted long enough.

        Requires N consecutive bars in new regime before switching.

        Args:
            symbol: Stock symbol
            new_regime: Candidate new regime

        Returns:
            True if persisted long enough
        """
        if symbol not in self.regime_history:
            self.regime_history[symbol] = deque(maxlen=self.config["min_bars_in_regime"])

        history = self.regime_history[symbol]
        history.append(new_regime)

        # Check if all recent bars agree on new regime
        if len(history) < self.config["min_bars_in_regime"]:
            return False

        all_agree = all(r == new_regime for r in history)

        if not all_agree:
            self.logger.debug(
                f"{symbol} regime persistence check: {list(history)} (need {self.config['min_bars_in_regime']} consecutive)"
            )

        return all_agree

    def update_regime(self, symbol: str, bars: pd.DataFrame, atr_percentile: float | None = None) -> str:
        """
        Update regime for symbol with stability checks.

        Args:
            symbol: Stock symbol
            bars: Recent price bars
            atr_percentile: Pre-calculated ATR percentile (optional)

        Returns:
            Current regime (may not have changed)
        """
        # Calculate or use provided ATR percentile
        if atr_percentile is None:
            from core.continuous_metrics import ContinuousMetrics

            metrics = ContinuousMetrics()
            atr_percentile = metrics.calculate_atr_percentile(symbol, bars)

        # Smooth the percentile
        smoothed_percentile = self._smooth_atr_percentile(symbol, atr_percentile)

        # Get current state
        current_state = self.regime_states.get(symbol)
        current_regime = current_state.current_regime if current_state else None

        # Classify with hysteresis
        candidate_regime = self._classify_regime(smoothed_percentile, current_regime)

        # If no change, update state and return
        if candidate_regime == current_regime:
            if current_state:
                current_state.regime_bar_count += 1
                current_state.atr_percentile = atr_percentile
                current_state.smoothed_atr_percentile = smoothed_percentile
                current_state.regime_strength = self._calculate_regime_strength(smoothed_percentile, current_regime)
            return current_regime or candidate_regime

        # Regime wants to change - apply stability checks

        # Check 1: Cooldown period
        if self._check_cooldown(symbol):
            self.logger.debug(f"{symbol} regime switch blocked: in cooldown period")
            return current_regime

        # Check 2: Persistence requirement
        if not self._check_persistence(symbol, candidate_regime):
            self.logger.debug(f"{symbol} regime switch blocked: not enough persistence")
            return current_regime

        # Check 3: Regime strength requirement
        regime_strength = self._calculate_regime_strength(smoothed_percentile, candidate_regime)
        if regime_strength < self.config["min_regime_strength"]:
            self.logger.debug(
                f"{symbol} regime switch blocked: weak regime "
                f"(strength={regime_strength:.2f} < {self.config['min_regime_strength']})"
            )
            return current_regime

        # All checks passed - switch regime
        self.logger.info(
            f"{symbol} REGIME SWITCH: {current_regime} → {candidate_regime} "
            f"(ATR={smoothed_percentile:.1f}%, strength={regime_strength:.2f})"
        )

        # Update state
        new_state = RegimeState(
            current_regime=candidate_regime,
            previous_regime=current_regime or candidate_regime,
            regime_start_time=datetime.now(),
            regime_bar_count=1,
            last_switch_time=datetime.now(),
            atr_percentile=atr_percentile,
            smoothed_atr_percentile=smoothed_percentile,
            regime_strength=regime_strength,
        )

        self.regime_states[symbol] = new_state

        return candidate_regime

    def get_current_regime(self, symbol: str) -> str | None:
        """Get current regime for symbol."""
        state = self.regime_states.get(symbol)
        return state.current_regime if state else None

    def regime_changed(self, symbol: str) -> bool:
        """Check if regime changed on last update."""
        state = self.regime_states.get(symbol)
        if state is None:
            return False
        return state.current_regime != state.previous_regime

    def get_regime_info(self, symbol: str) -> dict:
        """Get detailed regime information."""
        state = self.regime_states.get(symbol)
        if state is None:
            return {}

        return {
            "regime": state.current_regime,
            "previous_regime": state.previous_regime,
            "bars_in_regime": state.regime_bar_count,
            "time_in_regime": (datetime.now() - state.regime_start_time).seconds // 60,
            "atr_percentile": state.atr_percentile,
            "smoothed_atr_percentile": state.smoothed_atr_percentile,
            "regime_strength": state.regime_strength,
            "time_since_last_switch": (datetime.now() - state.last_switch_time).seconds // 60,
        }


def main():
    """Example usage."""
    import json
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load test data
    data_path = Path(__file__).parents[1] / "data" / "data_storage" / "proc_data"
    test_file = data_path / "proc_AAPL_day.json"

    if test_file.exists():
        with open(test_file) as f:
            data = json.load(f)

        bars = pd.DataFrame(data)

        # Create detector with stability features
        detector = StableRegimeDetector(
            config={
                "min_bars_in_regime": 3,
                "min_minutes_between_switches": 15,
                "hysteresis_band": 5,
                "use_smoothed_atr": True,
                "atr_ema_alpha": 0.3,
            }
        )

        # Simulate streaming bars
        print("Simulating regime detection with stability checks:\n")

        for i in range(200, min(250, len(bars))):
            window = bars.iloc[: i + 1]

            regime = detector.update_regime("AAPL", window)
            info = detector.get_regime_info("AAPL")

            if detector.regime_changed("AAPL"):
                print(f"Bar {i}: ★ REGIME CHANGE ★")

            print(
                f"Bar {i}: {regime} (ATR={info['smoothed_atr_percentile']:.1f}%, "
                f"strength={info['regime_strength']:.2f}, "
                f"bars={info['bars_in_regime']})"
            )

            if i < 210:  # Just show first few
                continue


if __name__ == "__main__":
    main()
