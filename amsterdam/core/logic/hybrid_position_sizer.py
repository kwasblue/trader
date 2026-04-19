"""
Hybrid Position Sizer - Combines confidence scoring + trend alignment for position sizing.

This module enables strategies to use daily indicators for trend context while
executing on minute bars for intraday trading.

Key Features:
- Confidence score determines IF you trade
- Trend alignment determines HOW MUCH (position size)
- Config toggle to enable/disable (can compare to current logic)

Position Sizing Matrix:
                       Against Trend    With Trend
                     +----------------+----------------+
     High Confidence |    Medium      |     Full       |
          (>0.7)     |     (50%)      |    (100%)      |
                     +----------------+----------------+
     Med Confidence  |    Small       |    Medium      |
        (0.5-0.7)    |     (25%)      |     (60%)      |
                     +----------------+----------------+
     Low Confidence  |   NO TRADE     |    Small       |
          (<0.5)     |      (0%)      |     (30%)      |
                     +----------------+----------------+

Usage:
    sizer = HybridPositionSizer(enabled=True)
    result = sizer.calculate(signal=1, confidence=0.75, daily_context={...})
    final_size = base_position_size * result.base_multiplier
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class HybridSizingResult:
    """Result of hybrid position sizing calculation."""

    base_multiplier: float  # 0.0 to 1.0
    confidence: float  # Signal confidence
    trend: str  # "bullish", "bearish", "neutral"
    aligned: bool  # Signal aligned with trend?
    reason: str  # Human-readable explanation


class HybridPositionSizer:
    """
    Combines confidence scoring + trend alignment for position sizing.

    The hybrid approach provides two levels of risk management:
    1. Confidence gating: Low confidence signals against trend are blocked
    2. Position sizing: Higher confidence + alignment = larger positions

    This allows the strategy to be more aggressive when conditions are
    favorable and more conservative when they're not.

    Attributes:
        enabled: Whether hybrid sizing is active
        high_threshold: Confidence threshold for "high" tier (default 0.7)
        low_threshold: Confidence threshold for "low" tier (default 0.5)
        MATRIX: Lookup table for (confidence_tier, aligned) -> multiplier
    """

    # Position size multipliers
    # Key: (confidence_tier, aligned) -> multiplier
    MATRIX: dict[tuple, float] = {
        ("high", True): 1.0,  # High confidence + with trend = full size
        ("high", False): 0.5,  # High confidence + against trend = half size
        ("med", True): 0.6,  # Medium confidence + with trend = 60%
        ("med", False): 0.25,  # Medium confidence + against trend = 25%
        ("low", True): 0.3,  # Low confidence + with trend = 30%
        ("low", False): 0.0,  # Low confidence + against trend = NO TRADE
    }

    def __init__(self, enabled: bool = True, config: dict[str, Any] | None = None):
        """
        Initialize the hybrid position sizer.

        Args:
            enabled: Whether hybrid sizing is active (False = pass-through)
            config: Optional configuration dict with:
                - high_confidence_threshold: float (default 0.7)
                - low_confidence_threshold: float (default 0.5)
                - trend_indicators: list of indicator names to use
        """
        self.enabled = enabled
        self.config = config or {}
        self.high_threshold = self.config.get("high_confidence_threshold", 0.7)
        self.low_threshold = self.config.get("low_confidence_threshold", 0.5)
        self.trend_indicators = self.config.get("trend_indicators", ["RSI", "MACD", "SMA_200"])

    def calculate(
        self,
        signal: int,
        confidence: float,
        daily_context: dict[str, Any] | None = None,
    ) -> HybridSizingResult:
        """
        Calculate position size multiplier based on confidence and trend.

        Args:
            signal: Trading signal (-1 = sell, 0 = hold, 1 = buy)
            confidence: Signal confidence (0.0 to 1.0)
            daily_context: Daily timeframe indicator values for trend detection.
                Expected keys: RSI, MACD, MACD_Signal, Close, SMA_200

        Returns:
            HybridSizingResult with multiplier and explanation
        """
        if not self.enabled:
            return HybridSizingResult(
                base_multiplier=1.0,
                confidence=confidence,
                trend="neutral",
                aligned=True,
                reason="Hybrid sizing disabled",
            )

        # Determine daily trend from indicators
        trend = self._get_trend(daily_context)

        # Check if signal aligns with trend
        aligned = self._is_aligned(signal, trend)

        # Classify confidence into tiers
        tier = self._get_confidence_tier(confidence)

        # Lookup multiplier from matrix
        multiplier = self.MATRIX.get((tier, aligned), 0.5)

        # Build human-readable reason
        reason = self._build_reason(tier, confidence, aligned, trend, multiplier)

        return HybridSizingResult(
            base_multiplier=multiplier,
            confidence=confidence,
            trend=trend,
            aligned=aligned,
            reason=reason,
        )

    def _get_confidence_tier(self, confidence: float) -> str:
        """
        Classify confidence value into tier.

        Args:
            confidence: Confidence value (0.0 to 1.0)

        Returns:
            Tier string: "high", "med", or "low"
        """
        if confidence >= self.high_threshold:
            return "high"
        elif confidence >= self.low_threshold:
            return "med"
        else:
            return "low"

    def _get_trend(self, daily_context: dict[str, Any] | None) -> str:
        """
        Determine daily trend from indicators.

        Uses a voting system across multiple indicators:
        - RSI > 50: bullish vote
        - MACD > Signal: bullish vote
        - Close > SMA_200: bullish vote

        Requires 2+ votes for bullish/bearish, otherwise neutral.

        Args:
            daily_context: Dict with indicator values

        Returns:
            Trend string: "bullish", "bearish", or "neutral"
        """
        if not daily_context:
            return "neutral"

        bullish = 0
        bearish = 0

        # RSI indicator
        rsi = daily_context.get("RSI")
        if rsi is not None:
            if rsi > 50:
                bullish += 1
            elif rsi < 50:
                bearish += 1

        # MACD vs Signal
        macd = daily_context.get("MACD")
        macd_signal = daily_context.get("MACD_Signal")
        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                bullish += 1
            elif macd < macd_signal:
                bearish += 1

        # Price vs SMA_200
        close = daily_context.get("Close")
        sma_200 = daily_context.get("SMA_200")
        if close is not None and sma_200 is not None:
            if close > sma_200:
                bullish += 1
            elif close < sma_200:
                bearish += 1

        # Require 2+ votes for directional bias
        if bullish >= 2:
            return "bullish"
        if bearish >= 2:
            return "bearish"
        return "neutral"

    def _is_aligned(self, signal: int, trend: str) -> bool:
        """
        Check if trading signal aligns with daily trend.

        A signal is aligned if:
        - Trend is neutral (always aligned)
        - Buy signal (1) with bullish trend
        - Sell signal (-1) with bearish trend
        - Hold signal (0) is always aligned

        Args:
            signal: Trading signal (-1, 0, or 1)
            trend: Trend string ("bullish", "bearish", or "neutral")

        Returns:
            True if signal aligns with trend
        """
        if trend == "neutral":
            return True
        if signal == 0:
            return True  # Hold is always aligned
        if signal == 1 and trend == "bullish":
            return True
        if signal == -1 and trend == "bearish":
            return True
        return False

    def _build_reason(self, tier: str, confidence: float, aligned: bool, trend: str, multiplier: float) -> str:
        """
        Build human-readable explanation for sizing decision.

        Args:
            tier: Confidence tier ("high", "med", "low")
            confidence: Actual confidence value
            aligned: Whether signal is aligned with trend
            trend: Trend direction
            multiplier: Resulting multiplier

        Returns:
            Explanation string
        """
        if multiplier == 0.0:
            return f"Blocked: low confidence ({confidence:.2f}) against {trend} trend"

        alignment_str = "with" if aligned else "against"
        return f"{tier} conf ({confidence:.2f}), {alignment_str} {trend} trend -> {multiplier:.0%}"

    def should_trade(
        self,
        signal: int,
        confidence: float,
        daily_context: dict[str, Any] | None = None,
    ) -> bool:
        """
        Quick check if a trade should be taken (multiplier > 0).

        Convenience method for use in trade approval logic.

        Args:
            signal: Trading signal (-1, 0, or 1)
            confidence: Signal confidence (0.0 to 1.0)
            daily_context: Daily indicator values

        Returns:
            True if trade should proceed, False if blocked
        """
        result = self.calculate(signal, confidence, daily_context)
        return result.base_multiplier > 0.0

    def get_daily_trend(self, daily_context: dict[str, Any] | None) -> str:
        """
        Public method to get daily trend.

        Can be used by strategies to incorporate trend into signal generation.

        Args:
            daily_context: Dict with indicator values

        Returns:
            Trend string: "bullish", "bearish", or "neutral"
        """
        return self._get_trend(daily_context)
