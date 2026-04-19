"""
Unit tests for HybridPositionSizer.

Tests all matrix combinations, trend detection, and alignment logic.
"""

from core.logic.hybrid_position_sizer import HybridPositionSizer, HybridSizingResult


class TestHybridPositionSizer:
    """Tests for HybridPositionSizer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sizer = HybridPositionSizer(enabled=True)

    # =========================================================================
    # DISABLED MODE TESTS
    # =========================================================================

    def test_disabled_mode_returns_full_multiplier(self):
        """When disabled, should return 1.0 multiplier (pass-through)."""
        sizer = HybridPositionSizer(enabled=False)
        result = sizer.calculate(signal=1, confidence=0.3, daily_context={})

        assert result.base_multiplier == 1.0
        assert result.reason == "Hybrid sizing disabled"

    def test_disabled_mode_ignores_low_confidence(self):
        """Disabled mode should not block low confidence trades."""
        sizer = HybridPositionSizer(enabled=False)
        result = sizer.calculate(signal=1, confidence=0.1, daily_context={"RSI": 30})

        assert result.base_multiplier == 1.0

    # =========================================================================
    # MATRIX COMBINATION TESTS
    # =========================================================================

    def test_high_confidence_with_trend_full_size(self):
        """High confidence + with trend = 100% position."""
        # Bullish context (RSI > 50, MACD > Signal, Close > SMA)
        daily_ctx = {
            "RSI": 60,
            "MACD": 1.0,
            "MACD_Signal": 0.5,
            "Close": 150,
            "SMA_200": 140,
        }
        result = self.sizer.calculate(signal=1, confidence=0.8, daily_context=daily_ctx)

        assert result.base_multiplier == 1.0
        assert result.trend == "bullish"
        assert result.aligned is True

    def test_high_confidence_against_trend_half_size(self):
        """High confidence + against trend = 50% position."""
        # Bearish context
        daily_ctx = {
            "RSI": 40,
            "MACD": -1.0,
            "MACD_Signal": 0.5,
            "Close": 130,
            "SMA_200": 140,
        }
        # Buy signal against bearish trend
        result = self.sizer.calculate(signal=1, confidence=0.8, daily_context=daily_ctx)

        assert result.base_multiplier == 0.5
        assert result.trend == "bearish"
        assert result.aligned is False

    def test_med_confidence_with_trend_60_percent(self):
        """Medium confidence + with trend = 60% position."""
        daily_ctx = {
            "RSI": 55,
            "MACD": 0.5,
            "MACD_Signal": 0.3,
            "Close": 150,
            "SMA_200": 145,
        }
        result = self.sizer.calculate(signal=1, confidence=0.6, daily_context=daily_ctx)

        assert result.base_multiplier == 0.6
        assert result.aligned is True

    def test_med_confidence_against_trend_25_percent(self):
        """Medium confidence + against trend = 25% position."""
        daily_ctx = {
            "RSI": 45,
            "MACD": -0.5,
            "MACD_Signal": 0.3,
            "Close": 135,
            "SMA_200": 145,
        }
        # Buy signal against bearish trend
        result = self.sizer.calculate(signal=1, confidence=0.6, daily_context=daily_ctx)

        assert result.base_multiplier == 0.25
        assert result.aligned is False

    def test_low_confidence_with_trend_30_percent(self):
        """Low confidence + with trend = 30% position."""
        daily_ctx = {
            "RSI": 55,
            "MACD": 0.5,
            "MACD_Signal": 0.3,
            "Close": 150,
            "SMA_200": 145,
        }
        result = self.sizer.calculate(signal=1, confidence=0.4, daily_context=daily_ctx)

        assert result.base_multiplier == 0.3
        assert result.aligned is True

    def test_low_confidence_against_trend_blocked(self):
        """Low confidence + against trend = NO TRADE (0%)."""
        daily_ctx = {
            "RSI": 45,
            "MACD": -0.5,
            "MACD_Signal": 0.3,
            "Close": 135,
            "SMA_200": 145,
        }
        # Buy signal against bearish trend with low confidence
        result = self.sizer.calculate(signal=1, confidence=0.3, daily_context=daily_ctx)

        assert result.base_multiplier == 0.0
        assert result.aligned is False
        assert "Blocked" in result.reason

    # =========================================================================
    # TREND DETECTION TESTS
    # =========================================================================

    def test_trend_bullish_requires_two_votes(self):
        """Trend should be bullish with 2+ bullish indicators."""
        # RSI bullish, MACD bullish, SMA neutral
        daily_ctx = {
            "RSI": 55,
            "MACD": 1.0,
            "MACD_Signal": 0.5,
            "Close": 145,
            "SMA_200": 145,  # Equal = neutral
        }
        result = self.sizer.calculate(signal=1, confidence=0.8, daily_context=daily_ctx)
        assert result.trend == "bullish"

    def test_trend_bearish_requires_two_votes(self):
        """Trend should be bearish with 2+ bearish indicators."""
        daily_ctx = {
            "RSI": 45,
            "MACD": -1.0,
            "MACD_Signal": 0.5,
            "Close": 135,
            "SMA_200": 140,
        }
        result = self.sizer.calculate(signal=-1, confidence=0.8, daily_context=daily_ctx)
        assert result.trend == "bearish"

    def test_trend_neutral_when_mixed(self):
        """Trend should be neutral with mixed indicators."""
        # 1 bullish (RSI), 1 bearish (MACD < Signal), 1 neutral (Close == SMA)
        daily_ctx = {
            "RSI": 55,  # Bullish
            "MACD": -0.5,  # Bearish (MACD < Signal)
            "MACD_Signal": 0.3,
            "Close": 140,  # Neutral vs SMA (equal)
            "SMA_200": 140,
        }
        result = self.sizer.calculate(signal=1, confidence=0.8, daily_context=daily_ctx)
        assert result.trend == "neutral"

    def test_trend_neutral_with_no_context(self):
        """Trend should be neutral with empty/None context."""
        result = self.sizer.calculate(signal=1, confidence=0.8, daily_context=None)
        assert result.trend == "neutral"

        result2 = self.sizer.calculate(signal=1, confidence=0.8, daily_context={})
        assert result2.trend == "neutral"

    # =========================================================================
    # ALIGNMENT TESTS
    # =========================================================================

    def test_buy_aligned_with_bullish(self):
        """Buy signal should be aligned with bullish trend."""
        daily_ctx = {"RSI": 60, "MACD": 1.0, "MACD_Signal": 0.5, "Close": 150, "SMA_200": 140}
        result = self.sizer.calculate(signal=1, confidence=0.8, daily_context=daily_ctx)
        assert result.aligned is True

    def test_sell_aligned_with_bearish(self):
        """Sell signal should be aligned with bearish trend."""
        daily_ctx = {"RSI": 40, "MACD": -1.0, "MACD_Signal": 0.5, "Close": 130, "SMA_200": 140}
        result = self.sizer.calculate(signal=-1, confidence=0.8, daily_context=daily_ctx)
        assert result.aligned is True

    def test_buy_not_aligned_with_bearish(self):
        """Buy signal should NOT be aligned with bearish trend."""
        daily_ctx = {"RSI": 40, "MACD": -1.0, "MACD_Signal": 0.5, "Close": 130, "SMA_200": 140}
        result = self.sizer.calculate(signal=1, confidence=0.8, daily_context=daily_ctx)
        assert result.aligned is False

    def test_sell_not_aligned_with_bullish(self):
        """Sell signal should NOT be aligned with bullish trend."""
        daily_ctx = {"RSI": 60, "MACD": 1.0, "MACD_Signal": 0.5, "Close": 150, "SMA_200": 140}
        result = self.sizer.calculate(signal=-1, confidence=0.8, daily_context=daily_ctx)
        assert result.aligned is False

    def test_hold_always_aligned(self):
        """Hold signal (0) should always be aligned."""
        daily_ctx = {"RSI": 60, "MACD": 1.0, "MACD_Signal": 0.5, "Close": 150, "SMA_200": 140}
        result = self.sizer.calculate(signal=0, confidence=0.8, daily_context=daily_ctx)
        assert result.aligned is True

    def test_any_signal_aligned_with_neutral(self):
        """Any signal should be aligned with neutral trend."""
        daily_ctx = {"RSI": 50}  # Neutral RSI, no other indicators
        result = self.sizer.calculate(signal=1, confidence=0.8, daily_context=daily_ctx)
        assert result.aligned is True

        result2 = self.sizer.calculate(signal=-1, confidence=0.8, daily_context=daily_ctx)
        assert result2.aligned is True

    # =========================================================================
    # CONFIDENCE THRESHOLD TESTS
    # =========================================================================

    def test_custom_high_threshold(self):
        """Should respect custom high confidence threshold."""
        sizer = HybridPositionSizer(enabled=True, config={"high_confidence_threshold": 0.8})
        daily_ctx = {"RSI": 60, "MACD": 1.0, "MACD_Signal": 0.5, "Close": 150, "SMA_200": 140}

        # 0.75 should be "med" with threshold of 0.8
        result = sizer.calculate(signal=1, confidence=0.75, daily_context=daily_ctx)
        assert result.base_multiplier == 0.6  # med + aligned

        # 0.85 should be "high"
        result2 = sizer.calculate(signal=1, confidence=0.85, daily_context=daily_ctx)
        assert result2.base_multiplier == 1.0  # high + aligned

    def test_custom_low_threshold(self):
        """Should respect custom low confidence threshold."""
        sizer = HybridPositionSizer(enabled=True, config={"low_confidence_threshold": 0.4})
        daily_ctx = {"RSI": 60, "MACD": 1.0, "MACD_Signal": 0.5, "Close": 150, "SMA_200": 140}

        # 0.35 should be "low" with threshold of 0.4
        result = sizer.calculate(signal=1, confidence=0.35, daily_context=daily_ctx)
        assert result.base_multiplier == 0.3  # low + aligned

        # 0.45 should be "med"
        result2 = sizer.calculate(signal=1, confidence=0.45, daily_context=daily_ctx)
        assert result2.base_multiplier == 0.6  # med + aligned

    # =========================================================================
    # EDGE CASE TESTS
    # =========================================================================

    def test_boundary_confidence_high(self):
        """Confidence exactly at high threshold should be high tier."""
        daily_ctx = {"RSI": 60, "MACD": 1.0, "MACD_Signal": 0.5, "Close": 150, "SMA_200": 140}
        result = self.sizer.calculate(signal=1, confidence=0.7, daily_context=daily_ctx)
        assert result.base_multiplier == 1.0  # high tier

    def test_boundary_confidence_low(self):
        """Confidence just below low threshold should be low tier."""
        daily_ctx = {"RSI": 60, "MACD": 1.0, "MACD_Signal": 0.5, "Close": 150, "SMA_200": 140}
        result = self.sizer.calculate(signal=1, confidence=0.49, daily_context=daily_ctx)
        assert result.base_multiplier == 0.3  # low tier + aligned

    def test_partial_daily_context(self):
        """Should handle partial daily context gracefully."""
        # Only RSI available
        daily_ctx = {"RSI": 60}
        result = self.sizer.calculate(signal=1, confidence=0.8, daily_context=daily_ctx)
        # Only 1 bullish vote, should be neutral
        assert result.trend == "neutral"

    def test_rsi_exactly_50_is_neutral(self):
        """RSI of exactly 50 should not vote either way."""
        daily_ctx = {"RSI": 50}
        result = self.sizer.calculate(signal=1, confidence=0.8, daily_context=daily_ctx)
        assert result.trend == "neutral"

    # =========================================================================
    # HELPER METHOD TESTS
    # =========================================================================

    def test_should_trade_returns_true_for_nonzero_multiplier(self):
        """should_trade should return True when trade allowed."""
        daily_ctx = {"RSI": 60, "MACD": 1.0, "MACD_Signal": 0.5, "Close": 150, "SMA_200": 140}
        assert self.sizer.should_trade(signal=1, confidence=0.8, daily_context=daily_ctx) is True

    def test_should_trade_returns_false_for_blocked_trade(self):
        """should_trade should return False when trade blocked."""
        daily_ctx = {"RSI": 40, "MACD": -1.0, "MACD_Signal": 0.5, "Close": 130, "SMA_200": 140}
        # Low confidence buy against bearish trend
        assert self.sizer.should_trade(signal=1, confidence=0.3, daily_context=daily_ctx) is False

    def test_get_daily_trend_public_method(self):
        """get_daily_trend should return correct trend."""
        bullish_ctx = {"RSI": 60, "MACD": 1.0, "MACD_Signal": 0.5, "Close": 150, "SMA_200": 140}
        bearish_ctx = {"RSI": 40, "MACD": -1.0, "MACD_Signal": 0.5, "Close": 130, "SMA_200": 140}

        assert self.sizer.get_daily_trend(bullish_ctx) == "bullish"
        assert self.sizer.get_daily_trend(bearish_ctx) == "bearish"
        assert self.sizer.get_daily_trend(None) == "neutral"


class TestHybridSizingResult:
    """Tests for HybridSizingResult dataclass."""

    def test_result_has_all_fields(self):
        """Result should have all required fields."""
        result = HybridSizingResult(
            base_multiplier=0.5, confidence=0.6, trend="bullish", aligned=True, reason="test reason"
        )

        assert result.base_multiplier == 0.5
        assert result.confidence == 0.6
        assert result.trend == "bullish"
        assert result.aligned is True
        assert result.reason == "test reason"
