"""
Tests for DynamicPositionSizer

Coverage:
- Position size calculation
- Risk-based sizing
- Market condition adjustments
"""
import pytest
from unittest.mock import MagicMock

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.position_sizer import DynamicPositionSizer


class TestDynamicPositionSizerInit:
    """Tests for DynamicPositionSizer initialization."""

    def test_init_valid_risk(self):
        """Test initialization with valid risk percentage."""
        sizer = DynamicPositionSizer(risk_percentage=0.01)
        assert sizer.risk_per_trade == 0.01

    def test_init_invalid_risk_zero(self):
        """Test initialization with zero risk raises error."""
        with pytest.raises(ValueError):
            DynamicPositionSizer(risk_percentage=0)

    def test_init_invalid_risk_negative(self):
        """Test initialization with negative risk raises error."""
        with pytest.raises(ValueError):
            DynamicPositionSizer(risk_percentage=-0.01)

    def test_init_invalid_risk_one(self):
        """Test initialization with risk=1 raises error."""
        with pytest.raises(ValueError):
            DynamicPositionSizer(risk_percentage=1.0)

    def test_init_sets_min_max_risk(self):
        """Test that min and max risk are set based on base risk."""
        sizer = DynamicPositionSizer(risk_percentage=0.02)
        assert sizer.min_risk_percentage == 0.01  # 0.02 * 0.5
        assert sizer.max_risk_percentage == 0.06  # 0.02 * 3


class TestAdjustRiskPercentage:
    """Tests for risk percentage adjustment."""

    @pytest.fixture
    def sizer(self):
        return DynamicPositionSizer(risk_percentage=0.02)

    def test_adjust_low_volatility(self, sizer):
        """Test risk increases in low volatility."""
        adjusted = sizer.adjust_risk_percentage("low_volatility")
        # Low volatility returns min(max_risk, risk * 1.25) = min(0.06, 0.025) = 0.025
        expected = min(sizer.max_risk_percentage, sizer.risk_per_trade * 1.25)
        assert adjusted == expected

    def test_adjust_normal(self, sizer):
        """Test risk stays normal in normal conditions."""
        adjusted = sizer.adjust_risk_percentage("normal")
        assert adjusted == sizer.risk_per_trade

    def test_adjust_high_volatility(self, sizer):
        """Test risk decreases in high volatility."""
        adjusted = sizer.adjust_risk_percentage("high_volatility")
        assert adjusted == sizer.min_risk_percentage


class TestCalculatePositionSize:
    """Tests for position size calculation."""

    @pytest.fixture
    def sizer(self):
        return DynamicPositionSizer(risk_percentage=0.01)

    def test_basic_calculation_buy(self, sizer):
        """Test basic position size calculation for buy."""
        size = sizer.calculate_position_size(
            price=100.0,
            stop_loss_price=95.0,  # $5 risk per share
            current_cash=10000.0,
            market_conditions="normal",
            signal=1  # Buy
        )
        # Risk = 1% of 10000 = $100
        # Per share risk = $5
        # Size = 100 / 5 = 20 shares
        assert size == 20

    def test_basic_calculation_sell(self, sizer):
        """Test basic position size calculation for sell."""
        size = sizer.calculate_position_size(
            price=100.0,
            stop_loss_price=105.0,  # $5 risk per share
            current_cash=10000.0,
            market_conditions="normal",
            signal=-1  # Sell
        )
        assert size == 20

    def test_low_volatility_increases_size(self, sizer):
        """Test that low volatility increases position size."""
        size_normal = sizer.calculate_position_size(
            price=100.0,
            stop_loss_price=95.0,
            current_cash=10000.0,
            market_conditions="normal",
            signal=1
        )
        size_low_vol = sizer.calculate_position_size(
            price=100.0,
            stop_loss_price=95.0,
            current_cash=10000.0,
            market_conditions="low_volatility",
            signal=1
        )
        assert size_low_vol > size_normal

    def test_high_volatility_decreases_size(self, sizer):
        """Test that high volatility decreases position size."""
        size_normal = sizer.calculate_position_size(
            price=100.0,
            stop_loss_price=95.0,
            current_cash=10000.0,
            market_conditions="normal",
            signal=1
        )
        size_high_vol = sizer.calculate_position_size(
            price=100.0,
            stop_loss_price=95.0,
            current_cash=10000.0,
            market_conditions="high_volatility",
            signal=1
        )
        assert size_high_vol < size_normal

    def test_returns_zero_for_zero_signal(self, sizer):
        """Test returns 0 for no signal."""
        size = sizer.calculate_position_size(
            price=100.0,
            stop_loss_price=95.0,
            current_cash=10000.0,
            market_conditions="normal",
            signal=0
        )
        assert size == 0

    def test_minimum_one_share(self, sizer):
        """Test position sizing with small cash returns 0 when risk < $5."""
        size = sizer.calculate_position_size(
            price=100.0,
            stop_loss_price=99.0,  # $1 risk per share
            current_cash=50.0,  # Very little cash
            market_conditions="high_volatility",
            signal=1
        )
        # With high vol (0.005 risk) and $50 cash, risk_per_trade = $0.25 < $5 min
        # Implementation returns 0 when risk_per_trade < $5
        assert size == 0

    def test_handles_wide_stop(self, sizer):
        """Test handling of wide stop loss."""
        size = sizer.calculate_position_size(
            price=100.0,
            stop_loss_price=50.0,  # 50% stop
            current_cash=10000.0,
            market_conditions="normal",
            signal=1
        )
        # Risk = $100, per share risk = $50
        # Size = 100 / 50 = 2
        assert size == 2

    def test_handles_tight_stop(self, sizer):
        """Test handling of tight stop loss."""
        size = sizer.calculate_position_size(
            price=100.0,
            stop_loss_price=99.50,  # $0.50 risk per share
            current_cash=10000.0,
            market_conditions="normal",
            signal=1
        )
        # Risk = $100, per share risk = $0.50
        # Position size = 100 / 0.50 = 200
        # But max_affordable = 10000 // 100 = 100
        # Returns min(200, 100) = 100
        assert size == 100


class TestEdgeCases:
    """Tests for edge cases."""

    def test_small_cash(self):
        """Test with very small cash returns 0 when risk < $5."""
        sizer = DynamicPositionSizer(risk_percentage=0.01)
        size = sizer.calculate_position_size(
            price=100.0,
            stop_loss_price=95.0,
            current_cash=100.0,  # Only $100
            market_conditions="normal",
            signal=1
        )
        # Risk = 1% of $100 = $1, which is < $5 minimum
        # Implementation returns 0 when risk_per_trade < $5
        assert size == 0

    def test_expensive_stock(self):
        """Test with expensive stock."""
        sizer = DynamicPositionSizer(risk_percentage=0.01)
        size = sizer.calculate_position_size(
            price=5000.0,
            stop_loss_price=4900.0,  # $100 risk per share
            current_cash=100000.0,
            market_conditions="normal",
            signal=1
        )
        # Risk = $1000, per share risk = $100
        # Size = 10
        assert size == 10
