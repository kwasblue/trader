"""
Tests for PositionManager - position lifecycle management.

Tests the REAL implementation for:
- Entry level calculation (SL/TP)
- Trailing stop updates
- Exit condition checks
- Partial exit logic
"""
import pytest
from datetime import datetime, timezone, timedelta

from core.logic.position_manager import PositionManager
from core.logic.symbol_state import SymbolState
from core.enums import OrderSide


class TestPositionManagerInit:
    """Test initialization."""

    def test_default_initialization(self):
        """Test default parameters."""
        pm = PositionManager()

        assert pm.exit_fraction == 0.25
        assert pm.min_bars_to_hold == 3
        assert pm.swing_mode is False
        assert pm.min_hold_days == 1

    def test_custom_initialization(self):
        """Test custom parameters."""
        pm = PositionManager(
            exit_fraction=0.5,
            min_bars_to_hold=5,
            swing_mode=True,
            min_hold_days=2
        )

        assert pm.exit_fraction == 0.5
        assert pm.min_bars_to_hold == 5
        assert pm.swing_mode is True
        assert pm.min_hold_days == 2

    def test_multipliers_stored(self):
        """Test that multipliers are stored correctly."""
        pm = PositionManager(
            tp_mults={"normal": 3.0, "high_volatility": 4.0},
            sl_mults={"normal": 2.0, "high_volatility": 2.5}
        )

        assert pm.tp_mults["normal"] == 3.0
        assert pm.tp_mults["high_volatility"] == 4.0
        assert pm.sl_mults["normal"] == 2.0
        assert pm.sl_mults["high_volatility"] == 2.5


class TestCalculateLevels:
    """Test entry level calculation."""

    @pytest.fixture
    def pm(self):
        return PositionManager(
            tp_mults={"normal": 2.0, "high_volatility": 3.0},
            sl_mults={"normal": 1.5, "high_volatility": 2.0}
        )

    def test_long_entry_levels(self, pm):
        """Test SL/TP calculation for long entry."""
        state = SymbolState(symbol="AAPL")

        pm.calculate_levels(
            state=state,
            price=150.0,
            atr=2.0,
            condition="normal",
            side=OrderSide.BUY
        )

        # Long: SL below entry, TP above entry
        assert state.stop_loss == 150.0 - (2.0 * 1.5)  # 147.0
        assert state.take_profit == 150.0 + (2.0 * 2.0)  # 154.0
        assert len(state.partial_exit_targets) == 2
        assert state.partial_exit_targets[0] == 150.0 + 2.0  # 152.0
        assert state.partial_exit_targets[1] == 150.0 + 4.0  # 154.0

    def test_short_entry_levels(self, pm):
        """Test SL/TP calculation for short entry."""
        state = SymbolState(symbol="AAPL")

        pm.calculate_levels(
            state=state,
            price=150.0,
            atr=2.0,
            condition="normal",
            side=OrderSide.SELL
        )

        # Short: SL above entry, TP below entry
        assert state.stop_loss == 150.0 + (2.0 * 1.5)  # 153.0
        assert state.take_profit == 150.0 - (2.0 * 2.0)  # 146.0
        assert len(state.partial_exit_targets) == 2
        assert state.partial_exit_targets[0] == 150.0 - 2.0  # 148.0
        assert state.partial_exit_targets[1] == 150.0 - 4.0  # 146.0

    def test_high_volatility_uses_different_multipliers(self, pm):
        """Test that high volatility regime uses different multipliers."""
        state = SymbolState(symbol="AAPL")

        pm.calculate_levels(
            state=state,
            price=150.0,
            atr=2.0,
            condition="high_volatility",
            side=OrderSide.BUY
        )

        # High vol: wider stops
        assert state.stop_loss == 150.0 - (2.0 * 2.0)  # 146.0 (vs 147.0 for normal)
        assert state.take_profit == 150.0 + (2.0 * 3.0)  # 156.0 (vs 154.0 for normal)

    def test_resets_tracking_fields(self, pm):
        """Test that calculate_levels resets tracking fields."""
        state = SymbolState(symbol="AAPL")
        state.bars_held = 10
        state.max_favorable_excursion = 5.0

        pm.calculate_levels(
            state=state,
            price=150.0,
            atr=2.0,
            condition="normal",
            side=OrderSide.BUY
        )

        assert state.bars_held == 0
        assert state.max_favorable_excursion is None
        assert state.entry_date is not None


class TestTrailingStop:
    """Test trailing stop updates."""

    @pytest.fixture
    def pm(self):
        return PositionManager(
            sl_mults={"normal": 1.5}
        )

    def test_long_trailing_stop_moves_up(self, pm):
        """Trailing stop should move up as price rises for long."""
        state = SymbolState(symbol="AAPL")
        state.side = "long"
        state.stop_loss = 147.0  # Initial SL at 147

        # Price rises to 155
        pm.update_trailing_stop(state, price=155.0, atr=2.0, condition="normal")

        # New SL should be 155 - (2.0 * 1.5) = 152.0, which is higher than 147
        assert state.stop_loss == 152.0

    def test_long_trailing_stop_doesnt_move_down(self, pm):
        """Trailing stop should not move down for long."""
        state = SymbolState(symbol="AAPL")
        state.side = "long"
        state.stop_loss = 150.0  # Current SL at 150

        # Price drops to 148
        pm.update_trailing_stop(state, price=148.0, atr=2.0, condition="normal")

        # New calculated SL would be 148 - 3 = 145, but shouldn't go below 150
        assert state.stop_loss == 150.0

    def test_short_trailing_stop_moves_down(self, pm):
        """Trailing stop should move down as price falls for short."""
        state = SymbolState(symbol="AAPL")
        state.side = "short"
        state.stop_loss = 153.0  # Initial SL at 153

        # Price falls to 145
        pm.update_trailing_stop(state, price=145.0, atr=2.0, condition="normal")

        # New SL should be 145 + 3 = 148.0, which is lower than 153
        assert state.stop_loss == 148.0

    def test_short_trailing_stop_doesnt_move_up(self, pm):
        """Trailing stop should not move up for short."""
        state = SymbolState(symbol="AAPL")
        state.side = "short"
        state.stop_loss = 150.0  # Current SL at 150

        # Price rises to 155
        pm.update_trailing_stop(state, price=155.0, atr=2.0, condition="normal")

        # New calculated SL would be 158, but shouldn't go above 150
        assert state.stop_loss == 150.0


class TestCheckExitConditions:
    """Test exit condition checks."""

    @pytest.fixture
    def pm(self):
        return PositionManager(min_bars_to_hold=3)

    def test_not_in_position_returns_false(self, pm):
        """Should return False if not in position."""
        state = SymbolState(symbol="AAPL")
        state.side = None

        should_exit, reason = pm.check_exit_conditions(state, price=150.0, signal=0)

        assert should_exit is False
        assert reason is None

    def test_stop_loss_hit_long(self, pm):
        """Stop loss hit should trigger exit for long (after first bar)."""
        state = SymbolState(symbol="AAPL")
        state.side = "long"
        state.current_position = 100  # Must set to be "in position"
        state.stop_loss = 145.0
        state.bars_held = 1  # After first bar - SL triggers normally

        should_exit, reason = pm.check_exit_conditions(state, price=144.0, signal=0)

        assert should_exit is True
        assert "Stop loss" in reason

    def test_stop_loss_hit_short(self, pm):
        """Stop loss hit should trigger exit for short (after first bar)."""
        state = SymbolState(symbol="AAPL")
        state.side = "short"
        state.current_position = -100  # Negative for short
        state.stop_loss = 155.0
        state.bars_held = 1  # After first bar - SL triggers normally

        should_exit, reason = pm.check_exit_conditions(state, price=156.0, signal=0)

        assert should_exit is True
        assert "Stop loss" in reason

    def test_same_bar_exit_blocked(self, pm):
        """Same-bar exits (bars_held=0) should be blocked to prevent churn."""
        state = SymbolState(symbol="AAPL")
        state.side = "long"
        state.current_position = 100
        state.entry_price = 150.0
        state.stop_loss = 145.0
        state.bars_held = 0  # Same bar as entry

        # Normal stop-loss hit should be blocked on bar 0
        should_exit, reason = pm.check_exit_conditions(state, price=144.0, signal=0)

        assert should_exit is False
        assert "Same-bar exit blocked" in reason

    def test_catastrophic_stop_allowed_on_same_bar(self, pm):
        """Catastrophic price moves (>2x SL distance) should allow same-bar exit."""
        state = SymbolState(symbol="AAPL")
        state.side = "long"
        state.current_position = 100
        state.entry_price = 150.0
        state.stop_loss = 145.0  # 5 points away
        state.bars_held = 0

        # Price crashes 15 points (3x the SL distance of 5)
        should_exit, reason = pm.check_exit_conditions(state, price=135.0, signal=0)

        assert should_exit is True
        assert "catastrophic" in reason.lower()

    def test_take_profit_hit_after_min_hold(self, pm):
        """Take profit should trigger exit after min hold period."""
        state = SymbolState(symbol="AAPL")
        state.side = "long"
        state.current_position = 100
        state.take_profit = 160.0
        state.bars_held = 5  # Above min_bars_to_hold

        should_exit, reason = pm.check_exit_conditions(state, price=161.0, signal=0)

        assert should_exit is True
        assert "Take profit" in reason

    def test_take_profit_blocked_before_min_hold(self, pm):
        """Take profit should NOT trigger before min hold period (but SL can)."""
        state = SymbolState(symbol="AAPL")
        state.side = "long"
        state.current_position = 100
        state.take_profit = 160.0
        state.stop_loss = 140.0  # Not hit
        state.bars_held = 1  # Below min_bars_to_hold

        should_exit, reason = pm.check_exit_conditions(state, price=161.0, signal=0)

        # TP not triggered because bars_held < min_bars_to_hold
        assert should_exit is False

    def test_signal_reversal_triggers_exit(self, pm):
        """Signal reversal should trigger exit after min hold."""
        state = SymbolState(symbol="AAPL")
        state.side = "long"
        state.current_position = 100
        state.bars_held = 5

        # Sell signal while long
        should_exit, reason = pm.check_exit_conditions(state, price=150.0, signal=-1)

        assert should_exit is True
        assert "reversal" in reason.lower()

    def test_signal_reversal_blocked_before_min_hold(self, pm):
        """Signal reversal should NOT trigger before min hold."""
        state = SymbolState(symbol="AAPL")
        state.side = "long"
        state.current_position = 100
        state.stop_loss = 140.0  # Not hit
        state.bars_held = 1  # Below min

        should_exit, reason = pm.check_exit_conditions(state, price=150.0, signal=-1)

        assert should_exit is False


class TestSwingMode:
    """Test swing mode (multi-day hold enforcement)."""

    def test_swing_mode_blocks_same_day_exit(self):
        """Swing mode should block same-day exits (except SL)."""
        pm = PositionManager(swing_mode=True, min_hold_days=1)

        state = SymbolState(symbol="AAPL")
        state.side = "long"
        state.current_position = 100
        state.entry_date = datetime.now(timezone.utc)  # Entered today
        state.take_profit = 160.0
        state.stop_loss = 140.0
        state.bars_held = 10  # Would normally allow TP

        # TP hit but same day - should block
        should_exit, reason = pm.check_exit_conditions(state, price=161.0, signal=0)

        assert should_exit is False
        assert reason is not None
        assert "Swing mode" in reason

    def test_swing_mode_allows_stop_loss_same_day(self):
        """Swing mode should still allow stop loss (protection) after first bar."""
        pm = PositionManager(swing_mode=True, min_hold_days=1)

        state = SymbolState(symbol="AAPL")
        state.side = "long"
        state.current_position = 100
        state.entry_date = datetime.now(timezone.utc)  # Entered today
        state.stop_loss = 145.0
        state.bars_held = 1  # After first bar (same-bar guard passed)

        # SL hit same day - should still trigger (swing mode allows SL)
        should_exit, reason = pm.check_exit_conditions(state, price=144.0, signal=0)

        assert should_exit is True
        assert "Stop loss" in reason

    def test_swing_mode_allows_exit_after_hold_days(self):
        """Swing mode should allow exit after min hold days."""
        pm = PositionManager(swing_mode=True, min_hold_days=1, min_bars_to_hold=0)

        state = SymbolState(symbol="AAPL")
        state.side = "long"
        state.current_position = 100
        state.entry_date = datetime.now(timezone.utc) - timedelta(days=2)  # 2 days ago
        state.take_profit = 160.0
        state.bars_held = 10

        should_exit, reason = pm.check_exit_conditions(state, price=161.0, signal=0)

        assert should_exit is True


class TestPartialExit:
    """Test partial exit logic."""

    def test_partial_exit_target_hit_long(self):
        """Partial exit target should trigger for long."""
        pm = PositionManager(min_bars_to_hold=0)

        state = SymbolState(symbol="AAPL")
        state.side = "long"
        state.partial_exit_targets = [152.0, 154.0]
        state.bars_held = 5

        should_exit, reason = pm.check_partial_exit(state, price=153.0)

        assert should_exit is True
        assert "152.00" in reason
        # First target should be removed
        assert len(state.partial_exit_targets) == 1
        assert state.partial_exit_targets[0] == 154.0

    def test_partial_exit_target_hit_short(self):
        """Partial exit target should trigger for short."""
        pm = PositionManager(min_bars_to_hold=0)

        state = SymbolState(symbol="AAPL")
        state.side = "short"
        state.partial_exit_targets = [148.0, 146.0]
        state.bars_held = 5

        should_exit, reason = pm.check_partial_exit(state, price=147.0)

        assert should_exit is True
        assert "148.00" in reason

    def test_no_partial_exit_if_target_not_hit(self):
        """No partial exit if target not reached."""
        pm = PositionManager()

        state = SymbolState(symbol="AAPL")
        state.side = "long"
        state.partial_exit_targets = [155.0, 160.0]

        should_exit, reason = pm.check_partial_exit(state, price=152.0)

        assert should_exit is False
        assert len(state.partial_exit_targets) == 2


class TestGetExitQuantity:
    """Test exit quantity calculation."""

    def test_full_exit_returns_full_position(self):
        """Full exit should return entire position."""
        pm = PositionManager(exit_fraction=0.25)

        qty = pm.get_exit_quantity(current_position=100, is_partial=False)

        assert qty == 100

    def test_partial_exit_returns_fraction(self):
        """Partial exit should return fraction of position."""
        pm = PositionManager(exit_fraction=0.25)

        qty = pm.get_exit_quantity(current_position=100, is_partial=True)

        assert qty == 25  # 25% of 100

    def test_partial_exit_minimum_one(self):
        """Partial exit should return at least 1."""
        pm = PositionManager(exit_fraction=0.1)

        qty = pm.get_exit_quantity(current_position=5, is_partial=True)

        assert qty == 1  # max(0.5, 1) = 1

    def test_handles_negative_position(self):
        """Should handle negative (short) positions."""
        pm = PositionManager(exit_fraction=0.25)

        qty = pm.get_exit_quantity(current_position=-100, is_partial=False)

        assert qty == 100  # abs(-100)


class TestExcursionTracking:
    """Test MFE/MAE tracking."""

    def test_update_excursions_long(self):
        """Test excursion tracking for long position."""
        pm = PositionManager()

        state = SymbolState(symbol="AAPL")
        state.side = "long"

        # Price moves favorably
        pm.update_excursions(state, current_price=155.0, avg_price=150.0)
        assert state.max_favorable_excursion == 5.0

        # Price moves more favorably
        pm.update_excursions(state, current_price=160.0, avg_price=150.0)
        assert state.max_favorable_excursion == 10.0

        # Price pulls back (adverse)
        pm.update_excursions(state, current_price=148.0, avg_price=150.0)
        assert state.max_adverse_excursion == -2.0

    def test_update_excursions_short(self):
        """Test excursion tracking for short position."""
        pm = PositionManager()

        state = SymbolState(symbol="AAPL")
        state.side = "short"

        # Price moves favorably (down for short)
        pm.update_excursions(state, current_price=145.0, avg_price=150.0)
        assert state.max_favorable_excursion == 5.0  # 150 - 145

        # Price moves against (up for short)
        pm.update_excursions(state, current_price=152.0, avg_price=150.0)
        assert state.max_adverse_excursion == -2.0  # 150 - 152
