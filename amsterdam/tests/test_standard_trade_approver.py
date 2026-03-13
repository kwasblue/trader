"""
Tests for StandardTradeApprover - the core trade gating logic.

Tests the REAL implementation, not mocks.
"""
import pytest
from datetime import datetime, timezone, timedelta

from core.logic.default_trade_logic import StandardTradeApprover
from core.logic.symbol_state import SymbolState
from core.contracts.types import SignalContext
from core.enums import OrderSide


class TestStandardTradeApproverInit:
    """Test initialization."""

    def test_default_initialization(self):
        """Test default parameters."""
        approver = StandardTradeApprover()

        assert approver.cooldown_seconds == 300
        assert approver.cooldown_bars == 5
        assert approver.cooldown_mode == 'bars'
        assert approver.max_positions == 10

    def test_custom_initialization(self):
        """Test custom parameters."""
        approver = StandardTradeApprover(
            cooldown_seconds=600,
            cooldown_bars=10,
            cooldown_mode='time',
            max_positions=5
        )

        assert approver.cooldown_seconds == 600
        assert approver.cooldown_bars == 10
        assert approver.cooldown_mode == 'time'
        assert approver.max_positions == 5

    def test_multipliers_stored(self):
        """Test that SL/TP multipliers are stored correctly."""
        approver = StandardTradeApprover(
            tp_mult_low=1.0,
            tp_mult_normal=2.5,
            tp_mult_high=4.0,
            sl_mult_low=0.5,
            sl_mult_normal=1.0,
            sl_mult_high=1.5
        )

        assert approver.tp_mults["low_volatility"] == 1.0
        assert approver.tp_mults["normal"] == 2.5
        assert approver.tp_mults["high_volatility"] == 4.0
        assert approver.sl_mults["low_volatility"] == 0.5
        assert approver.sl_mults["normal"] == 1.0
        assert approver.sl_mults["high_volatility"] == 1.5


class TestShouldTrade:
    """Test the main should_trade() method."""

    @pytest.fixture
    def approver(self):
        """Create approver with bar-based cooldown."""
        return StandardTradeApprover(
            cooldown_mode='bars',
            cooldown_bars=3,
            max_positions=5
        )

    @pytest.fixture
    def flat_state(self):
        """Create a flat (no position) state."""
        state = SymbolState(symbol="AAPL")
        state.current_position = 0
        state.side = None
        return state

    @pytest.fixture
    def long_state(self):
        """Create a long position state."""
        state = SymbolState(symbol="AAPL")
        state.current_position = 100
        state.side = "long"
        state.entry_price = 150.0
        return state

    def test_hold_signal_blocked(self, approver, flat_state):
        """Hold signal (0) should be blocked."""
        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=0,  # Hold
            price=150.0,
            atr=2.0,
            regime="normal",
            timestamp=datetime.now(timezone.utc)
        )

        allowed, reason = approver.should_trade(context, flat_state, account_positions=0)

        assert allowed is False
        assert "Hold signal" in reason

    def test_invalid_price_blocked(self, approver, flat_state):
        """Invalid price should be blocked."""
        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,
            price=0,  # Invalid
            atr=2.0,
            regime="normal",
            timestamp=datetime.now(timezone.utc)
        )

        allowed, reason = approver.should_trade(context, flat_state, account_positions=0)

        assert allowed is False
        assert "Invalid price" in reason

    def test_invalid_atr_blocked(self, approver, flat_state):
        """Invalid ATR should be blocked."""
        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,
            price=150.0,
            atr=0,  # Invalid
            regime="normal",
            timestamp=datetime.now(timezone.utc)
        )

        allowed, reason = approver.should_trade(context, flat_state, account_positions=0)

        assert allowed is False
        assert "Invalid ATR" in reason

    def test_buy_signal_allowed_when_flat(self, approver, flat_state):
        """Buy signal should be allowed when flat and no restrictions."""
        # Ensure cooldown is satisfied
        approver._bars_since_trade["AAPL"] = 10

        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,  # Buy
            price=150.0,
            atr=2.0,
            regime="normal",
            timestamp=datetime.now(timezone.utc),
            market_open=True
        )

        allowed, reason = approver.should_trade(context, flat_state, account_positions=0)

        assert allowed is True
        assert reason is None

    def test_sell_signal_allowed_when_flat(self, approver, flat_state):
        """Sell signal should be allowed when flat (short entry)."""
        approver._bars_since_trade["AAPL"] = 10

        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=-1,  # Sell (short entry)
            price=150.0,
            atr=2.0,
            regime="normal",
            timestamp=datetime.now(timezone.utc),
            market_open=True
        )

        allowed, reason = approver.should_trade(context, flat_state, account_positions=0)

        assert allowed is True
        assert reason is None

    def test_in_position_always_allowed(self, approver, long_state):
        """When in position, should_trade always returns True (exit logic is in PositionManager)."""
        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=-1,  # Sell signal
            price=155.0,
            atr=2.0,
            regime="normal",
            timestamp=datetime.now(timezone.utc)
        )

        allowed, reason = approver.should_trade(context, long_state, account_positions=1)

        assert allowed is True
        assert reason is None


class TestCooldown:
    """Test cooldown enforcement."""

    def test_bar_cooldown_blocks(self):
        """Bar-based cooldown should block trades."""
        approver = StandardTradeApprover(
            cooldown_mode='bars',
            cooldown_bars=5
        )

        state = SymbolState(symbol="AAPL")
        state.current_position = 0

        # Set bars since last trade to 2 (below cooldown of 5)
        approver._bars_since_trade["AAPL"] = 2

        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,
            price=150.0,
            atr=2.0,
            regime="normal",
            timestamp=datetime.now(timezone.utc),
            market_open=True
        )

        allowed, reason = approver.should_trade(context, state, account_positions=0)

        assert allowed is False
        assert "Cooldown" in reason
        assert "bars remaining" in reason

    def test_bar_cooldown_allows_after_elapsed(self):
        """Bar-based cooldown should allow after enough bars."""
        approver = StandardTradeApprover(
            cooldown_mode='bars',
            cooldown_bars=5
        )

        state = SymbolState(symbol="AAPL")
        state.current_position = 0

        # Set bars since last trade to 6 (above cooldown of 5)
        approver._bars_since_trade["AAPL"] = 6

        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,
            price=150.0,
            atr=2.0,
            regime="normal",
            timestamp=datetime.now(timezone.utc),
            market_open=True
        )

        allowed, reason = approver.should_trade(context, state, account_positions=0)

        assert allowed is True

    def test_time_cooldown_blocks(self):
        """Time-based cooldown should block trades."""
        approver = StandardTradeApprover(
            cooldown_mode='time',
            cooldown_seconds=300  # 5 minutes
        )

        state = SymbolState(symbol="AAPL")
        state.current_position = 0
        state.last_trade_time = datetime.now(timezone.utc) - timedelta(seconds=60)  # 1 min ago

        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,
            price=150.0,
            atr=2.0,
            regime="normal",
            timestamp=datetime.now(timezone.utc),
            market_open=True
        )

        allowed, reason = approver.should_trade(context, state, account_positions=0)

        assert allowed is False
        assert "Cooldown" in reason

    def test_time_cooldown_allows_after_elapsed(self):
        """Time-based cooldown should allow after enough time."""
        approver = StandardTradeApprover(
            cooldown_mode='time',
            cooldown_seconds=300
        )

        state = SymbolState(symbol="AAPL")
        state.current_position = 0
        state.last_trade_time = datetime.now(timezone.utc) - timedelta(seconds=400)  # 6+ min ago

        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,
            price=150.0,
            atr=2.0,
            regime="normal",
            timestamp=datetime.now(timezone.utc),
            market_open=True
        )

        allowed, reason = approver.should_trade(context, state, account_positions=0)

        assert allowed is True

    def test_on_bar_increments_counter(self):
        """on_bar() should increment the bars since trade counter."""
        approver = StandardTradeApprover(cooldown_mode='bars')

        # Initialize counter
        approver.on_trade("AAPL")
        assert approver._bars_since_trade["AAPL"] == 0

        # Increment
        approver.on_bar("AAPL")
        assert approver._bars_since_trade["AAPL"] == 1

        approver.on_bar("AAPL")
        assert approver._bars_since_trade["AAPL"] == 2

    def test_on_trade_resets_counter(self):
        """on_trade() should reset the bars since trade counter."""
        approver = StandardTradeApprover(cooldown_mode='bars')

        approver._bars_since_trade["AAPL"] = 10
        approver.on_trade("AAPL")

        assert approver._bars_since_trade["AAPL"] == 0


class TestPositionLimit:
    """Test position limit enforcement."""

    def test_position_limit_blocks(self):
        """Should block when at position limit."""
        approver = StandardTradeApprover(max_positions=3)

        state = SymbolState(symbol="AAPL")
        state.current_position = 0
        approver._bars_since_trade["AAPL"] = 100  # No cooldown

        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,
            price=150.0,
            atr=2.0,
            regime="normal",
            timestamp=datetime.now(timezone.utc),
            market_open=True
        )

        # At position limit (3 positions, max is 3)
        allowed, reason = approver.should_trade(context, state, account_positions=3)

        assert allowed is False
        assert "Position limit" in reason

    def test_position_limit_allows_below(self):
        """Should allow when below position limit."""
        approver = StandardTradeApprover(max_positions=5)

        state = SymbolState(symbol="AAPL")
        state.current_position = 0
        approver._bars_since_trade["AAPL"] = 100

        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,
            price=150.0,
            atr=2.0,
            regime="normal",
            timestamp=datetime.now(timezone.utc),
            market_open=True
        )

        allowed, reason = approver.should_trade(context, state, account_positions=2)

        assert allowed is True


class TestMarketHours:
    """Test market hours enforcement."""

    def test_market_closed_blocks_by_default(self):
        """Should block when market closed (default behavior)."""
        approver = StandardTradeApprover(allow_after_hours=False)

        state = SymbolState(symbol="AAPL")
        state.current_position = 0
        approver._bars_since_trade["AAPL"] = 100

        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,
            price=150.0,
            atr=2.0,
            regime="normal",
            timestamp=datetime.now(timezone.utc),
            market_open=False  # Market closed
        )

        allowed, reason = approver.should_trade(context, state, account_positions=0)

        assert allowed is False
        assert "Market closed" in reason

    def test_market_closed_allows_with_after_hours(self):
        """Should allow when market closed if after_hours enabled."""
        approver = StandardTradeApprover(allow_after_hours=True)

        state = SymbolState(symbol="AAPL")
        state.current_position = 0
        approver._bars_since_trade["AAPL"] = 100

        context = SignalContext.from_kwargs(
            symbol="AAPL",
            signal=1,
            price=150.0,
            atr=2.0,
            regime="normal",
            timestamp=datetime.now(timezone.utc),
            market_open=False
        )

        allowed, reason = approver.should_trade(context, state, account_positions=0)

        assert allowed is True


class TestCanEnterPosition:
    """Test can_enter_position() method."""

    def test_cannot_enter_when_in_position(self):
        """Cannot enter when already in position."""
        approver = StandardTradeApprover()

        state = SymbolState(symbol="AAPL")
        state.current_position = 100
        state.side = "long"

        allowed, reason = approver.can_enter_position(
            "AAPL", state, OrderSide.BUY,
            account_positions=1, market_open=True
        )

        assert allowed is False
        assert "Already in position" in reason


class TestCanExitPosition:
    """Test can_exit_position() method."""

    def test_cannot_exit_when_flat(self):
        """Cannot exit when not in position."""
        approver = StandardTradeApprover()

        state = SymbolState(symbol="AAPL")
        state.current_position = 0

        allowed, reason = approver.can_exit_position("AAPL", state, OrderSide.SELL)

        assert allowed is False
        assert "Not in position" in reason

    def test_can_exit_long_with_sell(self):
        """Can exit long position with sell order."""
        approver = StandardTradeApprover()

        state = SymbolState(symbol="AAPL")
        state.current_position = 100
        state.side = "long"

        allowed, reason = approver.can_exit_position("AAPL", state, OrderSide.SELL)

        assert allowed is True

    def test_cannot_exit_long_with_buy(self):
        """Cannot exit long position with buy order."""
        approver = StandardTradeApprover()

        state = SymbolState(symbol="AAPL")
        state.current_position = 100
        state.side = "long"

        allowed, reason = approver.can_exit_position(
            "AAPL", state, OrderSide.BUY, is_full_exit=True
        )

        assert allowed is False
        assert "Cannot buy to exit long" in reason


class TestHelperMethods:
    """Test helper methods."""

    def test_is_in_position(self):
        """Test is_in_position helper."""
        approver = StandardTradeApprover()

        flat_state = SymbolState(symbol="AAPL")
        flat_state.current_position = 0

        long_state = SymbolState(symbol="AAPL")
        long_state.current_position = 100

        assert approver.is_in_position(flat_state) is False
        assert approver.is_in_position(long_state) is True

    def test_is_long(self):
        """Test is_long helper."""
        approver = StandardTradeApprover()

        state = SymbolState(symbol="AAPL")
        state.current_position = 100

        assert approver.is_long(state) is True
        assert approver.is_short(state) is False

    def test_is_short(self):
        """Test is_short helper."""
        approver = StandardTradeApprover()

        state = SymbolState(symbol="AAPL")
        state.current_position = -100

        assert approver.is_short(state) is True
        assert approver.is_long(state) is False

    def test_get_position_side(self):
        """Test get_position_side helper."""
        approver = StandardTradeApprover()

        long_state = SymbolState(symbol="AAPL")
        long_state.current_position = 100

        short_state = SymbolState(symbol="AAPL")
        short_state.current_position = -100

        flat_state = SymbolState(symbol="AAPL")
        flat_state.current_position = 0

        assert approver.get_position_side(long_state) == "long"
        assert approver.get_position_side(short_state) == "short"
        assert approver.get_position_side(flat_state) is None
