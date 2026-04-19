"""
Tests for TradeGate

Coverage:
- Entry permission checks
- State management
- Regime persistence tracking
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.logic.trade_gate import TradeGate


class TestTradeGateInit:
    """Tests for TradeGate initialization."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        gate = TradeGate()
        assert gate.max_layers == 1
        assert gate.min_bars_between_layers == 2
        assert gate.regime_min_persist_bars == 3
        assert gate.flip_cooldown_bars == 1

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        gate = TradeGate(max_layers=3, min_bars_between_layers=5, regime_min_persist_bars=10, flip_cooldown_bars=3)
        assert gate.max_layers == 3
        assert gate.min_bars_between_layers == 5
        assert gate.regime_min_persist_bars == 10
        assert gate.flip_cooldown_bars == 3


class TestGateState:
    """Tests for gate state management."""

    @pytest.fixture
    def gate(self):
        return TradeGate()

    def test_get_state_creates_new(self, gate):
        """Test get_state creates new state for unknown symbol."""
        state = gate.get_state("AAPL")
        assert state is not None

    def test_get_state_returns_same(self, gate):
        """Test get_state returns same state for known symbol."""
        state1 = gate.get_state("AAPL")
        state2 = gate.get_state("AAPL")
        assert state1 is state2

    def test_state_has_expected_attributes(self, gate):
        """Test state has expected attributes."""
        state = gate.get_state("AAPL")
        assert hasattr(state, "did_action_this_bar")
        assert hasattr(state, "last_bar_id")


class TestCanEnter:
    """Tests for entry permission checks."""

    @pytest.fixture
    def gate(self):
        return TradeGate(max_layers=2, min_bars_between_layers=3, regime_min_persist_bars=2, flip_cooldown_bars=2)

    def test_can_enter_returns_tuple(self, gate):
        """Test can_enter returns tuple of (bool, reason)."""
        now = datetime.now(timezone.utc)
        # Initialize the state with on_new_bar first
        gate.on_new_bar("AAPL", bar_id=1, regime="trending")

        result = gate.can_enter(symbol="AAPL", side="long", ts=now, bar_id=1, regime="trending")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)

    def test_blocks_after_action_this_bar(self, gate):
        """Test blocks if already acted this bar."""
        now = datetime.now(timezone.utc)
        state = gate.get_state("AAPL")
        state.did_action_this_bar = True
        state.last_regime = "trending"

        can_enter, reason = gate.can_enter(symbol="AAPL", side="long", ts=now, bar_id=1, regime="trending")
        assert can_enter is False
        assert "already" in reason.lower()


class TestOnNewBar:
    """Tests for bar transition handling."""

    @pytest.fixture
    def gate(self):
        return TradeGate(regime_min_persist_bars=3)

    def test_on_new_bar_resets_action_flag(self, gate):
        """Test on_new_bar resets did_action_this_bar."""
        state = gate.get_state("AAPL")
        state.did_action_this_bar = True

        gate.on_new_bar("AAPL", bar_id=2, regime="trending")

        assert state.did_action_this_bar is False

    def test_on_new_bar_updates_regime_persist(self, gate):
        """Test on_new_bar updates regime persistence counter."""
        gate.on_new_bar("AAPL", bar_id=1, regime="trending")
        gate.on_new_bar("AAPL", bar_id=2, regime="trending")
        gate.on_new_bar("AAPL", bar_id=3, regime="trending")

        persist = gate.get_regime_persist("AAPL")
        assert persist >= 3

    def test_on_new_bar_resets_on_regime_change(self, gate):
        """Test regime persist resets on regime change."""
        gate.on_new_bar("AAPL", bar_id=1, regime="trending")
        gate.on_new_bar("AAPL", bar_id=2, regime="trending")
        gate.on_new_bar("AAPL", bar_id=3, regime="ranging")  # Change regime

        persist = gate.get_regime_persist("AAPL")
        assert persist == 1


class TestMultipleSymbols:
    """Tests for handling multiple symbols."""

    @pytest.fixture
    def gate(self):
        return TradeGate()

    def test_symbols_independent(self, gate):
        """Test symbols are tracked independently."""
        gate.on_new_bar("AAPL", bar_id=1, regime="trending")
        gate.on_new_bar("MSFT", bar_id=1, regime="ranging")

        aapl_state = gate.get_state("AAPL")
        msft_state = gate.get_state("MSFT")

        assert aapl_state is not msft_state
        assert aapl_state.last_regime == "trending"
        assert msft_state.last_regime == "ranging"

    def test_action_on_one_symbol_doesnt_block_other(self, gate):
        """Test action on one symbol doesn't block other."""
        # Initialize both
        gate.on_new_bar("AAPL", bar_id=1, regime="trending")
        gate.on_new_bar("MSFT", bar_id=1, regime="trending")

        # Mark AAPL as acted
        aapl_state = gate.get_state("AAPL")
        aapl_state.did_action_this_bar = True

        # MSFT should not be affected
        msft_state = gate.get_state("MSFT")
        assert msft_state.did_action_this_bar is False
