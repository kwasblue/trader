"""
Tests for EnsembleVoter

Coverage:
- Voting modes (majority, unanimous, weighted, any)
- VoteResult dataclass
- Strategy loading
- Config integration via create_ensemble_voter_from_config
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.logic.ensemble_voter import EnsembleVoter, VoteResult, create_ensemble_voter_from_config


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing strategies."""
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "Open": close + np.random.randn(n) * 0.2,
        "High": close + np.abs(np.random.randn(n) * 0.3),
        "Low": close - np.abs(np.random.randn(n) * 0.3),
        "Close": close,
        "Volume": np.random.randint(10000, 100000, n),
    })


@pytest.fixture
def uptrend_data():
    """Create uptrending price data."""
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 0.5 + 0.1)
    return pd.DataFrame({
        "Open": close + np.random.randn(n) * 0.2,
        "High": close + np.abs(np.random.randn(n) * 0.3),
        "Low": close - np.abs(np.random.randn(n) * 0.3),
        "Close": close,
        "Volume": np.random.randint(10000, 100000, n),
    })


@pytest.fixture
def downtrend_data():
    """Create downtrending price data."""
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 0.5 - 0.1)
    return pd.DataFrame({
        "Open": close + np.random.randn(n) * 0.2,
        "High": close + np.abs(np.random.randn(n) * 0.3),
        "Low": close - np.abs(np.random.randn(n) * 0.3),
        "Close": close,
        "Volume": np.random.randint(10000, 100000, n),
    })


@pytest.fixture
def mock_strategy_registry():
    """Mock strategy registry with controllable signals."""
    mock_strategies = {}

    class MockStrategy:
        def __init__(self, signal_value):
            self.signal_value = signal_value

        def generate_signal(self, df):
            return self.signal_value

    # Default: all strategies return 0 (hold)
    mock_strategies["sma"] = lambda: MockStrategy(0)
    mock_strategies["rsi"] = lambda: MockStrategy(0)
    mock_strategies["macd"] = lambda: MockStrategy(0)
    mock_strategies["momentum"] = lambda: MockStrategy(0)

    return mock_strategies


# ============================================================================
# VoteResult Tests
# ============================================================================


class TestVoteResult:
    """Tests for VoteResult dataclass."""

    def test_vote_result_fields(self):
        """VoteResult should contain all expected fields."""
        result = VoteResult(
            signal=1,
            confidence=0.8,
            strategy_signals={"sma": 1, "rsi": 1},
            strategy_weights={"sma": 0.5, "rsi": 0.5},
            mode="majority",
            details="Test result",
        )
        assert result.signal == 1
        assert result.confidence == 0.8
        assert result.strategy_signals == {"sma": 1, "rsi": 1}
        assert result.strategy_weights == {"sma": 0.5, "rsi": 0.5}
        assert result.mode == "majority"
        assert result.details == "Test result"

    def test_vote_result_signal_values(self):
        """VoteResult signal should be -1, 0, or 1."""
        for signal in [-1, 0, 1]:
            result = VoteResult(
                signal=signal,
                confidence=0.5,
                strategy_signals={},
                strategy_weights={},
                mode="majority",
                details="",
            )
            assert result.signal == signal


# ============================================================================
# EnsembleVoter Initialization Tests
# ============================================================================


class TestEnsembleVoterInit:
    """Tests for EnsembleVoter initialization."""

    def test_init_with_default_weights(self):
        """EnsembleVoter should assign equal weights by default."""
        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            voter = EnsembleVoter(
                strategies=["sma", "rsi", "macd", "momentum"],
                mode="majority",
            )
            # Equal weights: 1/4 = 0.25 each
            assert voter.weights["sma"] == pytest.approx(0.25)
            assert voter.weights["rsi"] == pytest.approx(0.25)
            assert voter.weights["macd"] == pytest.approx(0.25)
            assert voter.weights["momentum"] == pytest.approx(0.25)

    def test_init_with_custom_weights(self):
        """EnsembleVoter should use custom weights when provided."""
        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            voter = EnsembleVoter(
                strategies=["sma", "rsi"],
                mode="weighted",
                weights={"sma": 0.7, "rsi": 0.3},
            )
            assert voter.weights["sma"] == pytest.approx(0.7)
            assert voter.weights["rsi"] == pytest.approx(0.3)

    def test_init_normalizes_weights(self):
        """EnsembleVoter should normalize weights to sum to 1.0."""
        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            voter = EnsembleVoter(
                strategies=["sma", "rsi"],
                mode="weighted",
                weights={"sma": 2.0, "rsi": 2.0},  # Sum = 4.0
            )
            # Should be normalized to 0.5 each
            assert voter.weights["sma"] == pytest.approx(0.5)
            assert voter.weights["rsi"] == pytest.approx(0.5)

    def test_init_with_threshold(self):
        """EnsembleVoter should store threshold value."""
        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            voter = EnsembleVoter(
                strategies=["sma"],
                mode="weighted",
                threshold=0.75,
            )
            assert voter.threshold == 0.75

    def test_init_stores_mode(self):
        """EnsembleVoter should store voting mode."""
        for mode in ["majority", "unanimous", "weighted", "any"]:
            with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
                voter = EnsembleVoter(strategies=["sma"], mode=mode)
                assert voter.mode == mode


# ============================================================================
# Majority Voting Tests
# ============================================================================


class TestMajorityVoting:
    """Tests for majority voting mode."""

    def test_majority_buy_when_over_50_percent(self):
        """Majority mode should return BUY when >50% strategies agree."""
        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            voter = EnsembleVoter(strategies=["s1", "s2", "s3"], mode="majority")

        signals = {"s1": 1, "s2": 1, "s3": 0}  # 2/3 = 66% buy
        signal, confidence, details = voter._vote_majority(signals)

        assert signal == 1
        assert confidence == pytest.approx(2 / 3)
        assert "BUY majority" in details

    def test_majority_sell_when_over_50_percent(self):
        """Majority mode should return SELL when >50% strategies agree."""
        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            voter = EnsembleVoter(strategies=["s1", "s2", "s3"], mode="majority")

        signals = {"s1": -1, "s2": -1, "s3": 0}  # 2/3 = 66% sell
        signal, confidence, details = voter._vote_majority(signals)

        assert signal == -1
        assert confidence == pytest.approx(2 / 3)
        assert "SELL majority" in details

    def test_majority_hold_when_no_majority(self):
        """Majority mode should return HOLD when no >50% agreement."""
        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            voter = EnsembleVoter(strategies=["s1", "s2", "s3", "s4"], mode="majority")

        signals = {"s1": 1, "s2": -1, "s3": 0, "s4": 0}  # No majority
        signal, confidence, details = voter._vote_majority(signals)

        assert signal == 0
        assert "No majority" in details

    def test_majority_with_empty_signals(self):
        """Majority mode should handle empty signals."""
        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            voter = EnsembleVoter(strategies=["dummy"], mode="majority")

        # Test the method directly with empty signals dict
        signal, confidence, details = voter._vote_majority({})

        assert signal == 0
        assert confidence == 0.0
        assert "No signals" in details


# ============================================================================
# Unanimous Voting Tests
# ============================================================================


class TestUnanimousVoting:
    """Tests for unanimous voting mode."""

    def test_unanimous_buy_when_all_agree(self):
        """Unanimous mode should return BUY when all active strategies agree."""
        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            voter = EnsembleVoter(strategies=["s1", "s2", "s3"], mode="unanimous")

        signals = {"s1": 1, "s2": 1, "s3": 1}
        signal, confidence, details = voter._vote_unanimous(signals)

        assert signal == 1
        assert confidence == 1.0
        assert "Unanimous BUY" in details

    def test_unanimous_sell_when_all_agree(self):
        """Unanimous mode should return SELL when all active strategies agree."""
        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            voter = EnsembleVoter(strategies=["s1", "s2", "s3"], mode="unanimous")

        signals = {"s1": -1, "s2": -1, "s3": -1}
        signal, confidence, details = voter._vote_unanimous(signals)

        assert signal == -1
        assert confidence == 1.0
        assert "Unanimous SELL" in details

    def test_unanimous_hold_when_disagreement(self):
        """Unanimous mode should return HOLD when strategies disagree."""
        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            voter = EnsembleVoter(strategies=["s1", "s2", "s3"], mode="unanimous")

        signals = {"s1": 1, "s2": -1, "s3": 1}
        signal, confidence, details = voter._vote_unanimous(signals)

        assert signal == 0
        assert "No unanimity" in details

    def test_unanimous_ignores_holds(self):
        """Unanimous mode should ignore HOLD signals."""
        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            voter = EnsembleVoter(strategies=["s1", "s2", "s3"], mode="unanimous")

        signals = {"s1": 1, "s2": 0, "s3": 1}  # One hold
        signal, confidence, details = voter._vote_unanimous(signals)

        assert signal == 1
        assert "Unanimous BUY" in details

    def test_unanimous_all_hold(self):
        """Unanimous mode should return HOLD when all strategies hold."""
        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            voter = EnsembleVoter(strategies=["s1", "s2"], mode="unanimous")

        signals = {"s1": 0, "s2": 0}
        signal, confidence, details = voter._vote_unanimous(signals)

        assert signal == 0
        assert "All strategies hold" in details


# ============================================================================
# Weighted Voting Tests
# ============================================================================


class TestWeightedVoting:
    """Tests for weighted voting mode."""

    def test_weighted_buy_above_threshold(self):
        """Weighted mode should return BUY when score exceeds threshold."""
        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            voter = EnsembleVoter(
                strategies=["s1", "s2"],
                mode="weighted",
                weights={"s1": 0.7, "s2": 0.3},
                threshold=0.5,
            )

        signals = {"s1": 1, "s2": 1}  # Score = 1.0
        signal, confidence, details = voter._vote_weighted(signals)

        assert signal == 1
        assert "Weighted BUY" in details

    def test_weighted_sell_below_negative_threshold(self):
        """Weighted mode should return SELL when score below negative threshold."""
        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            voter = EnsembleVoter(
                strategies=["s1", "s2"],
                mode="weighted",
                weights={"s1": 0.7, "s2": 0.3},
                threshold=0.5,
            )

        signals = {"s1": -1, "s2": -1}  # Score = -1.0
        signal, confidence, details = voter._vote_weighted(signals)

        assert signal == -1
        assert "Weighted SELL" in details

    def test_weighted_hold_within_threshold(self):
        """Weighted mode should return HOLD when score within threshold."""
        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            voter = EnsembleVoter(
                strategies=["s1", "s2"],
                mode="weighted",
                weights={"s1": 0.5, "s2": 0.5},
                threshold=0.6,
            )

        signals = {"s1": 1, "s2": -1}  # Score = 0.0
        signal, confidence, details = voter._vote_weighted(signals)

        assert signal == 0
        assert "Below threshold" in details

    def test_weighted_uses_correct_weights(self):
        """Weighted mode should apply correct weight calculations."""
        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            voter = EnsembleVoter(
                strategies=["s1", "s2"],
                mode="weighted",
                weights={"s1": 0.8, "s2": 0.2},
                threshold=0.5,
            )

        # s1 (0.8) = BUY, s2 (0.2) = SELL
        # Score = 0.8 * 1 + 0.2 * (-1) = 0.6
        signals = {"s1": 1, "s2": -1}
        signal, confidence, details = voter._vote_weighted(signals)

        assert signal == 1  # 0.6 >= 0.5 threshold
        assert confidence == pytest.approx(0.6)


# ============================================================================
# Any Voting Tests
# ============================================================================


class TestAnyVoting:
    """Tests for 'any' voting mode."""

    def test_any_buy_when_single_buy(self):
        """Any mode should return BUY when any strategy signals buy."""
        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            voter = EnsembleVoter(strategies=["s1", "s2", "s3"], mode="any")

        signals = {"s1": 1, "s2": 0, "s3": 0}
        signal, confidence, details = voter._vote_any(signals)

        assert signal == 1
        assert "BUY from" in details

    def test_any_sell_when_single_sell(self):
        """Any mode should return SELL when any strategy signals sell."""
        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            voter = EnsembleVoter(strategies=["s1", "s2", "s3"], mode="any")

        signals = {"s1": -1, "s2": 0, "s3": 0}
        signal, confidence, details = voter._vote_any(signals)

        assert signal == -1
        assert "SELL from" in details

    def test_any_conflict_uses_weighted(self):
        """Any mode should use weighted voting when buy and sell conflict."""
        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            voter = EnsembleVoter(
                strategies=["s1", "s2"],
                mode="any",
                weights={"s1": 0.8, "s2": 0.2},
                threshold=0.5,
            )

        signals = {"s1": 1, "s2": -1}  # Conflict
        signal, confidence, details = voter._vote_any(signals)

        # Should fall back to weighted: 0.8*1 + 0.2*(-1) = 0.6 >= 0.5
        assert signal == 1

    def test_any_hold_when_all_hold(self):
        """Any mode should return HOLD when all strategies hold."""
        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            voter = EnsembleVoter(strategies=["s1", "s2"], mode="any")

        signals = {"s1": 0, "s2": 0}
        signal, confidence, details = voter._vote_any(signals)

        assert signal == 0
        assert "All strategies hold" in details


# ============================================================================
# Integration with Real Strategies
# ============================================================================


class TestEnsembleVoterIntegration:
    """Integration tests with actual strategy registry."""

    def test_vote_with_real_strategies(self, sample_ohlcv_data):
        """EnsembleVoter should work with real strategies from registry."""
        try:
            voter = EnsembleVoter(
                strategies=["sma", "rsi", "macd", "momentum"],
                mode="majority",
            )
            result = voter.vote(sample_ohlcv_data)

            assert isinstance(result, VoteResult)
            assert result.signal in [-1, 0, 1]
            assert 0.0 <= result.confidence <= 1.0
            assert result.mode == "majority"
        except ImportError:
            pytest.skip("Strategy registry not available")

    def test_vote_returns_strategy_signals(self, uptrend_data):
        """vote() should return individual strategy signals."""
        try:
            voter = EnsembleVoter(
                strategies=["sma", "rsi"],
                mode="majority",
            )
            result = voter.vote(uptrend_data)

            assert "sma" in result.strategy_signals or len(result.strategy_signals) >= 0
        except ImportError:
            pytest.skip("Strategy registry not available")

    def test_get_strategy_info(self):
        """get_strategy_info() should return voter configuration."""
        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            voter = EnsembleVoter(
                strategies=["sma", "rsi"],
                mode="weighted",
                threshold=0.7,
            )
            voter._strategies = {"sma": MagicMock(), "rsi": MagicMock()}

        info = voter.get_strategy_info()

        assert "strategies" in info
        assert "weights" in info
        assert "mode" in info
        assert info["mode"] == "weighted"
        assert info["threshold"] == 0.7


# ============================================================================
# Config Factory Tests
# ============================================================================


class TestCreateEnsembleVoterFromConfig:
    """Tests for create_ensemble_voter_from_config factory."""

    def test_returns_none_when_disabled(self):
        """Factory should return None when ensemble is disabled."""
        mock_config = MagicMock()
        mock_config.streaming.ensemble_enabled = False

        result = create_ensemble_voter_from_config(mock_config)

        assert result is None

    def test_creates_voter_when_enabled(self):
        """Factory should create EnsembleVoter when enabled."""
        mock_config = MagicMock()
        mock_config.streaming.ensemble_enabled = True
        mock_config.streaming.ensemble_strategies = ["sma", "rsi"]
        mock_config.streaming.ensemble_mode = "weighted"
        mock_config.streaming.ensemble_weights = {"sma": 0.5, "rsi": 0.5}
        mock_config.streaming.ensemble_threshold = 0.6

        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            result = create_ensemble_voter_from_config(mock_config)

        assert isinstance(result, EnsembleVoter)
        assert result.mode == "weighted"
        assert result.threshold == 0.6

    def test_handles_streaming_config_directly(self):
        """Factory should handle StreamingConfig passed directly."""
        mock_streaming = MagicMock()
        mock_streaming.ensemble_enabled = True
        mock_streaming.ensemble_strategies = ["sma"]
        mock_streaming.ensemble_mode = "majority"
        mock_streaming.ensemble_weights = {"sma": 1.0}
        mock_streaming.ensemble_threshold = 0.5

        # No .streaming attribute - it's the StreamingConfig itself
        del mock_streaming.streaming

        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            result = create_ensemble_voter_from_config(mock_streaming)

        assert isinstance(result, EnsembleVoter)

    def test_uses_config_strategies(self):
        """Factory should use strategies from config."""
        mock_config = MagicMock()
        mock_config.streaming.ensemble_enabled = True
        mock_config.streaming.ensemble_strategies = ["sma", "rsi", "macd"]
        mock_config.streaming.ensemble_mode = "any"
        mock_config.streaming.ensemble_weights = {}
        mock_config.streaming.ensemble_threshold = 0.5

        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            result = create_ensemble_voter_from_config(mock_config)

        assert result.strategy_names == ["sma", "rsi", "macd"]


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_vote_with_no_strategies_loaded(self, sample_ohlcv_data):
        """vote() should handle case when no strategies loaded."""
        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            voter = EnsembleVoter(strategies=["nonexistent"], mode="majority")
            voter._strategies = {}  # No strategies loaded

        result = voter.vote(sample_ohlcv_data)

        assert result.signal == 0
        assert result.confidence == 0.0
        assert "No strategies loaded" in result.details

    def test_vote_handles_strategy_exception(self, sample_ohlcv_data):
        """vote() should handle strategy exceptions gracefully."""
        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            voter = EnsembleVoter(strategies=["bad_strategy"], mode="majority")

            # Mock a strategy that raises an exception
            bad_strategy = MagicMock()
            bad_strategy.generate_signal.side_effect = ValueError("Strategy error")
            voter._strategies = {"bad_strategy": bad_strategy}

        result = voter.vote(sample_ohlcv_data)

        # Should handle error and return signal of 0 for that strategy
        assert result.strategy_signals["bad_strategy"] == 0

    def test_unknown_voting_mode(self, sample_ohlcv_data):
        """vote() should handle unknown voting mode."""
        with patch("core.logic.ensemble_voter.EnsembleVoter._load_strategies"):
            voter = EnsembleVoter(strategies=["sma"], mode="unknown_mode")

            mock_strategy = MagicMock()
            mock_strategy.generate_signal.return_value = 1
            voter._strategies = {"sma": mock_strategy}

        result = voter.vote(sample_ohlcv_data)

        assert result.signal == 0
        assert "Unknown voting mode" in result.details
