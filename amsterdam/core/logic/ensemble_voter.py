"""
Ensemble Voting System for Multi-Strategy Signal Aggregation

Combines signals from multiple strategies into a single trading decision.
Supports multiple voting modes:
- majority: Signal wins if >50% of strategies agree
- unanimous: All strategies must agree
- weighted: Weighted average of signals with threshold
- any: Any single strategy signal triggers action

Usage:
    from core.logic.ensemble_voter import EnsembleVoter

    voter = EnsembleVoter(
        strategies=["sma", "rsi", "macd", "momentum"],
        mode="weighted",
        weights={"sma": 0.3, "rsi": 0.25, "macd": 0.25, "momentum": 0.2},
        threshold=0.6
    )

    # Get ensemble signal
    signal, confidence, details = voter.vote(df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class VoteResult:
    """Result of ensemble voting."""

    signal: int  # -1 (sell), 0 (hold), 1 (buy)
    confidence: float  # 0.0 to 1.0
    strategy_signals: dict[str, int]  # Individual strategy signals
    strategy_weights: dict[str, float]  # Weights used
    mode: str  # Voting mode used
    details: str  # Human-readable explanation


class EnsembleVoter:
    """
    Aggregates signals from multiple strategies using configurable voting.

    The voter loads strategy instances and generates signals from each,
    then combines them according to the specified voting mode.
    """

    def __init__(
        self,
        strategies: list[str],
        mode: str = "majority",
        weights: dict[str, float] | None = None,
        threshold: float = 0.6,
    ):
        """
        Initialize the ensemble voter.

        Args:
            strategies: List of strategy names to use
            mode: Voting mode ("majority", "unanimous", "weighted", "any")
            weights: Strategy weights for weighted mode (defaults to equal)
            threshold: Minimum weighted score to trigger signal (for weighted mode)
        """
        self.strategy_names = strategies
        self.mode = mode
        self.threshold = threshold

        # Set default equal weights if not provided
        if weights is None:
            equal_weight = 1.0 / len(strategies)
            self.weights = {s: equal_weight for s in strategies}
        else:
            # Normalize weights to sum to 1.0
            total = sum(weights.values())
            self.weights = {s: w / total for s, w in weights.items()}

        # Load strategy instances
        self._strategies: dict[str, Any] = {}
        self._load_strategies()

        logger.info(
            f"EnsembleVoter initialized: {len(self._strategies)} strategies, "
            f"mode={mode}, threshold={threshold}"
        )

    def _load_strategies(self) -> None:
        """Load strategy instances from registry."""
        try:
            from strategies.strategy_registry.strategy_registry import STRATEGY_CLASS_REGISTRY

            for name in self.strategy_names:
                if name in STRATEGY_CLASS_REGISTRY:
                    self._strategies[name] = STRATEGY_CLASS_REGISTRY[name]()
                else:
                    logger.warning(f"Strategy '{name}' not found in registry")

        except ImportError as e:
            logger.error(f"Could not import strategy registry: {e}")

    def vote(self, df: pd.DataFrame) -> VoteResult:
        """
        Generate ensemble signal from all strategies.

        Args:
            df: OHLCV DataFrame with price history

        Returns:
            VoteResult with combined signal and details
        """
        if not self._strategies:
            return VoteResult(
                signal=0,
                confidence=0.0,
                strategy_signals={},
                strategy_weights=self.weights,
                mode=self.mode,
                details="No strategies loaded",
            )

        # Collect signals from each strategy
        signals: dict[str, int] = {}
        for name, strategy in self._strategies.items():
            try:
                signal = strategy.generate_signal(df)
                # Normalize to -1, 0, 1
                if signal > 0:
                    signals[name] = 1
                elif signal < 0:
                    signals[name] = -1
                else:
                    signals[name] = 0
            except Exception as e:
                logger.warning(f"Strategy '{name}' failed: {e}")
                signals[name] = 0

        # Apply voting mode
        if self.mode == "majority":
            result = self._vote_majority(signals)
        elif self.mode == "unanimous":
            result = self._vote_unanimous(signals)
        elif self.mode == "weighted":
            result = self._vote_weighted(signals)
        elif self.mode == "any":
            result = self._vote_any(signals)
        else:
            logger.error(f"Unknown voting mode: {self.mode}")
            result = (0, 0.0, "Unknown voting mode")

        signal, confidence, details = result

        return VoteResult(
            signal=signal,
            confidence=confidence,
            strategy_signals=signals,
            strategy_weights=self.weights,
            mode=self.mode,
            details=details,
        )

    def _vote_majority(self, signals: dict[str, int]) -> tuple[int, float, str]:
        """
        Majority voting: signal wins if >50% agree.

        Returns:
            Tuple of (signal, confidence, details)
        """
        if not signals:
            return (0, 0.0, "No signals")

        buy_count = sum(1 for s in signals.values() if s > 0)
        sell_count = sum(1 for s in signals.values() if s < 0)
        total = len(signals)

        buy_pct = buy_count / total
        sell_pct = sell_count / total

        if buy_pct > 0.5:
            return (1, buy_pct, f"BUY majority: {buy_count}/{total} ({buy_pct:.0%})")
        elif sell_pct > 0.5:
            return (-1, sell_pct, f"SELL majority: {sell_count}/{total} ({sell_pct:.0%})")
        else:
            return (0, max(buy_pct, sell_pct), f"No majority: BUY={buy_count}, SELL={sell_count}, HOLD={total - buy_count - sell_count}")

    def _vote_unanimous(self, signals: dict[str, int]) -> tuple[int, float, str]:
        """
        Unanimous voting: all strategies must agree (excluding holds).

        Returns:
            Tuple of (signal, confidence, details)
        """
        if not signals:
            return (0, 0.0, "No signals")

        # Filter out holds
        active_signals = {k: v for k, v in signals.items() if v != 0}

        if not active_signals:
            return (0, 0.0, "All strategies hold")

        unique_signals = set(active_signals.values())

        if len(unique_signals) == 1:
            signal = list(unique_signals)[0]
            action = "BUY" if signal > 0 else "SELL"
            return (signal, 1.0, f"Unanimous {action}: all {len(active_signals)} active strategies agree")
        else:
            return (0, 0.0, f"No unanimity: {len(active_signals)} strategies disagree")

    def _vote_weighted(self, signals: dict[str, int]) -> tuple[int, float, str]:
        """
        Weighted voting: weighted average of signals with threshold.

        Returns:
            Tuple of (signal, confidence, details)
        """
        if not signals:
            return (0, 0.0, "No signals")

        # Calculate weighted score (-1.0 to 1.0)
        weighted_score = 0.0
        total_weight = 0.0

        for name, signal in signals.items():
            weight = self.weights.get(name, 0.0)
            weighted_score += signal * weight
            total_weight += weight

        if total_weight > 0:
            weighted_score /= total_weight

        # Confidence is absolute value of score
        confidence = abs(weighted_score)

        if weighted_score >= self.threshold:
            return (1, confidence, f"Weighted BUY: score={weighted_score:.2f} >= {self.threshold}")
        elif weighted_score <= -self.threshold:
            return (-1, confidence, f"Weighted SELL: score={weighted_score:.2f} <= -{self.threshold}")
        else:
            return (0, confidence, f"Below threshold: |{weighted_score:.2f}| < {self.threshold}")

    def _vote_any(self, signals: dict[str, int]) -> tuple[int, float, str]:
        """
        Any voting: any single signal triggers action.
        If conflicting signals, uses weighted average to break tie.

        Returns:
            Tuple of (signal, confidence, details)
        """
        if not signals:
            return (0, 0.0, "No signals")

        buy_strategies = [k for k, v in signals.items() if v > 0]
        sell_strategies = [k for k, v in signals.items() if v < 0]

        if buy_strategies and not sell_strategies:
            confidence = len(buy_strategies) / len(signals)
            return (1, confidence, f"BUY from: {', '.join(buy_strategies)}")
        elif sell_strategies and not buy_strategies:
            confidence = len(sell_strategies) / len(signals)
            return (-1, confidence, f"SELL from: {', '.join(sell_strategies)}")
        elif buy_strategies and sell_strategies:
            # Conflict - use weighted to break tie
            return self._vote_weighted(signals)
        else:
            return (0, 0.0, "All strategies hold")

    def get_strategy_info(self) -> dict[str, Any]:
        """Get information about loaded strategies."""
        return {
            "strategies": list(self._strategies.keys()),
            "weights": self.weights,
            "mode": self.mode,
            "threshold": self.threshold,
            "loaded": len(self._strategies),
            "requested": len(self.strategy_names),
        }


def create_ensemble_voter_from_config(config) -> EnsembleVoter | None:
    """
    Create an EnsembleVoter from StreamingConfig.

    Args:
        config: TradingConfig or StreamingConfig

    Returns:
        EnsembleVoter if ensemble is enabled, None otherwise
    """
    # Handle both TradingConfig and StreamingConfig
    if hasattr(config, "streaming"):
        streaming = config.streaming
    else:
        streaming = config

    if not streaming.ensemble_enabled:
        return None

    return EnsembleVoter(
        strategies=streaming.ensemble_strategies,
        mode=streaming.ensemble_mode,
        weights=streaming.ensemble_weights,
        threshold=streaming.ensemble_threshold,
    )
