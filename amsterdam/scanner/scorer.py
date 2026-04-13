"""
Multi-factor scoring and ranking for scanned symbols.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from scanner.result import SymbolScore

logger = logging.getLogger(__name__)


class SymbolScorer:
    """Computes weighted composite scores and ranks symbols."""

    def __init__(
        self,
        weights: Dict[str, float],
        trade_threshold: float = 0.7,
        watch_threshold: float = 0.4,
        max_trade_symbols: int = 10,
        max_watch_symbols: int = 20,
    ):
        self._weights = weights
        self._trade_threshold = trade_threshold
        self._watch_threshold = watch_threshold
        self._max_trade = max_trade_symbols
        self._max_watch = max_watch_symbols

    def score(
        self,
        symbol: str,
        criteria_scores: Dict[str, float],
        metadata: Dict[str, Any],
    ) -> SymbolScore:
        """Compute weighted composite score for a symbol."""
        total_weight = 0.0
        weighted_sum = 0.0

        for criterion_name, raw_score in criteria_scores.items():
            weight = self._weights.get(criterion_name, 0.0)
            weighted_sum += raw_score * weight
            total_weight += weight

        total_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        total_score = max(0.0, min(1.0, total_score))

        if total_score >= self._trade_threshold:
            recommendation = "trade"
        elif total_score >= self._watch_threshold:
            recommendation = "watch"
        else:
            recommendation = "skip"

        return SymbolScore(
            symbol=symbol,
            total_score=round(total_score, 4),
            criteria_scores=criteria_scores,
            metadata=metadata,
            recommendation=recommendation,
        )

    def rank(self, scores: List[SymbolScore]) -> List[SymbolScore]:
        """Sort by score descending and enforce max caps."""
        sorted_scores = sorted(scores, key=lambda s: s.total_score, reverse=True)

        trade_count = 0
        watch_count = 0

        for s in sorted_scores:
            if s.recommendation == "trade":
                if trade_count >= self._max_trade:
                    s.recommendation = "watch"
                else:
                    trade_count += 1

            if s.recommendation == "watch":
                if watch_count >= self._max_watch:
                    s.recommendation = "skip"
                else:
                    watch_count += 1

        return sorted_scores
