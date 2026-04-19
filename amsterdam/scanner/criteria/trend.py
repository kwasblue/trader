"""
Trend-following screening criteria.
"""
from __future__ import annotations

import logging
from typing import Dict

import pandas as pd

from scanner.providers.base import ProviderData
from scanner.criteria.base import BaseCriterion, register_criterion

logger = logging.getLogger(__name__)


def _safe_get(data: Dict, key: str, default=None):
    val = data.get(key, default)
    if val is not None and pd.notna(val):
        return float(val)
    return default


@register_criterion
class TrendFollowing(BaseCriterion):
    """
    Evaluates trend alignment across multiple signals:
    - Price above SMA 20
    - Price above SMA 50 (if available)
    - MACD positive and above signal
    - Parabolic SAR below price (bullish)
    """

    @property
    def name(self) -> str:
        return "trend_following"

    def evaluate(self, symbol: str, provider_data: Dict[str, ProviderData]) -> float:
        tech = provider_data.get("technical")
        if not tech:
            return 0.0

        data = tech.data
        close = _safe_get(data, "close")
        if close is None:
            return 0.0

        components = 0
        score = 0.0

        # Price above SMA 20
        sma = _safe_get(data, "SMA")
        if sma is not None:
            components += 1
            if close > sma:
                score += 1.0

        # Price above SMA 50
        sma_50 = _safe_get(data, "SMA_50")
        if sma_50 is not None:
            components += 1
            if close > sma_50:
                score += 1.0

        # MACD positive and above signal
        macd = _safe_get(data, "MACD")
        macd_signal = _safe_get(data, "MACD_signal")
        if macd is not None and macd_signal is not None:
            components += 1
            if macd > 0 and macd > macd_signal:
                score += 1.0

        # PSAR below price (bullish)
        psar = _safe_get(data, "PSAR")
        if psar is not None:
            components += 1
            if close > psar:
                score += 1.0

        if components == 0:
            return 0.0

        return score / components


@register_criterion
class MovingAverageCross(BaseCriterion):
    """
    Detects recent golden cross (SMA 20 crossing above SMA 50).
    More recent crossovers score higher.
    """

    def __init__(self, lookback: int = 10):
        self._lookback = lookback

    @property
    def name(self) -> str:
        return "ma_crossover"

    def evaluate(self, symbol: str, provider_data: Dict[str, ProviderData]) -> float:
        tech = provider_data.get("technical")
        if not tech or tech.dataframe is None or tech.dataframe.empty:
            return 0.0

        df = tech.dataframe
        if "close" not in df.columns or len(df) < 50:
            return 0.0

        # Use SMA_20 column from indicators, or compute on the fly
        sma_50 = df["close"].rolling(window=50).mean()
        sma_20_col = "SMA_20" if "SMA_20" in df.columns else "SMA"
        sma_20 = df[sma_20_col] if sma_20_col in df.columns else df["close"].rolling(window=20).mean()

        # Look for golden cross in the last N bars
        tail = min(self._lookback, len(df) - 1)
        for i in range(1, tail + 1):
            idx = len(df) - i
            prev_idx = idx - 1
            if prev_idx < 0:
                break

            curr_20 = sma_20.iloc[idx]
            curr_50 = sma_50.iloc[idx]
            prev_20 = sma_20.iloc[prev_idx]
            prev_50 = sma_50.iloc[prev_idx]

            if pd.isna(curr_20) or pd.isna(curr_50) or pd.isna(prev_20) or pd.isna(prev_50):
                continue

            # Golden cross: SMA 20 crosses above SMA 50
            if prev_20 <= prev_50 and curr_20 > curr_50:
                # More recent = higher score
                recency = 1.0 - (i - 1) / tail
                return max(0.3, recency)

        # No crossover found but check if SMA20 > SMA50 (already in trend)
        if len(sma_20) > 0 and len(sma_50) > 0:
            last_20 = sma_20.iloc[-1]
            last_50 = sma_50.iloc[-1]
            if pd.notna(last_20) and pd.notna(last_50) and last_20 > last_50:
                return 0.2

        return 0.0
