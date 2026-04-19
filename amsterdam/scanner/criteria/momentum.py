"""
Momentum-based screening criteria.
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
class MomentumBreakout(BaseCriterion):
    """
    Scores symbols exhibiting momentum breakout characteristics:
    - Price above upper Bollinger Band
    - Price near N-day high
    - RSI in momentum zone (50-75)
    - MACD above signal line
    """

    def __init__(self, lookback: int = 20, rsi_min: float = 50, rsi_max: float = 75):
        self._lookback = lookback
        self._rsi_min = rsi_min
        self._rsi_max = rsi_max

    @property
    def name(self) -> str:
        return "momentum_breakout"

    def evaluate(self, symbol: str, provider_data: Dict[str, ProviderData]) -> float:
        tech = provider_data.get("technical")
        if not tech or tech.dataframe is None or tech.dataframe.empty:
            return 0.0

        df = tech.dataframe
        data = tech.data
        score = 0.0

        close = _safe_get(data, "close")
        if close is None:
            return 0.0

        # Bollinger breakout: close above upper band
        bb_upper = _safe_get(data, "BB_upper")
        if bb_upper and close > bb_upper:
            score += 0.3

        # Near N-day high
        if "close" in df.columns and len(df) >= self._lookback:
            high_n = df["close"].tail(self._lookback).max()
            if pd.notna(high_n) and high_n > 0:
                proximity = close / high_n
                if proximity >= 0.98:  # Within 2% of N-day high
                    score += 0.3

        # RSI in momentum zone
        rsi = _safe_get(data, "RSI")
        if rsi is not None:
            if self._rsi_min <= rsi <= self._rsi_max:
                score += 0.2

        # MACD above signal
        macd = _safe_get(data, "MACD")
        macd_signal = _safe_get(data, "MACD_signal")
        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                score += 0.2

        return min(score, 1.0)


@register_criterion
class RelativeStrength(BaseCriterion):
    """
    Compares a symbol's N-day return against a benchmark (SPY by default).
    Outperformance maps to higher scores.
    """

    def __init__(self, benchmark: str = "SPY", period: int = 20):
        self._benchmark = benchmark
        self._period = period

    @property
    def name(self) -> str:
        return "relative_strength"

    def evaluate(self, symbol: str, provider_data: Dict[str, ProviderData]) -> float:
        tech = provider_data.get("technical")
        bench_data = provider_data.get(f"technical_{self._benchmark}")

        if not tech or tech.dataframe is None or tech.dataframe.empty:
            return 0.0

        df = tech.dataframe
        if "close" not in df.columns or len(df) < self._period:
            return 0.0

        # Symbol return over period
        closes = df["close"].dropna()
        if len(closes) < self._period:
            return 0.0
        symbol_return = (closes.iloc[-1] / closes.iloc[-self._period] - 1.0) * 100

        # Benchmark return
        bench_return = 0.0
        if bench_data and bench_data.dataframe is not None and not bench_data.dataframe.empty:
            bench_closes = bench_data.dataframe["close"].dropna()
            if len(bench_closes) >= self._period:
                bench_return = (bench_closes.iloc[-1] / bench_closes.iloc[-self._period] - 1.0) * 100

        # Relative outperformance
        outperformance = symbol_return - bench_return

        # Scale: -5% or worse = 0.0, 0% = 0.5, +5% or more = 1.0
        score = (outperformance + 5.0) / 10.0
        return max(0.0, min(1.0, score))


@register_criterion
class VolumeSurge(BaseCriterion):
    """
    Scores based on current volume relative to 20-day average.
    Higher volume surge = higher score.
    """

    def __init__(self, surge_threshold: float = 1.5):
        self._surge_threshold = surge_threshold

    @property
    def name(self) -> str:
        return "volume_surge"

    def evaluate(self, symbol: str, provider_data: Dict[str, ProviderData]) -> float:
        tech = provider_data.get("technical")
        if not tech:
            return 0.0

        data = tech.data
        current_vol = _safe_get(data, "volume")
        avg_vol = _safe_get(data, "avg_volume_20")

        if not current_vol or not avg_vol or avg_vol == 0:
            return 0.0

        ratio = current_vol / avg_vol

        if ratio < self._surge_threshold:
            return 0.0
        elif ratio < 2.0:
            return 0.25
        elif ratio < 3.0:
            return 0.5
        elif ratio < 5.0:
            return 0.75
        else:
            return 1.0
