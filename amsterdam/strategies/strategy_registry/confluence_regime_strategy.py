"""
Confluence Regime Strategy

Thesis:
    Most edges come from NOT trading, not from better entry signals.
    This strategy only trades when multiple uncorrelated factors align:

    1. Regime alignment - Only trend-follow in trends, mean-revert in ranges
    2. Volatility contraction - Enter when vol is low (before expansion)
    3. Multi-signal confluence - Multiple indicators must agree
    4. Momentum confirmation - Price action confirms the setup

    The edge isn't in any single indicator - it's in the FILTER.
    We're looking for high-probability setups, not frequent trades.

Why this might work:
    - Volatility clustering: low vol periods precede moves
    - Regime persistence: trends and ranges tend to continue short-term
    - Confluence reduces false signals by requiring agreement
    - Adaptive parameters respond to market conditions

Why this might NOT work:
    - Fewer trades = higher variance in results
    - Regime detection lags (you catch the middle, not the start)
    - Over-filtering can miss good opportunities

Test it. Adapt it. Don't trust it blindly.
"""

import numpy as np
import pandas as pd
from core.base.base_strategy import BaseStrategy


class ConfluenceRegimeStrategy(BaseStrategy):
    """
    Multi-factor confluence strategy with regime awareness.

    Only generates signals when:
    1. Market regime is identified (trending or ranging)
    2. Volatility is below recent average (contraction before expansion)
    3. Multiple indicators agree on direction
    4. Price confirms with momentum
    """

    def __init__(
        self,
        # Regime detection
        regime_lookback: int = 50,
        trend_threshold: float = 0.6,  # ADX-like threshold

        # Volatility filter
        vol_lookback: int = 20,
        vol_percentile_max: float = 40,  # Only trade when vol is below 40th percentile

        # Indicator periods (adaptive based on vol)
        fast_period: int = 8,
        slow_period: int = 21,
        rsi_period: int = 14,

        # Confluence requirements
        min_confluence: int = 3,  # How many signals must agree (out of 4)

        **kwargs
    ):
        super().__init__(**kwargs)
        self.regime_lookback = regime_lookback
        self.trend_threshold = trend_threshold
        self.vol_lookback = vol_lookback
        self.vol_percentile_max = vol_percentile_max
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period
        self.min_confluence = min_confluence

    def generate_signal(self, data: pd.DataFrame) -> int:
        """Generate signal only when multiple factors align."""

        # Need enough data for all calculations
        min_periods = max(self.regime_lookback, self.slow_period, self.vol_lookback) + 10
        if len(data) < min_periods:
            return 0

        close = data["Close"] if "Close" in data.columns else data["close"]
        high = data["High"] if "High" in data.columns else data["high"]
        low = data["Low"] if "Low" in data.columns else data["low"]

        # 1. REGIME DETECTION
        # Using efficiency ratio: how much of the move is "signal" vs "noise"
        regime = self._detect_regime(close)

        # 2. VOLATILITY FILTER
        # Only trade when volatility is relatively low (contraction)
        if not self._volatility_contraction(close, high, low):
            return 0  # No trade in high vol

        # 3. GENERATE COMPONENT SIGNALS
        signals = []

        # Signal 1: Trend following (EMA crossover)
        ema_signal = self._ema_signal(close)
        signals.append(ema_signal)

        # Signal 2: RSI (but regime-aware)
        rsi_signal = self._rsi_signal(close, regime)
        signals.append(rsi_signal)

        # Signal 3: Price momentum (rate of change)
        momentum_signal = self._momentum_signal(close)
        signals.append(momentum_signal)

        # Signal 4: Higher timeframe trend (using longer lookback)
        htf_signal = self._higher_timeframe_signal(close)
        signals.append(htf_signal)

        # 4. CONFLUENCE CHECK
        # Only trade if enough signals agree
        buy_count = sum(1 for s in signals if s == 1)
        sell_count = sum(1 for s in signals if s == -1)

        if buy_count >= self.min_confluence:
            # Additional confirmation: regime should support direction
            if regime == "trending":
                return 1
            elif regime == "ranging" and rsi_signal == 1:
                # In ranging, only take RSI signals (mean reversion)
                return 1

        if sell_count >= self.min_confluence:
            if regime == "trending":
                return -1
            elif regime == "ranging" and rsi_signal == -1:
                return -1

        return 0  # No confluence = no trade

    def _detect_regime(self, close: pd.Series) -> str:
        """
        Detect market regime using Efficiency Ratio.

        ER = abs(price change) / sum(abs(daily changes))
        High ER = trending (price moves efficiently)
        Low ER = ranging (price moves but goes nowhere)
        """
        change = abs(close.iloc[-1] - close.iloc[-self.regime_lookback])
        volatility = close.diff().abs().iloc[-self.regime_lookback:].sum()

        if volatility == 0:
            return "unknown"

        efficiency_ratio = change / volatility

        if efficiency_ratio > self.trend_threshold:
            return "trending"
        else:
            return "ranging"

    def _volatility_contraction(self, close: pd.Series, high: pd.Series, low: pd.Series) -> bool:
        """
        Check if current volatility is below historical average.

        Thesis: Volatility clusters. Low vol periods often precede moves.
        We want to enter BEFORE the expansion, not during it.
        """
        # Use ATR as volatility measure
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(self.vol_lookback).mean()
        current_atr = atr.iloc[-1]

        # Calculate percentile of current ATR vs recent history
        recent_atr = atr.iloc[-self.regime_lookback:]
        percentile = (recent_atr < current_atr).sum() / len(recent_atr) * 100

        return percentile <= self.vol_percentile_max

    def _ema_signal(self, close: pd.Series) -> int:
        """EMA crossover signal."""
        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()

        if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
            return 1
        elif ema_fast.iloc[-1] < ema_slow.iloc[-1]:
            return -1
        return 0

    def _rsi_signal(self, close: pd.Series, regime: str) -> int:
        """
        RSI signal - but regime aware.

        Trending: RSI confirms momentum (>50 = bullish, <50 = bearish)
        Ranging: RSI for mean reversion (<30 = buy, >70 = sell)
        """
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()

        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        if regime == "trending":
            # Momentum confirmation
            if current_rsi > 55:
                return 1
            elif current_rsi < 45:
                return -1
        else:
            # Mean reversion
            if current_rsi < 30:
                return 1
            elif current_rsi > 70:
                return -1

        return 0

    def _momentum_signal(self, close: pd.Series) -> int:
        """Rate of change momentum."""
        roc = (close.iloc[-1] - close.iloc[-self.fast_period]) / close.iloc[-self.fast_period]

        # Threshold based on recent volatility
        recent_roc = close.pct_change(self.fast_period).iloc[-self.regime_lookback:]
        threshold = recent_roc.std()

        if roc > threshold:
            return 1
        elif roc < -threshold:
            return -1
        return 0

    def _higher_timeframe_signal(self, close: pd.Series) -> int:
        """
        Proxy for higher timeframe trend.
        Using longer SMA as a stand-in for daily trend on intraday.
        """
        sma_long = close.rolling(self.regime_lookback).mean()

        if close.iloc[-1] > sma_long.iloc[-1]:
            return 1
        elif close.iloc[-1] < sma_long.iloc[-1]:
            return -1
        return 0

    def generate_signals_vectorized(self, data: pd.DataFrame) -> list[int] | None:
        """Vectorized version for backtesting."""

        min_periods = max(self.regime_lookback, self.slow_period, self.vol_lookback) + 10
        if len(data) < min_periods:
            return None

        close = data["Close"] if "Close" in data.columns else data["close"]
        high = data["High"] if "High" in data.columns else data["high"]
        low = data["Low"] if "Low" in data.columns else data["low"]

        n = len(data)
        signals = np.zeros(n, dtype=int)

        # Pre-calculate indicators
        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 1e-10)))

        # ATR for volatility
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.vol_lookback).mean()

        # Longer SMA
        sma_long = close.rolling(self.regime_lookback).mean()

        # ROC
        roc = close.pct_change(self.fast_period)
        roc_std = roc.rolling(self.regime_lookback).std()

        for i in range(min_periods, n):
            # Efficiency ratio for regime
            change = abs(close.iloc[i] - close.iloc[i - self.regime_lookback])
            volatility = close.diff().abs().iloc[i - self.regime_lookback:i].sum()
            er = change / (volatility + 1e-10)
            regime = "trending" if er > self.trend_threshold else "ranging"

            # Volatility filter
            recent_atr = atr.iloc[i - self.regime_lookback:i]
            vol_percentile = (recent_atr < atr.iloc[i]).sum() / len(recent_atr) * 100
            if vol_percentile > self.vol_percentile_max:
                continue  # Skip high vol

            # Collect signals
            sig_ema = 1 if ema_fast.iloc[i] > ema_slow.iloc[i] else (-1 if ema_fast.iloc[i] < ema_slow.iloc[i] else 0)

            current_rsi = rsi.iloc[i]
            if regime == "trending":
                sig_rsi = 1 if current_rsi > 55 else (-1 if current_rsi < 45 else 0)
            else:
                sig_rsi = 1 if current_rsi < 30 else (-1 if current_rsi > 70 else 0)

            threshold = roc_std.iloc[i]
            sig_mom = 1 if roc.iloc[i] > threshold else (-1 if roc.iloc[i] < -threshold else 0)

            sig_htf = 1 if close.iloc[i] > sma_long.iloc[i] else (-1 if close.iloc[i] < sma_long.iloc[i] else 0)

            # Confluence
            sigs = [sig_ema, sig_rsi, sig_mom, sig_htf]
            buy_count = sum(1 for s in sigs if s == 1)
            sell_count = sum(1 for s in sigs if s == -1)

            if buy_count >= self.min_confluence:
                if regime == "trending" or sig_rsi == 1:
                    signals[i] = 1
            elif sell_count >= self.min_confluence:
                if regime == "trending" or sig_rsi == -1:
                    signals[i] = -1

        return signals.tolist()
