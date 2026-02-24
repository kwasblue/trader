import pandas as pd
import numpy as np
from typing import Optional

from indicators.ema import EMAIndicator
from indicators.sma import SMAIndicator
from indicators.macd import MACDIndicator
from indicators.rsi import RSIIndicator
from indicators.atr import ATRIndicator
from indicators.vwap import VWAPIndicator
from indicators.obv import OBVIndicator
from indicators.momentum import MomentumIndicator
from indicators.roc import ROCIndicator
from indicators.bollinger import BollingerBandsIndicator
from indicators.psar import PSARIndicator
from indicators.price_change import PriceChangeIndicator
from indicators.percent_change import PercentChangeIndicator


class TechnicalIndicators:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def apply_all(self, sma_window=20, ema_window=20):
        """Original method - kept for backward compatibility."""
        df = self.df.copy()
        df = SMAIndicator(df, window=sma_window).compute()
        df = EMAIndicator(df, window=ema_window).compute()
        df = MACDIndicator(df).compute()
        df = RSIIndicator(df).compute()
        df = ATRIndicator(df).compute()
        df = VWAPIndicator(df).compute()
        df = OBVIndicator(df).compute()
        df = MomentumIndicator(df).compute()
        df = ROCIndicator(df).compute()
        df = BollingerBandsIndicator(df).compute()
        df = PSARIndicator(df).compute()
        df = PriceChangeIndicator(df).compute()
        df = PercentChangeIndicator(df).compute()
        return df

    def apply_all_fast(
        self,
        sma_window: int = 20,
        ema_window: int = 20,
        atr_window: int = 14,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_window: int = 20,
        bb_std: float = 2.0,
        momentum_period: int = 10,
        roc_period: int = 10,
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Optimized vectorized indicator computation.

        Uses NumPy operations directly instead of per-indicator class overhead.
        Avoids unnecessary DataFrame copies and intermediate allocations.

        Args:
            sma_window: SMA period
            ema_window: EMA period
            atr_window: ATR period
            rsi_period: RSI period
            macd_fast: MACD fast period
            macd_slow: MACD slow period
            macd_signal: MACD signal period
            bb_window: Bollinger Bands window
            bb_std: Bollinger Bands standard deviations
            momentum_period: Momentum period
            roc_period: Rate of Change period
            inplace: If True, modifies self.df directly (faster)

        Returns:
            DataFrame with all indicators computed
        """
        df = self.df if inplace else self.df.copy()

        # Extract arrays once for efficiency
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values if 'Volume' in df.columns else np.zeros(len(df))
        n = len(close)

        # --- SMA ---
        df[f'SMA_{sma_window}'] = pd.Series(close).rolling(window=sma_window).mean().values

        # --- EMA ---
        df[f'EMA_{ema_window}'] = pd.Series(close).ewm(span=ema_window, adjust=False).mean().values

        # --- MACD (vectorized) ---
        ema_fast = pd.Series(close).ewm(span=macd_fast, adjust=False).mean()
        ema_slow = pd.Series(close).ewm(span=macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()
        df['MACD'] = macd_line.values
        df['MACD_Signal'] = signal_line.values
        df['MACD_Histogram'] = (macd_line - signal_line).values

        # --- RSI (vectorized) ---
        delta = np.diff(close, prepend=close[0])
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gains).ewm(com=rsi_period - 1, adjust=False).mean().values
        avg_loss = pd.Series(losses).ewm(com=rsi_period - 1, adjust=False).mean().values
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
        df['RSI'] = 100.0 - (100.0 / (1.0 + rs))

        # --- ATR (vectorized - avoid pd.concat) ---
        if n > 1:
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            tr_full = np.concatenate([[np.nan], tr])
            df['ATR'] = pd.Series(tr_full).rolling(window=atr_window).mean().values
        else:
            df['ATR'] = np.nan

        # --- VWAP (vectorized) ---
        typical_price = (high + low + close) / 3.0
        cum_tp_vol = np.cumsum(typical_price * volume)
        cum_vol = np.cumsum(volume)
        df['VWAP'] = np.divide(cum_tp_vol, cum_vol, out=np.zeros_like(cum_tp_vol), where=cum_vol != 0)

        # --- OBV (vectorized) ---
        obv = np.zeros(n)
        if n > 1:
            price_change = np.sign(np.diff(close))
            obv[1:] = np.cumsum(price_change * volume[1:])
        df['OBV'] = obv

        # --- Momentum ---
        momentum = np.full(n, np.nan)
        if n > momentum_period:
            momentum[momentum_period:] = close[momentum_period:] - close[:-momentum_period]
        df['Momentum'] = momentum

        # --- ROC (Rate of Change) ---
        roc = np.full(n, np.nan)
        if n > roc_period:
            prev_close = close[:-roc_period]
            roc[roc_period:] = ((close[roc_period:] - prev_close) / prev_close) * 100.0
        df['ROC'] = roc

        # --- Bollinger Bands ---
        sma_bb = pd.Series(close).rolling(window=bb_window).mean()
        std_bb = pd.Series(close).rolling(window=bb_window).std()
        df['Bollinger_Middle'] = sma_bb.values
        df['Bollinger_Upper'] = (sma_bb + bb_std * std_bb).values
        df['Bollinger_Lower'] = (sma_bb - bb_std * std_bb).values

        # --- Price Change ---
        df['Price_Change'] = np.concatenate([[0], np.diff(close)])

        # --- Percent Change ---
        pct_change = np.zeros(n)
        if n > 1:
            pct_change[1:] = (close[1:] - close[:-1]) / close[:-1] * 100.0
        df['Percent_Change'] = pct_change

        return df
