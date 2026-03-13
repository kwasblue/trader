import numpy as np
import pandas as pd
from typing import Optional, List
from core.base.base_strategy import BaseStrategy


class RSIStrategy(BaseStrategy):
    def generate_signal(self, data):
        window = self.params.get("window", 14)
        oversold = self.params.get("oversold", 30)
        overbought = self.params.get("overbought", 70)
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        data['Signal'] = np.where(data['RSI'] < oversold, 1, np.where(data['RSI'] > overbought, -1, 0))
        if data.empty or "Signal" not in data.columns:
            return 0
        return int(data["Signal"].iloc[-1])

    def generate_signals_vectorized(self, data: pd.DataFrame) -> Optional[List[int]]:
        """Vectorized RSI signal generation for fast backtesting."""
        window = self.params.get("window", 14)
        oversold = self.params.get("oversold", 30)
        overbought = self.params.get("overbought", 70)

        close = data['Close'].values if 'Close' in data.columns else data['close'].values

        # Calculate price changes
        delta = np.diff(close, prepend=close[0])

        # Separate gains and losses
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        # Calculate rolling averages using cumsum trick for speed
        n = len(close)
        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)

        # Initial SMA for first window
        if n >= window:
            avg_gain[window-1] = np.mean(gains[:window])
            avg_loss[window-1] = np.mean(losses[:window])

            # EMA-style smoothing for subsequent values
            alpha = 1.0 / window
            for i in range(window, n):
                avg_gain[i] = (avg_gain[i-1] * (window - 1) + gains[i]) / window
                avg_loss[i] = (avg_loss[i-1] * (window - 1) + losses[i]) / window

        # Calculate RSI
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
        rsi = 100 - (100 / (1 + rs))

        # Generate signals
        signals = np.where(rsi < oversold, 1, np.where(rsi > overbought, -1, 0))
        signals[:window] = 0  # No signal during warmup

        return signals.tolist()