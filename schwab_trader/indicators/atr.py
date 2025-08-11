from schwab_trader.core.base.base_indicator import BaseIndicator
import pandas as pd

class ATRIndicator(BaseIndicator):
    def __init__(self, df, window=14):
        super().__init__(df)
        self.window = window

    def compute(self):
        high_low = self.df['High'] - self.df['Low']
        high_close = (self.df['High'] - self.df['Close'].shift()).abs()
        low_close = (self.df['Low'] - self.df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df['ATR'] = tr.rolling(window=self.window).mean()
        return self.df