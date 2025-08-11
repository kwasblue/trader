from schwab_trader.core.base.base_indicator import BaseIndicator

class SMAIndicator(BaseIndicator):
    def __init__(self, df, window=20):
        super().__init__(df)
        self.window = window

    def compute(self):
        self.df[f"SMA_{self.window}"] = self.df["Close"].rolling(window=self.window).mean()
        return self.df