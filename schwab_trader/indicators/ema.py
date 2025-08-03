from indicators.base_indicator import BaseIndicator

class EMAIndicator(BaseIndicator):
    def __init__(self, df, window=20):
        super().__init__(df)
        self.window = window

    def compute(self):
        self.df[f"EMA_{self.window}"] = self.df["Close"].ewm(span=self.window, adjust=False).mean()
        return self.df