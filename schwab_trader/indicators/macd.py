from schwab_trader.core.base.base_indicator import BaseIndicator

class MACDIndicator(BaseIndicator):
    def compute(self):
        short_ema = self.df['Close'].ewm(span=12, adjust=False).mean()
        long_ema = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = short_ema - long_ema
        self.df['MACD_Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        return self.df