from schwab_trader.core.base.base_indicator import BaseIndicator

class VWAPIndicator(BaseIndicator):
    def compute(self):
        typical_price = (self.df['High'] + self.df['Low']) / 2
        self.df['VWAP'] = (typical_price * self.df['Volume']).cumsum() / self.df['Volume'].cumsum()
        return self.df