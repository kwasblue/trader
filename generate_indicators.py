import os

# === 1. Paths ===
BASE_DIR = r"C:\Users\kwasi\OneDrive\Documents\Personal Projects\schwab_trader"
INDICATOR_DIR = os.path.join(BASE_DIR, "indicators")
os.makedirs(INDICATOR_DIR, exist_ok=True)

# === 2. Write base_indicator.py ===
base_indicator = '''\
import pandas as pd

class BaseIndicator:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def compute(self) -> pd.DataFrame:
        raise NotImplementedError("Each indicator must implement compute().")
'''

with open(os.path.join(INDICATOR_DIR, "base_indicator.py"), "w") as f:
    f.write(base_indicator)

# === 3. Indicator definitions (add the full dictionary here) ===
indicators = {
    "sma": '''
from indicators.base_indicator import BaseIndicator

class SMAIndicator(BaseIndicator):
    def __init__(self, df, window=20):
        super().__init__(df)
        self.window = window

    def compute(self):
        self.df[f"SMA_{self.window}"] = self.df["Close"].rolling(window=self.window).mean()
        return self.df
''',

    "ema": '''
from indicators.base_indicator import BaseIndicator

class EMAIndicator(BaseIndicator):
    def __init__(self, df, window=20):
        super().__init__(df)
        self.window = window

    def compute(self):
        self.df[f"EMA_{self.window}"] = self.df["Close"].ewm(span=self.window, adjust=False).mean()
        return self.df
''',

    "macd": '''
from indicators.base_indicator import BaseIndicator

class MACDIndicator(BaseIndicator):
    def compute(self):
        short_ema = self.df['Close'].ewm(span=12, adjust=False).mean()
        long_ema = self.df['Close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = short_ema - long_ema
        self.df['MACD_Signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        return self.df
''',

    "bollinger": '''
from indicators.base_indicator import BaseIndicator

class BollingerBandsIndicator(BaseIndicator):
    def __init__(self, df, window=20):
        super().__init__(df)
        self.window = window

    def compute(self):
        tp = (self.df['Close'] + self.df['High'] + self.df['Low']) / 3
        sigma = tp.rolling(self.window).std(ddof=0)
        ma = tp.rolling(self.window).mean()
        self.df['Bollinger_Upper'] = ma + 2 * sigma
        self.df['Bollinger_Lower'] = ma - 2 * sigma
        return self.df
''',

    "rsi": '''
from indicators.base_indicator import BaseIndicator

class RSIIndicator(BaseIndicator):
    def __init__(self, df, periods=14):
        super().__init__(df)
        self.periods = periods

    def compute(self):
        delta = self.df['Close'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.ewm(com=self.periods - 1, adjust=False).mean()
        ma_down = down.ewm(com=self.periods - 1, adjust=False).mean()
        rs = ma_up / ma_down
        self.df['RSI'] = 100 - (100 / (1 + rs))
        return self.df
''',

    "atr": '''
from indicators.base_indicator import BaseIndicator
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
''',

    "vwap": '''
from indicators.base_indicator import BaseIndicator

class VWAPIndicator(BaseIndicator):
    def compute(self):
        typical_price = (self.df['High'] + self.df['Low']) / 2
        self.df['VWAP'] = (typical_price * self.df['Volume']).cumsum() / self.df['Volume'].cumsum()
        return self.df
''',

    "obv": '''
import numpy as np
from indicators.base_indicator import BaseIndicator

class OBVIndicator(BaseIndicator):
    def compute(self):
        self.df['OBV'] = (np.sign(self.df['Close'].diff()) * self.df['Volume']).fillna(0).cumsum()
        return self.df
''',

    "momentum": '''
from indicators.base_indicator import BaseIndicator

class MomentumIndicator(BaseIndicator):
    def __init__(self, df, window=10):
        super().__init__(df)
        self.window = window

    def compute(self):
        self.df['Momentum'] = self.df['Close'] - self.df['Close'].shift(self.window)
        return self.df
''',

    "roc": '''
from indicators.base_indicator import BaseIndicator

class ROCIndicator(BaseIndicator):
    def __init__(self, df, window=10):
        super().__init__(df)
        self.window = window

    def compute(self):
        self.df['ROC'] = self.df['Close'].pct_change(periods=self.window) * 100
        return self.df
''',

    "psar": '''
from indicators.base_indicator import BaseIndicator

class PSARIndicator(BaseIndicator):
    def compute(self):
        self.df['SAR'] = self.df['High'].rolling(5).max()  # Approximation
        return self.df
''',

    "price_change": '''
from indicators.base_indicator import BaseIndicator

class PriceChangeIndicator(BaseIndicator):
    def compute(self):
        self.df['Price_Change'] = self.df['Close'] - self.df['Open']
        return self.df
''',

    "percent_change": '''
from indicators.base_indicator import BaseIndicator

class PercentChangeIndicator(BaseIndicator):
    def compute(self):
        self.df['Percent_Change'] = ((self.df['Close'] - self.df['Open']) / self.df['Open']) * 100
        return self.df
'''
}

# === 4. Generate indicator files ===
for name, code in indicators.items():
    file_path = os.path.join(INDICATOR_DIR, f"{name}.py")
    with open(file_path, "w") as f:
        f.write(code.strip())

print(f"✅ Generated {len(indicators)} indicator modules in {INDICATOR_DIR}")
