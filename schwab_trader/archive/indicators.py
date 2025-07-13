import pandas as pd
import numpy as np

class TechnicalIndicators:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe
        self._validate_dataframe()

    def _validate_dataframe(self):
        """Checks if required columns are present in the DataFrame."""
        required_columns = ['Close', 'High', 'Low', 'Volume', 'Open']
        missing_columns = [col for col in required_columns if col not in self.dataframe.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in DataFrame: {', '.join(missing_columns)}")

    def ema(self, window: int) -> pd.DataFrame:
        """Calculates exponential moving average (EMA)."""
        self.dataframe[f'EMA {window}'] = self.dataframe['Close'].ewm(span=window, adjust=False).mean()
        return self.dataframe

    def sma(self, window: int) -> pd.DataFrame:
        """Calculates simple moving average (SMA)."""
        self.dataframe[f'SMA {window}'] = self.dataframe['Close'].rolling(window=window).mean()
        return self.dataframe

    def macd(self) -> pd.DataFrame:
        """Calculates MACD indicator."""
        short_ema = self.dataframe['Close'].ewm(span=12, adjust=False).mean()
        long_ema = self.dataframe['Close'].ewm(span=26, adjust=False).mean()
        self.dataframe['MACD'] = short_ema - long_ema
        self.dataframe['Signal Line'] = self.dataframe['MACD'].ewm(span=9, adjust=False).mean()
        return self.dataframe

    def bollinger_bands(self, window: int = 20) -> pd.DataFrame:
        """Calculates Bollinger Bands (upper and lower)."""
        tp = (self.dataframe['Close'] + self.dataframe['High'] + self.dataframe['Low']) / 3
        sigma = tp.rolling(window).std(ddof=0)
        moving_avg = tp.rolling(window).mean()
        self.dataframe['Bollinger Upper'] = moving_avg + 2 * sigma
        self.dataframe['Bollinger Lower'] = moving_avg - 2 * sigma
        return self.dataframe

    def rsi(self, periods: int = 14) -> pd.DataFrame:
        """Calculates Relative Strength Index (RSI)."""
        close_delta = self.dataframe['Close'].diff()
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)

        ma_up = up.ewm(com=periods - 1, adjust=False).mean()
        ma_down = down.ewm(com=periods - 1, adjust=False).mean()

        rsi = 100 - (100 / (1 + (ma_up / ma_down)))
        self.dataframe['RSI'] = rsi
        return self.dataframe

    def atr(self, window: int = 14) -> pd.DataFrame:
        """Calculates Average True Range (ATR) for volatility measurement."""
        high_low = self.dataframe['High'] - self.dataframe['Low']
        high_close = abs(self.dataframe['High'] - self.dataframe['Close'].shift())
        low_close = abs(self.dataframe['Low'] - self.dataframe['Close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.dataframe['ATR'] = true_range.rolling(window=window).mean()
        return self.dataframe

    def vwap(self) -> pd.DataFrame:
        """Calculates Volume Weighted Average Price (VWAP)."""
        self.dataframe['VWAP'] = (self.dataframe['Volume'] * (self.dataframe['High'] + self.dataframe['Low']) / 2).cumsum() / self.dataframe['Volume'].cumsum()
        return self.dataframe

    def obv(self) -> pd.DataFrame:
        """Calculates On-Balance Volume (OBV)."""
        self.dataframe['OBV'] = (np.sign(self.dataframe['Close'].diff()) * self.dataframe['Volume']).fillna(0).cumsum()
        return self.dataframe

    def momentum(self, window: int = 10) -> pd.DataFrame:
        """Calculates Momentum indicator."""
        self.dataframe['Momentum'] = self.dataframe['Close'] - self.dataframe['Close'].shift(window)
        return self.dataframe

    def roc(self, window: int = 10) -> pd.DataFrame:
        """Calculates Rate of Change (ROC)."""
        self.dataframe['ROC'] = self.dataframe['Close'].pct_change(periods=window) * 100
        return self.dataframe

    def parabolic_sar(self) -> pd.DataFrame:
        """Calculates Parabolic SAR indicator."""
        self.dataframe['SAR'] = self.dataframe['High'].rolling(5).max()  # Approximation
        return self.dataframe

    def price_change(self) -> pd.DataFrame:
        """Calculates daily price change."""
        self.dataframe['Price Change'] = self.dataframe['Close'] - self.dataframe['Open']
        return self.dataframe

    def percent_change(self) -> pd.DataFrame:
        """Calculates percent change from open to close."""
        self.dataframe['Percent Change'] = ((self.dataframe['Close'] - self.dataframe['Open']) / self.dataframe['Open']) * 100
        return self.dataframe


    def build_frame(self, sma_window: int, ema_window: int) -> pd.DataFrame:
        """Builds a DataFrame with multiple technical indicators."""
        self.sma(sma_window)
        self.ema(ema_window)
        self.macd()
        self.bollinger_bands()
        self.rsi()
        self.atr()
        self.vwap()
        self.obv()
        self.momentum()
        self.roc()
        self.parabolic_sar()
        self.price_change()
        self.percent_change()

        
        return self.dataframe
