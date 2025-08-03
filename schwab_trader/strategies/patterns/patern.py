import pandas as pd
import numpy as np

class PatternDetector:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def detect_moving_average_crossover(self, short_window: int, long_window: int):
        """
        Detects moving average crossovers.
        """
        self.data['Short_MA'] = self.data['Close'].rolling(window=short_window, min_periods=1).mean()
        self.data['Long_MA'] = self.data['Close'].rolling(window=long_window, min_periods=1).mean()
        
        self.data['Signal'] = 0
        self.data['Signal'][short_window:] = np.where(self.data['Short_MA'][short_window:] > self.data['Long_MA'][short_window:], 1, 0)
        
        self.data['Position'] = self.data['Signal'].diff()
        
        return self.data

    def detect_head_and_shoulders(self):
        """
        Detects head and shoulders pattern.
        """
        self.data['Head_And_Shoulders'] = 0
        for i in range(2, len(self.data) - 2):
            left_shoulder = self.data['Close'][i-2]
            head = self.data['Close'][i]
            right_shoulder = self.data['Close'][i+2]
            if head > left_shoulder and head > right_shoulder and left_shoulder == right_shoulder:
                self.data.at[i, 'Head_And_Shoulders'] = 1
        return self.data

    def detect_double_top(self):
        """
        Detects double top pattern.
        """
        self.data['Double_Top'] = 0
        for i in range(1, len(self.data) - 1):
            first_top = self.data['Close'][i-1]
            second_top = self.data['Close'][i+1]
            if first_top == second_top and self.data['Close'][i] < first_top:
                self.data.at[i, 'Double_Top'] = 1
        return self.data

    def detect_double_bottom(self):
        """
        Detects double bottom pattern.
        """
        self.data['Double_Bottom'] = 0
        for i in range(1, len(self.data) - 1):
            first_bottom = self.data['Close'][i-1]
            second_bottom = self.data['Close'][i+1]
            if first_bottom == second_bottom and self.data['Close'][i] > first_bottom:
                self.data.at[i, 'Double_Bottom'] = 1
        return self.data

    def detect_doji(self):
        """
        Detects doji candlestick pattern.
        """
        self.data['Doji'] = np.where(abs(self.data['Close'] - self.data['Open']) / (self.data['High'] - self.data['Low']) < 0.1, 1, 0)
        return self.data

    def detect_hammer(self):
        """
        Detects hammer candlestick pattern.
        """
        self.data['Hammer'] = np.where(
            ((self.data['High'] - self.data['Low']) > 3 * (self.data['Open'] - self.data['Close'])) &
            ((self.data['Close'] - self.data['Low']) / (.001 + self.data['High'] - self.data['Low']) > 0.6) &
            ((self.data['Open'] - self.data['Low']) / (.001 + self.data['High'] - self.data['Low']) > 0.6), 1, 0)
        return self.data

    def detect_hanging_man(self):
        """
        Detects hanging man candlestick pattern.
        """
        self.data['Hanging_Man'] = np.where(
            ((self.data['High'] - self.data['Low']) > 3 * (self.data['Open'] - self.data['Close'])) &
            ((self.data['High'] - self.data['Close']) / (.001 + self.data['High'] - self.data['Low']) > 0.6) &
            ((self.data['High'] - self.data['Open']) / (.001 + self.data['High'] - self.data['Low']) > 0.6), 1, 0)
        return self.data

    def detect_bullish_engulfing(self):
        """
        Detects bullish engulfing candlestick pattern.
        """
        self.data['Bullish_Engulfing'] = np.where(
            (self.data['Close'].shift(1) < self.data['Open'].shift(1)) &
            (self.data['Open'] < self.data['Close'].shift(1)) &
            (self.data['Close'] > self.data['Open'].shift(1)), 1, 0)
        return self.data

    def detect_bearish_engulfing(self):
        """
        Detects bearish engulfing candlestick pattern.
        """
        self.data['Bearish_Engulfing'] = np.where(
            (self.data['Close'].shift(1) > self.data['Open'].shift(1)) &
            (self.data['Open'] > self.data['Close'].shift(1)) &
            (self.data['Close'] < self.data['Open'].shift(1)), 1, 0)
        return self.data

    def detect_volume_spikes(self, threshold=1.5):
        """
        Detects volume spikes.
        """
        self.data['Volume_Spike'] = np.where(self.data['Volume'] > threshold * self.data['Volume'].rolling(window=20).mean(), 1, 0)
        return self.data

    def detect_volume_divergence(self):
        """
        Detects volume divergence.
        """
        self.data['Volume_Divergence'] = 0
        for i in range(1, len(self.data)):
            if self.data['Close'][i] > self.data['Close'][i-1] and self.data['Volume'][i] < self.data['Volume'][i-1]:
                self.data.at[i, 'Volume_Divergence'] = 1
            elif self.data['Close'][i] < self.data['Close'][i-1] and self.data['Volume'][i] > self.data['Volume'][i-1]:
                self.data.at[i, 'Volume_Divergence'] = 1
        return self.data



# Usage
""" 
data = pd.read_csv('path_to_your_stock_data.csv')  # Ensure your data has columns like 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'
detector = PatternDetector(data)
result = detector.detect_moving_average_crossover(short_window=50, long_window=200)
result = detector.detect_head_and_shoulders()
result = detector.detect_double_top()
result = detector.detect_double_bottom()
result = detector.detect_doji()
result = detector.detect_hammer()
result = detector.detect_hanging_man()
result = detector.detect_bullish_engulfing()
result = detector.detect_bearish_engulfing()
result = detector.detect_volume_spikes()
result = detector.detect_volume_divergence()

print(result[['Date', 'Close', 'Short_MA', 'Long_MA', 'Position', 'Head_And_Shoulders', 'Double_Top', 'Double_Bottom', 'Doji', 'Hammer', 'Hanging_Man', 'Bullish_Engulfing', 'Bearish_Engulfing', 'Volume_Spike', 'Volume_Divergence']])
 """