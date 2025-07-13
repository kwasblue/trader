import pandas as pd
import numpy as np
from ta.trend import ADXIndicator
import inspect

class Strategy:
    def __init__(self):
        self.strategy = None

    def simple_moving_average_strategy(self, data: pd.DataFrame, short_window=20, long_window=50, **kwargs) -> pd.DataFrame:
        """
        Simple Moving Average (SMA) Strategy.
        Buy when the short-term SMA crosses above the long-term SMA.
        Sell when the short-term SMA crosses below the long-term SMA.
        """
        data['SMA_Short'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
        data['SMA_Long'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
        data['Signal'] = np.where(data['SMA_Short'] > data['SMA_Long'], 1, -1)
        return data
    
    def exponential_moving_average_strategy(self, data: pd.DataFrame, short_window=20, long_window=50, **kwargs) -> pd.DataFrame:
        """
        Exponential Moving Average (EMA) Strategy.
        Buy when the short-term EMA crosses above the long-term EMA.
        Sell when the short-term EMA crosses below the long-term EMA.
        """
        data['EMA_Short'] = data['Close'].ewm(span=short_window, adjust=False).mean()
        data['EMA_Long'] = data['Close'].ewm(span=long_window, adjust=False).mean()
        data['Signal'] = np.where(data['EMA_Short'] > data['EMA_Long'], 1, -1)
        return data

    def mean_reversion_strategy(self, data: pd.DataFrame, window=10, **kwargs) -> pd.DataFrame:
        """
        Mean Reversion Strategy.
        Buy when the price is below the rolling mean.
        Sell when the price is above the rolling mean.
        """
        data['Rolling_Mean'] = data['Close'].rolling(window=window, min_periods=1, ).mean()
        data['Signal'] = np.where(data['Close'] > data['Rolling_Mean'], -1, 1)
        return data

    def breakout_strategy(self, data: pd.DataFrame, window=20, **kwargs) -> pd.DataFrame:
        """
        Breakout Strategy.
        Buy when the price breaks above the high of the past window days.
        Sell when the price breaks below the low of the past window days.
        """
        data['Rolling_High'] = data['High'].rolling(window=window, min_periods=1).max()
        data['Rolling_Low'] = data['Low'].rolling(window=window, min_periods=1).min()
        data['Signal'] = np.where(data['Close'] > data['Rolling_High'].shift(1), 1, 
                                  np.where(data['Close'] < data['Rolling_Low'].shift(1), -1, 0))
        return data

    def momentum_strategy(self, data: pd.DataFrame, window=10,**kwargs ) -> pd.DataFrame:
        """
        Momentum Strategy.
        Buy when the momentum (rate of change) is positive.
        Sell when the momentum is negative.
        """
        data['Momentum'] = data['Close'].diff(window)
        data['Signal'] = np.where(data['Momentum'] > 0, 1, -1)
        return data

    def rsi_strategy(self, data: pd.DataFrame, window=14, oversold=30, overbought=70, **kwargs) -> pd.DataFrame:
        """
        Relative Strength Index (RSI) Strategy.
        Buy when RSI crosses below the oversold threshold.
        Sell when RSI crosses above the overbought threshold.
        """
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        data['Signal'] = np.where(data['RSI'] < oversold, 1, np.where(data['RSI'] > overbought, -1, 0))
        return data

    def macd_strategy(self, data: pd.DataFrame, fast_window=12, slow_window=26, signal_window=9, **kwargs) -> pd.DataFrame:
        """
        MACD (Moving Average Convergence Divergence) Strategy.
        Buy when MACD crosses above the signal line.
        Sell when MACD crosses below the signal line.
        """
        data['EMA_Fast'] = data['Close'].ewm(span=fast_window, min_periods=1, adjust=False).mean()
        data['EMA_Slow'] = data['Close'].ewm(span=slow_window, min_periods=1, adjust=False).mean()
        data['MACD'] = data['EMA_Fast'] - data['EMA_Slow']
        data['MACD_Signal'] = data['MACD'].ewm(span=signal_window, min_periods=1, adjust=False).mean()

        data['Signal'] = np.where(data['MACD'] > data['MACD_Signal'], 1, -1)
        return data

    def bollinger_bands_strategy(self, data: pd.DataFrame, window=20, num_std=2, **kwargs) -> pd.DataFrame:
        """
        Bollinger Bands Strategy.
        Buy when the price crosses below the lower band.
        Sell when the price crosses above the upper band.
        """
        data['Rolling_Mean'] = data['Close'].rolling(window=window).mean()
        data['Rolling_Std'] = data['Close'].rolling(window=window).std()

        data['Upper_Band'] = data['Rolling_Mean'] + (data['Rolling_Std'] * num_std)
        data['Lower_Band'] = data['Rolling_Mean'] - (data['Rolling_Std'] * num_std)

        data['Signal'] = np.where(data['Close'] < data['Lower_Band'], 1, np.where(data['Close'] > data['Upper_Band'], -1, 0))
        return data

    def vwap_strategy(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
        data['Signal'] = np.where(data['Close'] < data['VWAP'], 1, -1)
        return data

    def donchian_channel_strategy(self, data: pd.DataFrame, window=20, **kwargs) -> pd.DataFrame:
        data['Donchian_High'] = data['High'].rolling(window=window).max()
        data['Donchian_Low'] = data['Low'].rolling(window=window).min()
        data['Signal'] = np.where(data['Close'] > data['Donchian_High'].shift(1), 1, 
                                  np.where(data['Close'] < data['Donchian_Low'].shift(1), -1, 0))
        return data

    def parabolic_sar_strategy(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        from ta.trend import PSARIndicator
        psar = PSARIndicator(data['High'], data['Low'], data['Close'])
        data['PSAR'] = psar.psar()
        data['Signal'] = np.where(data['Close'] > data['PSAR'], 1, -1)
        return data

    def stochastic_oscillator_strategy(self, data: pd.DataFrame, k_window=14, d_window=3, **kwargs) -> pd.DataFrame:
        data['Lowest_Low'] = data['Low'].rolling(window=k_window).min()
        data['Highest_High'] = data['High'].rolling(window=k_window).max()
        data['%K'] = 100 * (data['Close'] - data['Lowest_Low']) / (data['Highest_High'] - data['Lowest_Low'])
        data['%D'] = data['%K'].rolling(window=d_window).mean()
        data['Signal'] = np.where((data['%K'] > data['%D']) & (data['%K'] < 20), 1, 
                                  np.where((data['%K'] < data['%D']) & (data['%K'] > 80), -1, 0))
        return data

    def ichimoku_cloud_strategy(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        data['Tenkan_Sen'] = (data['High'].rolling(window=9).max() + data['Low'].rolling(window=9).min()) / 2
        data['Kijun_Sen'] = (data['High'].rolling(window=26).max() + data['Low'].rolling(window=26).min()) / 2
        data['Senkou_Span_A'] = ((data['Tenkan_Sen'] + data['Kijun_Sen']) / 2).shift(26)
        data['Senkou_Span_B'] = ((data['High'].rolling(window=52).max() + data['Low'].rolling(window=52).min()) / 2).shift(26)
        data['Signal'] = np.where(data['Close'] > data['Senkou_Span_A'], 1, 
                                  np.where(data['Close'] < data['Senkou_Span_B'], -1, 0))
        return data

    def adx_strategy(self, data: pd.DataFrame, window=14, threshold=25, **kwargs) -> pd.DataFrame:
        
        adx = ADXIndicator(data['High'], data['Low'], data['Close'], window=window)
        data['ADX'] = adx.adx()
        data['+DI'] = adx.adx_pos()
        data['-DI'] = adx.adx_neg()
        data['Signal'] = np.where((data['ADX'] > threshold) & (data['+DI'] > data['-DI']), 1, 
                                  np.where((data['ADX'] > threshold) & (data['-DI'] > data['+DI']), -1, 0))
        return data

    def moving_average_crossover(self, data: pd.DataFrame, short_window: int, long_window: int, **kwargs) -> pd.DataFrame:
        # Example strategy
        data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
        data['Long_MA'] = data['Close'].rolling(window=long_window).mean()
        data['Signal'] = np.where(data['Short_MA'] > data['Long_MA'], 1, -1)
        return data

    def trend_momentum_strategy(self, df:pd.DataFrame, **kwargs):
        """
        Apply the Trend Momentum strategy using EMA and RSI indicators.
        """
        # Calculate short-term and long-term EMAs
        short_ema = df['Close'].ewm(span=12, adjust=False).mean()  # Short-term EMA (12 periods)
        long_ema = df['Close'].ewm(span=26, adjust=False).mean()   # Long-term EMA (26 periods)

        # Calculate the RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Generate signals: Buy (1) or Sell (-1) based on EMA and RSI
        signals = pd.DataFrame(index=df.index)
        signals['Signal'] = 0  # Default to no position (0)
        
        # Buy signal: Short EMA crosses above Long EMA and RSI < 70
        signals.loc[(short_ema > long_ema) & (rsi < 70), 'Signal'] = 1

        # Sell signal: Short EMA crosses below Long EMA and RSI > 30
        signals.loc[(short_ema < long_ema) & (rsi > 30), 'Signal'] = -1
        
        # Add the signals to the dataframe
        df['Signal'] = signals['Signal']
        
        return df
    
    def combined_strategy(self, data: pd.DataFrame, strategy_methods: list, weights: list = None, combine_method: str = 'vote', **kwargs) -> pd.DataFrame:
        """
        Combine multiple strategies dynamically.
        
        Parameters:
            data (pd.DataFrame): Input DataFrame.
            strategy_methods (list): A list of strategy method names (strings) to apply.
            weights (list): Optional list of weights for each strategy signal (if combine_method=='weighted').
            combine_method (str): 'vote' to use majority vote or 'weighted' for a weighted sum.
            **kwargs: Additional parameters passed to each strategy method.
        
        Returns:
            pd.DataFrame: DataFrame with a composite 'Signal' column.
        """
        signals = []
        # Run each strategy and extract its Signal column
        for method_name in strategy_methods:
            strategy_function = getattr(self, method_name, None)
            if strategy_function is None:
                raise ValueError(f"Strategy '{method_name}' is not implemented.")
            
            # Use the inspect module to filter kwargs for only accepted parameters:
            sig = inspect.signature(strategy_function)
            valid_kwargs = {key: value for key, value in kwargs.items() if key in sig.parameters}
            
            strat_result = strategy_function(data.copy(), **valid_kwargs)
            if 'Signal' not in strat_result.columns:
                raise ValueError(f"Strategy '{method_name}' did not return a 'Signal' column.")
            signals.append(strat_result['Signal'])
        
        # Combine signals
        signals_array = np.array(signals)  # shape: (num_strategies, num_rows)
        if combine_method == 'vote':
            # Sum signals and take the sign: if majority are positive, the result is positive (1), etc.
            combined_signal = np.sign(np.sum(signals_array, axis=0))
        elif combine_method == 'weighted':
            if weights is None or len(weights) != len(strategy_methods):
                raise ValueError("Weights must be provided and match the number of strategy methods when using 'weighted' combine_method.")
            combined_signal = np.sign(np.sum(signals_array * np.array(weights)[:, None], axis=0))
        else:
            raise ValueError("Unknown combine method. Use 'vote' or 'weighted'.")
        
        data['Signal'] = combined_signal
        return data
    
    def logistic_regression_strategy(self, data: pd.DataFrame, model=None, **kwargs) -> pd.DataFrame:
        """
        Logistic Regression Strategy.
        Applies a trained logistic regression model (pipeline) to generate buy/sell signals
        based on probability thresholds.

        :param data: DataFrame with input features for prediction.
        :param model: A pre-trained sklearn Pipeline (with preprocessor, scaler, and model).
        :return: DataFrame with 'Signal' column.
        """
        if model is None:
            raise ValueError("A trained logistic regression model must be provided.")

        try:
            # Get transformed feature names from pipeline's preprocessor
            preprocessor = model.named_steps['preprocessor']
            raw_features = preprocessor.feature_names_in_
            
            # Ensure the data has all the required features
            X = data[raw_features].copy()

            # Transform and predict
            X_transformed = preprocessor.transform(X)
            preds_proba = model.named_steps['model'].predict_proba(X_transformed)[:, 1]

            # Create signal based on probability thresholds
            data['Signal'] = 0
            #data.loc[preds_proba > 0.55, 'Signal'] = 1
            #data.loc[preds_proba < 0.45, 'Signal'] = -1
            data['Signal'] = np.where(preds_proba > 0.52, 1, np.where(preds_proba < 0.48, -1, 0))
            data.to_csv('test.csv')

            return data

        except Exception as e:
            print(f"Model prediction failed: {e}")
            data['Signal'] = 0
            return data


    def _apply_strategy(self, data: pd.DataFrame, strategy_name: str, **kwargs):
        """
        Apply a specific strategy to the data using the provided parameters.
        Filters extra kwargs if the strategy method does not accept them.
        """
        strategy_function = getattr(self, strategy_name, None)
        if not strategy_function:
            raise ValueError(f"Strategy '{strategy_name}' is not implemented.")
        
        # Get the signature of the strategy function
        sig = inspect.signature(strategy_function)
        
        # Filter kwargs to only include those accepted by the function
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        
        # Call the strategy function with the filtered kwargs
        data = strategy_function(data, **valid_kwargs)
        data.to_csv('test.csv')
        return data

            
    def _calculate_returns(self, data: pd.DataFrame, initial_capital=10000, transaction_cost=0.001, risk_free_rate=0.02) -> pd.DataFrame:
        """
        Calculate returns, cumulative portfolio value, and adjust for transaction costs and risk-free rate.

        :param data: DataFrame containing the 'Signal' column and stock data
        :param initial_capital: Starting capital for the strategy
        :param transaction_cost: Transaction cost as a fraction (e.g., 0.001 for 0.1%)
        :param risk_free_rate: Annualized risk-free rate to adjust strategy returns
        :return: DataFrame with additional columns for returns and portfolio value
        """
        if 'Signal' not in data.columns:
            raise ValueError("The DataFrame must contain a 'Signal' column to calculate returns.")

        # Calculate daily percentage returns
        data['Daily_Return'] = data['Close'].pct_change()

        # Shift the Signal column to represent the previous day's position
        data['Position'] = data['Signal'].shift(1)

        # Calculate strategy returns with transaction costs
        data['Strategy_Return'] = data['Daily_Return'] * data['Position']
        data['Transaction_Cost'] = transaction_cost * abs(data['Signal'].diff().fillna(0))
        data['Adjusted_Strategy_Return'] = data['Strategy_Return'] - data['Transaction_Cost']

        # Convert annualized risk-free rate to daily and adjust for it
        daily_risk_free_rate = (1 + risk_free_rate) ** (1 / 252) - 1
        data['Excess_Return'] = data['Adjusted_Strategy_Return'] - daily_risk_free_rate

        # Calculate cumulative returns and portfolio value
        data['Cumulative_Strategy_Return'] = (1 + data['Adjusted_Strategy_Return']).cumprod()
        data['Portfolio_Value'] = initial_capital * data['Cumulative_Strategy_Return']

        return data
