from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.logger import Logger
from utils.configloader import ConfigLoader
from indicators.technical_indicators import TechnicalIndicators
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.fft import fft, ifft
import numpy as np
import pandas as pd


import pandas as pd
class Processor:
    def __init__(self, stock=None, frame=None):
        self.stock = stock
        self.frame = frame
        self.config = ConfigLoader().load_config()
        self.logs_dir = self.config["folders"]["logs"]
        self.logger = Logger('app.log', 'Processor', log_dir=self.logs_dir).get_logger()
        self.scaler = None  # Initialized during scaling

    def update(self, stock, frame):
        """Update the stock symbol and frame data."""
        self.stock = stock
        self.frame = frame

    def dataframe(self) -> pd.DataFrame:
        return self.frame.copy()

    def clean_stock_data(self) -> pd.DataFrame:
        try:
            df = self.dataframe()
            df.dropna(inplace=True)
            # df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
            df.sort_values(by='datetime', inplace=True)
            df.drop_duplicates(inplace=True)
            df.rename(
                columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'datetime': 'Date'
                },
                inplace=True,
            )
            return df
        except KeyError as e:
            self.logger.error(f"KeyError: Missing key {e} in the DataFrame. Please ensure all required columns are present.")
            return pd.DataFrame()

    def apply_indicators(self, sma_window: int, ema_window: int) -> pd.DataFrame:
        try:
            df = self.clean_stock_data()
            indicators = TechnicalIndicators(df)
            return indicators.apply_all(sma_window=sma_window, ema_window=ema_window)
        except ValueError as e:
            self.logger.error(f"ValueError: {e}. Please ensure the window sizes and data are valid.")
            return pd.DataFrame()

    def feature_engineering(self) -> pd.DataFrame:
        df = self.clean_stock_data()

        # Vectorized calculations
        high_low_diff = df['High'].values - df['Low'].values
        close_open_diff = df['Close'].values - df['Open'].values
        daily_return = df['Close'].pct_change().values
        rolling_mean_10 = df['Close'].rolling(window=10).mean().values
        rolling_std_10 = df['Close'].rolling(window=10).std().values

        # Building features
        features = {
            'Price_Range': high_low_diff,
            'Daily_Return': daily_return,
            'Rolling_Mean_10': rolling_mean_10,
            'Rolling_Std_10': rolling_std_10,
            'Close_Open_Diff': close_open_diff,
        }

        # Lagged features
        for lag in range(1, 6):
            features[f'Lag_Close_{lag}'] = df['Close'].shift(lag).values

        # Convert to DataFrame
        features_df = pd.DataFrame(features, index=df.index)
        features_df.dropna(inplace=True)
        return features_df

    def normalizing_scaling(self, df: pd.DataFrame, method='standard') -> pd.DataFrame:
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Unsupported scaling method")

        scaled_features = self.scaler.fit_transform(df.drop(columns=['Date', 'Close']))
        scaled_df = pd.DataFrame(scaled_features, columns=[f"{col}" for col in df.columns[2:]], index=df.index)
        scaled_df = pd.concat([df[['Date', 'Close']], scaled_df], axis=1)
        return scaled_df

    def denoise_fft(self, signal: np.ndarray, threshold: float = 0.1, low_freq_cutoff: float = 0.1, high_freq_cutoff: float = 0.9) -> np.ndarray:
        """
        Apply Fourier Transform based denoising with a cutoff for low and high frequencies.
        This is an enhanced version where a frequency band is retained based on thresholds.

        Parameters:
        - signal: The input signal (1D numpy array).
        - threshold: The threshold for zeroing out frequency components (complex magnitude).
        - low_freq_cutoff: Low frequency cutoff to retain components.
        - high_freq_cutoff: High frequency cutoff to retain components.

        Returns:
        - The denoised signal after applying the inverse FFT.
        """
        # Apply FFT to the signal
        fft_signal = fft(signal)
        
        # Get the frequency bins corresponding to the FFT components
        n = len(signal)
        freqs = np.fft.fftfreq(n)
        
        # Apply the frequency thresholding by zeroing out components outside the specified range
        fft_signal[(freqs < low_freq_cutoff) | (freqs > high_freq_cutoff)] = 0

        # Optionally apply a magnitude threshold for high-frequency noise rejection
        fft_signal[np.abs(fft_signal) < threshold] = 0
        
        # Inverse FFT to recover the denoised signal
        return np.abs(ifft(fft_signal))

    def apply_signal_processing(self, df: pd.DataFrame, columns_to_denoise: list) -> pd.DataFrame:
        """Apply denoising to specific columns of the DataFrame."""
        df_copy = df.copy()
        for col in columns_to_denoise:
            df_copy[col] = self.denoise_fft(df_copy[col].values)  # Using FFT-based denoising
        return df_copy

    def pca_feature_selection(self, df, n_components):
        # Ensure n_components is within the valid range
        n_components = min(n_components, df.drop(columns=['Date', 'Close']).shape[1])  # Exclude 'Date' and 'Close'

        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(df.drop(columns=['Date', 'Close']))  # Exclude 'Date' and 'Close' from PCA
        pca_df = pd.DataFrame(pca_features, columns=[f'PC{i+1}' for i in range(n_components)], index=df.index)
        pca_df = pd.concat([df[['Date', 'Close']], pca_df], axis=1)
        return pca_df


    def process(self, sma_window: int, ema_window: int, scaling_method: str = None, pca_components: int = 5, denoise_cols: list = None) -> pd.DataFrame:
        try:
            clean_data = self.clean_stock_data()
            indicators_data = self.apply_indicators(sma_window=sma_window, ema_window=ema_window).drop(
                columns=['Date', 'Open', 'Close', 'Volume', 'High', 'Low']
            )
            engineered_data = self.feature_engineering()
            
            # Adjust PCA components based on the number of features
            #num_features = clean_data.shape[1]  # Number of features in clean_data
            #pca_components = min(pca_components, num_features)  # Max out PCA components to available features

            # Apply signal processing (denoising)
            #if denoise_cols:
               # clean_data = self.apply_signal_processing(clean_data, denoise_cols)

            # Perform PCA
            # pca_data = self.pca_feature_selection(clean_data, n_components=pca_components)

            # Apply scaling
            scaled_data = self.normalizing_scaling(clean_data, method=scaling_method)

            # Combine all processed data
            combined_data = clean_data.merge(indicators_data, left_index=True, right_index=True, suffixes=('', '_ind'))
            combined_data = combined_data.merge(engineered_data, left_index=True, right_index=True, suffixes=('', '_eng'))
            #combined_data = combined_data.merge(pca_data, left_index=True, right_index=True, suffixes=('', '_pca'))
            combined_data = combined_data.merge(scaled_data, left_index=True, right_index=True, suffixes=('', '_scaled'))

            # Remove duplicates and clean up columns
            combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()].copy()
            combined_data.insert(0, 'Date', combined_data.pop('Date'))
            
            self.logger.info(f'Data Processed for {self.stock}')
            return combined_data

        except KeyError as e:
            self.logger.error(f"KeyError: {e}. Please ensure all required columns are present in the data.")
            return pd.DataFrame()
