from sklearn.preprocessing import StandardScaler, MinMaxScaler
from loggers.logger import Logger
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
        """
        Daily features for OHLCV data (no look-ahead).
        Returns a DataFrame indexed by Date with only feature columns.
        """
        import numpy as np
        import pandas as pd

        df = self.clean_stock_data().copy()  # expects ['Date','Open','High','Low','Close','Volume', ...]
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)

        # use adjusted close if present
        px = df.get("Adj Close", df["Close"]).astype(float)
        open_ = df["Open"].astype(float)
        high  = df["High"].astype(float)
        low   = df["Low"].astype(float)
        close = df["Close"].astype(float)
        vol   = df["Volume"].astype(float)

        out = pd.DataFrame(index=df.index)

        # ---- returns & momentum (daily horizon) ----
        ret1 = px.pct_change()
        out["ret_1d"] = ret1
        for w in (5, 10, 21, 63, 126, 252):   # ~1w, 2w, 1m, 1q, 6m, 1y
            out[f"ret_{w}d"] = px.pct_change(w)

        # Academic momentum (12-2): last 12m excluding most recent month
        # R_{t-21} / R_{t-252} - 1
        out["mom_12_2"] = (px.shift(21) / px.shift(252)) - 1

        # ---- volatility ----
        log_ret = np.log(px).diff()
        for w in (21, 63, 252):
            out[f"vol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252)  # annualized

        # True Range & ATR(14)
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs()
        ], axis=1).max(axis=1)
        out["TR"] = tr
        out["ATR14"] = tr.rolling(14, min_periods=14).mean()

        # Parkinson (range-based) volatility (annualized)
        with np.errstate(divide="ignore", invalid="ignore"):
            park = (1.0/(4*np.log(2))) * (np.log(high/low)**2)
        out["parkinson_20d"] = np.sqrt(park.rolling(20).mean()) * np.sqrt(252)

        # ---- trend / moving averages ----
        out["sma_20"]  = px.rolling(20).mean()
        out["sma_50"]  = px.rolling(50).mean()
        out["sma_200"] = px.rolling(200).mean()
        ema12 = px.ewm(span=12, adjust=False).mean()
        ema26 = px.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        out["ema12"] = ema12
        out["ema26"] = ema26
        out["macd"] = macd
        out["macd_signal"] = macd_sig
        out["macd_hist"] = macd - macd_sig
        out["ma_20_gap_pct"] = (px - out["sma_20"]) / px

        # ---- Bollinger & RSI ----
        bb_mu = px.rolling(20).mean()
        bb_sd = px.rolling(20).std()
        out["bb_pos"] = (px - bb_mu) / (2.0*bb_sd + 1e-12)  # ~[-1,1] outside bands
        # RSI(14) (EWMA version)
        delta = px.diff()
        up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        dn = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs = up / (dn + 1e-12)
        out["rsi14"] = 100 - 100/(1 + rs)

        # ---- gaps & intraday structure (useful on daily) ----
        out["overnight_ret"] = (open_ / prev_close) - 1.0
        out["intraday_ret"]  = (close / open_) - 1.0
        out["gap_bps"]       = out["overnight_ret"] * 1e4

        # ---- volume / liquidity ----
        out["dollar_vol"] = close * vol
        out["vol_z_20"]   = (vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + 1e-12)
        out["dvol_1d"]    = vol.pct_change()

        # ---- price location vs 52w extremes ----
        rolling_high = px.rolling(252).max()
        rolling_low  = px.rolling(252).min()
        out["pct_from_252d_high"] = (px / rolling_high) - 1.0
        out["pct_from_252d_low"]  = (px / rolling_low)  - 1.0

        # ---- seasonality (daily) ----
        # cyclical month-of-year & day-of-week
        idx = out.index
        if isinstance(idx, pd.DatetimeIndex):
            month = idx.month
            out["moy_sin"] = np.sin(2*np.pi*month/12.0)
            out["moy_cos"] = np.cos(2*np.pi*month/12.0)
            out["dow"] = idx.dayofweek.astype(float)

        # tidy up
        out = out.replace([np.inf, -np.inf], np.nan).dropna()
        return out

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
        

        
    def ml_process(
        self,
        sma_window: int,
        ema_window: int,
        *,
        scaling_method: str | None = None,   # "standard" | "minmax" | None
        pca_components: int = 5,
        denoise_cols: list[str] | None = None,
        include_scaled: bool = True,
        include_pca: bool = False,
        return_artifacts: bool = False,
    ):
        """
        Clean -> indicators -> engineered -> (optional) denoise -> combine -> scale/PCA (numeric only)

        Returns:
            DataFrame (and optionally artifacts dict with 'scaler','pca','feature_cols')
        """
        # --- 1) load/prepare base frames with a consistent DateTimeIndex ---
        base = self.clean_stock_data().copy()           # must contain 'Date'
        if "Date" in base.columns:
            base["Date"] = pd.to_datetime(base["Date"])
            base = base.set_index("Date", drop=True)

        if denoise_cols:
            # ensure cols exist before denoising
            cols = [c for c in denoise_cols if c in base.columns]
            if cols:
                base.loc[:, cols] = self.apply_signal_processing(base.loc[:, cols].copy(), cols)

        inds = self.apply_indicators(sma_window=sma_window, ema_window=ema_window).copy()
        if "Date" in inds.columns:
            inds["Date"] = pd.to_datetime(inds["Date"])
            inds = inds.set_index("Date", drop=True)
        # drop raw OHLCV from indicators frame to avoid duplicates
        inds = inds.drop(columns=[c for c in ("Open","High","Low","Close","Volume","Date") if c in inds.columns], errors="ignore")

        eng = self.feature_engineering().copy()
        if "Date" in eng.columns:
            eng["Date"] = pd.to_datetime(eng["Date"])
            eng = eng.set_index("Date", drop=True)

        # --- 2) align & combine on index (inner join keeps common timestamps) ---
        combined = base.join([inds, eng], how="inner")
        # keep a tidy column order: raw base then features
        combined = combined.loc[:, ~combined.columns.duplicated()].copy()

        # --- 3) build numeric matrix for transforms ---
        num = combined.select_dtypes(include=[np.number]).copy()
        # bound PCA components by numeric dimensionality and samples
        max_pca = max(1, min(pca_components, num.shape[1], max(1, num.shape[0]-1)))

        artifacts = {"feature_cols": list(num.columns)}

        # --- 4) scaling (optional; numeric only) ---
        scaled_df = None
        if scaling_method:
            if scaling_method.lower() in ("standard", "z", "zscore"):
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
            elif scaling_method.lower() in ("minmax", "min_max"):
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {scaling_method}")

            scaled_vals = scaler.fit_transform(num.values)
            scaled_df = pd.DataFrame(scaled_vals, index=num.index, columns=[f"{c}_scaled" for c in num.columns])
            artifacts["scaler"] = scaler

        # --- 5) PCA (optional; on the SAME numeric matrix used for scaling) ---
        pca_df = None
        if include_pca and max_pca > 0:
            # PCA generally expects zero-centered inputs; if you need PCA on scaled data,
            # swap `num` for `scaled_df` here.
            from sklearn.decomposition import PCA
            pca = PCA(n_components=max_pca, svd_solver="auto", whiten=False)
            X_for_pca = scaled_df.values if (scaled_df is not None) else num.values
            pcs = pca.fit_transform(X_for_pca)
            pca_df = pd.DataFrame(pcs, index=num.index, columns=[f"PC{i+1}" for i in range(pcs.shape[1])])
            artifacts["pca"] = pca

        # --- 6) assemble final frame ---
        out = combined.copy()
        if include_scaled and scaled_df is not None:
            out = out.join(scaled_df, how="left")
        if include_pca and pca_df is not None:
            out = out.join(pca_df, how="left")

        # put Date back as a column if you prefer
        out = out.reset_index().rename(columns={"index":"Date"})

        self.logger.info(f"ML data processed for {self.stock} | rows={len(out)} | cols={len(out.columns)}")
        return (out, artifacts) if return_artifacts else out
