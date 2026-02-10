"""
Data Processor Module

Provides data cleaning, feature engineering, and optional research features.

Note: sklearn and scipy are imported lazily inside methods that use them
(denoise_fft, pca_feature_selection, scale_features, feature_engineer with
include_pca=True or scaling_method). This keeps production trading code
from requiring these heavy ML dependencies.
"""
import numpy as np
import pandas as pd
import functools
from typing import Optional, Dict, Any, Tuple

from loggers.logger import Logger
from utils.configloader import ConfigLoader
from indicators.technical_indicators import TechnicalIndicators


def _parse_date_column(series: pd.Series) -> pd.Series:
    """
    Parse a date column handling both epoch milliseconds and string formats.

    Args:
        series: A pandas Series containing dates (epoch ms or strings)

    Returns:
        Series with datetime64 dtype
    """
    if series.dtype in ('int64', 'float64'):
        # Epoch milliseconds (from Alpaca/Schwab)
        return pd.to_datetime(series, unit='ms')
    else:
        # String or already datetime
        return pd.to_datetime(series)

# Optimization: Cache FFT frequency bins by length
_fft_freq_cache: Dict[int, np.ndarray] = {}


def _get_fft_freqs(n: int) -> np.ndarray:
    """Get cached FFT frequency bins for length n."""
    if n not in _fft_freq_cache:
        _fft_freq_cache[n] = np.fft.fftfreq(n)
    return _fft_freq_cache[n]


class Processor:
    """
    Optimized data processor with caching and reduced DataFrame copies.

    Optimizations:
    - Cached cleaned data (invalidated on frame update)
    - Cached feature engineering results
    - FFT frequency bin caching
    - Reduced DataFrame copies in pipeline
    - In-place operations where safe
    """

    def __init__(self, stock: Optional[str] = None, frame: Optional[pd.DataFrame] = None):
        self.stock = stock
        self.frame = frame
        self.config = ConfigLoader().load_config()
        self.logs_dir = self.config["folders"]["logs"]
        self.logger = Logger('app.log', 'Processor', log_dir=self.logs_dir).get_logger()
        self.scaler = None

        # Optimization: Cache for expensive computations
        self._clean_cache: Optional[pd.DataFrame] = None
        self._clean_cache_hash: Optional[int] = None
        self._features_cache: Optional[pd.DataFrame] = None
        self._features_cache_hash: Optional[int] = None

    def update(self, stock: str, frame: pd.DataFrame) -> None:
        """Update the stock symbol and frame data. Invalidates caches."""
        self.stock = stock
        self.frame = frame
        # Invalidate caches on data update
        self._clean_cache = None
        self._clean_cache_hash = None
        self._features_cache = None
        self._features_cache_hash = None

    def dataframe(self, copy: bool = True) -> pd.DataFrame:
        """
        Return the frame. Set copy=False for read-only operations.

        Args:
            copy: If True (default), return a copy. If False, return view.
        """
        if copy:
            return self.frame.copy()
        return self.frame

    def _frame_hash(self) -> int:
        """Compute a hash of the frame for cache invalidation."""
        if self.frame is None or len(self.frame) == 0:
            return 0
        # Use shape and numeric columns only for hash
        try:
            numeric_cols = self.frame.select_dtypes(include=[np.number])
            if len(numeric_cols.columns) > 0 and len(numeric_cols) > 0:
                first_val = numeric_cols.iloc[0].sum()
            else:
                first_val = 0
            return hash((len(self.frame), len(self.frame.columns), first_val))
        except Exception:
            return hash(len(self.frame))

    def clean_stock_data(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Clean and normalize stock data with optional caching.

        Args:
            use_cache: If True, return cached result if available.
        """
        # Check cache
        current_hash = self._frame_hash()
        if use_cache and self._clean_cache is not None and self._clean_cache_hash == current_hash:
            return self._clean_cache.copy()

        try:
            # Get a copy for modification
            df = self.dataframe(copy=True)

            # Chain operations for efficiency
            df = (df
                  .dropna()
                  .sort_values(by='datetime')
                  .drop_duplicates()
                  .rename(columns={
                      'open': 'Open',
                      'high': 'High',
                      'low': 'Low',
                      'close': 'Close',
                      'volume': 'Volume',
                      'datetime': 'Date'
                  }))

            # Cache result
            self._clean_cache = df
            self._clean_cache_hash = current_hash
            return df.copy()

        except KeyError as e:
            self.logger.error(f"KeyError: Missing key {e} in the DataFrame.")
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
            df["Date"] = _parse_date_column(df["Date"])
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

        # tidy up - replace inf values with NaN
        out = out.replace([np.inf, -np.inf], np.nan)

        # Only require short-term features to be present (not 252-day lookbacks)
        # This allows processing with less historical data
        essential_features = ["ret_1d", "vol_21d", "ATR14", "rsi14", "sma_20"]
        essential_existing = [f for f in essential_features if f in out.columns]

        if essential_existing:
            # Drop rows only where essential features are NaN
            out = out.dropna(subset=essential_existing)
        else:
            # Fallback: drop rows where ALL values are NaN
            out = out.dropna(how="all")

        return out

    def normalizing_scaling(self, df: pd.DataFrame, method='standard') -> pd.DataFrame:
        # Lazy import - only load sklearn when actually needed
        if method == 'standard':
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Unsupported scaling method")

        scaled_features = self.scaler.fit_transform(df.drop(columns=['Date', 'Close']))
        scaled_df = pd.DataFrame(scaled_features, columns=[f"{col}" for col in df.columns[2:]], index=df.index)
        scaled_df = pd.concat([df[['Date', 'Close']], scaled_df], axis=1)
        return scaled_df

    def denoise_fft(
        self,
        signal: np.ndarray,
        threshold: float = 0.1,
        low_freq_cutoff: float = 0.1,
        high_freq_cutoff: float = 0.9
    ) -> np.ndarray:
        """
        Apply FFT-based denoising with frequency band filtering.

        Optimizations:
        - Cached frequency bins by signal length
        - Early return for invalid signals
        - In-place array modification
        """
        # Lazy import - only load scipy when actually needed
        from scipy.fft import fft, ifft

        # Validate input
        if signal is None or len(signal) == 0:
            return signal
        if not np.all(np.isfinite(signal)):
            return signal  # Return unchanged if contains NaN/Inf

        n = len(signal)

        # Optimization: Use cached frequency bins
        freqs = _get_fft_freqs(n)

        # FFT and filter
        fft_signal = fft(signal)

        # Create mask once (avoid repeated boolean operations)
        freq_mask = (freqs < low_freq_cutoff) | (freqs > high_freq_cutoff)
        fft_signal[freq_mask] = 0

        # Magnitude threshold
        mag_mask = np.abs(fft_signal) < threshold
        fft_signal[mag_mask] = 0

        return np.abs(ifft(fft_signal))

    def apply_signal_processing(self, df: pd.DataFrame, columns_to_denoise: list) -> pd.DataFrame:
        """Apply denoising to specific columns of the DataFrame."""
        df_copy = df.copy()
        for col in columns_to_denoise:
            df_copy[col] = self.denoise_fft(df_copy[col].values)  # Using FFT-based denoising
        return df_copy

    def pca_feature_selection(self, df, n_components):
        # Lazy import - only load sklearn when actually needed
        from sklearn.decomposition import PCA

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
            base["Date"] = _parse_date_column(base["Date"])
            base = base.set_index("Date", drop=True)

        if denoise_cols:
            # ensure cols exist before denoising
            cols = [c for c in denoise_cols if c in base.columns]
            if cols:
                base.loc[:, cols] = self.apply_signal_processing(base.loc[:, cols].copy(), cols)

        inds = self.apply_indicators(sma_window=sma_window, ema_window=ema_window).copy()
        if "Date" in inds.columns:
            inds["Date"] = _parse_date_column(inds["Date"])
            inds = inds.set_index("Date", drop=True)
        # drop raw OHLCV from indicators frame to avoid duplicates
        inds = inds.drop(columns=[c for c in ("Open","High","Low","Close","Volume","Date") if c in inds.columns], errors="ignore")

        eng = self.feature_engineering().copy()
        if "Date" in eng.columns:
            eng["Date"] = _parse_date_column(eng["Date"])
            eng = eng.set_index("Date", drop=True)

        # --- 2) align & combine on index (inner join keeps common timestamps) ---
        combined = base.join([inds, eng], how="inner")
        # keep a tidy column order: raw base then features
        combined = combined.loc[:, ~combined.columns.duplicated()].copy()

        # --- 2b) Handle empty result from join (insufficient historical data) ---
        if len(combined) == 0:
            # Fall back to left join with base data if inner join produced nothing
            # This happens when feature_engineering dropna() removes all rows
            self.logger.warning(
                f"Inner join produced 0 rows for {self.stock}. "
                f"Falling back to base data with available indicators."
            )
            # Try joining with just indicators (less strict requirements)
            combined = base.join(inds, how="left")
            combined = combined.loc[:, ~combined.columns.duplicated()].copy()
            # Drop rows with NaN in essential columns only
            essential_cols = ["Open", "High", "Low", "Close", "Volume"]
            essential_existing = [c for c in essential_cols if c in combined.columns]
            if essential_existing:
                combined = combined.dropna(subset=essential_existing)

            if len(combined) == 0:
                self.logger.error(f"No valid data for {self.stock} after fallback")
                return (pd.DataFrame(), {}) if return_artifacts else pd.DataFrame()

        # --- 3) build numeric matrix for transforms ---
        num = combined.select_dtypes(include=[np.number]).copy()

        # --- 3b) Validate numeric data before scaling ---
        if len(num) == 0 or num.shape[1] == 0:
            self.logger.warning(
                f"No numeric columns for scaling in {self.stock}. Returning combined data without scaling."
            )
            out = combined.reset_index().rename(columns={"index": "Date"})
            return (out, {"feature_cols": []}) if return_artifacts else out

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
