"""
Technical data provider — wraps existing HistoricalBarLoader and TechnicalIndicators.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from core.historical_loader import HistoricalBarLoader
from indicators.technical_indicators import TechnicalIndicators
from scanner.providers.base import BaseDataProvider, ProviderData

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = str(ROOT / "data" / "data_storage" / "proc_data")


class TechnicalProvider(BaseDataProvider):
    """Fetches OHLCV data and applies all technical indicators."""

    def __init__(self, data_path: str = DEFAULT_DATA_PATH, lookback_bars: int = 200):
        self._data_path = data_path
        self._lookback_bars = lookback_bars
        self._loader = HistoricalBarLoader(data_path)

    @property
    def name(self) -> str:
        return "technical"

    def fetch(self, symbol: str) -> ProviderData:
        bars = self._loader.load_last_n_bars(symbol, n=self._lookback_bars)
        if not bars:
            logger.warning(f"[technical] No data for {symbol}")
            return ProviderData(symbol=symbol, provider_name=self.name)

        df = pd.DataFrame(bars)
        # Ensure capitalized columns for indicator compatibility
        col_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Apply all technical indicators (expects capitalized OHLCV columns)
        df = TechnicalIndicators(df).apply_all()

        # Normalize to lowercase for consistent access in criteria
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})

        # Build summary dict from latest row
        latest = df.iloc[-1] if len(df) > 0 else {}
        summary = {}
        # Map canonical names to actual indicator column names
        key_aliases = {
            "close": "close",
            "volume": "volume",
            "RSI": "RSI",
            "MACD": "MACD",
            "MACD_signal": "MACD_Signal",
            "SMA": "SMA_20",
            "EMA": "EMA_20",
            "ATR": "ATR",
            "OBV": "OBV",
            "BB_upper": "Bollinger_Upper",
            "BB_lower": "Bollinger_Lower",
            "Momentum": "Momentum",
            "ROC": "ROC",
            "PSAR": "SAR",
        }
        for out_key, col_key in key_aliases.items():
            if col_key in latest.index:
                val = latest[col_key]
                summary[out_key] = float(val) if pd.notna(val) else None

        # Add derived metrics
        if "volume" in df.columns and len(df) >= 20:
            avg_vol = df["volume"].tail(20).mean()
            summary["avg_volume_20"] = float(avg_vol) if pd.notna(avg_vol) else None
        if "close" in df.columns and len(df) >= 50:
            summary["SMA_50"] = float(df["close"].tail(50).mean())

        return ProviderData(
            symbol=symbol,
            provider_name=self.name,
            data=summary,
            dataframe=df,
        )

    def is_available(self) -> bool:
        return Path(self._data_path).exists()
