# core/historical_loader.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import json
import math

from loggers.logger import Logger


class HistoricalBarLoader:
    """
    Reads proc_{SYMBOL}_file.json and returns normalized bars with:
    - timestamp: datetime (UTC)
    - symbol: str
    - Open, High, Low, Close: float
    - Volume: int
    Keeps any extra fields from the source row as-is.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.logger = Logger("historical_loader.log", "HistoricalBarLoader").get_logger()

    def load_last_n_bars(self, symbol: str, n: int = 200) -> List[Dict[str, Any]]:
        fp = self.path / f"proc_{symbol}_file.json"
        if not fp.exists():
            self.logger.warning(f"No historical file found for {symbol} at {fp}")
            return []
        try:
            with open(fp, "r") as f:
                raw = json.load(f)  # Python's json accepts NaN by default
        except Exception as e:
            self.logger.exception(f"Failed to read {fp}: {e}")
            return []

        rows = raw.get("bars", raw) if isinstance(raw, dict) else raw
        if not isinstance(rows, list) or not rows:
            self.logger.warning(f"No bars in {fp}")
            return []

        tail = rows[-n:]
        out: List[Dict[str, Any]] = []
        for row in tail:
            nb = self._normalize_row(row, symbol)
            if nb:
                out.append(nb)

        out.sort(key=lambda r: r["timestamp"])
        return out

    def get_latest_close_price(self, symbol: str) -> Optional[float]:
        bars = self.load_last_n_bars(symbol, n=1)
        return float(bars[-1]["Close"]) if bars else None

    # --- helpers ---

    def _normalize_row(self, row: Dict[str, Any], fallback_symbol: str) -> Optional[Dict[str, Any]]:
        # Parse timestamp from epoch ms (your "Date" field)
        ts_raw = row.get("Date") or row.get("timestamp") or row.get("Datetime") or row.get("date")
        if ts_raw is None:
            return None
        try:
            # If it looks like epoch ms, convert
            if isinstance(ts_raw, (int, float)) and ts_raw > 10_000_000_000:  # > ~2001 in seconds
                ts = datetime.fromtimestamp(ts_raw / 1000.0, tz=timezone.utc)
            else:
                # seconds epoch or ISO string
                ts = datetime.fromtimestamp(float(ts_raw), tz=timezone.utc) if isinstance(ts_raw, (int, float)) \
                     else datetime.fromisoformat(str(ts_raw)).astimezone(timezone.utc)
        except Exception:
            return None

        # Capitalized OHLCV to match your strategies
        def _num(x):
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return None
            return float(x)

        o = _num(row.get("Open", row.get("open")))
        h = _num(row.get("High", row.get("high")))
        l = _num(row.get("Low", row.get("low")))
        c = _num(row.get("Close", row.get("close")))
        v = row.get("Volume", row.get("volume", 0))
        try:
            v = int(v) if v is not None else 0
        except Exception:
            v = 0

        if None in (o, h, l, c):
            # Skip incomplete bars
            return None

        symbol = str(row.get("Symbol", row.get("symbol", fallback_symbol)))

        # Keep all extra fields
        extra = {k: v for k, v in row.items()
                 if k not in {"Date", "timestamp", "Datetime", "date", "Open", "High", "Low", "Close", "Volume", "open", "high", "low", "close", "volume"}}

        return {
            "timestamp": ts,
            "symbol": symbol,
            "Open": o,
            "High": h,
            "Low": l,
            "Close": c,
            "Volume": v,
            **extra
        }
