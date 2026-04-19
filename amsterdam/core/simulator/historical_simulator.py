"""
Historical Data Simulator - Replays real market data from JSON files.

Unlike GBMSimulator which generates synthetic data using Geometric Brownian Motion,
this simulator replays actual historical OHLCV bars for more realistic backtesting
and ML training data generation.

Usage:
    from core.simulator.historical_simulator import HistoricalDataSimulator

    sim = HistoricalDataSimulator(
        symbols=["AAPL", "MSFT"],
        data_path="data/data_storage/proc_data"
    )

    for _ in range(1000):
        bars = sim.update_all()
        # Process bars...
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.contracts.events import EVENT_NEW_BAR, EVENT_PRICE_UPDATE
from core.events.eventhandler import get_event_handler
from loggers.logger import Logger


class HistoricalDataSimulator:
    """
    Replays historical OHLCV bars from JSON data files.

    Compatible with GBMSimulator interface for drop-in replacement.

    Attributes:
        symbols: List of symbols to simulate
        data_path: Path to directory containing proc_*.json files
        loop_data: If True, restart from beginning when data exhausted
    """

    def __init__(
        self,
        symbols: list[str],
        data_path: str = "data/data_storage/proc_data",
        loop_data: bool = True,
        start_index: int = 0,
        log_prices: bool = False,
    ):
        """
        Initialize the historical data simulator.

        Args:
            symbols: List of ticker symbols to load
            data_path: Path to directory with proc_SYMBOL_file.json files
            loop_data: Whether to loop when data is exhausted
            start_index: Starting bar index (for skipping warmup data)
            log_prices: Whether to log each price update
        """
        self.symbols = symbols
        self.data_path = Path(data_path)
        self.loop_data = loop_data
        self.log_prices = log_prices

        # Logger
        self.logger = Logger(
            log_file="historical_simulator.log", logger_name="HistoricalSimulator", propagate=True
        ).get_logger()

        # Event bus for price/bar events
        self.bus = get_event_handler()

        # Load data for each symbol
        self.data: dict[str, list[dict[str, Any]]] = {}
        self.indices: dict[str, int] = {}
        self.price_state: dict[str, float] = {}

        for symbol in symbols:
            bars = self._load_symbol_data(symbol)
            if bars:
                self.data[symbol] = bars
                self.indices[symbol] = min(start_index, len(bars) - 1)
                # Initialize price state from first bar
                self.price_state[symbol] = bars[self.indices[symbol]]["Close"]
                self.logger.info(f"Loaded {len(bars)} bars for {symbol}, starting at index {self.indices[symbol]}")
            else:
                self.logger.warning(f"No data found for {symbol}")

        self.logger.info(f"HistoricalDataSimulator initialized: {len(self.data)} symbols loaded")

    def _load_symbol_data(self, symbol: str) -> list[dict[str, Any]]:
        """
        Load historical bars for a symbol from JSON file.

        Looks for files in this order:
        1. proc_{symbol}_file.json in data_path
        2. raw_{symbol}_file.json in data_path
        3. raw_{symbol}_file.json in sibling raw_data folder

        Args:
            symbol: Ticker symbol (e.g., "AAPL")

        Returns:
            List of bar dictionaries, or empty list if not found
        """
        # Try processed data first
        proc_file = self.data_path / f"proc_{symbol}_file.json"
        if proc_file.exists():
            return self._parse_json_file(proc_file, symbol)

        # Try raw data in same directory
        raw_file = self.data_path / f"raw_{symbol}_file.json"
        if raw_file.exists():
            return self._parse_json_file(raw_file, symbol)

        # Fall back to raw data in sibling directory
        raw_path = self.data_path.parent / "raw_data" / f"raw_{symbol}_file.json"
        if raw_path.exists():
            return self._parse_json_file(raw_path, symbol)

        self.logger.warning(f"No data file found for {symbol}")
        return []

    def _parse_json_file(self, filepath: Path, symbol: str) -> list[dict[str, Any]]:
        """Parse a JSON data file and return list of bars."""
        try:
            with open(filepath) as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, dict):
                if "bars" in data:
                    bars = data["bars"]
                elif "candles" in data:
                    # Schwab/TDA format with millisecond timestamps
                    bars = data["candles"]
                else:
                    bars = []
            elif isinstance(data, list):
                bars = data
            else:
                self.logger.error(f"Unknown JSON structure in {filepath}")
                return []

            # Normalize bar format
            normalized = []
            for bar in bars:
                # Handle timestamp - could be ISO string, epoch ms, or Date field
                ts = bar.get("Date") or bar.get("timestamp") or bar.get("datetime")
                if isinstance(ts, (int, float)):
                    # Epoch milliseconds
                    ts = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()

                normalized.append(
                    {
                        "Date": ts,
                        "Open": float(bar.get("Open") or bar.get("open", 0)),
                        "High": float(bar.get("High") or bar.get("high", 0)),
                        "Low": float(bar.get("Low") or bar.get("low", 0)),
                        "Close": float(bar.get("Close") or bar.get("close", 0)),
                        "Volume": int(bar.get("Volume") or bar.get("volume", 0)),
                        "symbol": symbol,
                    }
                )

            return normalized

        except Exception as e:
            self.logger.error(f"Failed to load {filepath}: {e}")
            return []

    def generate_bar(self, symbol: str) -> dict[str, Any]:
        """
        Get the next historical bar for a symbol.

        Compatible with GBMSimulator.generate_bar() interface.

        Args:
            symbol: Ticker symbol

        Returns:
            Bar dictionary with timestamp, symbol, open, high, low, close, volume
        """
        if symbol not in self.data:
            raise ValueError(f"No data loaded for {symbol}")

        bars = self.data[symbol]
        idx = self.indices[symbol]

        # Check if we've exhausted data
        if idx >= len(bars):
            if self.loop_data:
                idx = 0
                self.indices[symbol] = 0
                self.logger.info(f"Looping {symbol} data from beginning")
            else:
                # Return last bar
                idx = len(bars) - 1

        bar_data = bars[idx]

        # Advance index for next call
        self.indices[symbol] = idx + 1

        # Update price state
        self.price_state[symbol] = bar_data["Close"]

        # Parse timestamp
        ts = bar_data["Date"]
        if isinstance(ts, str):
            try:
                timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                timestamp = datetime.now(timezone.utc)
        else:
            timestamp = datetime.now(timezone.utc)

        # Build bar in GBMSimulator format
        bar = {
            "timestamp": timestamp,
            "symbol": symbol,
            "open": round(bar_data["Open"], 2),
            "high": round(bar_data["High"], 2),
            "low": round(bar_data["Low"], 2),
            "close": round(bar_data["Close"], 2),
            "volume": bar_data["Volume"],
        }

        # Emit events (async, non-blocking)
        price_payload = {
            "symbol": symbol,
            "price": bar["close"],
            "timestamp": timestamp.isoformat(),
        }
        bar_payload = {
            "symbol": symbol,
            "timestamp": timestamp.isoformat(),
            "open": bar["open"],
            "high": bar["high"],
            "low": bar["low"],
            "close": bar["close"],
            "volume": bar["volume"],
        }

        asyncio.create_task(self.bus.emit(EVENT_PRICE_UPDATE, price_payload))
        asyncio.create_task(self.bus.emit(EVENT_NEW_BAR, bar_payload))

        if self.log_prices:
            self.logger.info(
                f"[{symbol}] Bar {idx} - O:{bar['open']} | H:{bar['high']} | "
                f"L:{bar['low']} | C:{bar['close']} | Vol:{bar['volume']}"
            )

        return bar

    def update_all(self) -> dict[str, dict[str, Any]]:
        """
        Generate new bars for all tracked symbols.

        Compatible with GBMSimulator.update_all() interface.

        Returns:
            Dictionary mapping symbol to bar data
        """
        return {sym: self.generate_bar(sym) for sym in self.symbols if sym in self.data}

    def get_remaining_bars(self, symbol: str) -> int:
        """Get number of bars remaining for a symbol."""
        if symbol not in self.data:
            return 0
        return len(self.data[symbol]) - self.indices[symbol]

    def get_total_bars(self, symbol: str) -> int:
        """Get total number of bars loaded for a symbol."""
        return len(self.data.get(symbol, []))

    def reset(self, start_index: int = 0) -> None:
        """Reset all symbols to start index."""
        for symbol in self.data:
            self.indices[symbol] = min(start_index, len(self.data[symbol]) - 1)
            self.price_state[symbol] = self.data[symbol][self.indices[symbol]]["Close"]
        self.logger.info(f"Reset all symbols to index {start_index}")
