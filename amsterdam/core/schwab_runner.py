# core/schwab_runner.py
"""
SchwabLiveRunner - Live trading runner for Schwab broker integration.

Extends BaseLiveRunner with Schwab-specific:
- SchwabBroker creation
- Quote-to-bar aggregation (Schwab streams quotes, not bars)
- Connection and streaming setup with reconnection logic
- Pre-flight token validation
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from core.base.base_live_runner import BaseLiveRunner
from core.broker.schwab_broker import SchwabBroker
from core.config_loader import TradingConfig, get_config
from core.contracts.events import EVENT_HEALTH_UPDATE
from core.credential_validator import CredentialStatus, CredentialValidator
from data.streaming.schwab_client import SchwabClient

ROOT = Path(__file__).resolve().parents[1]  # .../schwab_trader
load_dotenv(ROOT / ".venv" / ".env")


class SchwabLiveRunner(BaseLiveRunner):
    """
    Live trading runner for Schwab broker.

    Implements Schwab-specific quote handling, bar aggregation,
    and reconnection logic. Inherits common functionality from BaseLiveRunner.
    """

    BROKER_NAME = "Schwab"
    LOG_FILE_KEY = "SchwabLive"
    TRADE_LOG_FILE = "schwab_live_trades.csv"

    def __init__(self, symbols: list[str], client: SchwabClient | None = None, config: TradingConfig | None = None):
        """
        Initialize the Schwab live runner.

        Args:
            symbols: List of symbols to trade
            client: Optional pre-configured SchwabClient instance
            config: Optional TradingConfig instance (uses global config if not provided)
        """
        # Store client before calling super().__init__ which creates broker
        self._init_client = client
        self._init_config = config or get_config()

        # Reconnection state
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 5  # seconds

        # Quote callback registry
        self._quote_callbacks: dict[str, Callable] = {}

        # Bar aggregation state (Schwab streams quotes, we aggregate to bars)
        # Structure: {symbol: {"open": float, "high": float, "low": float, "close": float, "volume": int, "bar_id": int}}
        self._bar_aggregation: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"open": None, "high": None, "low": None, "close": None, "volume": 0, "bar_id": None}
        )

        # Call parent constructor
        super().__init__(symbols, config)

    # ==========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ==========================================================================

    def _create_broker(self) -> SchwabBroker:
        """Create and configure the Schwab broker instance."""
        client = self._init_client
        if client is None:
            api_key = os.getenv("SCHWAB_API_KEY")
            secret_key = os.getenv("SCHWAB_SECRET")
            if not api_key or not secret_key:
                raise ValueError("SCHWAB_API_KEY and SCHWAB_SECRET must be set in environment")
            client = SchwabClient(apikey=api_key, secretkey=secret_key)

        self._client = client
        config = getattr(self, "_init_config", None) or self.config
        return SchwabBroker(
            client=client,
            session=config.schwab.session,
        )

    def _canonicalize_bar(self, raw_data: Any) -> dict:
        """
        Convert Schwab quote/bar data to canonical format.

        This method is used by _canonicalize_schwab_quote internally.
        """
        # This is called from quote callback which passes dict
        return raw_data if isinstance(raw_data, dict) else {}

    async def _connect_broker(self) -> None:
        """Establish connection to Schwab streaming."""
        api_key = os.getenv("SCHWAB_API_KEY")
        secret_key = os.getenv("SCHWAB_SECRET")

        if not api_key or not secret_key:
            raise ValueError("Missing Schwab credentials")

        # Initialize streaming connection
        self.broker.connect_stream(api_key, secret_key)
        self._reconnect_attempts = 0
        await self._emit_health_status("connected", {"symbols": self.symbols})
        self.logger.info(f"Connected to Schwab streaming for: {', '.join(self.symbols)}")

    async def _start_streaming(self) -> asyncio.Task:
        """Start the Schwab data stream."""
        return asyncio.create_task(self.broker.start_stream())

    async def _disconnect_broker(self) -> None:
        """Disconnect from Schwab."""
        await self.broker.disconnect()

    def _subscribe_to_data(self) -> None:
        """Subscribe to quote data for all symbols."""
        for sym in self.symbols:
            callback = self._create_quote_callback(sym)
            self._quote_callbacks[sym] = callback
            self.broker.subscribe_quotes(callback, sym)

    # ==========================================================================
    # HOOK METHOD OVERRIDES
    # ==========================================================================

    async def _preflight_checks(self) -> None:
        """Check Schwab token status before starting."""
        validator = CredentialValidator()
        result = await validator.validate_schwab()

        if result.status == CredentialStatus.EXPIRED:
            self.logger.warning(
                "\n" + "=" * 60 + "\n"
                "SCHWAB TOKEN EXPIRED - Renewal required!\n"
                "Run: python -m data.streaming.authenticator\n"
                "=" * 60
            )
        elif result.status == CredentialStatus.EXPIRING_SOON:
            hours = result.expires_in // 3600 if result.expires_in else 0
            self.logger.warning(
                f"SCHWAB TOKEN EXPIRING in {hours} hours. Renew soon: python -m data.streaming.authenticator"
            )
        elif result.status == CredentialStatus.MISSING:
            self.logger.warning("Schwab credentials not configured. Run: python -m data.streaming.authenticator")
        elif result.status == CredentialStatus.VALID:
            days = result.expires_in // 86400 if result.expires_in else 0
            self.logger.info(f"Schwab credentials valid ({days} days until refresh expires)")

    async def _main_loop(self, stream_task: asyncio.Task) -> None:
        """Main run loop with reconnection handling."""
        try:
            while self._running:
                # Check if stream is still running
                if stream_task.done():
                    exc = stream_task.exception()
                    if exc:
                        self.logger.error(f"Stream task failed: {exc}")

                    # Attempt reconnection
                    if self._running and await self._reconnect():
                        stream_task = await self._start_streaming()
                    else:
                        break

                # Check if reconciler halted trading
                if self.reconciler.is_halted:
                    self.logger.critical("Trading halted by reconciler - exiting run loop")
                    break

                await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            self.logger.info("Runner cancelled")

    async def _cleanup(self, stream_task: asyncio.Task) -> None:
        """Cleanup with health status emission."""
        await super()._cleanup(stream_task)
        await self._emit_health_status("disconnected", {"reason": "shutdown"})

    # ==========================================================================
    # SCHWAB-SPECIFIC METHODS
    # ==========================================================================

    @staticmethod
    def _canonicalize_schwab_chart_bar(data: dict, symbol: str) -> dict:
        """
        Normalize Schwab CHART_EQUITY candle data to canonical bar format.

        Schwab CHART_EQUITY streaming field numbers (from API docs):
        - 1: Open Price
        - 2: High Price
        - 3: Low Price
        - 4: Close Price
        - 5: Volume
        - 6: Sequence (candle minute identifier)
        - 7: Chart Time (milliseconds since Epoch)

        Use this for actual candle bars from CHART_EQUITY service.
        """
        chart_time_ms = data.get("7") or data.get("chart_time")
        if chart_time_ms:
            ts = datetime.fromtimestamp(int(chart_time_ms) / 1000, tz=timezone.utc)
        else:
            ts = datetime.now(timezone.utc)

        return {
            "symbol": symbol,
            "timestamp": ts,
            "Open": float(data.get("1") or data.get("open", 0)),
            "High": float(data.get("2") or data.get("high", 0)),
            "Low": float(data.get("3") or data.get("low", 0)),
            "Close": float(data.get("4") or data.get("close", 0)),
            "Volume": int(data.get("5") or data.get("volume", 0)),
            "sequence": int(data.get("6") or data.get("sequence", 0)),
        }

    @staticmethod
    def _canonicalize_schwab_quote(quote: dict, symbol: str) -> dict:
        """
        Normalize Schwab quote format to canonical bar format.

        Schwab LEVELONE_EQUITIES streaming field numbers:
        - 1: Bid Price
        - 2: Ask Price
        - 3: Last Price
        - 8: Total Volume
        - 10: High Price (day's high)
        - 11: Low Price (day's low)
        - 12: Close Price (previous day's close)
        - 17: Open Price (day's open)
        - 35: Trade Time in Long (milliseconds since Epoch)
        """
        trade_time_ms = quote.get("35") or quote.get("trade_time")
        if trade_time_ms:
            ts = datetime.fromtimestamp(int(trade_time_ms) / 1000, tz=timezone.utc)
        else:
            ts = datetime.now(timezone.utc)

        bid_price = quote.get("1") or quote.get("bid_price") or quote.get("bidPrice", 0)
        ask_price = quote.get("2") or quote.get("ask_price") or quote.get("askPrice", 0)
        last_price = quote.get("3") or quote.get("last_price") or quote.get("lastPrice", 0)
        volume = quote.get("8") or quote.get("volume") or quote.get("totalVolume", 0)
        high_price = quote.get("10") or quote.get("high_price") or quote.get("highPrice", 0)
        low_price = quote.get("11") or quote.get("low_price") or quote.get("lowPrice", 0)
        prev_close = quote.get("12") or quote.get("close_price") or quote.get("closePrice", 0)
        open_price = quote.get("17") or quote.get("open_price") or quote.get("openPrice", 0)

        if last_price:
            price = float(last_price)
        elif bid_price and ask_price:
            price = (float(bid_price) + float(ask_price)) / 2
        else:
            price = float(bid_price or ask_price or 0)

        return {
            "symbol": symbol,
            "timestamp": ts,
            "Open": float(open_price) if open_price else price,
            "High": float(high_price) if high_price else price,
            "Low": float(low_price) if low_price else price,
            "Close": price,
            "Volume": int(volume) if volume else 0,
            "prev_close": float(prev_close) if prev_close else None,
        }

    def _aggregate_quote_to_bar(self, symbol: str, price: float, volume: int, bar_id: int) -> dict:
        """
        Aggregate quote data into OHLCV bar.

        Collects multiple quotes within the same time bucket into a proper bar.

        Args:
            symbol: Trading symbol
            price: Current quote price
            volume: Volume from this quote
            bar_id: Current bar bucket ID

        Returns:
            Dict with aggregated OHLCV data and bar_closed flag
        """
        agg = self._bar_aggregation[symbol]

        # Check if we're starting a new bar
        if agg["bar_id"] is None or agg["bar_id"] != bar_id:
            prev_bar = None
            if agg["bar_id"] is not None and agg["close"] is not None:
                prev_bar = {
                    "Open": agg["open"],
                    "High": agg["high"],
                    "Low": agg["low"],
                    "Close": agg["close"],
                    "Volume": agg["volume"],
                    "bar_closed": True,
                }

            # Start new bar
            agg["open"] = price
            agg["high"] = price
            agg["low"] = price
            agg["close"] = price
            agg["volume"] = volume
            agg["bar_id"] = bar_id

            if prev_bar:
                return prev_bar
            else:
                return {
                    "Open": price,
                    "High": price,
                    "Low": price,
                    "Close": price,
                    "Volume": volume,
                    "bar_closed": False,
                }

        # Update existing bar
        agg["high"] = max(agg["high"], price)
        agg["low"] = min(agg["low"], price)
        agg["close"] = price
        agg["volume"] += volume

        return {
            "Open": agg["open"],
            "High": agg["high"],
            "Low": agg["low"],
            "Close": agg["close"],
            "Volume": agg["volume"],
            "bar_closed": False,
        }

    def _create_quote_callback(self, symbol: str) -> Callable:
        """Create a quote callback bound to a specific symbol."""

        async def callback(quote: dict):
            await self._on_quote(symbol, quote)

        return callback

    async def _on_quote(self, symbol: str, quote: dict) -> None:
        """
        Handle incoming Schwab quote data.

        Aggregates quotes into OHLCV bars and processes through the strategy engine.

        Args:
            symbol: The symbol for this quote
            quote: Raw quote data from Schwab streaming
        """
        raw_bar = self._canonicalize_schwab_quote(quote, symbol)
        price = float(raw_bar["Close"])
        volume = int(raw_bar.get("Volume", 0))
        ts: datetime = raw_bar["timestamp"]

        self.logger.debug(f"[RAW QUOTE] {symbol} price={price}")

        bar_id = self._bar_bucket(ts)

        # Aggregate quotes into proper OHLCV bars
        aggregated = self._aggregate_quote_to_bar(symbol, price, volume, bar_id)
        bar_closed = aggregated.get("bar_closed", False)

        # Build bar with aggregated OHLCV data
        bar = {
            "symbol": symbol,
            "timestamp": ts,
            "Open": aggregated["Open"],
            "High": aggregated["High"],
            "Low": aggregated["Low"],
            "Close": aggregated["Close"],
            "Volume": aggregated["Volume"],
        }

        # Only add completed bars to history (override base behavior slightly)
        # We still call _process_bar but it will append to history
        # For Schwab, we want to control when bars are added
        if bar_closed:
            await self._process_bar(bar)
        else:
            # Still emit events and update prices for incomplete bars
            await self._process_partial_bar(bar)

    async def _process_partial_bar(self, bar: dict) -> None:
        """
        Process partial bar (still being aggregated).

        Updates prices and state without adding to history.
        """
        symbol = bar["symbol"]
        ts = bar["timestamp"]
        last_px = float(bar["Close"])

        # Update price for MTM
        self.portfolio.update_price(symbol, last_px)

        # Emit bar event for GUI
        await self.event_handler.emit(
            "BAR",
            {
                "timestamp": ts,
                "symbol": symbol,
                "open": float(bar["Open"]),
                "high": float(bar["High"]),
                "low": float(bar["Low"]),
                "close": float(bar["Close"]),
                "volume": int(bar.get("Volume", 0)),
            },
        )

    async def _emit_health_status(self, status: str, details: dict = None) -> None:
        """Emit health status event."""
        await self.event_handler.emit(
            EVENT_HEALTH_UPDATE,
            {
                "broker": "schwab",
                "status": status,
                "details": details or {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def _reconnect(self) -> bool:
        """Attempt to reconnect to Schwab streaming."""
        while self._running and self._reconnect_attempts < self._max_reconnect_attempts:
            self._reconnect_attempts += 1
            self.logger.warning(f"Reconnection attempt {self._reconnect_attempts}/{self._max_reconnect_attempts}")
            await self._emit_health_status(
                "reconnecting", {"attempt": self._reconnect_attempts, "max_attempts": self._max_reconnect_attempts}
            )

            await asyncio.sleep(self._reconnect_delay * self._reconnect_attempts)

            try:
                await self._connect_broker()
                self._subscribe_to_data()
                return True
            except Exception as e:
                self.logger.error(f"Reconnection failed: {e}")

        self.logger.error("Max reconnection attempts reached")
        await self._emit_health_status("disconnected", {"reason": "max_reconnect_attempts"})
        return False

    async def stop(self) -> None:
        """Stop the live trading runner (async version)."""
        self._running = False
        if hasattr(self.broker, "disconnect"):
            await self.broker.disconnect()
        self.logger.info("Stop requested")


# -------- Entrypoint --------
def _ensure_live_config(dir_path: str = "config"):
    """Ensure config files exist with defaults."""
    os.makedirs(dir_path, exist_ok=True)

    sr_path = os.path.join(dir_path, "strategy_routing.json")
    if not os.path.exists(sr_path):
        with open(sr_path, "w") as f:
            json.dump(
                {
                    "AAPL": {
                        "low_volatility": "sma_strategy",
                        "normal": "momentum_strategy",
                        "high_volatility": "mean_reversion_strategy",
                    },
                    "MSFT": {
                        "low_volatility": "sma_strategy",
                        "normal": "momentum_strategy",
                        "high_volatility": "mean_reversion_strategy",
                    },
                },
                f,
                indent=2,
            )

    sp_path = os.path.join(dir_path, "strategy_params.json")
    if not os.path.exists(sp_path):
        with open(sp_path, "w") as f:
            json.dump(
                {
                    "AAPL": {
                        "low_volatility": {"params": {"fast": 10, "slow": 30}},
                        "normal": {"params": {"lookback": 20}},
                        "high_volatility": {"params": {"window": 14}},
                    },
                    "MSFT": {
                        "low_volatility": {"params": {"fast": 10, "slow": 30}},
                        "normal": {"params": {"lookback": 20}},
                        "high_volatility": {"params": {"window": 14}},
                    },
                },
                f,
                indent=2,
            )

    tl_path = os.path.join(dir_path, "trade_logic_routing.json")
    if not os.path.exists(tl_path):
        with open(tl_path, "w") as f:
            json.dump(
                {
                    "AAPL": {
                        "low_volatility": {"trade_logic_class": "default", "params": {}},
                        "normal": {"trade_logic_class": "default", "params": {}},
                        "high_volatility": {"trade_logic_class": "default", "params": {}},
                    },
                    "MSFT": {
                        "low_volatility": {"trade_logic_class": "default", "params": {}},
                        "normal": {"trade_logic_class": "default", "params": {}},
                        "high_volatility": {"trade_logic_class": "default", "params": {}},
                    },
                },
                f,
                indent=2,
            )

    return sr_path, sp_path, tl_path


async def main():
    """Main entry point for running SchwabLiveRunner standalone."""
    _ensure_live_config("config")

    config = get_config()
    symbols = config.general.default_symbols or ["AAPL", "MSFT"]

    runner = SchwabLiveRunner(symbols=symbols, config=config)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
