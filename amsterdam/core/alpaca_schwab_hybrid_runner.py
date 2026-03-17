# core/alpaca_schwab_hybrid_runner.py
"""
AlpacaSchwabHybridRunner - Hybrid runner for Alpaca execution + Schwab data

Combines the best of both brokers:
- Alpaca for order execution (paper/live trading)
- Schwab for real-time websocket data feed

This runner:
- Uses AlpacaBroker for all trading operations
- Uses SchwabClient for streaming market data
- Aggregates Schwab quotes into OHLCV bars
- Validates Schwab tokens before starting
"""
from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

from core.base.base_live_runner import BaseLiveRunner
from core.broker.alpaca_broker import AlpacaBroker
from core.broker.schwab_broker import SchwabBroker
from core.config_loader import get_config, TradingConfig
from core.credential_validator import CredentialValidator, CredentialStatus
from core.contracts.events import EVENT_HEALTH_UPDATE
from data.streaming.schwab_client import SchwabClient

ROOT = Path(__file__).resolve().parents[1]  # .../amsterdam
load_dotenv(ROOT / ".venv" / ".env")
load_dotenv()


class AlpacaSchwabHybridRunner(BaseLiveRunner):
    """
    Hybrid runner combining Alpaca execution with Schwab data streaming.

    This allows you to:
    - Trade with Alpaca (paper/live)
    - Get real-time data from Schwab websocket
    - Keep using Alpaca's simple paper trading setup
    - Get Schwab's superior data quality
    """

    BROKER_NAME = "AlpacaSchwab"
    LOG_FILE_KEY = "AlpacaSchwabHybrid"
    TRADE_LOG_FILE = "alpaca_schwab_hybrid_trades.csv"

    def __init__(
        self,
        symbols: List[str],
        config: Optional[TradingConfig] = None
    ):
        """
        Initialize the hybrid runner.

        Args:
            symbols: List of symbols to trade
            config: Optional TradingConfig instance
        """
        # Store config
        self._init_config = config or get_config()

        # Schwab client for data streaming
        self._schwab_client: Optional[SchwabClient] = None

        # Reconnection state
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 5  # seconds

        # Quote callback registry
        self._quote_callbacks: Dict[str, Callable] = {}

        # Bar aggregation state (Schwab streams quotes, we aggregate to bars)
        self._bar_aggregation: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"open": None, "high": None, "low": None, "close": None, "volume": 0, "bar_id": None}
        )

        # Call parent constructor (creates broker via _create_broker)
        super().__init__(symbols, config)

    # ==========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS - BROKER (ALPACA)
    # ==========================================================================

    def _create_broker(self) -> AlpacaBroker:
        """Create Alpaca broker for order execution."""
        config = getattr(self, '_init_config', None) or self.config

        # Create Alpaca broker
        alpaca_broker = AlpacaBroker(
            api_key=os.getenv("ALPACA_API_KEY"),
            api_secret=os.getenv("ALPACA_SECRET_KEY"),
            paper=config.alpaca.paper,
            poll_timeout=getattr(config.alpaca, 'poll_timeout_seconds', 30),
        )

        self.logger.info("Using Alpaca broker for execution")
        return alpaca_broker

    def _canonicalize_bar(self, raw_data: Any) -> Dict:
        """
        Convert Schwab quote data to canonical bar format.

        This is called from our quote aggregation logic.
        """
        return raw_data if isinstance(raw_data, dict) else {}

    # ==========================================================================
    # CONNECTION MANAGEMENT - DUAL BROKER
    # ==========================================================================

    async def _connect_broker(self) -> None:
        """
        Establish connections to both Alpaca (execution) and Schwab (data).
        """
        # Connect to Alpaca for execution
        self.broker.api_key = os.getenv("ALPACA_API_KEY")
        self.broker.api_secret = os.getenv("ALPACA_SECRET_KEY")
        self.broker.connect_sync()
        self.logger.info("Connected to Alpaca for order execution")

        # Connect to Schwab for data streaming
        api_key = os.getenv("SCHWAB_API_KEY")
        secret_key = os.getenv("SCHWAB_SECRET")

        if not api_key or not secret_key:
            raise ValueError("Missing Schwab credentials for data streaming")

        # Create Schwab client for streaming only (not for trading)
        self._schwab_client = SchwabClient(apikey=api_key, secretkey=secret_key)

        # Create a temporary Schwab broker just for streaming connection
        self._schwab_stream_broker = SchwabBroker(
            client=self._schwab_client,
            session=self.config.schwab.session,
        )

        # Initialize Schwab streaming connection
        self._schwab_stream_broker.connect_stream(api_key, secret_key)
        self._reconnect_attempts = 0

        await self._emit_health_status("connected", {
            "execution": "alpaca",
            "data": "schwab",
            "symbols": self.symbols
        })

        self.logger.info(f"Connected to Schwab streaming for data: {', '.join(self.symbols)}")

    async def _start_streaming(self) -> asyncio.Task:
        """Start Schwab data stream."""
        return asyncio.create_task(self._schwab_stream_broker.start_stream())

    async def _disconnect_broker(self) -> None:
        """Disconnect from both Alpaca and Schwab."""
        # Disconnect Schwab streaming
        if hasattr(self, '_schwab_stream_broker'):
            await self._schwab_stream_broker.disconnect()
            self.logger.info("Disconnected from Schwab streaming")

        # Disconnect Alpaca
        self.broker.disconnect()
        self.logger.info("Disconnected from Alpaca")

    def _subscribe_to_data(self) -> None:
        """Subscribe to Schwab quote data for all symbols."""
        for sym in self.symbols:
            callback = self._create_quote_callback(sym)
            self._quote_callbacks[sym] = callback
            self._schwab_stream_broker.subscribe_quotes(callback, sym)

        self.logger.info(f"Subscribed to Schwab quotes for {len(self.symbols)} symbols")

    # ==========================================================================
    # PREFLIGHT CHECKS
    # ==========================================================================

    async def _preflight_checks(self) -> None:
        """
        Validate both Alpaca and Schwab credentials before starting.
        """
        validator = CredentialValidator()

        # Check Schwab token (for data streaming)
        schwab_result = await validator.validate_schwab()

        if schwab_result.status == CredentialStatus.EXPIRED:
            self.logger.warning(
                "\n" + "=" * 60 + "\n"
                "SCHWAB TOKEN EXPIRED - Renewal required for data streaming!\n"
                "Run: python -m data.streaming.authenticator\n"
                "=" * 60
            )
            raise RuntimeError("Schwab token expired - cannot stream data")
        elif schwab_result.status == CredentialStatus.EXPIRING_SOON:
            hours = schwab_result.expires_in // 3600 if schwab_result.expires_in else 0
            self.logger.warning(
                f"SCHWAB TOKEN EXPIRING in {hours} hours. "
                f"Renew soon: python -m data.streaming.authenticator"
            )
        elif schwab_result.status == CredentialStatus.MISSING:
            self.logger.error("Schwab credentials not configured!")
            raise RuntimeError("Schwab credentials missing - cannot stream data")
        elif schwab_result.status == CredentialStatus.VALID:
            days = schwab_result.expires_in // 86400 if schwab_result.expires_in else 0
            self.logger.info(f"✓ Schwab credentials valid ({days} days until refresh expires)")

        # Check Alpaca credentials (for execution)
        alpaca_result = await validator.validate_alpaca()

        if alpaca_result.status == CredentialStatus.VALID:
            self.logger.info("✓ Alpaca credentials valid")
        elif alpaca_result.status == CredentialStatus.MISSING:
            self.logger.error("Alpaca credentials not configured!")
            raise RuntimeError("Alpaca credentials missing - cannot execute trades")
        else:
            self.logger.warning(f"Alpaca credential check: {alpaca_result.status}")

    # ==========================================================================
    # RECONNECTION HANDLING
    # ==========================================================================

    async def _main_loop(self, stream_task: asyncio.Task) -> None:
        """Main run loop with reconnection handling for Schwab stream."""
        try:
            while self._running:
                # Check if stream is still running
                if stream_task.done():
                    exc = stream_task.exception()
                    if exc:
                        self.logger.error(f"Schwab stream task failed: {exc}")

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
            self.logger.info("Hybrid runner cancelled")

    async def _reconnect(self) -> bool:
        """Attempt to reconnect to Schwab streaming."""
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            self.logger.error(
                f"Max reconnection attempts ({self._max_reconnect_attempts}) reached. Giving up."
            )
            return False

        self._reconnect_attempts += 1
        self.logger.warning(
            f"Attempting reconnection {self._reconnect_attempts}/{self._max_reconnect_attempts}..."
        )

        await asyncio.sleep(self._reconnect_delay)

        try:
            # Try to reconnect Schwab streaming
            api_key = os.getenv("SCHWAB_API_KEY")
            secret_key = os.getenv("SCHWAB_SECRET")
            self._schwab_stream_broker.connect_stream(api_key, secret_key)

            # Re-subscribe to quotes
            self._subscribe_to_data()

            self.logger.info("Schwab stream reconnected successfully")
            self._reconnect_attempts = 0
            return True

        except Exception as e:
            self.logger.error(f"Reconnection failed: {e}")
            return False

    async def _emit_health_status(self, status: str, details: Dict = None) -> None:
        """Emit health status event."""
        await self.event_handler.emit(EVENT_HEALTH_UPDATE, {
            "broker": "hybrid",
            "status": status,
            "timestamp": self.scheduler.now_et().isoformat(),
            **(details or {})
        })

    async def _cleanup(self, stream_task: asyncio.Task) -> None:
        """Cleanup with health status emission."""
        await super()._cleanup(stream_task)
        await self._emit_health_status("disconnected", {"reason": "shutdown"})

    # ==========================================================================
    # SCHWAB DATA PROCESSING (Quote -> Bar Aggregation)
    # ==========================================================================

    @staticmethod
    def _canonicalize_schwab_quote(quote: Dict, symbol: str) -> Dict:
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
        trade_time_ms = quote.get('35') or quote.get('trade_time')
        if trade_time_ms:
            ts = datetime.fromtimestamp(int(trade_time_ms) / 1000, tz=timezone.utc)
        else:
            ts = datetime.now(timezone.utc)

        bid_price = quote.get('1') or quote.get('bid_price') or quote.get('bidPrice', 0)
        ask_price = quote.get('2') or quote.get('ask_price') or quote.get('askPrice', 0)
        last_price = quote.get('3') or quote.get('last_price') or quote.get('lastPrice', 0)
        volume = quote.get('8') or quote.get('volume') or quote.get('totalVolume', 0)
        high_price = quote.get('10') or quote.get('high_price') or quote.get('highPrice', 0)
        low_price = quote.get('11') or quote.get('low_price') or quote.get('lowPrice', 0)
        prev_close = quote.get('12') or quote.get('close_price') or quote.get('closePrice', 0)
        open_price = quote.get('17') or quote.get('open_price') or quote.get('openPrice', 0)

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

    def _aggregate_quote_to_bar(self, symbol: str, price: float, volume: int, bar_id: int) -> Dict:
        """
        Aggregate quote data into OHLCV bar.

        Collects multiple quotes within the same time bucket into a proper bar.
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
        async def callback(quote: Dict):
            await self._on_quote(symbol, quote)
        return callback

    async def _on_quote(self, symbol: str, quote: Dict) -> None:
        """
        Handle incoming Schwab quote data.

        Aggregates quotes into OHLCV bars and processes through the strategy engine.
        """
        raw_bar = self._canonicalize_schwab_quote(quote, symbol)
        price = float(raw_bar["Close"])
        volume = int(raw_bar.get("Volume", 0))
        ts: datetime = raw_bar["timestamp"]

        self.logger.debug(f"[SCHWAB QUOTE] {symbol} price={price}")

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

        # Only process completed bars
        if bar_closed:
            self.logger.debug(f"[BAR CLOSED] {symbol} {ts} O={bar['Open']} H={bar['High']} L={bar['Low']} C={bar['Close']} V={bar['Volume']}")
            await self._process_bar(bar)
        else:
            # For incomplete bars, just update current price tracking
            # (base runner handles this via _process_bar even for incomplete bars)
            pass
