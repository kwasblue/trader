from __future__ import annotations

import asyncio
import inspect
import uuid
from datetime import datetime, timezone
from typing import Any

import alpaca
from alpaca.data.enums import DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    OrderClass,
    QueryOrderStatus,
    TimeInForce,
)
from alpaca.trading.enums import (
    OrderSide as AlpacaOrderSide,
)
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
)

from core.app_types import BrokerSnapshot, OrderResult, PositionView
from core.base.base_broker_interface import BaseBrokerInterface
from core.contracts.events import (
    EVENT_ALERT,
    EVENT_HEALTH_UPDATE,
    EVENT_ORDER_STATUS,
    AlertPayload,
    HealthPayload,
    OrderStatusPayload,
)
from core.enums import OrderSide  # Our canonical enum
from core.events.eventhandler import EventHandler, get_event_handler
from core.utils.health import get_health_checker
from core.utils.retry import async_retry, retry
from loggers.logger import Logger


def _map_side(side: OrderSide | str) -> AlpacaOrderSide:
    """Convert our OrderSide enum or string to Alpaca's OrderSide enum."""
    if isinstance(side, OrderSide):
        side_str = side.value
    else:
        side_str = str(side).lower()
    return AlpacaOrderSide.BUY if side_str == "buy" else AlpacaOrderSide.SELL


def _map_tif(tif: str) -> TimeInForce:
    m = {
        "day": TimeInForce.DAY,
        "gtc": TimeInForce.GTC,
        "opg": TimeInForce.OPG,
        "cls": TimeInForce.CLS,
        "ioc": TimeInForce.IOC,
        "fok": TimeInForce.FOK,
    }
    return m.get(tif.lower(), TimeInForce.GTC)


class AlpacaBroker(BaseBrokerInterface):
    """
    Broker implementation for Alpaca using the Alpaca SDK (TradingClient + Data Stream).

    Minimal surface to match your BaseBrokerInterface plus the small helpers
    you originally used (connect/start_stream/subscribe_bars/execute_trade).
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        event_handler: EventHandler | None = None,
        paper: bool = True,
        poll_timeout: int = 30,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self.trading_client: TradingClient | None = None
        self.stream: StockDataStream | None = None
        self.data_rest: StockHistoricalDataClient | None = None
        self._last_price: dict[str, float] = {}
        self._connected = False
        self.logger = Logger("alpaca.log", "AlpacaBroker").get_logger()
        self.event_handler = get_event_handler()
        # Track subscriptions for re-subscribing after reconnection
        self._bar_subscriptions: dict[str, Any] = {}  # symbol -> callback
        self._streaming = False  # Flag to control streaming loop

        # Polling fallback
        self._last_bar_time: dict[str, datetime] = {}  # symbol -> last bar timestamp
        self._polling_active = False  # Whether polling fallback is active
        self._poll_interval = 60  # Poll every 60 seconds
        self._stale_threshold = 120  # Consider stale if no bar for 2 minutes
        self._poll_timeout = poll_timeout  # Timeout for poll REST requests

    # ---------------------------------------------------------
    # Connections / Streaming
    # ---------------------------------------------------------
    @async_retry(max_attempts=3, base_delay=2.0, circuit_breaker="alpaca_connect")
    async def connect(self) -> None:
        """
        Establish trading and streaming connections with retry logic.

        Returns:
            None (connection status stored in self._connected)
        """
        # Close any existing connections first to avoid connection limit issues
        if self.stream is not None:
            self.logger.warning("Closing existing stream before reconnecting...")
            self.disconnect()

        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=self.paper)
        # Websocket params for better connection stability
        # Increased ping_timeout to 60s to handle high-latency or congested network conditions
        ws_params = {"ping_interval": 30, "ping_timeout": 60, "close_timeout": 15}
        self.stream = StockDataStream(self.api_key, self.api_secret, feed=DataFeed.IEX, websocket_params=ws_params)
        self.data_rest = StockHistoricalDataClient(self.api_key, self.api_secret)
        self._connected = True
        self.logger.info("Connected to Alpaca.")

    @retry(max_attempts=3, base_delay=2.0, circuit_breaker="alpaca_connect")
    def connect_sync(self) -> None:
        """
        Sync wrapper for connect(). For backward compatibility.

        Use `await connect()` in async code.
        """
        # Close any existing connections first to avoid connection limit issues
        if self.stream is not None:
            self.logger.warning("Closing existing stream before reconnecting...")
            self.disconnect()

        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=self.paper)
        # Websocket params for better connection stability
        # Increased ping_timeout to 60s to handle high-latency or congested network conditions
        ws_params = {"ping_interval": 30, "ping_timeout": 60, "close_timeout": 15}
        self.stream = StockDataStream(self.api_key, self.api_secret, feed=DataFeed.IEX, websocket_params=ws_params)
        self.data_rest = StockHistoricalDataClient(self.api_key, self.api_secret)
        self._connected = True
        self.logger.info("Connected to Alpaca.")

    def disconnect(self):
        """Close all connections and clean up resources."""
        self._connected = False
        self._streaming = False  # Signal stream loop to exit
        if self.stream is not None:
            try:
                # Stop the websocket stream
                if hasattr(self.stream, "stop"):
                    self.stream.stop()
                elif hasattr(self.stream, "close"):
                    self.stream.close()
                self.logger.info("Stream disconnected.")
            except Exception as e:
                self.logger.warning(f"Error closing stream: {e}")
            finally:
                self.stream = None

        self.trading_client = None
        self.data_rest = None
        self.logger.info("Disconnected from Alpaca.")

        # Register health check
        health_checker = get_health_checker()
        health_checker.register("alpaca_broker", self._health_check, metadata={"paper": self.paper})

        if self.event_handler:
            payload: HealthPayload = {
                "broker": "alpaca",
                "status": "connected",
                "details": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            asyncio.create_task(self.event_handler.emit(EVENT_HEALTH_UPDATE, payload))

    async def _health_check(self) -> bool:
        """Health check for Alpaca connection."""
        try:
            if not self.trading_client:
                return False
            # Try to get account info as health check
            account = await asyncio.to_thread(self.trading_client.get_account)
            return account is not None
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False

    async def _heartbeat_loop(self, interval: int = 10):
        """Emit periodic broker heartbeat."""
        while self._streaming:
            if self.event_handler:
                payload: HealthPayload = {
                    "broker": "alpaca",
                    "status": "alive",
                    "details": {},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await self.event_handler.emit(EVENT_HEALTH_UPDATE, payload)
            await asyncio.sleep(interval)
        self.logger.debug("[heartbeat] Heartbeat loop ended")

    async def start_stream(self):
        """Start the data stream with automatic reconnection."""
        if not self.stream:
            self.logger.warning("Stream not initialized. Call connect() first.")
            if self.event_handler:
                await self.event_handler.emit(
                    EVENT_HEALTH_UPDATE,
                    {
                        "broker": "alpaca",
                        "status": "not_initialized",
                        "details": {"error": "Stream not initialized"},
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
            return

        self._streaming = True
        self.logger.info(
            f"[stream] starting... alpaca-py={getattr(alpaca, '__version__', 'unknown')} feed={getattr(self, 'feed', '?')}"
        )
        if self.event_handler:
            payload: HealthPayload = {
                "broker": "alpaca",
                "status": "stream_starting",
                "details": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await self.event_handler.emit(EVENT_HEALTH_UPDATE, payload)
        # launch heartbeat task
        asyncio.create_task(self._heartbeat_loop(interval=15))
        # Start polling fallback loop
        asyncio.create_task(self._polling_fallback_loop())

        reconnect_delay = 5  # Start with 5 seconds
        max_delay = 60  # Cap at 60 seconds

        while self._streaming:
            try:
                run_fn = self.stream.run
                if inspect.iscoroutinefunction(run_fn):
                    await run_fn()
                else:
                    await asyncio.to_thread(run_fn)

                # Check if we should stop
                if not self._streaming:
                    self.logger.info("[stream] Stop requested, exiting stream loop")
                    break

                # Stream exited cleanly - recreate stream, re-subscribe, and restart
                self.logger.warning("[RECONNECTING] Stream exited cleanly; reconnecting in 5s...")
                reconnect_delay = 5  # Reset on clean exit
                if self.event_handler:
                    payload: HealthPayload = {
                        "broker": "alpaca",
                        "status": "stream_exited",
                        "details": {},
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    await self.event_handler.emit(EVENT_HEALTH_UPDATE, payload)

                await asyncio.sleep(5)

                if not self._streaming:
                    break

                # Recreate stream and re-subscribe
                self._recreate_stream_and_resubscribe()

            except asyncio.CancelledError:
                self.logger.info("[stream] Stream cancelled, exiting")
                self._streaming = False
                break

            except Exception as e:
                if not self._streaming:
                    self.logger.info("[stream] Stop requested during error handling, exiting")
                    break

                self.logger.error(f"[RECONNECTING] Stream error: {e}; reconnecting in {reconnect_delay}s")
                if self.event_handler:
                    alert_payload: AlertPayload = {
                        "level": "error",
                        "message": f"Stream error: {e}",
                        "symbol": None,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    await self.event_handler.emit(EVENT_ALERT, alert_payload)

                    health_payload: HealthPayload = {
                        "broker": "alpaca",
                        "status": "reconnecting",
                        "details": {"error": str(e), "retry_in": reconnect_delay},
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    await self.event_handler.emit(EVENT_HEALTH_UPDATE, health_payload)

                await asyncio.sleep(reconnect_delay)
                # Exponential backoff: 5 -> 10 -> 20 -> 40 -> 60 (capped)
                reconnect_delay = min(reconnect_delay * 2, max_delay)

                if not self._streaming:
                    break

                # Recreate stream and re-subscribe
                self._recreate_stream_and_resubscribe()

        self.logger.info("[stream] Stream loop ended")

    def subscribe_bars(self, callback, symbol: str):
        if not self.stream:
            self.logger.warning("Stream not initialized. Call connect() first.")
            return

        if not asyncio.iscoroutinefunction(callback):

            async def _async_wrap(bar):
                return callback(bar)

            base_cb = _async_wrap
        else:
            base_cb = callback

        # Wrap callback to track when bars are received (for polling fallback)
        async def _tracking_callback(bar):
            self.record_bar_received(symbol)
            return await base_cb(bar)

        # Track subscription for re-subscribing after reconnection
        self._bar_subscriptions[symbol] = _tracking_callback

        self.stream.subscribe_bars(_tracking_callback, symbol)
        self.logger.info(f"[stream] subscribed bars: {symbol}")

        if self.event_handler:
            asyncio.create_task(
                self.event_handler.emit(
                    EVENT_HEALTH_UPDATE,
                    {
                        "broker": "alpaca",
                        "status": "subscribed",
                        "details": {"symbol": symbol},
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
            )

    def _resubscribe_all(self):
        """Re-subscribe to all tracked bar subscriptions after reconnection."""
        if not self.stream or not self._bar_subscriptions:
            return

        self.logger.info(f"[stream] Re-subscribing to {len(self._bar_subscriptions)} symbols...")
        for symbol, callback in self._bar_subscriptions.items():
            self.stream.subscribe_bars(callback, symbol)
            self.logger.info(f"[stream] re-subscribed bars: {symbol}")

    def _recreate_stream_and_resubscribe(self):
        """Recreate the stream object and re-subscribe to all bars."""
        # Close old stream to avoid connection leaks
        if self.stream is not None:
            self.logger.info("[stream] Closing old stream object...")
            try:
                # Try to close the websocket connection
                if hasattr(self.stream, "close"):
                    self.stream.close()
                elif hasattr(self.stream, "stop"):
                    self.stream.stop()
                elif hasattr(self.stream, "_ws") and self.stream._ws is not None:
                    # Direct websocket access as fallback
                    pass  # Let it be garbage collected
            except Exception as e:
                self.logger.debug(f"[stream] Error closing old stream (expected): {e}")
            finally:
                self.stream = None

        self.logger.info("[stream] Creating new stream object...")
        ws_params = {"ping_interval": 30, "ping_timeout": 30, "close_timeout": 10}
        self.stream = StockDataStream(self.api_key, self.api_secret, feed=DataFeed.IEX, websocket_params=ws_params)
        self._resubscribe_all()

        # Confirm reconnection
        symbols = list(self._bar_subscriptions.keys())
        self.logger.info(f"[RECONNECTED] Stream restored with {len(symbols)} subscriptions: {symbols}")

        if self.event_handler:
            asyncio.create_task(
                self.event_handler.emit(
                    EVENT_HEALTH_UPDATE,
                    {
                        "broker": "alpaca",
                        "status": "reconnected",
                        "details": {"symbols": symbols},
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
            )

    # ---------------------------------------------------------
    # Polling Fallback
    # ---------------------------------------------------------

    def record_bar_received(self, symbol: str):
        """Record that a bar was received for a symbol (called by streaming callback)."""
        self._last_bar_time[symbol] = datetime.now(timezone.utc)

    def _get_stale_symbols(self) -> list[str]:
        """Get symbols that haven't received a bar recently."""
        now = datetime.now(timezone.utc)
        stale = []
        for symbol in self._bar_subscriptions.keys():
            last_time = self._last_bar_time.get(symbol)
            if last_time is None or (now - last_time).total_seconds() > self._stale_threshold:
                stale.append(symbol)
        return stale

    async def _poll_bars(self, symbols: list[str]) -> dict[str, Any]:
        """
        Fetch latest bars for symbols via REST API.

        Returns dict of symbol -> bar data
        """
        if not self.data_rest or not symbols:
            self.logger.warning("[poll] data_rest not initialized, skipping poll")
            return {}

        # Check for valid API keys
        if not self.api_key or not self.api_secret:
            self.logger.error("[poll] API keys not set, cannot poll")
            return {}

        try:
            # Get the last 2 bars to ensure we have the most recent complete bar
            request = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Minute, limit=2)

            # Use wait_for to prevent hanging indefinitely on API calls
            bars = await asyncio.wait_for(
                asyncio.to_thread(self.data_rest.get_stock_bars, request), timeout=self._poll_timeout
            )

            result = {}
            for symbol in symbols:
                symbol_bars = bars.data.get(symbol, []) if bars.data else []
                if symbol_bars:
                    # Get the most recent bar
                    latest = symbol_bars[-1]
                    result[symbol] = latest

            return result

        except asyncio.TimeoutError:
            self.logger.error(f"[poll] Timeout after {self._poll_timeout}s fetching bars for {len(symbols)} symbols")
            return {}
        except Exception as e:
            self.logger.error(f"[poll] Error fetching bars: {e}")
            # If auth error, log more details
            if "unauthorized" in str(e).lower() or "401" in str(e):
                self.logger.error("[poll] Auth error - check ALPACA_API_KEY and ALPACA_SECRET_KEY env vars")
            return {}

    async def _polling_fallback_loop(self):
        """
        Background task that polls for bars when streaming is stale.

        Checks every poll_interval seconds. If any symbols haven't received
        a bar via streaming in stale_threshold seconds, fetches via REST.
        """
        self.logger.info(
            f"[poll] Polling fallback started (interval={self._poll_interval}s, stale={self._stale_threshold}s)"
        )

        while self._streaming:
            await asyncio.sleep(self._poll_interval)

            if not self._streaming:
                break

            # Check for stale symbols
            stale_symbols = self._get_stale_symbols()

            if stale_symbols:
                self._polling_active = True
                self.logger.info(f"[poll] Fetching bars for {len(stale_symbols)} stale symbols: {stale_symbols}")

                # Fetch bars via REST
                bars = await self._poll_bars(stale_symbols)

                # Route to callbacks
                for symbol, bar in bars.items():
                    callback = self._bar_subscriptions.get(symbol)
                    if callback:
                        try:
                            await callback(bar)
                            self.record_bar_received(symbol)
                            self.logger.debug(f"[poll] Delivered bar for {symbol}")
                        except Exception as e:
                            self.logger.error(f"[poll] Error in callback for {symbol}: {e}")

                if self.event_handler:
                    await self.event_handler.emit(
                        EVENT_HEALTH_UPDATE,
                        {
                            "broker": "alpaca",
                            "status": "polling_fallback",
                            "details": {"symbols": stale_symbols, "bars_fetched": len(bars)},
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )
            else:
                if self._polling_active:
                    self.logger.info("[poll] All symbols receiving streaming data, polling inactive")
                    self._polling_active = False

        self.logger.info("[poll] Polling fallback loop ended")

    # ---------------------------------------------------------
    # Convenience (from your original snippet)
    # ---------------------------------------------------------
    def execute_trade(self, trade: dict) -> bool:
        """Execute a trade using market order.

        trade = {"symbol": str, "qty": int, "side": "buy"|"sell"}
        """
        return self.submit_market_order(
            symbol=trade["symbol"],
            qty=trade["qty"],
            side=trade["side"],
        )

    @retry(max_attempts=3, base_delay=1.0, circuit_breaker="alpaca_orders")
    def submit_market_order(self, symbol: str, qty: int, side: str) -> bool:
        """
        Submit a market order through Alpaca with retry logic.

        Features:
        - Automatic retry on transient failures (3 attempts)
        - Circuit breaker to prevent cascading failures
        - Event emission for order status tracking
        """
        if not self.trading_client:
            self.logger.error("Trading client not initialized. Call `connect()` first.")
            if self.event_handler:
                payload: OrderStatusPayload = {
                    "order_id": "N/A",
                    "symbol": symbol,
                    "status": "rejected",
                    "filled_qty": 0.0,
                    "avg_price": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                asyncio.create_task(self.event_handler.emit(EVENT_ORDER_STATUS, payload))
            return False

        try:
            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=_map_side(side),
                time_in_force=TimeInForce.DAY,
            )
            res = self.trading_client.submit_order(order)

            self.logger.info(f"Submitted {side.upper()} order for {qty} {symbol}")

            if self.event_handler:
                payload: OrderStatusPayload = {
                    "order_id": getattr(res, "id", "N/A"),
                    "symbol": symbol,
                    "status": "submitted",
                    "filled_qty": 0.0,
                    "avg_price": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                asyncio.create_task(self.event_handler.emit(EVENT_ORDER_STATUS, payload))

            return True

        except Exception as e:
            self.logger.error(f"Order failed for {symbol}: {e}")
            if self.event_handler:
                payload: OrderStatusPayload = {
                    "order_id": "N/A",
                    "symbol": symbol,
                    "status": "rejected",
                    "filled_qty": 0.0,
                    "avg_price": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                asyncio.create_task(self.event_handler.emit(EVENT_ORDER_STATUS, payload))
            return False

    # ---------------------------------------------------------
    # BaseBrokerInterface implementation (async + sync)
    # ---------------------------------------------------------

    async def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: str = "gtc",
        max_retries: int = 3,
        retry_delay: float = 5.0,
        **kwargs,
    ) -> OrderResult:
        """
        Place an order asynchronously with retry logic.

        Uses client_order_id to prevent duplicate orders on retry.

        Returns:
            OrderResult with order details and status
        """
        if not self.trading_client:
            if self.event_handler:
                payload: OrderStatusPayload = {
                    "order_id": "N/A",
                    "symbol": symbol,
                    "status": "rejected",
                    "filled_qty": 0.0,
                    "avg_price": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await self.event_handler.emit(EVENT_ORDER_STATUS, payload)
            raise RuntimeError("Not connected: call connect() first")

        tif = _map_tif(time_in_force)
        s = _map_side(side)

        # Generate unique client_order_id to prevent duplicates on retry
        client_order_id = f"ord_{symbol}_{uuid.uuid4().hex[:12]}"

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:

                def _submit():
                    ot = (order_type or "market").lower()
                    if ot == "market":
                        req = MarketOrderRequest(
                            symbol=symbol, qty=qty, side=s, time_in_force=tif, client_order_id=client_order_id
                        )
                    elif ot == "limit":
                        if limit_price is None:
                            raise ValueError("limit_price is required for limit orders")
                        req = LimitOrderRequest(
                            symbol=symbol,
                            qty=qty,
                            side=s,
                            time_in_force=tif,
                            limit_price=limit_price,
                            client_order_id=client_order_id,
                        )
                    else:
                        raise ValueError(f"Unsupported order_type: {order_type}")
                    o = self.trading_client.submit_order(req)
                    return self._mk_order_result(o)

                res = await asyncio.to_thread(_submit)

                if self.event_handler:
                    payload: OrderStatusPayload = {
                        "order_id": res.order_id or "N/A",
                        "symbol": res.symbol or symbol,
                        "status": res.status or "submitted",
                        "filled_qty": res.filled_qty or 0.0,
                        "avg_price": res.avg_fill_price,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    await self.event_handler.emit(EVENT_ORDER_STATUS, payload)

                return res

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check if this is a duplicate order error
                if "duplicate" in error_str or "already exists" in error_str:
                    self.logger.info(f"[{symbol}] Order {client_order_id} already exists, fetching status...")
                    try:
                        existing = await self._get_order_by_client_id(client_order_id)
                        if existing:
                            return existing
                    except Exception as fetch_err:
                        self.logger.warning(f"[{symbol}] Could not fetch existing order: {fetch_err}")

                # Log and retry
                if attempt < max_retries:
                    self.logger.warning(
                        f"[{symbol}] Order failed (attempt {attempt}/{max_retries}): {e}, retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    self.logger.error(f"Order failed for {symbol} after {max_retries} attempts: {e}")
                    if self.event_handler:
                        payload: OrderStatusPayload = {
                            "order_id": "N/A",
                            "symbol": symbol,
                            "status": "rejected",
                            "filled_qty": 0.0,
                            "avg_price": None,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        await self.event_handler.emit(EVENT_ORDER_STATUS, payload)

        raise RuntimeError(f"Order failed after {max_retries} attempts: {last_error}")

    async def cancel_order(self, order_id: str) -> None:  # no return; emits instead
        if not self.trading_client:
            if self.event_handler:
                payload: OrderStatusPayload = {
                    "order_id": order_id,
                    "symbol": "N/A",
                    "status": "rejected",
                    "filled_qty": 0.0,
                    "avg_price": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await self.event_handler.emit(EVENT_ORDER_STATUS, payload)
            raise RuntimeError("Not connected: call connect() first")

        def _cancel():
            self.trading_client.cancel_order_by_id(order_id)
            try:
                o = self.trading_client.get_order_by_id(order_id)
                return self._mk_order_result(o)
            except Exception:
                return OrderResult(order_id=order_id, status="canceled")

        try:
            res = await asyncio.to_thread(_cancel)

            if self.event_handler:
                payload: OrderStatusPayload = {
                    "order_id": res.order_id or order_id,
                    "symbol": res.symbol or "N/A",
                    "status": res.status or "canceled",
                    "filled_qty": res.filled_qty or 0.0,
                    "avg_price": res.avg_fill_price,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await self.event_handler.emit(EVENT_ORDER_STATUS, payload)

        except Exception as e:
            self.logger.error(f"Cancel order failed for {order_id}: {e}")
            if self.event_handler:
                payload: OrderStatusPayload = {
                    "order_id": order_id,
                    "symbol": "N/A",
                    "status": "rejected",
                    "filled_qty": 0.0,
                    "avg_price": None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                await self.event_handler.emit(EVENT_ORDER_STATUS, payload)
            return res

    async def place_market_order(
        self,
        symbol: str,
        qty: int,
        side,
        price: float | None = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> OrderResult:
        """
        Place a market order asynchronously with retry logic.

        Uses client_order_id to prevent duplicate orders on retry.
        If we timeout waiting for response but order was placed,
        subsequent retries will detect the existing order.

        Args:
            symbol: Trading symbol
            qty: Quantity to trade
            side: OrderSide enum or string ("buy"/"sell")
            price: Optional price hint (ignored for market orders)
            max_retries: Maximum retry attempts (default 3)
            retry_delay: Seconds between retries (default 5)

        Returns:
            OrderResult with execution details
        """
        if not self.trading_client:
            raise RuntimeError("Not connected: call connect() first")

        # Handle both OrderSide enum and string
        if isinstance(side, str):
            alpaca_side = _map_side(side)
        else:
            alpaca_side = _map_side(side.value if hasattr(side, "value") else str(side))

        # Generate unique client_order_id to prevent duplicates on retry
        client_order_id = f"st_{symbol}_{uuid.uuid4().hex[:12]}"

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:

                def _submit():
                    return self.trading_client.submit_order(
                        MarketOrderRequest(
                            symbol=symbol,
                            qty=qty,
                            side=alpaca_side,
                            time_in_force=TimeInForce.DAY,
                            client_order_id=client_order_id,
                        )
                    )

                o = await asyncio.to_thread(_submit)
                res = self._mk_order_result(o)

                # Emit event
                if self.event_handler:
                    payload = {
                        "order_id": res.order_id,
                        "symbol": res.symbol,
                        "status": res.status,
                        "side": res.side,
                        "qty": res.qty,
                        "filled_qty": res.filled_qty,
                        "avg_price": res.avg_fill_price,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    await self.event_handler.emit(EVENT_ORDER_STATUS, payload)

                return res

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check if this is a duplicate order error (order already exists)
                if "duplicate" in error_str or "already exists" in error_str:
                    self.logger.info(f"[{symbol}] Order {client_order_id} already exists, fetching status...")
                    try:
                        existing = await self._get_order_by_client_id(client_order_id)
                        if existing:
                            return existing
                    except Exception as fetch_err:
                        self.logger.warning(f"[{symbol}] Could not fetch existing order: {fetch_err}")

                # Log and retry
                if attempt < max_retries:
                    self.logger.warning(
                        f"[{symbol}] Order failed (attempt {attempt}/{max_retries}): {e}, retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    self.logger.error(f"[{symbol}] Order failed after {max_retries} attempts: {e}")

        # All retries exhausted
        raise RuntimeError(f"Order failed after {max_retries} attempts: {last_error}")

    async def _get_order_by_client_id(self, client_order_id: str) -> OrderResult | None:
        """Fetch an order by client_order_id."""
        try:

            def _fetch():
                return self.trading_client.get_order_by_client_id(client_order_id)

            o = await asyncio.to_thread(_fetch)
            return self._mk_order_result(o)
        except Exception as e:
            self.logger.debug(f"Could not fetch order by client_id {client_order_id}: {e}")
            return None

    async def place_oco_order(
        self,
        symbol: str,
        qty: int,
        stop_price: float,
        limit_price: float,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> OrderResult:
        """
        Place OCO (One-Cancels-Other) bracket order asynchronously with retry logic.

        Uses Alpaca's bracket order_class with take_profit/stop_loss.
        Uses client_order_id to prevent duplicate orders on retry.

        Args:
            symbol: Trading symbol
            qty: Quantity to trade
            stop_price: Stop-loss trigger price
            limit_price: Take-profit limit price
            max_retries: Maximum retry attempts (default 3)
            retry_delay: Seconds between retries (default 5)

        Returns:
            OrderResult for the OCO order group
        """
        if not self.trading_client:
            raise RuntimeError("Not connected: call connect() first")

        # Generate unique client_order_id to prevent duplicates on retry
        client_order_id = f"oco_{symbol}_{uuid.uuid4().hex[:12]}"

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:

                def _submit():
                    req = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=AlpacaOrderSide.SELL,  # typical for exiting a long
                        time_in_force=TimeInForce.GTC,
                        order_class=OrderClass.BRACKET,  # type: ignore[arg-type]
                        take_profit={"limit_price": limit_price},
                        stop_loss={"stop_price": stop_price},
                        client_order_id=client_order_id,
                    )
                    return self.trading_client.submit_order(req)

                o = await asyncio.to_thread(_submit)
                res = self._mk_order_result(o)

                # Emit order status
                if self.event_handler:
                    payload = {
                        "order_id": res.order_id,
                        "symbol": res.symbol,
                        "status": res.status,
                        "side": res.side,
                        "qty": res.qty,
                        "filled_qty": res.filled_qty,
                        "avg_price": res.avg_fill_price,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    await self.event_handler.emit(EVENT_ORDER_STATUS, payload)

                return res

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check if this is a duplicate order error
                if "duplicate" in error_str or "already exists" in error_str:
                    self.logger.info(f"[{symbol}] OCO order {client_order_id} already exists, fetching status...")
                    try:
                        existing = await self._get_order_by_client_id(client_order_id)
                        if existing:
                            return existing
                    except Exception as fetch_err:
                        self.logger.warning(f"[{symbol}] Could not fetch existing order: {fetch_err}")

                # Log and retry
                if attempt < max_retries:
                    self.logger.warning(
                        f"[{symbol}] OCO order failed (attempt {attempt}/{max_retries}): {e}, "
                        f"retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    self.logger.error(f"[{symbol}] OCO order failed after {max_retries} attempts: {e}")

        raise RuntimeError(f"OCO order failed after {max_retries} attempts: {last_error}")

    async def get_position(self, symbol: str) -> PositionView | None:
        if not self.trading_client:
            raise RuntimeError("Not connected: call connect() first")

        def _get():
            try:
                p = self.trading_client.get_open_position(symbol)
                return self._mk_position_view(p)
            except Exception:
                return None

        return await asyncio.to_thread(_get)

    async def get_positions(self) -> list[PositionView]:
        """Get all open positions."""
        if not self.trading_client:
            raise RuntimeError("Not connected: call connect() first")

        def _get_all():
            positions_list = self.trading_client.get_all_positions()
            return [self._mk_position_view(p) for p in positions_list]

        return await asyncio.to_thread(_get_all)

    async def get_account_info(self) -> BrokerSnapshot:
        if not self.trading_client:
            raise RuntimeError("Not connected: call connect() first")

        def _acct():
            a = self.trading_client.get_account()
            # Debug: Log raw Alpaca account values
            self.logger.info(
                f"[ALPACA RAW] cash={getattr(a, 'cash', '?')}, "
                f"equity={getattr(a, 'equity', '?')}, "
                f"buying_power={getattr(a, 'buying_power', '?')}, "
                f"portfolio_value={getattr(a, 'portfolio_value', '?')}"
            )

            # Also fetch all positions
            positions_list = self.trading_client.get_all_positions()
            positions = {}
            for p in positions_list:
                symbol = getattr(p, "symbol", None)
                if symbol:
                    positions[symbol] = self._mk_position_view(p)
                    # Debug: Log raw position values
                    self.logger.info(
                        f"[ALPACA POS] {symbol}: qty={getattr(p, 'qty', '?')}, "
                        f"avg_entry={getattr(p, 'avg_entry_price', '?')}, "
                        f"current={getattr(p, 'current_price', '?')}, "
                        f"unrealized_pl={getattr(p, 'unrealized_pl', '?')}"
                    )
            return self._mk_broker_snapshot(a, positions)

        return await asyncio.to_thread(_acct)

    async def is_market_open(self) -> bool:
        if not self.trading_client:
            raise RuntimeError("Not connected: call connect() first")

        def _clock():
            c = self.trading_client.get_clock()
            return bool(getattr(c, "is_open", False))

        return await asyncio.to_thread(_clock)

    def get_default_account(self) -> str:
        if not self.trading_client:
            raise RuntimeError("Not connected: call connect() first")
        a = self.trading_client.get_account()
        return getattr(a, "account_number", getattr(a, "id", ""))

    def get_quote(self, symbol: str) -> float:
        if not self.data_rest:
            raise RuntimeError("Not connected: call connect() first")
        q = self.data_rest.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))
        quote = q[symbol]
        price = quote.ask_price or quote.bid_price or quote.midpoint
        if price is None:
            raise RuntimeError(f"No quote available for {symbol}")
        return float(price)

    def get_available_funds(self) -> float:
        """Get available funds (cash)."""
        if not self.trading_client:
            raise RuntimeError("Not connected: call connect() first")
        a = self.trading_client.get_account()
        return float(getattr(a, "cash", 0.0))

    def get_buying_power(self) -> float:
        """
        Get buying power from broker API.

        Returns:
            Available buying power in dollars (as reported by Alpaca)
        """
        if not self.trading_client:
            raise RuntimeError("Not connected: call connect() first")
        a = self.trading_client.get_account()
        bp = float(getattr(a, "buying_power", 0.0) or 0.0)
        self.logger.debug(f"Buying power from API: ${bp:,.2f}")
        return bp

    async def get_open_orders(self) -> list[OrderResult]:
        if not self.trading_client:
            raise RuntimeError("Not connected: call connect() first")

        def _orders():
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, limit=500)
            orders = self.trading_client.get_orders(filter=req)
            return [self._mk_order_result(o) for o in orders]

        return await asyncio.to_thread(_orders)

    async def get_order_status(self, order_id: str) -> OrderResult:
        if not self.trading_client:
            raise RuntimeError("Not connected: call connect() first")

        def _status():
            o = self.trading_client.get_order_by_id(order_id)
            return self._mk_order_result(o)

        return await asyncio.to_thread(_status)

    def mark_price(self, symbol: str, price: float) -> None:
        self._last_price[symbol] = float(price)

    # ---------------------------------------------------------
    # Minimal mappers to your domain types
    # ---------------------------------------------------------
    def _mk_order_result(self, o) -> OrderResult:
        try:
            return OrderResult(
                order_id=getattr(o, "id", None) or getattr(o, "client_order_id", None),
                client_order_id=getattr(o, "client_order_id", None),
                symbol=getattr(o, "symbol", None),
                side=str(getattr(o, "side", "")).lower(),
                qty=float(getattr(o, "qty", 0) or getattr(o, "quantity", 0) or 0),
                filled_qty=float(getattr(o, "filled_qty", 0) or 0),
                type=str(getattr(o, "type", "")).lower(),
                time_in_force=str(getattr(o, "time_in_force", "")).lower(),
                status=str(getattr(o, "status", "")).lower(),
                limit_price=_to_float(getattr(o, "limit_price", None)),
                stop_price=_to_float(getattr(o, "stop_price", None)),
                avg_fill_price=_to_float(getattr(o, "filled_avg_price", None)),
            )
        except Exception:
            # Fallback to a minimal dict-like if your dataclass signature differs
            return OrderResult(order_id=getattr(o, "id", None))

    def _mk_position_view(self, p) -> PositionView:
        current_price = _to_float(getattr(p, "current_price", None))
        return PositionView(
            symbol=getattr(p, "symbol", None),
            qty=float(getattr(p, "qty", 0) or 0),
            avg_entry_price=_to_float(getattr(p, "avg_entry_price", None)),
            market_price=current_price,
            last_price=current_price,  # Set last_price for compatibility
            unrealized_pl=_to_float(getattr(p, "unrealized_pl", None)),
            unrealized_plpc=_to_float(getattr(p, "unrealized_plpc", None)),
            side=str(getattr(p, "side", "")).lower(),
        )

    def _mk_broker_snapshot(self, a, positions: dict = None) -> BrokerSnapshot:
        return BrokerSnapshot(
            account_number=getattr(a, "account_number", getattr(a, "id", "")),
            status=getattr(a, "status", ""),
            cash=_to_float(getattr(a, "cash", 0.0)) or 0.0,
            buying_power=_to_float(getattr(a, "buying_power", 0.0)) or 0.0,
            equity=_to_float(getattr(a, "equity", 0.0)) or 0.0,
            portfolio_value=_to_float(getattr(a, "portfolio_value", 0.0)) or 0.0,
            positions=positions,
        )


from core.utils.type_helpers import to_float as _to_float
