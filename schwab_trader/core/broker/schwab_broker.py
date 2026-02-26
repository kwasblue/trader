# brokers/schwab_broker.py
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, time as dt_time
from typing import Optional, List, Dict, Any, Literal, Union, Callable

from core.base.base_broker_interface import BaseBrokerInterface
from core.app_types import OrderResult, PositionView, BrokerSnapshot
from core.events.eventhandler import get_event_handler
from core.contracts.events import EVENT_ORDER_STATUS, EVENT_HEALTH_UPDATE
from core.utils.retry import CircuitBreaker, CircuitBreakerConfig, async_retry
from data.streaming.schwab_client import SchwabClient
from data.streaming.streamer import SchwabStreamingClient
from loggers.logger import Logger


from core.enums import OrderSide

# Keep Side for backward compat with short/cover
Side = Literal["buy", "sell", "short", "cover"]


def _normalize_side(side: OrderSide | str) -> str:
    """Normalize OrderSide enum or string to lowercase string."""
    if isinstance(side, OrderSide):
        return side.value
    return str(side).lower()
OrderType = Literal["market", "limit", "stop", "stop_limit"]
TIF = Literal["day", "gtc", "fok"]
Session = Literal["NORMAL", "AM", "PM"]


from core.utils.type_helpers import to_float as _to_float


class SchwabBroker(BaseBrokerInterface):
    """
    Schwab broker adapter that matches your BaseBrokerInterface and mirrors your AlpacaBroker style.
    - Uses your existing SchwabClient (no changes required)
    - Supports market/limit/stop/stop-limit + OCO helper
    - Async methods run sync client calls in a thread (non-blocking)
    - Optional websocket streaming via your SchwabStreamingClient
    """

    SIDE_MAP: Dict[str, str] = {
        "buy": "BUY",
        "sell": "SELL",
        "short": "SELL_SHORT",
        "cover": "BUY_TO_COVER",
    }
    DURATION_MAP: Dict[str, str] = {
        "day": "DAY",
        "gtc": "GOOD_TILL_CANCEL",
        "fok": "FILL_OR_KILL",
    }

    def __init__(self, client: SchwabClient, session: Session = "NORMAL", account_number: Optional[str] = None):
        self.client = client
        self.session = session
        self.account_number = account_number or self.get_default_account()
        self._last_price: Dict[str, float] = {}
        self.logger = Logger("schwab_broker.log", "SchwabBroker").get_logger()

        # Event handler for emitting order/health events
        self._event_handler = get_event_handler()

        # Streaming (Alpaca-like façade)
        self.stream: Optional[SchwabStreamingClient] = None
        self._stream_task: Optional[asyncio.Task] = None
        self._stream_symbols: set[str] = set()

        # Quote callbacks registry
        self._quote_callbacks: Dict[str, List[Callable]] = {}

        # Circuit breaker for API calls - configurable via config
        from core.config_loader import get_config
        err_config = get_config().error_recovery
        self._circuit_breaker = CircuitBreaker(
            name="schwab_broker",
            config=CircuitBreakerConfig(
                failure_threshold=err_config.circuit_breaker_failure_threshold,
                timeout=err_config.circuit_breaker_timeout,
            )
        )

        # Health check callback
        self._health_callback: Optional[Callable] = None

        # Connection state
        self._connected = False

    # ---------------------------------------------------------------------
    # Connection Lifecycle
    # ---------------------------------------------------------------------
    async def connect(self) -> None:
        """
        Initialize connection to Schwab API.

        Connection status is stored in self._connected.
        Use is_connected property to check connection state.

        Raises:
            RuntimeError: If connection fails
        """
        try:
            # Verify account access
            account = await self._to_thread(self.client.account_number)
            if not account or not account.get("accountNumbers"):
                self.logger.error("Failed to retrieve account information")
                self._connected = False
                await self._emit_health_event("error", "Failed to retrieve account information")
                raise RuntimeError("Failed to retrieve Schwab account information")

            self._connected = True
            self.logger.info(f"Connected to Schwab API, account: {self.account_number}")

            # Emit health event
            await self._emit_health_event("connected")

        except Exception as e:
            self.logger.exception(f"Failed to connect to Schwab: {e}")
            self._connected = False
            await self._emit_health_event("error", str(e))
            raise

    @property
    def is_connected(self) -> bool:
        """Check if broker is connected."""
        return self._connected

    async def disconnect(self):
        """Disconnect from Schwab API and streaming."""
        self._connected = False
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Disconnected from Schwab API")
        await self._emit_health_event("disconnected")

    def set_health_callback(self, callback: Callable):
        """Set callback for health status updates."""
        self._health_callback = callback

    async def _emit_health_event(self, status: str, error: str = None):
        """Emit health status event."""
        payload = {
            "broker": "schwab",
            "status": status,
            "details": {
                "account": self.account_number,
                "connected": self._connected,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if error:
            payload["details"]["error"] = error

        await self._event_handler.emit(EVENT_HEALTH_UPDATE, payload)

        if self._health_callback:
            try:
                self._health_callback(status, payload)
            except Exception as e:
                self.logger.warning(f"Health callback error: {e}")

    # ---------------------------------------------------------------------
    # Internal utilities
    # ---------------------------------------------------------------------
    async def _to_thread(self, fn, *args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    def _instruction(self, side: str) -> str:
        try:
            return self.SIDE_MAP[side.lower()]
        except KeyError:
            raise ValueError(f"Unsupported side '{side}'. Use one of {list(self.SIDE_MAP)}")

    def _duration(self, tif: str) -> str:
        try:
            return self.DURATION_MAP[tif.lower()]
        except KeyError:
            raise ValueError(f"Unsupported time_in_force '{tif}'. Use one of {list(self.DURATION_MAP)}")

    @staticmethod
    def _fmt_price(v: Optional[Union[float, int]]) -> Optional[str]:
        if v is None:
            return None
        return f"{float(v):.2f}"

    # ---------------------------------------------------------------------
    # BaseBrokerInterface — ASYNC
    # ---------------------------------------------------------------------
    @async_retry(max_attempts=3, base_delay=1.0)
    async def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: str = "gtc",
        **kwargs,
    ) -> OrderResult:
        # Circuit breaker check
        if not self._circuit_breaker._should_allow_request():
            self.logger.warning("Circuit breaker is open, rejecting order")
            return OrderResult(
                order_id=None,
                status="rejected",
                symbol=symbol,
                side=side,
                qty=qty,
                raw={"error": "circuit_breaker_open"}
            )

        if qty <= 0:
            raise ValueError("qty must be > 0")

        duration = self._duration(time_in_force)
        instruction = self._instruction(side)
        ot = (order_type or "market").lower()

        # Build order with your client's builder; add price/stop fields afterward when needed
        if ot == "market":
            od = self.client.generate_order(
                orderType="MARKET", session=self.session, duration=duration,
                orderStrategyType="SINGLE", instruction=instruction,
                quantity=int(qty), symbol=symbol, assetType="EQUITY"
            )
        elif ot == "limit":
            if limit_price is None:
                raise ValueError("limit_price required for limit orders")
            od = self.client.generate_order(
                orderType="LIMIT", session=self.session, duration=duration,
                orderStrategyType="SINGLE", instruction=instruction,
                quantity=int(qty), symbol=symbol, assetType="EQUITY"
            )
            od["price"] = self._fmt_price(limit_price)
        elif ot == "stop":
            if stop_price is None:
                raise ValueError("stop_price required for stop orders")
            od = self.client.generate_order(
                orderType="STOP", session=self.session, duration=duration,
                orderStrategyType="SINGLE", instruction=instruction,
                quantity=int(qty), symbol=symbol, assetType="EQUITY"
            )
            od["stopPrice"] = self._fmt_price(stop_price)
        elif ot == "stop_limit":
            if stop_price is None or limit_price is None:
                raise ValueError("Both stop_price and limit_price required for stop_limit orders")
            od = self.client.generate_order(
                orderType="STOP_LIMIT", session=self.session, duration=duration,
                orderStrategyType="SINGLE", instruction=instruction,
                quantity=int(qty), symbol=symbol, assetType="EQUITY"
            )
            od["stopPrice"] = self._fmt_price(stop_price)
            od["price"] = self._fmt_price(limit_price)
        else:
            raise ValueError(f"Unsupported order_type: {order_type}")

        try:
            resp = await self._to_thread(self.client.place_orders, self.account_number, od)
            result = self._mk_order_result(resp, symbol=symbol, qty=qty, side=side,
                                           type=ot, limit_price=limit_price, stop_price=stop_price)

            # Record success for circuit breaker
            self._circuit_breaker.record_success()

            # Emit order status event
            await self._emit_order_event(result, "placed")

            self.logger.info(f"Order placed: {side} {qty} {symbol} @ {order_type}")
            return result

        except Exception as e:
            self._circuit_breaker.record_failure(e)
            self.logger.exception(f"Failed to place order: {e}")
            raise

    @async_retry(max_attempts=3, base_delay=1.0)
    async def cancel_order(self, order_id: str) -> OrderResult:
        """Cancel an order by ID."""
        if not self._circuit_breaker._should_allow_request():
            self.logger.warning("Circuit breaker is open, rejecting cancel request")
            return OrderResult(
                order_id=order_id,
                status="rejected",
                raw={"error": "circuit_breaker_open"}
            )

        try:
            resp = await self._to_thread(self.client.cancel_order, self.account_number, order_id)
            result = self._mk_order_result(resp)

            self._circuit_breaker.record_success()

            # Emit order cancellation event
            await self._emit_order_event(result, "cancelled")

            self.logger.info(f"Order cancelled: {order_id}")
            return result

        except Exception as e:
            self._circuit_breaker.record_failure(e)
            self.logger.exception(f"Failed to cancel order {order_id}: {e}")
            raise

    async def _emit_order_event(self, order_result: OrderResult, action: str):
        """Emit order status event to the event bus."""
        payload = {
            "order_id": order_result.order_id,
            "symbol": order_result.symbol,
            "side": order_result.side,
            "qty": order_result.qty,
            "status": order_result.status,
            "action": action,
            "filled_qty": order_result.filled_qty,
            "avg_fill_price": order_result.avg_fill_price,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self._event_handler.emit(EVENT_ORDER_STATUS, payload)

    @async_retry(max_attempts=3, base_delay=1.0)
    async def get_position(self, symbol: str) -> Optional[PositionView]:
        acct = await self._to_thread(self.client.accounts_number, self.account_number)
        positions = (acct.get("securitiesAccount", {}) or {}).get("positions", []) or []
        for p in positions:
            if ((p.get("instrument", {}) or {}).get("symbol") == symbol):
                return self._mk_position_view(p)
        return None

    @async_retry(max_attempts=3, base_delay=1.0)
    async def get_account_info(self) -> BrokerSnapshot:
        acct = await self._to_thread(self.client.accounts_number, self.account_number)
        # Extract positions from account response
        positions_list = (acct.get("securitiesAccount", {}) or {}).get("positions", []) or []
        positions = {}
        for p in positions_list:
            symbol = (p.get("instrument", {}) or {}).get("symbol")
            if symbol:
                positions[symbol] = self._mk_position_view(p)
        return self._mk_broker_snapshot(acct, positions)

    async def is_market_open(self) -> bool:
        """
        Check if the US equity market is currently open.

        Uses Eastern Time to determine market hours:
        - Regular hours: 9:30 AM - 4:00 PM ET, Mon-Fri
        - Pre-market: 4:00 AM - 9:30 AM ET (if SCHWAB_SESSION == "AM")
        - After-hours: 4:00 PM - 8:00 PM ET (if SCHWAB_SESSION == "PM")

        Returns:
            True if market is open for the configured session type
        """
        try:
            # Get current time in Eastern Time
            from zoneinfo import ZoneInfo
            et_tz = ZoneInfo("America/New_York")
            now_et = datetime.now(et_tz)

            # Check if it's a weekday
            if now_et.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False

            current_time = now_et.time()

            # Define market hours
            market_open = dt_time(9, 30)
            market_close = dt_time(16, 0)
            premarket_open = dt_time(4, 0)
            afterhours_close = dt_time(20, 0)

            # Check based on session type
            if self.session == "NORMAL":
                return market_open <= current_time <= market_close
            elif self.session == "AM":
                # Pre-market through regular hours
                return premarket_open <= current_time <= market_close
            elif self.session == "PM":
                # Regular hours through after-hours
                return market_open <= current_time <= afterhours_close
            else:
                # Default to regular hours
                return market_open <= current_time <= market_close

        except ImportError:
            # Fallback if zoneinfo not available (Python < 3.9)
            try:
                import pytz
                et_tz = pytz.timezone("America/New_York")
                now_et = datetime.now(et_tz)

                if now_et.weekday() >= 5:
                    return False

                current_time = now_et.time()
                market_open = dt_time(9, 30)
                market_close = dt_time(16, 0)

                return market_open <= current_time <= market_close
            except ImportError:
                # If no timezone library available, assume market is CLOSED for safety
                self.logger.warning("No timezone library available, assuming market CLOSED (safe default)")
                return False
        except Exception as e:
            # SAFETY: On error, assume market is CLOSED to prevent unintended trades
            self.logger.warning(f"Error checking market hours: {e}, assuming CLOSED (safe default)")
            return False

    @async_retry(max_attempts=3, base_delay=1.0)
    async def get_open_orders(self) -> List[OrderResult]:
        resp = await self._to_thread(self.client.all_orders, self.account_number)
        orders = resp if isinstance(resp, list) else (resp.get("orders", []) if isinstance(resp, dict) else [])
        return [self._mk_order_result(o) for o in orders]

    @async_retry(max_attempts=3, base_delay=1.0)
    async def get_order_status(self, order_id: str) -> OrderResult:
        resp = await self._to_thread(self.client.get_order_by_id, self.account_number, order_id)
        return self._mk_order_result(resp)

    # ---------------------------------------------------------------------
    # BaseBrokerInterface — ASYNC order helpers
    # ---------------------------------------------------------------------
    async def place_market_order(self, symbol: str, qty: int, side, price: float | None = None) -> OrderResult:
        """
        Place a market order asynchronously.

        Args:
            symbol: Trading symbol
            qty: Quantity to trade
            side: OrderSide enum or string ("buy"/"sell")
            price: Optional price hint for mark-to-market

        Returns:
            OrderResult with execution details
        """
        # Handle both OrderSide enum and string
        if hasattr(side, 'value'):
            side_str = side.value.lower()
        else:
            side_str = str(side).lower()

        od = self.client.generate_order(
            orderType="MARKET", session=self.session, duration="DAY",
            orderStrategyType="SINGLE", instruction=self._instruction(side_str),
            quantity=int(qty), symbol=symbol, assetType="EQUITY"
        )
        resp = await self._to_thread(self.client.place_orders, self.account_number, od)
        if price is not None:
            self.mark_price(symbol, float(price))
        result = self._mk_order_result(resp, symbol=symbol, qty=qty, side=side_str, type="market")

        # Emit order event
        await self._emit_order_event(result, "placed")

        return result

    async def place_oco_order(self, symbol: str, qty: int, stop_price: float, limit_price: float) -> OrderResult:
        """
        Place OCO (One-Cancels-Other) bracket order asynchronously.

        Creates two orders: a stop-loss and a take-profit. When one
        executes, the other is automatically cancelled.

        Args:
            symbol: Trading symbol
            qty: Quantity to trade
            stop_price: Stop-loss trigger price
            limit_price: Take-profit limit price

        Returns:
            OrderResult for the OCO order group
        """
        # Typical long exit: SELL limit (TP) + SELL stop (SL)
        child_limit = self.client.generate_order(
            orderType="LIMIT", session=self.session, duration="DAY",
            orderStrategyType="SINGLE", instruction="SELL",
            quantity=int(qty), symbol=symbol, assetType="EQUITY"
        )
        child_limit["price"] = self._fmt_price(limit_price)

        child_stop = self.client.generate_order(
            orderType="STOP", session=self.session, duration="DAY",
            orderStrategyType="SINGLE", instruction="SELL",
            quantity=int(qty), symbol=symbol, assetType="EQUITY"
        )
        child_stop["stopPrice"] = self._fmt_price(stop_price)

        oco = {
            "orderStrategyType": "OCO",
            "session": self.session,
            "duration": "DAY",
            "childOrderStrategies": [child_limit, child_stop],
        }
        resp = await self._to_thread(self.client.place_orders, self.account_number, oco)
        result = self._mk_order_result(resp, symbol=symbol, qty=qty, type="oco",
                                       limit_price=limit_price, stop_price=stop_price)

        # Emit order event
        await self._emit_order_event(result, "placed")

        return result

    # ---------------------------------------------------------------------
    # BaseBrokerInterface — INFO
    # ---------------------------------------------------------------------
    def get_default_account(self) -> str:
        data = self.client.account_number()
        return data.get("accountNumbers", [{}])[0].get("accountNumber")

    def get_quote(self, symbol: str, fallback_to_cache: bool = True) -> float:
        """
        Get quote for a symbol.

        Args:
            symbol: Trading symbol
            fallback_to_cache: If True, return last known price if API fails

        Returns:
            Current price for the symbol

        Raises:
            RuntimeError: If no quote available and no cached price
        """
        try:
            raw = self.client.quote(symbol)
            # Normalize common shapes: {"AAPL": {...}} or {"quotes":{"AAPL": {...}}}
            d = None
            if isinstance(raw, dict):
                d = raw.get(symbol)
                if d is None and isinstance(raw.get("quotes", {}), dict):
                    d = raw["quotes"].get(symbol)
            price = None
            if isinstance(d, dict):
                price = d.get("lastPrice") or d.get("last") or d.get("mark") or d.get("close")
            if price is not None:
                self.mark_price(symbol, float(price))
                return float(price)
        except Exception as e:
            self.logger.warning(f"Quote API failed for {symbol}: {e}")

        # Fallback to cached price
        if fallback_to_cache and symbol in self._last_price:
            self.logger.warning(f"Using cached price for {symbol}: ${self._last_price[symbol]:.2f}")
            return self._last_price[symbol]

        raise RuntimeError(f"No quote available for {symbol}")

    def get_available_funds(self) -> float:
        acct = self.client.accounts_number(self.account_number)
        return float((acct.get("securitiesAccount", {}) or {}).get("currentBalances", {}).get("availableFunds", 0.0))

    def get_buying_power(self) -> float:
        """
        Get buying power from broker API.

        Returns:
            Available buying power in dollars
        """
        acct = self.client.accounts_number(self.account_number)
        balances = (acct.get("securitiesAccount", {}) or {}).get("currentBalances", {})
        return float(balances.get("buyingPower", 0.0) or 0.0)

    def mark_price(self, symbol: str, price: float) -> None:
        self._last_price[symbol] = float(price)

    # ---------------------------------------------------------------------
    # Streaming API (Alpaca-like)
    # ---------------------------------------------------------------------
    def connect_stream(self, api_key: str, secret_key: str):
        """Initialize the Schwab websocket streamer (no network call yet)."""
        self.stream = SchwabStreamingClient(api_key, secret_key)
        # Wire up the quote dispatcher
        self.stream.set_quote_callback(self._dispatch_quote)

    def subscribe_quotes(self, callback: Callable, symbol: str):
        """
        Register a quote handler for a symbol.

        The callback will be invoked with quote data when new quotes arrive.
        Works before or after stream start.

        Args:
            callback: Async or sync function to call with quote data
            symbol: The symbol to subscribe to
        """
        if symbol not in self._quote_callbacks:
            self._quote_callbacks[symbol] = []
        if callback not in self._quote_callbacks[symbol]:
            self._quote_callbacks[symbol].append(callback)
        self._stream_symbols.add(symbol)
        self.logger.debug(f"Subscribed to quotes for {symbol}")

    def unsubscribe_quotes(self, callback: Callable, symbol: str):
        """
        Unsubscribe a callback from quote updates.

        Args:
            callback: The callback to remove
            symbol: The symbol to unsubscribe from
        """
        if symbol in self._quote_callbacks and callback in self._quote_callbacks[symbol]:
            self._quote_callbacks[symbol].remove(callback)
            self.logger.debug(f"Unsubscribed from quotes for {symbol}")

    async def _dispatch_quote(self, symbol: str, quote: Dict):
        """
        Dispatch quote to all registered callbacks for a symbol.

        Args:
            symbol: The symbol this quote is for
            quote: The quote data
        """
        callbacks = self._quote_callbacks.get(symbol, [])
        for cb in callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(quote)
                else:
                    cb(quote)
            except Exception as e:
                self.logger.exception(f"Error in quote callback for {symbol}: {e}")

    async def start_stream(self):
        """Start the websocket loop with currently-subscribed symbols."""
        if not self.stream:
            raise RuntimeError("Stream not initialized. Call connect_stream(...) first.")
        if self._stream_task and not self._stream_task.done():
            return  # already running
        symbols = sorted(self._stream_symbols) or []
        self._stream_task = asyncio.create_task(self.stream.run(symbols))
        await asyncio.sleep(0)

    async def stop_stream(self):
        """Stop the streaming connection."""
        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        self._stream_task = None
        self.logger.info("Streaming stopped")

    def subscribe_bars(self, callback: Callable, symbol: str):
        """
        Alias for subscribe_quotes for compatibility with AlpacaBroker interface.

        Note: Schwab streaming provides quotes, not bars. The runner must
        aggregate quotes into bars if needed.
        """
        self.subscribe_quotes(callback, symbol)

    # ---------------------------------------------------------------------
    # Mappers: Schwab payloads -> your domain types
    # ---------------------------------------------------------------------
    def _mk_order_result(self, resp: Any, **hint) -> OrderResult:
        if not isinstance(resp, dict):
            return OrderResult(order_id=None, status="unknown", raw=resp)  # type: ignore[arg-type]

        legs = resp.get("orderLegCollection", []) or resp.get("legs", []) or []
        symbol = hint.get("symbol")
        qty = hint.get("qty")
        side = hint.get("side")
        if legs:
            leg0 = legs[0]
            symbol = symbol or ((leg0.get("instrument", {}) or {}).get("symbol"))
            qty = qty or leg0.get("quantity")
            side = side or leg0.get("instruction")

        order_id = resp.get("orderId") or resp.get("id") or resp.get("order_id")
        status = resp.get("status") or resp.get("orderStatus") or resp.get("state") or resp.get("error")
        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=(str(side).lower() if side else None),
            qty=_to_float(qty) or hint.get("qty"),
            type=hint.get("type"),
            time_in_force=None,
            status=(str(status).lower() if status else "submitted"),
            limit_price=_to_float(hint.get("limit_price")),
            stop_price=_to_float(hint.get("stop_price")),
            filled_qty=_to_float(resp.get("filledQuantity")) or 0.0,
            avg_fill_price=_to_float(resp.get("averagePrice")),
            raw=resp,
        )

    def _mk_position_view(self, p: Dict[str, Any]) -> PositionView:
        inst = p.get("instrument", {}) or {}
        sym = inst.get("symbol")
        long_q = _to_float(p.get("longQuantity")) or 0.0
        short_q = _to_float(p.get("shortQuantity")) or 0.0
        qty = long_q if long_q > 0 else (short_q if short_q > 0 else 0.0)
        side = "short" if short_q > 0 else ("long" if long_q > 0 else "flat")
        avg = _to_float(p.get("averagePrice")) or 0.0
        mp = self._last_price.get(sym or "", None)
        return PositionView(
            symbol=sym,
            qty=float(qty),
            avg_entry_price=float(avg),
            market_price=(float(mp) if mp is not None else None),
            side=side,
        )

    def _mk_broker_snapshot(self, acct: Dict[str, Any], positions: dict = None) -> BrokerSnapshot:
        bal = (acct.get("securitiesAccount", {}) or {}).get("currentBalances", {}) or {}
        return BrokerSnapshot(
            account_number=self.account_number,
            status=str((acct.get("securitiesAccount", {}) or {}).get("type", "")),
            cash=_to_float(bal.get("availableFunds")) or 0.0,
            buying_power=_to_float(bal.get("buyingPower")) or (_to_float(bal.get("availableFunds")) or 0.0),
            equity=_to_float(bal.get("liquidationValue")) or 0.0,
            portfolio_value=_to_float(bal.get("liquidationValue")) or 0.0,
            positions=positions,
        )
