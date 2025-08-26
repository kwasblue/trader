# brokers/schwab_broker.py
from __future__ import annotations

import asyncio
from typing import Optional, List, Dict, Any, Literal, Union

from core.base.base_broker_interface import BaseBrokerInterface
from core.app_types import OrderResult, PositionView, BrokerSnapshot
from data.streaming.schwab_client import SchwabClient
from data.streaming.streamer import SchwabStreamingClient
from loggers.logger import Logger


Side = Literal["buy", "sell", "short", "cover"]
OrderType = Literal["market", "limit", "stop", "stop_limit"]
TIF = Literal["day", "gtc", "fok"]
Session = Literal["NORMAL", "AM", "PM"]


def _to_float(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except Exception:
        return None


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

        # Streaming (Alpaca-like façade)
        self.stream: Optional[SchwabStreamingClient] = None
        self._stream_task: Optional[asyncio.Task] = None
        self._stream_symbols: set[str] = set()

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

        resp = await self._to_thread(self.client.place_orders, self.account_number, od)
        return self._mk_order_result(resp, symbol=symbol, qty=qty, side=side,
                                     type=ot, limit_price=limit_price, stop_price=stop_price)

    async def cancel_order(self, order_id: str) -> OrderResult:
        resp = await self._to_thread(self.client.cancel_order, self.account_number, order_id)
        return self._mk_order_result(resp)

    async def get_position(self, symbol: str) -> Optional[PositionView]:
        acct = await self._to_thread(self.client.accounts_number, self.account_number)
        positions = (acct.get("securitiesAccount", {}) or {}).get("positions", []) or []
        for p in positions:
            if ((p.get("instrument", {}) or {}).get("symbol") == symbol):
                return self._mk_position_view(p)
        return None

    async def get_account_info(self) -> BrokerSnapshot:
        acct = await self._to_thread(self.client.accounts_number, self.account_number)
        return self._mk_broker_snapshot(acct)

    async def is_market_open(self) -> bool:
        # If you later expose a market-hours endpoint, call it here.
        return True

    async def get_open_orders(self) -> List[OrderResult]:
        resp = await self._to_thread(self.client.all_orders, self.account_number)
        orders = resp if isinstance(resp, list) else (resp.get("orders", []) if isinstance(resp, dict) else [])
        return [self._mk_order_result(o) for o in orders]

    async def get_order_status(self, order_id: str) -> OrderResult:
        resp = await self._to_thread(self.client.get_order_by_id, self.account_number, order_id)
        return self._mk_order_result(resp)

    # ---------------------------------------------------------------------
    # BaseBrokerInterface — SYNC helpers
    # ---------------------------------------------------------------------
    def place_market_order(self, symbol: str, qty: int, side: str, price: float | None = None) -> OrderResult:
        od = self.client.generate_order(
            orderType="MARKET", session=self.session, duration="DAY",
            orderStrategyType="SINGLE", instruction=self._instruction(side),
            quantity=int(qty), symbol=symbol, assetType="EQUITY"
        )
        resp = self.client.place_orders(self.account_number, od)
        if price is not None:
            self.mark_price(symbol, float(price))
        return self._mk_order_result(resp, symbol=symbol, qty=qty, side=side, type="market")

    def place_oco_order(self, symbol: str, qty: int, stop_price: float, limit_price: float) -> OrderResult:
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
        resp = self.client.place_orders(self.account_number, oco)
        return self._mk_order_result(resp, symbol=symbol, qty=qty, type="oco",
                                     limit_price=limit_price, stop_price=stop_price)

    # ---------------------------------------------------------------------
    # BaseBrokerInterface — INFO
    # ---------------------------------------------------------------------
    def get_default_account(self) -> str:
        data = self.client.account_number()
        return data.get("accountNumbers", [{}])[0].get("accountNumber")

    def get_quote(self, symbol: str) -> float:
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
        raise RuntimeError(f"No quote available for {symbol}")

    def get_available_funds(self) -> float:
        acct = self.client.accounts_number(self.account_number)
        return float((acct.get("securitiesAccount", {}) or {}).get("currentBalances", {}).get("availableFunds", 0.0))

    def mark_price(self, symbol: str, price: float) -> None:
        self._last_price[symbol] = float(price)

    # ---------------------------------------------------------------------
    # Streaming API (Alpaca-like)
    # ---------------------------------------------------------------------
    def connect_stream(self, api_key: str, secret_key: str):
        """Initialize the Schwab websocket streamer (no network call yet)."""
        self.stream = SchwabStreamingClient(api_key, secret_key)

    def subscribe_quotes(self, callback, symbol: str):
        """Register a quote handler for a symbol; works before/after start."""
        if not self.stream:
            raise RuntimeError("Stream not initialized. Call connect_stream(api_key, secret_key) first.")
        self.stream.on_quote(symbol, callback)
        self._stream_symbols.add(symbol)

    async def start_stream(self):
        """Start the websocket loop with currently-subscribed symbols."""
        if not self.stream:
            raise RuntimeError("Stream not initialized. Call connect_stream(...) first.")
        if self._stream_task and not self._stream_task.done():
            return  # already running
        symbols = sorted(self._stream_symbols) or []
        self._stream_task = asyncio.create_task(self.stream.run(symbols))
        await asyncio.sleep(0)

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
            filled_qty=_to_float(resp.get("filledQuantity")),
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

    def _mk_broker_snapshot(self, acct: Dict[str, Any]) -> BrokerSnapshot:
        bal = (acct.get("securitiesAccount", {}) or {}).get("currentBalances", {}) or {}
        return BrokerSnapshot(
            account_number=self.account_number,
            status=str((acct.get("securitiesAccount", {}) or {}).get("type", "")),
            cash=_to_float(bal.get("availableFunds")) or 0.0,
            buying_power=_to_float(bal.get("buyingPower")) or (_to_float(bal.get("availableFunds")) or 0.0),
            equity=_to_float(bal.get("liquidationValue")) or 0.0,
            portfolio_value=_to_float(bal.get("liquidationValue")) or 0.0,
        )
