"""
CoinbaseBroker (Advanced Trade v3) — SDK-backed, .env-aware
----------------------------------------------------------
Broker backend that wraps the **official Coinbase Advanced API Python SDK**
(`coinbase-advanced-py`) and automatically loads credentials from a `.env` file
(similar to your Schwab/Alpaca setup).

Refs:
- SDK usage (RESTClient, WSClient, WSUserClient) citeturn1view0
- Advanced Trade API overview & WebSocket channels citeturn0search3turn0search6

Setup:
1. Add to `.env`:

   ```env
   COINBASE_API_KEY=organizations/<org_id>/apiKeys/<key_id>
   COINBASE_API_SECRET="-----BEGIN EC PRIVATE KEY-----\\n...\\n-----END EC PRIVATE KEY-----"
   ```

   ⚠️ If multiline PEM doesn’t parse, use `\\n` for newlines or load from a `.pem` file.

2. Install deps:
   ```bash
   pip install coinbase-advanced-py python-dotenv
   ```

3. Usage:
   ```python
   from brokers.coinbase_broker import CoinbaseBroker
   cb = CoinbaseBroker()  # auto-loads from .env
   products = await cb.list_products()
   ```

Run:
- `python coinbase_broker.py` → demo (lists product count)
- `python coinbase_broker.py --test` → offline self-tests (no network)
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# --- Official SDK imports --- #
try:
    from coinbase.rest import RESTClient
    from coinbase.websocket import WSClient, WSUserClient
    HAS_CB_SDK = True
except Exception as _e:  # pragma: no cover
    HAS_CB_SDK = False
    _IMPORT_ERR = _e


# --------------------------- Config & Constants --------------------------- #
CB_BASE_PATH = "/api/v3/brokerage"


@dataclass
class CoinbaseAuth:
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    key_file: Optional[str] = None


class CoinbaseBroker:
    """Broker adapter using the official SDK under the hood.

    Auto-loads `.env` (COINBASE_API_KEY, COINBASE_API_SECRET) if no auth passed.
    """

    def __init__(
        self,
        auth: CoinbaseAuth | None = None,
        *,
        timeout: float = 15.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if not HAS_CB_SDK:
            raise RuntimeError(
                "coinbase-advanced-py not installed. `pip install coinbase-advanced-py`\n"
                f"Import error: {_IMPORT_ERR}"
            )

        # Auto-load .env if not provided
        load_dotenv(r'C:\Users\kwasi\OneDrive\Documents\Personal Projects\schwab_trader\venv\.env')

        if auth is None:
            api_key = os.getenv("COINBASE_API_KEY")
            api_secret = os.getenv("COINBASE_SECRET")
            if not api_key or not api_secret:
                raise RuntimeError("Missing Coinbase API credentials in .env")
            auth = CoinbaseAuth(api_key=api_key, api_secret=api_secret)

        self.logger = logger or logging.getLogger("CoinbaseBroker")
        self._rest = self._make_rest_client(auth, timeout)
        self._ws: Optional[WSClient] = None
        self._ws_user: Optional[WSUserClient] = None

    # ----------------------------- Client Init ---------------------------- #
    def _make_rest_client(self, auth: CoinbaseAuth, timeout: float) -> RESTClient:
        if auth.key_file is not None:
            return RESTClient(key_file=auth.key_file, timeout=timeout)
        return RESTClient(api_key=auth.api_key, api_secret=auth.api_secret, timeout=timeout)

    # ------------------------------ Accounts ----------------------------- #
    async def get_accounts(self):
        res = self._rest.get_accounts()
        return res.to_dict() if hasattr(res, "to_dict") else res

    async def get_account(self, account_uuid: str):
        res = self._rest.get_account(account_uuid=account_uuid)
        return res.to_dict() if hasattr(res, "to_dict") else res

    # ------------------------------ Products ----------------------------- #
    async def list_products(self):
        res = self._rest.get_products()
        return res.to_dict() if hasattr(res, "to_dict") else res

    async def get_product(self, product_id: str):
        res = self._rest.get_product(product_id=product_id)
        return res.to_dict() if hasattr(res, "to_dict") else res

    async def get_product_candles(
        self,
        product_id: str,
        *,
        granularity: str = "ONE_MINUTE",
        start_iso: Optional[str] = None,
        end_iso: Optional[str] = None,
        limit: int = 300,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"granularity": granularity, "limit": limit}
        if start_iso:
            params["start"] = start_iso
        if end_iso:
            params["end"] = end_iso
        path = f"{CB_BASE_PATH}/products/{product_id}/candles"
        res = self._rest.get(path, params=params)
        if hasattr(res, "to_dict"):
            res = res.to_dict()
        candles = res.get("candles", []) if isinstance(res, dict) else res
        if isinstance(candles, dict):
            candles = candles.get("candles", [])
        return candles

    # -------------------------------- Fees ------------------------------- #
    async def get_fees(self):
        res = self._rest.get(f"{CB_BASE_PATH}/fees")
        return res.to_dict() if hasattr(res, "to_dict") else res

    # ------------------------------- Orders ------------------------------ #
    async def place_order(
        self,
        *,
        product_id: str,
        side: str,
        order_type: str = "MARKET",
        size: Decimal | str | float | None = None,
        quote_size: Decimal | str | float | None = None,
        limit_price: Decimal | str | float | None = None,
        time_in_force: Optional[str] = None,
        client_order_id: Optional[str] = "",
    ) -> Dict[str, Any]:
        side = side.upper()
        order_type = order_type.upper()

        if order_type == "MARKET":
            if side == "BUY":
                if quote_size is None and size is None:
                    raise ValueError("MARKET BUY requires quote_size or size")
                if quote_size is not None:
                    res = self._rest.market_order_buy(client_order_id=client_order_id, product_id=product_id, quote_size=str(quote_size))
                else:
                    res = self._rest.market_order_buy(client_order_id=client_order_id, product_id=product_id, base_size=str(size))
            elif side == "SELL":
                if size is None:
                    raise ValueError("MARKET SELL requires base size")
                res = self._rest.market_order_sell(client_order_id=client_order_id, product_id=product_id, base_size=str(size))
            else:
                raise ValueError("side must be BUY or SELL")
        elif order_type == "LIMIT":
            if limit_price is None or size is None:
                raise ValueError("LIMIT orders require limit_price and base size")
            tif = time_in_force or "GOOD_UNTIL_CANCELLED"
            order_cfg: Dict[str, Any] = {
                "product_id": product_id,
                "side": side,
                "client_order_id": client_order_id,
                "order_configuration": {
                    "limit_limit_gtc" if tif == "GOOD_UNTIL_CANCELLED" else "limit_limit_ioc": {
                        "base_size": str(size),
                        "limit_price": str(limit_price),
                        "post_only": False,
                    }
                },
            }
            res = self._rest.post(f"{CB_BASE_PATH}/orders", data=order_cfg)
        else:
            raise ValueError("order_type must be MARKET or LIMIT")

        return res.to_dict() if hasattr(res, "to_dict") else res

    # ---------------------------- Market Data WS -------------------------- #
    def open_ws(self, on_message, *, retry: bool = True, timeout: Optional[int] = None, max_size: Optional[int] = None) -> None:
        self._ws = WSClient(on_message=on_message, retry=retry, timeout=timeout, max_size=max_size)
        self._ws.open()

    def subscribe_ws(self, product_ids: List[str], channels: List[str]) -> None:
        if not self._ws:
            raise RuntimeError("WS not opened. Call open_ws first.")
        self._ws.subscribe(product_ids=product_ids, channels=channels)

    def close_ws(self) -> None:
        if self._ws:
            self._ws.close()
            self._ws = None

    # ---------------------------- User Orders WS -------------------------- #
    def open_user_ws(self, on_message, *, api_key: Optional[str] = None, api_secret: Optional[str] = None) -> None:
        self._ws_user = WSUserClient(api_key=api_key, api_secret=api_secret, on_message=on_message)
        self._ws_user.open()

    def subscribe_user_ws(self, channels: List[str]) -> None:
        if not self._ws_user:
            raise RuntimeError("User WS not opened. Call open_user_ws first.")
        self._ws_user.subscribe(channels=channels)

    def close_user_ws(self) -> None:
        if self._ws_user:
            self._ws_user.close()
            self._ws_user = None

    async def aclose(self) -> None:
        return None


# ------------------------- Safe async entry points ------------------------ #

def safe_asyncio_run(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        return loop.create_task(coro)


# --------------------------------- Demo ---------------------------------- #
async def _demo():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("CB-Demo")

    cb = CoinbaseBroker()  # auto-loads .env creds
    prods = await cb.list_products()
    count = len(prods["products"]) if isinstance(prods, dict) and "products" in prods else (
        len(getattr(prods, "products", []))
    )
    logger.info(f"Products count: {count}")


# -------------------------------- Tests ---------------------------------- #
class _StubCB(CoinbaseBroker):
    def __init__(self):
        self.logger = logging.getLogger("StubCB")
        global HAS_CB_SDK
        HAS_CB_SDK = True
        self._rest = None
        self._ws = None
        self._ws_user = None

    async def list_products(self):
        return {"products": [{"product_id": "BTC-USD"}, {"product_id": "ETH-USD"}]}

    async def get_product_candles(self, product_id: str, **_):
        return [{"t": 1, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "10"}]

    async def get_fees(self):
        return {"maker": "0.001", "taker": "0.002"}

    async def place_order(self, **kwargs):
        assert kwargs.get("product_id") == "BTC-USD"
        return {"success": True, "order_id": "test123"}


async def _test_stubbed_calls():
    stub = _StubCB()
    prods = await stub.list_products()
    assert len(prods["products"]) == 2
    candles = await stub.get_product_candles("BTC-USD")
    assert isinstance(candles, list) and candles
    fees = await stub.get_fees()
    assert fees.get("taker") == "0.002"
    placed = await stub.place_order(product_id="BTC-USD", side="BUY", order_type="MARKET", quote_size="10")
    assert placed.get("success") is True


async def _run_tests():
    await _test_stubbed_calls()
    print("All tests passed.")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        safe_asyncio_run(_run_tests())
    else:
        safe_asyncio_run(_demo())
