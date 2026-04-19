# core/alpaca_runner.py
"""
AlpacaLiveRunner - Live trading runner for Alpaca broker integration.

Extends BaseLiveRunner with Alpaca-specific:
- AlpacaBroker creation
- Bar canonicalization for Alpaca streaming format
- Connection and streaming setup
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv

from core.base.base_live_runner import BaseLiveRunner
from core.broker.alpaca_broker import AlpacaBroker
from core.config_loader import get_config, TradingConfig
from core.credential_validator import CredentialValidator

ROOT = Path(__file__).resolve().parents[1]  # .../schwab_trader
load_dotenv(ROOT / ".venv" / ".env")


class AlpacaLiveRunner(BaseLiveRunner):
    """
    Live trading runner for Alpaca broker.

    Implements Alpaca-specific bar handling and streaming setup.
    Inherits all common functionality from BaseLiveRunner.
    """

    BROKER_NAME = "Alpaca"
    LOG_FILE_KEY = "AlpacaLive"
    TRADE_LOG_FILE = "live_trades.csv"

    def __init__(self, symbols: list[str], config: Optional[TradingConfig] = None):
        """
        Initialize the Alpaca live runner.

        Args:
            symbols: List of symbols to trade
            config: Optional TradingConfig instance (uses global config if not provided)
        """
        # Store config before calling super().__init__ which needs it
        self._init_config = config or get_config()

        # Credential validator for preflight
        self.credential_validator = CredentialValidator()

        # Call parent constructor (creates broker, engine, etc.)
        super().__init__(symbols, config)

    # ==========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ==========================================================================

    def _create_broker(self) -> AlpacaBroker:
        """Create and configure the Alpaca broker instance."""
        config = getattr(self, '_init_config', None) or self.config
        return AlpacaBroker(
            api_key=os.getenv("ALPACA_API_KEY"),
            api_secret=os.getenv("ALPACA_SECRET_KEY"),
            paper=config.alpaca.paper,
            poll_timeout=getattr(config.alpaca, 'poll_timeout_seconds', 30),
        )

    def _canonicalize_bar(self, raw_bar: Any) -> Dict:
        """
        Convert Alpaca bar to canonical format.

        Args:
            raw_bar: Raw bar from Alpaca streaming

        Returns:
            Canonical bar dict
        """
        ts = raw_bar.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return {
            "symbol": getattr(raw_bar, "symbol", getattr(raw_bar, "S", None)),
            "timestamp": ts,
            "Open": getattr(raw_bar, "open", getattr(raw_bar, "o", None)),
            "High": getattr(raw_bar, "high", getattr(raw_bar, "h", None)),
            "Low": getattr(raw_bar, "low", getattr(raw_bar, "l", None)),
            "Close": getattr(raw_bar, "close", getattr(raw_bar, "c", None)),
            "Volume": getattr(raw_bar, "volume", getattr(raw_bar, "v", 0)) or 0,
        }

    async def _connect_broker(self) -> None:
        """Establish connection to Alpaca."""
        self.broker.api_key = os.getenv("ALPACA_API_KEY")
        self.broker.api_secret = os.getenv("ALPACA_SECRET_KEY")
        self.broker.connect_sync()

    async def _start_streaming(self) -> asyncio.Task:
        """Start the Alpaca data stream."""
        return asyncio.create_task(self.broker.start_stream())

    async def _disconnect_broker(self) -> None:
        """Disconnect from Alpaca."""
        self.broker.disconnect()

    def _subscribe_to_data(self) -> None:
        """Subscribe to bar data for all symbols."""
        for sym in self.symbols:
            self.broker.subscribe_bars(self._on_bar, sym)

    # ==========================================================================
    # BAR CALLBACK
    # ==========================================================================

    async def _on_bar(self, raw_bar: Any) -> None:
        """
        Handle incoming Alpaca bar data.

        Args:
            raw_bar: Raw bar from Alpaca streaming
        """
        bar = self._canonicalize_bar(raw_bar)
        self.logger.debug(f"[RAW BAR] {bar['symbol']} {bar['timestamp']} c={bar['Close']}")
        await self._process_bar(bar)


# -------- Entrypoint --------
def _ensure_live_config(dir_path: str = "config"):
    """Ensure config files exist with defaults."""
    os.makedirs(dir_path, exist_ok=True)

    sr_path = os.path.join(dir_path, "strategy_routing.json")
    if not os.path.exists(sr_path):
        with open(sr_path, "w") as f:
            json.dump({
                "AAPL": {
                    "low_volatility": "sma_strategy",
                    "normal": "momentum_strategy",
                    "high_volatility": "mean_reversion_strategy"
                },
                "MSFT": {
                    "low_volatility": "sma_strategy",
                    "normal": "momentum_strategy",
                    "high_volatility": "mean_reversion_strategy"
                }
            }, f, indent=2)

    sp_path = os.path.join(dir_path, "strategy_params.json")
    if not os.path.exists(sp_path):
        with open(sp_path, "w") as f:
            json.dump({
                "AAPL": {
                    "low_volatility": {"params": {"fast": 10, "slow": 30}},
                    "normal": {"params": {"lookback": 20}},
                    "high_volatility": {"params": {"window": 14}}
                },
                "MSFT": {
                    "low_volatility": {"params": {"fast": 10, "slow": 30}},
                    "normal": {"params": {"lookback": 20}},
                    "high_volatility": {"params": {"window": 14}}
                }
            }, f, indent=2)

    tl_path = os.path.join(dir_path, "trade_logic_routing.json")
    if not os.path.exists(tl_path):
        with open(tl_path, "w") as f:
            json.dump({
                "AAPL": {
                    "low_volatility": {"trade_logic_class": "default", "params": {}},
                    "normal": {"trade_logic_class": "default", "params": {}},
                    "high_volatility": {"trade_logic_class": "default", "params": {}}
                },
                "MSFT": {
                    "low_volatility": {"trade_logic_class": "default", "params": {}},
                    "normal": {"trade_logic_class": "default", "params": {}},
                    "high_volatility": {"trade_logic_class": "default", "params": {}}
                }
            }, f, indent=2)

    return sr_path, sp_path, tl_path


async def main():
    """Main entry point for running AlpacaLiveRunner standalone."""
    _ensure_live_config("config")

    config = get_config()
    symbols = config.general.default_symbols or ["AAPL", "MSFT"]

    runner = AlpacaLiveRunner(symbols=symbols, config=config)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
