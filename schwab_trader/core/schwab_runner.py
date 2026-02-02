# core/schwab_runner.py
"""
SchwabLiveRunner - Live trading runner for Schwab broker integration.

Mirrors the AlpacaLiveRunner pattern:
- Bar callback handler with strategy evaluation
- Historical data seeding
- P&L tracking and event emission
- Integration with execution engine and trade logic manager
"""
from __future__ import annotations
from pathlib import Path
import sys
import os
import json
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]  # .../schwab_trader
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asyncio
import contextlib
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import pandas as pd

from utils.settings import Settings
from loggers.factory import get_module_logger
from loggers.file_trade_logger import FileTradeLogger

from core.logic.portfolio_state import PortfolioState
from core.logic.symbol_state import SymbolState
from core.logic.trade_gate import TradeGate
from core.logic.strategy_routing_manager import StrategyRoutingManager
from core.position_sizer import DynamicPositionSizer
from core.drawdown_monitor import DrawdownMonitor
from core.historical_loader import HistoricalBarLoader
from core.events.eventhandler import EventHandler, get_event_handler
from core.events.events import (
    EVENT_NEW_BAR, BarPayload,
    EVENT_STRATEGY_SIGNAL, StrategySignalPayload,
    EVENT_PNL_UPDATE, PnLPayload,
    EVENT_HEALTH_UPDATE
)

from core.simulator.simulation import compute_atr, classify_regime
from core.broker.schwab_broker import SchwabBroker
from core.logic.live_execution_engine import LiveExecutionEngine
from core.logic.trade_logic_manager import DynamicTradeLogicManager
from core.executor import LiveExecutor
from core.state_reconciler import StateReconciler, ReconcilerConfig
from data.streaming.schwab_client import SchwabClient

load_dotenv(ROOT / ".venv" / ".env")


class SchwabLiveRunner:
    """
    Live trading runner for Schwab broker.

    Provides:
    - Real-time bar processing from Schwab streaming
    - Strategy signal generation and routing
    - Position and P&L tracking
    - Event emission for GUI integration
    - Drawdown monitoring and trade gates
    """

    def __init__(self, settings: Settings, symbols: List[str], client: Optional[SchwabClient] = None):
        """
        Initialize the Schwab live runner.

        Args:
            settings: Application settings containing credentials and config
            symbols: List of symbols to trade
            client: Optional pre-configured SchwabClient instance
        """
        self.settings = settings
        self.symbols = symbols
        self.event_handler = get_event_handler()

        # Logging + trade log
        self.logger = get_module_logger(module_name='SchwabLiveRunner', file_key='SchwabLive')
        self.trade_logger = FileTradeLogger(log_file='schwab_live_trades.csv', logger_name='SchwabTradeLogger')

        # Initialize Schwab client if not provided
        if client is None:
            api_key = settings.get("SCHWAB_API_KEY") or os.getenv("SCHWAB_API_KEY")
            secret_key = settings.get("SCHWAB_SECRET") or os.getenv("SCHWAB_SECRET")
            if not api_key or not secret_key:
                raise ValueError("SCHWAB_API_KEY and SCHWAB_SECRET must be set in settings or environment")
            client = SchwabClient(apikey=api_key, secretkey=secret_key)

        self._client = client

        # Broker (Schwab)
        self.broker = SchwabBroker(
            client=client,
            session=settings.get("SCHWAB_SESSION", "NORMAL"),
        )

        # State tracking
        self.portfolio = PortfolioState()
        self.symbol_state: Dict[str, SymbolState] = defaultdict(SymbolState)
        self.history: Dict[str, List[Dict]] = defaultdict(list)
        self.atr_hist: Dict[str, List[float]] = defaultdict(list)

        # Risk & gates
        self.trade_gate = TradeGate(
            max_layers=settings.get("MAX_PYRAMID_LAYERS", 2),
            min_bars_between_layers=settings.get("MIN_BARS_BETWEEN_LAYERS", 2),
            regime_min_persist_bars=settings.get("REGIME_MIN_PERSIST_BARS", 1),
            flip_cooldown_bars=settings.get("FLIP_COOLDOWN_BARS", 1),
        )
        self.ddm = DrawdownMonitor(
            max_symbol_drawdown=settings.get("MAX_SYMBOL_DD", 0.12),
            max_portfolio_drawdown=settings.get("MAX_PORTFOLIO_DD", 0.15),
            symbol_cooldown_seconds=settings.get("DDM_COOLDOWN_BARS", 5),
        )

        # Sizer, router, executor, engine
        self.sizer = DynamicPositionSizer(
            risk_percentage=settings.get("BASE_RISK_PCT", 0.05)
        )
        self.router = StrategyRoutingManager(str(ROOT.parent / "config" / "strategy_routing.json"))
        self.executor = LiveExecutor(
            broker=self.broker,
            event_handler=self.event_handler,
        )
        self.engine = LiveExecutionEngine(
            broker=self.broker,
            executor=self.executor,
            sizer=self.sizer,
            performance_tracker=self.trade_logger,
            trade_logic_manager=DynamicTradeLogicManager(str(ROOT.parent / "config" / "trade_logic_routing.json")),
            portfolio=self.portfolio,
        )

        # Wire optional attrs
        if hasattr(self.engine, "trade_gate"):
            self.engine.trade_gate = self.trade_gate
        if hasattr(self.engine, "drawdown_monitor"):
            self.engine.drawdown_monitor = self.ddm

        # State reconciler - ensures local state matches broker
        reconciler_config = ReconcilerConfig(
            reconcile_interval=settings.get("RECONCILE_INTERVAL", 60),
            halt_on_critical=settings.get("HALT_ON_MISMATCH", True),
            auto_correct_minor=settings.get("AUTO_CORRECT_MINOR", True),
        )
        self.reconciler = StateReconciler(
            broker=self.broker,
            portfolio=self.portfolio,
            config=reconciler_config,
            on_halt=self._on_reconciler_halt,
        )

        # Bar tracking
        self._last_bar_id: Dict[str, int] = {}
        self._last_ddm_date: Optional[datetime] = None
        self._running = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 5  # seconds

        # Quote callback registry
        self._quote_callbacks: Dict[str, Any] = {}

    # ---------- Reconciler Callback ----------
    def _on_reconciler_halt(self, message: str):
        """Called when reconciler detects critical state mismatch."""
        self.logger.critical(f"TRADING HALTED: {message}")
        self.logger.critical("Manual intervention required. Review positions at broker.")
        self._running = False  # Stop the run loop

    # ---------- Helpers ----------
    @staticmethod
    def _canonicalize_schwab_quote(quote: Dict, symbol: str) -> Dict:
        """
        Normalize Schwab quote format to canonical bar format.

        Schwab streaming quotes have different field names than Alpaca bars.
        """
        ts = datetime.now(timezone.utc)

        # Schwab streaming fields:
        # key=symbol, 1=last_price, 2=bid, 3=ask, 29=close
        last_price = quote.get('last_price') or quote.get('1') or quote.get('lastPrice', 0)
        bid_price = quote.get('bid_price') or quote.get('2') or quote.get('bidPrice', 0)
        ask_price = quote.get('ask_price') or quote.get('3') or quote.get('askPrice', 0)
        close_price = quote.get('close_price') or quote.get('29') or quote.get('closePrice', last_price)
        volume = quote.get('volume') or quote.get('v', 0)

        # Create a synthetic bar from quote data
        mid = (float(bid_price or 0) + float(ask_price or 0)) / 2 if bid_price and ask_price else float(last_price or 0)
        price = float(last_price) if last_price else mid

        return {
            "symbol": symbol,
            "timestamp": ts,
            "Open": price,
            "High": price,
            "Low": price,
            "Close": price,
            "Volume": int(volume) if volume else 0,
        }

    @staticmethod
    def _bar_bucket(ts: datetime, timeframe_sec: int = 60) -> int:
        """Calculate bar bucket ID for deduplication."""
        return int(ts.timestamp() // timeframe_sec)

    def _df_from_history(self, symbol: str) -> pd.DataFrame:
        """Create DataFrame from recent history."""
        rows = self.history[symbol][-300:]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _symbol_mv(self, symbol: str, last_px: float) -> float:
        """Calculate market value for a symbol position."""
        pos = self.portfolio.positions.get(symbol)
        return (pos.qty * last_px) if pos else 0.0

    async def _bar_debug_logger(self, payload: Dict):
        """Debug logger for bar events."""
        try:
            self.logger.debug(
                f"[EVENT BAR] {payload['symbol']} {payload['timestamp']} "
                f"o={payload['open']} h={payload['high']} l={payload['low']} "
                f"c={payload['close']} v={payload['volume']}"
            )
        except Exception:
            pass

    # ---------- Seeding ----------
    async def seed(self, lookback_bars: int = 200):
        """
        Seed historical data for warmup indicators.

        Args:
            lookback_bars: Number of historical bars to load per symbol
        """
        loader = HistoricalBarLoader(str(ROOT / "data" / "data_storage" / "proc_data"))
        for sym in self.symbols:
            try:
                for b in loader.load_last_n_bars(sym, n=lookback_bars):
                    self.history[sym].append({
                        "timestamp": b["timestamp"],
                        "symbol": b["symbol"],
                        "Open": b["Open"], "High": b["High"], "Low": b["Low"],
                        "Close": b["Close"], "Volume": b.get("Volume", 0),
                    })
                df = self._df_from_history(sym)
                atr = compute_atr(df, period=14)
                if atr is not None:
                    self.atr_hist[sym].append(atr)
                self.logger.info(f"Seeded {sym} with {len(self.history[sym])} bars")
            except Exception as e:
                self.logger.warning(f"Failed to seed {sym}: {e}")
        self.logger.info(f"Seeded {len(self.symbols)} symbols with up to {lookback_bars} bars.")

    # ---------- Quote/Bar Callback ----------
    async def on_schwab_quote(self, symbol: str, quote: Dict):
        """
        Handle incoming Schwab quote data.

        Converts quotes to bar format and processes through the strategy engine.

        Args:
            symbol: The symbol for this quote
            quote: Raw quote data from Schwab streaming
        """
        bar = self._canonicalize_schwab_quote(quote, symbol)
        self.logger.debug(f"[RAW QUOTE] {symbol} c={bar['Close']}")

        ts: datetime = bar["timestamp"]

        # Emit raw bar event
        await self.event_handler.emit("BAR", {
            "timestamp": ts,
            "symbol": symbol,
            "open": float(bar["Open"]),
            "high": float(bar["High"]),
            "low": float(bar["Low"]),
            "close": float(bar["Close"]),
            "volume": int(bar.get("Volume", 0)),
        })

        bar_id = self._bar_bucket(ts)
        prev_bar_id = self._last_bar_id.get(symbol)
        bar_closed = (prev_bar_id is None) or (bar_id != prev_bar_id)
        self._last_bar_id[symbol] = bar_id

        # Track history + MTM
        self.history[symbol].append(bar)
        last_px = float(bar["Close"])
        self.portfolio.update_price(symbol, last_px)

        # Update symbol state
        state = self.symbol_state.setdefault(symbol, SymbolState(symbol=symbol))
        state.portfolio_value = self.portfolio.total_equity()
        state.ts = ts
        state.bar_id = bar_id
        state.bar_closed = bar_closed

        # Indicators & regime
        df = self._df_from_history(symbol)
        atr = compute_atr(df, period=14)
        if atr is not None:
            self.atr_hist[symbol].append(atr)
        regime = classify_regime(atr, self.atr_hist[symbol])

        # Drawdown monitor (daily tick + updates)
        if self._last_ddm_date is None or ts.date() != self._last_ddm_date:
            self.ddm.start_new_day(portfolio_equity=self.portfolio.total_equity())
            self._last_ddm_date = ts.date()
        self.ddm.update_portfolio(self.portfolio.total_equity())
        self.ddm.update_symbol(symbol, self._symbol_mv(symbol, last_px))

        # Emit P&L update
        await self.event_handler.emit(EVENT_PNL_UPDATE, PnLPayload(
            portfolio_value=self.portfolio.total_equity(),
            equity_curve=self.portfolio.equity_history,
            unrealized=self.portfolio.total_unrealized(),
            realized=self.portfolio.realized_pnl,
            drawdown=self.ddm.get_portfolio_drawdown(),
            timestamp=ts.isoformat(),
        ))

        # Strategy & signal
        strategy = self.router.get_strategy(symbol, regime)
        strategy_name = type(strategy).__name__
        try:
            raw_signal = strategy.generate_signal(df)
            signal = int(raw_signal if isinstance(raw_signal, (int, float)) else getattr(raw_signal, "signal", 0))
            await self.event_handler.emit(EVENT_STRATEGY_SIGNAL, StrategySignalPayload(
                symbol=symbol,
                strategy=strategy_name,
                signal={-1: "sell", 0: "hold", 1: "buy"}.get(signal, "hold"),
                confidence=None,
                timestamp=ts.isoformat(),
            ))
        except Exception as e:
            self.logger.exception(f"[{symbol}] Strategy error in {strategy_name}: {e}")
            signal = 0

        # Trade gate context
        self.trade_gate.on_new_bar(symbol, bar_id, regime)
        gs = self.trade_gate.get_state(symbol)
        state.regime = regime
        state.regime_persist = gs.regime_persist

        # Hand off to execution engine
        await self.engine.handle_signal(
            symbol=symbol,
            state=state,
            signal=signal,
            price=last_px,
            atr=float(atr or 0.0),
            regime=regime,
            strategy_name=strategy_name,
        )

        # Telemetry
        pos = self.portfolio.positions.get(symbol)
        qty = pos.qty if pos else 0
        self.logger.debug(
            f"[{symbol}] bar={bar_id} closed={bar_closed} regime={regime} "
            f"persist={gs.regime_persist} qty={qty} equity={self.portfolio.total_equity():.2f}"
        )

        # Emit processed bar event
        await self.event_handler.emit(EVENT_NEW_BAR, BarPayload(
            symbol=symbol,
            open=float(bar["Open"]),
            high=float(bar["High"]),
            low=float(bar["Low"]),
            close=float(bar["Close"]),
            volume=int(bar.get("Volume", 0)),
            timestamp=ts.isoformat(),
        ))

    def _create_quote_callback(self, symbol: str):
        """Create a quote callback bound to a specific symbol."""
        async def callback(quote: Dict):
            await self.on_schwab_quote(symbol, quote)
        return callback

    # ---------- Health Check ----------
    async def _emit_health_status(self, status: str, details: Dict = None):
        """Emit health status event."""
        await self.event_handler.emit(EVENT_HEALTH_UPDATE, {
            "broker": "schwab",
            "status": status,
            "details": details or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    # ---------- Connection Management ----------
    async def connect(self) -> bool:
        """
        Establish connection to Schwab streaming.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            api_key = self.settings.get("SCHWAB_API_KEY") or os.getenv("SCHWAB_API_KEY")
            secret_key = self.settings.get("SCHWAB_SECRET") or os.getenv("SCHWAB_SECRET")

            if not api_key or not secret_key:
                self.logger.error("Missing Schwab credentials")
                return False

            # Initialize streaming connection
            self.broker.connect_stream(api_key, secret_key)

            # Subscribe to quotes for each symbol
            for sym in self.symbols:
                callback = self._create_quote_callback(sym)
                self._quote_callbacks[sym] = callback
                self.broker.subscribe_quotes(callback, sym)

            self._reconnect_attempts = 0
            await self._emit_health_status("connected", {"symbols": self.symbols})
            self.logger.info(f"Connected to Schwab streaming for: {', '.join(self.symbols)}")
            return True

        except Exception as e:
            self.logger.exception(f"Failed to connect to Schwab: {e}")
            await self._emit_health_status("error", {"error": str(e)})
            return False

    async def _reconnect(self):
        """Attempt to reconnect to Schwab streaming."""
        while self._running and self._reconnect_attempts < self._max_reconnect_attempts:
            self._reconnect_attempts += 1
            self.logger.warning(
                f"Reconnection attempt {self._reconnect_attempts}/{self._max_reconnect_attempts}"
            )
            await self._emit_health_status("reconnecting", {
                "attempt": self._reconnect_attempts,
                "max_attempts": self._max_reconnect_attempts
            })

            await asyncio.sleep(self._reconnect_delay * self._reconnect_attempts)

            if await self.connect():
                return True

        self.logger.error("Max reconnection attempts reached")
        await self._emit_health_status("disconnected", {"reason": "max_reconnect_attempts"})
        return False

    # ---------- Run ----------
    async def run(self):
        """
        Start the live trading runner.

        Connects to Schwab streaming and processes quotes until stopped.
        """
        self._running = True

        # Seed historical data
        await self.seed(self.settings.get("SEED_BARS", 200))

        # Subscribe to bar debug logging
        await self.event_handler.subscribe("BAR", self._bar_debug_logger)

        # Connect to Schwab
        if not await self.connect():
            self.logger.error("Failed to establish initial connection")
            return

        # CRITICAL: Sync state with broker before trading
        self.logger.info("Syncing portfolio state with broker...")
        sync_result = await self.reconciler.full_sync()
        if not sync_result.success:
            self.logger.error("Failed to sync with broker - check credentials and connectivity")
        else:
            self.logger.info(
                f"Portfolio synced: {sync_result.broker_positions} positions, "
                f"${sync_result.broker_cash:,.2f} cash"
            )

        # Start streaming
        stream_task = asyncio.create_task(self.broker.start_stream())
        self.logger.info(f"SchwabLiveRunner started for: {', '.join(self.symbols)}")

        # Start periodic state reconciliation
        reconcile_interval = self.settings.get("RECONCILE_INTERVAL", 60)
        if reconcile_interval > 0:
            await self.reconciler.start_periodic(interval_seconds=reconcile_interval)
            self.logger.info(f"State reconciliation enabled (every {reconcile_interval}s)")

        try:
            while self._running:
                # Check if stream is still running
                if stream_task.done():
                    exc = stream_task.exception()
                    if exc:
                        self.logger.error(f"Stream task failed: {exc}")

                    # Attempt reconnection
                    if self._running and await self._reconnect():
                        stream_task = asyncio.create_task(self.broker.start_stream())
                    else:
                        break

                await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            self.logger.info("Runner cancelled")
        finally:
            self._running = False

            # Stop reconciler
            await self.reconciler.stop_periodic()
            self.logger.info("State reconciler stopped")

            stream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stream_task
            self.trade_logger.flush()
            await self._emit_health_status("disconnected", {"reason": "shutdown"})
            self.logger.info("SchwabLiveRunner shut down cleanly.")

    async def stop(self):
        """Stop the live trading runner."""
        self._running = False
        self.logger.info("Stop requested")


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
    """Main entry point for running SchwabLiveRunner standalone."""
    # Ensure config files exist
    sr_path, sp_path, tl_path = _ensure_live_config("config")

    # Load settings
    settings = Settings(
        root="config",
        include_root=True,
        runtime_overrides={
            "strategy_routing_path": sr_path,
            "strategy_params_path": sp_path,
            "trade_logic_routing_path": tl_path,
        },
    )

    # Get symbols from settings or default
    symbols = settings.get_list("symbols") or settings.get_list("SYMBOLS") or ["AAPL", "MSFT"]

    runner = SchwabLiveRunner(settings, symbols)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
