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

from loggers.factory import get_module_logger
from loggers.file_trade_logger import FileTradeLogger

from core.logic.portfolio_state import PortfolioState
from core.logic.symbol_state import SymbolState
from core.logic.trade_gate import TradeGate
from core.logic.strategy_routing_manager import StrategyRoutingManager
from core.position_sizer import KellyPositionSizer
from core.drawdown_monitor import DrawdownMonitor
from core.historical_loader import HistoricalBarLoader
from core.historical_data_updater import HistoricalDataUpdater
from core.unified_data_pipeline import UnifiedDataPipeline
from core.events.eventhandler import EventHandler, get_event_handler
from core.contracts.events import (
    EVENT_NEW_BAR, BarPayload,
    EVENT_STRATEGY_SIGNAL, StrategySignalPayload,
    EVENT_PNL_UPDATE, PnLPayload,
    EVENT_HEALTH_UPDATE,
    EVENT_POSITION_UPDATE, PositionPayload
)
from core.config_loader import get_config, create_position_sizer, create_drawdown_monitor, TradingConfig

from core.simulator.simulation import compute_atr, classify_regime
from core.credential_validator import CredentialValidator, CredentialStatus
from core.broker.schwab_broker import SchwabBroker
from core.logic.live_execution_engine import LiveExecutionEngine
from core.logic.trade_logic_manager import DynamicTradeLogicManager
from core.app_types import SignalContext
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

    def __init__(self, symbols: List[str], client: Optional[SchwabClient] = None, config: Optional[TradingConfig] = None):
        """
        Initialize the Schwab live runner.

        Args:
            symbols: List of symbols to trade
            client: Optional pre-configured SchwabClient instance
            config: Optional TradingConfig instance (uses global config if not provided)
        """
        self.symbols = symbols
        self.event_handler = get_event_handler()

        # Load centralized config
        self.config = config or get_config()

        # Logging + trade log
        self.logger = get_module_logger(module_name='SchwabLiveRunner', file_key='SchwabLive')
        self.trade_logger = FileTradeLogger(log_file='schwab_live_trades.csv', logger_name='SchwabTradeLogger')

        # Initialize Schwab client if not provided
        if client is None:
            api_key = os.getenv("SCHWAB_API_KEY")
            secret_key = os.getenv("SCHWAB_SECRET")
            if not api_key or not secret_key:
                raise ValueError("SCHWAB_API_KEY and SCHWAB_SECRET must be set in environment")
            client = SchwabClient(apikey=api_key, secretkey=secret_key)

        self._client = client

        # Broker (Schwab)
        self.broker = SchwabBroker(
            client=client,
            session=self.config.schwab.session,
        )

        # State tracking
        self.portfolio = PortfolioState()
        self.symbol_state: Dict[str, SymbolState] = defaultdict(SymbolState)
        self.history: Dict[str, List[Dict]] = defaultdict(list)
        self.atr_hist: Dict[str, List[float]] = defaultdict(list)

        risk_cfg = self.config.risk

        # Risk & gates (uses config values)
        self.trade_gate = TradeGate(
            max_layers=risk_cfg.max_pyramid_layers,
            min_bars_between_layers=risk_cfg.min_bars_between_layers,
            regime_min_persist_bars=1,
            flip_cooldown_bars=1,
        )

        # DrawdownMonitor from config (returns None if disabled)
        self.ddm = create_drawdown_monitor(self.config)

        # Sizer from config
        self.sizer = create_position_sizer(self.config)
        self.router = StrategyRoutingManager(str(ROOT / "config" / "strategy_routing.json"))
        self.executor = LiveExecutor(
            broker=self.broker,
            event_handler=self.event_handler,
        )
        self.engine = LiveExecutionEngine(
            broker=self.broker,
            executor=self.executor,
            sizer=self.sizer,
            performance_tracker=self.trade_logger,
            trade_logic_manager=DynamicTradeLogicManager(str(ROOT / "config" / "trade_logic_routing.json")),
            portfolio=self.portfolio,
            event_handler=self.event_handler,  # For GUI visibility of trade decisions
            drawdown_monitor=self.ddm,  # Pass DrawdownMonitor for risk control
        )

        # Wire optional attrs
        if hasattr(self.engine, "trade_gate"):
            self.engine.trade_gate = self.trade_gate

        # State reconciler - ensures local state matches broker
        reconciler_config = ReconcilerConfig(
            reconcile_interval=60,
            halt_on_critical=True,
            auto_correct_minor=True,
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

        # Bar aggregation state (for building OHLCV bars from quotes)
        # Structure: {symbol: {"open": float, "high": float, "low": float, "close": float, "volume": int, "bar_id": int}}
        self._bar_aggregation: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"open": None, "high": None, "low": None, "close": None, "volume": 0, "bar_id": None}
        )

        # Unified data pipeline (supports Alpaca and Schwab with fallback)
        self.data_pipeline = UnifiedDataPipeline(
            data_path=str(ROOT / "data" / "data_storage" / "proc_data"),
            raw_data_path=str(ROOT / "data" / "data_storage" / "raw_data"),
        )

        # Also keep the simple updater for quick freshness checks
        self.data_updater = HistoricalDataUpdater(
            api_key=os.getenv("ALPACA_API_KEY"),
            api_secret=os.getenv("ALPACA_SECRET_KEY"),
            data_path=str(ROOT / "data" / "data_storage" / "proc_data"),
        )

        # Background update task reference
        self._update_task: Optional[asyncio.Task] = None

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

        Schwab LEVELONE_EQUITIES streaming field numbers (from API docs):
        - 1: Bid Price
        - 2: Ask Price
        - 3: Last Price (price at which the last trade was matched)
        - 8: Total Volume (aggregated shares traded throughout the day)
        - 10: High Price (day's high)
        - 11: Low Price (day's low)
        - 12: Close Price (previous day's closing price)
        - 17: Open Price (day's open)
        - 35: Trade Time in Long (milliseconds since Epoch)
        """
        # Parse timestamp from trade time field, fallback to now
        trade_time_ms = quote.get('35') or quote.get('trade_time')
        if trade_time_ms:
            ts = datetime.fromtimestamp(int(trade_time_ms) / 1000, tz=timezone.utc)
        else:
            ts = datetime.now(timezone.utc)

        # Extract prices using correct Schwab field numbers
        bid_price = quote.get('1') or quote.get('bid_price') or quote.get('bidPrice', 0)
        ask_price = quote.get('2') or quote.get('ask_price') or quote.get('askPrice', 0)
        last_price = quote.get('3') or quote.get('last_price') or quote.get('lastPrice', 0)
        volume = quote.get('8') or quote.get('volume') or quote.get('totalVolume', 0)
        high_price = quote.get('10') or quote.get('high_price') or quote.get('highPrice', 0)
        low_price = quote.get('11') or quote.get('low_price') or quote.get('lowPrice', 0)
        prev_close = quote.get('12') or quote.get('close_price') or quote.get('closePrice', 0)
        open_price = quote.get('17') or quote.get('open_price') or quote.get('openPrice', 0)

        # Use last price as the primary price, fallback to mid if not available
        if last_price:
            price = float(last_price)
        elif bid_price and ask_price:
            price = (float(bid_price) + float(ask_price)) / 2
        else:
            price = float(bid_price or ask_price or 0)

        # For intraday bars, use day's OHLC if available, otherwise use last price
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

    @staticmethod
    def _canonicalize_schwab_chart_bar(data: Dict, symbol: str) -> Dict:
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
        # Parse timestamp from chart time field
        chart_time_ms = data.get('7') or data.get('chart_time')
        if chart_time_ms:
            ts = datetime.fromtimestamp(int(chart_time_ms) / 1000, tz=timezone.utc)
        else:
            ts = datetime.now(timezone.utc)

        return {
            "symbol": symbol,
            "timestamp": ts,
            "Open": float(data.get('1') or data.get('open', 0)),
            "High": float(data.get('2') or data.get('high', 0)),
            "Low": float(data.get('3') or data.get('low', 0)),
            "Close": float(data.get('4') or data.get('close', 0)),
            "Volume": int(data.get('5') or data.get('volume', 0)),
            "sequence": int(data.get('6') or data.get('sequence', 0)),
        }

    @staticmethod
    def _bar_bucket(ts: datetime, timeframe_sec: int = 60) -> int:
        """Calculate bar bucket ID for deduplication."""
        return int(ts.timestamp() // timeframe_sec)

    def _aggregate_quote_to_bar(self, symbol: str, price: float, volume: int, bar_id: int) -> Dict:
        """
        Aggregate quote data into OHLCV bar.

        Collects multiple quotes within the same time bucket into a proper bar
        with Open, High, Low, Close, and cumulative Volume.

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
            # Finalize previous bar data before starting new one
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
                # First bar, return current state
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
    async def seed(self, lookback_bars: int = 200, max_stale_minutes: int = 60):
        """
        Seed historical data for all symbols.

        Uses the unified data pipeline which:
        - Automatically selects best data source (Alpaca or Schwab)
        - Falls back if one source is unavailable
        - Processes data through the full ML pipeline

        Args:
            lookback_bars: Number of bars to load for each symbol
            max_stale_minutes: If data is older than this, fetch fresh data
        """
        # Check available data sources
        self.logger.info("Checking data source availability...")
        sources = await self.data_pipeline.check_sources()
        self.logger.info(f"Recommended data source: {sources['recommended']}")

        if sources['recommended'] == 'none':
            self.logger.error("No data sources available! Check credentials.")
            # Try to continue with existing data

        # Check data freshness and update if needed
        symbols_to_update = []
        for sym in self.symbols:
            freshness = self.data_updater.get_data_freshness(sym)
            if freshness is None:
                self.logger.warning(f"[{sym}] No historical data found - will fetch")
                symbols_to_update.append(sym)
            elif freshness['age_minutes'] > max_stale_minutes:
                self.logger.info(
                    f"[{sym}] Data is stale ({freshness['age_minutes']} min old) - will update"
                )
                symbols_to_update.append(sym)
            else:
                self.logger.info(
                    f"[{sym}] Data is fresh ({freshness['age_minutes']} min old, "
                    f"{freshness['bar_count']} bars)"
                )

        # Fetch fresh data for stale symbols using unified pipeline
        # This handles source selection and full data processing
        if symbols_to_update:
            self.logger.info(f"Updating historical data for: {symbols_to_update}")
            await self.data_pipeline.update_symbols(
                symbols_to_update,
                days=30,  # Fetch more days for proper processing
                source=None,  # Auto-select best source
                process_data=True,  # Run through ML pipeline
            )

        # Load data from files
        loader = HistoricalBarLoader(str(ROOT / "data" / "data_storage" / "proc_data"))
        for sym in self.symbols:
            bars = loader.load_last_n_bars(sym, n=lookback_bars)
            if not bars:
                self.logger.warning(f"[{sym}] No bars loaded after update attempt")
                continue

            for b in bars:
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

            self.logger.info(f"[{sym}] Seeded with {len(bars)} bars")

        self.logger.info(f"Seeded {len(self.symbols)} symbols with up to {lookback_bars} bars.")

    # ---------- Quote/Bar Callback ----------
    async def on_schwab_quote(self, symbol: str, quote: Dict):
        """
        Handle incoming Schwab quote data.

        Aggregates quotes into OHLCV bars within time buckets and processes
        through the strategy engine when bars close.

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
        prev_bar_id = self._last_bar_id.get(symbol)

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

        self._last_bar_id[symbol] = bar_id

        # Track history + MTM (only add completed bars to history)
        if bar_closed:
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

        # Drawdown monitor (daily tick + updates) - only if enabled
        if self.ddm is not None:
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
            drawdown=self.ddm.get_portfolio_drawdown() if self.ddm else 0.0,
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

        # Hand off to execution engine using SignalContext
        context = SignalContext(
            symbol=symbol,
            signal=signal,
            price=last_px,
            atr=float(atr or 0.0),
            regime=regime,
            timestamp=ts,
            strategy_name=strategy_name,
            market_open=True,  # Schwab streaming only runs during market hours
            metadata={'state': state}
        )
        await self.engine.handle_signal_context(context)

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

    # ---------- Periodic Data Update ----------
    async def _periodic_data_update(self, interval_minutes: int = 60):
        """
        Background task to periodically update historical data.

        Uses the unified pipeline which:
        - Auto-selects best data source (Alpaca or Schwab)
        - Processes data through ML pipeline
        - Handles source failover

        This ensures historical data files stay fresh for strategy warmup
        and indicator calculations.
        """
        while True:
            await asyncio.sleep(interval_minutes * 60)

            try:
                self.logger.info("Running periodic historical data update...")

                # Use unified pipeline for full processing
                results = await self.data_pipeline.update_symbols(
                    self.symbols,
                    days=5,  # Fetch enough for processing
                    source=None,  # Auto-select
                    process_data=True,  # Full ML pipeline
                )
                total_bars = sum(results.values())
                self.logger.info(f"Periodic update complete: {total_bars} total bars fetched")

            except Exception as e:
                self.logger.exception(f"Periodic data update failed: {e}")

    # ---------- Health Check ----------
    async def _emit_health_status(self, status: str, details: Dict = None):
        """Emit health status event."""
        await self.event_handler.emit(EVENT_HEALTH_UPDATE, {
            "broker": "schwab",
            "status": status,
            "details": details or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    # ---------- Preflight Checks ----------
    async def _preflight_credential_check(self) -> None:
        """
        Check Schwab token status before starting.

        Logs warnings/errors about token expiry to alert the user.
        Does not block startup - just provides visibility.
        """
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
                f"SCHWAB TOKEN EXPIRING in {hours} hours. "
                f"Renew soon: python -m data.streaming.authenticator"
            )

        elif result.status == CredentialStatus.MISSING:
            self.logger.warning(
                "Schwab credentials not configured. "
                "Run: python -m data.streaming.authenticator"
            )

        elif result.status == CredentialStatus.VALID:
            days = result.expires_in // 86400 if result.expires_in else 0
            self.logger.info(f"Schwab credentials valid ({days} days until refresh expires)")

    # ---------- Connection Management ----------
    async def connect(self) -> bool:
        """
        Establish connection to Schwab streaming.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            api_key = os.getenv("SCHWAB_API_KEY")
            secret_key = os.getenv("SCHWAB_SECRET")

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

        # Preflight: Check Schwab token status and alert if renewal needed
        await self._preflight_credential_check()

        # Seed historical data (fetches fresh if stale)
        max_stale = self.config.data.max_stale_minutes
        await self.seed(self.config.data.seed_bars, max_stale_minutes=max_stale)

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

            # Get actual broker values for GUI (not calculated)
            broker_snapshot = await self.broker.get_account_info()

            # Emit initial position updates for GUI
            for symbol, pos_view in broker_snapshot.positions.items():
                last_price = pos_view.market_price or pos_view.avg_entry_price
                unrealized = pos_view.unrealized_pl or 0.0
                await self.event_handler.emit(EVENT_POSITION_UPDATE, PositionPayload(
                    symbol=symbol,
                    qty=pos_view.qty,
                    avg_price=pos_view.avg_entry_price,
                    avg=pos_view.avg_entry_price,  # GUI model uses 'avg'
                    last=last_price,  # GUI model uses 'last'
                    unrealized=unrealized,
                    unreal=unrealized,  # GUI model uses 'unreal'
                    realized=0.0,
                    market_value=pos_view.qty * last_price,
                    side=pos_view.side or ("long" if pos_view.qty > 0 else "short"),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))

            # Emit initial PnL for GUI using broker's actual values
            await self.event_handler.emit(EVENT_PNL_UPDATE, PnLPayload(
                portfolio_value=broker_snapshot.equity,
                equity_curve=[broker_snapshot.equity],
                unrealized=sum(p.unrealized_pl or 0.0 for p in broker_snapshot.positions.values()),
                realized=0.0,
                drawdown=0.0,
                cash=broker_snapshot.cash,
                buying_power=broker_snapshot.buying_power,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))
            self.logger.info(f"[GUI] Emitted initial state: equity=${broker_snapshot.equity:,.2f}, "
                           f"cash=${broker_snapshot.cash:,.2f}, positions={len(broker_snapshot.positions)}")

        # Start streaming
        stream_task = asyncio.create_task(self.broker.start_stream())
        self.logger.info(f"SchwabLiveRunner started for: {', '.join(self.symbols)}")

        # Start periodic state reconciliation
        reconcile_interval = 60
        await self.reconciler.start_periodic(interval_seconds=reconcile_interval)
        self.logger.info(f"State reconciliation enabled (every {reconcile_interval}s)")

        # Start periodic historical data updates (runs in background)
        update_interval = self.config.data.historical_update_interval_minutes
        if update_interval > 0:
            self._update_task = asyncio.create_task(
                self._periodic_data_update(interval_minutes=update_interval)
            )
            self.logger.info(f"Periodic data updates enabled (every {update_interval} min)")

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

            # Cancel background update task
            if self._update_task:
                self._update_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._update_task

            stream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stream_task

            # Disconnect broker (async)
            await self.broker.disconnect()

            # Shutdown event handler (waits for pending tasks, closes thread pool)
            await self.event_handler.shutdown()

            self.trade_logger.flush()
            await self._emit_health_status("disconnected", {"reason": "shutdown"})
            self.logger.info("SchwabLiveRunner shut down cleanly.")

    async def stop(self):
        """Stop the live trading runner."""
        self._running = False
        # Disconnect broker
        if hasattr(self.broker, 'disconnect'):
            await self.broker.disconnect()
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
    _ensure_live_config("config")

    # Load config
    config = get_config()

    # Get symbols from config or default
    symbols = config.general.default_symbols or ["AAPL", "MSFT"]

    runner = SchwabLiveRunner(symbols=symbols, config=config)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
