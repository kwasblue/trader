# core/base/base_live_runner.py
"""
BaseLiveRunner - Abstract base class for live trading runners.

Consolidates common functionality between AlpacaLiveRunner and SchwabLiveRunner:
- Historical data seeding
- Bar processing pipeline
- P&L tracking and event emission
- Drawdown monitoring and trade gates
- State reconciliation
- Periodic data updates

Subclasses implement broker-specific:
- Connection/disconnection
- Bar canonicalization
- Streaming setup and subscriptions
"""
from __future__ import annotations

import asyncio
import contextlib
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd

from loggers.factory import get_module_logger
from loggers.file_trade_logger import FileTradeLogger

from core.logic.portfolio_state import PortfolioState
from core.logic.symbol_state import SymbolState
from core.logic.trade_gate import TradeGate
from core.logic.strategy_routing_manager import StrategyRoutingManager
from core.drawdown_monitor import DrawdownMonitor
from core.historical_loader import HistoricalBarLoader
from core.historical_data_updater import HistoricalDataUpdater
from core.unified_data_pipeline import UnifiedDataPipeline
from core.events.eventhandler import get_event_handler
from core.contracts.events import (
    EVENT_NEW_BAR, BarPayload,
    EVENT_STRATEGY_SIGNAL, StrategySignalPayload,
    EVENT_PNL_UPDATE, PnLPayload,
    EVENT_POSITION_UPDATE, PositionPayload,
    EVENT_HEALTH_UPDATE,
)
from core.config_loader import get_config, create_position_sizer, create_drawdown_monitor, TradingConfig
from core.simulator.simulation import compute_atr, classify_regime
from core.logic.live_execution_engine import LiveExecutionEngine
from core.logic.trade_logic_manager import DynamicTradeLogicManager
from core.app_types import SignalContext
from core.executor import LiveExecutor
from core.state_reconciler import StateReconciler, ReconcilerConfig

if TYPE_CHECKING:
    from core.base.base_broker_interface import BaseBrokerInterface


# Get ROOT path for config files
ROOT = Path(__file__).resolve().parents[2]  # .../schwab_trader


class BaseLiveRunner(ABC):
    """
    Abstract base class for live trading runners.

    Provides common infrastructure for both Alpaca and Schwab runners:
    - Portfolio state and symbol state tracking
    - Risk management (trade gates, drawdown monitoring)
    - Execution engine with trade logic routing
    - Historical data pipeline
    - Event emission for GUI

    Subclasses must implement broker-specific abstract methods.
    """

    # ---------- Class Constants (Override in subclasses) ----------
    BROKER_NAME: str = "Base"
    LOG_FILE_KEY: str = "LiveRunner"
    TRADE_LOG_FILE: str = "live_trades.csv"

    def __init__(self, symbols: List[str], config: Optional[TradingConfig] = None):
        """
        Initialize the live runner.

        Args:
            symbols: List of symbols to trade
            config: Optional TradingConfig instance (uses global config if not provided)
        """
        self.symbols = symbols
        self.event_handler = get_event_handler()

        # Load centralized config
        self.config = config or get_config()

        # Logging + trade log
        self.logger = get_module_logger(
            module_name=f'{self.BROKER_NAME}LiveRunner',
            file_key=self.LOG_FILE_KEY
        )
        self.trade_logger = FileTradeLogger(
            log_file=self.TRADE_LOG_FILE,
            logger_name=f'{self.BROKER_NAME}TradeLogger'
        )

        # Create broker (subclass implements this)
        self.broker: BaseBrokerInterface = self._create_broker()

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
        self.executor = LiveExecutor(broker=self.broker)
        self.engine = LiveExecutionEngine(
            broker=self.broker,
            executor=self.executor,
            sizer=self.sizer,
            performance_tracker=self.trade_logger,
            trade_logic_manager=DynamicTradeLogicManager(str(ROOT / "config" / "trade_logic_routing.json")),
            portfolio=self.portfolio,
            sync_on_start=False,  # Sync is done by reconciler in run()
            event_handler=self.event_handler,
            drawdown_monitor=self.ddm,
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

    # ==========================================================================
    # ABSTRACT METHODS - Subclasses must implement
    # ==========================================================================

    @abstractmethod
    def _create_broker(self) -> "BaseBrokerInterface":
        """
        Create and configure the broker instance.

        Returns:
            Configured broker instance
        """
        pass

    @abstractmethod
    def _canonicalize_bar(self, raw_data: Any) -> Dict:
        """
        Convert broker-specific bar/quote data to canonical format.

        Args:
            raw_data: Raw bar or quote data from broker

        Returns:
            Dict with keys: symbol, timestamp, Open, High, Low, Close, Volume
        """
        pass

    @abstractmethod
    async def _connect_broker(self) -> None:
        """
        Establish connection to the broker.

        Called at the start of run() before streaming begins.
        """
        pass

    @abstractmethod
    async def _start_streaming(self) -> asyncio.Task:
        """
        Start the broker's data streaming.

        Returns:
            asyncio.Task for the streaming loop
        """
        pass

    @abstractmethod
    async def _disconnect_broker(self) -> None:
        """
        Disconnect from the broker.

        Called during cleanup in run().
        """
        pass

    @abstractmethod
    def _subscribe_to_data(self) -> None:
        """
        Subscribe to data feeds for all symbols.

        Called after broker connection is established.
        """
        pass

    # ==========================================================================
    # HOOK METHODS - Can be overridden for customization
    # ==========================================================================

    async def _preflight_checks(self) -> None:
        """
        Run pre-flight checks before starting.

        Override for broker-specific validation (e.g., token expiry).
        """
        pass

    async def _main_loop(self, stream_task: asyncio.Task) -> None:
        """
        Main run loop logic.

        Override to customize loop behavior (e.g., reconnection handling).

        Args:
            stream_task: The streaming task to monitor
        """
        while self._running:
            # Check if reconciler halted trading
            if self.reconciler.is_halted:
                self.logger.critical("Trading halted by reconciler - exiting run loop")
                break
            await asyncio.sleep(0.5)

    async def _cleanup(self, stream_task: asyncio.Task) -> None:
        """
        Cleanup after run loop ends.

        Override to add broker-specific cleanup.

        Args:
            stream_task: The streaming task to cancel
        """
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

        # Disconnect broker
        await self._disconnect_broker()

        # Shutdown event handler
        await self.event_handler.shutdown()

        self.trade_logger.flush()
        self.logger.info(f"{self.BROKER_NAME}LiveRunner shut down cleanly.")

    # ==========================================================================
    # CONCRETE METHODS - Shared implementation
    # ==========================================================================

    def _on_reconciler_halt(self, message: str):
        """Called when reconciler detects critical state mismatch."""
        self.logger.critical(f"TRADING HALTED: {message}")
        self.logger.critical("Manual intervention required. Review positions at broker.")
        self._running = False

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

        # Fetch fresh data for stale symbols
        if symbols_to_update:
            self.logger.info(f"Updating historical data for: {symbols_to_update}")
            await self.data_pipeline.update_symbols(
                symbols_to_update,
                days=30,
                source=None,  # Auto-select best source
                process_data=True,
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

    async def _process_bar(self, bar: Dict) -> None:
        """
        Process a canonicalized bar through the strategy pipeline.

        This is the core bar processing logic shared by all runners.

        Args:
            bar: Canonical bar dict with symbol, timestamp, OHLCV
        """
        symbol = bar["symbol"]
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

        # Hand off to execution engine
        context = SignalContext(
            symbol=symbol,
            signal=signal,
            price=last_px,
            atr=float(atr or 0.0),
            regime=regime,
            timestamp=ts,
            strategy_name=strategy_name,
            market_open=True,
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

    async def _periodic_data_update(self, interval_minutes: int = 60):
        """
        Background task to periodically update historical data.

        Uses the unified pipeline for full processing.
        """
        while True:
            await asyncio.sleep(interval_minutes * 60)

            try:
                self.logger.info("Running periodic historical data update...")

                results = await self.data_pipeline.update_symbols(
                    self.symbols,
                    days=5,
                    source=None,
                    process_data=True,
                )
                total_bars = sum(results.values())
                self.logger.info(f"Periodic update complete: {total_bars} total bars fetched")

            except Exception as e:
                self.logger.exception(f"Periodic data update failed: {e}")

    async def _sync_and_emit_initial_state(self) -> bool:
        """
        Sync state with broker and emit initial GUI events.

        Returns:
            True if sync was successful
        """
        self.logger.info("Syncing portfolio state with broker...")
        sync_result = await self.reconciler.full_sync()

        if not sync_result.success:
            self.logger.error("Failed to sync with broker - check credentials and connectivity")
            return False

        self.logger.info(
            f"Portfolio synced: {sync_result.broker_positions} positions, "
            f"${sync_result.broker_cash:,.2f} cash"
        )

        # Get actual broker values for GUI
        broker_snapshot = await self.broker.get_account_info()

        # Emit initial position updates for GUI
        for symbol, pos_view in broker_snapshot.positions.items():
            last_price = pos_view.market_price or pos_view.avg_entry_price
            unrealized = pos_view.unrealized_pl or 0.0
            await self.event_handler.emit(EVENT_POSITION_UPDATE, PositionPayload(
                symbol=symbol,
                qty=pos_view.qty,
                avg_price=pos_view.avg_entry_price,
                avg=pos_view.avg_entry_price,
                last=last_price,
                unrealized=unrealized,
                unreal=unrealized,
                realized=0.0,
                market_value=pos_view.qty * last_price,
                side=pos_view.side or ("long" if pos_view.qty > 0 else "short"),
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))

        # Emit initial PnL for GUI
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

        self.logger.info(
            f"[GUI] Emitted initial state: equity=${broker_snapshot.equity:,.2f}, "
            f"cash=${broker_snapshot.cash:,.2f}, positions={len(broker_snapshot.positions)}"
        )

        return True

    # ==========================================================================
    # TEMPLATE METHOD - The main run loop
    # ==========================================================================

    async def run(self):
        """
        Start the live trading runner.

        This is the main entry point that orchestrates:
        1. Pre-flight checks
        2. Historical data seeding
        3. Broker connection and streaming
        4. State reconciliation
        5. Main run loop
        6. Cleanup
        """
        self._running = True

        # Pre-flight checks (broker-specific)
        await self._preflight_checks()

        # Seed historical data
        max_stale = self.config.data.max_stale_minutes
        await self.seed(self.config.data.seed_bars, max_stale_minutes=max_stale)

        # Subscribe to bar debug logging
        await self.event_handler.subscribe("BAR", self._bar_debug_logger)

        # Connect to broker
        await self._connect_broker()

        # Subscribe to data feeds
        self._subscribe_to_data()

        # Sync state with broker
        await self._sync_and_emit_initial_state()

        # Start streaming
        stream_task = await self._start_streaming()
        self.logger.info(f"{self.BROKER_NAME}LiveRunner started for: {', '.join(self.symbols)}")

        # Start periodic state reconciliation
        reconcile_interval = 60
        await self.reconciler.start_periodic(interval_seconds=reconcile_interval)
        self.logger.info(f"State reconciliation enabled (every {reconcile_interval}s)")

        # Start periodic historical data updates
        update_interval = self.config.data.historical_update_interval_minutes
        if update_interval > 0:
            self._update_task = asyncio.create_task(
                self._periodic_data_update(interval_minutes=update_interval)
            )
            self.logger.info(f"Periodic data updates enabled (every {update_interval} min)")

        try:
            await self._main_loop(stream_task)
        finally:
            self._running = False
            await self._cleanup(stream_task)

    def stop(self):
        """Stop the live trading runner."""
        self._running = False
        self.logger.info("Stop requested")
