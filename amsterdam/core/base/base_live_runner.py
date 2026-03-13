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
from collections import defaultdict, deque
from dataclasses import asdict
from datetime import datetime, date, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, Deque, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from loggers.factory import get_module_logger
from loggers.file_trade_logger import FileTradeLogger

from core.logic.portfolio_state import PortfolioState
from core.logic.symbol_state import SymbolState
from core.logic.trade_gate import TradeGate
from core.logic.strategy_routing_manager import StrategyRoutingManager
from core.logic.hybrid_position_sizer import HybridPositionSizer
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
from core.config_loader import get_config, create_position_sizer, create_drawdown_monitor, create_position_manager, TradingConfig
from core.simulator.simulation import compute_atr, classify_regime, IncrementalATR
from core.logic.live_execution_engine import LiveExecutionEngine
from core.logic.trade_logic_manager import DynamicTradeLogicManager
from core.app_types import SignalContext
from core.executor import LiveExecutor
from core.state_reconciler import StateReconciler, ReconcilerConfig

if TYPE_CHECKING:
    from core.base.base_broker_interface import BaseBrokerInterface


# Get ROOT path for config files
ROOT = Path(__file__).resolve().parents[2]  # .../schwab_trader


class NumpyBarBuffer:
    """
    Efficient circular buffer for OHLCV data using NumPy arrays.

    Provides O(1) append operations and efficient array access for
    calculations like ATR. Converts to DataFrame lazily only when
    needed for strategy signal generation.

    Features:
    - Pre-allocated arrays to avoid memory allocation per bar
    - Circular buffer behavior for bounded memory usage
    - Cached array views to avoid redundant computation
    - Lazy DataFrame conversion
    """

    def __init__(self, maxlen: int = 300):
        """
        Initialize the buffer with pre-allocated arrays.

        Args:
            maxlen: Maximum number of bars to store
        """
        self.maxlen = maxlen
        self.open = np.zeros(maxlen, dtype=np.float64)
        self.high = np.zeros(maxlen, dtype=np.float64)
        self.low = np.zeros(maxlen, dtype=np.float64)
        self.close = np.zeros(maxlen, dtype=np.float64)
        self.volume = np.zeros(maxlen, dtype=np.int64)
        self.timestamps = np.empty(maxlen, dtype='datetime64[ns]')
        self._idx = 0  # Next write position
        self._count = 0  # Number of items stored
        # Caching for get_arrays() and to_dataframe()
        self._cached_arrays: Optional[Tuple[np.ndarray, ...]] = None
        self._cached_df: Optional[pd.DataFrame] = None
        self._cache_idx = -1  # Index when cache was last valid

    def append(self, bar: Dict) -> None:
        """
        Append a bar to the circular buffer.

        Args:
            bar: Dict with Open, High, Low, Close, Volume, timestamp keys
        """
        idx = self._idx % self.maxlen
        self.open[idx] = bar["Open"]
        self.high[idx] = bar["High"]
        self.low[idx] = bar["Low"]
        self.close[idx] = bar["Close"]
        self.volume[idx] = bar.get("Volume", 0)

        # Handle timestamp conversion
        ts = bar["timestamp"]
        if isinstance(ts, datetime):
            # Convert tz-aware datetime to UTC and strip tzinfo for np.datetime64
            if ts.tzinfo is not None:
                ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
            self.timestamps[idx] = np.datetime64(ts, 'ns')
        else:
            self.timestamps[idx] = ts

        self._idx += 1
        self._count = min(self._count + 1, self.maxlen)
        # Invalidate cache on new data
        self._cached_arrays = None
        self._cached_df = None

    def __len__(self) -> int:
        """Return the number of bars in the buffer."""
        return self._count

    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get ordered OHLCV arrays for calculations.

        Returns cached arrays if available, otherwise computes and caches.

        Returns:
            Tuple of (open, high, low, close, volume) arrays in chronological order
        """
        if self._cached_arrays is not None:
            return self._cached_arrays

        if self._count == 0:
            empty = np.array([], dtype=np.float64)
            result = (empty, empty, empty, empty, np.array([], dtype=np.int64))
            self._cached_arrays = result
            return result

        if self._count < self.maxlen:
            # Buffer not yet full - return views (no copy needed for read-only)
            result = (
                self.open[:self._count],
                self.high[:self._count],
                self.low[:self._count],
                self.close[:self._count],
                self.volume[:self._count],
            )
        else:
            # Buffer is full - use concatenation (faster than np.roll for this case)
            start = self._idx % self.maxlen
            result = (
                np.concatenate([self.open[start:], self.open[:start]]),
                np.concatenate([self.high[start:], self.high[:start]]),
                np.concatenate([self.low[start:], self.low[:start]]),
                np.concatenate([self.close[start:], self.close[:start]]),
                np.concatenate([self.volume[start:], self.volume[:start]]),
            )

        self._cached_arrays = result
        return result

    def get_timestamps(self) -> np.ndarray:
        """Get ordered timestamp array."""
        if self._count == 0:
            return np.array([], dtype='datetime64[ns]')

        if self._count < self.maxlen:
            return self.timestamps[:self._count]
        else:
            start = self._idx % self.maxlen
            return np.concatenate([self.timestamps[start:], self.timestamps[:start]])

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert buffer to DataFrame for strategy compatibility.

        Returns cached DataFrame if available. This is a lazy conversion -
        only called when strategy needs DataFrame.
        """
        if self._cached_df is not None:
            return self._cached_df

        if self._count == 0:
            return pd.DataFrame()

        o, h, l, c, v = self.get_arrays()
        ts = self.get_timestamps()

        df = pd.DataFrame({
            'timestamp': pd.to_datetime(ts),
            'Open': o,
            'High': h,
            'Low': l,
            'Close': c,
            'Volume': v,
        })
        self._cached_df = df
        return df


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
        # ATR history for regime classification
        self.atr_hist: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=300))
        # NumPy bar buffers for efficient OHLCV storage (replaces history deque)
        self.np_buffers: Dict[str, NumpyBarBuffer] = defaultdict(NumpyBarBuffer)
        # Incremental ATR calculators for O(1) updates
        self.atr_calc: Dict[str, IncrementalATR] = defaultdict(IncrementalATR)

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
        # Create PositionManager with config values (min_bars_to_hold, swing_mode, etc.)
        self.position_manager = create_position_manager(self.config)
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
            position_manager=self.position_manager,
        )

        # Provide ATR history reference for meta-model feature computation
        self.engine.set_atr_hist_reference(self.atr_hist)

        # Log trading rules
        tl = self.config.trade_logic
        cutoff_minute = getattr(tl, 'trading_cutoff_minute_et', 0)
        self.logger.info(
            f"Trading rules: min_bars_to_hold={tl.min_bars_to_hold}, "
            f"cutoff={tl.trading_cutoff_hour_et:02d}:{cutoff_minute:02d} ET, "
            f"close_eod={tl.close_positions_eod}"
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
        self._last_reset_minute: Optional[int] = None  # Track when reservations were last reset
        self._running = False
        self._eod_close_triggered: bool = False  # Track if EOD close has been triggered today

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

        # Daily indicator context for hybrid position sizing
        # Maps symbol -> dict of daily indicator values (RSI, MACD, SMA_200, etc.)
        self.daily_context: Dict[str, Dict[str, Any]] = {}
        self._last_daily_refresh: Optional[date] = None

        # Hybrid position sizer for confidence + trend-based sizing
        hybrid_config = getattr(self.config, 'hybrid_sizing', None)
        if hybrid_config and hybrid_config.enabled:
            self.hybrid_sizer = HybridPositionSizer(
                enabled=True,
                config=asdict(hybrid_config)
            )
            self.logger.info(
                f"Hybrid position sizing ENABLED: "
                f"high_conf={hybrid_config.high_confidence_threshold}, "
                f"low_conf={hybrid_config.low_confidence_threshold}"
            )
        else:
            self.hybrid_sizer = HybridPositionSizer(enabled=False)
            self.logger.info("Hybrid position sizing DISABLED (pass-through mode)")

        # Pass hybrid sizer to execution engine
        self.engine.hybrid_sizer = self.hybrid_sizer

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

    async def _close_all_positions_eod(self) -> None:
        """
        Close all positions at end of day.

        Called when past trading_cutoff_hour_et and close_positions_eod is True.
        Uses broker's actual positions to avoid state mismatch errors.
        """
        self.logger.info("=" * 60)
        self.logger.info("EOD POSITION CLOSE: Closing all positions")
        self.logger.info("=" * 60)

        try:
            broker_positions = await self.broker.get_positions()
        except Exception as e:
            self.logger.error(f"EOD close failed - couldn't get positions: {e}")
            return

        if not broker_positions:
            self.logger.info("EOD close: No positions to close")
            return

        closed_count = 0
        for pos in broker_positions:
            if pos.qty == 0:
                continue

            try:
                qty = abs(int(pos.qty))
                side = "sell" if pos.qty > 0 else "buy"

                self.logger.info(f"[{pos.symbol}] EOD closing: {side} {qty} shares")

                await self.broker.place_market_order(
                    symbol=pos.symbol,
                    qty=qty,
                    side=side
                )
                closed_count += 1

                # Update local state
                state = self.symbol_state.get(pos.symbol)
                if state:
                    state.side = None
                    state.bars_held = 0

            except Exception as e:
                self.logger.error(f"[{pos.symbol}] EOD close failed: {e}")

        self.logger.info(f"EOD close complete: {closed_count} positions closed")

    @staticmethod
    def _bar_bucket(ts: datetime, timeframe_sec: int = 60) -> int:
        """Calculate bar bucket ID for deduplication."""
        return int(ts.timestamp() // timeframe_sec)

    def _df_from_history(self, symbol: str) -> pd.DataFrame:
        """Create DataFrame from recent history using NumpyBarBuffer."""
        return self.np_buffers[symbol].to_dataframe()

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
                bar_dict = {
                    "timestamp": b["timestamp"],
                    "symbol": b["symbol"],
                    "Open": b["Open"], "High": b["High"], "Low": b["Low"],
                    "Close": b["Close"], "Volume": b.get("Volume", 0),
                }
                # Populate NumPy buffer (single source of truth for history)
                self.np_buffers[sym].append(bar_dict)
                # Warm up incremental ATR calculator
                self.atr_calc[sym].update(
                    float(b["High"]),
                    float(b["Low"]),
                    float(b["Close"])
                )

            # Get final ATR after seeding
            atr = self.atr_calc[sym].current_atr
            if atr is not None:
                self.atr_hist[sym].append(atr)

            # Load daily context for hybrid sizing
            # Use the last bar's daily indicators (RSI, MACD, SMA_200, Close)
            if bars:
                last_bar = bars[-1]
                self.daily_context[sym] = {
                    'RSI': last_bar.get('RSI'),
                    'MACD': last_bar.get('MACD'),
                    'MACD_Signal': last_bar.get('MACD_Signal'),
                    'SMA_200': last_bar.get('SMA_200'),
                    'Close': last_bar.get('Close'),
                }
                rsi_val = self.daily_context[sym].get('RSI')
                if rsi_val is not None:
                    self.logger.info(f"[{sym}] Loaded daily context: RSI={rsi_val:.1f}")
                else:
                    self.logger.debug(f"[{sym}] Daily context loaded (RSI not available)")

            self.logger.info(f"[{sym}] Seeded with {len(bars)} bars")

        self._last_daily_refresh = datetime.now(timezone.utc).date()
        self.logger.info(f"Seeded {len(self.symbols)} symbols with up to {lookback_bars} bars.")

    async def _process_bar(self, bar: Dict) -> None:
        """
        Process a canonicalized bar through the strategy pipeline.

        This is the core bar processing logic shared by all runners.
        Uses batched event emissions for efficiency.

        Args:
            bar: Canonical bar dict with symbol, timestamp, OHLCV
        """
        symbol = bar["symbol"]
        ts: datetime = bar["timestamp"]

        # Check trading cutoff time (convert to ET for comparison)
        cutoff_hour_et = getattr(self.config.trade_logic, 'trading_cutoff_hour_et', 15)
        cutoff_minute_et = getattr(self.config.trade_logic, 'trading_cutoff_minute_et', 45)

        # Proper timezone conversion to Eastern Time (handles DST automatically)
        et_tz = ZoneInfo("America/New_York")
        if ts.tzinfo is None:
            # Assume naive datetime is UTC (Alpaca streams in UTC)
            ts_utc = ts.replace(tzinfo=timezone.utc)
        else:
            ts_utc = ts
        ts_et = ts_utc.astimezone(et_tz)

        # Compare time as total minutes since midnight for precise cutoff
        current_minutes = ts_et.hour * 60 + ts_et.minute
        cutoff_minutes = cutoff_hour_et * 60 + cutoff_minute_et
        past_cutoff = current_minutes >= cutoff_minutes

        # Reset EOD close flag at start of new trading day (before cutoff)
        if not past_cutoff:
            self._eod_close_triggered = False

        # Close all positions at EOD if enabled and past cutoff
        close_eod = getattr(self.config.trade_logic, 'close_positions_eod', False)
        if past_cutoff and close_eod and not self._eod_close_triggered:
            self.logger.info(
                f"EOD TRIGGER: Bar time {ts_et.strftime('%H:%M:%S')} ET >= cutoff {cutoff_hour_et:02d}:{cutoff_minute_et:02d} ET"
            )
            self._eod_close_triggered = True
            await self._close_all_positions_eod()

        # Reset position sizer reservations at the start of each new minute
        # This prevents accumulated reservations from blocking new trades
        current_minute = ts.minute + ts.hour * 60
        if self._last_reset_minute is None or current_minute != self._last_reset_minute:
            self.sizer.reset_bar_reservations()
            self._last_reset_minute = current_minute

        bar_id = self._bar_bucket(ts)
        prev_bar_id = self._last_bar_id.get(symbol)
        bar_closed = (prev_bar_id is None) or (bar_id != prev_bar_id)
        self._last_bar_id[symbol] = bar_id

        # Track history in NumPy buffer (single source of truth)
        self.np_buffers[symbol].append(bar)
        last_px = float(bar["Close"])
        self.portfolio.update_price(symbol, last_px)

        # Update symbol state
        state = self.symbol_state.setdefault(symbol, SymbolState(symbol=symbol))
        state.portfolio_value = self.portfolio.total_equity()
        state.ts = ts
        state.bar_id = bar_id
        state.bar_closed = bar_closed

        # Incremental ATR calculation - O(1) instead of O(period) per bar
        atr = self.atr_calc[symbol].update(
            float(bar["High"]),
            float(bar["Low"]),
            float(bar["Close"])
        )
        if atr is not None:
            self.atr_hist[symbol].append(atr)
        regime = classify_regime(atr, list(self.atr_hist[symbol]))

        # Drawdown monitor (daily tick + updates) - only if enabled
        if self.ddm is not None:
            if self._last_ddm_date is None or ts.date() != self._last_ddm_date:
                self.ddm.start_new_day(portfolio_equity=self.portfolio.total_equity())
                self._last_ddm_date = ts.date()
            self.ddm.update_portfolio(self.portfolio.total_equity())
            self.ddm.update_symbol(symbol, self._symbol_mv(symbol, last_px))

        # Strategy & signal - use NumpyBarBuffer for efficient DataFrame conversion
        df = self.np_buffers[symbol].to_dataframe()
        strategy = self.router.get_strategy(symbol, regime)
        strategy_name = type(strategy).__name__
        try:
            raw_signal = strategy.generate_signal(df)
            signal = int(raw_signal if isinstance(raw_signal, (int, float)) else getattr(raw_signal, "signal", 0))
        except Exception as e:
            self.logger.exception(f"[{symbol}] Strategy error in {strategy_name}: {e}")
            signal = 0

        # Trade gate context
        self.trade_gate.on_new_bar(symbol, bar_id, regime)
        gs = self.trade_gate.get_state(symbol)
        state.regime = regime
        state.regime_persist = gs.regime_persist

        # Prepare event payloads (defer emission until after critical path)
        bar_payload = {
            "timestamp": ts,
            "symbol": symbol,
            "open": float(bar["Open"]),
            "high": float(bar["High"]),
            "low": float(bar["Low"]),
            "close": float(bar["Close"]),
            "volume": int(bar.get("Volume", 0)),
        }

        pnl_payload = PnLPayload(
            portfolio_value=self.portfolio.total_equity(),
            equity_curve=self.portfolio.equity_history,
            unrealized=self.portfolio.total_unrealized(),
            realized=self.portfolio.realized_pnl,
            drawdown=self.ddm.get_portfolio_drawdown() if self.ddm else 0.0,
            timestamp=ts.isoformat(),
        )

        signal_payload = StrategySignalPayload(
            symbol=symbol,
            strategy=strategy_name,
            signal={-1: "sell", 0: "hold", 1: "buy"}.get(signal, "hold"),
            confidence=None,
            timestamp=ts.isoformat(),
        )

        new_bar_payload = BarPayload(
            symbol=symbol,
            open=float(bar["Open"]),
            high=float(bar["High"]),
            low=float(bar["Low"]),
            close=float(bar["Close"]),
            volume=int(bar.get("Volume", 0)),
            timestamp=ts.isoformat(),
        )

        # Get daily context for hybrid sizing
        daily_ctx = self.daily_context.get(symbol, {})

        # Block new entries after cutoff time (but allow exits)
        # If past cutoff, force signal to 0 (hold) unless we're in a position
        adjusted_signal = signal
        if past_cutoff and not state.is_in_position:
            if signal != 0:
                self.logger.info(
                    f"[{symbol}] Entry blocked: past {cutoff_hour_et}:00 ET cutoff "
                    f"(current ET hour: {et_hour})"
                )
                adjusted_signal = 0

        # Hand off to execution engine (critical path - await directly)
        context = SignalContext(
            symbol=symbol,
            signal=adjusted_signal,
            price=last_px,
            atr=float(atr or 0.0),
            regime=regime,
            timestamp=ts,
            strategy_name=strategy_name,
            market_open=True,
            metadata={
                'state': state,
                'daily_context': daily_ctx,
                'past_cutoff': past_cutoff,
            }
        )
        await self.engine.handle_signal_context(context)

        # Batch emit all events after critical path completes
        # Using gather() reduces await overhead from 4 context switches to 1
        await asyncio.gather(
            self.event_handler.emit("BAR", bar_payload),
            self.event_handler.emit(EVENT_PNL_UPDATE, pnl_payload),
            self.event_handler.emit(EVENT_STRATEGY_SIGNAL, signal_payload),
            self.event_handler.emit(EVENT_NEW_BAR, new_bar_payload),
        )

        # Telemetry
        pos = self.portfolio.positions.get(symbol)
        qty = pos.qty if pos else 0
        self.logger.debug(
            f"[{symbol}] bar={bar_id} closed={bar_closed} regime={regime} "
            f"persist={gs.regime_persist} qty={qty} equity={self.portfolio.total_equity():.2f}"
        )

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

    # ==========================================================================
    # HELPER METHODS
    # ==========================================================================

    def _compute_atr_percentile(self, symbol: str, current_atr: float) -> float:
        """
        Compute the percentile of current ATR within recent history.

        Used for meta-model feature extraction - helps identify if current
        volatility is high or low relative to recent history.

        Args:
            symbol: Trading symbol
            current_atr: Current ATR value

        Returns:
            Percentile (0.0 to 1.0), or 0.5 if insufficient history
        """
        hist = list(self.atr_hist[symbol])
        if len(hist) < 10:
            return 0.5  # Not enough history for meaningful percentile

        # Count how many historical values are <= current ATR
        count_below = sum(1 for h in hist if h <= current_atr)
        return count_below / len(hist)
