# core/alpaca_runner.py
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
from core.events.events import (EVENT_NEW_BAR, BarPayload, EVENT_STRATEGY_SIGNAL, StrategySignalPayload,
    EVENT_PNL_UPDATE, PnLPayload)

from core.simulator.simulation import compute_atr, classify_regime  # reuse your helpers
from core.broker.alpaca_broker import AlpacaBroker
from core.logic.live_execution_engine import LiveExecutionEngine
from core.logic.trade_logic_manager import DynamicTradeLogicManager
from core.executor import LiveExecutor
from core.historical_data_updater import HistoricalDataUpdater
from core.unified_data_pipeline import UnifiedDataPipeline
from core.credential_validator import CredentialValidator
from core.state_reconciler import StateReconciler, ReconcilerConfig
load_dotenv(ROOT / ".venv" / ".env")

class AlpacaLiveRunner:
    def __init__(self, settings: Settings, symbols: list[str]):
        self.settings = settings
        self.symbols = symbols
        self.event_handler = get_event_handler()

        # logging + trade log
        self.logger = get_module_logger(module_name='AlpacaLiveRunner', file_key='AlpacaLive')
        self.trade_logger = FileTradeLogger(log_file='live_trades.csv', logger_name='LiveTradeLogger')

        # broker (Alpaca)
        self.broker = AlpacaBroker(
            api_key=settings.get("ALPACA_API_KEY") or os.getenv("ALPACA_API_KEY"),
            api_secret=settings.get("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY"),
            paper=bool(settings.get("ALPACA_PAPER", True)),
        )

        # state
        self.portfolio = PortfolioState()
        self.symbol_state: dict[str, SymbolState] = defaultdict(SymbolState)
        self.history: dict[str, list[dict]] = defaultdict(list)
        self.atr_hist: dict[str, list[float]] = defaultdict(list)

        # risk & gates
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

        # sizer, router, executor, engine
        self.sizer = DynamicPositionSizer(
            risk_percentage=settings.get("BASE_RISK_PCT", 0.05)
        )
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
            sync_on_start=False,  # Sync is done by reconciler in run()
        )
        # If your engine has optional attrs (per our earlier patch), wire them:
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
        self._reconciler_task = None

        # bar tracking
        self._last_bar_id: dict[str, int] = {}
        self._last_ddm_date = None
        self._running = False

        # Unified data pipeline (supports Alpaca and Schwab with fallback)
        self.data_pipeline = UnifiedDataPipeline(
            data_path=str(ROOT / "data" / "data_storage" / "proc_data"),
            raw_data_path=str(ROOT / "data" / "data_storage" / "raw_data"),
        )

        # Also keep the simple updater for quick freshness checks
        self.data_updater = HistoricalDataUpdater(
            api_key=settings.get("ALPACA_API_KEY") or os.getenv("ALPACA_API_KEY"),
            api_secret=settings.get("ALPACA_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY"),
            data_path=str(ROOT / "data" / "data_storage" / "proc_data"),
        )

        # Credential validator
        self.credential_validator = CredentialValidator()

        # Background update task reference
        self._update_task: asyncio.Task | None = None
        

    # ---------- reconciler callback ----------
    def _on_reconciler_halt(self, message: str):
        """Called when reconciler detects critical state mismatch."""
        self.logger.critical(f"TRADING HALTED: {message}")
        self.logger.critical("Manual intervention required. Review positions at broker.")
        # Could emit event to GUI, send notification, etc.

    # ---------- helpers ----------
    @staticmethod
    def _canonicalize_alpaca_bar(bar) -> dict:
        ts = bar.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return {
        "symbol":    getattr(bar, "symbol", getattr(bar, "S", None)),
        "timestamp": getattr(bar, "timestamp", getattr(bar, "t", None)),
        "Open":      getattr(bar, "open", getattr(bar, "o", None)),
        "High":      getattr(bar, "high", getattr(bar, "h", None)),
        "Low":       getattr(bar, "low", getattr(bar, "l", None)),
        "Close":     getattr(bar, "close", getattr(bar, "c", None)),
        "Volume":    getattr(bar, "volume", getattr(bar, "v", 0)) or 0,
    }

    @staticmethod
    def _bar_bucket(ts: datetime, timeframe_sec: int = 60) -> int:
        return int(ts.timestamp() // timeframe_sec)

    def _df_from_history(self, symbol: str) -> pd.DataFrame:
        rows = self.history[symbol][-300:]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _symbol_mv(self, symbol: str, last_px: float) -> float:
        pos = self.portfolio.positions.get(symbol)
        return (pos.qty * last_px) if pos else 0.0

    async def _bar_debug_logger(self, payload: dict):
        try:
            self.logger.debug(
                f"[EVENT BAR] {payload['symbol']} {payload['timestamp']} "
                f"o={payload['open']} h={payload['high']} l={payload['low']} "
                f"c={payload['close']} v={payload['volume']}"
            )
        except Exception:
            pass
    # ---------- seeding ----------
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

    # ---------- bar callback ----------
    async def on_alpaca_bar(self, raw_bar):
        bar = self._canonicalize_alpaca_bar(raw_bar)
        self.logger.debug(f"[RAW BAR] {raw_bar.symbol} {raw_bar.timestamp} c={raw_bar.close}")
        symbol = bar["symbol"]
        ts: datetime = bar["timestamp"]
        
        await self.event_handler.emit("BAR", {
            "timestamp": ts,
            "symbol": symbol,
            "open":  float(bar["Open"]),
            "high":  float(bar["High"]),
            "low":   float(bar["Low"]),
            "close": float(bar["Close"]),
            "volume": int(bar.get("Volume", 0)),
        })
        bar_id = self._bar_bucket(ts)
        prev_bar_id = self._last_bar_id.get(symbol)
        bar_closed = (prev_bar_id is None) or (bar_id != prev_bar_id)
        self._last_bar_id[symbol] = bar_id

        # track history + MTM (for charts/telemetry)
        self.history[symbol].append(bar)
        last_px = float(bar["Close"])
        self.portfolio.update_price(symbol, last_px)  # mark-to-market

        #state = self.symbol_state[symbol]
        state = self.symbol_state.setdefault(symbol, SymbolState(symbol=symbol))
        state.portfolio_value = self.portfolio.total_equity()
        state.ts = ts
        state.bar_id = bar_id
        state.bar_closed = bar_closed

        # indicators & regime
        df = self._df_from_history(symbol)
        atr = compute_atr(df, period=14)
        if atr is not None:
            self.atr_hist[symbol].append(atr)
        regime = classify_regime(atr, self.atr_hist[symbol])

        # drawdown monitor (daily tick + updates)
        if self._last_ddm_date is None or ts.date() != self._last_ddm_date:
            self.ddm.start_new_day(portfolio_equity=self.portfolio.total_equity())
            self._last_ddm_date = ts.date()
        self.ddm.update_portfolio(self.portfolio.total_equity())
        self.ddm.update_symbol(symbol, self._symbol_mv(symbol, last_px))
        await self.event_handler.emit(EVENT_PNL_UPDATE, PnLPayload(
            portfolio_value=self.portfolio.total_equity(),
            equity_curve=self.portfolio.equity_history,
            unrealized=self.portfolio.total_unrealized(),
            realized=self.portfolio.realized_pnl,
            drawdown=self.ddm.get_portfolio_drawdown(),
            timestamp=ts.isoformat(),
        ))

        # strategy & signal
        strategy = self.router.get_strategy(symbol, regime)
        strategy_name = type(strategy).__name__
        try:
            raw_signal = strategy.generate_signal(df)
            signal = int(raw_signal if isinstance(raw_signal, (int, float)) else getattr(raw_signal, "signal", 0))
            await self.event_handler.emit(EVENT_STRATEGY_SIGNAL, StrategySignalPayload(
                symbol=symbol,
                strategy=strategy_name,
                signal={-1: "sell", 0: "hold", 1: "buy"}.get(signal, "hold"),
                confidence=None,  # you could calculate model confidence here
                timestamp=ts.isoformat(),
            ))

        except Exception as e:
            self.logger.exception(f"[{symbol}] Strategy error in {strategy_name}: {e}")
            signal = 0

        # trade gate context for this bar
        self.trade_gate.on_new_bar(symbol, bar_id, regime)
        gs = self.trade_gate.get_state(symbol)
        state.regime = regime
        state.regime_persist = gs.regime_persist

        # hand off to engine (engine should enforce gates; runner is context-only)
        await self.engine.handle_signal(
            symbol=symbol,
            state=state,
            signal=signal,
            price=last_px,
            atr=float(atr or 0.0),
            regime=regime,
            strategy_name=strategy_name,
        )

        # telemetry
        pos = self.portfolio.positions.get(symbol)
        qty = pos.qty if pos else 0
        self.logger.debug(
            f"[{symbol}] bar={bar_id} closed={bar_closed} regime={regime} "
            f"persist={gs.regime_persist} qty={qty} equity={self.portfolio.total_equity():.2f}"
        )
        await self.event_handler.emit(EVENT_NEW_BAR, BarPayload(
            symbol=symbol,
            open=float(bar["Open"]),
            high=float(bar["High"]),
            low=float(bar["Low"]),
            close=float(bar["Close"]),
            volume=int(bar.get("Volume", 0)),
            timestamp=ts.isoformat(),
        ))

    # ---------- run ----------
    async def run(self):
        self._running = True

        # Seed historical data (fetches fresh if stale)
        max_stale = self.settings.get("MAX_STALE_MINUTES", 60)
        await self.seed(self.settings.get("SEED_BARS", 200), max_stale_minutes=max_stale)
        await self.event_handler.subscribe("BAR", self._bar_debug_logger)

        # connect + subscribe
        self.broker.api_key = os.getenv("ALPACA_API_KEY")
        self.broker.api_secret = os.getenv("ALPACA_SECRET_KEY")
        self.broker.connect()

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

        for sym in self.symbols:
            self.broker.subscribe_bars(self.on_alpaca_bar, sym)

        stream_task = asyncio.create_task(self.broker.start_stream())
        self.logger.info(f"LiveRunner (Alpaca) started for: {', '.join(self.symbols)}")

        # Start periodic state reconciliation
        reconcile_interval = self.settings.get("RECONCILE_INTERVAL", 60)
        if reconcile_interval > 0:
            await self.reconciler.start_periodic(interval_seconds=reconcile_interval)
            self.logger.info(f"State reconciliation enabled (every {reconcile_interval}s)")

        # Start periodic historical data updates (runs in background)
        update_interval = self.settings.get("HISTORICAL_UPDATE_INTERVAL_MINUTES", 60)
        if update_interval > 0:
            self._update_task = asyncio.create_task(
                self._periodic_data_update(interval_minutes=update_interval)
            )
            self.logger.info(f"Periodic data updates enabled (every {update_interval} min)")

        try:
            while self._running:
                # Check if reconciler halted trading
                if self.reconciler.is_halted:
                    self.logger.critical("Trading halted by reconciler - exiting run loop")
                    break
                await asyncio.sleep(0.5)
        finally:
            self._running = False
            # Stop reconciler
            await self.reconciler.stop_periodic()
            self.logger.info("State reconciler stopped")

            # Cancel background tasks
            if self._update_task:
                self._update_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._update_task
            stream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stream_task

            # CRITICAL: Disconnect broker to release websocket connection
            self.broker.disconnect()

            self.trade_logger.flush()
            self.logger.info("LiveRunner shut down cleanly.")

    def stop(self):
        """Stop the live trading runner."""
        self._running = False
        # Disconnect broker to release websocket
        if hasattr(self.broker, 'disconnect'):
            self.broker.disconnect()
        self.logger.info("Stop requested")

    async def _periodic_data_update(self, interval_minutes: int = 60):
        """
        Background task to periodically update historical data.

        Uses the unified pipeline which:
        - Auto-selects best data source
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


# -------- entrypoint --------
# ----- defaults written only if missing -----
def _ensure_live_config(dir_path: str = "config"):
    os.makedirs(dir_path, exist_ok=True)

    sr_path = os.path.join(dir_path, "strategy_routing.json")
    if not os.path.exists(sr_path):
        with open(sr_path, "w") as f:
            json.dump({
                "AAPL": {
                    "low_volatility":  "sma_strategy",
                    "normal":          "momentum_strategy",
                    "high_volatility": "mean_reversion_strategy"
                },
                "MSFT": {
                    "low_volatility":  "sma_strategy",
                    "normal":          "momentum_strategy",
                    "high_volatility": "mean_reversion_strategy"
                }
            }, f, indent=2)

    sp_path = os.path.join(dir_path, "strategy_params.json")
    if not os.path.exists(sp_path):
        with open(sp_path, "w") as f:
            json.dump({
                "AAPL": {
                    "low_volatility": {"params": {"fast": 10, "slow": 30}},
                    "normal":         {"params": {"lookback": 20}},
                    "high_volatility":{"params": {"window": 14}}
                },
                "MSFT": {
                    "low_volatility": {"params": {"fast": 10, "slow": 30}},
                    "normal":         {"params": {"lookback": 20}},
                    "high_volatility":{"params": {"window": 14}}
                }
            }, f, indent=2)

    tl_path = os.path.join(dir_path, "trade_logic_routing.json")
    if not os.path.exists(tl_path):
        with open(tl_path, "w") as f:
            json.dump({
                "AAPL": {
                    "low_volatility":  {"trade_logic_class": "default", "params": {}},
                    "normal":          {"trade_logic_class": "default", "params": {}},
                    "high_volatility": {"trade_logic_class": "default", "params": {}}
                },
                "MSFT": {
                    "low_volatility":  {"trade_logic_class": "default", "params": {}},
                    "normal":          {"trade_logic_class": "default", "params": {}},
                    "high_volatility": {"trade_logic_class": "default", "params": {}}
                }
            }, f, indent=2)

    return sr_path, sp_path, tl_path

async def main():
    # 1) ensure config JSONs exist (flat under ./config)
    sr_path, sp_path, tl_path = _ensure_live_config("config")

    # 2) load settings; merge ALL *.json/*.yml directly in ./config
    settings = Settings(
        root="config",
        include_root=True,  # <- important for your flat layout
        # optional: env="dev",
        runtime_overrides={
            # expose your three files as first-class keys
            "strategy_routing_path": sr_path,
            "strategy_params_path": sp_path,
            "trade_logic_routing_path": tl_path,
        },
    )

    # 3) symbols – keep your existing key, with a sane default
    symbols = settings.get_list("symbols") or settings.get_list("SYMBOLS") or ["AAPL", "MSFT"]

    runner = AlpacaLiveRunner(settings, symbols)
    await runner.run()

if __name__ == "__main__":
    asyncio.run(main())