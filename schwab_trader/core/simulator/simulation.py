"""
Production-Ready Simulation Runner
All critical issues fixed with proper error handling and state management
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asyncio
import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Deque, Any, Optional, Union
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from loggers.logger import Logger
from core.events.eventhandler import EventHandler, get_event_handler
from core.simulator.gbm_simulator import GBMSimulator
from core.logic.strategy_routing_manager import StrategyRoutingManager
from core.simulator.strategy_router import StrategyRouter
from core.logic.trade_logic_manager import DynamicTradeLogicManager
from core.logic.mock_execution_engine import MockExecutionEngine
from core.logic.symbol_state import SymbolState
from core.logic.portfolio_state import PortfolioState
from core.position_sizer import DynamicPositionSizer2
from loggers.file_trade_logger import FileTradeLogger
from core.historical_loader import HistoricalBarLoader
from core.drawdown_monitor import DrawdownMonitor
from core.logic.trade_gate import TradeGate
from core.broker.mock_broker import MockBroker
from utils.replay import replay_equity_from_trades

from core.events.events import (
    EVENT_NEW_BAR,
    EVENT_STRATEGY_SIGNAL,
    EVENT_PNL_UPDATE,
    PnLPayload,
    BarPayload,
    StrategySignalPayload,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SimConfig:
    """Simulation configuration with sensible defaults"""
    
    # Core settings
    symbols: list[str]
    steps: int = 600
    bar_sleep: float = 0.05
    starting_cash: float = 100_000.0
    
    # Paths (relative to project root)
    strategy_routing: str = "config/strategy_routing.json"
    strategy_params: str = "config/strategy_params.json"
    trade_logic_routing: str = "config/trade_logic_routing.json"
    historical_data_path: str = "data/data_storage/proc_data"
    
    # ✅ FIXED: Separate file and directory for trade logs
    trade_log_file: str = "trades_sim.csv"
    trade_log_dir: str = "logs"
    
    # Position sizing
    risk_percentage: float = 0.15
    max_trade_pct: float = 0.05
    max_holding_pct: float = 0.10
    
    # Drawdown limits (applied consistently)
    max_symbol_drawdown: float = 0.30
    max_symbol_daily_drawdown: float = 0.10
    symbol_cooldown_seconds: int = 20
    max_portfolio_drawdown: float = 0.25
    max_portfolio_daily_drawdown: float = 0.10
    portfolio_cooldown_seconds: int = 60
    
    # Trade gates
    max_pyramid_layers: int = 3
    min_bars_between_layers: int = 2
    regime_min_persist_bars: int = 2
    flip_cooldown_bars: int = 1
    
    # History buffers
    max_history_bars: int = 500
    max_atr_history: int = 300
    atr_period: int = 14
    warmup_bars: int = 200
    
    def __post_init__(self):
        """Validate configuration and ensure directories exist"""
        if not self.symbols:
            raise ValueError("symbols list cannot be empty")
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        
        # Ensure all directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create all necessary directories"""
        dirs_to_create = [
            Path(self.trade_log_dir),  # ✅ Just the directory
            Path(self.strategy_routing).parent,
            Path(self.strategy_params).parent,
            Path(self.trade_logic_routing).parent,
        ]
        
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def trade_log_path(self) -> str:
        """Get full trade log path"""
        return os.path.join(self.trade_log_dir, self.trade_log_file)


# ============================================================================
# BAR UTILITIES
# ============================================================================

class BarProcessor:
    """Utilities for processing and validating bars"""
    
    @staticmethod
    def to_canonical_bar(raw: dict) -> dict:
        """
        Convert raw bar dict to canonical format with validation.
        
        Raises:
            ValueError: If required fields are missing or invalid
        """
        try:
            m = {k.lower(): v for k, v in raw.items()}
            
            # Extract timestamp
            ts = m.get("timestamp", m.get("date"))
            if ts is None:
                raise ValueError("bar missing timestamp/date")
            
            if isinstance(ts, (int, float)):
                ts = pd.to_datetime(ts, unit="ms", utc=True)
            else:
                ts = pd.to_datetime(ts, utc=True)
            
            # Extract OHLCV with validation
            def get_float(key: str, alt: Optional[str] = None) -> float:
                v = m.get(key, m.get(alt) if alt else None)
                if v is None:
                    raise ValueError(f"bar missing '{key}'")
                return float(v)
            
            return {
                "timestamp": ts,
                "symbol": str(m["symbol"]),
                "Open": get_float("open"),
                "High": get_float("high"),
                "Low": get_float("low"),
                "Close": get_float("close"),
                "Volume": int(m.get("volume", 0)),
            }
        
        except Exception as e:
            raise ValueError(f"Invalid bar data: {e}") from e


# ============================================================================
# INDICATOR CALCULATIONS
# ============================================================================

class Indicators:
    """Reusable indicator calculations"""
    
    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """
        Compute Average True Range.
        
        Returns:
            ATR value or None if insufficient data
        """
        try:
            if df.shape[0] < period + 1:
                return None
            
            high = df["High"].astype(float)
            low = df["Low"].astype(float)
            close = df["Close"].astype(float)
            prev_close = close.shift(1)
            
            tr = pd.concat([
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)
            
            atr = tr.rolling(period).mean()
            v = atr.iloc[-1]
            
            return float(v) if pd.notna(v) else None
        
        except Exception as e:
            logging.error(f"ATR computation failed: {e}")
            return None
    
    @staticmethod
    def classify_regime(atr_value: Optional[float], 
                       atr_window: Deque[float]) -> str:
        """
        Classify market regime by ATR quantiles.
        
        Returns:
            'low_volatility', 'normal', or 'high_volatility'
        """
        if atr_value is None or len(atr_window) < 10:
            return "normal"
        
        try:
            s = pd.Series(list(atr_window))
            q25 = s.quantile(0.25)
            q75 = s.quantile(0.75)
            
            if atr_value < q25:
                return "low_volatility"
            elif atr_value > q75:
                return "high_volatility"
            else:
                return "normal"
        
        except Exception:
            return "normal"


# ============================================================================
# SIGNAL EXTRACTION
# ============================================================================

class SignalExtractor:
    """Robust signal extraction from various strategy outputs"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def extract(self, sig: Any) -> int:
        """
        Normalize various strategy outputs to {-1, 0, 1}.
        
        Handles:
        - int/float/bool
        - dict with 'signal' key
        - pd.Series
        - pd.DataFrame
        
        Returns:
            int in {-1, 0, 1}
        """
        try:
            if sig is None:
                return 0
            
            # Simple numeric types
            if isinstance(sig, (int, np.integer)):
                return int(np.clip(sig, -1, 1))
            
            if isinstance(sig, (float, np.floating)):
                if np.isnan(sig):
                    return 0
                return int(np.clip(np.sign(sig), -1, 1))
            
            if isinstance(sig, bool):
                return 1 if sig else 0
            
            # Dictionary
            if isinstance(sig, dict):
                for key in ("signal", "Signal", "trade_signal"):
                    if key in sig:
                        return self.extract(sig[key])
                self.logger.warning("Dict signal missing recognized keys")
                return 0
            
            # Pandas Series
            if isinstance(sig, pd.Series):
                if len(sig) == 0:
                    return 0
                return self.extract(sig.iloc[-1])
            
            # Pandas DataFrame
            if isinstance(sig, pd.DataFrame):
                if sig.empty:
                    return 0
                
                for col in ("signal", "Signal", "Position", "position", "trade_signal"):
                    if col in sig.columns:
                        return self.extract(sig[col].iloc[-1])
                
                self.logger.warning("DataFrame signal missing recognized columns")
                return 0
            
            # Unknown type
            self.logger.warning(f"Unknown signal type: {type(sig)}")
            return 0
        
        except Exception as e:
            self.logger.error(f"Signal extraction failed: {e}")
            return 0


# ============================================================================
# SIMULATION RUNNER
# ============================================================================

class SimulationRunner:
    """
    Production-ready simulation runner with:
    - Consistent configuration usage
    - Proper error handling
    - Resource cleanup
    - Thread-safe state management
    """
    
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.logger = Logger(
            "simulation.log", 
            self.__class__.__name__
        ).get_logger()
        
        # Core state
        self.portfolio = PortfolioState(cash=cfg.starting_cash)
        self.symbol_state: Dict[str, SymbolState] = {
            s: SymbolState(symbol=s) for s in cfg.symbols
        }
        
        # History buffers
        self.history: Dict[str, Deque[dict]] = {
            s: deque(maxlen=cfg.max_history_bars) for s in cfg.symbols
        }
        self.atr_hist: Dict[str, Deque[float]] = {
            s: deque(maxlen=cfg.max_atr_history) for s in cfg.symbols
        }
        
        # Bar tracking
        self._last_bar_id: Dict[str, int] = {}
        self._last_ddm_date: Optional[datetime] = None
        
        # Utilities
        self.bar_processor = BarProcessor()
        self.indicators = Indicators()
        self.signal_extractor = SignalExtractor(self.logger)
        
        # Initialize components
        self._init_simulator()
        self._init_routing()
        self._init_risk_management()
        self._init_execution()
        
        self.logger.info(f"SimulationRunner initialized for {cfg.symbols}")
    
    def _init_simulator(self):
        """Initialize price simulator"""
        self.sim = GBMSimulator(
            self.cfg.symbols,
            base_price=300.0,
            log_prices=False
        )
    
    def _init_routing(self):
        """Initialize strategy routing"""
        self.strategy_routing = StrategyRoutingManager(
            self.cfg.strategy_routing
        )
        self.strategy_router = StrategyRouter(
            self.strategy_routing,
            self.cfg.strategy_params
        )
        self.trade_logic_manager = DynamicTradeLogicManager(
            self.cfg.trade_logic_routing
        )
    
    def _init_risk_management(self):
        """Initialize risk management components"""
        # ✅ Drawdown monitor (using config values consistently!)
        self.ddm = DrawdownMonitor(
            max_symbol_drawdown=self.cfg.max_symbol_drawdown,
            max_symbol_daily_drawdown=self.cfg.max_symbol_daily_drawdown,
            symbol_cooldown_seconds=self.cfg.symbol_cooldown_seconds,
            max_portfolio_drawdown=self.cfg.max_portfolio_drawdown,
            max_portfolio_daily_drawdown=self.cfg.max_portfolio_daily_drawdown,
            portfolio_cooldown_seconds=self.cfg.portfolio_cooldown_seconds,
        )
        
        # Trade gates
        self.trade_gate = TradeGate(
            max_layers=self.cfg.max_pyramid_layers,
            min_bars_between_layers=self.cfg.min_bars_between_layers,
            regime_min_persist_bars=self.cfg.regime_min_persist_bars,
            flip_cooldown_bars=self.cfg.flip_cooldown_bars,
        )
    
    def _init_execution(self):
        """Initialize execution components"""
        # Event handler
        self.events = get_event_handler()
        
        # Broker
        self.broker = MockBroker(
            starting_cash=self.cfg.starting_cash,
            event_handler=self.events
        )
        
        # Position sizer
        self.sizer = DynamicPositionSizer2(
            risk_percentage=self.cfg.risk_percentage,
            max_trade_pct=self.cfg.max_trade_pct,
            max_holding_pct=self.cfg.max_holding_pct
        )
        
        # ✅ Trade logger - Pass filename and directory separately
        self.trade_logger = FileTradeLogger(
            log_file=self.cfg.trade_log_file,  # Just "trades_sim.csv"
            log_dir=self.cfg.trade_log_dir      # Just "logs"
        )
        
        # Historical data loader
        self.loader = HistoricalBarLoader(
            path=self.cfg.historical_data_path
        )
        
        # Execution engine
        self.engine = MockExecutionEngine(
            broker=self.broker,
            sizer=self.sizer,
            performance_tracker=self.trade_logger,
            trade_logic_manager=self.trade_logic_manager,
            drawdown_monitor=self.ddm,
            portfolio=self.portfolio,
            event_handler=self.events
        )
        self.engine.symbolstates = self.symbol_state
    
    # ========================================================================
    # BAR PROCESSING
    # ========================================================================
    
    def _bar_bucket(self, ts: datetime, timeframe_sec: int = 60) -> int:
        """Convert timestamp to bar bucket ID"""
        return int(ts.timestamp() // timeframe_sec)
    
    def _df_from_history(self, symbol: str) -> pd.DataFrame:
        """Build DataFrame from history buffer"""
        try:
            if not self.history[symbol]:
                return pd.DataFrame()
            
            df = pd.DataFrame(list(self.history[symbol]))
            df = df.set_index(pd.to_datetime(df["timestamp"]), drop=True)
            return df[["Open", "High", "Low", "Close", "Volume"]].copy()
        
        except Exception as e:
            self.logger.error(f"DataFrame construction failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def _symbol_mv(self, symbol: str, last_price: float) -> float:
        """Calculate symbol market value"""
        try:
            pos = self.portfolio.positions.get(symbol)
            if not pos:
                return 0.0
            return float(pos.qty * last_price)
        except Exception as e:
            self.logger.error(f"Market value calculation failed for {symbol}: {e}")
            return 0.0
    
    async def _on_bar(self, bar: dict) -> None:
        """
        Process a single bar update.
        
        This is the main event handler called for each bar.
        ✅ Fixed: Removed duplicate MTM and DDM updates.
        """
        try:
            # Validate and canonicalize bar
            bar = self.bar_processor.to_canonical_bar(bar)
            symbol = bar["symbol"]
            ts: datetime = bar["timestamp"]
            
            # Ensure UTC timezone
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            
            # Bar bucket tracking
            bar_id = self._bar_bucket(ts)
            prev_bar_id = self._last_bar_id.get(symbol)
            bar_closed = (prev_bar_id is not None and bar_id != prev_bar_id)
            self._last_bar_id[symbol] = bar_id
            
            # Reset reservations on bar close
            if bar_closed:
                self.sizer.reset_bar_reservations()
            
            # Update history
            self.history[symbol].append(bar)
            
            # ✅ MTM update (ONCE!)
            price = float(bar["Close"])
            self.portfolio.update_price(symbol, price)
            
            # Update state
            state = self.symbol_state[symbol]
            state.portfolio_value = self.portfolio.total_equity()
            state.ts = ts
            state.bar_id = bar_id
            state.bar_closed = bar_closed
            
            # Build DataFrame
            df = self._df_from_history(symbol)
            if df.empty:
                self.logger.warning(f"{symbol}: Empty DataFrame, skipping")
                return
            
            # Compute indicators
            atr = self.indicators.compute_atr(df, period=self.cfg.atr_period)
            if atr is not None:
                self.atr_hist[symbol].append(atr)
            
            regime = self.indicators.classify_regime(atr, self.atr_hist[symbol])
            
            # ✅ Drawdown monitor updates (ONCE per symbol!)
            if self._last_ddm_date is None or ts.date() != self._last_ddm_date:
                self.ddm.start_new_day(
                    portfolio_equity=self.portfolio.total_equity()
                )
                self._last_ddm_date = ts.date()
            
            equity = self.portfolio.total_equity()
            self.ddm.update_portfolio(equity)
            
            sym_mv = self._symbol_mv(symbol, price)
            self.ddm.update_symbol(symbol, sym_mv)  # ONCE!
            
            # Get strategy and generate signal
            strategy = self.strategy_router.get_strategy(symbol, regime)
            strategy_name = type(strategy).__name__
            
            try:
                raw_signal = strategy.generate_signal(df)
                signal = self.signal_extractor.extract(raw_signal)
            except Exception as e:
                self.logger.exception(
                    f"[{symbol}] Strategy error in {strategy_name}: {e}"
                )
                signal = 0
            
            # Emit signal event
            payload: StrategySignalPayload = {
                "symbol": symbol,
                "timestamp": ts.isoformat(),
                "signal": signal,
                "strategy": strategy_name,
                "confidence": 0.0,
                "atr": atr,
                "regime": regime,
            }
            await self.events.publish(EVENT_STRATEGY_SIGNAL, payload)
            
            # Update trade gates
            self.trade_gate.on_new_bar(symbol, bar_id, regime)
            gate_state = self.trade_gate.get_state(symbol)
            
            # Determine position side
            pos = self.portfolio.positions.get(symbol)
            cur_qty = pos.qty if pos else 0
            cur_side = (
                "long" if cur_qty > 0 else
                "short" if cur_qty < 0 else
                None
            )
            
            # Gate flags (only entries require bar_closed)
            can_enter_long = (
                bar_closed and 
                self.trade_gate.can_enter(
                    symbol, "long", ts, bar_id, regime, allow_pyramiding=False
                )[0]
            )
            can_enter_short = (
                bar_closed and 
                self.trade_gate.can_enter(
                    symbol, "short", ts, bar_id, regime, allow_pyramiding=False
                )[0]
            )
            can_pyr_long = (
                bar_closed and 
                self.trade_gate.can_enter(
                    symbol, "long", ts, bar_id, regime, allow_pyramiding=True
                )[0]
            )
            can_pyr_short = (
                bar_closed and 
                self.trade_gate.can_enter(
                    symbol, "short", ts, bar_id, regime, allow_pyramiding=True
                )[0]
            )
            
            # Update state with gate info
            state.regime = regime
            state.regime_persist = gate_state.regime_persist if gate_state else 0
            state.cur_side = cur_side
            state.can_enter_long = can_enter_long
            state.can_enter_short = can_enter_short
            state.can_pyramid_long = can_pyr_long
            state.can_pyramid_short = can_pyr_short
            
            # Pass to execution engine
            self.engine.handle_signal(
                symbol=symbol,
                state=state,
                signal=signal,
                price=price,
                atr=atr if atr is not None else 0.0,
                regime=regime,
                strategy_name=strategy_name,
            )
        
        except Exception as e:
            self.logger.exception(f"Bar processing failed: {e}")
    
    # ========================================================================
    # SIMULATION LOOP
    # ========================================================================
    
    async def _bar_producer(self) -> None:
        """Generate and emit bars"""
        try:
            await self.events.start()
            
            for step in range(self.cfg.steps):
                all_bars = self.sim.update_all()
                
                for bar in all_bars.values():
                    try:
                        canonical = self.bar_processor.to_canonical_bar(bar)
                        payload: BarPayload = {
                            "symbol": canonical["symbol"],
                            "timestamp": canonical["timestamp"].isoformat(),
                            "open": canonical["Open"],
                            "high": canonical["High"],
                            "low": canonical["Low"],
                            "close": canonical["Close"],
                            "volume": canonical["Volume"],
                        }
                        await self.events.publish(EVENT_NEW_BAR, payload)
                    except Exception as e:
                        self.logger.error(f"Bar emission failed: {e}")
                
                await asyncio.sleep(self.cfg.bar_sleep)
                
                # Yield control periodically
                if step % 5 == 0:
                    await asyncio.sleep(0)
            
            self.logger.info("Bar production complete")
        
        except Exception as e:
            self.logger.exception(f"Bar producer failed: {e}")
    
    async def _bar_consumer(self) -> None:
        """Subscribe to and process bars"""
        try:
            # Emit initial PnL
            pnl_payload: PnLPayload = {
                "portfolio_value": self.portfolio.total_equity(),
                "realized": self.portfolio.realized_pnl,
                "unrealized": self.portfolio.unrealized_pnl,
                "drawdown": self.portfolio.drawdown,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await self.events.emit(EVENT_PNL_UPDATE, pnl_payload)
            
            # Subscribe to bars
            async def bar_handler(event):
                await self._on_bar(event.payload)
            
            await self.events.subscribe(EVENT_NEW_BAR, bar_handler)
            
        except Exception as e:
            self.logger.exception(f"Bar consumer failed: {e}")
    
    def _seed(self, warmup: int = None):
        """
        Seed history with warm-up data.
        Thread-safe initialization before async loop starts.
        """
        if warmup is None:
            warmup = self.cfg.warmup_bars
        
        try:
            for symbol in self.cfg.symbols:
                # Try loading historical data
                bars = self.loader.load_last_n_bars(symbol, n=warmup)
                
                if not bars:
                    # Fallback: generate synthetic bars
                    self.logger.warning(
                        f"{symbol}: No historical data, using synthetic"
                    )
                    for _ in range(warmup):
                        self.history[symbol].append(
                            self.sim.generate_bar(symbol)
                        )
                else:
                    self.history[symbol].extend(bars)
                
                # Compute initial ATR
                df = self._df_from_history(symbol)
                atr = self.indicators.compute_atr(df, self.cfg.atr_period)
                if atr is not None:
                    self.atr_hist[symbol].append(atr)
            
            self.logger.info(f"Seeded {len(self.cfg.symbols)} symbols with {warmup} bars")
        
        except Exception as e:
            self.logger.exception(f"Seeding failed: {e}")
    
    async def run(self) -> None:
        """
        Main simulation loop with proper cleanup.
        """
        try:
            self.logger.info(
                f"Starting simulation: {self.cfg.symbols} | "
                f"steps={self.cfg.steps}"
            )
            
            # Seed historical data
            self._seed()
            
            # Subscribe engine to signals
            await self.engine.subscribe_signals()
            
            # Run producer and consumer concurrently
            await asyncio.gather(
                self._bar_consumer(),
                self._bar_producer(),
            )
            
            # Log final state
            self.logger.info(
                f"Simulation complete. "
                f"Final equity: ${self.portfolio.total_equity():,.2f} | "
                f"Unrealized: ${self.portfolio.total_unrealized():,.2f}"
            )
        
        except Exception as e:
            self.logger.exception(f"Simulation failed: {e}")
        
        finally:
            # ✅ Cleanup resources
            try:
                self.trade_logger.flush()
                await self.events.stop()
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")


# ============================================================================
# CONFIG FILE GENERATOR
# ============================================================================

def ensure_config_files(config_dir: str = "config") -> dict:
    """
    Create default config files if they don't exist.
    
    Returns:
        dict with config file paths
    """
    os.makedirs(config_dir, exist_ok=True)
    
    paths = {
        "strategy_routing": f"{config_dir}/strategy_routing.json",
        "strategy_params": f"{config_dir}/strategy_params.json",
        "trade_logic_routing": f"{config_dir}/trade_logic_routing.json",
    }
    
    # Strategy routing
    if not os.path.exists(paths["strategy_routing"]):
        with open(paths["strategy_routing"], "w") as f:
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
    
    # Strategy params
    if not os.path.exists(paths["strategy_params"]):
        with open(paths["strategy_params"], "w") as f:
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
    
    # Trade logic routing
    if not os.path.exists(paths["trade_logic_routing"]):
        with open(paths["trade_logic_routing"], "w") as f:
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
    
    return paths


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """CLI entry point with comprehensive error handling"""
    try:
        # Ensure config files exist
        config_paths = ensure_config_files()
        
        # Create configuration
        cfg = SimConfig(
            symbols=["AAPL", "MSFT"],
            steps=2000,
            bar_sleep=0.05,
            starting_cash=100_000.0,
            **config_paths
        )
        
        # Run simulation
        runner = SimulationRunner(cfg)
        await runner.run()
        
        # Replay equity from trades
        last_prices = {}
        for sym in cfg.symbols:
            if runner.history[sym]:
                last_prices[sym] = runner.history[sym][-1]["Close"]
        
        if last_prices:
            try:
                snap = replay_equity_from_trades(
                    csv_path=cfg.trade_log_path,  # ✅ Uses property that builds full path
                    starting_cash=cfg.starting_cash,
                    final_prices=last_prices
                )
                
                print("\n" + "="*60)
                print("📊 SIMULATION RESULTS")
                print("="*60)
                print(f"Final Equity:    ${snap['equity']:>15,.2f}")
                print(f"Cash:            ${snap['cash']:>15,.2f}")
                print(f"Total Positions: {len(snap['positions']):>15}")
                print("\nPosition Details:")
                for sym, pos in snap['positions'].items():
                    print(f"  {sym}: {pos['qty']} shares @ ${pos['avg_price']:.2f}")
                print("="*60)
                
            except Exception as e:
                logging.error(f"Replay failed: {e}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Simulation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())