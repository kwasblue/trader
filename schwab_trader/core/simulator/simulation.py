"""
Updated Simulation Runner - Uses Improved Architecture

Works with:
- Improved base classes
- Proper execution engine orchestration
- Event-driven architecture
- Clean separation of concerns
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
from dataclasses import dataclass
from typing import Dict, Deque, Any, Optional
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from loggers.logger import Logger
from loggers.file_trade_logger import FileTradeLogger
from indicators.technical_indicators import TechnicalIndicators
from core.broker.mock_broker import MockBroker
from core.mock_executor import MockExecutor
from core.position_sizer import KellyPositionSizer
from core.events.eventhandler import EventHandler
from core.simulator.gbm_simulator import GBMSimulator
from core.logic.strategy_routing_manager import StrategyRoutingManager
from core.logic.trade_logic_manager import DynamicTradeLogicManager
from core.logic.mock_execution_engine import MockExecutionEngine
from core.logic.symbol_state import SymbolState
from core.logic.portfolio_state import PortfolioState
from core.drawdown_monitor import DrawdownMonitor
from core.enums import OrderSide
from core.contracts.types import SignalContext

from core.contracts.events import (
    EVENT_NEW_BAR,
    EVENT_STRATEGY_SIGNAL,
    EVENT_PNL_UPDATE,
    EVENT_REGIME_UPDATE,
    EVENT_HALTED,
    PnLPayload,
    BarPayload,
    StrategySignalPayload,
    RegimePayload,
)
from core.config_loader import get_config


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """
    Compute the Average True Range (ATR) from OHLC data.

    Args:
        df: DataFrame with 'High', 'Low', 'Close' columns
        period: ATR lookback period (default: 14)

    Returns:
        The latest ATR value, or None if insufficient data
    """
    if df.empty or len(df) < period + 1:
        return None

    # Get column names (handle both 'High' and 'high' conventions)
    high_col = 'High' if 'High' in df.columns else 'high'
    low_col = 'Low' if 'Low' in df.columns else 'low'
    close_col = 'Close' if 'Close' in df.columns else 'close'

    high = df[high_col].values
    low = df[low_col].values
    close = df[close_col].values

    # Calculate True Range components
    tr1 = high[1:] - low[1:]  # High - Low
    tr2 = np.abs(high[1:] - close[:-1])  # |High - Prev Close|
    tr3 = np.abs(low[1:] - close[:-1])   # |Low - Prev Close|

    # True Range is max of the three
    tr = np.maximum(np.maximum(tr1, tr2), tr3)

    if len(tr) < period:
        return None

    # Simple moving average of TR for ATR
    atr = np.mean(tr[-period:])
    return float(atr)


def classify_regime(atr: Optional[float], atr_history: list) -> str:
    """
    Classify market regime based on ATR relative to historical values.

    Args:
        atr: Current ATR value
        atr_history: List of historical ATR values

    Returns:
        Regime string: 'low_volatility', 'normal', or 'high_volatility'
    """
    if atr is None or len(atr_history) < 20:
        return "normal"  # Default when insufficient data

    # Calculate dynamic thresholds from history
    recent_atr = list(atr_history)[-50:] if len(atr_history) >= 50 else list(atr_history)
    atr_mean = np.mean(recent_atr)
    atr_std = np.std(recent_atr) if len(recent_atr) > 1 else 0.0

    high_threshold = atr_mean + atr_std
    low_threshold = atr_mean - 0.5 * atr_std

    if atr > high_threshold:
        return "high_volatility"
    elif atr < low_threshold:
        return "low_volatility"
    else:
        return "normal"


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SimConfig:
    """Simulation configuration"""

    # Core settings
    symbols: list[str]
    steps: int = 600
    bar_sleep: float = 0.05
    starting_cash: float = 100_000.0

    # Paths
    strategy_routing: str = "config/strategy_routing.json"
    trade_logic_routing: str = "config/trade_logic_routing.json"
    historical_data_path: str = "data/data_storage/proc_data"
    trade_log_file: str = "trades_sim.csv"
    trade_log_dir: str = "logs"

    # Position sizing
    risk_per_trade: float = 0.01  # 1% risk per trade
    max_trade_pct: float = 0.10   # Max 10% of portfolio per trade
    max_holding_pct: float = 0.25  # Max 25% in any single position

    # Drawdown limits (intraday = from peak, daily = from day start)
    drawdown_monitor_enabled: bool = False      # Enable/disable drawdown monitor
    max_symbol_drawdown: float = 0.30           # 30% symbol intraday
    max_symbol_daily_drawdown: float = 0.15     # 15% symbol daily
    max_portfolio_drawdown: float = 0.25        # 25% portfolio intraday
    max_portfolio_daily_drawdown: float = 0.10  # 10% portfolio daily

    # History buffers
    max_history_bars: int = 500
    max_atr_history: int = 300
    atr_period: int = 14
    warmup_bars: int = 200

    @classmethod
    def from_config_file(cls, symbols: list[str] = None) -> "SimConfig":
        """
        Create SimConfig from the global trading_config.json file.

        Args:
            symbols: Override symbols (if None, uses config default)

        Returns:
            SimConfig populated from config file
        """
        cfg = get_config()

        return cls(
            symbols=symbols or cfg.general.default_symbols,
            steps=cfg.simulation.steps,
            bar_sleep=cfg.simulation.bar_sleep,
            starting_cash=cfg.simulation.starting_cash,
            risk_per_trade=cfg.risk.risk_per_trade,
            max_trade_pct=cfg.position_sizer.max_trade_pct,
            max_holding_pct=cfg.position_sizer.max_holding_pct,
            drawdown_monitor_enabled=cfg.drawdown_monitor.enabled,
            max_symbol_drawdown=cfg.drawdown_monitor.max_symbol_drawdown,
            max_symbol_daily_drawdown=cfg.drawdown_monitor.max_symbol_daily_drawdown,
            max_portfolio_drawdown=cfg.drawdown_monitor.max_portfolio_drawdown,
            max_portfolio_daily_drawdown=cfg.drawdown_monitor.max_portfolio_daily_drawdown,
            atr_period=cfg.indicators.atr_period,
            warmup_bars=cfg.simulation.warmup_bars,
        )
    
    def __post_init__(self):
        """Validate and create directories"""
        if not self.symbols:
            raise ValueError("symbols list cannot be empty")
        
        Path(self.trade_log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.strategy_routing).parent.mkdir(parents=True, exist_ok=True)
        Path(self.trade_logic_routing).parent.mkdir(parents=True, exist_ok=True)
    
    @property
    def trade_log_path(self) -> str:
        return os.path.join(self.trade_log_dir, self.trade_log_file)


# ============================================================================
# SIMULATION RUNNER
# ============================================================================

class SimulationRunner:
    """
    Simulation runner using improved architecture
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        # Logger - own file with propagation to app.log
        self.logger = Logger(
            log_file="simulation.log",
            logger_name="SimulationRunner",
            propagate=True
        ).get_logger()

        # Stop flag for graceful shutdown
        self._stop_requested = False

        # Core state
        self.portfolio = PortfolioState(cash=cfg.starting_cash)
        self.symbol_states: Dict[str, SymbolState] = {
            s: SymbolState(symbol=s) for s in cfg.symbols
        }

        # History buffers
        self.history: Dict[str, Deque[dict]] = {
            s: deque(maxlen=cfg.max_history_bars) for s in cfg.symbols
        }
        self.atr_hist: Dict[str, Deque[float]] = {
            s: deque(maxlen=cfg.max_atr_history) for s in cfg.symbols
        }

        # Equity curve history for PNL events
        self.equity_history: Deque[float] = deque(maxlen=1000)
        self.equity_history.append(cfg.starting_cash)

        # Utilities

        # Initialize components
        self._init_components()

        self.logger.info(f"SimulationRunner initialized for {cfg.symbols}")

    def stop(self):
        """Request graceful stop of the simulation."""
        self._stop_requested = True
        self.logger.info("Stop requested")
    
    def _init_components(self):
        """Initialize all components"""
        # Simulator
        self.sim = GBMSimulator(
            self.cfg.symbols,
            base_price=300.0
        )
        
        # Event handler - use global singleton so GUI receives events
        # NOTE: Must be created BEFORE trade logic manager to pass event handler
        from core.events.eventhandler import get_event_handler
        self.events = get_event_handler()
        print(f"[SIM] EventHandler instance ID: {id(self.events)}")

        # Strategy routing
        self.strategy_routing = StrategyRoutingManager(
            self.cfg.strategy_routing
        )

        # Trade logic routing (with event handler for signal emission)
        self.trade_logic_manager = DynamicTradeLogicManager(
            self.cfg.trade_logic_routing,
            event_handler=self.events
        )
        
        # Broker
        self.broker = MockBroker(self.cfg.starting_cash)
        
        # Executor
        self.executor = MockExecutor(self.broker)
        
        # Position sizer - use config values
        self.sizer = KellyPositionSizer(
            risk_percentage=self.cfg.risk_per_trade,
            max_trade_pct=self.cfg.max_trade_pct,
            max_holding_pct=self.cfg.max_holding_pct
        )
        self.logger.info(f"[SIM] Position sizer: risk={self.cfg.risk_per_trade:.1%}, max_trade={self.cfg.max_trade_pct:.1%}")
        
        # Trade logger
        self.trade_logger = FileTradeLogger(
            log_file=self.cfg.trade_log_file,
            log_dir=self.cfg.trade_log_dir
        )
        
        # Drawdown monitor - controlled by config
        if self.cfg.drawdown_monitor_enabled:
            self.ddm = DrawdownMonitor(
                max_symbol_drawdown=self.cfg.max_symbol_drawdown,
                max_symbol_daily_drawdown=self.cfg.max_symbol_daily_drawdown,
                max_portfolio_drawdown=self.cfg.max_portfolio_drawdown,
                max_portfolio_daily_drawdown=self.cfg.max_portfolio_daily_drawdown
            )
            self.logger.info("[SIM] Drawdown monitor ENABLED")
        else:
            self.ddm = None
            self.logger.info("[SIM] Drawdown monitor DISABLED (via config)")

        # Execution engine
        self.engine = MockExecutionEngine(
            broker=self.broker,
            executor=self.executor,
            sizer=self.sizer,
            performance_tracker=self.trade_logger,
            trade_logic_manager=self.trade_logic_manager,
            portfolio=self.portfolio,
            drawdown_monitor=self.ddm,
            event_handler=self.events
        )
    
    # ========================================================================
    # BAR PROCESSING
    # ========================================================================
    
    def _df_from_history(self, symbol: str) -> pd.DataFrame:
        """Build DataFrame from history"""
        if not self.history[symbol]:
            return pd.DataFrame()
        
        df = pd.DataFrame(list(self.history[symbol]))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        return df[["Open", "High", "Low", "Close", "Volume"]].copy()
    
    def _classify_regime(self, indicators_row, atr_history):
        """Classify market regime based on indicator values"""
        atr = indicators_row.get('ATR')
        rsi = indicators_row.get('RSI')
        bb_upper = indicators_row.get('Bollinger_Upper')
        bb_lower = indicators_row.get('Bollinger_Lower')
        close = indicators_row.get('Close')
        
        # Default to UNKNOWN if key indicators are missing
        if pd.isna(atr) or pd.isna(close):
            return "UNKNOWN"
        
        # Calculate Bollinger Band width if we have the bands
        bb_width = None
        if not pd.isna(bb_upper) and not pd.isna(bb_lower):
            bb_width = (bb_upper - bb_lower) / close
        
        # Calculate dynamic ATR thresholds based on history
        if len(atr_history) >= 20:
            atr_list = list(atr_history)
            atr_mean = np.mean(atr_list[-20:])
            atr_std = np.std(atr_list[-20:])
            high_atr_threshold = atr_mean + atr_std
            low_atr_threshold = atr_mean - 0.5 * atr_std
        else:
            return "UNKNOWN"  # Not enough history yet
        
        # High volatility regime
        if atr > high_atr_threshold:
            return "HIGH_VOLATILITY"
        
        # Low volatility regime  
        if atr < low_atr_threshold:
            return "LOW_VOLATILITY"
        
        # Range-bound (using RSI and Bollinger Bands)
        if not pd.isna(rsi) and bb_width is not None:
            if 40 < rsi < 60 and bb_width < 0.02:  # Narrow range
                return "RANGE_BOUND"
        
        # Trending (default when volatility is normal)
        return "TRENDING"
        
    async def _on_bar(self, event) -> None:
        """Process bar event"""
        try:
            payload = event.payload
            symbol = payload["symbol"]
            
            # Build bar dict
            bar = {
                "timestamp": pd.to_datetime(payload["timestamp"]),
                "symbol": symbol,
                "Open": payload["open"],
                "High": payload["high"],
                "Low": payload["low"],
                "Close": payload["close"],
                "Volume": payload["volume"]
            }
            
            # Update history
            self.history[symbol].append(bar)
            
            # Update portfolio with price
            price = float(bar["Close"])
            self.portfolio.update_price(symbol, price)
            
            # Build DataFrame
            df = self._df_from_history(symbol)
            if df.empty or len(df) < self.cfg.atr_period:
                return
            
            # Compute indicators
            indicators = TechnicalIndicators(df).apply_all()
            
            atr = indicators['ATR'].iloc[-1]
            if atr and not pd.isna(atr):
                self.atr_hist[symbol].append(atr)
            
            # Classify regime based on current indicators - PASS THE SERIES, NOT ATR
            regime = self._classify_regime(indicators.iloc[-1], self.atr_hist[symbol])
                    
            # Get strategy and generate signal
            strategy = self.strategy_routing.get_strategy(symbol, regime)
            strategy_name = strategy.__class__.__name__
            
            try:
                signal = strategy.generate_signal(df)
                if not isinstance(signal, int):
                    signal = 0
            except Exception as e:
                self.logger.error(f"Strategy error for {symbol}: {e}")
                signal = 0

            # Emit strategy signal event (for Strategies tab)
            signal_payload = {
                "symbol": symbol,
                "strategy": strategy_name,
                "signal": "buy" if signal == 1 else "sell" if signal == -1 else "hold",
                "confidence": None,
                "regime": regime,  # Include regime for Strategies tab
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await self.events.emit(EVENT_STRATEGY_SIGNAL, signal_payload)

            # Emit regime update event (for Market/Strategies tab)
            regime_payload: RegimePayload = {
                "symbol": symbol,
                "volatility": regime,  # low_volatility, normal, high_volatility
                "trend": "bullish" if signal == 1 else "bearish" if signal == -1 else "sideways",
                "market": "neutral",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await self.events.emit(EVENT_REGIME_UPDATE, regime_payload)

            # Get state
            state = self.symbol_states[symbol]

            # Create context and handle signal via engine
            context = SignalContext.from_kwargs(
                symbol=symbol,
                signal=signal,
                price=price,
                atr=atr or 0.0,
                regime=regime,
                timestamp=datetime.now(timezone.utc),
                strategy_name=strategy_name,
                df=df,
            )
            await self.engine.handle_signal(context, state)

            # Emit PNL update after each bar (for continuous equity curve in monitor)
            equity = self.portfolio.total_equity()
            self.equity_history.append(equity)

            pnl_payload: PnLPayload = {
                "portfolio_value": equity,
                "equity_curve": list(self.equity_history)[-100:],  # Limit size
                "unrealized": self.portfolio.total_unrealized(),
                "realized": self.portfolio.total_realized(),
                "drawdown": self.portfolio.drawdown(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Debug: Log emission
            if len(self.equity_history) % 10 == 0:  # Every 10 bars
                print(f"[SIM] Emitting PNL: equity=${equity:,.2f}, bus_id={id(self.events)}")

            await self.events.emit(EVENT_PNL_UPDATE, pnl_payload)

        except Exception as e:
            self.logger.error(f"Bar processing failed: {e}", exc_info=True)
    
    # ========================================================================
    # SIMULATION LOOP
    # ========================================================================
    
    async def _bar_producer(self) -> None:
        """Generate and emit bars"""
        try:
            for step in range(self.cfg.steps):
                # Check for stop request
                if self._stop_requested:
                    self.logger.info(f"Simulation stopped at step {step}")
                    break

                # Reset position sizer reservations for new bar cycle
                # This prevents accumulated reservations from blocking new trades
                self.sizer.reset_bar_reservations()

                all_bars = self.sim.update_all()

                for bar in all_bars.values():
                    # Check stop flag before each bar emission
                    if self._stop_requested:
                        break

                    # Emit bar event
                    payload: BarPayload = {
                        "symbol": bar["symbol"],
                        "timestamp": bar["timestamp"].isoformat() if hasattr(bar["timestamp"], 'isoformat') else str(bar["timestamp"]),
                        "open": float(bar["open"]),
                        "high": float(bar["high"]),
                        "low": float(bar["low"]),
                        "close": float(bar["close"]),
                        "volume": int(bar.get("volume", 0)),
                    }
                    await self.events.emit(EVENT_NEW_BAR, payload)

                if self._stop_requested:
                    break

                await asyncio.sleep(self.cfg.bar_sleep)

            if self._stop_requested:
                self.logger.info("Bar production stopped by user")
            else:
                self.logger.info("Bar production complete")

        except asyncio.CancelledError:
            self.logger.info("Bar producer cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Bar producer failed: {e}", exc_info=True)
    
    async def _bar_consumer(self) -> None:
        """Subscribe to bar events"""
        try:
            # Subscribe to bars
            await self.events.subscribe(EVENT_NEW_BAR, self._on_bar)

            # Subscribe to HALT events from GUI
            await self.events.subscribe(EVENT_HALTED, self._on_halt)

            # NOTE: Do NOT call engine.subscribe_signals() here!
            # The simulation directly calls engine.handle_signal() in _on_bar,
            # so subscribing to EVENT_STRATEGY_SIGNAL would cause duplicate processing.

            # But DO subscribe to GUI events (flatten, cancel, halt, manual orders)
            if hasattr(self.engine, '_subscribe_gui_events'):
                await self.engine._subscribe_gui_events()

        except Exception as e:
            self.logger.error(f"Bar consumer failed: {e}", exc_info=True)

    async def _on_halt(self, event) -> None:
        """Handle HALT event from GUI."""
        payload = event.payload if hasattr(event, 'payload') else event
        halted = payload.get('halted', False)
        if halted:
            self.logger.info("HALT received - stopping simulation")
            self.stop()
    
    def _seed(self):
        """Seed with warm-up bars"""
        for symbol in self.cfg.symbols:
            for _ in range(self.cfg.warmup_bars):
                bar = self.sim.generate_bar(symbol)
                bar["timestamp"] = datetime.now(timezone.utc)
                self.history[symbol].append(bar)
    
    async def run(self) -> None:
        """Run simulation"""
        try:
            self.logger.info(f"Starting simulation: {self.cfg.symbols}, steps={self.cfg.steps}")

            # Seed history
            self._seed()

            # IMPORTANT: Subscribe to events FIRST before starting producer
            # This prevents the race condition where producer emits before consumer subscribes
            await self._bar_consumer()

            # Now run the producer
            await self._bar_producer()
            
            # Final stats
            final_equity = self.portfolio.total_equity()
            self.logger.info(f"Simulation complete. Final equity: ${final_equity:,.2f}")
            
            print("\n" + "="*60)
            print("📊 SIMULATION RESULTS")
            print("="*60)
            print(f"Final Equity:    ${final_equity:>15,.2f}")
            print(f"Cash:            ${self.portfolio.cash:>15,.2f}")
            print(f"Unrealized PnL:  ${self.portfolio.total_unrealized():>15,.2f}")
            print(f"Realized PnL:    ${self.portfolio.realized_pnl:>15,.2f}")
            print(f"Total Positions: {self.portfolio.num_positions:>15}")
            print("="*60)
        
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}", exc_info=True)
        
        finally:
            self.trade_logger.flush()
            self.trade_logger.close()


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Run simulation"""
    cfg = SimConfig(
        symbols=["AAPL", "MSFT"],
        steps=500,
        bar_sleep=0.01,
        starting_cash=100_000.0
    )
    
    runner = SimulationRunner(cfg)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())