#%% core/simulator/run_sim.py
from __future__ import annotations

import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # .../schwab_trader
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
#%%
import asyncio
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Deque, Any, Optional
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from loggers.logger import Logger
from core.eventhandler import EventHandler
from core.simulator.gbm_simulator import GBMSimulator
from core.logic.strategy_routing_manager import StrategyRoutingManager
from core.simulator.strategy_router import StrategyRouter
from core.logic.trade_logic_manager import DynamicTradeLogicManager
from core.logic.mock_execution_engine import MockExecutionEngine
from core.logic.symbol_state import SymbolState
from core.logic.portfolio_state import PortfolioState
from core.position_sizer import DynamicPositionSizer
from loggers.file_trade_logger import FileTradeLogger
from core.historical_loader import HistoricalBarLoader
from core.drawdown_monitor import DrawdownMonitor
from core.logic.trade_gate import TradeGate
from utils.replay import replay_equity_from_trades

# If your mock broker doesn't expose `submit_market_order`, adapt it:
from core.broker.mock_broker import MockBroker 
#%%

# -----------------------------
# Small utilities
# -----------------------------
# core/simulator/simulation.py
# inside SimulationRunner


CANONICAL = ("Open", "High", "Low", "Close", "Volume")

def to_canonical_bar(raw: dict) -> dict:
    m = {k.lower(): v for k, v in raw.items()}
    # timestamp -> pandas Timestamp (UTC)
    ts = m.get("timestamp", m.get("date"))
    import pandas as pd
    if ts is None:
        raise ValueError("bar missing timestamp/date")
    if isinstance(ts, (int, float)):  # epoch ms
        ts = pd.to_datetime(ts, unit="ms", utc=True)
    else:
        ts = pd.to_datetime(ts, utc=True)

    # coerce numeric types (np.float64 -> float)
    def f(key, alt=None, cast=float):
        v = m.get(key, m.get(alt) if alt else None)
        if v is None:
            raise KeyError(f"bar missing '{key}'")
        return cast(v)

    return {
        "timestamp": ts,
        "symbol": m["symbol"],
        "Open":  f("open"),
        "High":  f("high"),
        "Low":   f("low"),
        "Close": f("close"),
        "Volume": int(m.get("volume", 0)),
    }


def compute_atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
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



def classify_regime(atr_value: float, atr_window: Deque[float]) -> str:
    """
    Classify regime by ATR quantiles over a rolling window.
    """
    if atr_value is None or len(atr_window) < 10:
        return "normal"
    s = pd.Series(list(atr_window))
    q25 = s.quantile(0.25)
    q75 = s.quantile(0.75)
    if atr_value < q25:
        return "low_volatility"
    if atr_value > q75:
        return "high_volatility"
    return "normal"


# -----------------------------
# Runner
# -----------------------------

@dataclass
class SimConfig:
    symbols: list[str]
    steps: int = 600
    bar_sleep: float = 0.05
    strategy_routing_path: str = "config/strategy_routing.json"
    strategy_params_path: str = "config/strategy_params.json"
    trade_logic_routing_path: str = "config/trade_logic_routing.json"
    ddm: dict = field(default_factory=lambda: {
        "max_symbol_drawdown": 0.30,
        "max_symbol_daily_drawdown": 0.10,
        "symbol_cooldown_seconds": 20,
        "max_portfolio_drawdown": 0.25,
        "max_portfolio_daily_drawdown": 0.10,
        "portfolio_cooldown_seconds": 60,
    })
class SimulationRunner:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.logger = Logger("simulation.log", self.__class__.__name__).get_logger()

        # States
        self.portfolio = PortfolioState(cash=100_000.0)
        self.symbol_state: Dict[str, SymbolState] = {s: SymbolState(symbol=s) for s in cfg.symbols}

        # History buffers
        self.history: Dict[str, Deque[dict]] = {s: deque(maxlen=500) for s in cfg.symbols}
        self.atr_hist: Dict[str, Deque[float]] = {s: deque(maxlen=300) for s in cfg.symbols}

        # Core components
        self.sim = GBMSimulator(cfg.symbols, base_price=300.0, log_prices=False)

        # Routing + Strategy loader
        self.strategy_routing = StrategyRoutingManager(cfg.strategy_routing_path)
        self.strategy_router = StrategyRouter(self.strategy_routing, cfg.strategy_params_path)

        # Trade logic manager
        self.trade_logic_manager = DynamicTradeLogicManager(cfg.trade_logic_routing_path)

        # Broker + sizing + trade logger + engine
        self.broker = MockBroker(starting_cash=self.portfolio.cash)
        self.sizer = DynamicPositionSizer(risk_percentage=0.01)
        self.trade_logger = FileTradeLogger(log_file="trades_sim.csv")

        # historical data
        self.loader = HistoricalBarLoader(path="data/data_storage/proc_data")
        
        # Event hub
        self.events = EventHandler()
        self.ddm = DrawdownMonitor(
            max_symbol_drawdown=0.02,           # start tiny to confirm it triggers
            max_symbol_daily_drawdown=0.01,
            symbol_cooldown_seconds=10,
            max_portfolio_drawdown=0.02,
            max_portfolio_daily_drawdown=0.01,
            portfolio_cooldown_seconds=10,
        )
        # 
        self._last_ddm_date = None

        # init the execution engine
        self.engine = MockExecutionEngine(
                    broker=self.broker,
                    sizer=self.sizer,
                    performance_tracker=self.trade_logger,
                    trade_logic_manager=self.trade_logic_manager,
                    drawdown_monitor=self.ddm,
                    portfolio=self.portfolio

            )
        
        # initialize the trade gates. Helps deal with weird bars
        self.trade_gate = TradeGate(
            max_layers=3,
            min_bars_between_layers=2,
            regime_min_persist_bars=2,
            flip_cooldown_bars=1,
        )

        self._last_bar_id = {}   # symbol -> int
    
    def _extract_signal(self, sig) -> int:
        """
        Normalize various strategy outputs to a single int in {-1,0,1}.
        Accepted forms:
        - int/float/bool
        - dict with 'signal' (or 'Signal', 'trade_signal')
        - pd.Series -> last value
        - pd.DataFrame -> last value of 'signal'/'Signal'/'Position' if present
        """
        if sig is None:
            return 0

        # simple numerics & bools
        if isinstance(sig, (int, np.integer)):
            return int(sig)
        if isinstance(sig, (float, np.floating)):
            return int(np.sign(sig)) if not np.isnan(sig) else 0
        if isinstance(sig, bool):
            return 1 if sig else 0

        # dicts
        if isinstance(sig, dict):
            for k in ("signal", "Signal", "trade_signal"):
                if k in sig:
                    v = sig[k]
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        return int(np.sign(v)) if isinstance(v, float) else int(v)
                    if isinstance(v, pd.Series) and len(v):
                        return int(np.sign(v.iloc[-1])) if isinstance(v.iloc[-1], float) else int(v.iloc[-1])
            return 0

        # Series
        if isinstance(sig, pd.Series) and len(sig):
            v = sig.iloc[-1]
            if isinstance(v, (int, float, np.integer, np.floating)):
                return int(np.sign(v)) if isinstance(v, float) else int(v)
            if isinstance(v, bool):
                return 1 if v else 0
            return 0

        # DataFrame
        if isinstance(sig, pd.DataFrame) and not sig.empty:
            for col in ("signal", "Signal", "Position", "position", "trade_signal"):
                if col in sig.columns:
                    v = sig[col].iloc[-1]
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        return int(np.sign(v)) if isinstance(v, float) else int(v)
                    if isinstance(v, bool):
                        return 1 if v else 0
                    return 0
            # last resort: 0
            self.logger.warning("[SignalCoerce] DataFrame returned but no recognizable signal column. Defaulting to 0.")
            return 0

        # fallback
        self.logger.warning(f"[SignalCoerce] Unhandled signal type: {type(sig)}. Defaulting to 0.")
        return 0
    
    def _bar_bucket(self, ts: datetime, timeframe_sec: int = 60) -> int:
        return int(ts.timestamp() // timeframe_sec)

    async def _on_bar(self, bar: dict) -> None:
        bar = to_canonical_bar(bar)
        symbol = bar["symbol"]

        # --- NEW: bar_id + bar_closed inference (no structural change) ---
        ts: datetime = bar["timestamp"]
        if ts.tzinfo is None:
            from datetime import timezone
            ts = ts.replace(tzinfo=timezone.utc)
        bar_id = int(ts.timestamp() // 60)
        prev_bar_id = self._last_bar_id.get(symbol)
        bar_closed = prev_bar_id is not None and bar_id != prev_bar_id
        self._last_bar_id[symbol] = bar_id
        # ---------------------------------------------------------------

        self.history[symbol].append(bar)

        # MTM update for portfolio (keep it once)
        self.portfolio.update_price(symbol, bar["Close"])
        self.symbol_state[symbol].portfolio_value = self.portfolio.total_equity()

        # 👇 build the DataFrame the same way every time
        df = self._df_from_history(symbol)

        atr = compute_atr(df, period=14)
        if atr is not None:
            self.atr_hist[symbol].append(atr)

        regime = classify_regime(atr, self.atr_hist[symbol])

        # Ask the router for a strategy (caches the instance)
        strategy = self.strategy_router.get_strategy(symbol, regime)
        strategy_name = type(strategy).__name__

        # (you already did price update above)
        state = self.symbol_state[symbol]
        state.portfolio_value = self.portfolio.total_equity()

        # ---- Drawdown monitor updates ----
        if self._last_ddm_date is None or ts.date() != self._last_ddm_date:
            self.ddm.start_new_day(portfolio_equity=self.portfolio.total_equity())
            self._last_ddm_date = ts.date()

        equity = self.portfolio.total_equity()
        self.ddm.update_portfolio(equity)
        sym_mv = self._symbol_mv(symbol, float(bar["Close"]))
        self.ddm.update_symbol(symbol, sym_mv)

        price = float(bar["Close"])

        # approximate per-symbol equity value (position qty * last price)
        pos = self.portfolio.positions.get(bar["symbol"])
        sym_equity = (pos.qty * price) if pos else 0.0
        self.ddm.update_symbol(bar["symbol"], sym_equity)
        # ----------------------------------

        try:
            raw_signal = strategy.generate_signal(df)
            signal = self._extract_signal(raw_signal)
        except Exception as e:
            self.logger.exception(f"[{symbol}] Strategy error in {strategy_name}: {e}")
            signal = 0

        # --- NEW: update gate context and expose tiny flags to engine/state ---
        self.trade_gate.on_new_bar(symbol, bar_id, regime)
        st = self.trade_gate.get_state(symbol)  # GateState

        # Current side (for engines that need it)
        cur_qty = (pos.qty if pos else 0)
        cur_side = ("long" if cur_qty > 0 else "short" if cur_qty < 0 else None)

        # Gate booleans: entries only allowed at close; exits always allowed upstream
        can_enter_long  = (bar_closed and self.trade_gate.can_enter(symbol, "long",  bar_id, allow_pyramiding=False)[0])
        can_enter_short = (bar_closed and self.trade_gate.can_enter(symbol, "short", bar_id, allow_pyramiding=False)[0])
        can_pyr_long    = (bar_closed and self.trade_gate.can_enter(symbol, "long",  bar_id, allow_pyramiding=True)[0])
        can_pyr_short   = (bar_closed and self.trade_gate.can_enter(symbol, "short", bar_id, allow_pyramiding=True)[0])

        # Stash for engine (no behavior change here unless engine reads them)
        state.bar_id = bar_id
        state.bar_closed = bar_closed
        state.regime = regime
        state.regime_persist = (st.regime_persist if st else 0)
        state.cur_side = cur_side
        state.can_enter_long = can_enter_long
        state.can_enter_short = can_enter_short
        state.can_pyramid_long = can_pyr_long
        state.can_pyramid_short = can_pyr_short
        # ----------------------------------------------------------------------

        # Call your engine exactly like before (optionally pass gates)
        self.engine.handle_signal(
            symbol=symbol,
            state=state,
            signal=signal,
            price=price,
            atr=atr if atr is not None else 0.0,
            regime=regime,
            strategy_name=strategy_name,
        )


    def _symbol_mv(self, symbol: str, last_price: float) -> float:
        """Approx symbol market value from PortfolioState."""
        try:
            pos = self.portfolio.positions.get(symbol)  # adapt to your PortfolioState API
            qty = getattr(pos, "qty", 0)
            return float(qty) * float(last_price)
        except Exception:
            return 0.0
    
    def _df_from_history(self, symbol: str) -> pd.DataFrame:
        df = pd.DataFrame(list(self.history[symbol]))
        df = df.set_index(pd.to_datetime(df["timestamp"]), drop=True)
        return df[["Open", "High", "Low", "Close", "Volume"]].copy()


    async def _bar_producer(self) -> None:
        """
        Generate GBM bars and publish them to the event bus.
        """
        await self.events.start()
        for step in range(self.cfg.steps):
            all_bars = self.sim.update_all()
            for bar in all_bars.values():
                # fire-and-forget publish
                await self.events.publish("BAR", bar)
            await asyncio.sleep(self.cfg.bar_sleep)
        self.logger.info("Simulation finished producing bars.")

    async def _bar_consumer(self) -> None:
        """
        Subscribe to BAR events and process them.
        """
        async def bar_handler(event):
            await self._on_bar(event.payload)

        await self.events.subscribe("BAR", bar_handler)
        # Block until producer finishes draining queue (or just idle here)
        # In this design, consumer work happens via EventHandler's dispatcher.

    def _seed(self, warmup: int = 200):
        for s in self.cfg.symbols:
            bars = self.loader.load_last_n_bars(s, n=warmup)
            if not bars:
                # fallback: synth bars
                for _ in range(warmup):
                    self.history[s].append(self.sim.generate_bar(s))
            else:
                self.history[s].extend(bars)

            df = self._df_from_history(s)
            atr = compute_atr(df, 14)
            if atr is not None:
                self.atr_hist[s].append(atr)    

    async def run(self) -> None:
        self.logger.info(f"Starting sim for {self.cfg.symbols} | steps={self.cfg.steps}")
        await asyncio.gather(
            self._bar_consumer(),
            self._bar_producer(),
        )
        self.logger.info(
            f"Done. Final equity: ${self.portfolio.total_equity():,.2f} | "
            f"Unrealized: ${self.portfolio.total_unrealized():,.2f}"
        )


# ------------- CLI-ish entry -------------
async def main():
    # Ensure default config files exist (minimal, safe defaults)
    os.makedirs("config", exist_ok=True)

    # Strategy routing (which strategy to use by symbol+regime)
    sr_path = "config/strategy_routing.json"
    if not os.path.exists(sr_path):
        with open(sr_path, "w") as f:
            json.dump({
                "AAPL": {
                    "low_volatility": "sma_strategy",
                    "normal": "momentum_strategy",
                    "high_volatility": "mean_reversion_strategy"
                }
            }, f, indent=2)

    # Strategy params (injected at instantiation)
    sp_path = "config/strategy_params.json"
    if not os.path.exists(sp_path):
        with open(sp_path, "w") as f:
            json.dump({
                "AAPL": {
                    "low_volatility": {"params": {"fast": 10, "slow": 30}},
                    "normal": {"params": {"lookback": 20}},
                    "high_volatility": {"params": {"window": 14}}
                }
            }, f, indent=2)

    # Trade logic routing (which trade logic to use for symbol+regime)
    tl_path = "config/trade_logic_routing.json"
    if not os.path.exists(tl_path):
        with open(tl_path, "w") as f:
            json.dump({
                "AAPL": {
                    "low_volatility": {"trade_logic_class": "default", "params": {}},
                    "normal": {"trade_logic_class": "default", "params": {}},
                    "high_volatility": {"trade_logic_class": "default", "params": {}}
                }
            }, f, indent=2)

    cfg = SimConfig(
        symbols=["AAPL", "MSFT"],
        steps=1000,
        bar_sleep=0.1,
        strategy_routing_path=sr_path,
        strategy_params_path=sp_path,
        trade_logic_routing_path=tl_path,
    )

    runner = SimulationRunner(cfg)
    await runner.run()

    last_prices = {
        sym: (runner.history[sym][-1]["Close"] if runner.history[sym] else None)
        for sym in runner.cfg.symbols
    }
    # filter out Nones
    last_prices = {k: v for k, v in last_prices.items() if v is not None}

    snap = replay_equity_from_trades(
        csv_path="logs/trades_sim.csv",      # adjust if you wrote elsewhere
        starting_cash=100_000.0,
        final_prices=last_prices
    )
    print("Replay equity:", snap["equity"], "cash:", snap["cash"], "positions:", snap["positions"])


if __name__ == "__main__":
    asyncio.run(main())

# %%
