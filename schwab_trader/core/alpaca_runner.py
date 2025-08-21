# core/live_runner.py
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
from core.eventhandler import EventHandler

from core.simulator.simulation import compute_atr, classify_regime  # reuse your helpers
from core.broker.alpaca_broker import AlpacaBroker
from core.logic.live_execution_engine import LiveExecutionEngine
from core.logic.trade_logic_manager import DynamicTradeLogicManager
load_dotenv(r"/home/kwasi/Projects/trader/schwab_trader/venv/.env")

class AlpacaLiveRunner:
    def __init__(self, settings: Settings, symbols: list[str]):
        self.settings = settings
        self.symbols = symbols
        self.event_handler = EventHandler()

        # logging + trade log
        self.logger = get_module_logger(module_name='AlpacaLiveRunner', file_key='AlpacaLive')
        self.trade_logger = FileTradeLogger(log_file='live_trades.csv', logger_name='LiveTradeLogger')

        # broker (Alpaca)
        self.broker = AlpacaBroker(
            api_key=settings.get("ALPACA_KEY_ID"),
            api_secret=settings.get("ALPACA_SECRET_KEY"),
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

        # sizer, router, engine
        self.sizer = DynamicPositionSizer(
            risk_percentage=settings.get("BASE_RISK_PCT", 0.05)
        )
        self.router = StrategyRoutingManager('/home/kwasi/Projects/trader/config/strategy_routing.json')
        self.engine = LiveExecutionEngine(
            broker=self.broker,
            sizer=self.sizer,
            performance_tracker=self.trade_logger,
            trade_logic_manager=DynamicTradeLogicManager('/home/kwasi/Projects/trader/config/trade_logic_routing.json'),
        )
        # If your engine has optional attrs (per our earlier patch), wire them:
        if hasattr(self.engine, "trade_gate"):
            self.engine.trade_gate = self.trade_gate
        if hasattr(self.engine, "drawdown_monitor"):
            self.engine.drawdown_monitor = self.ddm

        # bar tracking
        self._last_bar_id: dict[str, int] = {}
        self._last_ddm_date = None
        

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
    async def seed(self, lookback_bars: int = 200):
        loader = HistoricalBarLoader('/home/kwasi/Projects/trader/schwab_trader/data/data_storage/proc_data')
        for sym in self.symbols:
            for b in loader.load_last_n_bars(sym, n=lookback_bars):
                # assume loader returns dict-like
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
        self.logger.info(f"Seeded {len(self.symbols)} symbols with {lookback_bars} bars.")

    # ---------- bar callback ----------
    async def on_alpaca_bar(self, raw_bar):
        bar = self._canonicalize_alpaca_bar(raw_bar)
        self.logger.debug(f"[RAW BAR] {raw_bar.symbol} {raw_bar.timestamp} c={raw_bar.close}")
        print(f"[RUNNER RAW BAR] {raw_bar.symbol} {raw_bar.timestamp} c={raw_bar.close}", flush=True)
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

        # strategy & signal
        strategy = self.router.get_strategy(symbol, regime)
        strategy_name = type(strategy).__name__
        try:
            raw_signal = strategy.generate_signal(df)
            signal = int(raw_signal if isinstance(raw_signal, (int, float)) else getattr(raw_signal, "signal", 0))
        except Exception as e:
            self.logger.exception(f"[{symbol}] Strategy error in {strategy_name}: {e}")
            signal = 0

        # trade gate context for this bar
        self.trade_gate.on_new_bar(symbol, bar_id, regime)
        gs = self.trade_gate.get_state(symbol)
        state.regime = regime
        state.regime_persist = gs.regime_persist

        # hand off to engine (engine should enforce gates; runner is context-only)
        self.engine.handle_signal(
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

    # ---------- run ----------
    async def run(self):
        await self.seed(self.settings.get("SEED_BARS", 200))
        await self.event_handler.subscribe("BAR", self._bar_debug_logger)

        # connect + subscribe
        self.broker.api_key = os.getenv("ALPACA_API_KEY")
        self.broker.api_secret = os.getenv("ALPACA_SECRET")
        self.broker.connect()
        for sym in self.symbols:
            self.broker.subscribe_bars(self.on_alpaca_bar, sym)

        stream_task = asyncio.create_task(self.broker.start_stream())
        self.logger.info(f"LiveRunner (Alpaca) started for: {', '.join(self.symbols)}")

        try:
            while True:
                await asyncio.sleep(0.5)
        finally:
            stream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stream_task
            self.trade_logger.flush()
            self.logger.info("LiveRunner shut down cleanly.")


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