# core/simulator/live_plotter.py

from __future__ import annotations

import asyncio
import threading
from collections import deque
from dataclasses import dataclass
from typing import Dict, Deque, List, Optional, Tuple, Any

import matplotlib
import matplotlib.pyplot as plt

# Use your custom logger
try:
    from schwab_trader.loggers.logger import Logger as _Logger
except Exception:
    # Fallback if the path is different in your env
    from loggers.logger import Logger as _Logger


@dataclass
class TradeMarker:
    ts: Any
    price: float
    side: str  # "BUY" | "SELL" | "COVER" | etc.


class AsyncLivePlotter:
    """
    Async-friendly live plotter designed for simulations.

    - Non-blocking: runs in an asyncio task at a fixed refresh rate.
    - Thread-safe updates: all public record_* methods are safe to call
      from your simulator/strategy callbacks.
    - Low overhead: skips frames when nothing changed, caps refresh rate.
    - Headless friendly: can switch to Agg backend and save snapshots to disk.

    Usage
    -----
        plotter = AsyncLivePlotter(symbols=["AAPL","MSFT"], window=500, refresh_hz=4)
        await plotter.start()   # inside your main async sim
        ...
        plotter.update_bar("AAPL", ts, close, volume)
        plotter.record_trade("AAPL", ts, close, "BUY")
        plotter.record_pnl("AAPL", ts, cum_pnl)
        ...
        await plotter.stop()
    """

    def __init__(
        self,
        symbols: List[str],
        window: int = 500,
        refresh_hz: int = 5,
        backend: Optional[str] = None,    # e.g. "Agg" for headless
        save_snapshots: bool = False,
        snapshot_path: str = "plots/live_snapshot.png",
    ):
        if backend:
            matplotlib.use(backend, force=True)

        self.symbols = list(symbols)
        self.window = window
        self.refresh_hz = max(1, int(refresh_hz))
        self.save_snapshots = save_snapshots
        self.snapshot_path = snapshot_path

        # Buffers
        self._prices: Dict[str, Deque[float]] = {s: deque(maxlen=window) for s in self.symbols}
        self._times: Dict[str, Deque[Any]] = {s: deque(maxlen=window) for s in self.symbols}
        self._volumes: Dict[str, Deque[float]] = {s: deque(maxlen=window) for s in self.symbols}
        self._signals: Dict[str, Deque[Tuple[Any, int]]] = {s: deque(maxlen=window) for s in self.symbols}
        self._pnl: Dict[str, Deque[Tuple[Any, float]]] = {s: deque(maxlen=window) for s in self.symbols}
        self._trades: Dict[str, List[TradeMarker]] = {s: [] for s in self.symbols}

        # Plot objects
        self._fig: Optional[plt.Figure] = None
        self._axes: Dict[str, plt.Axes] = {}

        # Control
        self._lock = threading.Lock()
        self._dirty = False
        self._running = False
        self._task: Optional[asyncio.Task] = None

        self._logger = _Logger("plotter.log", self.__class__.__name__).get_logger()
        self._logger.info("AsyncLivePlotter initialized")

    # -------------------------- Public API (thread-safe) --------------------------

    def update_bar(self, symbol: str, ts: Any, close: float, volume: float = 0.0) -> None:
        if symbol not in self._prices:
            return
        with self._lock:
            self._times[symbol].append(ts)
            self._prices[symbol].append(close)
            self._volumes[symbol].append(volume)
            self._dirty = True

    def record_signal(self, symbol: str, ts: Any, signal: int) -> None:
        if symbol not in self._signals:
            return
        with self._lock:
            self._signals[symbol].append((ts, int(signal)))
            self._dirty = True

    def record_trade(self, symbol: str, ts: Any, price: float, side: str) -> None:
        if symbol not in self._trades:
            return
        with self._lock:
            self._trades[symbol].append(TradeMarker(ts, float(price), side.upper()))
            self._dirty = True

    def record_pnl(self, symbol: str, ts: Any, pnl: float) -> None:
        if symbol not in self._pnl:
            return
        with self._lock:
            self._pnl[symbol].append((ts, float(pnl)))
            self._dirty = True

    async def start(self) -> None:
        """
        Start the plotter's async loop. Must be awaited from an asyncio context.
        """
        if self._running:
            return
        self._setup_figure()
        self._running = True
        self._task = asyncio.create_task(self._run())
        self._logger.info("AsyncLivePlotter started")

    async def stop(self) -> None:
        """
        Stop the plotter and close the figure. Safe to call multiple times.
        """
        if not self._running:
            return
        self._running = False
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except asyncio.TimeoutError:
                self._logger.warning("Plotter task did not stop in time; cancelling.")
                self._task.cancel()
        plt.close(self._fig)
        self._fig = None
        self._axes.clear()
        self._logger.info("AsyncLivePlotter stopped")

    # -------------------------- Internals --------------------------

    def _setup_figure(self) -> None:
        rows = len(self.symbols)
        height = max(3.0, 2.5 * rows)
        self._fig, axs = plt.subplots(rows, 1, figsize=(10, height), sharex=True)
        if rows == 1:
            axs = [axs]
        for ax, sym in zip(axs, self.symbols):
            self._axes[sym] = ax
            ax.set_title(sym)
            ax.grid(True)
        plt.tight_layout()
        try:
            plt.ion()
            self._fig.canvas.manager.set_window_title("Live Simulation")
        except Exception:
            pass  # headless backends won't have a manager

    async def _run(self) -> None:
        interval = 1.0 / float(self.refresh_hz)
        while self._running:
            try:
                if self._dirty:
                    self._draw_frame()
                    self._dirty = False
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.exception(f"Plotter loop error: {e}")
                await asyncio.sleep(interval)

    def _draw_frame(self) -> None:
        # Snapshot under lock
        with self._lock:
            times = {s: list(self._times[s]) for s in self.symbols}
            prices = {s: list(self._prices[s]) for s in self.symbols}
            volumes = {s: list(self._volumes[s]) for s in self.symbols}
            signals = {s: list(self._signals[s]) for s in self.symbols}
            pnl = {s: list(self._pnl[s]) for s in self.symbols}
            trades = {s: list(self._trades[s]) for s in self.symbols}

        for sym in self.symbols:
            ax = self._axes[sym]
            ax.clear()
            ax.grid(True)
            ax.set_title(sym)

            # Price
            if times[sym] and prices[sym]:
                ax.plot(times[sym], prices[sym], label="Price")

            # Trades
            if trades[sym]:
                buys_x = [m.ts for m in trades[sym] if m.side in ("BUY", "COVER")]
                buys_y = [m.price for m in trades[sym] if m.side in ("BUY", "COVER")]
                sells_x = [m.ts for m in trades[sym] if m.side not in ("BUY", "COVER")]
                sells_y = [m.price for m in trades[sym] if m.side not in ("BUY", "COVER")]
                if buys_x:
                    ax.scatter(buys_x, buys_y, marker="^", s=40, label="Buy/Cover")
                if sells_x:
                    ax.scatter(sells_x, sells_y, marker="v", s=40, label="Sell/Short")

            # Signals (optional markers at current price)
            if signals[sym] and prices[sym]:
                latest_px = prices[sym][-1]
                for (ts, sig) in signals[sym][-min(20, len(signals[sym])):]:
                    if sig == 1:
                        ax.plot(ts, latest_px, "^", alpha=0.3)
                    elif sig == -1:
                        ax.plot(ts, latest_px, "vdefaultdict,", alpha=0.3)

            # PnL (as dashed overlay)
            if pnl[sym]:
                ts, vals = zip(*pnl[sym])
                ax.plot(ts, vals, linestyle="--", label="PnL")

            # You can uncomment this to visualize normalized volume:
            # if volumes[sym] and times[sym]:
            #     vmax = max(volumes[sym]) or 1.0
            #     vnorm = [0.2 * (v / vmax) for v in volumes[sym]]
            #     ax.fill_between(times[sym], min(prices[sym]) - 0.5, min(prices[sym]) - 0.5 + vnorm, alpha=0.2, step="pre", label="Volume (norm)")

            ax.legend(loc="upper left", fontsize=8)

        try:
            self._fig.tight_layout()
            self._fig.canvas.draw_idle()
            # A tiny pause lets UI backends process events; on Agg it’s a no-op
            plt.pause(0.001)
            if self.save_snapshots:
                self._fig.savefig(self.snapshot_path, dpi=120, bbox_inches="tight")
        except Exception as e:
            self._logger.debug(f"Draw frame warning: {e}")
