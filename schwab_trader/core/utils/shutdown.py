"""
Graceful shutdown handling for trading system.

Ensures clean shutdown with position handling and state persistence.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Awaitable

logger = logging.getLogger(__name__)


class ShutdownPhase(Enum):
    """Phases of graceful shutdown."""
    RUNNING = "running"
    STOPPING = "stopping"          # Stop accepting new work
    DRAINING = "draining"          # Wait for in-flight work
    CLOSING_POSITIONS = "closing"  # Close/flatten positions if configured
    SAVING_STATE = "saving"        # Persist state
    CLEANUP = "cleanup"            # Release resources
    STOPPED = "stopped"


class ShutdownManager:
    """
    Manages graceful shutdown sequence for trading systems.

    Shutdown sequence:
    1. Stop accepting new signals/orders
    2. Wait for in-flight orders to complete
    3. Optionally close/flatten positions
    4. Save system state to persistence
    5. Close connections and release resources
    """

    def __init__(
        self,
        flatten_on_shutdown: bool = False,
        drain_timeout: float = 30.0,
        save_state: bool = True,
    ):
        """
        Initialize shutdown manager.

        Args:
            flatten_on_shutdown: Whether to close all positions on shutdown
            drain_timeout: Max seconds to wait for in-flight work
            save_state: Whether to save state before shutdown
        """
        self.flatten_on_shutdown = flatten_on_shutdown
        self.drain_timeout = drain_timeout
        self.save_state = save_state

        self.phase = ShutdownPhase.RUNNING
        self._shutdown_event = asyncio.Event()
        self._handlers: Dict[ShutdownPhase, List[Callable[[], Awaitable[None]]]] = {
            phase: [] for phase in ShutdownPhase
        }
        self._signal_handlers_installed = False

    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self.phase != ShutdownPhase.RUNNING

    def should_accept_work(self) -> bool:
        """Check if system should accept new work."""
        return self.phase == ShutdownPhase.RUNNING

    def register_handler(
        self,
        phase: ShutdownPhase,
        handler: Callable[[], Awaitable[None]],
    ) -> None:
        """
        Register a handler for a shutdown phase.

        Args:
            phase: Shutdown phase to handle
            handler: Async function to call during that phase
        """
        self._handlers[phase].append(handler)

    def install_signal_handlers(self) -> None:
        """Install OS signal handlers for graceful shutdown."""
        if self._signal_handlers_installed:
            return

        def signal_handler(sig, frame):
            logger.info(f"[Shutdown] Received signal {sig}, initiating shutdown...")
            asyncio.create_task(self.shutdown())

        # Handle SIGINT (Ctrl+C) and SIGTERM
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        self._signal_handlers_installed = True
        logger.info("[Shutdown] Signal handlers installed")

    async def shutdown(self) -> None:
        """
        Execute graceful shutdown sequence.
        """
        if self.phase != ShutdownPhase.RUNNING:
            logger.warning("[Shutdown] Already shutting down")
            return

        logger.info("[Shutdown] Starting graceful shutdown sequence...")
        start_time = datetime.now(timezone.utc)

        try:
            # Phase 1: Stop accepting new work
            self.phase = ShutdownPhase.STOPPING
            logger.info("[Shutdown] Phase 1: Stopping new work acceptance")
            await self._run_handlers(ShutdownPhase.STOPPING)

            # Phase 2: Drain in-flight work
            self.phase = ShutdownPhase.DRAINING
            logger.info(f"[Shutdown] Phase 2: Draining in-flight work (timeout: {self.drain_timeout}s)")
            try:
                await asyncio.wait_for(
                    self._run_handlers(ShutdownPhase.DRAINING),
                    timeout=self.drain_timeout
                )
            except asyncio.TimeoutError:
                logger.warning("[Shutdown] Drain timeout reached, proceeding with shutdown")

            # Phase 3: Close positions if configured
            if self.flatten_on_shutdown:
                self.phase = ShutdownPhase.CLOSING_POSITIONS
                logger.info("[Shutdown] Phase 3: Closing positions")
                await self._run_handlers(ShutdownPhase.CLOSING_POSITIONS)
            else:
                logger.info("[Shutdown] Phase 3: Skipping position close (not configured)")

            # Phase 4: Save state
            if self.save_state:
                self.phase = ShutdownPhase.SAVING_STATE
                logger.info("[Shutdown] Phase 4: Saving system state")
                await self._run_handlers(ShutdownPhase.SAVING_STATE)
            else:
                logger.info("[Shutdown] Phase 4: Skipping state save (not configured)")

            # Phase 5: Cleanup
            self.phase = ShutdownPhase.CLEANUP
            logger.info("[Shutdown] Phase 5: Cleanup and resource release")
            await self._run_handlers(ShutdownPhase.CLEANUP)

            # Done
            self.phase = ShutdownPhase.STOPPED
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(f"[Shutdown] Graceful shutdown complete in {elapsed:.2f}s")

        except Exception as e:
            logger.exception(f"[Shutdown] Error during shutdown: {e}")
            self.phase = ShutdownPhase.STOPPED
            raise

        finally:
            self._shutdown_event.set()

    async def _run_handlers(self, phase: ShutdownPhase) -> None:
        """Run all handlers for a phase."""
        handlers = self._handlers.get(phase, [])
        if not handlers:
            return

        tasks = [asyncio.create_task(h()) for h in handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for handler, result in zip(handlers, results):
            if isinstance(result, Exception):
                logger.error(
                    f"[Shutdown] Handler {handler.__name__} failed: {result}"
                )

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown to complete."""
        await self._shutdown_event.wait()


# Global shutdown manager
_shutdown_manager: Optional[ShutdownManager] = None


def get_shutdown_manager(
    flatten_on_shutdown: bool = False,
    drain_timeout: float = 30.0,
    save_state: bool = True,
) -> ShutdownManager:
    """Get or create the global shutdown manager."""
    global _shutdown_manager
    if _shutdown_manager is None:
        _shutdown_manager = ShutdownManager(
            flatten_on_shutdown=flatten_on_shutdown,
            drain_timeout=drain_timeout,
            save_state=save_state,
        )
    return _shutdown_manager


def create_shutdown_handlers(
    event_handler: Any = None,
    portfolio_state: Any = None,
    broker: Any = None,
    persistence: Any = None,
) -> ShutdownManager:
    """
    Create shutdown manager with standard handlers.

    Args:
        event_handler: EventHandler instance
        portfolio_state: PortfolioState instance
        broker: Broker instance
        persistence: StatePersistence instance

    Returns:
        Configured ShutdownManager
    """
    manager = get_shutdown_manager()

    # Handler: Stop event handler
    if event_handler:
        async def stop_events():
            logger.info("[Shutdown] Stopping event handler...")
            if hasattr(event_handler, 'shutdown'):
                await event_handler.shutdown()

        manager.register_handler(ShutdownPhase.STOPPING, stop_events)

    # Handler: Wait for pending orders
    if broker:
        async def drain_orders():
            logger.info("[Shutdown] Waiting for pending orders...")
            if hasattr(broker, 'get_pending_orders'):
                pending = await broker.get_pending_orders()
                if pending:
                    logger.info(f"[Shutdown] {len(pending)} pending orders")
                    # Wait a bit for orders to complete
                    await asyncio.sleep(5.0)

        manager.register_handler(ShutdownPhase.DRAINING, drain_orders)

    # Handler: Close positions
    if broker and portfolio_state:
        async def close_positions():
            logger.info("[Shutdown] Closing all positions...")
            if hasattr(portfolio_state, 'positions'):
                for symbol, pos in portfolio_state.positions.items():
                    if pos.qty != 0:
                        try:
                            side = "sell" if pos.qty > 0 else "buy"
                            await broker.place_market_order(
                                symbol=symbol,
                                side=side,
                                quantity=abs(pos.qty),
                            )
                            logger.info(f"[Shutdown] Closed position: {symbol}")
                        except Exception as e:
                            logger.error(f"[Shutdown] Failed to close {symbol}: {e}")

        manager.register_handler(ShutdownPhase.CLOSING_POSITIONS, close_positions)

    # Handler: Save state
    if persistence and portfolio_state:
        async def save_state():
            logger.info("[Shutdown] Saving system state...")
            try:
                state_id = f"shutdown_{uuid.uuid4().hex[:8]}"
                positions = []
                if hasattr(portfolio_state, 'positions'):
                    positions = [
                        {'symbol': s, 'qty': p.qty, 'avg_price': p.avg_price}
                        for s, p in portfolio_state.positions.items()
                    ]

                persistence.save_system_state(
                    state_id=state_id,
                    cash_balance=getattr(portfolio_state, 'cash', 0),
                    total_equity=getattr(portfolio_state, 'total_equity', lambda: 0)(),
                    positions=positions,
                    pending_orders=[],
                    metadata={'shutdown_type': 'graceful'},
                )
                logger.info(f"[Shutdown] State saved: {state_id}")
            except Exception as e:
                logger.error(f"[Shutdown] Failed to save state: {e}")

        manager.register_handler(ShutdownPhase.SAVING_STATE, save_state)

    # Handler: Close broker connection
    if broker:
        async def close_broker():
            logger.info("[Shutdown] Closing broker connection...")
            if hasattr(broker, 'disconnect'):
                await broker.disconnect()
            elif hasattr(broker, 'close'):
                broker.close()

        manager.register_handler(ShutdownPhase.CLEANUP, close_broker)

    # Handler: Close persistence
    if persistence:
        async def close_persistence():
            logger.info("[Shutdown] Closing persistence...")
            if hasattr(persistence, 'close'):
                persistence.close()

        manager.register_handler(ShutdownPhase.CLEANUP, close_persistence)

    return manager
