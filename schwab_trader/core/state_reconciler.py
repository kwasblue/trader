"""
State Reconciler - Ensures local state matches broker state.

Critical for production trading to prevent state drift between
the trading system and the actual broker positions.

Features:
- Startup sync: Load all positions from broker before trading
- Periodic reconciliation: Check positions match every N seconds
- Order verification: Confirm fills match expectations
- Mismatch detection: Alert/halt on state divergence
- Auto-correction: Fix minor discrepancies from rounding

Usage:
    from core.state_reconciler import StateReconciler

    reconciler = StateReconciler(broker, portfolio, config)

    # On startup
    await reconciler.full_sync()

    # Start periodic reconciliation
    await reconciler.start_periodic(interval_seconds=60)

    # After order execution
    await reconciler.verify_order(order_id, expected_qty)

    # Check current state
    mismatches = await reconciler.check_positions()
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from core.app_types import BrokerSnapshot, PositionView
from loggers.logger import Logger
from core.events.eventhandler import get_event_handler
from core.events import events

if TYPE_CHECKING:
    from core.logic.portfolio_state import PortfolioState
    from core.base.base_broker_interface import BaseBrokerInterface

# Dedicated log file for state reconciliation
logger = Logger(
    log_file="state_reconciler.log",
    logger_name="StateReconciler",
    propagate=True
).get_logger()


class ReconcileAction(Enum):
    """Actions to take on mismatch."""
    ALERT = "alert"          # Log and emit alert, continue trading
    HALT = "halt"            # Stop trading until resolved
    AUTO_CORRECT = "auto"    # Automatically sync from broker


@dataclass
class PositionMismatch:
    """Details of a position mismatch."""
    symbol: str
    local_qty: float
    broker_qty: float
    local_avg_price: float
    broker_avg_price: float
    qty_diff: float
    price_diff: float
    timestamp: str
    severity: str  # 'minor', 'major', 'critical'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "local_qty": self.local_qty,
            "broker_qty": self.broker_qty,
            "local_avg_price": self.local_avg_price,
            "broker_avg_price": self.broker_avg_price,
            "qty_diff": self.qty_diff,
            "price_diff": self.price_diff,
            "timestamp": self.timestamp,
            "severity": self.severity,
        }


@dataclass
class ReconcileResult:
    """Result of a reconciliation check."""
    success: bool
    timestamp: str
    positions_checked: int
    mismatches: List[PositionMismatch] = field(default_factory=list)
    broker_positions: int = 0
    local_positions: int = 0
    cash_match: bool = True
    local_cash: float = 0.0
    broker_cash: float = 0.0
    action_taken: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "timestamp": self.timestamp,
            "positions_checked": self.positions_checked,
            "mismatches": [m.to_dict() for m in self.mismatches],
            "broker_positions": self.broker_positions,
            "local_positions": self.local_positions,
            "cash_match": self.cash_match,
            "action_taken": self.action_taken,
        }


@dataclass
class ReconcilerConfig:
    """Configuration for state reconciler."""
    # Thresholds
    qty_tolerance: float = 0.001          # Qty diff tolerance (for fractional shares)
    price_tolerance: float = 0.01         # Price diff tolerance ($)
    cash_tolerance: float = 1.0           # Cash diff tolerance ($)

    # Behavior
    reconcile_interval: int = 15          # Seconds between periodic checks (reduced from 60)
    stale_threshold_seconds: int = 30     # Portfolio state staleness threshold
    halt_on_critical: bool = True         # Halt trading on critical mismatch
    auto_correct_minor: bool = True       # Auto-sync minor mismatches

    # Severity thresholds
    minor_qty_diff: float = 1.0           # Qty diff for 'minor'
    major_qty_diff: float = 10.0          # Qty diff for 'major', above is 'critical'

    # Order verification
    order_verify_timeout: float = 30.0    # Seconds to wait for order fill
    order_verify_retries: int = 3         # Retries for order verification


class StateReconciler:
    """
    Reconciles local trading state with broker state.

    Ensures the system's view of positions matches reality at the broker.
    Critical for production trading to prevent dangerous state drift.
    """

    def __init__(
        self,
        broker: "BaseBrokerInterface",
        portfolio: "PortfolioState",
        config: Optional[ReconcilerConfig] = None,
        on_mismatch: Optional[Callable[[List[PositionMismatch]], None]] = None,
        on_halt: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the state reconciler.

        Args:
            broker: Broker interface for fetching actual positions
            portfolio: Local portfolio state to reconcile
            config: Reconciler configuration
            on_mismatch: Callback when mismatches detected
            on_halt: Callback when trading should halt
        """
        self.broker = broker
        self.portfolio = portfolio
        self.config = config or ReconcilerConfig()
        self.on_mismatch = on_mismatch
        self.on_halt = on_halt

        self.event_handler = get_event_handler()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._halted = False
        self._last_result: Optional[ReconcileResult] = None

        # Statistics
        self._sync_count = 0
        self._mismatch_count = 0
        self._auto_correct_count = 0

        logger.info(
            f"StateReconciler initialized: interval={self.config.reconcile_interval}s, "
            f"halt_on_critical={self.config.halt_on_critical}"
        )

    @property
    def is_halted(self) -> bool:
        """Check if trading is halted due to state mismatch."""
        return self._halted

    @property
    def last_result(self) -> Optional[ReconcileResult]:
        """Get the last reconciliation result."""
        return self._last_result

    # ========================================================================
    # FULL SYNC (Startup)
    # ========================================================================

    async def full_sync(self) -> ReconcileResult:
        """
        Perform full sync from broker - use at startup.

        Fetches all positions from broker and overwrites local state.
        This is the authoritative sync that establishes baseline.

        Returns:
            ReconcileResult with sync details
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        logger.info("Starting full sync from broker...")

        try:
            # Get broker snapshot
            snapshot = await self.broker.get_account_info()

            if snapshot is None:
                logger.error("Failed to get broker snapshot")
                return ReconcileResult(
                    success=False,
                    timestamp=timestamp,
                    positions_checked=0,
                    action_taken="sync_failed",
                )

            # Record pre-sync state for logging
            local_positions = len(self.portfolio.positions)
            broker_positions = len(snapshot.positions) if snapshot.positions else 0

            # Sync portfolio from snapshot
            self.portfolio.sync_from_snapshot(snapshot)

            self._sync_count += 1

            result = ReconcileResult(
                success=True,
                timestamp=timestamp,
                positions_checked=broker_positions,
                broker_positions=broker_positions,
                local_positions=local_positions,
                local_cash=self.portfolio.cash,
                broker_cash=snapshot.cash,
                cash_match=True,
                action_taken="full_sync",
            )

            self._last_result = result

            logger.info(
                f"Full sync complete: {broker_positions} positions, "
                f"${snapshot.cash:,.2f} cash, ${snapshot.equity:,.2f} equity"
            )

            # Emit sync event
            await self._emit_reconcile_event(result)

            return result

        except Exception as e:
            logger.error(f"Full sync failed: {e}")
            return ReconcileResult(
                success=False,
                timestamp=timestamp,
                positions_checked=0,
                action_taken=f"sync_error: {e}",
            )

    # ========================================================================
    # POSITION CHECK (Periodic)
    # ========================================================================

    async def check_positions(self) -> ReconcileResult:
        """
        Check local positions against broker positions.

        Compares each position and identifies mismatches.
        Does NOT auto-correct - use reconcile() for that.

        Returns:
            ReconcileResult with any mismatches found
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        mismatches: List[PositionMismatch] = []

        try:
            # Get broker snapshot
            snapshot = await self.broker.get_account_info()

            if snapshot is None:
                logger.error("Failed to get broker snapshot for check")
                return ReconcileResult(
                    success=False,
                    timestamp=timestamp,
                    positions_checked=0,
                )

            broker_positions = snapshot.positions or {}
            local_positions = self.portfolio.positions

            # Check all symbols (union of both)
            all_symbols = set(broker_positions.keys()) | set(local_positions.keys())

            for symbol in all_symbols:
                broker_pos = broker_positions.get(symbol)
                local_pos = local_positions.get(symbol)

                broker_qty = broker_pos.qty if broker_pos else 0.0
                local_qty = local_pos.qty if local_pos else 0.0
                broker_price = broker_pos.avg_entry_price if broker_pos else 0.0
                local_price = local_pos.avg_price if local_pos else 0.0

                qty_diff = abs(local_qty - broker_qty)
                price_diff = abs(local_price - broker_price)

                # Check if mismatch exceeds tolerance
                if qty_diff > self.config.qty_tolerance:
                    severity = self._classify_severity(qty_diff)

                    mismatch = PositionMismatch(
                        symbol=symbol,
                        local_qty=local_qty,
                        broker_qty=broker_qty,
                        local_avg_price=local_price,
                        broker_avg_price=broker_price,
                        qty_diff=qty_diff,
                        price_diff=price_diff,
                        timestamp=timestamp,
                        severity=severity,
                    )
                    mismatches.append(mismatch)

                    logger.warning(
                        f"Position mismatch [{severity}]: {symbol} "
                        f"local={local_qty} broker={broker_qty} diff={qty_diff}"
                    )

            # Check cash
            cash_diff = abs(self.portfolio.cash - snapshot.cash)
            cash_match = cash_diff <= self.config.cash_tolerance

            if not cash_match:
                logger.warning(
                    f"Cash mismatch: local=${self.portfolio.cash:,.2f} "
                    f"broker=${snapshot.cash:,.2f} diff=${cash_diff:,.2f}"
                )

            result = ReconcileResult(
                success=len(mismatches) == 0 and cash_match,
                timestamp=timestamp,
                positions_checked=len(all_symbols),
                mismatches=mismatches,
                broker_positions=len(broker_positions),
                local_positions=len(local_positions),
                cash_match=cash_match,
                local_cash=self.portfolio.cash,
                broker_cash=snapshot.cash,
            )

            self._last_result = result

            if mismatches:
                self._mismatch_count += len(mismatches)

            return result

        except Exception as e:
            logger.error(f"Position check failed: {e}")
            return ReconcileResult(
                success=False,
                timestamp=timestamp,
                positions_checked=0,
            )

    # ========================================================================
    # RECONCILE (Check + Action)
    # ========================================================================

    async def reconcile(self) -> ReconcileResult:
        """
        Check positions and take action on mismatches.

        Based on configuration:
        - Minor mismatches: auto-correct if enabled
        - Major mismatches: alert
        - Critical mismatches: halt trading if enabled

        Returns:
            ReconcileResult with actions taken
        """
        result = await self.check_positions()

        if result.success:
            logger.debug("Reconciliation passed - no mismatches")
            result.action_taken = "none_needed"
            return result

        # Handle mismatches
        critical_mismatches = [m for m in result.mismatches if m.severity == "critical"]
        major_mismatches = [m for m in result.mismatches if m.severity == "major"]
        minor_mismatches = [m for m in result.mismatches if m.severity == "minor"]

        actions_taken = []

        # Auto-correct minor mismatches
        if minor_mismatches and self.config.auto_correct_minor:
            logger.info(f"Auto-correcting {len(minor_mismatches)} minor mismatches")
            await self.full_sync()
            self._auto_correct_count += len(minor_mismatches)
            actions_taken.append("auto_corrected")

        # Alert on major mismatches
        if major_mismatches:
            await self._emit_alert(
                f"Major position mismatches detected: {len(major_mismatches)} positions",
                "warning",
                result
            )
            actions_taken.append("alerted")

            if self.on_mismatch:
                self.on_mismatch(major_mismatches)

        # Halt on critical mismatches
        if critical_mismatches and self.config.halt_on_critical:
            self._halted = True
            msg = f"CRITICAL: {len(critical_mismatches)} position mismatches - HALTING TRADING"
            logger.critical(msg)

            await self._emit_alert(msg, "error", result)
            actions_taken.append("halted")

            if self.on_halt:
                self.on_halt(msg)

        # Also check cash mismatch
        if not result.cash_match:
            await self._emit_alert(
                f"Cash mismatch: local=${result.local_cash:,.2f} broker=${result.broker_cash:,.2f}",
                "warning",
                result
            )

        result.action_taken = ", ".join(actions_taken) if actions_taken else "none"

        # Emit reconcile event
        await self._emit_reconcile_event(result)

        return result

    # ========================================================================
    # ORDER VERIFICATION
    # ========================================================================

    async def verify_order(
        self,
        order_id: str,
        symbol: str,
        expected_qty: float,
        expected_side: str,  # 'buy' or 'sell'
    ) -> bool:
        """
        Verify an order was filled as expected.

        Polls order status until filled or timeout.

        Args:
            order_id: Broker order ID
            symbol: Symbol traded
            expected_qty: Expected fill quantity
            expected_side: Expected order side

        Returns:
            True if order verified, False if mismatch or timeout
        """
        logger.info(f"Verifying order {order_id}: {expected_side} {expected_qty} {symbol}")

        for attempt in range(self.config.order_verify_retries):
            try:
                order = await self.broker.get_order_status(order_id)

                if order is None:
                    logger.warning(f"Order {order_id} not found (attempt {attempt + 1})")
                    await asyncio.sleep(2)
                    continue

                status = order.status.lower() if order.status else ""

                if status in ("filled", "completed"):
                    filled_qty = order.filled_qty or 0
                    qty_match = abs(filled_qty - expected_qty) <= self.config.qty_tolerance

                    if qty_match:
                        logger.info(f"Order {order_id} verified: filled {filled_qty} {symbol}")
                        return True
                    else:
                        logger.warning(
                            f"Order {order_id} qty mismatch: expected={expected_qty} filled={filled_qty}"
                        )
                        return False

                elif status in ("cancelled", "rejected", "expired"):
                    logger.error(f"Order {order_id} not filled: status={status}")
                    return False

                else:
                    # Still pending
                    logger.debug(f"Order {order_id} status={status}, waiting...")
                    await asyncio.sleep(self.config.order_verify_timeout / self.config.order_verify_retries)

            except Exception as e:
                logger.error(f"Order verification error: {e}")
                await asyncio.sleep(2)

        logger.error(f"Order {order_id} verification timed out")
        return False

    # ========================================================================
    # PERIODIC RECONCILIATION
    # ========================================================================

    async def start_periodic(self, interval_seconds: Optional[int] = None) -> None:
        """
        Start periodic reconciliation in background.

        Args:
            interval_seconds: Override config interval
        """
        if self._running:
            logger.warning("Periodic reconciliation already running")
            return

        interval = interval_seconds or self.config.reconcile_interval
        self._running = True

        logger.info(f"Starting periodic reconciliation every {interval}s")

        self._task = asyncio.create_task(self._periodic_loop(interval))

    async def stop_periodic(self) -> None:
        """Stop periodic reconciliation."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("Periodic reconciliation stopped")

    async def _periodic_loop(self, interval: int) -> None:
        """Background loop for periodic reconciliation."""
        while self._running:
            try:
                await asyncio.sleep(interval)

                if not self._running:
                    break

                logger.debug("Running periodic reconciliation...")
                result = await self.reconcile()

                if result.success:
                    logger.debug("Periodic reconciliation passed")
                else:
                    logger.warning(f"Periodic reconciliation found issues: {result.action_taken}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic reconciliation error: {e}")
                await asyncio.sleep(5)  # Brief pause on error

    # ========================================================================
    # HALT MANAGEMENT
    # ========================================================================

    def clear_halt(self) -> None:
        """Clear halt state after manual review."""
        if self._halted:
            logger.info("Halt cleared by operator")
            self._halted = False

    async def force_sync_and_clear(self) -> ReconcileResult:
        """Force sync from broker and clear halt state."""
        result = await self.full_sync()
        self.clear_halt()
        return result

    # ========================================================================
    # HELPERS
    # ========================================================================

    def _classify_severity(self, qty_diff: float) -> str:
        """Classify mismatch severity based on quantity difference."""
        if qty_diff <= self.config.minor_qty_diff:
            return "minor"
        elif qty_diff <= self.config.major_qty_diff:
            return "major"
        else:
            return "critical"

    async def _emit_alert(
        self,
        message: str,
        level: str,
        result: ReconcileResult
    ) -> None:
        """Emit alert event."""
        payload = {
            "type": "reconciliation",
            "message": message,
            "level": level,
            "timestamp": result.timestamp,
            "details": result.to_dict(),
        }

        try:
            await self.event_handler.emit(events.EVENT_ALERT, payload)
        except Exception as e:
            logger.error(f"Failed to emit alert: {e}")

    async def _emit_reconcile_event(self, result: ReconcileResult) -> None:
        """Emit reconciliation result event."""
        payload = {
            "type": "reconcile_result",
            "success": result.success,
            "timestamp": result.timestamp,
            "positions_checked": result.positions_checked,
            "mismatch_count": len(result.mismatches),
            "action_taken": result.action_taken,
        }

        try:
            # Use alert event for now, could add dedicated event type
            if not result.success:
                await self.event_handler.emit(events.EVENT_ALERT, {
                    "type": "reconciliation",
                    "message": f"Reconciliation: {result.action_taken}",
                    "level": "info" if result.success else "warning",
                    "timestamp": result.timestamp,
                })
        except Exception as e:
            logger.error(f"Failed to emit reconcile event: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get reconciler statistics."""
        return {
            "sync_count": self._sync_count,
            "mismatch_count": self._mismatch_count,
            "auto_correct_count": self._auto_correct_count,
            "is_halted": self._halted,
            "is_running": self._running,
            "last_result": self._last_result.to_dict() if self._last_result else None,
        }
