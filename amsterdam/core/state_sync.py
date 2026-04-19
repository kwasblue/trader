"""
State Synchronizer - Ensures Portfolio and Symbol states stay in sync.

Problem: Position state is tracked in both PortfolioState._position_states AND
SymbolState.position_state. They can diverge. Also, SymbolState.current_position
is not updated when PortfolioState.apply_fill_safe() is called.

This module provides a centralized synchronizer that:
1. Applies fills to both Portfolio and Symbol states atomically
2. Propagates position changes from Portfolio to Symbol states
3. Ensures consistency during concurrent operations
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loggers.logger import Logger

if TYPE_CHECKING:
    from core.logic.portfolio_state import PortfolioState
    from core.logic.symbol_state import SymbolState


class StateSynchronizer:
    """
    Synchronizes state between PortfolioState and SymbolState.

    Ensures that:
    - Fills are applied to both Portfolio and Symbol atomically
    - Position changes in Portfolio are reflected in Symbol states
    - State consistency is maintained during concurrent operations

    Usage:
        sync = StateSynchronizer(portfolio, symbol_states)

        # Apply fill and sync to symbol state
        await sync.apply_fill_and_sync("AAPL", "buy", 100, 150.25)

        # Sync all symbol states from portfolio
        await sync.sync_all_from_portfolio()
    """

    def __init__(
        self,
        portfolio: PortfolioState,
        symbol_states: dict[str, SymbolState],
    ):
        """
        Initialize the state synchronizer.

        Args:
            portfolio: Portfolio state instance
            symbol_states: Dictionary mapping symbols to their SymbolState
        """
        self.portfolio = portfolio
        self.symbol_states = symbol_states

        # Synchronization lock to prevent race conditions
        self._sync_lock = asyncio.Lock()

        # Logger
        self.logger = Logger(log_file="state_sync.log", logger_name="StateSynchronizer", propagate=True).get_logger()

        self.logger.info(f"StateSynchronizer initialized with {len(symbol_states)} symbol states")

    async def apply_fill_and_sync(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
    ) -> None:
        """
        Apply fill to portfolio AND sync to symbol state.

        This is the primary method for applying fills. It ensures both
        Portfolio and Symbol states are updated atomically.

        Args:
            symbol: Trading symbol
            side: Order side ("buy" or "sell")
            qty: Quantity filled
            price: Fill price
        """
        async with self._sync_lock:
            # 1. Apply fill to portfolio (thread-safe)
            await self.portfolio.apply_fill_safe(symbol, side, qty, price)

            # 2. Sync to symbol state
            await self._sync_symbol_from_portfolio(symbol)

            self.logger.debug(f"[{symbol}] Fill applied and synced: {side} {qty}@${price:.2f}")

    async def _sync_symbol_from_portfolio(self, symbol: str) -> None:
        """
        Sync a single symbol's state from portfolio.

        Updates SymbolState.current_position and related fields
        to match PortfolioState.

        Args:
            symbol: Trading symbol to sync
        """
        position = self.portfolio.get_position(symbol)
        state = self.symbol_states.get(symbol)

        if state is None:
            # No symbol state to sync
            return

        if position:
            # Update symbol state from portfolio position
            state.update_from_portfolio(qty=position.qty, avg_price=position.avg_price)
            self.logger.debug(f"[{symbol}] Symbol state synced: qty={position.qty}, avg=${position.avg_price:.2f}")
        else:
            # Position closed - reset symbol state
            if state.current_position != 0:
                state.current_position = 0
                state.side = None
                state.entry_price = None
                self.logger.debug(f"[{symbol}] Symbol state reset (position closed)")

    async def sync_all_from_portfolio(self) -> int:
        """
        Sync all symbol states from portfolio.

        Useful after broker reconciliation or startup to ensure
        all states are consistent.

        Returns:
            Number of symbols synced
        """
        async with self._sync_lock:
            synced_count = 0

            for symbol in self.symbol_states:
                await self._sync_symbol_from_portfolio(symbol)
                synced_count += 1

            self.logger.info(f"Synced {synced_count} symbol states from portfolio")
            return synced_count

    async def sync_from_broker_snapshot(
        self,
        broker_positions: dict[str, any],
    ) -> int:
        """
        Sync symbol states from broker position snapshot.

        Called after reconciliation to update all states.

        Args:
            broker_positions: Dictionary of broker positions

        Returns:
            Number of symbols updated
        """
        async with self._sync_lock:
            updated_count = 0

            for symbol, pos in broker_positions.items():
                state = self.symbol_states.get(symbol)
                if state:
                    qty = getattr(pos, "qty", 0)
                    avg_price = getattr(pos, "avg_price", 0.0)
                    state.update_from_portfolio(qty=qty, avg_price=avg_price)
                    updated_count += 1
                    self.logger.debug(f"[{symbol}] Updated from broker: qty={qty}, avg=${avg_price:.2f}")

            self.logger.info(f"Synced {updated_count} symbol states from broker snapshot")
            return updated_count

    def register_symbol(self, symbol: str, state: SymbolState) -> None:
        """
        Register a new symbol state for synchronization.

        Args:
            symbol: Trading symbol
            state: SymbolState instance
        """
        self.symbol_states[symbol] = state
        self.logger.debug(f"[{symbol}] Registered for state sync")

    def unregister_symbol(self, symbol: str) -> None:
        """
        Unregister a symbol from synchronization.

        Args:
            symbol: Trading symbol
        """
        if symbol in self.symbol_states:
            del self.symbol_states[symbol]
            self.logger.debug(f"[{symbol}] Unregistered from state sync")

    async def verify_consistency(self) -> dict[str, dict]:
        """
        Verify that Portfolio and Symbol states are consistent.

        Returns:
            Dictionary of inconsistencies found, empty if all consistent
        """
        async with self._sync_lock:
            inconsistencies = {}

            for symbol, state in self.symbol_states.items():
                position = self.portfolio.get_position(symbol)

                # Check position quantity consistency
                portfolio_qty = position.qty if position else 0
                symbol_qty = state.current_position

                if portfolio_qty != symbol_qty:
                    inconsistencies[symbol] = {
                        "type": "qty_mismatch",
                        "portfolio_qty": portfolio_qty,
                        "symbol_qty": symbol_qty,
                    }
                    self.logger.warning(
                        f"[{symbol}] Inconsistency: portfolio_qty={portfolio_qty}, symbol_qty={symbol_qty}"
                    )

                # Check average price consistency
                if position and state.entry_price:
                    if abs(position.avg_price - state.entry_price) > 0.01:
                        if symbol not in inconsistencies:
                            inconsistencies[symbol] = {}
                        inconsistencies[symbol]["price_mismatch"] = {
                            "portfolio_avg": position.avg_price,
                            "symbol_entry": state.entry_price,
                        }

            if not inconsistencies:
                self.logger.info("State consistency check passed")
            else:
                self.logger.warning(f"Found {len(inconsistencies)} state inconsistencies")

            return inconsistencies

    async def fix_inconsistencies(self) -> int:
        """
        Fix any state inconsistencies by syncing from portfolio.

        Portfolio is considered the source of truth.

        Returns:
            Number of fixes applied
        """
        inconsistencies = await self.verify_consistency()

        if not inconsistencies:
            return 0

        fixed_count = 0
        for symbol in inconsistencies:
            await self._sync_symbol_from_portfolio(symbol)
            fixed_count += 1
            self.logger.info(f"[{symbol}] State inconsistency fixed")

        return fixed_count
