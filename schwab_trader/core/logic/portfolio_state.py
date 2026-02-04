"""
Portfolio State - Track positions, cash, and P&L

Maintains real-time portfolio state including:
- Cash balance
- Open positions
- Realized/unrealized P&L
- Equity history
- Drawdown tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timezone
from loggers.logger import Logger

from core.app_types import BrokerSnapshot, OrderResult
from core.enums import OrderSide




@dataclass
class SymbolPosition:
    """
    Represents a position in a single symbol.
    
    Attributes:
        qty: Position quantity (+ for long, - for short, 0 for flat)
        avg_price: Average entry price
        last_price: Most recent market price
        unrealized_pnl: Current unrealized profit/loss
        realized_pnl: Realized P&L from this position
        entry_time: When position was opened
    """
    qty: int = 0
    avg_price: float = 0.0
    last_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    entry_time: Optional[datetime] = None
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.qty > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.qty < 0
    
    @property
    def is_flat(self) -> bool:
        """Check if position is flat."""
        return self.qty == 0
    
    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return abs(self.qty) * self.last_price
    
    @property
    def cost_basis(self) -> float:
        """Cost basis of position."""
        return abs(self.qty) * self.avg_price
    
    def update_unrealized_pnl(self) -> None:
        """Recalculate unrealized P&L based on current price."""
        if self.qty == 0:
            self.unrealized_pnl = 0.0
        elif self.qty > 0:
            # Long position
            self.unrealized_pnl = (self.last_price - self.avg_price) * self.qty
        else:
            # Short position
            self.unrealized_pnl = (self.avg_price - self.last_price) * abs(self.qty)
    
    def __repr__(self) -> str:
        side = "LONG" if self.is_long else "SHORT" if self.is_short else "FLAT"
        return (
            f"Position({side} qty={self.qty}, avg=${self.avg_price:.2f}, "
            f"last=${self.last_price:.2f}, upnl=${self.unrealized_pnl:.2f})"
        )


@dataclass
class PortfolioState:
    """
    Portfolio state tracker.

    Tracks:
    - Cash balance
    - Open positions
    - Realized/unrealized P&L
    - Equity history for metrics
    - Total portfolio value
    - State freshness (last sync/update times)

    Example:
        portfolio = PortfolioState(initial_cash=100000)

        # Apply trade
        portfolio.apply_fill("AAPL", "buy", 100, 150.25)

        # Update price
        portfolio.update_price("AAPL", 152.50)

        # Check metrics
        print(f"Equity: ${portfolio.total_equity():,.2f}")
        print(f"Unrealized: ${portfolio.total_unrealized():,.2f}")
        print(f"Drawdown: {portfolio.current_drawdown():.2%}")

        # Check staleness
        if portfolio.is_stale():
            await reconciler.full_sync()
    """

    cash: float = 100_000.0
    positions: Dict[str, SymbolPosition] = field(default_factory=dict)
    realized_pnl: float = 0.0
    equity_history: List[float] = field(default_factory=list)
    total_value: float = 100_000.0  # Cached total value

    # State freshness tracking
    last_sync_time: Optional[datetime] = None  # Last full sync with broker
    last_update_time: Optional[datetime] = None  # Last any update (trade, price, sync)

    # Logger - own file with propagation to app.log
    port_logger = Logger(
        log_file="portfolio_state.log",
        logger_name="PortfolioState",
        propagate=True
    ).get_logger()
    
    
    def __post_init__(self):
        """Initialize equity history with starting cash."""
        if not self.equity_history:
            self.equity_history.append(self.cash)
            self.total_value = self.cash
        self.port_logger.info(f"PortfolioState initialized: cash=${self.cash:,.2f}")

    # ========================================================================
    # STATE FRESHNESS METHODS
    # ========================================================================

    def is_stale(self, threshold_seconds: float = 30.0) -> bool:
        """
        Check if portfolio state is stale (not recently updated).

        Used by reconciler and execution engine to detect when local state
        may have drifted from broker state.

        Args:
            threshold_seconds: Number of seconds before considering state stale

        Returns:
            True if last update was more than threshold_seconds ago,
            or if never updated
        """
        if self.last_update_time is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self.last_update_time).total_seconds()
        return elapsed > threshold_seconds

    def mark_updated(self) -> None:
        """
        Mark portfolio as recently updated.

        Called after:
        - Price updates
        - Trade fills
        - Position changes

        This resets the staleness timer.
        """
        self.last_update_time = datetime.now(timezone.utc)

    def mark_synced(self) -> None:
        """
        Mark portfolio as synced with broker.

        Called after:
        - Full broker sync
        - Reconciliation

        Updates both sync and update times.
        """
        now = datetime.now(timezone.utc)
        self.last_sync_time = now
        self.last_update_time = now
        self.port_logger.debug("Portfolio marked as synced")
            
    
    # ========================================================================
    # PROPERTIES
    # ========================================================================
    
    @property
    def equity(self) -> float:
        """
        Current total equity (convenience property).
        
        Returns:
            Latest equity value from history
        """
        if self.equity_history:
            return self.equity_history[-1]
        return self.total_equity()
    
    @property
    def num_positions(self) -> int:
        """Number of open positions."""
        return sum(1 for p in self.positions.values() if not p.is_flat)
    
    @property
    def long_positions(self) -> Dict[str, SymbolPosition]:
        """Dictionary of long positions."""
        return {sym: pos for sym, pos in self.positions.items() if pos.is_long}
    
    @property
    def short_positions(self) -> Dict[str, SymbolPosition]:
        """Dictionary of short positions."""
        return {sym: pos for sym, pos in self.positions.items() if pos.is_short}
    
    # ========================================================================
    # CORE METHODS
    # ========================================================================
    
    def update_price(self, symbol: str, price: float) -> None:
        """
        Update market price for a symbol.

        Recalculates unrealized P&L and tracks equity.

        Args:
            symbol: Trading symbol
            price: New market price
        """
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        pos.last_price = float(price)
        pos.update_unrealized_pnl()

        # Update total value and history
        self.total_value = self.total_equity()
        self.equity_history.append(self.total_value)

        # Mark as updated
        self.mark_updated()
    
    def apply_fill(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float
    ) -> None:
        """
        Apply a trade fill to the portfolio.
        
        Updates:
        - Position quantity and average price
        - Cash balance
        - Realized P&L (if closing/reducing)
        - Equity history
        
        Args:
            symbol: Trading symbol
            side: Order side ("buy", "sell", "long", "short", etc.)
            qty: Quantity traded
            price: Fill price
            
        Raises:
            ValueError: If side is invalid or qty is non-positive
        """
        qty = int(qty)
        if qty <= 0:
            raise ValueError(f"Quantity must be positive, got {qty}")
        
        # Normalize side
        side = side.lower().strip()
        if side in ("long", "buy", "cover"):
            trade_qty = +qty
        elif side in ("short", "sell"):
            trade_qty = -qty
        else:
            raise ValueError(f"Unknown side: {side}")
        
        # Get or create position
        pos = self.positions.setdefault(symbol, SymbolPosition())
        px = float(price)
        
        old_qty = pos.qty
        new_qty = old_qty + trade_qty
        
        # Calculate realized P&L for closing/reducing trades
        realized_from_trade = self._calculate_realized_pnl(
            old_qty, trade_qty, pos.avg_price, px
        )
        self.realized_pnl += realized_from_trade
        pos.realized_pnl += realized_from_trade
        
        # Update cash
        self.cash -= trade_qty * px
        
        # Update position
        self._update_position(pos, old_qty, new_qty, trade_qty, px)
        
        # Track equity
        self.total_value = self.total_equity()
        self.equity_history.append(self.total_value)

        # Mark as updated
        self.mark_updated()

        self.port_logger.info(
            f"[{symbol}] Fill applied: {side} {qty}@${px:.2f}, "
            f"new_qty={new_qty}, cash=${self.cash:,.2f}, "
            f"realized_pnl=${realized_from_trade:.2f}"
        )
    
    def update_position(self, symbol: str, order_result: OrderResult) -> None:
        """
        Update position from OrderResult (convenience method).
        
        Args:
            symbol: Trading symbol
            order_result: Result from broker execution
        """
        side = "buy" if order_result.side == OrderSide.BUY else "sell"
        self.apply_fill(
            symbol=symbol,
            side=side,
            qty=order_result.filled_qty,
            price=order_result.avg_price
        )
    
    # ========================================================================
    # P&L CALCULATIONS
    # ========================================================================
    
    def _calculate_realized_pnl(
        self,
        old_qty: int,
        trade_qty: int,
        avg_price: float,
        fill_price: float
    ) -> float:
        """
        Calculate realized P&L from a trade.
        
        Returns:
            Realized P&L from closing/reducing position
        """
        # Closing long position
        if old_qty > 0 and trade_qty < 0:
            closed_qty = min(old_qty, -trade_qty)
            return (fill_price - avg_price) * closed_qty
        
        # Closing short position
        elif old_qty < 0 and trade_qty > 0:
            closed_qty = min(-old_qty, trade_qty)
            return (avg_price - fill_price) * closed_qty
        
        # Not closing/reducing
        return 0.0
    
    def total_unrealized(self) -> float:
        """
        Calculate total unrealized P&L across all positions.
        
        Returns:
            Sum of unrealized P&L for all positions
        """
        total = 0.0
        for pos in self.positions.values():
            if not pos.is_flat:
                pos.update_unrealized_pnl()
                total += pos.unrealized_pnl
        return total
    
    def unrealized_pnl(self, symbol: str) -> float:
        """
        Get unrealized P&L for specific symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Unrealized P&L for symbol
        """
        if symbol not in self.positions:
            return 0.0
        
        pos = self.positions[symbol]
        pos.update_unrealized_pnl()
        return pos.unrealized_pnl
    
    def total_equity(self) -> float:
        """
        Calculate total portfolio equity.
        
        Returns:
            Cash + unrealized P&L
        """
        return self.cash + self.total_unrealized()
    
    def total_realized(self) -> float:
        """
        Get total realized P&L.
        
        Returns:
            Total realized P&L from closed positions
        """
        return self.realized_pnl
    
    # ========================================================================
    # POSITION MANAGEMENT
    # ========================================================================
    
    def _update_position(
        self,
        pos: SymbolPosition,
        old_qty: int,
        new_qty: int,
        trade_qty: int,
        price: float
    ) -> None:
        """
        Update position quantity and average price.
        
        Handles:
        - Opening new position
        - Adding to position
        - Reducing position
        - Closing position
        - Flipping position (long -> short or vice versa)
        """
        # Set entry time if opening new position
        if old_qty == 0 and new_qty != 0:
            pos.entry_time = datetime.now(timezone.utc)
        
        # Same direction (adding to position)
        if old_qty == 0 or (old_qty > 0 and new_qty > 0) or (old_qty < 0 and new_qty < 0):
            if old_qty == 0:
                # New position
                pos.avg_price = price
            else:
                # Adding to existing position
                total_cost = (pos.avg_price * abs(old_qty)) + (price * abs(trade_qty))
                total_qty = abs(old_qty) + abs(trade_qty)
                pos.avg_price = total_cost / total_qty
            pos.qty = new_qty
        
        # Reducing or closing position
        elif (old_qty > 0 and new_qty >= 0) or (old_qty < 0 and new_qty <= 0):
            pos.qty = new_qty
            if new_qty == 0:
                # Position closed
                pos.avg_price = 0.0
                pos.entry_time = None
        
        # Flipping position (long -> short or short -> long)
        else:
            pos.qty = new_qty
            pos.avg_price = price
            pos.entry_time = datetime.now(timezone.utc)
        
        # Always update last price
        pos.last_price = price
        pos.update_unrealized_pnl()
    
    # ========================================================================
    # METRICS
    # ========================================================================
    
    def current_drawdown(self) -> float:
        """
        Calculate current drawdown from peak equity.
        
        Returns:
            Drawdown as decimal (negative value, e.g., -0.05 for 5% drawdown)
        """
        if not self.equity_history:
            return 0.0
        
        peak = max(self.equity_history)
        current = self.equity_history[-1]
        
        if peak == 0:
            return 0.0
        
        return (current - peak) / peak
    
    def drawdown(self) -> float:
        """Alias for current_drawdown()."""
        return self.current_drawdown()
    
    def max_drawdown(self) -> float:
        """
        Calculate maximum drawdown from equity history.
        
        Returns:
            Maximum drawdown as decimal (negative value)
        """
        if len(self.equity_history) < 2:
            return 0.0
        
        peak = self.equity_history[0]
        max_dd = 0.0
        
        for equity in self.equity_history:
            if equity > peak:
                peak = equity
            
            dd = (equity - peak) / peak if peak > 0 else 0.0
            max_dd = min(max_dd, dd)
        
        return max_dd
    
    # ========================================================================
    # BROKER SYNCHRONIZATION
    # ========================================================================
    
    def sync_from_snapshot(self, snap: BrokerSnapshot) -> None:
        """
        Sync portfolio state from broker snapshot.
        
        Used to synchronize internal state with live broker.
        Critical for live trading to avoid state drift.
        
        Args:
            snap: Broker snapshot containing current state
        """
        self.cash = snap.cash
        
        # Clear and rebuild positions
        self.positions.clear()
        for symbol, pos_value in snap.positions.items():
            self.positions[symbol] = SymbolPosition(
                qty=pos_value.qty,
                avg_price=pos_value.avg_price,
                last_price=pos_value.last_price
            )
            self.positions[symbol].update_unrealized_pnl()
        
        # Update equity history
        portfolio_value = snap.portfolio_value or self.total_equity()
        self.total_value = portfolio_value
        self.equity_history.append(portfolio_value)

        # Mark as synced
        self.mark_synced()

        self.port_logger.info(
            f"Portfolio synced from broker: ${portfolio_value:,.2f}, "
            f"{len(self.positions)} positions"
        )
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_position(self, symbol: str) -> Optional[SymbolPosition]:
        """
        Get position for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position or None if not found
        """
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """
        Check if symbol has open position.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if position exists and is not flat
        """
        pos = self.positions.get(symbol)
        return pos is not None and not pos.is_flat
    
    def reset(self) -> None:
        """
        Reset portfolio to initial state.
        
        WARNING: Clears all positions and history!
        """
        initial_cash = self.equity_history[0] if self.equity_history else self.cash
        self.cash = initial_cash
        self.positions.clear()
        self.realized_pnl = 0.0
        self.equity_history = [initial_cash]
        self.total_value = initial_cash
        
        self.port_logger.warning("Portfolio reset to initial state")
    
    def __repr__(self) -> str:
        return (
            f"Portfolio(equity=${self.equity:,.2f}, cash=${self.cash:,.2f}, "
            f"positions={self.num_positions}, unrealized=${self.total_unrealized():,.2f})"
        )