"""
Meta Trade Types - Dataclasses for ML-ready trade logging.

These types capture entry conditions and trade outcomes for training
a meta-model that can improve trade selection.

Usage:
    from core.contracts.meta_types import TradeEntryContext, TradeExitContext
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass
class TradeEntryContext:
    """
    Captures all relevant context at trade entry time.

    Used to log features that can be correlated with trade outcomes
    for ML model training.

    Example:
        entry = TradeEntryContext(
            trade_id="20240224_AAPL_001",
            timestamp=datetime.now(),
            symbol="AAPL",
            side="buy",
            qty=50,
            price=150.25,
            strategy="momentum",
            regime="normal",
            atr=2.34,
            atr_percentile=0.72,
            drawdown_portfolio_pct=-0.02,
            drawdown_symbol_pct=-0.01,
            position_size_pct=0.05,
            hour_of_day=15,
            day_of_week=3,
            minutes_since_open=360,
            bars_in_regime=12,
            hours_since_last_trade=4.5,
            signal_strength=1,
        )
    """

    # Trade identification
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # "buy" or "sell"
    qty: int
    price: float

    # Strategy context
    strategy: str
    regime: str

    # Volatility features
    atr: float
    atr_percentile: float  # Percentile of current ATR in recent history

    # Risk state
    drawdown_portfolio_pct: float  # Current portfolio drawdown (negative)
    drawdown_symbol_pct: float  # Current symbol-level drawdown (negative)
    position_size_pct: float  # Position size as % of portfolio

    # Time features
    hour_of_day: int  # 0-23
    day_of_week: int  # 0=Monday, 6=Sunday
    minutes_since_open: int  # Minutes since market open (9:30 ET)

    # Trade timing features
    bars_in_regime: int  # How long current regime has persisted
    hours_since_last_trade: float  # Time since last trade for this symbol
    signal_strength: int  # Signal value (-1, 0, 1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert datetime to ISO string
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class TradeExitContext:
    """
    Captures trade outcome at exit time.

    Links back to entry via trade_id for joining entry features
    with outcomes during training.

    Example:
        exit = TradeExitContext(
            trade_id="20240224_AAPL_001",
            timestamp=datetime.now(),
            price=152.10,
            pnl_dollars=92.50,
            pnl_percent=0.0123,
            hold_bars=75,
            mae_percent=-0.008,
            mfe_percent=0.018,
            exit_reason="take_profit",
        )
    """

    # Trade identification
    trade_id: str
    timestamp: datetime

    # Exit details
    price: float
    pnl_dollars: float
    pnl_percent: float

    # Hold duration
    hold_bars: int

    # Excursion metrics
    mae_percent: float  # Max Adverse Excursion (worst drawdown during trade)
    mfe_percent: float  # Max Favorable Excursion (best profit during trade)

    # Exit classification
    exit_reason: str  # "take_profit", "stop_loss", "signal_reversal", "time_exit", etc.

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert datetime to ISO string
        d["timestamp"] = self.timestamp.isoformat()
        return d


__all__ = [
    "TradeEntryContext",
    "TradeExitContext",
]
