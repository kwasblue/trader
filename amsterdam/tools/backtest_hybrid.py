#!/usr/bin/env python3
"""
Backtest script to compare current logic vs hybrid logic.

This script runs historical simulations comparing two strategies:
1. Current logic: No trend filter, full position sizes
2. Hybrid logic: Confidence + trend-based position sizing

The hybrid approach uses daily indicators for trend context while
executing on minute bars for intraday trading.

Usage:
    # Run comparison with default settings
    python tools/backtest_hybrid.py --compare

    # Run with specific symbols
    python tools/backtest_hybrid.py --symbols AAPL MSFT GOOGL --compare

    # Run with custom date range
    python tools/backtest_hybrid.py --days 90 --compare

    # Run only hybrid mode (for quick testing)
    python tools/backtest_hybrid.py --hybrid-only

Output:
    Comparison report showing:
    - Total trades
    - Win rate
    - Total P&L
    - Max drawdown
    - Sharpe ratio
    - Trade distribution by confidence/trend
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import math

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.logic.hybrid_position_sizer import HybridPositionSizer, HybridSizingResult


@dataclass
class TradeRecord:
    """Record of a single trade for analysis."""
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    side: str  # "long" or "short"
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    pnl: float = 0.0
    pnl_percent: float = 0.0
    confidence: float = 1.0
    trend: str = "neutral"
    aligned: bool = True
    confidence_tier: str = "high"
    sizing_multiplier: float = 1.0


@dataclass
class BacktestResult:
    """Results from a single backtest run."""
    name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    avg_position_size: float = 0.0
    trades: List[TradeRecord] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        """Calculate win rate as percentage."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100


@dataclass
class ComparisonReport:
    """Comparison of current vs hybrid backtest results."""
    current: BacktestResult
    hybrid: BacktestResult
    period_start: datetime
    period_end: datetime
    symbols: List[str]

    def print_report(self) -> None:
        """Print formatted comparison report."""
        print()
        print("=" * 80)
        print("           BACKTEST COMPARISON: Current vs Hybrid")
        print("=" * 80)
        print()
        print(f"Period: {self.period_start.strftime('%Y-%m-%d')} to {self.period_end.strftime('%Y-%m-%d')}")
        print(f"Symbols: {', '.join(self.symbols[:5])}{'...' if len(self.symbols) > 5 else ''} ({len(self.symbols)} total)")
        print()
        print(f"{'Metric':<25} {'CURRENT':>15} {'HYBRID':>15} {'CHANGE':>15}")
        print("-" * 80)

        # Total trades
        trades_change = self.hybrid.total_trades - self.current.total_trades
        trades_pct = (trades_change / self.current.total_trades * 100) if self.current.total_trades > 0 else 0
        print(f"{'Total Trades':<25} {self.current.total_trades:>15} {self.hybrid.total_trades:>15} {trades_pct:>+14.0f}%")

        # Win rate
        wr_change = self.hybrid.win_rate - self.current.win_rate
        print(f"{'Win Rate':<25} {self.current.win_rate:>14.1f}% {self.hybrid.win_rate:>14.1f}% {wr_change:>+14.1f}%")

        # Total P&L
        pnl_change = self.hybrid.total_pnl - self.current.total_pnl
        print(f"{'Total P&L':<25} ${self.current.total_pnl:>13,.0f} ${self.hybrid.total_pnl:>13,.0f} ${pnl_change:>+13,.0f}")

        # Max drawdown
        dd_change = self.hybrid.max_drawdown - self.current.max_drawdown
        print(f"{'Max Drawdown':<25} {self.current.max_drawdown:>14.1f}% {self.hybrid.max_drawdown:>14.1f}% {dd_change:>+14.1f}%")

        # Sharpe ratio
        sharpe_change = self.hybrid.sharpe_ratio - self.current.sharpe_ratio
        print(f"{'Sharpe Ratio':<25} {self.current.sharpe_ratio:>15.2f} {self.hybrid.sharpe_ratio:>15.2f} {sharpe_change:>+15.2f}")

        # Avg position size
        pos_change = (self.hybrid.avg_position_size / self.current.avg_position_size - 1) * 100 if self.current.avg_position_size > 0 else 0
        print(f"{'Avg Position Size':<25} ${self.current.avg_position_size:>13,.0f} ${self.hybrid.avg_position_size:>13,.0f} {pos_change:>+14.0f}%")

        # Print trade distribution for hybrid
        if self.hybrid.trades:
            print()
            print("TRADE DISTRIBUTION (Hybrid):")
            self._print_trade_distribution()

        # Recommendation
        print()
        if self.hybrid.sharpe_ratio > self.current.sharpe_ratio and self.hybrid.max_drawdown < self.current.max_drawdown:
            print("RECOMMENDATION: Hybrid logic shows improvement. Consider enabling.")
        elif self.hybrid.sharpe_ratio > self.current.sharpe_ratio:
            print("RECOMMENDATION: Hybrid shows better returns but similar drawdown. Consider testing further.")
        elif self.hybrid.max_drawdown < self.current.max_drawdown:
            print("RECOMMENDATION: Hybrid shows lower risk but similar returns. Good for risk reduction.")
        else:
            print("RECOMMENDATION: Current logic performing better. Keep hybrid disabled.")

        print("=" * 80)
        print()

    def _print_trade_distribution(self) -> None:
        """Print distribution of trades by confidence/alignment."""
        # Group trades by (confidence_tier, aligned)
        groups: Dict[tuple, List[TradeRecord]] = {}
        for trade in self.hybrid.trades:
            key = (trade.confidence_tier, trade.aligned)
            if key not in groups:
                groups[key] = []
            groups[key].append(trade)

        # Print each group
        labels = {
            ("high", True): "High conf + with trend",
            ("high", False): "High conf + against trend",
            ("med", True): "Med conf + with trend",
            ("med", False): "Med conf + against trend",
            ("low", True): "Low conf + with trend",
            ("low", False): "Low conf + against trend",
        }

        for key, label in labels.items():
            trades = groups.get(key, [])
            if not trades:
                count = 0
                win_rate = 0.0
            else:
                count = len(trades)
                winners = sum(1 for t in trades if t.pnl > 0)
                win_rate = (winners / count) * 100 if count > 0 else 0

            status = "(blocked)" if key == ("low", False) else ""
            print(f"  {label + ':':<30} {count:>4} trades ({win_rate:>5.1f}% win rate) {status}")


class HybridBacktester:
    """
    Runs historical simulation comparing:
    1. Current logic (no trend filter)
    2. Hybrid logic (confidence + trend sizing)
    """

    def __init__(
        self,
        symbols: List[str],
        data_path: str = "data/data_storage/proc_data",
        starting_cash: float = 100_000.0,
    ):
        """
        Initialize the backtester.

        Args:
            symbols: List of symbols to backtest
            data_path: Path to historical data files
            starting_cash: Starting cash for simulation
        """
        self.symbols = symbols
        self.data_path = Path(data_path)
        self.starting_cash = starting_cash

        # Initialize hybrid sizer
        self.hybrid_sizer = HybridPositionSizer(
            enabled=True,
            config={
                "high_confidence_threshold": 0.7,
                "low_confidence_threshold": 0.5,
            }
        )

    def load_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Load historical bars for a symbol."""
        # Try processed data first
        proc_file = self.data_path / f"proc_{symbol}_file.json"
        if proc_file.exists():
            with open(proc_file, "r") as f:
                data = json.load(f)
                return data.get("bars", data) if isinstance(data, dict) else data

        # Fall back to raw data
        raw_file = self.data_path / f"raw_{symbol}_file.json"
        if raw_file.exists():
            with open(raw_file, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data.get("candles", data.get("bars", []))
                return data

        print(f"Warning: No data file found for {symbol}")
        return []

    def generate_mock_signals(self, bars: List[Dict]) -> List[Dict[str, Any]]:
        """
        Generate mock trading signals from bar data.

        Uses simple RSI crossover strategy for signal generation.
        Real backtesting would use actual strategy signals.
        """
        signals = []

        for i, bar in enumerate(bars):
            if i < 14:  # Need RSI warmup
                continue

            # Get indicators if available
            rsi = bar.get("RSI", 50)
            close = bar.get("Close", bar.get("close", 0))

            # Simple RSI strategy for mock signals
            signal = 0
            confidence = 0.5

            if rsi < 30:
                signal = 1  # Buy oversold
                confidence = min(0.9, (30 - rsi) / 30 + 0.5)
            elif rsi > 70:
                signal = -1  # Sell overbought
                confidence = min(0.9, (rsi - 70) / 30 + 0.5)

            # Build daily context from bar indicators
            daily_context = {
                "RSI": rsi,
                "MACD": bar.get("MACD", 0),
                "MACD_Signal": bar.get("MACD_Signal", 0),
                "SMA_200": bar.get("SMA_200", close),
                "Close": close,
            }

            signals.append({
                "bar": bar,
                "signal": signal,
                "confidence": confidence,
                "daily_context": daily_context,
            })

        return signals

    def run_backtest(
        self,
        hybrid_enabled: bool,
        bars_limit: Optional[int] = None,
    ) -> BacktestResult:
        """
        Run backtest with or without hybrid sizing.

        Args:
            hybrid_enabled: If True, use hybrid position sizing
            bars_limit: Limit number of bars to process (for faster testing)

        Returns:
            BacktestResult with trade statistics
        """
        name = "Hybrid" if hybrid_enabled else "Current"
        result = BacktestResult(name=name)

        cash = self.starting_cash
        positions: Dict[str, Dict] = {}  # symbol -> position info
        equity_curve: List[float] = [cash]
        peak_equity = cash
        max_dd = 0.0

        for symbol in self.symbols:
            bars = self.load_data(symbol)
            if not bars:
                continue

            if bars_limit:
                bars = bars[:bars_limit]

            signals = self.generate_mock_signals(bars)

            for sig_data in signals:
                bar = sig_data["bar"]
                signal = sig_data["signal"]
                confidence = sig_data["confidence"]
                daily_context = sig_data["daily_context"]
                price = bar.get("Close", bar.get("close", 0))

                if signal == 0:
                    continue

                # Calculate base position size (simplified)
                base_qty = int((cash * 0.05) / price) if price > 0 else 0

                if base_qty <= 0:
                    continue

                # Apply hybrid sizing if enabled
                final_qty = base_qty
                sizing_result = None

                if hybrid_enabled:
                    sizing_result = self.hybrid_sizer.calculate(
                        signal=signal,
                        confidence=confidence,
                        daily_context=daily_context,
                    )
                    final_qty = int(base_qty * sizing_result.base_multiplier)

                    if final_qty <= 0:
                        continue  # Trade blocked

                # Check if we have an existing position
                if symbol in positions:
                    pos = positions[symbol]
                    # Check for exit condition (opposite signal)
                    if (pos["side"] == "long" and signal == -1) or \
                       (pos["side"] == "short" and signal == 1):
                        # Exit position
                        exit_price = price
                        entry_price = pos["entry_price"]
                        qty = pos["quantity"]

                        if pos["side"] == "long":
                            pnl = (exit_price - entry_price) * qty
                        else:
                            pnl = (entry_price - exit_price) * qty

                        pnl_pct = (pnl / (entry_price * qty)) * 100 if entry_price > 0 else 0

                        trade = TradeRecord(
                            symbol=symbol,
                            entry_time=pos["entry_time"],
                            exit_time=datetime.now(timezone.utc),
                            side=pos["side"],
                            entry_price=entry_price,
                            exit_price=exit_price,
                            quantity=qty,
                            pnl=pnl,
                            pnl_percent=pnl_pct,
                            confidence=pos.get("confidence", 1.0),
                            trend=pos.get("trend", "neutral"),
                            aligned=pos.get("aligned", True),
                            confidence_tier=pos.get("confidence_tier", "high"),
                            sizing_multiplier=pos.get("sizing_multiplier", 1.0),
                        )

                        result.trades.append(trade)
                        result.total_trades += 1
                        result.total_pnl += pnl

                        if pnl > 0:
                            result.winning_trades += 1
                        else:
                            result.losing_trades += 1

                        cash += entry_price * qty + pnl
                        del positions[symbol]
                else:
                    # Enter new position
                    if cash >= final_qty * price:
                        side = "long" if signal == 1 else "short"
                        positions[symbol] = {
                            "side": side,
                            "entry_price": price,
                            "quantity": final_qty,
                            "entry_time": datetime.now(timezone.utc),
                            "confidence": confidence,
                            "trend": sizing_result.trend if sizing_result else "neutral",
                            "aligned": sizing_result.aligned if sizing_result else True,
                            "confidence_tier": self._get_tier(confidence),
                            "sizing_multiplier": sizing_result.base_multiplier if sizing_result else 1.0,
                        }
                        cash -= final_qty * price
                        result.avg_position_size += final_qty * price

                # Update equity curve
                position_value = sum(
                    p["quantity"] * price for p in positions.values()
                )
                equity = cash + position_value
                equity_curve.append(equity)

                # Track drawdown
                if equity > peak_equity:
                    peak_equity = equity
                dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
                max_dd = max(max_dd, dd)

        result.max_drawdown = max_dd

        # Calculate average position size
        if result.total_trades > 0:
            result.avg_position_size /= result.total_trades

        # Calculate Sharpe ratio (simplified)
        if len(equity_curve) > 1:
            returns = [
                (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
                for i in range(1, len(equity_curve))
                if equity_curve[i-1] > 0
            ]
            if returns:
                avg_return = sum(returns) / len(returns)
                std_return = math.sqrt(sum((r - avg_return) ** 2 for r in returns) / len(returns))
                if std_return > 0:
                    result.sharpe_ratio = (avg_return * 252) / (std_return * math.sqrt(252))

        return result

    def _get_tier(self, confidence: float) -> str:
        """Get confidence tier from value."""
        if confidence >= 0.7:
            return "high"
        elif confidence >= 0.5:
            return "med"
        return "low"

    def run_comparison(self, bars_limit: Optional[int] = None) -> ComparisonReport:
        """
        Run both current and hybrid backtests and compare.

        Args:
            bars_limit: Limit bars per symbol (for faster testing)

        Returns:
            ComparisonReport with both results
        """
        print("Running current logic backtest...")
        current = self.run_backtest(hybrid_enabled=False, bars_limit=bars_limit)

        print("Running hybrid logic backtest...")
        hybrid = self.run_backtest(hybrid_enabled=True, bars_limit=bars_limit)

        return ComparisonReport(
            current=current,
            hybrid=hybrid,
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            symbols=self.symbols,
        )


async def run_comparison_async(args: argparse.Namespace) -> None:
    """Run the backtest comparison asynchronously."""
    # Resolve data path
    data_path = args.data_path
    if not Path(data_path).exists():
        data_path = str(ROOT / args.data_path)

    backtester = HybridBacktester(
        symbols=args.symbols,
        data_path=data_path,
        starting_cash=args.cash,
    )

    if args.hybrid_only:
        print("\nRunning hybrid-only backtest...")
        result = backtester.run_backtest(hybrid_enabled=True, bars_limit=args.bars)
        print(f"\nResults for {result.name}:")
        print(f"  Total trades: {result.total_trades}")
        print(f"  Win rate: {result.win_rate:.1f}%")
        print(f"  Total P&L: ${result.total_pnl:,.2f}")
        print(f"  Max drawdown: {result.max_drawdown:.1f}%")
        print(f"  Sharpe ratio: {result.sharpe_ratio:.2f}")
    else:
        report = backtester.run_comparison(bars_limit=args.bars)
        report.print_report()


def main():
    parser = argparse.ArgumentParser(
        description="Backtest comparison: Current logic vs Hybrid logic",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--symbols", "-s",
        nargs="+",
        default=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
        help="Symbols to backtest"
    )
    parser.add_argument(
        "--bars", "-b",
        type=int,
        default=None,
        help="Limit bars per symbol (for faster testing)"
    )
    parser.add_argument(
        "--data-path",
        default="data/data_storage/proc_data",
        help="Path to historical data files"
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=100_000.0,
        help="Starting cash"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=True,
        help="Run comparison (default behavior)"
    )
    parser.add_argument(
        "--hybrid-only",
        action="store_true",
        help="Run only hybrid mode (skip comparison)"
    )

    args = parser.parse_args()

    # Validate symbols have data
    data_path = Path(args.data_path)
    if not data_path.exists():
        data_path = ROOT / args.data_path

    if data_path.exists():
        available = []
        for sym in args.symbols:
            proc_file = data_path / f"proc_{sym}_file.json"
            raw_file = data_path / f"raw_{sym}_file.json"
            if proc_file.exists() or raw_file.exists():
                available.append(sym)
            else:
                print(f"Warning: No data file for {sym}, skipping")
        args.symbols = available

    if not args.symbols:
        print("Error: No valid symbols with data files")
        sys.exit(1)

    asyncio.run(run_comparison_async(args))


if __name__ == "__main__":
    main()
