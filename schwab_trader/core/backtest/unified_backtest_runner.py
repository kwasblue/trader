"""
Unified Backtest Runner - Run any strategy with optional hybrid sizing.

This module provides a comprehensive backtesting framework that:
1. Runs any strategy from the registry
2. Optionally integrates hybrid position sizing for trend alignment
3. Tracks trades by trend alignment (with/against trend)
4. Calculates comprehensive metrics including trend-specific stats

Usage:
    from core.backtest.unified_backtest_runner import UnifiedBacktestRunner, BacktestConfig

    config = BacktestConfig(
        strategy_name="sma",
        strategy_params={"fast": 10, "slow": 30},
        use_hybrid_sizing=True
    )

    runner = UnifiedBacktestRunner(data)
    result = runner.run(config)

    print(f"Return: {result.metrics.total_return:.2%}")
    print(f"Trades with trend: {result.metrics.trades_with_trend}")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import logging

from strategies.strategy_registry import load_strategy, list_strategies
from core.backtest.validation import validate_ohlcv_data
from core.backtest.slippage_models import SlippageModel, FixedSlippage
from core.logic.hybrid_position_sizer import HybridPositionSizer, HybridSizingResult

logger = logging.getLogger(__name__)


# Strategy categories for comparison
STRATEGY_CATEGORIES = {
    "trend_following": ["sma", "ema", "macd", "adx", "ichimoku", "psar", "donchian", "breakout"],
    "mean_reversion": ["rsi", "bollinger", "stochastic", "meanreversion", "vwap"],
    "momentum": ["momentum"],
    "ml": ["logisticregression"],
    "ensemble": ["combined"],
}


def get_strategy_category(strategy_name: str) -> str:
    """Get the category for a strategy."""
    for category, strategies in STRATEGY_CATEGORIES.items():
        if strategy_name.lower() in strategies:
            return category
    return "unknown"


def get_strategies_by_category(category: str) -> List[str]:
    """Get all strategies in a category."""
    return STRATEGY_CATEGORIES.get(category.lower(), [])


@dataclass
class BacktestConfig:
    """Configuration for running a backtest."""

    strategy_name: str
    strategy_params: Dict[str, Any] = field(default_factory=dict)

    # Position sizing
    position_sizing: str = "fixed"  # "fixed", "volatility_scaled", "risk_parity"
    position_size: float = 0.1  # Fraction of capital per trade

    # Risk management
    stop_loss_atr: float = 2.0
    take_profit_atr: float = 3.0

    # Capital
    initial_capital: float = 10000.0
    transaction_cost: float = 0.001

    # Hybrid sizing
    use_hybrid_sizing: bool = False
    hybrid_config: Optional[Dict[str, Any]] = None

    # Confidence settings (for hybrid sizing)
    default_confidence: float = 0.6  # Default if strategy doesn't provide confidence


@dataclass
class TradeRecord:
    """Record of a single trade."""

    entry_idx: int
    entry_price: float
    entry_signal: int  # 1 for buy, -1 for sell
    quantity: int

    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # "signal", "stop_loss", "take_profit"

    pnl: float = 0.0
    pnl_pct: float = 0.0

    # Hybrid sizing info
    trend: str = "neutral"
    aligned: bool = True
    confidence: float = 0.6
    size_multiplier: float = 1.0


@dataclass
class BacktestMetrics:
    """Comprehensive metrics from a backtest."""

    # Core metrics
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0

    # Trade metrics
    num_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # Trend-aligned metrics (hybrid sizing)
    trades_with_trend: int = 0
    trades_against_trend: int = 0
    win_rate_with_trend: float = 0.0
    win_rate_against_trend: float = 0.0
    pnl_with_trend: float = 0.0
    pnl_against_trend: float = 0.0

    # Additional metrics
    final_value: float = 0.0
    cagr: float = 0.0
    calmar_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "num_trades": self.num_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_pnl": self.avg_trade_pnl,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "trades_with_trend": self.trades_with_trend,
            "trades_against_trend": self.trades_against_trend,
            "win_rate_with_trend": self.win_rate_with_trend,
            "win_rate_against_trend": self.win_rate_against_trend,
            "pnl_with_trend": self.pnl_with_trend,
            "pnl_against_trend": self.pnl_against_trend,
            "final_value": self.final_value,
            "cagr": self.cagr,
            "calmar_ratio": self.calmar_ratio,
        }


@dataclass
class BacktestResult:
    """Complete result from a backtest run."""

    config: BacktestConfig
    metrics: BacktestMetrics

    # Equity curve
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # Daily returns
    returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # Drawdown series
    drawdown: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # Trade log
    trades: List[TradeRecord] = field(default_factory=list)

    # Signal series
    signals: pd.Series = field(default_factory=lambda: pd.Series(dtype=int))

    # Full result DataFrame
    result_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        records = []
        for t in self.trades:
            records.append({
                "entry_idx": t.entry_idx,
                "entry_price": t.entry_price,
                "signal": t.entry_signal,
                "quantity": t.quantity,
                "exit_idx": t.exit_idx,
                "exit_price": t.exit_price,
                "exit_reason": t.exit_reason,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "trend": t.trend,
                "aligned": t.aligned,
                "confidence": t.confidence,
                "size_multiplier": t.size_multiplier,
            })

        return pd.DataFrame(records)


class UnifiedBacktestRunner:
    """
    Run backtests on any strategy with optional hybrid sizing.

    This runner provides:
    - Execution of any registered strategy
    - Optional hybrid position sizing based on trend alignment
    - Comprehensive trade tracking with trend statistics
    - Full metrics calculation including trend-specific metrics
    """

    def __init__(
        self,
        data: pd.DataFrame,
        slippage_model: Optional[SlippageModel] = None,
    ):
        """
        Initialize the backtest runner.

        Args:
            data: OHLCV DataFrame with historical data
            slippage_model: Optional slippage model for realistic execution
        """
        self.data = data.copy()
        self.slippage = slippage_model or FixedSlippage(0.0005)

        # Validate and prepare data
        result = validate_ohlcv_data(self.data, fix_issues=True)
        if not result.is_valid:
            raise ValueError(f"Invalid data: {result.errors}")
        if result.cleaned_data is not None:
            self.data = result.cleaned_data

        # Ensure column names are consistent (capitalize first letter)
        self._normalize_columns()

        # Pre-calculate ATR if not present
        if "ATR" not in self.data.columns:
            self._calculate_atr()

        # Build daily context for hybrid sizing
        self._daily_context = self._build_daily_context()

        logger.info(f"UnifiedBacktestRunner initialized with {len(self.data)} bars")

    def _normalize_columns(self) -> None:
        """Normalize column names to expected format."""
        column_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }

        for old, new in column_map.items():
            if old in self.data.columns and new not in self.data.columns:
                self.data = self.data.rename(columns={old: new})

    def _calculate_atr(self, period: int = 14) -> None:
        """Calculate ATR and add to data."""
        high = self.data["High"].values
        low = self.data["Low"].values
        close = self.data["Close"].values

        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]

        self.data["ATR"] = pd.Series(tr).rolling(period).mean().values

    def _build_daily_context(self) -> Dict[int, Dict[str, Any]]:
        """
        Build daily context for each bar for hybrid sizing.

        Returns a dict mapping bar index to context with trend indicators.
        """
        context = {}

        # Calculate indicators if not present
        df = self.data.copy()

        # RSI
        if "RSI" not in df.columns:
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.inf)
            df["RSI"] = 100 - (100 / (1 + rs))

        # MACD
        if "MACD" not in df.columns:
            ema12 = df["Close"].ewm(span=12, adjust=False).mean()
            ema26 = df["Close"].ewm(span=26, adjust=False).mean()
            df["MACD"] = ema12 - ema26
            df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # SMA 200
        if "SMA_200" not in df.columns:
            df["SMA_200"] = df["Close"].rolling(200).mean()

        # Build context for each row
        for i in range(len(df)):
            row = df.iloc[i]
            context[i] = {
                "RSI": row.get("RSI") if pd.notna(row.get("RSI")) else None,
                "MACD": row.get("MACD") if pd.notna(row.get("MACD")) else None,
                "MACD_Signal": row.get("MACD_Signal") if pd.notna(row.get("MACD_Signal")) else None,
                "Close": row["Close"],
                "SMA_200": row.get("SMA_200") if pd.notna(row.get("SMA_200")) else None,
            }

        return context

    def run(self, config: BacktestConfig) -> BacktestResult:
        """
        Run a backtest with the given configuration.

        Args:
            config: BacktestConfig with strategy and sizing settings

        Returns:
            BacktestResult with metrics, equity curve, and trades
        """
        logger.info(f"Running backtest: {config.strategy_name} (hybrid={config.use_hybrid_sizing})")

        # Load strategy
        strategy = load_strategy(config.strategy_name, params=config.strategy_params)

        # Generate signals
        signals = self._generate_signals(strategy)

        # Initialize hybrid sizer if enabled
        hybrid_sizer = None
        if config.use_hybrid_sizing:
            hybrid_sizer = HybridPositionSizer(
                enabled=True,
                config=config.hybrid_config
            )

        # Run simulation
        result_df, trades = self._simulate(
            signals=signals,
            config=config,
            hybrid_sizer=hybrid_sizer,
        )

        # Calculate metrics
        metrics = self._calculate_metrics(result_df, trades, config)

        # Build result
        result = BacktestResult(
            config=config,
            metrics=metrics,
            equity_curve=result_df["Portfolio_Value"],
            returns=result_df["Strategy_Return"],
            drawdown=result_df["Drawdown"],
            trades=trades,
            signals=pd.Series(signals, index=self.data.index if hasattr(self.data, 'index') else None),
            result_df=result_df,
        )

        logger.info(
            f"Backtest complete: Return={metrics.total_return:.2%}, "
            f"Sharpe={metrics.sharpe_ratio:.2f}, Trades={metrics.num_trades}"
        )

        return result

    def _generate_signals(self, strategy) -> np.ndarray:
        """Generate signals for all bars."""
        n = len(self.data)
        signals = np.zeros(n, dtype=int)

        # Try vectorized generation first
        if hasattr(strategy, "generate_signals_vectorized"):
            vec_signals = strategy.generate_signals_vectorized(self.data)
            if vec_signals is not None:
                return np.array(vec_signals)

        # Fall back to row-by-row
        for i in range(n):
            try:
                sig = strategy.generate_signal(self.data.iloc[:i+1])
                signals[i] = int(sig) if sig in (-1, 0, 1) else 0
            except Exception:
                signals[i] = 0

        return signals

    def _simulate(
        self,
        signals: np.ndarray,
        config: BacktestConfig,
        hybrid_sizer: Optional[HybridPositionSizer] = None,
    ) -> Tuple[pd.DataFrame, List[TradeRecord]]:
        """
        Simulate trading with the given signals.

        Returns:
            Tuple of (result_df, trades_list)
        """
        n = len(self.data)
        close = self.data["Close"].values
        high = self.data["High"].values
        low = self.data["Low"].values
        atr = self.data["ATR"].values
        volume = self.data["Volume"].values if "Volume" in self.data.columns else np.ones(n) * 1e6

        # State arrays
        position = np.zeros(n)
        cash = np.zeros(n)
        portfolio_value = np.zeros(n)

        cash[0] = config.initial_capital
        portfolio_value[0] = config.initial_capital

        trades: List[TradeRecord] = []
        current_trade: Optional[TradeRecord] = None

        for i in range(1, n):
            cash[i] = cash[i-1]
            position[i] = position[i-1]

            current_price = close[i]
            current_atr = atr[i] if not np.isnan(atr[i]) else current_price * 0.02
            signal = signals[i]

            # Get hybrid sizing info
            size_multiplier = 1.0
            trend = "neutral"
            aligned = True
            confidence = config.default_confidence

            if hybrid_sizer is not None and signal != 0:
                daily_ctx = self._daily_context.get(i, {})
                sizing_result = hybrid_sizer.calculate(
                    signal=signal,
                    confidence=confidence,
                    daily_context=daily_ctx,
                )
                size_multiplier = sizing_result.base_multiplier
                trend = sizing_result.trend
                aligned = sizing_result.aligned

            # Skip trade if hybrid sizing blocks it
            if size_multiplier <= 0:
                signal = 0

            # Calculate position size
            if config.position_sizing == "volatility_scaled":
                target_vol = 0.15
                daily_vol = current_atr / current_price
                vol_mult = target_vol / (daily_vol * np.sqrt(252)) if daily_vol > 0 else 1
                vol_mult = np.clip(vol_mult, 0.1, 3.0)
            else:
                vol_mult = 1.0

            base_trade_value = config.initial_capital * config.position_size * vol_mult
            trade_value = base_trade_value * size_multiplier
            quantity = int(trade_value / current_price) if current_price > 0 else 0

            # Process trades
            if signal == 1 and position[i] <= 0 and quantity > 0:
                # Close short if any
                if position[i] < 0 and current_trade is not None:
                    exec_price = self._get_exec_price(current_price, abs(int(position[i])), "buy", volume[i], current_atr)
                    cost = abs(position[i]) * exec_price * (1 + config.transaction_cost)
                    cash[i] -= cost

                    current_trade.exit_idx = i
                    current_trade.exit_price = exec_price
                    current_trade.exit_reason = "signal"
                    current_trade.pnl = (current_trade.entry_price - exec_price) * abs(current_trade.quantity)
                    current_trade.pnl_pct = current_trade.pnl / (current_trade.entry_price * abs(current_trade.quantity))
                    trades.append(current_trade)
                    position[i] = 0

                # Open long
                exec_price = self._get_exec_price(current_price, quantity, "buy", volume[i], current_atr)
                cost = quantity * exec_price * (1 + config.transaction_cost)

                if cost <= cash[i]:
                    cash[i] -= cost
                    position[i] = quantity

                    current_trade = TradeRecord(
                        entry_idx=i,
                        entry_price=exec_price,
                        entry_signal=1,
                        quantity=quantity,
                        trend=trend,
                        aligned=aligned,
                        confidence=confidence,
                        size_multiplier=size_multiplier,
                    )

            elif signal == -1 and position[i] >= 0 and quantity > 0:
                # Close long if any
                if position[i] > 0 and current_trade is not None:
                    exec_price = self._get_exec_price(current_price, int(position[i]), "sell", volume[i], current_atr)
                    proceeds = position[i] * exec_price * (1 - config.transaction_cost)
                    cash[i] += proceeds

                    current_trade.exit_idx = i
                    current_trade.exit_price = exec_price
                    current_trade.exit_reason = "signal"
                    current_trade.pnl = (exec_price - current_trade.entry_price) * current_trade.quantity
                    current_trade.pnl_pct = current_trade.pnl / (current_trade.entry_price * current_trade.quantity)
                    trades.append(current_trade)
                    current_trade = None
                    position[i] = 0

            # Check stop loss for long positions
            if position[i] > 0 and current_trade is not None:
                stop_price = current_trade.entry_price - config.stop_loss_atr * current_atr
                if low[i] <= stop_price:
                    exec_price = stop_price * (1 - 0.001)  # Extra slippage on stop
                    proceeds = position[i] * exec_price * (1 - config.transaction_cost)
                    cash[i] += proceeds

                    current_trade.exit_idx = i
                    current_trade.exit_price = exec_price
                    current_trade.exit_reason = "stop_loss"
                    current_trade.pnl = (exec_price - current_trade.entry_price) * current_trade.quantity
                    current_trade.pnl_pct = current_trade.pnl / (current_trade.entry_price * current_trade.quantity)
                    trades.append(current_trade)
                    current_trade = None
                    position[i] = 0

            # Check take profit for long positions
            if position[i] > 0 and current_trade is not None:
                tp_price = current_trade.entry_price + config.take_profit_atr * current_atr
                if high[i] >= tp_price:
                    exec_price = tp_price
                    proceeds = position[i] * exec_price * (1 - config.transaction_cost)
                    cash[i] += proceeds

                    current_trade.exit_idx = i
                    current_trade.exit_price = exec_price
                    current_trade.exit_reason = "take_profit"
                    current_trade.pnl = (exec_price - current_trade.entry_price) * current_trade.quantity
                    current_trade.pnl_pct = current_trade.pnl / (current_trade.entry_price * current_trade.quantity)
                    trades.append(current_trade)
                    current_trade = None
                    position[i] = 0

            portfolio_value[i] = cash[i] + position[i] * current_price

        # Close any open position at end
        if position[-1] != 0 and current_trade is not None:
            exec_price = close[-1]
            if position[-1] > 0:
                current_trade.pnl = (exec_price - current_trade.entry_price) * current_trade.quantity
            else:
                current_trade.pnl = (current_trade.entry_price - exec_price) * abs(current_trade.quantity)
            current_trade.exit_idx = n - 1
            current_trade.exit_price = exec_price
            current_trade.exit_reason = "end_of_data"
            current_trade.pnl_pct = current_trade.pnl / (current_trade.entry_price * abs(current_trade.quantity))
            trades.append(current_trade)

        # Build result DataFrame
        result = self.data.copy()
        result["Signal"] = signals
        result["Position"] = position
        result["Cash"] = cash
        result["Portfolio_Value"] = portfolio_value
        result["Strategy_Return"] = pd.Series(portfolio_value).pct_change().fillna(0)

        # Calculate drawdown
        peak = np.maximum.accumulate(portfolio_value)
        result["Drawdown"] = (peak - portfolio_value) / peak

        return result, trades

    def _get_exec_price(
        self,
        price: float,
        quantity: int,
        side: str,
        volume: float,
        atr: float,
    ) -> float:
        """Get execution price with slippage."""
        volatility_pct = (atr / price * 100) if price > 0 else 1.0
        return self.slippage.calculate_slippage(price, quantity, side, volume, volatility_pct)

    def _calculate_metrics(
        self,
        result_df: pd.DataFrame,
        trades: List[TradeRecord],
        config: BacktestConfig,
    ) -> BacktestMetrics:
        """Calculate comprehensive metrics."""
        returns = result_df["Strategy_Return"].values
        portfolio_value = result_df["Portfolio_Value"].values

        # Total return
        total_return = (portfolio_value[-1] - config.initial_capital) / config.initial_capital

        # Sharpe ratio (annualized)
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Sortino ratio
        downside = returns[returns < 0]
        if len(downside) > 0 and np.std(downside) > 0:
            sortino = np.mean(returns) / np.std(downside) * np.sqrt(252)
        else:
            sortino = sharpe

        # Max drawdown
        peak = np.maximum.accumulate(portfolio_value)
        drawdown = (peak - portfolio_value) / peak
        max_drawdown = np.max(drawdown)

        # Trade metrics
        num_trades = len(trades)

        if num_trades > 0:
            pnls = [t.pnl for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]

            win_rate = len(wins) / num_trades if num_trades > 0 else 0.0
            avg_trade_pnl = np.mean(pnls) if pnls else 0.0
            avg_win = np.mean(wins) if wins else 0.0
            avg_loss = np.mean(losses) if losses else 0.0

            total_wins = sum(wins)
            total_losses = abs(sum(losses))
            profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")
        else:
            win_rate = 0.0
            avg_trade_pnl = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = 0.0

        # Trend-aligned metrics
        trades_with_trend = [t for t in trades if t.aligned]
        trades_against_trend = [t for t in trades if not t.aligned]

        pnl_with_trend = sum(t.pnl for t in trades_with_trend)
        pnl_against_trend = sum(t.pnl for t in trades_against_trend)

        wins_with_trend = len([t for t in trades_with_trend if t.pnl > 0])
        wins_against_trend = len([t for t in trades_against_trend if t.pnl > 0])

        win_rate_with_trend = wins_with_trend / len(trades_with_trend) if trades_with_trend else 0.0
        win_rate_against_trend = wins_against_trend / len(trades_against_trend) if trades_against_trend else 0.0

        # CAGR
        n_years = len(result_df) / 252
        if n_years > 0 and portfolio_value[-1] > 0:
            cagr = (portfolio_value[-1] / config.initial_capital) ** (1 / n_years) - 1
        else:
            cagr = 0.0

        # Calmar ratio
        calmar = cagr / max_drawdown if max_drawdown > 0 else 0.0

        return BacktestMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            num_trades=num_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_trade_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            trades_with_trend=len(trades_with_trend),
            trades_against_trend=len(trades_against_trend),
            win_rate_with_trend=win_rate_with_trend,
            win_rate_against_trend=win_rate_against_trend,
            pnl_with_trend=pnl_with_trend,
            pnl_against_trend=pnl_against_trend,
            final_value=portfolio_value[-1],
            cagr=cagr,
            calmar_ratio=calmar,
        )

    def run_strategy(
        self,
        strategy_name: str,
        strategy_params: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = False,
        **kwargs,
    ) -> BacktestResult:
        """
        Convenience method to run a single strategy.

        Args:
            strategy_name: Name of strategy from registry
            strategy_params: Strategy parameters
            use_hybrid: Whether to use hybrid sizing
            **kwargs: Additional BacktestConfig parameters

        Returns:
            BacktestResult
        """
        config = BacktestConfig(
            strategy_name=strategy_name,
            strategy_params=strategy_params or {},
            use_hybrid_sizing=use_hybrid,
            **kwargs,
        )
        return self.run(config)

    def compare_hybrid(
        self,
        strategy_name: str,
        strategy_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[BacktestResult, BacktestResult]:
        """
        Run strategy with and without hybrid sizing for comparison.

        Args:
            strategy_name: Name of strategy
            strategy_params: Strategy parameters
            **kwargs: Additional BacktestConfig parameters

        Returns:
            Tuple of (standard_result, hybrid_result)
        """
        # Standard run
        standard_config = BacktestConfig(
            strategy_name=strategy_name,
            strategy_params=strategy_params or {},
            use_hybrid_sizing=False,
            **kwargs,
        )
        standard_result = self.run(standard_config)

        # Hybrid run
        hybrid_config = BacktestConfig(
            strategy_name=strategy_name,
            strategy_params=strategy_params or {},
            use_hybrid_sizing=True,
            **kwargs,
        )
        hybrid_result = self.run(hybrid_config)

        return standard_result, hybrid_result


def list_available_strategies() -> List[str]:
    """Get list of all available strategies."""
    return list_strategies()


def list_categories() -> Dict[str, List[str]]:
    """Get all strategy categories with their strategies."""
    return STRATEGY_CATEGORIES.copy()
