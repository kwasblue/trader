# API Reference

Complete API documentation for Schwab Trader.

---

## Core Module

### Backtester

```python
class Backtester:
    """Main backtesting engine."""

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 10000,
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize backtester.

        Args:
            data: Historical OHLCV data with columns: Date, Open, High, Low, Close
            initial_capital: Starting capital
            transaction_cost: Cost per trade as fraction (0.001 = 0.1%)
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """

    def run_backtest(
        self,
        strategy_name: str,
        strategy_params: dict = None,
        sizer: DynamicPositionSizer = None
    ) -> pd.DataFrame:
        """
        Run backtest simulation.

        Args:
            strategy_name: Name of strategy (e.g., 'sma', 'ema', 'macd')
            strategy_params: Strategy parameters dict
            sizer: Position sizer instance

        Returns:
            DataFrame with columns: Date, Portfolio_Value, Cash, Position,
                                   Price, Drawdown, Strategy_Return
        """

    def evaluate_performance(
        self,
        portfolio_df: pd.DataFrame,
        market_data: pd.DataFrame = None
    ) -> dict:
        """
        Calculate performance metrics.

        Args:
            portfolio_df: Backtest results DataFrame
            market_data: Optional market data for beta/alpha

        Returns:
            Dict with keys: Standard Deviation, Sharpe Ratio, Sortino Ratio,
                          Max Drawdown, Value at Risk (VaR),
                          and optionally Beta, Alpha, Treynor Ratio
        """

    def plot_results(
        self,
        data: pd.DataFrame,
        strategy: str,
        save_path: str = None
    ):
        """Plot strategy vs buy-and-hold performance."""

    def generate_report(
        self,
        portfolio_df: pd.DataFrame,
        strategy_name: str,
        performance: dict,
        file_name: str = "backtest_report.pdf"
    ) -> str:
        """Generate PDF report with metrics and charts."""
```

---

### VectorizedBacktester

```python
class VectorizedBacktester:
    """High-performance vectorized backtester."""

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 10000,
        transaction_cost: float = 0.001,
        slippage_model: SlippageModel = None
    ):
        """
        Initialize vectorized backtester.

        Args:
            data: OHLCV DataFrame
            initial_capital: Starting capital
            transaction_cost: Transaction cost fraction
            slippage_model: SlippageModel instance (default: FixedSlippage)
        """

    def run(
        self,
        strategy_name: str,
        strategy_params: dict = None,
        position_sizing: str = 'fixed',
        position_size: float = 0.1,
        stop_loss_atr: float = 2.0,
        take_profit_atr: float = 3.0
    ) -> pd.DataFrame:
        """
        Run vectorized backtest.

        Args:
            strategy_name: Strategy name
            strategy_params: Strategy parameters
            position_sizing: 'fixed', 'risk_parity', 'volatility_scaled'
            position_size: Base position size as capital fraction
            stop_loss_atr: Stop loss in ATR multiples
            take_profit_atr: Take profit in ATR multiples

        Returns:
            DataFrame with Portfolio_Value, Position, Cash, Drawdown, Strategy_Return
        """

    def get_metrics(self, result: pd.DataFrame) -> dict:
        """
        Calculate performance metrics.

        Returns:
            Dict with keys: total_return, sharpe_ratio, sortino_ratio,
                          max_drawdown, win_rate, profit_factor, num_trades, final_value
        """
```

---

### KellyPositionSizer

```python
class KellyPositionSizer:
    """Production position sizer with full risk management features."""

    def __init__(
        self,
        risk_percentage: float,
        *,
        fee_rate: float = 0.001,
        max_trade_pct: Optional[float] = None,
        max_holding_pct: Optional[float] = None,
        allow_fractional: bool = False,
        lot_size: int = 1,
    ):
        """
        Initialize position sizer.

        Args:
            risk_percentage: Risk per trade as capital fraction (0.02 = 2%)
            fee_rate: Transaction fee rate (0.001 = 0.1%)
            max_trade_pct: Maximum trade notional as equity fraction
            max_holding_pct: Maximum holding notional as equity fraction
            allow_fractional: Allow fractional shares
            lot_size: Minimum lot size for rounding
        """

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        account_value: float,
        signal_strength: float = 1.0,
        atr: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        **kwargs
    ) -> int:
        """
        Calculate position size based on risk.

        Args:
            symbol: Trading symbol
            price: Current stock price
            account_value: Total account value (not used, uses portfolio)
            signal_strength: Signal confidence
            atr: Average True Range
            stop_loss_price: Stop loss price
            **kwargs: Additional args (signal, portfolio, market_conditions)

        Returns:
            Number of shares to trade
        """

    def reset_reserved(self, symbol: Optional[str] = None):
        """Reset reserved notional (per symbol or all)."""

# Factory function
from core.config_loader import create_position_sizer

sizer = create_position_sizer()  # Creates from config/trading_config.json
```

---

### DrawdownMonitor

```python
class DrawdownMonitor:
    """Unified risk guard for per-symbol and portfolio-level drawdown control."""

    def __init__(
        self,
        max_symbol_drawdown: float = 0.30,
        max_symbol_daily_drawdown: float = 0.15,
        symbol_cooldown_seconds: int = 5,
        max_portfolio_drawdown: float = 0.25,
        max_portfolio_daily_drawdown: float = 0.10,
        portfolio_cooldown_seconds: int = 5,
    ):
        """
        Initialize drawdown monitor.

        Args:
            max_symbol_drawdown: Max intraday drawdown per symbol (0.30 = 30%)
            max_symbol_daily_drawdown: Max daily drawdown per symbol
            symbol_cooldown_seconds: Cooldown after symbol unlock
            max_portfolio_drawdown: Max intraday portfolio drawdown
            max_portfolio_daily_drawdown: Max daily portfolio drawdown
            portfolio_cooldown_seconds: Cooldown after portfolio unlock
        """

    def start_new_day(
        self,
        portfolio_equity: Optional[float] = None,
        per_symbol_equity: Optional[Dict[str, float]] = None,
    ) -> None:
        """Reset daily baselines. Call once at session start."""

    def update_portfolio(self, portfolio_equity: float) -> bool:
        """Update portfolio drawdown state. Returns True if trading allowed."""

    def update_symbol(self, symbol: str, symbol_equity: float) -> bool:
        """Update symbol drawdown state. Returns True if trading allowed."""

    def can_trade(self, symbol: str) -> bool:
        """Check if trading allowed (portfolio not locked AND symbol not locked)."""

    def is_portfolio_blocked(self) -> bool:
        """Check if portfolio is blocked (locked or in cooldown)."""

    def is_symbol_blocked(self, symbol: str) -> bool:
        """Check if symbol is blocked (locked or in cooldown)."""

    def get_portfolio_drawdown(self) -> float:
        """Current portfolio drawdown vs peak (negative = DD)."""

    def unlock_symbol(self, symbol: str) -> None:
        """Manually unlock a symbol (starts cooldown)."""

    def unlock_portfolio(self) -> None:
        """Manually unlock portfolio (starts cooldown)."""

    # Async wrappers for use in async contexts
    async def can_trade_async(self, symbol: str) -> bool: ...
    async def update_portfolio_async(self, portfolio_equity: float) -> bool: ...
    async def update_symbol_async(self, symbol: str, symbol_equity: float) -> bool: ...

# Factory function
from core.config_loader import create_drawdown_monitor

ddm = create_drawdown_monitor()  # Creates from config, returns None if disabled
```

---

### StandardTradeApprover

```python
class StandardTradeApprover(TradeApprover):
    """Pure gating approver for trade decisions."""

    def __init__(
        self,
        tp_mult_low: float = 1.5,
        tp_mult_normal: float = 2.0,
        tp_mult_high: float = 3.0,
        sl_mult_low: float = 1.0,
        sl_mult_normal: float = 1.5,
        sl_mult_high: float = 2.0,
        exit_fraction: float = 0.25,
        trailing_stop: bool = True,
        min_bars_to_hold: int = 3,
        swing_mode: bool = False,
        min_hold_days: int = 1,
        cooldown_seconds: int = 300,
        cooldown_bars: int = 5,
        cooldown_mode: str = 'bars',
        max_positions: int = 10,
        allow_after_hours: bool = False,
        event_handler: Optional[EventHandler] = None,
    ):
        """Initialize trade approver with gating rules."""

    def should_trade(
        self,
        context: SignalContext,
        state: SymbolState,
        account_positions: int = 0,
    ) -> Tuple[bool, Optional[str]]:
        """
        Pure gating check - are we ALLOWED to trade?

        Returns:
            (True, None) if allowed
            (False, reason) if gated
        """

    def can_enter_position(
        self, symbol: str, state: SymbolState, side: OrderSide, **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Check if new position can be opened."""

    def can_exit_position(
        self, symbol: str, state: SymbolState, side: OrderSide, **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Check if position can be closed."""

# Factory function
from core.config_loader import create_trade_approver

approver = create_trade_approver()  # Creates from config
```

---

### PositionManager

```python
class PositionManager:
    """Manages position lifecycle: entry levels, stops, targets, exits."""

    def __init__(
        self,
        tp_mults: Optional[Dict[str, float]] = None,
        sl_mults: Optional[Dict[str, float]] = None,
        exit_fraction: float = 0.25,
        min_bars_to_hold: int = 3,
        swing_mode: bool = False,
        min_hold_days: int = 1,
    ):
        """Initialize position manager."""

    def calculate_levels(
        self, state: SymbolState, price: float, atr: float,
        condition: str, side: OrderSide
    ) -> None:
        """Calculate and set SL/TP/partial targets on state."""

    def update_trailing_stop(
        self, state: SymbolState, price: float, atr: float, condition: str
    ) -> None:
        """Update trailing stop based on current price."""

    def update_excursions(
        self, state: SymbolState, current_price: float, avg_price: float
    ) -> None:
        """Track MFE/MAE for trade analytics."""

    def check_exit_conditions(
        self, state: SymbolState, price: float, signal: int
    ) -> Tuple[bool, Optional[str]]:
        """Check all exit conditions for a position."""

    def check_partial_exit(
        self, state: SymbolState, price: float
    ) -> Tuple[bool, Optional[str]]:
        """Check if partial exit target was hit."""

    def get_exit_quantity(
        self, current_position: float, is_partial: bool = False
    ) -> int:
        """Calculate exit quantity."""

# Factory function
from core.config_loader import create_position_manager

pm = create_position_manager()  # Creates from config
```

---

## Strategy Module

### BaseStrategy

```python
class BaseStrategy(ABC):
    """Abstract base class for all strategies."""

    def __init__(self, params: dict = None, **kwargs):
        """
        Initialize strategy.

        Args:
            params: Strategy parameters dict
            **kwargs: Additional keyword arguments
        """
        self.params = params or kwargs

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Generate trading signal for latest bar.

        Args:
            data: OHLCV DataFrame

        Returns:
            1: Buy signal
            -1: Sell signal
            0: Hold/No signal
        """
        pass

    def generate_signals_vectorized(
        self,
        data: pd.DataFrame
    ) -> Optional[List[int]]:
        """
        Generate signals for all bars (vectorized).

        Args:
            data: OHLCV DataFrame

        Returns:
            List of signals for each bar, or None if not implemented
        """
        return None
```

### Strategy Registry Functions

```python
def load_strategy(name: str, params: dict = None) -> BaseStrategy:
    """
    Load a strategy by name.

    Args:
        name: Strategy name (e.g., 'sma', 'ema', 'macd')
        params: Strategy parameters

    Returns:
        Strategy instance

    Raises:
        ValueError: If strategy not found
    """

def list_strategies() -> List[str]:
    """
    List all available strategy names.

    Returns:
        List of strategy names
    """
```

---

## Backtest Suite Module

### Data Validation

```python
def validate_ohlcv_data(
    data: pd.DataFrame,
    required_columns: List[str] = None,
    fix_issues: bool = True
) -> ValidationResult:
    """
    Validate OHLCV data for backtesting.

    Args:
        data: DataFrame to validate
        required_columns: Required column names (default: Date, OHLC)
        fix_issues: If True, attempt to fix issues

    Returns:
        ValidationResult with is_valid, errors, warnings, cleaned_data
    """

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    cleaned_data: Optional[pd.DataFrame]
```

### Slippage Models

```python
class SlippageModel:
    """Base class for slippage models."""

    def calculate_slippage(
        self,
        price: float,
        quantity: int,
        side: str,
        volume: float = None,
        volatility: float = None
    ) -> float:
        """Calculate execution price after slippage."""

class FixedSlippage(SlippageModel):
    def __init__(self, slippage_pct: float = 0.001): ...

class RandomSlippage(SlippageModel):
    def __init__(self, min_pct: float = -0.001, max_pct: float = 0.001): ...

class VolumeBasedSlippage(SlippageModel):
    def __init__(
        self,
        base_slippage: float = 0.0001,
        volume_impact: float = 0.1,
        max_slippage: float = 0.02
    ): ...

class VolatilityAdjustedSlippage(SlippageModel):
    def __init__(
        self,
        base_slippage: float = 0.0005,
        volatility_multiplier: float = 2.0,
        max_slippage: float = 0.03
    ): ...
```

### Optimization Functions

```python
def grid_search(
    data: pd.DataFrame,
    strategy_name: str,
    param_grid: Dict[str, List[Any]],
    metric: str = 'sharpe_ratio',
    initial_capital: float = 10000,
    transaction_cost: float = 0.001,
    n_jobs: int = 1,
    verbose: bool = True
) -> OptimizationResult:
    """
    Grid search over strategy parameters.

    Args:
        data: OHLCV DataFrame
        strategy_name: Strategy to optimize
        param_grid: Dict mapping param names to value lists
        metric: 'sharpe_ratio', 'total_return', 'sortino_ratio', 'max_drawdown'
        initial_capital: Starting capital
        transaction_cost: Transaction cost fraction
        n_jobs: Parallel jobs (1 = sequential)
        verbose: Print progress

    Returns:
        OptimizationResult with best_params, best_metric, all_results
    """

@dataclass
class OptimizationResult:
    best_params: Dict[str, Any]
    best_metric: float
    all_results: List[Dict[str, Any]]
    metric_name: str
```

### Walk-Forward Analysis

```python
def walk_forward_analysis(
    data: pd.DataFrame,
    strategy_name: str,
    param_grid: Dict[str, List[Any]],
    train_size: int = 252,
    test_size: int = 63,
    step_size: int = 63,
    metric: str = 'sharpe_ratio',
    initial_capital: float = 10000,
    verbose: bool = True
) -> WalkForwardResult:
    """
    Walk-forward analysis with rolling windows.

    Args:
        data: OHLCV DataFrame
        strategy_name: Strategy to test
        param_grid: Parameters to optimize
        train_size: Training window size (bars)
        test_size: Testing window size (bars)
        step_size: Roll forward amount (bars)
        metric: Optimization metric
        initial_capital: Starting capital
        verbose: Print progress

    Returns:
        WalkForwardResult with windows, overall_return, overall_sharpe,
                         out_of_sample_returns, in_sample_params
    """

@dataclass
class WalkForwardResult:
    windows: List[Dict[str, Any]]
    overall_return: float
    overall_sharpe: float
    out_of_sample_returns: List[float]
    in_sample_params: List[Dict[str, Any]]
```

### Monte Carlo Simulation

```python
def monte_carlo_simulation(
    trades: List[Dict],
    initial_capital: float = 10000,
    n_simulations: int = 1000,
    seed: int = None
) -> MonteCarloResult:
    """
    Monte Carlo simulation by randomizing trade order.

    Args:
        trades: List of trade dicts with 'pnl' key
        initial_capital: Starting capital
        n_simulations: Number of simulations
        seed: Random seed

    Returns:
        MonteCarloResult with mean_return, median_return, std_return,
                        percentiles, max_drawdowns, sharpe_ratios,
                        final_values, confidence_interval_95
    """

@dataclass
class MonteCarloResult:
    mean_return: float
    median_return: float
    std_return: float
    percentiles: Dict[int, float]
    max_drawdowns: List[float]
    sharpe_ratios: List[float]
    final_values: List[float]
    confidence_interval_95: Tuple[float, float]
```

### Benchmark Comparison

```python
def compare_to_benchmark(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02
) -> BenchmarkComparison:
    """
    Compare strategy to benchmark.

    Args:
        strategy_returns: Strategy return series
        benchmark_returns: Benchmark return series
        risk_free_rate: Annual risk-free rate

    Returns:
        BenchmarkComparison with strategy_return, benchmark_return,
                           excess_return, strategy_sharpe, benchmark_sharpe,
                           beta, alpha, information_ratio, tracking_error,
                           up_capture, down_capture
    """

@dataclass
class BenchmarkComparison:
    strategy_return: float
    benchmark_return: float
    excess_return: float
    strategy_sharpe: float
    benchmark_sharpe: float
    beta: float
    alpha: float
    information_ratio: float
    tracking_error: float
    up_capture: float
    down_capture: float
```

---

## Data Module

### DataStore

```python
class DataStore:
    """Thread-safe SQLite data storage."""

    def __init__(self, db_path: str = "data/trading.db"):
        """Initialize with database path."""

    def __enter__(self) -> 'DataStore':
        """Context manager entry - opens connection."""

    def __exit__(self, *args):
        """Context manager exit - closes connection."""

    def open_db(self):
        """Open database connection."""

    def close_db(self):
        """Close database connection."""

    def create_database(self, table_name: str):
        """Create table for symbol data."""

    def fill_database(self, table_name: str, df: pd.DataFrame):
        """Insert DataFrame into table."""

    def read_data(
        self,
        table_name: str,
        limit: int = None,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """Read data from table."""

    def upsert_data(self, table_name: str, df: pd.DataFrame):
        """Insert or update data."""

    def table_exists(self, table_name: str) -> bool:
        """Check if table exists."""

    def list_tables(self) -> List[str]:
        """List all tables."""

    def get_row_count(self, table_name: str) -> int:
        """Get number of rows in table."""

    def delete_data(
        self,
        table_name: str,
        before_date: str = None,
        after_date: str = None
    ):
        """Delete data from table."""
```

---

## Monitoring Module

### DataFeeder

```python
class DataFeeder(QObject):
    """Bridges async EventBus to Qt signals."""

    class Signals(QObject):
        ohlc = Signal(str, pd.DataFrame)      # symbol, data
        symbols = Signal(list)                 # position rows
        orders = Signal(list)                  # order rows
        trades = Signal(list)                  # trade rows
        equity_point = Signal(str, float)      # timestamp, value
        realized_point = Signal(str, float, float)  # ts, realized, unrealized
        risk_stats = Signal(float, float, float)    # unreal, real, drawdown
        alerts = Signal(list)                  # alert dicts
        cooldown = Signal(bool)                # is_halted
        health = Signal(dict)                  # health status
        regime_breakdown = Signal(dict)        # regime info
        log = Signal(str)                      # log message

    async def start(self):
        """Start subscribing to EventBus."""

    async def stop(self):
        """Stop and unsubscribe."""
```

### StateAggregator

```python
class StateAggregator(QObject):
    """Aggregates state into unified snapshots."""

    snapshot_ready = Signal(dict)  # Emitted every 1 second

    def __init__(self, feeder: DataFeeder):
        """Initialize with DataFeeder to aggregate from."""

    def get_snapshot(self) -> dict:
        """Get current state snapshot."""
```

### SymbolsTableModel

```python
class SymbolsTableModel(QAbstractTableModel):
    """Table model for positions display."""

    HEADERS = [
        "Symbol", "Side", "Qty", "Avg", "Last",
        "Unreal", "Realized", "Total", "PnL %",
        "ATR", "Regime", "Risk"
    ]

    def update_position(self, pos: dict):
        """Update single position row."""

    def remove_position(self, symbol: str):
        """Remove position row."""

    def update_from_df(self, df: pd.DataFrame):
        """Update all rows from DataFrame."""

    def replace_rows(self, rows: List[dict]):
        """Replace all rows."""
```

---

## Indicators Module

### ATRIndicator

```python
class ATRIndicator:
    """Average True Range indicator."""

    def __init__(self, df: pd.DataFrame, window: int = 14):
        """Initialize with data and window."""

    def compute(self) -> pd.Series:
        """Calculate ATR."""
```

### BollingerBands

```python
class BollingerBands:
    """Bollinger Bands indicator."""

    def __init__(self, df: pd.DataFrame, window: int = 20, num_std: float = 2):
        """Initialize with data, window, and std multiplier."""

    def compute(self) -> pd.DataFrame:
        """Calculate bands. Returns DataFrame with upper, middle, lower."""
```

---

## Event Types

```python
# Core events
EVENT_NEW_BAR = "new_bar"
EVENT_STRATEGY_SIGNAL = "strategy_signal"
EVENT_ORDER_SUBMITTED = "order_submitted"
EVENT_ORDER_FILLED = "order_filled"
EVENT_ORDER_CANCELED = "order_canceled"
EVENT_ORDER_REJECTED = "order_rejected"
EVENT_POSITION_UPDATE = "position_update"
EVENT_PNL_UPDATE = "pnl_update"

# Risk events
EVENT_DRAWDOWN_ALERT = "drawdown_alert"
EVENT_HALT_STATE = "halt_state"
EVENT_COOLDOWN_STATE = "cooldown_state"

# System events
EVENT_LOG = "log"
EVENT_HEALTH_UPDATE = "health_update"
EVENT_REGIME_UPDATE = "regime_update"

# User events
EVENT_MANUAL_ORDER = "manual_order"
EVENT_FLATTEN_ALL = "flatten_all"
EVENT_CANCEL_ALL = "cancel_all"
```
