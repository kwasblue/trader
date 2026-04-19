"""
Base Strategy - Abstract base class for trading strategies

This module defines the contract for strategy implementations that generate
trading signals based on market data analysis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Dict, Any, Optional, List
import pandas as pd
import logging

from core.tracing import trace

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    Strategies are responsible for analyzing market data and generating signals.
    They are completely decoupled from execution logic - they only decide WHAT
    to do, not HOW to do it.
    
    Signal Convention:
    - +1: Buy/Long signal
    - -1: Sell/Short signal  
    - 0: Hold/No action
    
    Design Philosophy:
    - Strategy generates signals (analysis and decision)
    - Executor interprets signals (risk management and execution)
    - Broker executes orders (routing and fills)
    
    Flexible Parameter Handling:
    The constructor accepts **kwargs to allow flexible parameter injection
    from configuration files or routing systems. Subclasses can either:
    1. Use self.params dict to access parameters generically
    2. Declare explicit __init__ parameters and call super().__init__(**kwargs)
    
    Example:
        # Option 1: Generic access via self.params
        class MyStrategy(BaseStrategy):
            def generate_signal(self, data):
                threshold = self.params.get('threshold', 0.02)
                # ... strategy logic
        
        # Option 2: Explicit parameters
        class MyStrategy(BaseStrategy):
            def __init__(self, threshold: float = 0.02, **kwargs):
                super().__init__(**kwargs)
                self.threshold = threshold
            
            def generate_signal(self, data):
                # ... use self.threshold
    """
    
    def __init__(self, **kwargs):
        """
        Initialize base strategy with flexible parameters.
        
        Args:
            **kwargs: Strategy parameters passed from config or router
                Common parameters:
                - name: Strategy name (defaults to class name)
                - symbols: List of symbols to trade
                - timeframe: Data timeframe (e.g., "1m", "1h", "1d")
                - ... any strategy-specific parameters
        """
        # Store all parameters for generic access
        self.params = dict(kwargs)
        
        # Extract common parameters
        self.name = kwargs.get("name", self.__class__.__name__)
        self.symbols = kwargs.get("symbols", [])
        self.timeframe = kwargs.get("timeframe", "1d")
        
        # Optional: Enable/disable logging
        self.verbose = kwargs.get("verbose", False)
        
        logger.info(f"Initialized {self.name} with params: {self.params}")
    
    # ========================================================================
    # ABSTRACT METHODS
    # ========================================================================
    
    @abstractmethod
    def generate_signal(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> int:
        """
        Generate trading signal based on market data.
        
        This is the core method that each strategy must implement.
        Analyze the provided data and return a signal indicating what
        action should be taken.
        
        Args:
            data: Market data in one of two formats:
                1. pd.DataFrame: Historical OHLCV data with columns:
                   - open, high, low, close, volume
                   - May include additional indicator columns
                   - Index should be datetime
                
                2. Dict[str, Any]: Single bar/tick data with keys:
                   - 'open', 'high', 'low', 'close', 'volume'
                   - 'timestamp' or 'datetime'
                   - Any additional fields
        
        Returns:
            Signal as integer:
            - +1: Buy/Long signal (bullish)
            - -1: Sell/Short signal (bearish)
            - 0: Hold/No action (neutral or no signal)
        
        Raises:
            ValueError: If data format is invalid
            
        Example:
            def generate_signal(self, data):
                if isinstance(data, pd.DataFrame):
                    # Use latest values from dataframe
                    close = data['close'].iloc[-1]
                    sma_fast = data['close'].rolling(10).mean().iloc[-1]
                    sma_slow = data['close'].rolling(50).mean().iloc[-1]
                else:
                    # Use dict values
                    close = data['close']
                    # You'd need to maintain state for indicators
                    sma_fast = self._calculate_sma_fast(close)
                    sma_slow = self._calculate_sma_slow(close)
                
                if sma_fast > sma_slow:
                    return 1  # Buy signal
                elif sma_fast < sma_slow:
                    return -1  # Sell signal
                else:
                    return 0  # Hold
        """
        pass
    
    # ========================================================================
    # OPTIONAL METHODS (Can be overridden)
    # ========================================================================
    
    def initialize(self) -> None:
        """
        Initialize strategy state before first signal generation.
        
        Override this method to:
        - Load historical data
        - Pre-calculate indicators
        - Set up any stateful components
        - Validate parameters
        
        Called once before strategy starts generating signals.
        """
        pass
    
    @trace
    def on_bar(self, bar: Dict[str, Any]) -> int:
        """
        Process a single bar and generate signal.

        This is a convenience method that wraps generate_signal for
        real-time bar-by-bar processing. Override if you need special
        handling for streaming data.

        Args:
            bar: Single bar data dict with OHLCV fields

        Returns:
            Signal as integer (1, -1, or 0)
        """
        return self.generate_signal(bar)

    @trace
    def generate_signal_traced(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> int:
        """
        Traced wrapper for generate_signal.

        Use this method when you want execution tracing for signal generation.
        Calls the abstract generate_signal method with tracing enabled.

        Args:
            data: Market data (DataFrame or dict)

        Returns:
            Signal as integer (1, -1, or 0)
        """
        return self.generate_signal(data)
    
    def on_data(self, data: pd.DataFrame, vectorized: bool = True) -> pd.DataFrame:
        """
        Process batch data and generate signals for each row.

        Args:
            data: Historical OHLCV DataFrame
            vectorized: If True, use vectorized generation (10-50x faster).
                        Set False for strict row-by-row simulation.

        Returns:
            DataFrame with added 'signal' column

        Performance:
            - vectorized=True: O(1) calls to generate_signals_vectorized
            - vectorized=False: O(n) calls to generate_signal (slower but exact)
        """
        if vectorized:
            # Try vectorized method first (much faster for backtesting)
            signals = self.generate_signals_vectorized(data)
            if signals is not None:
                data_copy = data.copy()
                data_copy['signal'] = signals
                return data_copy

        # Fallback to row-by-row (slower but compatible with all strategies)
        n = len(data)
        signals = [0] * n  # Pre-allocate

        for i in range(n):
            signals[i] = self.generate_signal(data.iloc[:i+1])

        data_copy = data.copy()
        data_copy['signal'] = signals
        return data_copy

    def generate_signals_vectorized(self, data: pd.DataFrame) -> Optional[List[int]]:
        """
        Generate signals for entire DataFrame at once (vectorized).

        Override this method for 10-50x faster backtesting performance.
        Default implementation returns None, falling back to row-by-row.

        Args:
            data: Full historical OHLCV DataFrame

        Returns:
            List of signals (same length as data), or None to use fallback

        Example:
            def generate_signals_vectorized(self, data):
                close = data['Close']
                sma_fast = close.rolling(10).mean()
                sma_slow = close.rolling(50).mean()

                signals = np.where(sma_fast > sma_slow, 1,
                           np.where(sma_fast < sma_slow, -1, 0))
                return signals.tolist()
        """
        return None  # Subclasses override for vectorized operation
    
    def cleanup(self) -> None:
        """
        Clean up strategy resources.
        
        Override this method to:
        - Close file handles
        - Save state
        - Release resources
        
        Called when strategy is stopped or system shuts down.
        """
        pass
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """
        Get strategy parameter with default fallback.
        
        Args:
            key: Parameter name
            default: Default value if parameter not found
            
        Returns:
            Parameter value or default
        """
        return self.params.get(key, default)
    
    def set_param(self, key: str, value: Any) -> None:
        """
        Set or update a strategy parameter.
        
        Args:
            key: Parameter name
            value: Parameter value
        """
        self.params[key] = value
        logger.debug(f"{self.name}: Set {key} = {value}")
    
    def validate_data(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> bool:
        """
        Validate that data has required fields.
        
        Args:
            data: Market data to validate
            
        Returns:
            True if data is valid
            
        Raises:
            ValueError: If data format is invalid
        """
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        
        if isinstance(data, pd.DataFrame):
            missing = [f for f in required_fields if f not in data.columns]
            if missing:
                raise ValueError(f"DataFrame missing required columns: {missing}")
            
            if len(data) == 0:
                raise ValueError("DataFrame is empty")
            
            return True
        
        elif isinstance(data, dict):
            missing = [f for f in required_fields if f not in data]
            if missing:
                raise ValueError(f"Bar data missing required fields: {missing}")
            
            return True
        
        else:
            raise ValueError(
                f"Invalid data type: {type(data)}. "
                "Expected pd.DataFrame or dict"
            )
    
    def log(self, message: str, level: str = "info") -> None:
        """
        Log a message if verbose mode is enabled.
        
        Args:
            message: Message to log
            level: Log level (debug, info, warning, error)
        """
        if not self.verbose:
            return
        
        log_fn = getattr(logger, level, logger.info)
        log_fn(f"[{self.name}] {message}")
    
    # ========================================================================
    # SPECIAL METHODS
    # ========================================================================
    
    def __repr__(self) -> str:
        """String representation of strategy."""
        return f"{self.name}(params={self.params})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.name