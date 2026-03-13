"""
Trade Logger - Record and track all trading activity

Provides framework for logging trades, performance, errors, and summaries
to various destinations (files, databases, cloud services, etc.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import logging

from core.enums import OrderSide
from core.app_types import OrderResult

logger = logging.getLogger(__name__)


class TradeLoggerBase(ABC):
    """
    Abstract base class for trade logging and performance tracking.
    
    Trade loggers record all trading activity for:
    - Audit trail (compliance, debugging)
    - Performance analysis (metrics, reports)
    - Real-time monitoring (dashboards, alerts)
    - Historical analysis (backtesting validation)
    
    Design Philosophy:
    - Logger only RECORDS activity
    - Does NOT make trading decisions
    - Does NOT enforce rules
    - Can output to multiple destinations (file, DB, API, etc.)
    
    Common Implementations:
    - FileTradeLogger: Log to files
    - DatabaseTradeLogger: Log to SQL database
    - CloudTradeLogger: Log to cloud service
    - MultiTradeLogger: Log to multiple destinations
    
    Example:
        logger = FileTradeLogger("trades.csv")
        
        # Log trade
        logger.log_trade(
            symbol="AAPL",
            action=OrderSide.BUY,
            price=150.25,
            quantity=100,
            strategy="momentum"
        )
        
        # Log summary
        logger.log_summary({
            'total_pnl': 1234.56,
            'win_rate': 0.65,
            'sharpe_ratio': 1.8
        })
    """
    
    def __init__(self, **kwargs):
        """
        Initialize trade logger.
        
        Args:
            **kwargs: Logger-specific configuration
        """
        self.params = dict(kwargs)
    
    # ========================================================================
    # ABSTRACT METHODS - TRADE LOGGING
    # ========================================================================
    
    @abstractmethod
    def log_trade(
        self,
        symbol: str,
        action: OrderSide,
        price: float,
        quantity: float,
        timestamp: Optional[datetime] = None,
        order_id: Optional[str] = None,
        commission: float = 0.0,
        slippage: float = 0.0,
        pnl: Optional[float] = None,
        strategy: Optional[str] = None,
        regime: Optional[str] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        trailing_stop: Optional[float] = None,
        cash_before: Optional[float] = None,
        cash_after: Optional[float] = None,
        position_before: Optional[float] = None,
        position_after: Optional[float] = None,
        notes: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Log a trade execution.
        
        Records all details of a completed trade for audit trail and
        performance analysis.
        
        Args:
            symbol: Trading symbol
            action: Order side (OrderSide.BUY or OrderSide.SELL)
            price: Execution price
            quantity: Quantity traded
            timestamp: Execution time (defaults to now if None)
            order_id: Unique order identifier
            commission: Commission paid
            slippage: Slippage incurred
            pnl: Realized P&L from this trade (if closing position)
            strategy: Strategy that generated signal
            regime: Market regime at time of trade
            sl: Stop loss price (if set)
            tp: Take profit price (if set)
            trailing_stop: Trailing stop distance (if set)
            cash_before: Cash balance before trade
            cash_after: Cash balance after trade
            position_before: Position size before trade
            position_after: Position size after trade
            notes: Additional notes or metadata
            **kwargs: Additional custom fields
            
        Example:
            logger.log_trade(
                symbol="AAPL",
                action=OrderSide.BUY,
                price=150.25,
                quantity=100,
                commission=1.50,
                strategy="momentum",
                regime="trending",
                cash_before=100000,
                cash_after=84973.50
            )
        """
        pass
    
    @abstractmethod
    def log_order_status(
        self,
        order_id: str,
        symbol: str,
        status: str,
        side: OrderSide,
        quantity: float,
        filled_quantity: float = 0.0,
        avg_price: Optional[float] = None,
        timestamp: Optional[datetime] = None,
        reason: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Log order status change.
        
        Records order lifecycle events (submitted, filled, cancelled, etc.)
        
        Args:
            order_id: Unique order identifier
            symbol: Trading symbol
            status: Order status (pending, filled, cancelled, rejected, etc.)
            side: Order side (BUY or SELL)
            quantity: Total order quantity
            filled_quantity: Quantity filled so far
            avg_price: Average fill price
            timestamp: Status change time
            reason: Reason for rejection/cancellation
            **kwargs: Additional fields
        """
        pass
    
    # ========================================================================
    # ABSTRACT METHODS - GENERAL LOGGING
    # ========================================================================
    
    @abstractmethod
    def log_error(
        self,
        message: str,
        error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Log an error or exception.
        
        Args:
            message: Error description
            error: Exception object (if applicable)
            context: Additional context (symbol, order_id, etc.)
            timestamp: Error time (defaults to now)
            
        Example:
            try:
                broker.place_order(...)
            except OrderRejectedError as e:
                logger.log_error(
                    "Order rejected",
                    error=e,
                    context={'symbol': 'AAPL', 'reason': e.reason}
                )
        """
        pass
    
    @abstractmethod
    def log_info(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Log general information.
        
        Used for:
        - Signal generation
        - Regime changes
        - Position sizing decisions
        - Risk management actions
        
        Args:
            message: Info message
            context: Additional context data
            timestamp: Log time (defaults to now)
            
        Example:
            logger.log_info(
                "Market regime changed",
                context={'from': 'ranging', 'to': 'trending'}
            )
        """
        pass
    
    @abstractmethod
    def log_warning(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Log warning message.
        
        Used for:
        - Risk limit approaches
        - Unusual market conditions
        - Performance degradation
        
        Args:
            message: Warning message
            context: Additional context
            timestamp: Warning time
        """
        pass
    
    # ========================================================================
    # ABSTRACT METHODS - PERFORMANCE TRACKING
    # ========================================================================
    
    @abstractmethod
    def log_summary(
        self,
        summary_stats: Dict[str, Any],
        period: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Log performance summary.
        
        Records portfolio metrics at regular intervals (daily, weekly, etc.)
        
        Args:
            summary_stats: Performance metrics dict with keys like:
                - 'total_pnl': Total P&L
                - 'win_rate': Win rate
                - 'sharpe_ratio': Sharpe ratio
                - 'max_drawdown': Maximum drawdown
                - 'total_trades': Number of trades
                - 'portfolio_value': Current portfolio value
                - etc.
            period: Period covered (e.g., "daily", "weekly", "2024-01-15")
            timestamp: Summary time (defaults to now)
            
        Example:
            logger.log_summary({
                'date': '2024-01-15',
                'total_pnl': 1234.56,
                'win_rate': 0.65,
                'sharpe_ratio': 1.8,
                'max_drawdown': -0.05,
                'total_trades': 45,
                'portfolio_value': 105234.56
            }, period="daily")
        """
        pass
    
    @abstractmethod
    def log_position_update(
        self,
        symbol: str,
        quantity: float,
        avg_price: float,
        current_price: float,
        unrealized_pnl: float,
        timestamp: Optional[datetime] = None,
        **kwargs
    ) -> None:
        """
        Log position status update.
        
        Records current position state for monitoring.
        
        Args:
            symbol: Trading symbol
            quantity: Current position size (+ long, - short)
            avg_price: Average entry price
            current_price: Current market price
            unrealized_pnl: Current unrealized P&L
            timestamp: Update time
            **kwargs: Additional fields (market_value, etc.)
        """
        pass
    
    # ========================================================================
    # ABSTRACT METHODS - PERSISTENCE
    # ========================================================================
    
    @abstractmethod
    def flush(self) -> None:
        """
        Flush any buffered logs to persistent storage.
        
        Call this:
        - At end of trading session
        - Before system shutdown
        - Periodically for safety
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        Close logger and release resources.
        
        Call this when done logging (end of day, system shutdown).
        """
        pass
    
    # ========================================================================
    # OPTIONAL METHODS - QUERYING
    # ========================================================================
    
    def get_trades(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        strategy: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve logged trades (optional).
        
        Override this if your logger supports querying.
        
        Args:
            symbol: Filter by symbol
            start_date: Filter from date
            end_date: Filter to date
            strategy: Filter by strategy
            
        Returns:
            List of trade records
        """
        raise NotImplementedError("Query not supported by this logger")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics (optional).
        
        Override this if your logger tracks metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        raise NotImplementedError("Metrics not supported by this logger")
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _get_timestamp(self, timestamp: Optional[datetime] = None) -> datetime:
        """Get timestamp, defaulting to current UTC time."""
        return timestamp or datetime.now(timezone.utc)
    
    def _format_trade_id(
        self,
        symbol: str,
        timestamp: datetime,
        side: OrderSide
    ) -> str:
        """Generate unique trade ID."""
        return f"{symbol}_{side.value}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    
    def log_from_order_result(self, result: OrderResult, **kwargs) -> None:
        """
        Convenience method to log from OrderResult.
        
        Args:
            result: OrderResult from broker
            **kwargs: Additional fields (strategy, regime, etc.)
        """
        self.log_trade(
            symbol=result.symbol,
            action=result.side,
            price=result.avg_price,
            quantity=result.filled_qty,
            order_id=result.order_id,
            commission=getattr(result, 'commission', 0.0),
            **kwargs
        )
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """Get configuration parameter."""
        return self.params.get(key, default)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.flush()
        self.close()