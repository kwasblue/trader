"""
Custom Exceptions for Trading System

This module defines all custom exceptions used throughout the trading system,
organized by category for better error handling and debugging.
"""

from typing import Optional


# ============================================================================
# BASE EXCEPTIONS
# ============================================================================

class TradingSystemError(Exception):
    """Base exception for all trading system errors."""
    pass


# ============================================================================
# BROKER EXCEPTIONS
# ============================================================================

class BrokerError(TradingSystemError):
    """Base exception for broker-related errors."""
    
    def __init__(self, message: str, broker_name: Optional[str] = None):
        self.broker_name = broker_name
        super().__init__(message)


class ConnectionError(BrokerError):
    """Raised when unable to connect to broker."""
    pass


class AuthenticationError(BrokerError):
    """Raised when broker authentication fails."""
    pass


class OrderError(BrokerError):
    """Base exception for order-related errors."""
    pass


class OrderNotFoundError(OrderError):
    """Raised when order ID doesn't exist."""
    
    def __init__(self, order_id: str, message: Optional[str] = None):
        self.order_id = order_id
        msg = message or f"Order not found: {order_id}"
        super().__init__(msg)


class OrderRejectedError(OrderError):
    """Raised when broker rejects an order."""
    
    def __init__(self, reason: str, order_details: Optional[dict] = None):
        self.reason = reason
        self.order_details = order_details
        super().__init__(f"Order rejected: {reason}")


class OrderCancellationError(OrderError):
    """Raised when order cancellation fails."""
    pass


class InsufficientFundsError(BrokerError):
    """Raised when account has insufficient funds for an order."""
    
    def __init__(self, required: float, available: float):
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient funds: required ${required:,.2f}, "
            f"available ${available:,.2f}"
        )


class PositionError(BrokerError):
    """Base exception for position-related errors."""
    pass


class PositionNotFoundError(PositionError):
    """Raised when position doesn't exist."""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        super().__init__(f"No position found for symbol: {symbol}")


class InsufficientPositionError(PositionError):
    """Raised when trying to sell more shares than held."""
    
    def __init__(self, symbol: str, requested: float, available: float):
        self.symbol = symbol
        self.requested = requested
        self.available = available
        super().__init__(
            f"Insufficient position in {symbol}: "
            f"requested {requested}, available {available}"
        )


# ============================================================================
# MARKET DATA EXCEPTIONS
# ============================================================================

class MarketDataError(TradingSystemError):
    """Base exception for market data errors."""
    pass


class SymbolNotFoundError(MarketDataError):
    """Raised when symbol is invalid or not found."""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        super().__init__(f"Symbol not found: {symbol}")


class MarketClosedError(MarketDataError):
    """Raised when attempting to trade while market is closed."""
    
    def __init__(self, message: Optional[str] = None):
        msg = message or "Market is currently closed"
        super().__init__(msg)


class DataUnavailableError(MarketDataError):
    """Raised when market data is temporarily unavailable."""
    pass


class StaleDataError(MarketDataError):
    """Raised when market data is too old to be reliable."""
    
    def __init__(self, symbol: str, age_seconds: float, max_age: float):
        self.symbol = symbol
        self.age_seconds = age_seconds
        self.max_age = max_age
        super().__init__(
            f"Stale data for {symbol}: {age_seconds:.1f}s old "
            f"(max: {max_age:.1f}s)"
        )


# ============================================================================
# STRATEGY EXCEPTIONS
# ============================================================================

class StrategyError(TradingSystemError):
    """Base exception for strategy-related errors."""
    pass


class StrategyInitializationError(StrategyError):
    """Raised when strategy fails to initialize."""
    pass


class StrategyConfigurationError(StrategyError):
    """Raised when strategy configuration is invalid."""
    pass


class SignalGenerationError(StrategyError):
    """Raised when strategy fails to generate signals."""
    pass


class RiskViolationError(StrategyError):
    """Raised when trade would violate risk management rules."""
    
    def __init__(self, rule: str, message: str):
        self.rule = rule
        super().__init__(f"Risk violation ({rule}): {message}")


# ============================================================================
# CONFIGURATION EXCEPTIONS
# ============================================================================

class ConfigurationError(TradingSystemError):
    """Base exception for configuration errors."""
    pass


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing."""
    
    def __init__(self, key: str):
        self.key = key
        super().__init__(f"Missing required configuration: {key}")


class InvalidConfigError(ConfigurationError):
    """Raised when configuration value is invalid."""
    
    def __init__(self, key: str, value: any, reason: str):
        self.key = key
        self.value = value
        self.reason = reason
        super().__init__(f"Invalid config for {key}={value}: {reason}")


# ============================================================================
# VALIDATION EXCEPTIONS
# ============================================================================

class ValidationError(TradingSystemError):
    """Base exception for validation errors."""
    pass


class InvalidParameterError(ValidationError):
    """Raised when function parameter is invalid."""
    
    def __init__(self, param_name: str, value: any, reason: str):
        self.param_name = param_name
        self.value = value
        self.reason = reason
        super().__init__(f"Invalid {param_name}={value}: {reason}")


class InvalidOrderError(ValidationError):
    """Raised when order parameters are invalid."""
    pass


# ============================================================================
# BACKTESTING EXCEPTIONS
# ============================================================================

class BacktestError(TradingSystemError):
    """Base exception for backtesting errors."""
    pass


class BacktestDataError(BacktestError):
    """Raised when backtest data is invalid or incomplete."""
    pass


class BacktestConfigError(BacktestError):
    """Raised when backtest configuration is invalid."""
    pass


# ============================================================================
# RISK MANAGEMENT EXCEPTIONS
# ============================================================================

class RiskManagementError(TradingSystemError):
    """Base exception for risk management errors."""
    pass


class PositionLimitExceededError(RiskManagementError):
    """Raised when position size exceeds limits."""
    
    def __init__(self, symbol: str, requested: float, limit: float):
        self.symbol = symbol
        self.requested = requested
        self.limit = limit
        super().__init__(
            f"Position limit exceeded for {symbol}: "
            f"requested {requested}, limit {limit}"
        )


class DailyLossLimitExceededError(RiskManagementError):
    """Raised when daily loss limit is exceeded."""
    
    def __init__(self, current_loss: float, limit: float):
        self.current_loss = current_loss
        self.limit = limit
        super().__init__(
            f"Daily loss limit exceeded: "
            f"${current_loss:,.2f} loss (limit: ${limit:,.2f})"
        )


class MaxDrawdownExceededError(RiskManagementError):
    """Raised when maximum drawdown is exceeded."""
    
    def __init__(self, current_dd: float, max_dd: float):
        self.current_drawdown = current_dd
        self.max_drawdown = max_dd
        super().__init__(
            f"Max drawdown exceeded: {current_dd:.2%} "
            f"(limit: {max_dd:.2%})"
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_error_context(error: Exception) -> dict:
    """
    Extract structured context from an exception.
    
    Args:
        error: Exception instance
        
    Returns:
        Dictionary with error context for logging
    """
    context = {
        "error_type": type(error).__name__,
        "message": str(error),
    }
    
    # Add custom attributes if available
    if hasattr(error, "__dict__"):
        for key, value in error.__dict__.items():
            if not key.startswith("_"):
                context[key] = value
    
    return context
