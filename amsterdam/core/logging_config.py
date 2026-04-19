"""
Logging Configuration - Centralized logging setup with correlation IDs

Activates the structured logging utilities from core/utils/logging_utils.py
and provides easy access to component loggers.

Features:
- Correlation ID support for tracing operations
- Optional JSON structured logging (for production)
- Sensitive data masking
- Component-specific loggers

Usage:
    from core.logging_config import get_component_logger, generate_correlation_id

    logger = get_component_logger("LiveExecutionEngine")

    # With correlation ID
    correlation_id = generate_correlation_id()
    logger.info(f"[{correlation_id}] Processing signal for AAPL")

    # With structured context
    logger = get_component_logger("TradeValidator", structured=True)
    logger.info("Validation passed", extra={"symbol": "AAPL", "action": "entry"})

Environment Variables:
    STRUCTURED_LOGS: Set to "true" to enable JSON output (default: "false")
    LOG_LEVEL: Log level (default: "INFO")
"""

from __future__ import annotations

import logging
import os
from typing import Any

from core.utils.logging_utils import (
    ContextLogger,
    StructuredFormatter,
    async_with_correlation_id,
    generate_correlation_id,
    get_correlation_id,
    set_correlation_id,
    setup_structured_logging,
    with_correlation_id,
)

# Re-export for convenience
__all__ = [
    "get_component_logger",
    "generate_correlation_id",
    "set_correlation_id",
    "get_correlation_id",
    "setup_structured_logging",
    "async_with_correlation_id",
    "with_correlation_id",
    "StructuredFormatter",
    "ContextLogger",
]


def get_component_logger(
    name: str,
    structured: bool | None = None,
    context: dict[str, Any] | None = None,
) -> ContextLogger:
    """
    Get a logger with correlation ID support and optional JSON formatting.

    Args:
        name: Logger name (e.g., "LiveExecutionEngine", "StateReconciler")
        structured: Use JSON format. If None, checks STRUCTURED_LOGS env var.
        context: Initial context dict to include in all log messages

    Returns:
        ContextLogger with correlation ID support

    Example:
        logger = get_component_logger("LiveExecutionEngine")
        logger.info("Signal received", extra={"symbol": "AAPL"})

        # JSON output (if STRUCTURED_LOGS=true):
        # {"timestamp": "...", "level": "INFO", "logger": "LiveExecutionEngine",
        #  "correlation_id": "abc123", "message": "Signal received",
        #  "extra": {"symbol": "AAPL"}}

        # Text output (default):
        # 14:30:00 | INFO     | LiveExecutionEngine | [abc123] Signal received
    """
    # Get base logger
    logger = logging.getLogger(name)

    # Determine if structured logging is enabled
    if structured is None:
        structured = os.environ.get("STRUCTURED_LOGS", "false").lower() == "true"

    # Set level from environment
    level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    # Add handler if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)

        if structured:
            handler.setFormatter(StructuredFormatter())
        else:
            # Text format with correlation ID placeholder
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S")
            )

        logger.addHandler(handler)

    # Return wrapped logger with context support
    return ContextLogger(logger, context or {})


def get_trading_logger(
    symbol: str,
    strategy: str | None = None,
    correlation_id: str | None = None,
) -> ContextLogger:
    """
    Get a logger pre-configured for trading operations.

    Includes symbol and optional strategy in all log messages.

    Args:
        symbol: Trading symbol
        strategy: Strategy name (optional)
        correlation_id: Correlation ID (generates new if not provided)

    Returns:
        ContextLogger with trading context

    Example:
        logger = get_trading_logger("AAPL", "momentum")
        logger.info("Entry signal detected")
        # Output: 14:30:00 | INFO | Trading | [abc123] [AAPL] [momentum] Entry signal detected
    """
    if correlation_id:
        set_correlation_id(correlation_id)

    context = {"symbol": symbol}
    if strategy:
        context["strategy"] = strategy

    return get_component_logger("Trading", context=context)


def format_log_message(
    message: str, correlation_id: str | None = None, symbol: str | None = None, **kwargs: Any
) -> str:
    """
    Format a log message with optional prefixes.

    Helper for consistent log formatting across components.

    Args:
        message: Base message
        correlation_id: Correlation ID to prefix
        symbol: Symbol to prefix
        **kwargs: Additional key=value pairs to include

    Returns:
        Formatted message string

    Example:
        msg = format_log_message(
            "Order placed",
            correlation_id="abc123",
            symbol="AAPL",
            side="buy",
            qty=10
        )
        # Returns: "[abc123] [AAPL] Order placed (side=buy, qty=10)"
    """
    parts = []

    if correlation_id:
        parts.append(f"[{correlation_id}]")

    if symbol:
        parts.append(f"[{symbol}]")

    parts.append(message)

    if kwargs:
        extras = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        parts.append(f"({extras})")

    return " ".join(parts)
