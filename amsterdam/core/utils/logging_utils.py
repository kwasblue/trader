"""
Structured logging utilities with correlation IDs and sensitive data masking.

Provides production-ready logging for trading systems.
"""
from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from functools import wraps

# Context variable for correlation ID (thread-safe)
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    'correlation_id', default=''
)


def get_correlation_id() -> str:
    """Get current correlation ID or generate new one."""
    cid = correlation_id_var.get()
    if not cid:
        cid = generate_correlation_id()
        correlation_id_var.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Set correlation ID for current context."""
    correlation_id_var.set(cid)


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())[:8]


# Patterns for sensitive data masking
SENSITIVE_PATTERNS: List[tuple] = [
    # API keys and secrets
    (re.compile(r'(api[_-]?key|apikey)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{8,})', re.I), r'\1=***MASKED***'),
    (re.compile(r'(secret|password|token|auth)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{8,})', re.I), r'\1=***MASKED***'),
    # Bearer tokens
    (re.compile(r'(Bearer\s+)([a-zA-Z0-9._-]{20,})', re.I), r'\1***MASKED***'),
    # Account numbers (mask middle digits)
    (re.compile(r'(account[_-]?(number|id)?)["\']?\s*[:=]\s*["\']?(\d{4})(\d+)(\d{4})', re.I), r'\1=\3****\5'),
    # Credit card patterns
    (re.compile(r'\b(\d{4})[-\s]?(\d{4})[-\s]?(\d{4})[-\s]?(\d{4})\b'), r'\1-****-****-\4'),
    # SSN patterns
    (re.compile(r'\b(\d{3})[-\s]?(\d{2})[-\s]?(\d{4})\b'), r'***-**-\3'),
    # Email addresses (partial mask)
    (re.compile(r'([a-zA-Z0-9._%+-]+)(@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'), lambda m: m.group(1)[:2] + '***' + m.group(2)),
]

# Fields to always mask completely
SENSITIVE_FIELDS: Set[str] = {
    'password', 'secret', 'api_key', 'apikey', 'api_secret',
    'access_token', 'refresh_token', 'token', 'authorization',
    'private_key', 'secret_key', 'credentials', 'auth',
}


def mask_sensitive_data(data: Any, depth: int = 0) -> Any:
    """
    Recursively mask sensitive data in dictionaries, lists, and strings.

    Args:
        data: Data to mask
        depth: Current recursion depth (to prevent infinite loops)

    Returns:
        Masked data
    """
    if depth > 10:  # Prevent infinite recursion
        return data

    if isinstance(data, dict):
        return {
            k: '***MASKED***' if k.lower() in SENSITIVE_FIELDS
            else mask_sensitive_data(v, depth + 1)
            for k, v in data.items()
        }

    if isinstance(data, list):
        return [mask_sensitive_data(item, depth + 1) for item in data]

    if isinstance(data, str):
        result = data
        for pattern, replacement in SENSITIVE_PATTERNS:
            if callable(replacement):
                result = pattern.sub(replacement, result)
            else:
                result = pattern.sub(replacement, result)
        return result

    return data


class StructuredFormatter(logging.Formatter):
    """
    JSON structured log formatter with correlation ID and context.

    Outputs logs in JSON format for easy parsing by log aggregators.
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_correlation_id: bool = True,
        mask_sensitive: bool = True,
    ):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_correlation_id = include_correlation_id
        self.mask_sensitive = mask_sensitive

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if self.include_timestamp:
            log_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        if self.include_correlation_id:
            cid = correlation_id_var.get()
            if cid:
                log_data["correlation_id"] = cid

        # Add source location
        log_data["source"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        extra_fields = {
            k: v for k, v in record.__dict__.items()
            if k not in {
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'pathname', 'process', 'processName', 'relativeCreated',
                'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
                'message', 'taskName',
            }
        }
        if extra_fields:
            log_data["extra"] = extra_fields

        # Mask sensitive data
        if self.mask_sensitive:
            log_data = mask_sensitive_data(log_data)

        return json.dumps(log_data, default=str)


class ContextLogger(logging.LoggerAdapter):
    """
    Logger adapter that automatically includes context in log messages.

    Adds correlation ID, symbol, and other context to all log messages.
    """

    def __init__(self, logger: logging.Logger, context: Optional[Dict[str, Any]] = None):
        super().__init__(logger, context or {})

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        # Add correlation ID
        extra = kwargs.get('extra', {})
        extra['correlation_id'] = get_correlation_id()

        # Add context from adapter
        extra.update(self.extra)

        kwargs['extra'] = extra
        return msg, kwargs

    def with_context(self, **context) -> "ContextLogger":
        """Create a new logger with additional context."""
        new_context = {**self.extra, **context}
        return ContextLogger(self.logger, new_context)


def get_structured_logger(
    name: str,
    level: int = logging.INFO,
    context: Optional[Dict[str, Any]] = None,
) -> ContextLogger:
    """
    Get a structured logger with context support.

    Args:
        name: Logger name
        level: Log level
        context: Initial context dict

    Returns:
        ContextLogger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return ContextLogger(logger, context or {})


def setup_structured_logging(
    level: int = logging.INFO,
    json_output: bool = True,
    stream: Any = None,
) -> None:
    """
    Configure root logger for structured logging.

    Args:
        level: Log level
        json_output: Use JSON format (True for production)
        stream: Output stream (default: stderr)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add new handler
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(level)

    if json_output:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | [%(correlation_id)s] %(message)s',
            datefmt='%H:%M:%S'
        ))

    root_logger.addHandler(handler)


def with_correlation_id(func):
    """
    Decorator to ensure function has a correlation ID.

    Creates new correlation ID if none exists.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not correlation_id_var.get():
            correlation_id_var.set(generate_correlation_id())
        return func(*args, **kwargs)
    return wrapper


def async_with_correlation_id(func):
    """
    Async decorator to ensure function has a correlation ID.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not correlation_id_var.get():
            correlation_id_var.set(generate_correlation_id())
        return await func(*args, **kwargs)
    return wrapper
