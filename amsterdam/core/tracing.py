"""
Execution Tracing - Trace function calls through the trading system

Provides visibility into the execution flow for debugging and analysis.
Toggle via config: logging.trace_enabled = true

Usage:
    from core.tracing import trace, get_tracer

    @trace
    def my_function(x, y):
        return x + y

    # Or trace a block:
    with get_tracer().span("my_operation"):
        do_something()

Output (trace.log):
    {"ts": "2024-01-15T10:30:00.123", "event": "ENTER", "func": "handle_signal_context", "symbol": "AAPL", "signal": 1}
    {"ts": "2024-01-15T10:30:00.124", "event": "ENTER", "func": "_check_trade_approval", "symbol": "AAPL"}
    {"ts": "2024-01-15T10:30:00.125", "event": "EXIT", "func": "_check_trade_approval", "result": [true, null], "elapsed_ms": 1.2}
    {"ts": "2024-01-15T10:30:00.130", "event": "EXIT", "func": "handle_signal_context", "elapsed_ms": 7.5}
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Thread-local storage for call stack tracking
_local = threading.local()


class Tracer:
    """
    Execution tracer for the trading system.

    Captures function entry/exit with timing and key parameters.
    Writes structured JSON to trace.log for analysis.
    """

    _instance: Tracer | None = None

    def __init__(self):
        self.enabled = False
        self.log_file: Path | None = None
        self._file_handle = None
        self._lock = threading.Lock()
        self._depth = 0

        # Parameters to extract from args (for readable traces)
        self.interesting_params = {
            "symbol",
            "signal",
            "price",
            "side",
            "qty",
            "action_type",
            "regime",
            "strategy_name",
            "reason",
            "result",
        }

    @classmethod
    def get_instance(cls) -> Tracer:
        """Get singleton tracer instance."""
        if cls._instance is None:
            cls._instance = Tracer()
        return cls._instance

    def configure(self, enabled: bool = True, log_path: str = "logs/trace.log"):
        """
        Configure the tracer.

        Args:
            enabled: Whether tracing is active
            log_path: Path to trace log file
        """
        self.enabled = enabled

        if enabled:
            self.log_file = Path(log_path)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

            # Open file for append
            if self._file_handle:
                self._file_handle.close()
            self._file_handle = open(self.log_file, "a")

            self._write_event({"event": "TRACE_START", "message": "Tracing enabled"})

    def _write_event(self, event: dict[str, Any]):
        """Write event to trace log."""
        if not self.enabled or not self._file_handle:
            return

        event["ts"] = datetime.now(timezone.utc).isoformat()
        event["depth"] = getattr(_local, "depth", 0)

        with self._lock:
            try:
                self._file_handle.write(json.dumps(event) + "\n")
                self._file_handle.flush()
            except Exception:
                pass  # Don't let tracing errors affect execution

    def enter(self, func_name: str, args_dict: dict[str, Any] = None):
        """Record function entry."""
        if not self.enabled:
            return

        # Increment depth
        _local.depth = getattr(_local, "depth", 0) + 1

        event = {
            "event": "ENTER",
            "func": func_name,
        }

        # Add interesting parameters
        if args_dict:
            for key in self.interesting_params:
                if key in args_dict and args_dict[key] is not None:
                    val = args_dict[key]
                    # Make JSON serializable
                    if hasattr(val, "value"):  # Enum
                        val = val.value
                    elif hasattr(val, "__class__") and val.__class__.__name__ == "SymbolState":
                        val = f"<SymbolState:{getattr(val, 'symbol', '?')}>"
                    elif not isinstance(val, (str, int, float, bool, type(None))):
                        val = str(val)[:100]
                    event[key] = val

        self._write_event(event)

    def exit(self, func_name: str, result: Any = None, elapsed_ms: float = None, error: str = None):
        """Record function exit."""
        if not self.enabled:
            return

        event = {
            "event": "EXIT",
            "func": func_name,
        }

        if elapsed_ms is not None:
            event["elapsed_ms"] = round(elapsed_ms, 3)

        if error:
            event["error"] = str(error)[:200]
        elif result is not None:
            # Summarize result
            if isinstance(result, tuple) and len(result) == 2:
                # Likely (bool, reason) tuple
                event["result"] = list(result)
            elif isinstance(result, bool):
                event["result"] = result
            elif hasattr(result, "success"):
                event["result"] = {"success": result.success}
            elif result is not None:
                event["result"] = str(result)[:100]

        self._write_event(event)

        # Decrement depth
        _local.depth = max(0, getattr(_local, "depth", 1) - 1)

    @contextmanager
    def span(self, name: str, **kwargs):
        """
        Context manager for tracing a block of code.

        Usage:
            with tracer.span("calculate_size", symbol="AAPL"):
                size = calculate(...)
        """
        if not self.enabled:
            yield
            return

        self.enter(name, kwargs)
        start = time.perf_counter()

        try:
            yield
            elapsed = (time.perf_counter() - start) * 1000
            self.exit(name, elapsed_ms=elapsed)
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            self.exit(name, elapsed_ms=elapsed, error=str(e))
            raise

    def close(self):
        """Close trace log file."""
        if self._file_handle:
            self._write_event({"event": "TRACE_END"})
            self._file_handle.close()
            self._file_handle = None


def get_tracer() -> Tracer:
    """Get the global tracer instance."""
    return Tracer.get_instance()


def trace(func: Callable = None, *, extract: list[str] = None):
    """
    Decorator to trace function entry/exit.

    Usage:
        @trace
        def my_function(symbol, price):
            ...

        @trace(extract=['symbol', 'signal'])
        async def handle_signal(self, context):
            ...

    Args:
        func: Function to wrap
        extract: Parameter names to extract and log
    """

    def decorator(fn):
        # Get parameter names from signature
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            if not tracer.enabled:
                return fn(*args, **kwargs)

            # Build args dict
            args_dict = _build_args_dict(fn, args, kwargs, param_names, extract)

            tracer.enter(fn.__qualname__, args_dict)
            start = time.perf_counter()

            try:
                result = fn(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                tracer.exit(fn.__qualname__, result=result, elapsed_ms=elapsed)
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                tracer.exit(fn.__qualname__, elapsed_ms=elapsed, error=str(e))
                raise

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            if not tracer.enabled:
                return await fn(*args, **kwargs)

            # Build args dict
            args_dict = _build_args_dict(fn, args, kwargs, param_names, extract)

            tracer.enter(fn.__qualname__, args_dict)
            start = time.perf_counter()

            try:
                result = await fn(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                tracer.exit(fn.__qualname__, result=result, elapsed_ms=elapsed)
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                tracer.exit(fn.__qualname__, elapsed_ms=elapsed, error=str(e))
                raise

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator


def _build_args_dict(fn, args, kwargs, param_names, extract_params):
    """Build dictionary of interesting arguments."""
    args_dict = {}

    # Map positional args to names
    for i, arg in enumerate(args):
        if i < len(param_names):
            name = param_names[i]
            if name != "self":
                args_dict[name] = arg

    # Add kwargs
    args_dict.update(kwargs)

    # Extract nested attributes (e.g., context.symbol)
    if "context" in args_dict:
        ctx = args_dict["context"]
        for attr in ["symbol", "signal", "price", "regime", "strategy_name"]:
            if hasattr(ctx, attr):
                args_dict[attr] = getattr(ctx, attr)
        del args_dict["context"]  # Don't log full context object

    # Filter to extract list if specified
    if extract_params:
        args_dict = {k: v for k, v in args_dict.items() if k in extract_params}

    return args_dict


def configure_tracing_from_config():
    """
    Configure tracing from trading_config.json.

    Call this at startup to enable tracing if configured.
    """
    try:
        from core.config_loader import get_config

        cfg = get_config()

        trace_enabled = getattr(cfg.logging, "trace_enabled", False)
        trace_file = getattr(cfg.logging, "trace_file", "logs/trace.log")

        if trace_enabled:
            tracer = get_tracer()
            tracer.configure(enabled=True, log_path=trace_file)
            print(f"[TRACE] Tracing enabled -> {trace_file}")
    except Exception as e:
        print(f"[TRACE] Failed to configure: {e}")


# Auto-configure on import if already configured
try:
    configure_tracing_from_config()
except Exception:
    pass
