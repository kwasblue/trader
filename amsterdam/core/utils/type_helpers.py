# core/utils/type_helpers.py
"""
Type conversion and normalization utilities.

Shared helpers for safely converting and normalizing values across the codebase.
Centralizes common patterns to avoid duplication.
"""

from __future__ import annotations

from typing import Any


def to_float(value: Any) -> float | None:
    """
    Safely convert a value to float.

    Args:
        value: Any value to convert (string, int, float, None, etc.)

    Returns:
        Float value if conversion succeeds, None otherwise

    Examples:
        >>> to_float("123.45")
        123.45
        >>> to_float(None)
        None
        >>> to_float("invalid")
        None
    """
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def to_int(value: Any) -> int | None:
    """
    Safely convert a value to int.

    Args:
        value: Any value to convert

    Returns:
        Int value if conversion succeeds, None otherwise
    """
    if value is None:
        return None
    try:
        return int(float(value))  # Handle "123.0" strings
    except (ValueError, TypeError):
        return None


def to_bool(value: Any) -> bool | None:
    """
    Safely convert a value to bool.

    Handles string values like "true", "false", "1", "0".

    Args:
        value: Any value to convert

    Returns:
        Bool value if conversion succeeds, None for unrecognized values
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lower = value.lower().strip()
        if lower in ("true", "1", "yes", "on"):
            return True
        if lower in ("false", "0", "no", "off"):
            return False
    return None


def coalesce(*values: Any) -> Any:
    """
    Return the first non-None value.

    Args:
        *values: Values to check

    Returns:
        First non-None value, or None if all are None

    Example:
        >>> coalesce(None, None, 42, 100)
        42
    """
    for v in values:
        if v is not None:
            return v
    return None
