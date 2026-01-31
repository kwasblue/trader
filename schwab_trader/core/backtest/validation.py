"""
Data Validation Module

Provides validation and cleaning utilities for OHLCV data used in backtesting.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    cleaned_data: Optional[pd.DataFrame] = None


def validate_ohlcv_data(
    data: pd.DataFrame,
    required_columns: List[str] = None,
    fix_issues: bool = True
) -> ValidationResult:
    """
    Validate OHLCV data for backtesting.

    Checks:
    - Required columns exist
    - No NaN/inf values in critical columns
    - Prices are positive
    - High >= Low, High >= Open/Close, Low <= Open/Close
    - Dates are sorted and unique
    - No unrealistic price moves (> 50% in one bar)

    Args:
        data: DataFrame with OHLCV data
        required_columns: Columns that must exist
        fix_issues: If True, attempt to fix issues

    Returns:
        ValidationResult with status and optionally cleaned data
    """
    if required_columns is None:
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close']

    errors = []
    warnings_list = []
    df = data.copy()

    # Check required columns
    missing = set(required_columns) - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return ValidationResult(False, errors, warnings_list)

    # Normalize column names
    col_map = {}
    for col in df.columns:
        if col.lower() == 'open':
            col_map[col] = 'Open'
        elif col.lower() == 'high':
            col_map[col] = 'High'
        elif col.lower() == 'low':
            col_map[col] = 'Low'
        elif col.lower() == 'close':
            col_map[col] = 'Close'
        elif col.lower() == 'volume':
            col_map[col] = 'Volume'
    if col_map:
        df = df.rename(columns=col_map)

    # Check for NaN values
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                if fix_issues:
                    df[col] = df[col].ffill().bfill()
                    warnings_list.append(f"Fixed {nan_count} NaN values in {col}")
                else:
                    errors.append(f"Found {nan_count} NaN values in {col}")

    # Check for inf values
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                errors.append(f"Found {inf_count} infinite values in {col}")

    # Check prices are positive
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            neg_count = (df[col] <= 0).sum()
            if neg_count > 0:
                errors.append(f"Found {neg_count} non-positive values in {col}")

    # Check OHLC consistency
    if all(c in df.columns for c in ['Open', 'High', 'Low', 'Close']):
        # High should be >= all others
        high_violations = ((df['High'] < df['Open']) |
                          (df['High'] < df['Close']) |
                          (df['High'] < df['Low'])).sum()
        if high_violations > 0:
            if fix_issues:
                df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
                warnings_list.append(f"Fixed {high_violations} High violations")
            else:
                errors.append(f"Found {high_violations} rows where High < other prices")

        # Low should be <= all others
        low_violations = ((df['Low'] > df['Open']) |
                         (df['Low'] > df['Close']) |
                         (df['Low'] > df['High'])).sum()
        if low_violations > 0:
            if fix_issues:
                df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)
                warnings_list.append(f"Fixed {low_violations} Low violations")
            else:
                errors.append(f"Found {low_violations} rows where Low > other prices")

    # Check for unrealistic price moves (> 50% in one bar)
    if 'Close' in df.columns:
        pct_change = df['Close'].pct_change().abs()
        extreme_moves = (pct_change > 0.5).sum()
        if extreme_moves > 0:
            warnings_list.append(f"Found {extreme_moves} bars with >50% price change")

    # Check date sorting
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        if not df['Date'].is_monotonic_increasing:
            if fix_issues:
                df = df.sort_values('Date').reset_index(drop=True)
                warnings_list.append("Sorted data by Date")
            else:
                errors.append("Dates are not sorted in ascending order")

        # Check for duplicates
        dup_count = df['Date'].duplicated().sum()
        if dup_count > 0:
            if fix_issues:
                df = df.drop_duplicates(subset='Date', keep='last')
                warnings_list.append(f"Removed {dup_count} duplicate dates")
            else:
                errors.append(f"Found {dup_count} duplicate dates")

    is_valid = len(errors) == 0
    return ValidationResult(is_valid, errors, warnings_list, df if fix_issues else None)
