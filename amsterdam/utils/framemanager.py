"""Thread-safe singleton registry for pandas DataFrames."""

from __future__ import annotations

import threading
from collections.abc import Iterator, MutableMapping
from enum import Enum

import pandas as pd


class AddStrategy(str, Enum):
    """How to handle a key collision when adding a DataFrame."""

    OVERWRITE = "overwrite"
    APPEND = "append"
    IGNORE = "ignore"
    RAISE = "raise"
    UNIQUE = "unique"


class DataFrameManager(MutableMapping[str, pd.DataFrame]):
    """
    Thread-safe in-memory registry of pandas DataFrames keyed by string.

    Implemented as a process-wide singleton: every call to DataFrameManager()
    returns the same instance with shared state. Supports dict-like access
    (manager["key"], "key" in manager, len(manager), iteration).
    """

    _instance: DataFrameManager | None = None
    _singleton_lock = threading.Lock()

    def __new__(cls) -> DataFrameManager:
        # Double-checked locking: avoid taking the lock on the hot path
        # once the instance has been created.
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._init_state()
                    cls._instance = instance
        return cls._instance

    def _init_state(self) -> None:
        """One-time state setup. Called from __new__, never from __init__."""
        self._dataframes: dict[str, pd.DataFrame] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add(
        self,
        key: str,
        dataframe: pd.DataFrame,
        strategy: AddStrategy | str = AddStrategy.OVERWRITE,
    ) -> str:
        """
        Add a DataFrame, returning the key it was actually stored under.

        The returned key matters for AddStrategy.UNIQUE, which may suffix
        the key to avoid collisions.
        """
        strategy = AddStrategy(strategy)

        with self._lock:
            if key not in self._dataframes:
                self._dataframes[key] = dataframe
                return key

            if strategy is AddStrategy.OVERWRITE:
                self._dataframes[key] = dataframe
                return key
            if strategy is AddStrategy.APPEND:
                self._dataframes[key] = pd.concat(
                    [self._dataframes[key], dataframe], ignore_index=True
                )
                return key
            if strategy is AddStrategy.IGNORE:
                return key
            if strategy is AddStrategy.RAISE:
                raise KeyError(f"Key '{key}' already exists.")
            if strategy is AddStrategy.UNIQUE:
                suffix = 1
                new_key = f"{key}_{suffix}"
                while new_key in self._dataframes:
                    suffix += 1
                    new_key = f"{key}_{suffix}"
                self._dataframes[new_key] = dataframe
                return new_key

            # Unreachable — Enum coercion above would have raised.
            raise ValueError(f"Unhandled strategy: {strategy}")

    def get(self, key: str, default: pd.DataFrame | None = None) -> pd.DataFrame | None:
        """Retrieve a DataFrame by key, or return default if missing."""
        with self._lock:
            return self._dataframes.get(key, default)

    def describe(self, key: str) -> pd.DataFrame:
        """Return df.describe() for the stored DataFrame."""
        with self._lock:
            if key not in self._dataframes:
                raise KeyError(f"No DataFrame found with key '{key}'.")
            return self._dataframes[key].describe()

    def keys(self) -> list[str]:  # type: ignore[override]
        """Snapshot list of all keys."""
        with self._lock:
            return list(self._dataframes.keys())

    def clear(self) -> None:
        """Remove all stored DataFrames."""
        with self._lock:
            self._dataframes.clear()

    # ------------------------------------------------------------------
    # MutableMapping protocol — gives us dict-like access for free
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> pd.DataFrame:
        with self._lock:
            if key not in self._dataframes:
                raise KeyError(key)
            return self._dataframes[key]

    def __setitem__(self, key: str, value: pd.DataFrame) -> None:
        with self._lock:
            self._dataframes[key] = value

    def __delitem__(self, key: str) -> None:
        with self._lock:
            if key not in self._dataframes:
                raise KeyError(key)
            del self._dataframes[key]

    def __iter__(self) -> Iterator[str]:
        # Snapshot to avoid "dict changed size during iteration" under threading.
        with self._lock:
            return iter(list(self._dataframes.keys()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._dataframes)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._dataframes

    def __repr__(self) -> str:
        with self._lock:
            keys = list(self._dataframes.keys())
            total_rows = sum(len(df) for df in self._dataframes.values())
        return f"DataFrameManager(keys={keys}, total_rows={total_rows})"

    # ------------------------------------------------------------------
    # Test helper — only use in tests.
    # ------------------------------------------------------------------

    @classmethod
    def _reset_singleton(cls) -> None:
        """Drop the singleton instance. Intended for test isolation only."""
        with cls._singleton_lock:
            cls._instance = None