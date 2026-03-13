import threading
import pandas as pd

import threading

class DataFrameManager:
    _instance = None
    _lock = threading.Lock()  # Lock for thread safety

    def __new__(cls, *args, **kwargs):
        # Implementing Singleton pattern
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DataFrameManager, cls).__new__(cls, *args, **kwargs)
                cls._instance.dataframes = {}
        return cls._instance

    def __init__(self):
        """Initializes only once for the singleton instance"""
        if not hasattr(self, '_initialized'):  # Prevent re-initialization
            self._initialized = True
            self.dataframes = {}

    def add_dataframe(self, key, dataframe, strategy="overwrite"):
        """
        Add a DataFrame to the manager, handling duplicates based on the chosen strategy.
        """
        with self._lock:
            if key in self.dataframes:
                if strategy == "overwrite":
                    self.dataframes[key] = dataframe
                elif strategy == "append":
                    self.dataframes[key] = pd.concat([self.dataframes[key], dataframe], ignore_index=True)
                elif strategy == "ignore":
                    pass  # Keep the existing DataFrame
                elif strategy == "raise":
                    raise KeyError(f"Key '{key}' already exists in the DataFrameManager.")
                elif strategy == "unique":
                    new_key = f"{key}_{len(self.dataframes)}"
                    self.dataframes[new_key] = dataframe
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
            else:
                self.dataframes[key] = dataframe

    def get_dataframe(self, key: str):
        """Retrieve a DataFrame by its key."""
        with self._lock:
            if key not in self.dataframes:
                raise KeyError(f"No DataFrame found with key '{key}'.")
            return pd.DataFrame.from_dict(self.dataframes[key])

    def remove_dataframe(self, key: str):
        """Remove a DataFrame by its key."""
        with self._lock:
            if key in self.dataframes:
                del self.dataframes[key]
            else:
                raise KeyError(f"No DataFrame found with key '{key}'.")

    def list_keys(self) -> list:
        """List all the keys of stored DataFrames."""
        with self._lock:
            return list(self.dataframes.keys())

    def clear_all(self):
        """Remove all DataFrames from the manager."""
        with self._lock:
            self.dataframes.clear()

    def size(self) -> int:
        """Get the number of DataFrames currently stored."""
        with self._lock:
            return len(self.dataframes)

    def has_key(self, key: str) -> bool:
        """Check if a DataFrame exists for a given key."""
        with self._lock:
            return key in self.dataframes

    def describe(self, key: str) -> str:
        """Get a summary of a stored DataFrame."""
        with self._lock:
            if key not in self.dataframes:
                raise KeyError(f"No DataFrame found with key '{key}'.")
            return self.dataframes[key].describe()
