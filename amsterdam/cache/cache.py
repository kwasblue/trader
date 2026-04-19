from pathlib import Path
import json
from datetime import datetime
from typing import Optional, Dict, Any
from core.config_loader import get_app_path


class CacheManager:
    _instance = None

    def __new__(cls, cache_file: str = "cache/system_cache.json"):
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, cache_file: str = "cache/system_cache.json"):
        if self._initialized:
            return
        app_path = get_app_path()

        self.cache_path = (app_path / cache_file).resolve()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_data = self._load_cache()

        CacheManager._initialized = True

    def _load_cache(self) -> Dict[str, Any]:
        if self.cache_path.exists():
            with self.cache_path.open("r") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with self.cache_path.open("w") as f:
            json.dump(self.cache_data, f, indent=4)

    def get(self, section: str, key: Optional[str] = None) -> Any:
        section_data = self.cache_data.get(section, {})
        if key:
            return section_data.get(key)
        return section_data
    
    def get_last_processed_date(self, section, stock: str):
        return self.get(section, stock)

    def update(self, section: str, key: str, value: Any):
        # Create a copy of the whole cache to avoid mutation during iteration
        cache_copy = self.cache_data.copy()

        # Create or update the section
        section_data = cache_copy.get(section, {}).copy()
        section_data[key] = value
        cache_copy[section] = section_data

        # Replace the internal cache with the safe copy
        self.cache_data = cache_copy

        # Save to disk
        self._save_cache()

    def delete(self, section: str, key: Optional[str] = None):
        if key:
            self.cache_data.get(section, {}).pop(key, None)
        else:
            self.cache_data.pop(section, None)
        self._save_cache()

    def clear_all(self):
        self.cache_data = {}
        self._save_cache()




