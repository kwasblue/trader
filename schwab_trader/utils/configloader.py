import json
from pathlib import Path

class ConfigLoader:
    _config = None  # Class-level cache

    @staticmethod
    def load_config():
        if ConfigLoader._config is None:  # Load only once
            config_path = Path(r'C:\Users\kwasi\OneDrive\Documents\Personal Projects\schwab_trader\config\config2.json')
            with config_path.open() as f:
                ConfigLoader._config = json.load(f)
        return ConfigLoader._config
