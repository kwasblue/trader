import json
from pathlib import Path
import os

class ConfigLoader:
    _config = None

    @staticmethod
    def load_config(config_filename='utils/config.json'):
        if ConfigLoader._config is None:
            raw_config_path = os.getenv("CONFIG_PATH", config_filename)

            # Resolve the path to config.json relative to the project root
            config_loader_dir = Path(__file__).resolve().parent         # .../utils
            project_root = config_loader_dir.parent                     # .../schwab_trader
            config_path = (project_root / raw_config_path).resolve()

            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found at: {config_path}")

            with config_path.open() as f:
                config_data = json.load(f)

            # ✅ FIX: resolve folders relative to project root
            base_path = project_root
            folders = config_data.get("folders", {})

            for key, rel_path in folders.items():
                folders[key] = str((base_path / rel_path).resolve())

            config_data["folders"] = folders
            config_data["app_path"] = str(project_root.resolve())

            #print(config_data)  # optional for debugging

            ConfigLoader._config = config_data

        return ConfigLoader._config
