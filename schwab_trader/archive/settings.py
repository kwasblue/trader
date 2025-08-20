import json, os
from jsonschema import validate  # optional; or skip if you don’t want dependency

class Settings:
    def __init__(self, config_path: str, schema_path: str | None = None):
        with open(config_path) as f:
            self._cfg = json.load(f)
        if schema_path:
            with open(schema_path) as f:
                schema = json.load(f)
            validate(self._cfg, schema)

    def get(self, path: str, default=None):
        cur = self._cfg
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur: return default
            cur = cur[part]
        return cur
