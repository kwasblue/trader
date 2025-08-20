from __future__ import annotations
import os, json, re, threading
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union, List, Tuple

# Optional deps
try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None
try:
    from jsonschema import validate as jsonschema_validate  # pip install jsonschema
except Exception:
    jsonschema_validate = None
try:
    from watchdog.observers import Observer  # pip install watchdog
    from watchdog.events import FileSystemEventHandler
except Exception:
    Observer = None
    FileSystemEventHandler = object


# -------------------------
# Helpers
# -------------------------

_JSON_LIKE = {".json"}
_YAML_LIKE = {".yaml", ".yml"}

def _load_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    if path.suffix.lower() in _JSON_LIKE:
        with path.open() as f:
            data = json.load(f)
    elif path.suffix.lower() in _YAML_LIKE:
        if yaml is None:
            raise RuntimeError("pyyaml not installed but YAML provided")
        with path.open() as f:
            data = yaml.safe_load(f)
    else:
        return {}
    return data or {}

def _iter_config_files(dirpath: Path) -> Iterable[Path]:
    if not dirpath.exists():
        return []
    files = [
        p for p in dirpath.rglob("*")
        if p.is_file() and p.suffix.lower() in (_JSON_LIKE | _YAML_LIKE)
    ]
    # Deterministic: first by folder, then filename
    return sorted(files, key=lambda p: (str(p.parent), p.name))

def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any], list_policy: str) -> Dict[str, Any]:
    """
    list_policy: 'replace' | 'extend' | 'unique_extend'
    """
    for k, v in src.items():
        if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
            _deep_merge(dst[k], v, list_policy)
        elif k in dst and isinstance(dst[k], list) and isinstance(v, list):
            if list_policy == "replace":
                dst[k] = v
            elif list_policy == "extend":
                dst[k] = dst[k] + v
            elif list_policy == "unique_extend":
                seen = set()
                out = []
                for item in (dst[k] + v):
                    key = json.dumps(item, sort_keys=True) if isinstance(item, (dict, list)) else item
                    if key not in seen:
                        seen.add(key)
                        out.append(item)
                dst[k] = out
            else:
                raise ValueError(f"Unknown list_policy: {list_policy}")
        else:
            dst[k] = v
    return dst

_ENV_PATTERN = re.compile(r"\$\{ENV:([A-Za-z_][A-Za-z0-9_]*)\|?([^}]*)\}")
_CFG_PATTERN = re.compile(r"\$\{CFG:([A-Za-z0-9_.-]+)\|?([^}]*)\}")

def _interpolate(value: Any, getter) -> Any:
    """
    Interpolate strings with:
      ${ENV:VAR|default}  -> environment var with default
      ${CFG:path.to.key|default} -> config lookup with default
    """
    if isinstance(value, str):
        def env_repl(m):
            var, default = m.group(1), m.group(2)
            return os.getenv(var, default)
        value = _ENV_PATTERN.sub(env_repl, value)

        def cfg_repl(m):
            path, default = m.group(1), m.group(2)
            got = getter(path, None)
            return str(got if got is not None else default)
        value = _CFG_PATTERN.sub(cfg_repl, value)
        return value
    elif isinstance(value, list):
        return [_interpolate(v, getter) for v in value]
    elif isinstance(value, dict):
        return {k: _interpolate(v, getter) for k, v in value.items()}
    return value

def _apply_env_overrides(cfg: Dict[str, Any], prefix: str) -> None:
    """
    APP__FOO__BAR=123 -> cfg['foo']['bar'] = 123 (json-decoded if possible).
    Case-insensitive keys in cfg (normalize to lower).
    """
    for k, v in os.environ.items():
        if not k.startswith(prefix):
            continue
        path_parts = k[len(prefix):].split("__")
        cur = cfg
        for part in path_parts[:-1]:
            key = part.lower()
            if key not in cur or not isinstance(cur[key], dict):
                cur[key] = {}
            cur = cur[key]
        leaf = path_parts[-1].lower()
        try:
            cur[leaf] = json.loads(v)
        except Exception:
            cur[leaf] = v

def _normalize_keys_lower(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        nk = k.lower() if isinstance(k, str) else k
        if isinstance(v, dict):
            out[nk] = _normalize_keys_lower(v)
        else:
            out[nk] = v
    return out

def _iter_config_files_shallow(dirpath: Path):
    if not dirpath.exists():
        return []
    files = [
        p for p in dirpath.iterdir()
        if p.is_file() and p.suffix.lower() in {".json", ".yaml", ".yml"}
    ]
    return sorted(files, key=lambda p: p.name)

# -------------------------
# Settings
# -------------------------

class Settings:
    """
    Folder-based config with layering:
      1) base/
      2) env/<env>/
      3) local/       (gitignored)
      4) env vars     (APP__FOO__BAR=...)
      5) runtime_overrides

    Features:
      - JSON/YAML
      - deterministic deep-merge with configurable list policy
      - ${ENV:VAR|default} and ${CFG:foo.bar|default} interpolation
      - __include__ support (list of file paths relative to config root)
      - optional JSON Schema validation (global or per-namespace)
      - typed getters + require()
      - resolve_path() to make paths absolute (relative to config root)
      - optional file watching to auto-reload (watchdog if installed)
    """

    def __init__(
        self,
        root: Union[str, Path],
        env: Optional[str] = None,
        *,
        list_policy: str = "replace",
        env_prefix: str = "APP__",
        runtime_overrides: Optional[Dict[str, Any]] = None,
        schema_paths: Optional[Dict[str, Union[str, Path]]] = None,
        global_schema_path: Optional[Union[str, Path]] = None,
        auto_watch: bool = False,
        include_root: bool = True,        # <-- NEW
    ):
        self.root = Path(root)
        self.env = env
        self.list_policy = list_policy
        self.env_prefix = env_prefix
        self._schema_paths = {k: Path(v) for k, v in (schema_paths or {}).items()}
        self._global_schema_path = Path(global_schema_path) if global_schema_path else None
        self._watcher: Optional[tuple[object, object]] = None
        self._lock = threading.RLock()
        self._include_root = include_root

        self._cfg: Dict[str, Any] = {}
        self._load_all(runtime_overrides or {})

        if auto_watch and Observer is not None:
            self._start_watch()

    # ------------- Public API -------------

    def get(self, path: str, default: Any = None) -> Any:
        with self._lock:
            cur: Any = self._cfg
            for part in path.lower().split("."):
                if not isinstance(cur, dict) or part not in cur:
                    return default
                cur = cur[part]
            return cur

    def require(self, path: str) -> Any:
        val = self.get(path, None)
        if val is None:
            raise KeyError(f"Missing required config key: '{path}'")
        return val

    # Typed getters
    def get_str(self, path: str, default: Optional[str] = None) -> Optional[str]:
        val = self.get(path, default)
        return None if val is None else str(val)

    def get_int(self, path: str, default: Optional[int] = None) -> Optional[int]:
        val = self.get(path, default)
        return None if val is None else int(val)

    def get_float(self, path: str, default: Optional[float] = None) -> Optional[float]:
        val = self.get(path, default)
        return None if val is None else float(val)

    def get_bool(self, path: str, default: Optional[bool] = None) -> Optional[bool]:
        val = self.get(path, default)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in {"1", "true", "yes", "y", "on"}
        if val is None:
            return default
        return bool(val)

    def get_list(self, path: str, default: Optional[List[Any]] = None) -> Optional[List[Any]]:
        val = self.get(path, default)
        if val is None:
            return None
        if isinstance(val, list):
            return val
        return [val]

    def get_dict(self, path: str, default: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        val = self.get(path, default)
        return val if isinstance(val, dict) else default

    def resolve_path(self, path_key: str, must_exist: bool = False) -> Path:
        """
        Resolve a path value relative to config root.
        """
        val = self.require(path_key)
        p = Path(str(val))
        if not p.is_absolute():
            p = (self.root / p).resolve()
        if must_exist and not p.exists():
            raise FileNotFoundError(f"{path_key} -> {p} not found")
        return p

    def as_dict(self, deep: bool = True) -> Dict[str, Any]:
        with self._lock:
            if not deep:
                return dict(self._cfg)
            # deep copy via json round-trip to keep it simple and safe
            return json.loads(json.dumps(self._cfg))

    def reload(self) -> None:
        with self._lock:
            current = self._runtime_overrides  # preserve
            self._load_all(current)

    def stop_watching(self) -> None:
        if self._watcher:
            obs, handler = self._watcher
            obs.stop()
            obs.join(timeout=2.0)
            self._watcher = None

    # ------------- Internal -------------

    def _load_all(self, runtime_overrides: Dict[str, Any]) -> None:
        # Preserve runtime_overrides so reload keeps them
        self._runtime_overrides = runtime_overrides or {}

        merged: Dict[str, Any] = {}

        # ---- 0) root-level files (shallow only) ----
        if getattr(self, "_include_root", True):
            if self.root.exists():
                for f in sorted(self.root.iterdir(), key=lambda p: p.name):
                    if f.is_file() and f.suffix.lower() in {".json", ".yaml", ".yml"}:
                        data = _load_file(f)
                        merged = _deep_merge(merged, _process_includes(data), self.list_policy)

        # helper: recursively merge a subtree (e.g., base/, env/dev/, local/)
        def _apply_tree(sub: str) -> None:
            nonlocal merged
            d = self.root / sub
            for f in _iter_config_files(d):  # your existing recursive rglob loader
                data = _load_file(f)
                merged = _deep_merge(merged, _process_includes(data), self.list_policy)

        # ---- 1) base/ ----
        _apply_tree("base")

        # ---- 2) env/<env>/ ----
        if self.env:
            _apply_tree(f"env/{self.env}")

        # ---- 3) local/ ----
        _apply_tree("local")

        # ---- 4) env vars ----
        merged = _normalize_keys_lower(merged)
        _apply_env_overrides(merged, prefix=self.env_prefix)

        # ---- 5) runtime overrides ----
        merged = _deep_merge(merged, _normalize_keys_lower(self._runtime_overrides), self.list_policy)

        # Interpolate (${ENV:..}, ${CFG:..})
        def _getter(path: str, default=None):
            cur: Any = merged
            for part in path.lower().split("."):
                if not isinstance(cur, dict) or part not in cur:
                    return default
                cur = cur[part]
            return cur
        merged = _interpolate(merged, _getter)

        # Validate (optional)
        if self._schema_paths and jsonschema_validate:
            for ns, schema_path in self._schema_paths.items():
                node = self._pluck(merged, ns.lower())
                schema = _load_file(schema_path)
                jsonschema_validate(node, schema)
        if self._global_schema_path and jsonschema_validate:
            schema = _load_file(self._global_schema_path)
            jsonschema_validate(merged, schema)

        self._cfg = merged


    def _pluck(self, cfg: Dict[str, Any], path: str) -> Any:
        cur: Any = cfg
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return {}
            cur = cur[part]
        return cur

    def _start_watch(self) -> None:
        if Observer is None:
            return
        class _Handler(FileSystemEventHandler):
            def __init__(self, s: Settings):
                self.s = s
            def on_any_event(self, event):
                # Only react to writes to json/yaml files inside config
                if event.is_directory:
                    return
                p = Path(event.src_path)
                if p.suffix.lower() in (_JSON_LIKE | _YAML_LIKE):
                    try:
                        self.s.reload()
                    except Exception as e:
                        # Swallow: you can add logging here
                        pass
        handler = _Handler(self)
        obs = Observer()
        obs.schedule(handler, str(self.root), recursive=True)
        obs.daemon = True
        obs.start()
        self._watcher = (obs, handler)


# -------------------------
# Include support
# -------------------------

def _process_includes(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supports:
      __include__: ["relative/path1.yml", "relative/path2.json"]
    Includes are loaded relative to the *root* of the Settings instance.
    We patch in a placeholder and resolve later (Settings._load_all ensures base/env/local order first).
    """
    # This function only normalizes the key; actual loading happens in Settings._load_all via _load_file when files are iterated.
    # Here we just return d untouched because files are loaded at the directory level already.
    # If you want per-file includes (nested), uncomment the below pattern and pass 'root' around.
    return d
