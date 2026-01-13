"""
Settings Module - Production-Ready Configuration Management

Features:
- Multi-environment support (base, env-specific, local overrides)
- JSON/YAML file loading with deep merge
- Environment variable overrides
- String interpolation (${ENV:VAR}, ${CFG:path})
- JSON Schema validation
- File watching for hot-reload
- Type-safe getters
- Thread-safe operations

Usage:
    settings = Settings(
        root="config",
        env="production",
        auto_watch=True
    )
    
    api_key = settings.get_str("api.key")
    timeout = settings.get_int("api.timeout", default=30)
"""

from __future__ import annotations

import os
import json
import re
import threading
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union, List, Set, TYPE_CHECKING
from dataclasses import dataclass

# Optional dependencies with graceful degradation
try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None
    HAS_YAML = False

try:
    from jsonschema import validate as jsonschema_validate, ValidationError as JSONSchemaValidationError
    HAS_JSONSCHEMA = True
except ImportError:
    jsonschema_validate = None
    JSONSchemaValidationError = None
    HAS_JSONSCHEMA = False

# Type checking imports (not available at runtime if not installed)
if TYPE_CHECKING:
    from watchdog.observers import Observer as WatchdogObserver
    from watchdog.events import FileSystemEventHandler as WatchdogHandler

# Runtime imports
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HAS_WATCHDOG = True
except ImportError:
    Observer = None
    FileSystemEventHandler = object
    HAS_WATCHDOG = False


# ============================================================================
# CONSTANTS
# ============================================================================

_JSON_EXTENSIONS: Set[str] = {".json"}
_YAML_EXTENSIONS: Set[str] = {".yaml", ".yml"}
_ALL_CONFIG_EXTENSIONS: Set[str] = _JSON_EXTENSIONS | _YAML_EXTENSIONS

# Regex patterns for interpolation
_ENV_PATTERN = re.compile(r"\$\{ENV:([A-Za-z_][A-Za-z0-9_]*)\|?([^}]*)\}")
_CFG_PATTERN = re.compile(r"\$\{CFG:([A-Za-z0-9_.-]+)\|?([^}]*)\}")

# Logging
logger = logging.getLogger(__name__)


# ============================================================================
# EXCEPTIONS
# ============================================================================

class SettingsError(Exception):
    """Base exception for settings-related errors"""
    pass


class RequiredKeyError(SettingsError):
    """Raised when a required configuration key is missing"""
    pass


class ValidationError(SettingsError):
    """Raised when configuration fails schema validation"""
    pass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SettingsConfig:
    """Configuration for Settings behavior"""
    
    # Merge policy for lists: 'replace', 'extend', 'unique_extend'
    list_policy: str = "replace"
    
    # Prefix for environment variable overrides (e.g., APP__FOO__BAR)
    env_prefix: str = "APP__"
    
    # Whether to include root-level config files
    include_root: bool = True
    
    # Whether to normalize all keys to lowercase
    normalize_keys: bool = True
    
    # Whether to enable file watching
    auto_watch: bool = False
    
    # File watch debounce delay (seconds)
    watch_debounce: float = 0.5


# ============================================================================
# FILE LOADING
# ============================================================================

def _load_file(path: Path) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file.
    
    Args:
        path: Path to configuration file
        
    Returns:
        Parsed configuration dictionary
        
    Raises:
        SettingsError: If file cannot be loaded or parsed
    """
    if not path.exists():
        logger.debug(f"Config file not found: {path}")
        return {}
    
    try:
        suffix = path.suffix.lower()
        
        if suffix in _JSON_EXTENSIONS:
            with path.open('r', encoding='utf-8') as f:
                data = json.load(f)
                
        elif suffix in _YAML_EXTENSIONS:
            if not HAS_YAML:
                raise SettingsError(
                    f"Cannot load {path}: pyyaml not installed. "
                    "Install with: pip install pyyaml"
                )
            with path.open('r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        else:
            logger.warning(f"Unsupported file extension: {suffix}")
            return {}
        
        return data or {}
        
    except json.JSONDecodeError as e:
        raise SettingsError(f"Invalid JSON in {path}: {e}") from e
    except yaml.YAMLError as e:
        raise SettingsError(f"Invalid YAML in {path}: {e}") from e
    except Exception as e:
        raise SettingsError(f"Failed to load {path}: {e}") from e


def _iter_config_files(dirpath: Path, recursive: bool = True) -> List[Path]:
    """
    Iterate over configuration files in a directory.
    
    Args:
        dirpath: Directory to search
        recursive: Whether to search recursively
        
    Returns:
        Sorted list of configuration file paths
    """
    if not dirpath.exists():
        return []
    
    try:
        if recursive:
            files = [
                p for p in dirpath.rglob("*")
                if p.is_file() and p.suffix.lower() in _ALL_CONFIG_EXTENSIONS
            ]
        else:
            files = [
                p for p in dirpath.iterdir()
                if p.is_file() and p.suffix.lower() in _ALL_CONFIG_EXTENSIONS
            ]
        
        # Sort deterministically: by parent directory, then filename
        return sorted(files, key=lambda p: (str(p.parent), p.name))
        
    except Exception as e:
        logger.error(f"Error scanning directory {dirpath}: {e}")
        return []


# ============================================================================
# MERGING
# ============================================================================

def _deep_merge(
    dst: Dict[str, Any],
    src: Dict[str, Any],
    list_policy: str = "replace"
) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dst: Destination dictionary (mutated in place)
        src: Source dictionary
        list_policy: How to handle list merging:
            - 'replace': Replace destination list
            - 'extend': Append to destination list
            - 'unique_extend': Extend with unique values only
            
    Returns:
        Merged dictionary (same as dst)
        
    Raises:
        ValueError: If list_policy is invalid
    """
    valid_policies = {"replace", "extend", "unique_extend"}
    if list_policy not in valid_policies:
        raise ValueError(
            f"Invalid list_policy: {list_policy}. "
            f"Must be one of: {', '.join(valid_policies)}"
        )
    
    for key, value in src.items():
        # Both are dicts: recurse
        if key in dst and isinstance(dst[key], dict) and isinstance(value, dict):
            _deep_merge(dst[key], value, list_policy)
            
        # Both are lists: apply policy
        elif key in dst and isinstance(dst[key], list) and isinstance(value, list):
            if list_policy == "replace":
                dst[key] = value
            elif list_policy == "extend":
                dst[key] = dst[key] + value
            elif list_policy == "unique_extend":
                # Deduplicate while preserving order
                seen: Set[str] = set()
                result: List[Any] = []
                
                for item in (dst[key] + value):
                    # Use JSON representation as key for complex types
                    item_key = (
                        json.dumps(item, sort_keys=True)
                        if isinstance(item, (dict, list))
                        else str(item)
                    )
                    
                    if item_key not in seen:
                        seen.add(item_key)
                        result.append(item)
                
                dst[key] = result
        else:
            # Simple override
            dst[key] = value
    
    return dst


# ============================================================================
# INTERPOLATION
# ============================================================================

def _interpolate(value: Any, getter) -> Any:
    """
    Interpolate configuration values.
    
    Supports:
        ${ENV:VAR_NAME|default}       - Environment variable with fallback
        ${CFG:path.to.key|default}    - Config reference with fallback
        
    Args:
        value: Value to interpolate
        getter: Function to get config values (path, default) -> value
        
    Returns:
        Interpolated value
    """
    if isinstance(value, str):
        # Environment variable interpolation
        def env_repl(match):
            var_name, default = match.group(1), match.group(2)
            result = os.getenv(var_name, default)
            logger.debug(f"ENV interpolation: {var_name} -> {result}")
            return result
        
        value = _ENV_PATTERN.sub(env_repl, value)
        
        # Config reference interpolation
        def cfg_repl(match):
            path, default = match.group(1), match.group(2)
            result = getter(path, None)
            resolved = str(result if result is not None else default)
            logger.debug(f"CFG interpolation: {path} -> {resolved}")
            return resolved
        
        value = _CFG_PATTERN.sub(cfg_repl, value)
        return value
        
    elif isinstance(value, list):
        return [_interpolate(v, getter) for v in value]
        
    elif isinstance(value, dict):
        return {k: _interpolate(v, getter) for k, v in value.items()}
    
    return value


# ============================================================================
# ENVIRONMENT VARIABLE OVERRIDES
# ============================================================================

def _apply_env_overrides(cfg: Dict[str, Any], prefix: str) -> None:
    """
    Apply environment variable overrides.
    
    Environment variables like APP__FOO__BAR=value will set cfg['foo']['bar'].
    Values are parsed as JSON if possible, otherwise kept as strings.
    
    Args:
        cfg: Configuration dictionary (mutated in place)
        prefix: Environment variable prefix (e.g., "APP__")
    """
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        
        # Parse path: APP__FOO__BAR -> ['foo', 'bar']
        path_parts = env_key[len(prefix):].split("__")
        
        # Navigate to parent
        current = cfg
        for part in path_parts[:-1]:
            key = part.lower()
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        
        # Set leaf value
        leaf_key = path_parts[-1].lower()
        try:
            # Try to parse as JSON (handles booleans, numbers, etc.)
            current[leaf_key] = json.loads(env_value)
            logger.debug(f"ENV override: {env_key} = {current[leaf_key]} (parsed)")
        except (json.JSONDecodeError, ValueError):
            # Keep as string
            current[leaf_key] = env_value
            logger.debug(f"ENV override: {env_key} = {env_value} (string)")


def _normalize_keys_lower(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively normalize all dictionary keys to lowercase.
    
    Args:
        data: Dictionary to normalize
        
    Returns:
        New dictionary with lowercase keys
    """
    result = {}
    for key, value in data.items():
        normalized_key = key.lower() if isinstance(key, str) else key
        
        if isinstance(value, dict):
            result[normalized_key] = _normalize_keys_lower(value)
        else:
            result[normalized_key] = value
    
    return result


# ============================================================================
# FILE WATCHING
# ============================================================================

if HAS_WATCHDOG:
    class _ConfigFileHandler(FileSystemEventHandler):
        """Handler for configuration file changes"""
        
        def __init__(self, settings: 'Settings', debounce: float = 0.5):
            self.settings = settings
            self.debounce = debounce
            self._timer: Optional[threading.Timer] = None
            self._lock = threading.Lock()
        
        def on_any_event(self, event):
            """Handle file system events"""
            if event.is_directory:
                return
            
            path = Path(event.src_path)
            if path.suffix.lower() not in _ALL_CONFIG_EXTENSIONS:
                return
            
            logger.info(f"Config file changed: {path}")
            
            # Debounce rapid changes
            with self._lock:
                if self._timer:
                    self._timer.cancel()
                
                self._timer = threading.Timer(
                    self.debounce,
                    self._reload_settings
                )
                self._timer.daemon = True
                self._timer.start()
        
        def _reload_settings(self):
            """Reload settings after debounce period"""
            try:
                self.settings.reload()
                logger.info("Settings reloaded successfully")
            except Exception as e:
                logger.error(f"Failed to reload settings: {e}")


# ============================================================================
# MAIN SETTINGS CLASS
# ============================================================================

class Settings:
    """
    Hierarchical configuration management with environment support.
    
    Loading Order (later overrides earlier):
        1. root/*.{json,yaml} (if include_root=True)
        2. base/**/*.{json,yaml}
        3. env/<environment>/**/*.{json,yaml}
        4. local/**/*.{json,yaml}
        5. Environment variables (APP__*)
        6. Runtime overrides
    
    Example:
        settings = Settings(
            root="config",
            env="production",
            auto_watch=True
        )
        
        db_host = settings.get_str("database.host")
        db_port = settings.get_int("database.port", default=5432)
    """
    
    # Declare types for type checkers only
    if TYPE_CHECKING:
        _observer: Optional[WatchdogObserver]
        _handler: Optional[WatchdogHandler]
    
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
        include_root: bool = True,
        normalize_keys: bool = True,
    ):
        """
        Initialize Settings.
        
        Args:
            root: Root directory containing configuration files
            env: Environment name (e.g., "dev", "prod")
            list_policy: How to merge lists ('replace', 'extend', 'unique_extend')
            env_prefix: Prefix for environment variable overrides
            runtime_overrides: Dictionary of runtime configuration overrides
            schema_paths: Dict mapping namespace to schema file path
            global_schema_path: Path to global JSON schema
            auto_watch: Enable automatic reloading on file changes
            include_root: Whether to load files from root directory
            normalize_keys: Whether to normalize keys to lowercase
        """
        self.root = Path(root).resolve()
        self.env = env
        self.list_policy = list_policy
        self.env_prefix = env_prefix
        self.include_root = include_root
        self.normalize_keys = normalize_keys
        
        # Schema validation
        self._schema_paths = {
            k: Path(v) for k, v in (schema_paths or {}).items()
        }
        self._global_schema_path = (
            Path(global_schema_path) if global_schema_path else None
        )
        
        # Thread safety
        self._lock = threading.RLock()
        
        # File watching - use Any at runtime to avoid type issues with Optional[None]
        self._observer: Any = None
        self._handler: Any = None
        
        # Configuration data
        self._cfg: Dict[str, Any] = {}
        self._runtime_overrides = runtime_overrides or {}
        
        # Initial load
        self._load_all()
        
        # Start file watcher
        if auto_watch:
            self._start_watch()
        
        logger.info(
            f"Settings initialized: root={self.root}, env={self.env}, "
            f"auto_watch={auto_watch}"
        )
    
    # ========================================================================
    # PUBLIC API - GETTERS
    # ========================================================================
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated path.
        
        Args:
            path: Dot-separated path (e.g., "database.host")
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        with self._lock:
            current: Any = self._cfg
            parts = path.lower().split(".") if self.normalize_keys else path.split(".")
            
            for part in parts:
                if not isinstance(current, dict) or part not in current:
                    return default
                current = current[part]
            
            return current
    
    def require(self, path: str) -> Any:
        """
        Get required configuration value.
        
        Args:
            path: Dot-separated path
            
        Returns:
            Configuration value
            
        Raises:
            RequiredKeyError: If path not found
        """
        value = self.get(path, None)
        if value is None:
            raise RequiredKeyError(f"Missing required config key: '{path}'")
        return value
    
    def get_str(self, path: str, default: Optional[str] = None) -> Optional[str]:
        """Get string value"""
        value = self.get(path, default)
        return None if value is None else str(value)
    
    def get_int(self, path: str, default: Optional[int] = None) -> Optional[int]:
        """Get integer value"""
        value = self.get(path, default)
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Cannot convert '{path}' to int: {value}")
            return default
    
    def get_float(self, path: str, default: Optional[float] = None) -> Optional[float]:
        """Get float value"""
        value = self.get(path, default)
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Cannot convert '{path}' to float: {value}")
            return default
    
    def get_bool(self, path: str, default: Optional[bool] = None) -> Optional[bool]:
        """
        Get boolean value.
        
        Accepts: True/False, "true"/"false", "yes"/"no", "1"/"0", "on"/"off"
        """
        value = self.get(path, default)
        
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False
        
        if value is None:
            return default
        
        return bool(value)
    
    def get_list(self, path: str, default: Optional[List[Any]] = None) -> Optional[List[Any]]:
        """Get list value (converts single values to list)"""
        value = self.get(path, default)
        if value is None:
            return None
        if isinstance(value, list):
            return value
        return [value]
    
    def get_dict(
        self,
        path: str,
        default: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get dictionary value"""
        value = self.get(path, default)
        return value if isinstance(value, dict) else default
    
    def resolve_path(self, path_key: str, must_exist: bool = False) -> Path:
        """
        Resolve a path value relative to config root.
        
        Args:
            path_key: Configuration key containing path
            must_exist: Whether path must exist
            
        Returns:
            Resolved absolute path
            
        Raises:
            RequiredKeyError: If path_key not found
            FileNotFoundError: If must_exist=True and path doesn't exist
        """
        value = self.require(path_key)
        path = Path(str(value))
        
        if not path.is_absolute():
            path = (self.root / path).resolve()
        
        if must_exist and not path.exists():
            raise FileNotFoundError(f"{path_key} -> {path} not found")
        
        return path
    
    def as_dict(self, deep_copy: bool = True) -> Dict[str, Any]:
        """
        Get configuration as dictionary.
        
        Args:
            deep_copy: Whether to return a deep copy
            
        Returns:
            Configuration dictionary
        """
        with self._lock:
            if not deep_copy:
                return dict(self._cfg)
            
            # Deep copy via JSON round-trip (safe and simple)
            return json.loads(json.dumps(self._cfg, default=str))
    
    # ========================================================================
    # PUBLIC API - CONTROL
    # ========================================================================
    
    def reload(self) -> None:
        """Reload configuration from files"""
        with self._lock:
            logger.info("Reloading settings...")
            self._load_all()
    
    def stop_watching(self) -> None:
        """Stop file watching"""
        if self._observer:
            logger.info("Stopping file watcher...")
            self._observer.stop()
            self._observer.join(timeout=2.0)
            self._observer = None
            self._handler = None
    
    def __del__(self):
        """Cleanup on deletion"""
        self.stop_watching()
    
    # ========================================================================
    # INTERNAL - LOADING
    # ========================================================================
    
    def _load_all(self) -> None:
        """Load and merge all configuration sources"""
        merged: Dict[str, Any] = {}
        
        try:
            # 1. Root-level files (if enabled)
            if self.include_root:
                for file_path in _iter_config_files(self.root, recursive=False):
                    data = _load_file(file_path)
                    merged = _deep_merge(merged, data, self.list_policy)
            
            # 2. base/ directory
            self._merge_directory(merged, "base")
            
            # 3. env/<environment>/ directory
            if self.env:
                self._merge_directory(merged, f"env/{self.env}")
            
            # 4. local/ directory (for local overrides, should be .gitignored)
            self._merge_directory(merged, "local")
            
            # 5. Normalize keys if enabled
            if self.normalize_keys:
                merged = _normalize_keys_lower(merged)
            
            # 6. Environment variable overrides
            _apply_env_overrides(merged, self.env_prefix)
            
            # 7. Runtime overrides
            if self._runtime_overrides:
                runtime = (
                    _normalize_keys_lower(self._runtime_overrides)
                    if self.normalize_keys
                    else self._runtime_overrides
                )
                merged = _deep_merge(merged, runtime, self.list_policy)
            
            # 8. Interpolation
            merged = _interpolate(merged, self._getter_for_interpolation(merged))
            
            # 9. Validation
            self._validate(merged)
            
            # Store
            self._cfg = merged
            
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _merge_directory(self, merged: Dict[str, Any], subdir: str) -> None:
        """Merge all files from a subdirectory"""
        dir_path = self.root / subdir
        for file_path in _iter_config_files(dir_path, recursive=True):
            data = _load_file(file_path)
            merged = _deep_merge(merged, data, self.list_policy)
    
    def _getter_for_interpolation(self, cfg: Dict[str, Any]):
        """Create getter function for interpolation"""
        def getter(path: str, default=None):
            current: Any = cfg
            parts = path.lower().split(".") if self.normalize_keys else path.split(".")
            
            for part in parts:
                if not isinstance(current, dict) or part not in current:
                    return default
                current = current[part]
            
            return current
        
        return getter
    
    def _validate(self, cfg: Dict[str, Any]) -> None:
        """Validate configuration against JSON schemas"""
        if not HAS_JSONSCHEMA:
            if self._schema_paths or self._global_schema_path:
                logger.warning(
                    "JSON Schema validation requested but jsonschema not installed. "
                    "Install with: pip install jsonschema"
                )
            return
        
        try:
            # Validate namespaces
            for namespace, schema_path in self._schema_paths.items():
                node = self._pluck(cfg, namespace.lower() if self.normalize_keys else namespace)
                schema = _load_file(schema_path)
                jsonschema_validate(node, schema)
                logger.debug(f"Validated namespace: {namespace}")
            
            # Validate global
            if self._global_schema_path:
                schema = _load_file(self._global_schema_path)
                jsonschema_validate(cfg, schema)
                logger.debug("Validated global schema")
                
        except Exception as e:
            if JSONSchemaValidationError and isinstance(e, JSONSchemaValidationError):
                raise ValidationError(f"Schema validation failed: {e}") from e
            raise
    
    def _pluck(self, cfg: Dict[str, Any], path: str) -> Any:
        """Extract nested value by path"""
        current: Any = cfg
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                return {}
            current = current[part]
        return current
    
    # ========================================================================
    # INTERNAL - FILE WATCHING
    # ========================================================================
    
    def _start_watch(self) -> None:
        """Start watching configuration files for changes"""
        if not HAS_WATCHDOG:
            logger.warning(
                "File watching requested but watchdog not installed. "
                "Install with: pip install watchdog"
            )
            return
        
        if self._observer:
            logger.warning("File watcher already running")
            return
        
        try:
            self._handler = _ConfigFileHandler(self)
            self._observer = Observer()
            self._observer.schedule(
                self._handler,
                str(self.root),
                recursive=True
            )
            self._observer.daemon = True
            self._observer.start()
            
            logger.info(f"File watcher started for: {self.root}")
            
        except Exception as e:
            logger.error(f"Failed to start file watcher: {e}")
            self._observer = None
            self._handler = None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_global_settings: Optional[Settings] = None


def initialize_settings(**kwargs) -> Settings:
    """
    Initialize global settings instance.
    
    Args:
        **kwargs: Arguments passed to Settings constructor
        
    Returns:
        Settings instance
    """
    global _global_settings
    _global_settings = Settings(**kwargs)
    return _global_settings


def get_settings() -> Settings:
    """
    Get global settings instance.
    
    Returns:
        Settings instance
        
    Raises:
        RuntimeError: If settings not initialized
    """
    if _global_settings is None:
        raise RuntimeError(
            "Settings not initialized. Call initialize_settings() first."
        )
    return _global_settings