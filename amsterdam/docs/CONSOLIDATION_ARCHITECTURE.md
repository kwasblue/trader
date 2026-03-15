# Amsterdam Trading System - Consolidation Architecture

## Overview

This document defines the canonical architecture patterns for the Amsterdam trading system after consolidation, distinguishing between current production code and legacy components scheduled for removal.

## Canonical Paths

These are the production-ready, actively maintained components:

### Entrypoints
- **`autoamsterdam.py`** - Daemon mode for autonomous trading
- **`run_trading.py`** - GUI mode for interactive trading
- **`cli/main.py`** - CLI wrapper for system operations

### Core Data & Events
- **`core/unified_data_pipeline.py`** - Canonical data pipeline (handles all market data ingestion)
- **`core/contracts/events.py`** - Canonical event definitions (TypedDict-based, 45 events, used by 21+ files)

### Backtesting & Optimization
- **`core/backtest/unified_backtest_runner.py`** - Feature-complete backtester (28KB, production-ready)
- **`core/backtest/daily_optimize_v3.py`** - Current optimizer version

### Execution
- **`core/execution/base_executor.py`** - Abstract executor interface
- **`core/execution/live_executor.py`** - Live trading executor
- **`core/execution/mock_executor.py`** - Mock executor for testing

### Runners
- **`core/runners/runner_factory.py`** - Factory pattern for runner creation (clean design, no consolidation needed)

## Bootstrap Layer (Phase 2)

**Location**: `app/bootstrap.py`

The unified initialization system used by all entrypoints:

```python
from app.bootstrap import bootstrap_app

ctx = bootstrap_app(
    mode='daemon',  # or 'gui', 'cli'
    symbols=['AAPL', 'MSFT'],
    broker='schwab',
    log_level=logging.INFO
)

# ctx provides:
# - ctx.config: TradingConfig
# - ctx.logger: logging.Logger
# - ctx.symbols: List[str]
# - ctx.mode: str
# - ctx.root_path: Path
# - ctx.metadata: dict
```

## Legacy Components (To Be Removed)

### Deprecated - Removal in v2.0.0

| Component | Status | Replacement | Production Usage |
|-----------|--------|-------------|------------------|
| `core/events/event_models.py` | ⚠️ Orphaned | `core/contracts/events.py` | **0 files** |
| `core/backtest/backtester.py` | ⚠️ Superseded | `unified_backtest_runner.py` | Legacy only |
| `core/backtest/daily_optimize.py` (v1) | ⚠️ Superseded | `daily_optimize_v3.py` | None |
| `core/backtest/daily_optimize_v2.py` | ⚠️ Superseded | `daily_optimize_v3.py` | None |
| `core/events/events.py` | ⚠️ Deprecated shim | `core/contracts/events.py` | Contains warnings |

**Actions**:
- Phase 4: Add deprecation warnings to all legacy components
- Phase 5: Remove deprecated code after one release cycle

## Initialization Patterns

### Before Consolidation (Anti-Pattern)

Each entrypoint had different initialization:

```python
# autoamsterdam.py - Direct Logger() construction
from loggers.logger import Logger
self.logger = Logger("autotrader.log", "AutoTrader").get_logger()
self.config = get_config()

# run_trading.py - init_root_logger() calls
from loggers.bootstrap import init_root_logger
init_root_logger(log_dir="logs", root_file="app.log")
logger = logging.getLogger("TradingApp")

# cli/main.py - Subprocess delegation, no unified bootstrap
```

### After Consolidation (Canonical)

All entrypoints use unified bootstrap:

```python
from app.bootstrap import bootstrap_app

ctx = bootstrap_app(mode='daemon', symbols=['AAPL'], broker='schwab')
app = Application(context=ctx)
```

## Design Principles

1. **Single Canonical Path**: One clear way to do each thing
2. **Backwards Compatibility**: Deprecation warnings before removal
3. **Explicit Over Implicit**: Document architectural decisions
4. **Lazy Loading**: Components initialized on demand via DI container
5. **Testability**: Bootstrap layer makes testing easier

## Directory Structure

```
amsterdam/
├── app/                    # NEW: Bootstrap & DI layer
│   ├── __init__.py
│   ├── bootstrap.py        # Canonical initialization
│   └── container.py        # Dependency injection
├── core/
│   ├── contracts/
│   │   └── events.py       # CANONICAL: Event definitions
│   ├── events/
│   │   ├── event_models.py # DEPRECATED: Zero usage
│   │   └── events.py       # DEPRECATED: Shim with warnings
│   ├── backtest/
│   │   ├── unified_backtest_runner.py  # CANONICAL
│   │   ├── backtester.py               # DEPRECATED
│   │   ├── daily_optimize_v3.py        # CURRENT
│   │   ├── daily_optimize_v2.py        # DEPRECATED
│   │   └── daily_optimize.py           # DEPRECATED
│   ├── execution/          # Clean design, no changes
│   └── runners/            # Clean design, no changes
├── docs/
│   ├── CONSOLIDATION_ARCHITECTURE.md  # This file
│   ├── CONSOLIDATION_MIGRATION_GUIDE.md  # Upgrade paths
│   └── CONSOLIDATION_PLAN.md          # Full implementation plan
├── autoamsterdam.py        # Daemon entrypoint
├── run_trading.py          # GUI entrypoint
└── cli/main.py            # CLI entrypoint
```

## Migration Timeline

- **Phase 1**: Documentation (Complete)
- **Phase 2**: Bootstrap layer creation
- **Phase 3**: Entrypoint migration
- **Phase 4**: Deprecation warnings
- **Phase 5**: Code removal (after 1 release cycle)

See `CONSOLIDATION_PLAN.md` for detailed implementation steps.

## Questions?

For migration guidance, see `CONSOLIDATION_MIGRATION_GUIDE.md`.
For implementation details, see `CONSOLIDATION_PLAN.md`.
