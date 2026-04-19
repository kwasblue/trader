# Amsterdam Trading System - Consolidation Plan

## Executive Summary

**Goal**: Eliminate architectural fragmentation by consolidating overlapping patterns, deprecating unused code, and creating a single canonical application assembly path.

**Approach**: Incremental, 5-phase plan prioritizing documentation → consolidation → cleanup

**Timeline**: 3-4 weeks | **Risk**: Low (backwards compatible, testable phases)

---

## Key Findings from Exploration

### ✅ What's Already Good
- **Runners**: Clean factory pattern (no consolidation needed)
- **Data Pipeline**: `core/unified_data_pipeline.py` is canonical (no issues)
- **Executors**: Clear base → Live/Mock inheritance (well-designed)

### ⚠️ Needs Consolidation

**1. Multiple Entrypoints with Different Init Patterns**
- `autoamsterdam.py` - Uses `Logger()` directly, complex autonomous logic
- `run_trading.py` - Uses `init_root_logger()`, GUI initialization
- `cli/main.py` - Delegates via subprocess, no unified bootstrap
- **Issue**: Each has different config loading, logging setup, dependency wiring

**2. Event Schema Duplication**
- `core/contracts/events.py` - CANONICAL (45 events, 21 files use it)
- `core/events/event_models.py` - ORPHANED (6 Pydantic models, ZERO production usage)
- `core/events/events.py` - DEPRECATED shim (already has warnings)
- **Action**: Remove orphaned event_models.py

**3. Legacy Optimizer Versions**
- `daily_optimize_v3.py` - CURRENT
- `daily_optimize.py`, `daily_optimize_v2.py` - SUPERSEDED
- **Action**: Deprecate v1/v2, eventually rename v3 → canonical

**4. Legacy Backtester**
- `unified_backtest_runner.py` - CANONICAL (28KB, feature-complete)
- `backtester.py` - LEGACY (10KB, superseded)
- **Action**: Deprecate old backtester

---

## Implementation Plan

### Phase 1: Documentation & Clarity ✅ COMPLETE

**Goal**: Make implicit canonical choices explicit

#### Deliverables

**1.1 Create `docs/CONSOLIDATION_ARCHITECTURE.md`**
Document canonical vs legacy paths

**1.2 Create `docs/CONSOLIDATION_MIGRATION_GUIDE.md`**
Provide upgrade paths for deprecated imports

**1.3 Create `docs/CONSOLIDATION_PLAN.md`**
This document - full implementation plan

#### Verification
```bash
cat docs/CONSOLIDATION_ARCHITECTURE.md
cat docs/CONSOLIDATION_MIGRATION_GUIDE.md
cat docs/CONSOLIDATION_PLAN.md
pytest tests/  # No functional changes
```

---

### Phase 2: Create Unified Bootstrap Layer

**Goal**: Centralize initialization without breaking existing code

#### Deliverables

**2.1 Create `app/__init__.py`**
```python
from app.bootstrap import bootstrap_app, AppContext
from app.container import AppContainer

__all__ = ['bootstrap_app', 'AppContext', 'AppContainer']
```

**2.2 Create `app/bootstrap.py`** (CORE COMPONENT)
```python
"""Unified application bootstrap for all entrypoints"""

from dataclasses import dataclass
from typing import List, Literal, Optional
import logging
from pathlib import Path
from dotenv import load_dotenv
from loggers.bootstrap import init_root_logger
from core.config_loader import get_config, TradingConfig

ROOT = Path(__file__).resolve().parents[1]

@dataclass
class AppContext:
    """Application context returned by bootstrap"""
    config: TradingConfig
    logger: logging.Logger
    mode: str  # daemon/gui/cli
    symbols: List[str]
    root_path: Path
    metadata: dict  # broker, trading_mode, operation

def bootstrap_app(
    mode: Literal['daemon', 'gui', 'cli'],
    symbols: Optional[List[str]] = None,
    broker: Optional[str] = None,
    trading_mode: Optional[str] = None,
    operation: Optional[str] = None,
    log_level: int = logging.INFO,
    console_logging: bool = True,
) -> AppContext:
    """
    Canonical initialization path for all Amsterdam entrypoints.

    Returns AppContext with config, logger, symbols, metadata.
    """
    # 1. Load env
    load_dotenv(ROOT / ".venv" / ".env")
    load_dotenv()

    # 2. Init unified logger
    log_files = {'daemon': 'autoamsterdam.log', 'gui': 'app.log', 'cli': 'cli.log'}
    init_root_logger(
        log_dir=str(ROOT / "logs"),
        root_file=log_files[mode],
        level=log_level,
        console=console_logging,
    )
    logger = logging.getLogger(f"Amsterdam.{mode.capitalize()}")

    # 3. Load config
    config = get_config()

    # 4. Resolve symbols
    symbols = symbols or config.general.default_symbols or []

    # 5. Log initialization
    logger.info(f"AMSTERDAM {mode.upper()} MODE INITIALIZED")
    logger.info(f"Symbols: {symbols}")

    # 6. Build context
    metadata = {}
    if broker: metadata['broker'] = broker
    if trading_mode: metadata['trading_mode'] = trading_mode
    if operation: metadata['operation'] = operation

    return AppContext(
        config=config,
        logger=logger,
        mode=mode,
        symbols=symbols,
        root_path=ROOT,
        metadata=metadata,
    )
```

**2.3 Create `app/container.py`** (Dependency Injection)
```python
"""DI container for lazy component initialization"""

from typing import Optional
from app.bootstrap import AppContext
from core.runners.base_runner import BaseLiveRunner
from core.runners.runner_factory import RunnerFactory
from core.events.eventhandler import get_event_handler, EventHandler

class AppContainer:
    """Dependency injection container for lazy component initialization"""

    def __init__(self, context: AppContext):
        self.ctx = context
        self._runner: Optional[BaseLiveRunner] = None
        self._event_handler: Optional[EventHandler] = None

    def get_runner(self) -> BaseLiveRunner:
        """Lazy-load runner from factory"""
        if self._runner is None:
            broker = self.ctx.metadata.get('broker', 'schwab')
            self._runner = RunnerFactory.create(broker, self.ctx.symbols, self.ctx.config)
        return self._runner

    def get_event_handler(self) -> EventHandler:
        """Get singleton event handler"""
        if self._event_handler is None:
            self._event_handler = get_event_handler()
        return self._event_handler

    def cleanup(self):
        """Cleanup resources"""
        if self._runner is not None:
            if hasattr(self._runner, 'cleanup'):
                self._runner.cleanup()
        if self._event_handler is not None:
            if hasattr(self._event_handler, 'cleanup'):
                self._event_handler.cleanup()
```

#### Critical Files
- `/Users/kwasiaddo/projects/trader/amsterdam/app/__init__.py`
- `/Users/kwasiaddo/projects/trader/amsterdam/app/bootstrap.py` ⭐ CORE
- `/Users/kwasiaddo/projects/trader/amsterdam/app/container.py`

#### Verification
```bash
# Unit tests
pytest tests/app/test_bootstrap.py -v

# Integration test
python -c "from app.bootstrap import bootstrap_app; ctx = bootstrap_app('cli', symbols=['AAPL']); print('✓ Works')"

# Existing entrypoints still work
python autoamsterdam.py --dry-run --symbols AAPL &
sleep 5 && pkill -f autoamsterdam
```

---

### Phase 3: Migrate Entrypoints

**Goal**: Refactor entrypoints to use canonical bootstrap

#### 3.1 Migrate `autoamsterdam.py`

Refactor to use `bootstrap_app()` instead of direct Logger() calls.

#### 3.2 Migrate `run_trading.py`

Refactor to use `bootstrap_app()` instead of direct init_root_logger() calls.

#### 3.3 Update `cli/main.py`

Add bootstrap for operations that need initialization.

#### Critical Files
- `/Users/kwasiaddo/projects/trader/amsterdam/autoamsterdam.py`
- `/Users/kwasiaddo/projects/trader/amsterdam/run_trading.py`
- `/Users/kwasiaddo/projects/trader/amsterdam/cli/main.py`

#### Verification
```bash
# Test migrated entrypoints
python autoamsterdam.py --dry-run --symbols AAPL
python run_trading.py --mode simulation --symbols AAPL
amsterdam preflight -v

# Full test suite
pytest tests/ -v

# No import errors
python -c "import autoamsterdam; import run_trading; import cli.main"
```

---

### Phase 4: Deprecate Legacy

**Goal**: Add deprecation warnings without removing code

#### 4.1 Deprecate `core/events/event_models.py`

Add to top of file:
```python
"""
DEPRECATED: Use core.contracts.events instead.
This module has zero production usage and will be removed in v2.0.0.
"""
import warnings
warnings.warn(
    "core.events.event_models is deprecated. Use core.contracts.events.",
    DeprecationWarning,
    stacklevel=2
)
```

#### 4.2 Deprecate Legacy Optimizers

Add to `daily_optimize.py` and `daily_optimize_v2.py`:
```python
"""DEPRECATED: Use daily_optimize_v3.py"""
import warnings
warnings.warn("Use daily_optimize_v3.py", DeprecationWarning, stacklevel=2)
```

#### 4.3 Deprecate `core/backtest/backtester.py`

```python
"""DEPRECATED: Use unified_backtest_runner.py"""
import warnings
warnings.warn("Use unified_backtest_runner.py", DeprecationWarning, stacklevel=2)
```

#### 4.4 Update `CONSOLIDATION_ARCHITECTURE.md`

Document deprecation timeline.

#### Critical Files
- `/Users/kwasiaddo/projects/trader/amsterdam/core/events/event_models.py`
- `/Users/kwasiaddo/projects/trader/amsterdam/core/backtest/backtester.py`
- `/Users/kwasiaddo/projects/trader/amsterdam/core/backtest/daily_optimize.py`
- `/Users/kwasiaddo/projects/trader/amsterdam/core/backtest/daily_optimize_v2.py`

#### Verification
```bash
# Verify warnings emitted
python -W all -c "from core.events.event_models import OrderStatusEvent" 2>&1 | grep DeprecationWarning

# Deprecated code still works
pytest tests/core/backtest/test_backtester.py -v

# No new warnings in production
pytest tests/ 2>&1 | grep "DeprecationWarning.*amsterdam" || echo "✓ Clean"
```

---

### Phase 5: Remove Deprecated Code (After 1 release cycle)

**Goal**: Clean removal after migration period

#### Actions
```bash
# Remove orphaned code (safe, zero usage)
git rm core/events/event_models.py

# Remove legacy backtester
git rm core/backtest/backtester.py

# Remove old optimizers, rename v3 to canonical
git rm core/backtest/daily_optimize.py core/backtest/daily_optimize_v2.py
git mv core/backtest/daily_optimize_v3.py core/backtest/daily_optimize.py

# Remove deprecated shim
git rm core/events/events.py
```

#### Update Documentation
Remove deprecation notices, add to changelog.

#### Verification
```bash
# No broken imports
python -c "import core; import amsterdam"

# Full test suite
pytest tests/ -v --cov=core --cov=strategies

# CLI still works
amsterdam preflight
amsterdam gui --mode simulation &
sleep 5 && pkill -f amsterdam

# Check for remaining references
grep -r "event_models\|backtester\.py\|daily_optimize_v[12]" --include="*.py" . || echo "✓ Clean"
```

---

## Risk Mitigation

### Rollback Strategy
- **Phase 1**: Delete doc files
- **Phase 2**: Delete `app/` directory
- **Phase 3**: Remove bootstrap calls, revert to old code
- **Phase 4**: Remove warnings
- **Phase 5**: Git revert specific commits

### Production Safety
- ✅ Backwards compatible until Phase 5
- ✅ Deprecation warnings don't affect functionality
- ✅ Testing at each phase gate
- ✅ No changes to deployed systems until ready

---

## Success Metrics

**Code Quality**
- ✅ Single canonical init path (`app/bootstrap.py`)
- ✅ Zero orphaned code
- ✅ Clear legacy markers
- ✅ Documented architecture

**Developer Experience**
- ✅ Clear entry points (reduced onboarding time)
- ✅ Easier testing (bootstrap_app() reusable)
- ✅ Consistent patterns

**Maintainability**
- ✅ Fewer code paths
- ✅ Explicit canonical choices
- ✅ Migration guide for external deps

---

## Timeline Summary

| Phase | Duration | Risk | Focus |
|-------|----------|------|-------|
| 1: Documentation ✅ | 2-3 days | None | Clarity |
| 2: Bootstrap Layer | 3-4 days | Low | Foundation |
| 3: Migration | 5-7 days | Medium | Integration |
| 4: Deprecation | 2-3 days | Low | Warnings |
| 5: Removal | 2-3 days | Low | Cleanup |

**Total**: 3-4 weeks + deprecation period (1 release cycle)

---

## End-to-End Verification

After all phases complete:

```bash
# 1. Bootstrap works for all modes
python -c "
from app.bootstrap import bootstrap_app
for mode in ['daemon', 'gui', 'cli']:
    ctx = bootstrap_app(mode, symbols=['AAPL'])
    assert ctx.config is not None
    assert ctx.logger is not None
    print(f'✓ {mode} bootstrap works')
"

# 2. All entrypoints use bootstrap
grep -r "bootstrap_app" autoamsterdam.py run_trading.py cli/main.py || echo "⚠️  Not all migrated"

# 3. No orphaned code
test ! -f core/events/event_models.py && echo "✓ event_models removed"
test ! -f core/backtest/backtester.py && echo "✓ backtester removed"

# 4. Documentation complete
test -f docs/CONSOLIDATION_ARCHITECTURE.md && test -f docs/CONSOLIDATION_MIGRATION_GUIDE.md && echo "✓ Docs exist"

# 5. Full test suite passes
pytest tests/ -v --cov=core --cov=app

# 6. Production smoke tests
python autoamsterdam.py --dry-run --symbols AAPL &
sleep 10 && pkill -f autoamsterdam && echo "✓ Daemon works"

python run_trading.py --mode simulation --symbols AAPL --speed 1.0 &
sleep 10 && pkill -f run_trading && echo "✓ GUI works"

amsterdam preflight && echo "✓ CLI works"
```

---

**This plan provides a clear, incremental path to consolidation while maintaining stability and backwards compatibility.**
