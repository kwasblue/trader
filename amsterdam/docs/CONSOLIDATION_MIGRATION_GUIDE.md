# Amsterdam Trading System - Migration Guide

## Overview

This guide provides upgrade paths for code using deprecated components being removed in v2.0.0.

## Quick Reference

| Deprecated | Use Instead | Status |
|------------|-------------|--------|
| `core.events.event_models` | `core.contracts.events` | Orphaned (0 usage) |
| `core.backtest.backtester` | `core.backtest.unified_backtest_runner` | Superseded |
| `daily_optimize.py` (v1, v2) | `daily_optimize_v3.py` | Superseded |
| Direct `Logger()` construction | `app.bootstrap.bootstrap_app()` | New pattern |
| Direct `init_root_logger()` | `app.bootstrap.bootstrap_app()` | New pattern |

---

## Event System Migration

### ❌ Deprecated: event_models.py

**Status**: Zero production usage, will be removed in v2.0.0

```python
# OLD (deprecated)
from core.events.event_models import OrderStatusEvent

event = OrderStatusEvent(
    order_id="123",
    status="filled",
    symbol="AAPL"
)
```

### ✅ Use: core.contracts.events

```python
# NEW (canonical)
from core.contracts.events import OrderStatusUpdateEvent

event: OrderStatusUpdateEvent = {
    "type": "order_status_update",
    "order_id": "123",
    "status": "filled",
    "symbol": "AAPL",
    "timestamp": datetime.now(timezone.utc)
}
```

**Why?**
- TypedDict-based events are simpler and more flexible
- `core.contracts.events` is already used by 21+ production files
- No runtime overhead from Pydantic validation
- Better IDE autocomplete

---

## Backtesting Migration

### ❌ Deprecated: backtester.py

**Status**: Superseded by unified_backtest_runner.py

```python
# OLD (deprecated)
from core.backtest.backtester import Backtester

bt = Backtester(
    symbol="AAPL",
    strategy=my_strategy,
    start_date="2023-01-01",
    end_date="2023-12-31"
)
results = bt.run()
```

### ✅ Use: unified_backtest_runner.py

```python
# NEW (canonical)
from core.backtest.unified_backtest_runner import UnifiedBacktestRunner

runner = UnifiedBacktestRunner(
    symbols=["AAPL"],
    strategy=my_strategy,
    start_date="2023-01-01",
    end_date="2023-12-31",
    mode="historical"  # or "simulation"
)
results = runner.run()
```

**Why?**
- Supports multiple symbols
- Unified interface for historical and simulation modes
- Better performance metrics
- 28KB vs 10KB (more features)

---

## Optimizer Migration

### ❌ Deprecated: daily_optimize.py (v1, v2)

**Status**: Superseded by v3

```python
# OLD (deprecated)
from core.backtest.daily_optimize import optimize_parameters
# or
from core.backtest.daily_optimize_v2 import optimize_parameters_v2

results = optimize_parameters(symbol="AAPL", ...)
```

### ✅ Use: daily_optimize_v3.py

```python
# NEW (current)
from core.backtest.daily_optimize_v3 import optimize_parameters

results = optimize_parameters(
    symbol="AAPL",
    start_date="2023-01-01",
    end_date="2023-12-31",
    param_grid={
        "atr_period": [10, 14, 20],
        "sma_period": [20, 50, 100]
    }
)
```

**Why?**
- Better parallel processing
- Improved metrics calculation
- More robust error handling
- Will be renamed to `daily_optimize.py` in v2.0.0

---

## Initialization Migration

### ❌ Deprecated: Direct Logger/init_root_logger

**Status**: Fragmented patterns being consolidated

#### Old Pattern 1: autoamsterdam.py

```python
# OLD (deprecated)
from loggers.logger import Logger

class AutoTrader:
    def __init__(self, symbols):
        self.logger = Logger("autotrader.log", "AutoTrader").get_logger()
        self.config = get_config()
        self.symbols = symbols
```

#### Old Pattern 2: run_trading.py

```python
# OLD (deprecated)
from loggers.bootstrap import init_root_logger

def main():
    init_root_logger(log_dir="logs", root_file="app.log", level=logging.DEBUG)
    logger = logging.getLogger("TradingApp")
    config = get_config()
```

### ✅ Use: Unified Bootstrap

```python
# NEW (canonical)
from app.bootstrap import bootstrap_app, AppContext

# In main() or factory function
ctx = bootstrap_app(
    mode='daemon',  # or 'gui', 'cli'
    symbols=['AAPL', 'MSFT'],
    broker='schwab',
    log_level=logging.INFO,
    console_logging=True
)

# Use context
class AutoTrader:
    def __init__(self, context: AppContext):
        self.logger = context.logger
        self.config = context.config
        self.symbols = context.symbols
        self.root_path = context.root_path
```

**Why?**
- Single initialization path for all entrypoints
- Consistent logging setup
- Easy to test (just pass mock AppContext)
- Clear dependency injection

---

## Backwards Compatibility

All migrations maintain backwards compatibility until v2.0.0:

### Deprecation Warnings

When using deprecated code, you'll see warnings:

```python
# Using deprecated event_models.py
DeprecationWarning: core.events.event_models is deprecated. Use core.contracts.events.

# Using old backtester
DeprecationWarning: Use unified_backtest_runner.py
```

### Suppressing Warnings (Not Recommended)

```python
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
```

**Better**: Fix the code now rather than suppressing warnings.

---

## Migration Timeline

### Phase 4 (Current)
- ✅ Deprecation warnings added
- ✅ Old code still works
- ⚠️ Warnings emitted when using deprecated code

### v2.0.0 (Future Release)
- ❌ Deprecated code removed
- ✅ Only canonical paths remain
- ✅ Cleaner codebase

**Recommended**: Migrate during Phase 4 to avoid breaking changes in v2.0.0.

---

## Testing Your Migration

### Verify No Deprecated Imports

```bash
# Check for deprecated imports in your code
grep -r "event_models" --include="*.py" your_code/
grep -r "from core.backtest.backtester import" --include="*.py" your_code/
grep -r "daily_optimize_v[12]" --include="*.py" your_code/
```

### Run Tests with Warnings

```bash
# See deprecation warnings
python -W all -m pytest tests/

# Or in code
pytest tests/ -W default::DeprecationWarning
```

### Verify Bootstrap Works

```python
# Test bootstrap initialization
from app.bootstrap import bootstrap_app

ctx = bootstrap_app(mode='cli', symbols=['AAPL'])
assert ctx.config is not None
assert ctx.logger is not None
assert 'AAPL' in ctx.symbols
print("✓ Bootstrap works")
```

---

## Common Migration Patterns

### Pattern 1: Testing Code

**Before**:
```python
def test_strategy():
    logger = Logger("test.log", "Test").get_logger()
    config = get_config()
    # ... test code
```

**After**:
```python
def test_strategy():
    from app.bootstrap import bootstrap_app

    ctx = bootstrap_app(mode='cli', symbols=['AAPL'])
    # Use ctx.logger, ctx.config
    # ... test code
```

### Pattern 2: Standalone Scripts

**Before**:
```python
# script.py
from loggers.bootstrap import init_root_logger
from core.config_loader import get_config

init_root_logger(log_dir="logs", root_file="script.log")
logger = logging.getLogger("Script")
config = get_config()
```

**After**:
```python
# script.py
from app.bootstrap import bootstrap_app

ctx = bootstrap_app(mode='cli', operation='script')
logger = ctx.logger
config = ctx.config
```

### Pattern 3: Class Constructors

**Before**:
```python
class MyClass:
    def __init__(self, symbols, broker):
        self.logger = Logger("myclass.log", "MyClass").get_logger()
        self.config = get_config()
        self.symbols = symbols
        self.broker = broker
```

**After**:
```python
from typing import Optional
from app.bootstrap import AppContext, bootstrap_app

class MyClass:
    def __init__(self, context: Optional[AppContext] = None, **kwargs):
        # Backwards compatibility
        if context is None:
            import warnings
            warnings.warn(
                "Direct construction deprecated, use bootstrap_app()",
                DeprecationWarning
            )
            context = bootstrap_app(
                mode='cli',
                symbols=kwargs.get('symbols', []),
                broker=kwargs.get('broker')
            )

        self.logger = context.logger
        self.config = context.config
        self.symbols = context.symbols
        self.broker = context.metadata.get('broker')
```

---

## Need Help?

- **Architecture questions**: See `CONSOLIDATION_ARCHITECTURE.md`
- **Implementation details**: See `CONSOLIDATION_PLAN.md`
- **Issues**: Check GitHub issues or create a new one

---

## Summary Checklist

Before v2.0.0, ensure:

- [ ] No imports from `core.events.event_models`
- [ ] Using `unified_backtest_runner.py` instead of `backtester.py`
- [ ] Using `daily_optimize_v3.py` instead of v1/v2
- [ ] All entrypoints use `bootstrap_app()`
- [ ] Tests pass with `-W default::DeprecationWarning`
- [ ] No deprecated imports in your codebase

**Migration complete when all boxes are checked.**
