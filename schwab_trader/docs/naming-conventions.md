# Naming Conventions

This document defines naming standards for the trading system codebase.

## Suffix Conventions

| Suffix | Purpose | Example |
|--------|---------|---------|
| `*State` | Domain data objects (mutable state containers) | `PortfolioState`, `SymbolState` |
| `*Policy` | Trade logic, risk rules, gating decisions | `ExitPolicy`, `RiskPolicy` |
| `*Manager` | Orchestrates multiple components, owns lifecycle | `PositionManager`, `TradeLogicManager` |
| `*Service` | Use-case orchestration, business logic | `TradingService`, `ReconciliationService` |
| `*Adapter` | External integrations (brokers, storage, APIs) | `AlpacaAdapter`, `SchwabAdapter` |
| `*Repository` | Persistence access (read/write data) | `OrderRepository`, `BarRepository` |
| `*Runner` | Process lifecycle control | `AlpacaLiveRunner`, `SchwabLiveRunner` |
| `*Factory` | Creates instances based on config/type | `RunnerFactory`, `PositionSizerFactory` |
| `*Router` | Routes requests to appropriate handlers | `TradeApproverRouter`, `StrategyRouter` |
| `*Handler` | Handles events or callbacks | `EventHandler`, `BarHandler` |
| `*Validator` | Validates data or conditions | `TradeValidator`, `CredentialValidator` |
| `*Monitor` | Observes and tracks metrics | `DrawdownMonitor`, `HealthMonitor` |
| `*Gate` | Controls access/flow (permit/deny) | `TradeGate` |
| `*Approver` | Makes approval decisions | `TradeApprover`, `StandardTradeApprover` |
| `*Sizer` | Calculates sizes/quantities | `PositionSizer`, `KellyPositionSizer` |
| `*Loader` | Loads data from storage | `HistoricalBarLoader`, `ConfigLoader` |
| `*Pipeline` | Multi-stage data processing | `UnifiedDataPipeline`, `FeaturePipeline` |

---

## Current vs Recommended Names

### Brokers (Adapters)
| Current | Status | Notes |
|---------|--------|-------|
| `AlpacaBroker` | OK | Could be `AlpacaAdapter` |
| `SchwabBroker` | OK | Could be `SchwabAdapter` |
| `MockBroker` | OK | Could be `MockAdapter` |
| `BaseBrokerInterface` | OK | Abstract interface |

### Execution
| Current | Status | Notes |
|---------|--------|-------|
| `LiveExecutionEngine` | OK | Orchestrates execution |
| `MockExecutionEngine` | OK | Test double |
| `ExecutionEngineBase` | OK | Abstract base |
| `Executor` | Ambiguous | Consider `OrderExecutor` |

### Trade Logic
| Current | Recommended | Notes |
|---------|-------------|-------|
| `StandardTradeApprover` | OK | Clear purpose |
| `TradeApprover` | OK | Base class |
| `PositionManager` | OK | Manages position lifecycle |
| `TradeGate` | OK | Gating logic |
| `TradeValidator` | OK | Validates trades |

### State
| Current | Status | Notes |
|---------|--------|-------|
| `PortfolioState` | OK | Domain state |
| `SymbolState` | OK | Domain state |
| `SymbolPosition` | OK | Data object |

### Data
| Current | Recommended | Notes |
|---------|-------------|-------|
| `UnifiedDataPipeline` | OK | Multi-stage processing |
| `HistoricalBarLoader` | OK | Loads data |
| `DataStore` | Could be `BarRepository` | Persistence layer |
| `Processor` | Could be `DataProcessor` | More specific |
| `Aggregator` | DEPRECATED | Use UnifiedDataPipeline |

### Runners
| Current | Status | Notes |
|---------|--------|-------|
| `AlpacaLiveRunner` | OK | Lifecycle control |
| `SchwabLiveRunner` | OK | Lifecycle control |
| `BaseLiveRunner` | OK | Abstract base |

---

## File Naming

### Conventions
- Use `snake_case` for all Python files
- Base classes: `*_base.py` or `base_*.py` (prefer `*_base.py`)
- Interfaces: `*_interface.py`
- Tests: `test_*.py`

### Current Inconsistencies
| File | Pattern | Notes |
|------|---------|-------|
| `base_broker_interface.py` | `base_*` | OK |
| `base_live_runner.py` | `base_*` | OK |
| `position_sizer_base.py` | `*_base` | OK |
| `executor_base.py` | `*_base` | OK |
| `execution_engine_base.py` | `*_base` | OK |

**Decision:** Both patterns are acceptable. Don't rename existing files.

---

## Module Organization

### Recommended Structure
```
core/
├── adapters/           # External integrations
│   ├── alpaca/
│   ├── schwab/
│   └── coinbase/
├── domain/             # Domain models and state
│   ├── portfolio_state.py
│   ├── symbol_state.py
│   └── order.py
├── services/           # Business logic orchestration
│   ├── execution_service.py
│   └── reconciliation_service.py
├── policies/           # Trade/risk policies
│   ├── exit_policy.py
│   └── entry_policy.py
└── infrastructure/     # Cross-cutting concerns
    ├── persistence/
    ├── events/
    └── logging/
```

**Note:** This is a target structure for future refactoring. Current structure is functional.

---

## Variable Naming

### Conventions
- Use `snake_case` for variables and functions
- Use `PascalCase` for classes
- Use `UPPER_SNAKE_CASE` for constants
- Prefix private methods/attributes with `_`
- Prefix "very private" with `__` (rare)

### Domain-Specific Terms
| Term | Meaning |
|------|---------|
| `qty` | Quantity (shares) |
| `px` | Price |
| `sl` | Stop loss |
| `tp` | Take profit |
| `atr` | Average True Range |
| `mfe` | Maximum Favorable Excursion |
| `mae` | Maximum Adverse Excursion |
| `pnl` | Profit and Loss |

---

## Deprecation Naming

When deprecating a module:
1. Add `DEPRECATED` or `LEGACY` to the docstring
2. Add `warnings.warn()` call with `DeprecationWarning`
3. Point to the replacement in the docstring

Example:
```python
"""
Module Name - DEPRECATED

Use NewModule instead:
    from core.new_module import NewModule

This module is maintained for backward compatibility only.
"""
import warnings

class OldClass:
    def __init__(self):
        warnings.warn(
            "OldClass is deprecated. Use NewClass instead.",
            DeprecationWarning,
            stacklevel=2
        )
```

---

## Summary

**Key Principles:**
1. Names should reveal intent
2. Suffixes indicate architectural role
3. Don't rename existing stable code without good reason
4. Document deprecations clearly
5. New code should follow these conventions
