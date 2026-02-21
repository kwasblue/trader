# Architecture Overview

This document describes the canonical architecture of the trading system.

## Design Principles

1. **Single Responsibility**: Each component has ONE job with ONE clear name
2. **Clear Layers**: Event system → Trade logic → Execution → Broker
3. **No Duplication**: State lives in one place, not duplicated across layers

---

## Core Components

### Event System

**Canonical Module**: `core/events/eventhandler.py`

```
EventHandler (Singleton)
└── Single event bus for all pub/sub communication
```

The event system provides async pub/sub for decoupled component communication.
All events flow through this single bus.

### Trade Logic

```
TradeApprover (base)           # Gates: should we trade?
├── StandardTradeApprover      # Concrete gating with cooldowns, position limits
└── TradeApproverRouter        # Routes to symbol/strategy/regime-specific approvers

PositionManager                # Manages: SL/TP/trailing/exits
└── Handles position lifecycle after trade approval
```

**Responsibilities**:
- `TradeApprover`: Decides IF we should trade (cooldowns, limits, conditions)
- `PositionManager`: Decides HOW to manage open positions (stops, targets, exits)

**Exit Timing Rules** (owned by PositionManager):
- `min_bars_to_hold`: Minimum bars before TP/reversal exits
- `swing_mode`: Enforce multi-day holds
- `min_hold_days`: Days to hold in swing mode before exit

### Execution Stack

```
ExecutionEngineBase                        # Abstract base with canonical API
├── handle_signal(context, state)          # Primary entry point
├── handle_signal_context(context)         # Abstract (async engines implement)
└── handle_signal_legacy(...)              # Deprecated backward compat

GenericExecutionEngine (sync)              # For backtesting/simulation
├── Synchronous signal handling
└── Direct order execution

MockExecutionEngine (async)                # For paper trading
├── Async signal handling
└── Simulated order fills

LiveExecutionEngine (async)                # For live trading
├── Handles signals from strategies
├── Coordinates trade approval, sizing, validation
├── Emits events for GUI/monitoring
└── Manages GUI commands (manual orders, flatten, cancel)

LiveExecutor                               # Thin broker adapter
├── Places orders via broker
├── Cancels orders
└── Queries order status
```

**Key Insight**: The Engine is the orchestrator; the Executor is just an adapter.
All engines share the same `handle_signal(context, state)` signature.

---

## API Contract: SignalContext-First

All engines use a unified API:

```python
# Primary entry point (all engines)
engine.handle_signal(context: SignalContext, state: SymbolState) -> Optional[OrderResult]

# SignalContext contains all signal data
SignalContext(
    symbol="AAPL",
    signal=1,           # -1, 0, 1
    price=150.0,
    atr=2.5,
    regime="normal",
    timestamp=datetime.now(timezone.utc),
    strategy_name="momentum",
    confidence=0.8,
    market_open=True,
    metadata={}         # Additional context
)

# SymbolState contains per-symbol trading state
SymbolState(
    symbol="AAPL",
    side="long",        # "long", "short", None
    entry_price=148.0,
    stop_loss=145.0,
    take_profit=155.0,
    bars_held=5,
    ...
)
```

**Deprecated**: `handle_signal_legacy()` accepts loose parameters for backward compatibility.

---

## Data Flow

```
Strategy Signal
    ↓
SignalContext.from_kwargs(...)  →  Immutable context object
    ↓
Engine.handle_signal(context, state)
    ↓
TradeApprover.should_trade(context, state)  →  Pure gating (allowed?)
    ↓
╔══════════════════════════════════════════════════╗
║ IF NOT IN POSITION (entry):                      ║
║   → PositionSizer.calculate_position_size()      ║
║   → TradeValidator.validate()                    ║
║   → LiveExecutor.buy/sell()  →  Broker           ║
║   → PositionManager.calculate_levels() (SL/TP)   ║
╠══════════════════════════════════════════════════╣
║ IF IN POSITION:                                  ║
║   → PositionManager.check_exit_conditions()      ║
║   → IF should_exit:                              ║
║       → LiveExecutor.buy/sell() → Broker         ║
║   → ELSE:                                        ║
║       → PositionManager.update_trailing_stop()   ║
║       → PositionManager.update_excursions()      ║
╚══════════════════════════════════════════════════╝
    ↓
Event emission (NEW_TRADE, POSITION_UPDATE, etc.)
```

---

## Configuration

| File | Purpose |
|------|---------|
| `core/config_loader.py` | Trading configuration (strategies, risk params) |
| `utils/settings.py` | Infrastructure configuration (API keys, paths) |

---

## Key Files

| File | Responsibility |
|------|----------------|
| `core/events/eventhandler.py` | Single canonical event bus |
| `core/logic/position_manager.py` | Position lifecycle (SL/TP/trailing) |
| `core/logic/default_trade_logic.py` | Trade approval (gating) |
| `core/logic/live_execution_engine.py` | Orchestrator (signals → orders) |
| `core/executor.py` | Thin broker adapter |
| `core/logic/portfolio_state.py` | Portfolio state (positions, cash) |
| `core/logic/symbol_state.py` | Per-symbol trading state |

---

## Anti-Patterns Avoided

1. **No duplicate position tracking**: Portfolio is authoritative, not Executor
2. **No event emission in Executor**: Events belong in Engine layer
3. **No signal handling in Executor**: Engine handles signals, Executor places orders
4. **No dead code**: Unused classes/methods have been removed
5. **Single event bus**: EventHandler is the ONLY event emitter (no duplicate buses)
6. **Clear separation**: TradeApprover gates, PositionManager manages positions
7. **Engines don't register logic**: Use TradeApproverRouter directly, engines just consume it
8. **SymbolState is dumb**: Just fields + properties, PositionManager does all "smart" checks
9. **Executors don't size**: Engines compute qty, executors just place orders with given qty
10. **No method duplication**: Shared helpers stay in base, subclasses inherit (not copy)

---

## Inheritance Pattern

```
ExecutionEngineBase (owns shared methods)
├── handle_signal(context, state)       # Base default (calls handle_signal_context)
├── handle_signal_legacy(...)           # Sync shim for backward compat
├── _determine_action(...)              # Shared by all engines
├── _setup_approval_state(...)          # Shared by all engines
├── _get_exit_quantity(...)             # Shared by all engines
└── _post_execution(...)                # Default, can be overridden

GenericExecutionEngine (sync)
├── handle_signal(context, state)       # OVERRIDE: sync implementation
├── handle_signal_legacy                # INHERITED from base (sync)
├── _determine_action                   # INHERITED from base
├── _check_trade_approval(...)          # Own: uses approver_router
└── _post_execution(...)                # OVERRIDE: sync version

MockExecutionEngine (async)
├── handle_signal(context, state)       # OVERRIDE: async implementation
├── handle_signal_legacy                # OVERRIDE: async (calls await handle_signal)
├── _determine_action                   # INHERITED from base
└── _execute_mock_trade(...)            # Own: mock fill logic

LiveExecutionEngine (async)
├── handle_signal(context, state)       # OVERRIDE: async implementation
├── handle_signal_legacy                # OVERRIDE: async (calls await handle_signal)
├── _determine_action                   # INHERITED from base
├── _check_daily_loss_limit_breached()  # Own: live-specific guardrail
└── _reconcile_positions()              # Own: live-specific sync
```

**Rule**: If a method is identical to base, DELETE IT and inherit. Only override when behavior differs.

---

## Testing

Run verification:

\`\`\`bash
# All tests pass
pytest tests/ -v

# Verify no duplicate position tracking
grep -r "self.positions\s*=" --include="*.py" core/

# Verify no event emission in Executor
grep -r "_emit_" core/executor.py  # Should return nothing

# Verify imports work
python -c "from core.logic.live_execution_engine import LiveExecutionEngine; print('OK')"
python -c "from core.logic.position_manager import PositionManager; print('OK')"
\`\`\`
