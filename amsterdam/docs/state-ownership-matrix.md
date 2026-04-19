# State Ownership Matrix

This document defines clear ownership boundaries for state management components.

## Authority Hierarchy

```
EXTERNAL TRUTH (eventual consistency)
    └── Broker API (positions, cash, orders)
            │
            ▼
INTERNAL TRUTH (operational state)
    └── PortfolioState (canonical position/cash/P&L)
            │
            ▼
WORKING STATE (per-symbol execution context)
    └── SymbolState (stops, targets, bars held)
            │
            ▼
POLICY LAYER (decisions, no mutation)
    ├── PositionManager (exit rules, trailing logic)
    ├── TradeApprover (entry/exit gating)
    └── TradeGate (cooldowns, pyramiding limits)
            │
            ▼
AUDIT LAYER (correction only)
    └── StateReconciler (sync local ↔ broker)
```

---

## Component Responsibilities

### PortfolioState
**Role:** Canonical source of truth for positions, cash, and P&L.

| Owns | Does NOT Own |
|------|--------------|
| Position quantities | Stop/target levels |
| Average entry prices | Strategy assignments |
| Cash balance | Cooldown state |
| Realized/unrealized P&L | Pyramid layer counts |
| Equity history | Bars held |
| Position lifecycle states | Order routing decisions |

**Key Methods:**
- `apply_fill()` - THE canonical way to update positions from trades
- `apply_fill_safe()` - Thread-safe async version with lock
- `sync_from_snapshot()` - Full broker sync (startup, reconciliation)
- `sync_position_from_broker()` - Single-symbol sync from broker
- `remove_position()` - Remove position closed on broker
- `set_position_state()` - Track lifecycle (validates transitions)

**Rules:**
- All fill updates MUST go through `apply_fill()` or `apply_fill_safe()`
- Broker syncs use `sync_from_snapshot()` or `sync_position_from_broker()`
- NEVER directly assign to `portfolio.positions[symbol]`
- Position lifecycle state changes require valid transitions

---

### SymbolState
**Role:** Per-symbol working state for trade execution (dumb container).

| Owns | Does NOT Own |
|------|--------------|
| Stop loss level | Position quantity (synced from Portfolio) |
| Take profit level | Cash balance |
| Partial exit targets | P&L calculations |
| Bars held counter | Order execution |
| Entry date (for swing mode) | Lifecycle transitions |
| Strategy name | Cooldown timers |
| MFE/MAE tracking | |

**Key Methods:**
- `update_from_portfolio()` - Sync qty/price from PortfolioState
- `reset()` - Clear state when position closed

**Rules:**
- SymbolState is a DATA CONTAINER, not a decision maker
- Position quantity comes FROM PortfolioState, not the other way
- PositionManager SETS levels, SymbolState STORES them

---

### PositionManager
**Role:** Pure policy for position lifecycle (exits, stops, targets).

| Owns | Does NOT Own |
|------|--------------|
| Exit condition logic | Position quantity |
| Trailing stop calculation | Cash balance |
| Partial exit rules | Order placement |
| MFE/MAE update logic | State persistence |
| TP/SL multipliers | Cooldown tracking |

**Key Methods:**
- `calculate_levels()` - Set SL/TP/partials on SymbolState
- `update_trailing_stop()` - Ratchet stop on SymbolState
- `check_exit_conditions()` - Return (should_exit, reason)

**Rules:**
- PositionManager READS from SymbolState, WRITES levels TO SymbolState
- Never mutates PortfolioState directly
- Returns decisions, does not execute them

---

### TradeApprover (StandardTradeApprover)
**Role:** Gatekeeper for trade entry/exit permissions.

| Owns | Does NOT Own |
|------|--------------|
| Entry permission logic | Position state |
| Cooldown checking | Stop/target levels |
| Position limit checks | Order execution |
| Market hours validation | P&L tracking |
| TP/SL multiplier storage | |

**Key Methods:**
- `should_trade()` - Return (approved, reason)
- `can_enter_position()` - Check entry conditions
- `can_exit_position()` - Check exit conditions

**Rules:**
- Pure gating: returns True/False with reason
- Does NOT execute trades
- Does NOT modify position state

---

### TradeGate
**Role:** Per-symbol action tracking and pyramiding limits.

| Owns | Does NOT Own |
|------|--------------|
| `did_action_this_bar` flag | Position quantity |
| Pyramid layer count | Cash balance |
| Bars since last layer | Stop/target levels |
| Flip cooldown timer | Order execution |
| Regime persistence counter | |

**Key Methods:**
- `can_enter()` - Check if entry allowed (pyramiding, cooldowns)
- `mark_action()` - Record that action was taken
- `on_new_bar()` - Reset per-bar flags

**Rules:**
- Tracks ACTIONS, not POSITIONS
- Used by ExecutionEngine to prevent duplicate signals
- Complements (not replaces) TradeApprover

---

### StateReconciler
**Role:** Audit bridge between local state and broker.

| Owns | Does NOT Own |
|------|--------------|
| Mismatch detection | Routine position updates |
| Sync scheduling | Trade execution |
| Correction decisions | Order placement |
| Halt triggers | Strategy logic |

**Key Methods:**
- `full_sync()` - Complete state sync from broker
- `check_positions()` - Detect mismatches
- `verify_order()` - Confirm fill matches expectation

**Rules:**
- Only runs on startup, periodically, or after suspected drift
- May call `PortfolioState.sync_from_snapshot()`
- Can trigger HALT state on critical mismatches

---

### OrderRegistry
**Role:** Local order cache for fast lookups.

| Owns | Does NOT Own |
|------|--------------|
| Order ID → status mapping | Position state |
| Pending order tracking | Fill application |
| Conflict detection | Cash balance |
| Order lifecycle metadata | Execution decisions |

**Key Methods:**
- `register()` - Add new order
- `update_status()` - Update order status
- `get_pending_for_symbol()` - Check for open orders
- `has_conflicting_order()` - Wash trade prevention

**Rules:**
- Cache only - broker is source of truth for orders
- Does NOT apply fills (that's PortfolioState)
- Used for fast lookups without broker queries

---

## State Flow Diagram

```
[Signal Generated]
       │
       ▼
[TradeApprover.should_trade()] ──No──► [Reject]
       │ Yes
       ▼
[TradeGate.can_enter()] ──No──► [Reject: cooldown/layers]
       │ Yes
       ▼
[PositionSizer.calculate_position_size()]
       │
       ▼
[ExecutionEngine submits order]
       │
       ▼
[OrderRegistry.register(order_id)]
       │
       ▼
[Broker executes fill]
       │
       ▼
[PortfolioState.apply_fill()] ◄── CANONICAL UPDATE
       │
       ▼
[SymbolState.update_from_portfolio()]
       │
       ▼
[PositionManager.calculate_levels()] → sets SL/TP on SymbolState
       │
       ▼
[TradeGate.mark_action()]
```

---

## Anti-Patterns to Avoid

1. **Never update SymbolState.current_position directly**
   - Always sync from PortfolioState via `update_from_portfolio()`

2. **Never call PortfolioState.sync_from_snapshot() during normal trading**
   - Only StateReconciler does this on startup/mismatch

3. **Never make PositionManager apply fills**
   - It calculates exits, it doesn't execute them

4. **Never let TradeGate own position truth**
   - It tracks actions/layers, not actual positions

5. **Never bypass apply_fill() for position changes**
   - All fills must go through this method for proper P&L tracking

---

## Validation Checklist

When reviewing code changes, verify:

- [ ] All fills go through `PortfolioState.apply_fill()`
- [ ] SymbolState syncs FROM PortfolioState (not vice versa)
- [ ] PositionManager only writes to SymbolState fields (stops, targets)
- [ ] TradeApprover returns decisions, doesn't mutate state
- [ ] StateReconciler only runs at defined sync points
- [ ] OrderRegistry doesn't apply fills

---

## File Locations

| Component | File |
|-----------|------|
| PortfolioState | `core/logic/portfolio_state.py` |
| SymbolState | `core/logic/symbol_state.py` |
| PositionManager | `core/logic/position_manager.py` |
| TradeApprover | `core/logic/default_trade_logic.py` |
| TradeGate | `core/logic/trade_gate.py` |
| StateReconciler | `core/state_reconciler.py` |
| OrderRegistry | `core/order_registry.py` |
