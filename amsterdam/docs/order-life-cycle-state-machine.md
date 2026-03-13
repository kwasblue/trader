# Order Lifecycle State Machine

This document defines the order lifecycle and state transitions for the trading system.

## Overview

The system tracks two parallel state machines:
1. **OrderStatus** - Broker-level order state (per-order)
2. **PositionState** - Position lifecycle state (per-symbol)

These must stay synchronized to prevent race conditions, duplicate orders, and state drift.

---

## Order Status State Machine

```
                     ┌─────────────────────────────────────────┐
                     │              ORDER LIFECYCLE            │
                     └─────────────────────────────────────────┘

    ┌─────────┐     ┌──────────┐     ┌─────────┐     ┌────────┐
    │ PENDING │────►│ ACCEPTED │────►│ PARTIAL │────►│ FILLED │
    └─────────┘     └──────────┘     └─────────┘     └────────┘
         │               │                │
         │               │                │
         ▼               ▼                ▼
    ┌──────────┐    ┌───────────┐   ┌───────────┐
    │ REJECTED │    │ CANCELLED │   │ CANCELLED │
    └──────────┘    └───────────┘   └───────────┘
         │
         ▼
    ┌─────────┐
    │ EXPIRED │  (for DAY orders at market close)
    └─────────┘
```

### States

| State | Description | is_open | is_closed |
|-------|-------------|---------|-----------|
| `PENDING` | Order submitted, awaiting broker acknowledgment | ✓ | |
| `ACCEPTED` | Broker accepted, awaiting execution | ✓ | |
| `PARTIAL` | Partially filled | ✓ | |
| `FILLED` | Completely filled | | ✓ |
| `CANCELLED` | User or system cancelled | | ✓ |
| `REJECTED` | Broker rejected | | ✓ |
| `EXPIRED` | Day order expired at close | | ✓ |

### Valid Transitions

```python
VALID_ORDER_TRANSITIONS = {
    OrderStatus.PENDING: {OrderStatus.ACCEPTED, OrderStatus.REJECTED},
    OrderStatus.ACCEPTED: {OrderStatus.PARTIAL, OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED},
    OrderStatus.PARTIAL: {OrderStatus.FILLED, OrderStatus.CANCELLED},
    OrderStatus.FILLED: set(),      # Terminal
    OrderStatus.CANCELLED: set(),   # Terminal
    OrderStatus.REJECTED: set(),    # Terminal
    OrderStatus.EXPIRED: set(),     # Terminal
}
```

---

## Position State Machine

```
                     ┌─────────────────────────────────────────┐
                     │           POSITION LIFECYCLE            │
                     └─────────────────────────────────────────┘

                           entry order
                             placed
    ┌──────┐              ┌─────────────────┐
    │ NONE │─────────────►│  PENDING_ENTRY  │
    └──────┘              └─────────────────┘
       ▲                          │
       │                          │ filled
       │ filled/cancelled         ▼
       │                     ┌────────┐
       │◄────────────────────│  OPEN  │◄──────────────┐
       │  exit order         └────────┘               │
       │  filled                  │                   │ fill/cancel
                                  │                   │
                     ┌────────────┴────────────┐      │
                     │                         │      │
              exit   ▼                   add   ▼      │
           ┌─────────────────┐    ┌─────────────────┐ │
           │  PENDING_EXIT   │    │  PENDING_ADD    │─┘
           └─────────────────┘    └─────────────────┘
                     │
                     │ filled
                     ▼
                ┌──────┐
                │ NONE │
                └──────┘
```

### States

| State | Description | allows_new_orders | is_pending |
|-------|-------------|-------------------|------------|
| `NONE` | No position, no pending orders | ✓ | |
| `PENDING_ENTRY` | Entry order placed, awaiting fill | | ✓ |
| `OPEN` | Position active | ✓ | |
| `PENDING_EXIT` | Exit order placed, awaiting fill | | ✓ |
| `PENDING_ADD` | Adding to position, awaiting fill | | ✓ |

### Valid Transitions

```python
VALID_POSITION_TRANSITIONS = {
    PositionState.NONE: {PositionState.PENDING_ENTRY},
    PositionState.PENDING_ENTRY: {PositionState.OPEN, PositionState.NONE},
    PositionState.OPEN: {PositionState.PENDING_EXIT, PositionState.PENDING_ADD, PositionState.NONE},
    PositionState.PENDING_EXIT: {PositionState.OPEN, PositionState.NONE},
    PositionState.PENDING_ADD: {PositionState.OPEN},
}
```

---

## State Ownership

| Component | State Owned | Responsibility |
|-----------|-------------|----------------|
| `OrderRegistry` | `TrackedOrder.status` | Local order cache, fast lookups |
| `PortfolioState` | `_position_states[symbol]` | Position lifecycle, validates transitions |
| `SymbolState` | `pending_order_id` | Links to active order |
| `Broker` | Order status | Source of truth for orders |

---

## Order Flow Sequence

```
┌─────────────┐   ┌──────────────────┐   ┌─────────────────┐   ┌─────────────┐
│ExecutionEng │   │  PortfolioState  │   │  OrderRegistry  │   │   Broker    │
└──────┬──────┘   └────────┬─────────┘   └────────┬────────┘   └──────┬──────┘
       │                   │                      │                   │
       │ 1. set_position_state(PENDING_ENTRY)     │                   │
       │──────────────────►│                      │                   │
       │                   │                      │                   │
       │ 2. register(order_id, symbol, side, qty) │                   │
       │─────────────────────────────────────────►│                   │
       │                   │                      │                   │
       │ 3. place_order(symbol, qty, side)        │                   │
       │───────────────────────────────────────────────────────────── ►│
       │                   │                      │                   │
       │◄──────────────────────────────────────────────────────────── │
       │                   │  4. order_id         │                   │
       │                   │                      │                   │
       │ 5. apply_fill_safe(symbol, side, qty, price)                 │
       │──────────────────►│                      │                   │
       │                   │                      │                   │
       │ 6. update_status(order_id, "filled")     │                   │
       │─────────────────────────────────────────►│                   │
       │                   │                      │                   │
       │ 7. set_position_state(OPEN)              │                   │
       │──────────────────►│                      │                   │
       │                   │                      │                   │
```

---

## Detailed State Transitions

### Entry Flow

```
Signal: BUY AAPL 100 shares

1. PRE-VALIDATION
   ├── Check: position_state == NONE or OPEN
   ├── Check: No conflicting orders in OrderRegistry
   └── Check: Buying power sufficient

2. SET PENDING STATE
   └── portfolio.set_position_state("AAPL", PENDING_ENTRY)

3. REGISTER ORDER
   └── order_registry.register(
         order_id="ABC123",
         symbol="AAPL",
         side="buy",
         qty=100,
         status="pending"
       )

4. EXECUTE
   └── broker.place_order("AAPL", 100, "buy")

5. ON SUCCESS
   ├── portfolio.apply_fill_safe("AAPL", "buy", 100, 150.25)
   ├── order_registry.update_status("ABC123", "filled", 100)
   └── portfolio.set_position_state("AAPL", OPEN)

5. ON FAILURE/REJECTION
   ├── order_registry.update_status("ABC123", "rejected")
   └── portfolio.set_position_state("AAPL", NONE)
```

### Exit Flow

```
Signal: SELL AAPL 100 shares (closing position)

1. PRE-VALIDATION
   ├── Check: position_state == OPEN
   ├── Check: Position exists with qty >= 100
   └── Check: No conflicting orders

2. SET PENDING STATE
   └── portfolio.set_position_state("AAPL", PENDING_EXIT)

3. REGISTER ORDER
   └── order_registry.register(
         order_id="DEF456",
         symbol="AAPL",
         side="sell",
         qty=100,
         status="pending"
       )

4. EXECUTE
   └── broker.place_order("AAPL", 100, "sell")

5. ON SUCCESS (full exit)
   ├── portfolio.apply_fill_safe("AAPL", "sell", 100, 152.50)
   ├── order_registry.update_status("DEF456", "filled", 100)
   └── portfolio.set_position_state("AAPL", NONE)

5. ON PARTIAL FILL
   ├── portfolio.apply_fill_safe("AAPL", "sell", 50, 152.50)
   ├── order_registry.update_status("DEF456", "partial", 50)
   └── portfolio.set_position_state("AAPL", OPEN)  # Still have shares
```

### Pyramid (Add to Position) Flow

```
Signal: BUY AAPL 50 shares (adding to existing long)

1. PRE-VALIDATION
   ├── Check: position_state == OPEN
   ├── Check: TradeGate.pyramid_layers < max_pyramid_layers
   └── Check: Buying power sufficient

2. SET PENDING STATE
   └── portfolio.set_position_state("AAPL", PENDING_ADD)

3. REGISTER & EXECUTE
   └── (same as entry flow)

4. ON SUCCESS
   ├── portfolio.apply_fill_safe("AAPL", "buy", 50, 151.00)
   ├── order_registry.update_status("GHI789", "filled", 50)
   ├── portfolio.set_position_state("AAPL", OPEN)
   └── trade_gate.increment_layer("AAPL")
```

---

## Error Recovery

### Order Rejection

```
Error: Broker rejected order

1. order_registry.update_status(order_id, "rejected")
2. portfolio.set_position_state(symbol, previous_state)
   ├── PENDING_ENTRY → NONE
   ├── PENDING_EXIT → OPEN
   └── PENDING_ADD → OPEN
3. Log error and emit alert
```

### Fill Mismatch

```
Error: Fill price/qty differs from expectation

1. StateReconciler detects mismatch on next sync
2. Corrects PortfolioState via sync_from_snapshot()
3. Logs discrepancy for audit
4. If severe: triggers HALT state
```

### Orphaned Orders

```
Error: Order in registry but not on broker

1. StateReconciler finds order not in broker open orders
2. If old (> 1 hour): remove from registry
3. If recent: query broker for order status
4. Update registry to match broker truth
```

---

## Concurrency Protection

### Per-Symbol Locking

```python
# PortfolioState uses asyncio.Lock
async with self._lock:
    self._position_states[symbol] = new_state
```

### OrderRegistry Locking

```python
# OrderRegistry uses asyncio.Lock
async with self._lock:
    self._orders[order_id] = order
```

### Blocked Actions by State

| Current State | Blocked Actions |
|--------------|-----------------|
| `PENDING_ENTRY` | New entry orders |
| `PENDING_EXIT` | New exit orders |
| `PENDING_ADD` | New add orders |

---

## Reconciliation

### Periodic Sync

```
Every 60 seconds (configurable):

1. StateReconciler.check_positions()
2. Compare PortfolioState.positions with broker.get_positions()
3. For mismatches:
   ├── Log discrepancy
   ├── Check OrderRegistry for pending fills
   └── Correct if no pending orders
```

### Full Sync (Startup)

```
On startup:

1. StateReconciler.full_sync()
2. broker.get_account_info()
3. portfolio.sync_from_snapshot(snapshot)
4. Clear OrderRegistry (stale from previous session)
5. Set all position_states based on positions:
   ├── qty != 0 → OPEN
   └── qty == 0 → NONE
```

---

## Component Files

| Component | File |
|-----------|------|
| OrderStatus enum | `core/enums.py:55` |
| PositionState enum | `core/enums.py:125` |
| OrderRegistry | `core/order_registry.py` |
| TrackedOrder | `core/order_registry.py:24` |
| PortfolioState | `core/logic/portfolio_state.py` |
| LiveExecutionEngine | `core/logic/live_execution_engine.py` |
| StateReconciler | `core/state_reconciler.py` |

---

## Anti-Patterns

1. **Never check order status via broker during hot path**
   - Use OrderRegistry for fast lookups
   - Broker queries are for reconciliation only

2. **Never modify position_states directly**
   - Always use `portfolio.set_position_state()` for validation

3. **Never skip pending state before order placement**
   - Prevents duplicate orders from race conditions

4. **Never assume fill == order qty**
   - Always handle partial fills

5. **Never delete OrderRegistry entries immediately**
   - Keep for reconciliation; cleanup via `cleanup_closed()`

---

## Verification Checklist

When reviewing order handling code, verify:

- [ ] `set_position_state(PENDING_*)` before `place_order()`
- [ ] `order_registry.register()` with order_id
- [ ] `apply_fill_safe()` after successful execution
- [ ] `update_status()` to registry after fill
- [ ] `set_position_state(OPEN|NONE)` after fill
- [ ] Error paths revert position_state properly
- [ ] Partial fills handled (don't assume full fill)
- [ ] Conflicting orders cancelled before new orders
