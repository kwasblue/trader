# Data Pipeline Architecture

This document defines the official data ingestion and read paths.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA SOURCES                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                       │
│  │  Alpaca  │  │  Schwab  │  │ (future) │                       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                       │
│       │             │             │                              │
│       └─────────────┴─────────────┘                              │
│                     │                                            │
│                     ▼                                            │
│         ┌─────────────────────┐                                  │
│         │ UnifiedDataPipeline │  ◄── PRIMARY INGESTION          │
│         │  (auto source       │                                  │
│         │   selection +       │                                  │
│         │   failover)         │                                  │
│         └──────────┬──────────┘                                  │
│                    │                                             │
│         ┌──────────┼──────────┐                                  │
│         ▼          ▼          ▼                                  │
│   ┌──────────┐ ┌────────┐ ┌────────┐                            │
│   │ Processor│ │  JSON  │ │ SQLite │                            │
│   │(features)│ │ Files  │ │   DB   │                            │
│   └──────────┘ └────────┘ └────────┘                            │
│                    │                                             │
│                    ▼                                             │
│         ┌─────────────────────┐                                  │
│         │ HistoricalBarLoader │  ◄── PRIMARY READ PATH          │
│         │  (reads JSON files) │                                  │
│         └─────────────────────┘                                  │
│                    │                                             │
│                    ▼                                             │
│         ┌─────────────────────┐                                  │
│         │   Trading System    │                                  │
│         │  (runners, GUI,     │                                  │
│         │   backtests)        │                                  │
│         └─────────────────────┘                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Official Components

### PRIMARY: UnifiedDataPipeline
**File:** `core/unified_data_pipeline.py`
**Role:** Canonical ingestion path for all historical data.

**Features:**
- Auto-selects Alpaca or Schwab based on credential availability
- Automatic failover between sources
- Full processing pipeline (indicators, features)
- Stores to JSON files AND SQLite
- Async operation

**Usage:**
```python
from core.unified_data_pipeline import UnifiedDataPipeline

pipeline = UnifiedDataPipeline()

# Update symbols (auto-select source)
results = await pipeline.update_symbols(['AAPL', 'MSFT'], days=30)

# Force specific source
results = await pipeline.update_symbols(['AAPL'], source='alpaca')

# Run as periodic background task
await pipeline.run_periodic(symbols=['AAPL'], interval_minutes=60)
```

---

### PRIMARY: HistoricalBarLoader
**File:** `core/historical_loader.py`
**Role:** Canonical read path for historical bars.

**Features:**
- Reads `proc_{SYMBOL}_file.json` files
- Normalizes to standard bar format
- Falls back to raw data if processed unavailable

**Usage:**
```python
from core.historical_loader import HistoricalBarLoader

loader = HistoricalBarLoader(path="data/data_storage/proc_data")
bars = loader.load_last_n_bars("AAPL", n=200)
```

---

### UTILITY: Processor
**File:** `data/processor.py`
**Role:** Data cleaning and feature engineering.

**Features:**
- Technical indicator calculation
- Feature engineering (PCA, scaling)
- FFT denoising
- Used internally by UnifiedDataPipeline

---

### UTILITY: DataStore
**File:** `data/datastorage.py`
**Role:** SQLite storage backend.

**Features:**
- Thread-safe SQLite operations
- Used internally by UnifiedDataPipeline
- Can be used directly for SQL queries

---

## Legacy/Deprecated Components

### DEPRECATED: Aggregator
**File:** `data/aggregate.py`
**Status:** DEPRECATED - Use UnifiedDataPipeline instead.

```python
# OLD (deprecated)
from data.aggregate import Aggregator
aggregator = Aggregator(apikey, secret)

# NEW
from core.unified_data_pipeline import UnifiedDataPipeline
pipeline = UnifiedDataPipeline()
```

---

### LEGACY: HistoricalDataUpdater
**File:** `core/historical_data_updater.py`
**Status:** LEGACY - Alpaca-specific updater.

**Note:** This provides Alpaca-specific functionality used by `BaseLiveRunner.seed()`.
Consider consolidating into UnifiedDataPipeline in future refactor.

**Current usage:**
- `base_live_runner.py` - For seeding warmup data
- `preflight.py` - For data freshness checks

---

## Storage Locations

| Type | Location | Format |
|------|----------|--------|
| Processed data | `data/data_storage/proc_data/proc_{SYMBOL}_file.json` | JSON |
| Raw data | `data/data_storage/raw_data/raw_{SYMBOL}_file.json` | JSON |
| SQLite DB | `data/stock_base.db` | SQLite |
| Cache metadata | `cache/system_cache.json` | JSON |

---

## Data Flow by Use Case

### Live Trading (Alpaca)
```
1. BaseLiveRunner.seed()
   → HistoricalDataUpdater.update_symbols()
   → HistoricalBarLoader.load_last_n_bars()
2. Streaming bars from Alpaca WebSocket
```

### Live Trading (Schwab)
```
1. BaseLiveRunner.seed()
   → UnifiedDataPipeline.update_symbols(source='schwab')
   → HistoricalBarLoader.load_last_n_bars()
2. Streaming quotes aggregated to bars
```

### Backtesting
```
1. UnifiedDataPipeline.update_symbols() (if stale)
2. HistoricalBarLoader.load_last_n_bars()
3. Strategy execution on historical bars
```

### GUI Simulation
```
1. HistoricalBarLoader.load_last_n_bars()
2. GBM simulation generates synthetic bars
```

### Preflight Checks
```
1. Check data freshness via cache metadata
2. UnifiedDataPipeline.update_symbols() if stale
```

---

## Future Consolidation

**Goal:** Merge `HistoricalDataUpdater` into `UnifiedDataPipeline`

**Blockers:**
- `HistoricalDataUpdater` has Alpaca-specific streaming integration
- `BaseLiveRunner.seed()` depends on it

**Plan:**
1. Add Alpaca-specific methods to UnifiedDataPipeline
2. Update BaseLiveRunner to use UnifiedDataPipeline
3. Deprecate HistoricalDataUpdater
4. Remove after migration

---

## Module Status Summary

| Module | Status | Action |
|--------|--------|--------|
| `core/unified_data_pipeline.py` | PRIMARY | Use for ingestion |
| `core/historical_loader.py` | PRIMARY | Use for reading |
| `data/processor.py` | UTILITY | Internal use |
| `data/datastorage.py` | UTILITY | Internal use |
| `data/aggregate.py` | DEPRECATED | Do not use |
| `core/historical_data_updater.py` | LEGACY | Plan migration |
