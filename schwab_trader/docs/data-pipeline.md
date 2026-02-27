# Data Pipeline

The Schwab Trader data pipeline manages historical and real-time market data from multiple sources.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data Pipeline                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │   Alpaca    │     │   Schwab    │     │    CSV      │        │
│  │    API      │     │    API      │     │   Files     │        │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘        │
│         │                   │                   │                │
│         └───────────────────┴───────────────────┘                │
│                             │                                    │
│                             ▼                                    │
│                  ┌─────────────────────┐                        │
│                  │  UnifiedDataPipeline │                        │
│                  └──────────┬──────────┘                        │
│                             │                                    │
│              ┌──────────────┼──────────────┐                    │
│              ▼              ▼              ▼                    │
│     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐         │
│     │  Raw Data    │ │  Processor   │ │    Cache     │         │
│     │   Storage    │ │  (ML/TA)     │ │   Manager    │         │
│     └──────────────┘ └──────────────┘ └──────────────┘         │
│              │              │              │                    │
│              ▼              ▼              ▼                    │
│     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐         │
│     │  JSON Files  │ │   SQLite     │ │  Cache JSON  │         │
│     └──────────────┘ └──────────────┘ └──────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### UnifiedDataPipeline

The main interface for all data operations.

```python
from core.unified_data_pipeline import UnifiedDataPipeline

pipeline = UnifiedDataPipeline(
    data_path="data/data_storage/proc_data",
    raw_data_path="data/data_storage/raw_data",
    store_to_db=True,
    store_to_files=True
)

# Update multiple symbols
results = await pipeline.update_symbols(
    symbols=['AAPL', 'MSFT'],
    days=30,
    source=None,  # Auto-select best source
    process_data=True
)
```

### HistoricalDataUpdater

Focused on Alpaca historical data:

```python
from core.historical_data_updater import HistoricalDataUpdater

updater = HistoricalDataUpdater(
    api_key="your_key",
    api_secret="your_secret",
    data_path="data/proc_data"
)

# Update single symbol
count = await updater.update_symbol('AAPL', days=30)

# Update multiple symbols concurrently
results = await updater.update_symbols(['AAPL', 'MSFT'], days=30)
```

## Data Sources

### Alpaca (Primary)

- 1-minute bars via REST API
- Real-time streaming via WebSocket
- Free tier: 200 requests/minute

```python
# Check Alpaca availability
from core.credential_validator import can_use_alpaca

if await can_use_alpaca():
    print("Alpaca is available")
```

### Schwab (Secondary)

- 1-minute bars via REST API
- OAuth2 token-based authentication
- Requires manual token refresh

```python
# Check Schwab availability
from core.credential_validator import can_use_schwab

if await can_use_schwab():
    print("Schwab is available")
```

### Auto-Selection

The pipeline automatically selects the best available source:

```python
# Source priority: Alpaca > Schwab
source = await pipeline._select_best_source()
# Returns: 'alpaca', 'schwab', or 'none'
```

## Storage

### File Storage

Data is stored as JSON files:

```
data/data_storage/
├── raw_data/           # Raw API responses
│   ├── AAPL_raw.json
│   └── MSFT_raw.json
└── proc_data/          # Processed data
    ├── proc_AAPL_file.json
    └── proc_MSFT_file.json
```

### Database Storage

Processed data is also stored in SQLite:

```python
# Get data from database
df = pipeline.get_data_from_db('AAPL')

# Get data from file
df = pipeline.get_data_from_file('AAPL')

# Get from either (prefers file)
df = pipeline.get_data('AAPL')
```

### Cache

A cache tracks data freshness:

```json
{
  "AAPL": {
    "last_update": "2024-01-15T16:00:00Z",
    "bar_count": 1234,
    "source": "alpaca"
  }
}
```

## Data Processing

Raw bars are processed through the ML pipeline:

1. **Feature Engineering**
   - Technical indicators (SMA, EMA, RSI, etc.)
   - Price transformations
   - Volume analysis

2. **Scaling**
   - StandardScaler normalization
   - MinMax scaling (optional)

3. **Feature Selection**
   - PCA dimensionality reduction (optional)

```python
from data.processor import Processor

# Process raw DataFrame
processor = Processor(stock='AAPL', frame=raw_df)
processed_df = processor.ml_process(
    sma_window=200,
    ema_window=50,
    scaling_method="standard",
    include_scaled=True,
    include_pca=False
)
```

## Data Freshness

Check if data needs updating:

```python
# Get freshness info
info = pipeline.get_cache_info('AAPL')
# {
#     'last_update': '2024-01-15T16:00:00Z',
#     'age_minutes': 45,
#     'is_stale': False,
#     'bar_count': 1234
# }

# List available symbols
symbols = pipeline.list_available_symbols()
```

## Smart Reprocessing

The pipeline intelligently avoids unnecessary reprocessing:

### How It Works

1. **Timestamp Comparison**: Compares raw data timestamp vs processed data timestamp
2. **Skip if Current**: If processed data is up-to-date with raw data, reprocessing is skipped
3. **Force Reprocess**: Use `force_reprocess=True` to override and force reprocessing

```python
# Check if processed data is current
is_current = pipeline._is_processed_data_current('AAPL')

# Get timestamps for comparison
raw_ts = pipeline._get_last_raw_timestamp('AAPL')    # Last bar in raw_AAPL_file.json
proc_ts = pipeline._get_last_processed_timestamp('AAPL')  # Last bar in proc_AAPL_file.json

# Force reprocessing even if current
await pipeline.update_symbols(['AAPL'], force_reprocess=True)
```

### Processing Behavior

When data is updated, the pipeline processes **ALL raw data**, not just new bars:

```
Fetch New Bars → Append to raw_data → Load ALL raw_data → Process ALL → Save to proc_data
```

This ensures:
- Technical indicators (SMA_200, etc.) are computed correctly from full history
- No data gaps or incorrect indicator values
- Consistent processed data regardless of when it was last updated

### Command Line

```bash
# Normal update (skips reprocessing if current)
python -m core.unified_data_pipeline --symbols AAPL MSFT --days 5

# Force reprocessing
python -m core.unified_data_pipeline --symbols AAPL MSFT --days 5 --force-reprocess
```

### Manual Reprocessing Tool

For rebuilding processed data without fetching new data:

```bash
# Reprocess specific symbols from raw data
python tools/reprocess_raw_data.py --symbols AAPL MSFT NVDA

# Reprocess all symbols with raw data files
python tools/reprocess_raw_data.py --all
```

## Periodic Updates

Schedule automatic updates:

```python
# Run updates every 60 minutes
await pipeline.run_periodic(
    symbols=['AAPL', 'MSFT'],
    interval_minutes=60,
    days=5
)
```

## API Reference

### UnifiedDataPipeline

```python
class UnifiedDataPipeline:
    def __init__(
        self,
        data_path: Optional[str] = None,
        raw_data_path: Optional[str] = None,
        db_name: str = "stock_base.db",
        store_to_db: bool = True,
        store_to_files: bool = True
    )

    async def update_symbols(
        self,
        symbols: List[str],
        days: int = 30,
        source: Optional[str] = None,
        force_full: bool = False,
        process_data: bool = True
    ) -> Dict[str, int]

    async def check_sources(self) -> Dict[str, Any]

    def get_data(self, symbol: str) -> pd.DataFrame

    def get_data_from_db(self, symbol: str) -> pd.DataFrame

    def get_data_from_file(self, symbol: str) -> pd.DataFrame

    def list_available_symbols(self) -> List[str]

    def get_cache_info(self, symbol: str) -> Optional[Dict[str, Any]]
```

### HistoricalDataUpdater

```python
class HistoricalDataUpdater:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        data_path: Optional[str] = None,
        timeframe: TimeFrame = TimeFrame.Minute
    )

    async def update_symbol(
        self,
        symbol: str,
        days: int = 30,
        force_full: bool = False
    ) -> int

    async def update_symbols(
        self,
        symbols: List[str],
        days: int = 30,
        max_concurrent: int = 5
    ) -> Dict[str, int]

    def get_data_freshness(self, symbol: str) -> Optional[Dict[str, Any]]
```

## Configuration

### trading_config.json

```json
{
  "data": {
    "default_source": "alpaca",
    "update_days": 30,
    "store_to_db": true,
    "store_to_files": true,
    "stale_threshold_hours": 1
  }
}
```

## Logging

Data pipeline logs to `logs/unified_pipeline.log`:

```
2024-01-15 09:30:00 INFO [UnifiedDataPipeline] Update started for ['AAPL', 'MSFT']
2024-01-15 09:30:01 INFO [UnifiedDataPipeline] Source: alpaca
2024-01-15 09:30:05 INFO [UnifiedDataPipeline] [AAPL] Fetched 1234 bars
2024-01-15 09:30:06 INFO [UnifiedDataPipeline] [AAPL] Processed and saved
2024-01-15 09:30:10 INFO [UnifiedDataPipeline] Update complete: 2/2 success
```

## Best Practices

1. **Use auto-selection for source**
   ```python
   await pipeline.update_symbols(symbols, source=None)
   ```

2. **Enable both storage options**
   ```python
   pipeline = UnifiedDataPipeline(store_to_db=True, store_to_files=True)
   ```

3. **Check freshness before trading**
   ```python
   info = pipeline.get_cache_info('AAPL')
   if info and info['is_stale']:
       await pipeline.update_symbols(['AAPL'])
   ```

4. **Use incremental updates**
   ```python
   await updater.update_symbol('AAPL', days=5, force_full=False)
   ```

## Troubleshooting

### "No data source available"

1. Check credentials: `python preflight.py -v`
2. Verify API keys in `.env`
3. Check network connectivity

### "Failed to fetch bars"

1. Check API rate limits
2. Verify symbol is valid
3. Check market hours (data may not be available outside hours)

### "Processing failed"

1. Check data format from source
2. Verify Processor configuration
3. Check for missing required columns

## Related Documentation

- [Pre-Flight Checks](preflight.md)
- [Configuration Guide](configuration.md)
- [Architecture](architecture.md#data-layer)
