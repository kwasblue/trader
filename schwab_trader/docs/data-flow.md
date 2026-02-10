# Data Flow Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA SOURCES                                    │
│                                                                             │
│   ┌─────────────┐              ┌─────────────┐                             │
│   │   ALPACA    │              │   SCHWAB    │                             │
│   │  (Preferred)│              │ (Fallback)  │                             │
│   │             │              │             │                             │
│   │ - Free tier │              │ - OAuth req │                             │
│   │ - No expiry │              │ - Token exp │                             │
│   └──────┬──────┘              └──────┬──────┘                             │
│          │                            │                                     │
│          └────────────┬───────────────┘                                     │
│                       ▼                                                     │
│            ┌─────────────────────┐                                         │
│            │ UnifiedDataPipeline │  ◄── Auto-selects best source           │
│            │ (core/unified_data_ │      Validates credentials              │
│            │  pipeline.py)       │      Handles fallback                   │
│            └──────────┬──────────┘                                         │
└───────────────────────┼─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              STORAGE LAYER                                   │
│                                                                             │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐         │
│   │   RAW DATA      │   │ PROCESSED DATA  │   │    DATABASE     │         │
│   │                 │   │                 │   │                 │         │
│   │ raw_{SYM}_file  │   │ proc_{SYM}_file │   │  stock_base.db  │         │
│   │     .json       │   │     .json       │   │  (SQLite)       │         │
│   │                 │   │                 │   │                 │         │
│   │ data/data_      │   │ data/data_      │   │ stock_table     │         │
│   │ storage/raw_data│   │ storage/proc_   │   │                 │         │
│   │                 │   │ data            │   │                 │         │
│   └────────┬────────┘   └────────┬────────┘   └────────┬────────┘         │
│            │                     │                     │                   │
│            │    ┌────────────────┴─────────────────────┘                   │
│            │    │                                                          │
│            │    ▼                                                          │
│            │   ┌─────────────────┐                                         │
│            │   │  CACHE MANAGER  │                                         │
│            │   │                 │                                         │
│            │   │ cache/system_   │  ◄── Tracks last processed dates       │
│            │   │ cache.json      │      Enables incremental updates        │
│            │   └─────────────────┘                                         │
│            │                                                               │
└────────────┼───────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PROCESSING PIPELINE                              │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      Processor (data/processor.py)                   │  │
│   │                                                                      │  │
│   │  1. clean_stock_data()     - Normalize columns, handle NaN          │  │
│   │  2. apply_indicators()     - SMA, EMA, RSI, MACD, Bollinger, ATR   │  │
│   │  3. feature_engineering()  - Returns, momentum, volatility features │  │
│   │  4. scale_features()       - StandardScaler or MinMaxScaler        │  │
│   │  5. (optional) PCA         - Dimensionality reduction               │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRADING LAYER                                   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │              HistoricalBarLoader (core/historical_loader.py)         │  │
│   │                                                                      │  │
│   │  - Reads proc_{SYM}_file.json                                       │  │
│   │  - Normalizes bar format (timestamp, OHLCV)                         │  │
│   │  - Provides load_last_n_bars(symbol, n)                             │  │
│   └──────────────────────────────┬──────────────────────────────────────┘  │
│                                  │                                          │
│                                  ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    Runner (Alpaca/Schwab)                            │  │
│   │                                                                      │  │
│   │  STARTUP (seed):                                                    │  │
│   │    1. Check data freshness via cache                                │  │
│   │    2. If stale → UnifiedDataPipeline.update_symbols()              │  │
│   │    3. Load bars via HistoricalBarLoader                            │  │
│   │    4. Build history buffer for each symbol                         │  │
│   │    5. Compute initial ATR                                          │  │
│   │                                                                      │  │
│   │  LIVE TRADING (on_bar callback):                                    │  │
│   │    1. Receive bar from broker websocket                            │  │
│   │    2. Append to history buffer                                      │  │
│   │    3. Update portfolio mark-to-market                              │  │
│   │    4. Compute ATR & classify regime                                │  │
│   │    5. Run strategy → generate signal                               │  │
│   │    6. Execute via execution engine                                  │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Fetch Flow (Historical)

### 1. Source Selection
```python
# UnifiedDataPipeline._select_best_source()
1. CredentialValidator.validate_all()
2. Check Alpaca: API key valid? → can_fetch_data=True
3. Check Schwab: Token valid & not expired? → can_fetch_data=True
4. Return 'alpaca' (preferred) or 'schwab' (fallback) or 'none'
```

### 2. Fetching Data

**Alpaca Path:**
```python
# _fetch_alpaca(symbol, days)
StockHistoricalDataClient.get_stock_bars(
    symbol=symbol,
    timeframe=TimeFrame.Day,
    start=now - days,
    feed=DataFeed.IEX
)
→ Returns list of {datetime, open, high, low, close, volume}
```

**Schwab Path:**
```python
# _fetch_schwab(symbol, days)
SchwabClient.daily_price_history(symbol, start=epoch_ms)
→ Returns {'candles': [{datetime, open, high, low, close, volume}, ...]}
```

### 3. Storage

```
Raw bars → _save_raw_data()
         → data/data_storage/raw_data/raw_{SYMBOL}_file.json

         → _process_and_save()
             → Processor.ml_process()
                 → clean_stock_data()      # normalize
                 → apply_indicators()       # SMA, EMA, RSI, etc.
                 → feature_engineering()    # returns, momentum
                 → scale_features()         # optional normalization

             → _save_processed_data_file()
             → data/data_storage/proc_data/proc_{SYMBOL}_file.json

             → _save_processed_data_db()
             → stock_base.db (stock_table)

             → _update_cache()
             → cache/system_cache.json
```

---

## Data Retrieval Flow (Trading)

### At Startup (seed)

```python
# Runner.seed(lookback_bars=200)

1. Check freshness:
   data_updater.get_data_freshness(symbol)
   → Reads cache/system_cache.json
   → Returns {age_minutes, bar_count, last_date}

2. If stale (age > max_stale_minutes):
   data_pipeline.update_symbols(symbols, days=30)
   → Fetches fresh data from Alpaca/Schwab
   → Processes through full pipeline
   → Saves to files + DB + cache

3. Load processed data:
   HistoricalBarLoader.load_last_n_bars(symbol, n=200)
   → Reads proc_{SYMBOL}_file.json
   → Normalizes to {timestamp, symbol, Open, High, Low, Close, Volume}

4. Build history buffer:
   for bar in bars:
       self.history[symbol].append(bar)

5. Compute initial ATR:
   df = self._df_from_history(symbol)
   atr = compute_atr(df, period=14)
```

### During Live Trading

```python
# Runner.on_alpaca_bar(raw_bar) or on_schwab_quote(quote)

1. Receive live bar from websocket:
   bar = _canonicalize_bar(raw_bar)
   → {timestamp, symbol, Open, High, Low, Close, Volume}

2. Emit event for GUI:
   event_handler.emit("BAR", bar)

3. Update history:
   self.history[symbol].append(bar)

4. Mark-to-market:
   portfolio.update_price(symbol, bar.Close)

5. Compute indicators:
   df = self._df_from_history(symbol)
   atr = compute_atr(df, period=14)
   regime = classify_regime(atr, atr_history)

6. Strategy evaluation:
   signal = strategy.generate_signal(df)
   → Returns -1 (sell), 0 (hold), or 1 (buy)

7. Execution:
   if signal != 0:
       execution_engine.process_signal(signal, context)
```

---

## File Locations

| File | Purpose | Format |
|------|---------|--------|
| `data/data_storage/raw_data/raw_{SYM}_file.json` | Raw OHLCV from API | `{candles: [{datetime, o, h, l, c, v}, ...]}` |
| `data/data_storage/proc_data/proc_{SYM}_file.json` | Processed with indicators | `{bars: [{Date, Open, High, Low, Close, Volume, SMA_200, EMA_50, RSI, ...}]}` |
| `stock_base.db` | SQLite database | Table: `stock_table` |
| `cache/system_cache.json` | Processing timestamps | `{stock_files: {SYM: last_epoch_ms}}` |

---

## Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `UnifiedDataPipeline` | `core/unified_data_pipeline.py` | Fetch, process, store historical data |
| `CredentialValidator` | `core/credential_validator.py` | Check API credentials & token status |
| `Processor` | `data/processor.py` | Clean data, add indicators & features |
| `HistoricalBarLoader` | `core/historical_loader.py` | Load processed data for trading |
| `DataStore` | `data/datastorage.py` | SQLite database interface |
| `CacheManager` | `cache/cache.py` | Track processing timestamps |
| `AlpacaLiveRunner` | `core/alpaca_runner.py` | Live trading with Alpaca |
| `SchwabLiveRunner` | `core/schwab_runner.py` | Live trading with Schwab |

---

## Data Freshness Logic

```python
# When is data considered "stale"?

max_stale_minutes = 60  # configurable

freshness = data_updater.get_data_freshness(symbol)
# Returns:
# {
#   'age_minutes': 45,
#   'bar_count': 200,
#   'last_date': '2024-01-15T16:00:00Z'
# }

if freshness['age_minutes'] > max_stale_minutes:
    # Fetch fresh data
    await data_pipeline.update_symbols([symbol])
```

---

## Indicators Added by Processor

The `Processor.ml_process()` adds these columns:

**Moving Averages:**
- `SMA_200` - 200-period Simple Moving Average
- `EMA_50` - 50-period Exponential Moving Average

**Momentum:**
- `RSI` - Relative Strength Index (14-period)
- `MACD`, `MACD_Signal`, `MACD_Hist` - MACD indicator

**Volatility:**
- `ATR` - Average True Range
- `BB_Upper`, `BB_Middle`, `BB_Lower` - Bollinger Bands

**Engineered Features:**
- `Return` - Daily return
- `Log_Return` - Log return
- `Momentum_5`, `Momentum_10` - N-day momentum
- `Volatility_5`, `Volatility_10` - N-day volatility
- Various ratios and normalized values
