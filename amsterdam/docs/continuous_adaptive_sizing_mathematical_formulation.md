# Continuous Adaptive Position Sizing: Mathematical Formulation

**Version:** 2.1.0
**Date:** March 2026
**Author:** Trading System Architecture
**Status:** Production-Ready Sizing Component

---

## Abstract

This document provides the **corrected and complete** mathematical formulation for the Continuous Adaptive Position Sizing system. The system dynamically adjusts position sizes based on:

1. **Market Volatility** (ATR percentile) - inverted score favoring low volatility
2. **Strategy Performance** (Rolling Sharpe ratio) - with Bayesian shrinkage and exponential smoothing

**Key Formula:**

$$C_{\text{final}} = \text{clamp}\left(\alpha \cdot \sqrt{V \times P}, \ C_{\text{floor}}, \ C_{\text{ceiling}}\right)$$

where:
- $V$ = volatility score $\in [0, 1]$ (inverted ATR percentile)
- $P$ = performance score $\in [0, 1]$ (normalized Sharpe with shrinkage & smoothing)
- $\alpha$ = scaling constant (allows leverage, default 1.5)
- Geometric mean $\sqrt{V \times P}$ ensures **both** metrics must be favorable
- Floor/ceiling bounds prevent extreme sizing

**Scope and Production Deployment:**

This module outputs **candidate position multipliers** for individual positions. Production deployment requires integration with portfolio-level controls:
- Portfolio exposure caps
- Sector/correlation limits
- Liquidity checks
- Account drawdown overrides
- Gross/net exposure management
- Max concurrent positions limits

This is a **production-ready sizing component**, not a standalone trading system.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Volatility Measurement](#2-volatility-measurement)
3. [Performance Measurement](#3-performance-measurement)
4. [Bayesian Shrinkage](#4-bayesian-shrinkage)
5. [Exponential Smoothing](#5-exponential-smoothing)
6. [Risk Score Aggregation](#6-risk-score-aggregation)
7. [Position Size Calculation](#7-position-size-calculation)
8. [Trade Filtering (Skip Logic)](#8-trade-filtering-skip-logic)
9. [Parameter Configuration](#9-parameter-configuration)
10. [Complete Algorithm](#10-complete-algorithm)
11. [Worked Examples](#11-worked-examples)
12. [Mathematical Properties](#12-mathematical-properties)
13. [Implementation Notes](#13-implementation-notes)

---

## 0. Formal Assumptions and Scope

### 0.1 Scope

This module is a **position sizing component**, not a complete trading system. It assumes:

**What this module does:**
- Calculates adaptive position multipliers for individual positions
- Adjusts for current market volatility (ATR percentile)
- Adjusts for recent strategy performance (Sharpe ratio)
- Provides skip recommendations for very unfavorable conditions

**What this module does NOT do:**
- Portfolio-level exposure management
- Correlation or sector concentration limits
- Liquidity or slippage analysis
- Broker constraint enforcement
- Account drawdown monitoring
- Event risk filtering
- Max concurrent positions management

### 0.2 Required External Controls

Production deployment requires integration with:

1. **Portfolio risk manager:** Enforce total exposure caps, sector limits, correlation constraints
2. **Liquidity manager:** Check available liquidity, estimate slippage, enforce max % of ADV
3. **Account monitor:** Track account equity, enforce drawdown limits, margin requirements
4. **Event filter:** Block trades during earnings, Fed announcements, known catalysts
5. **Broker interface:** Enforce position limits, check buying power, handle rejections

### 0.3 Data Quality Assumptions

The system assumes:

- **Price data:** Clean OHLCV bars with no gaps, outliers filtered
- **Trade history:** Accurate PnL, entry/exit times, return percentages
- **Timestamp consistency:** All trades timestamped in same timezone
- **Lookback availability:** Sufficient bars for ATR calculation (≥264 for 250-bar lookback + 14-bar period)

### 0.4 Statistical Assumptions

- **Rolling Sharpe:** Recent performance has *some* predictive value (weak EMH violation)
- **ATR percentile:** Historical volatility distribution is *somewhat* stationary
- **Bayesian prior:** Expected baseline Sharpe ≈ 0.5 for typical strategies (calibrate to your system)
- **Trade independence:** Returns are not perfectly IID, but shrinkage + smoothing mitigate this

### 0.5 Operational Assumptions

- **State persistence:** Smoothed Sharpe history is persisted across restarts
- **Update frequency:** Metrics updated on trade close, not intra-bar
- **Single-threaded:** No concurrent writes to smoothed history (add locking if multi-threaded)

---

## 1. System Overview

### 1.1 Position Sizing Formula

Final position size is computed as:

$$\text{Position}_{\text{final}} = \text{Capital} \times \text{MaxPct} \times C_{\text{final}}$$

where $C_{\text{final}}$ is the continuous adaptive multiplier derived from market conditions.

### 1.2 Multiplier Calculation Pipeline

```
Raw Metrics (ATR, Sharpe)
    ↓
Normalization (V_score, P_score) ∈ [0,1]
    ↓
Bayesian Shrinkage (P_score) - reduces noise
    ↓
Exponential Smoothing (P_score) - prevents twitchiness
    ↓
Geometric Mean: C_raw = √(V × P) ∈ [0,1]
    ↓
Skip Check: if C_raw < threshold → skip trade
    ↓
Scaling: C_scaled = α × C_raw (allows >1.0)
    ↓
Bounds: C_final = clamp(C_scaled, floor, ceiling)
```

### 1.3 Key Design Principles

1. **Geometric Mean** - Conservative aggregation (both V and P must be good)
2. **Bayesian Shrinkage** - Blend sample with prior to reduce noise when n is small
3. **Exponential Smoothing** - Prevent rapid oscillation in position sizes
4. **Skip-Before-Floor** - Check raw score to avoid unreachable skip threshold
5. **Scaling Constant** - Allow leverage while maintaining safety bounds

---

## 2. Volatility Measurement

### 2.1 Average True Range (ATR)

The ATR measures intraday price volatility over period $n$:

$$\text{ATR}_n = \frac{1}{n} \sum_{i=1}^{n} \text{TR}_i$$

where True Range at bar $i$ is:

$$\text{TR}_i = \max\left(H_i - L_i, \left|H_i - C_{i-1}\right|, \left|L_i - C_{i-1}\right|\right)$$

**Parameters:**
- $n = 14$ (standard ATR period)
- $H_i, L_i, C_i$ = High, Low, Close at bar $i$

### 2.2 ATR Percentile Rank

To contextualize current volatility against historical distribution:

$$\text{ATR}_{\text{pct}} = \frac{\sum_{j=1}^{N_{\text{hist}}} \mathbb{1}[\text{ATR}_j < \text{ATR}_{\text{curr}}]}{N_{\text{hist}}} \times 100$$

where:
- $\text{ATR}_{\text{curr}}$ = Current ATR value
- $N_{\text{hist}} = 250$ historical bars
- $\mathbb{1}[\cdot]$ = indicator function (1 if true, 0 otherwise)

**Interpretation:**
- $\text{ATR}_{\text{pct}} = 10$ → current volatility is lower than 10% of history (very calm)
- $\text{ATR}_{\text{pct}} = 90$ → current volatility is higher than 90% of history (very volatile)

### 2.3 Volatility Score (Inverted)

Since **lower volatility** is preferable for larger positions:

$$V_{\text{score}} = 1 - \frac{\text{ATR}_{\text{pct}}}{100} = \frac{100 - \text{ATR}_{\text{pct}}}{100}$$

**Properties:**
- $V_{\text{score}} \in [0, 1]$
- Low volatility (ATR$_{\text{pct}} = 0$) → $V_{\text{score}} = 1.0$ ✓ (favorable)
- High volatility (ATR$_{\text{pct}} = 100$) → $V_{\text{score}} = 0.0$ ✗ (unfavorable)

---

## 3. Performance Measurement

### 3.1 Sample Sharpe Ratio (Raw)

From the last $N$ trades (within lookback window $W$ days):

$$S_{\text{sample}} = \frac{\bar{R}}{\sigma_R} \times \sqrt{T}$$

where:
- $\bar{R} = \frac{1}{N}\sum_{i=1}^{N} R_i$ (mean return %)
- $\sigma_R = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(R_i - \bar{R})^2}$ (std dev %)
- $T = 252$ (annualization factor - trading days per year)
- $N \geq N_{\text{min}} = 10$ (minimum trades required)

**Issue with Sample Sharpe:**
- Noisy when $N$ is small (e.g., 10-15 trades)
- Can wildly overestimate/underestimate true performance
- **Solution:** Bayesian shrinkage (see next section)

---

## 4. Bayesian Shrinkage

### 4.1 Motivation

When trade count $N$ is low, sample Sharpe is unreliable. We blend it with a **prior belief** about expected performance.

### 4.2 Formula

$$S_{\text{shrunk}} = \frac{N \cdot S_{\text{sample}} + N_{\text{prior}} \cdot S_{\text{prior}}}{N + N_{\text{prior}}}$$

**Parameters:**
- $S_{\text{prior}} = 0.5$ (prior belief: moderately positive Sharpe)
- $N_{\text{prior}} = 5$ (prior weight: equivalent to 5 trades of belief)
- $N$ = actual number of recent trades

### 4.3 Behavior

| Scenario | $N$ | Weight on Sample | Weight on Prior | Result |
|----------|-----|------------------|-----------------|--------|
| Cold start | 10 | 10/15 = 67% | 5/15 = 33% | Modest shrinkage |
| Warmed up | 30 | 30/35 = 86% | 5/35 = 14% | Light shrinkage |
| Mature | 100 | 100/105 = 95% | 5/105 = 5% | Minimal shrinkage |

**Effect:** Prevents overconfidence from lucky streaks when $N$ is small.

### 4.4 Example

- Sample Sharpe = 2.5 (from 10 trades - likely noisy)
- Prior Sharpe = 0.5
- Shrunk Sharpe = $(10 \times 2.5 + 5 \times 0.5) / 15 = 27.5 / 15 = 1.83$

The shrunk value is more conservative and realistic.

---

## 5. Exponential Smoothing

### 5.1 Motivation

Even after shrinkage, Sharpe can fluctuate as new trades arrive. To prevent rapid position size oscillation, we apply **exponential moving average (EMA)**.

### 5.2 Formula

$$S_{\text{smoothed}}^{(t)} = \beta \cdot S_{\text{shrunk}}^{(t)} + (1 - \beta) \cdot S_{\text{smoothed}}^{(t-1)}$$

where:
- $\beta = 0.3$ (smoothing parameter)
- $S_{\text{shrunk}}^{(t)}$ = current shrunk Sharpe
- $S_{\text{smoothed}}^{(t-1)}$ = previous smoothed value
- First value: $S_{\text{smoothed}}^{(0)} = S_{\text{shrunk}}^{(0)}$ (no history)

### 5.3 Behavior

| $\beta$ | New Weight | Old Weight | Behavior |
|---------|------------|------------|----------|
| 0.1 | 10% | 90% | Very smooth, slow to adapt |
| 0.3 | 30% | 70% | Balanced (default) |
| 0.5 | 50% | 50% | Moderate smoothing |
| 1.0 | 100% | 0% | No smoothing |

**Default $\beta = 0.3$** balances responsiveness with stability.

### 5.4 Example

- Previous smoothed Sharpe = 1.2
- New shrunk Sharpe = 1.8
- New smoothed Sharpe = $0.3 \times 1.8 + 0.7 \times 1.2 = 0.54 + 0.84 = 1.38$

Position size adjusts gradually, not abruptly.

---

## 6. Risk Score Aggregation

### 6.1 Performance Score (Normalized)

After shrinkage and smoothing, normalize to $[0, 1]$:

$$P_{\text{score}} = \begin{cases}
0 & \text{if } S_{\text{smoothed}} \leq 0 \\
\min\left(1, \frac{S_{\text{smoothed}}}{S_{\text{max}}}\right) & \text{if } S_{\text{smoothed}} > 0
\end{cases}$$

where $S_{\text{max}} = 2.0$ (Sharpe above this is clamped to 1.0).

**Properties:**
- $P_{\text{score}} \in [0, 1]$
- Negative Sharpe → 0 (no confidence)
- Sharpe = 1.0 → $P_{\text{score}} = 0.5$
- Sharpe ≥ 2.0 → $P_{\text{score}} = 1.0$ (excellent)

### 6.2 Combined Raw Score (Geometric Mean)

$$C_{\text{raw}} = \sqrt{V_{\text{score}} \times P_{\text{score}}}$$

**Why Geometric Mean?**

- **Conservative:** Both $V$ and $P$ must be good
- **Penalizes imbalance:** One low score drags down result
- **Range:** $C_{\text{raw}} \in [0, 1]$ (since both inputs $\in [0, 1]$)

### 6.3 Comparison: Geometric vs Arithmetic

| Scenario | $V_{\text{score}}$ | $P_{\text{score}}$ | Geometric | Arithmetic |
|----------|------------|------------|-----------|------------|
| Both good | 0.8 | 0.8 | 0.80 | 0.80 |
| Balanced | 0.6 | 0.6 | 0.60 | 0.60 |
| Imbalanced | 0.9 | 0.3 | 0.52 | 0.60 |
| One poor | 0.8 | 0.1 | 0.28 | 0.45 |

Geometric mean is **more conservative** when metrics disagree.

---

## 7. Position Size Calculation

### 7.1 Complete Formula

$$C_{\text{final}} = \text{clamp}\left(\alpha \cdot C_{\text{raw}}, \ C_{\text{floor}}, \ C_{\text{ceiling}}\right)$$

where:

$$C_{\text{scaled}} = \alpha \cdot C_{\text{raw}}$$

$$C_{\text{final}} = \begin{cases}
C_{\text{floor}} & \text{if } C_{\text{scaled}} < C_{\text{floor}} \\
C_{\text{ceiling}} & \text{if } C_{\text{scaled}} > C_{\text{ceiling}} \\
C_{\text{scaled}} & \text{otherwise}
\end{cases}$$

**Default Parameters:**
- $\alpha = 1.5$ (scaling constant - allows leverage)
- $C_{\text{floor}} = 0.30$ (minimum 30% of base position)
- $C_{\text{ceiling}} = 1.50$ (maximum 150% of base position)

### 7.2 Why Scaling Constant $\alpha > 1$?

Without scaling ($\alpha = 1$):
- $C_{\text{raw}} \in [0, 1]$
- Maximum multiplier = 1.0 (no leverage)

With scaling ($\alpha = 1.5$):
- $C_{\text{scaled}} \in [0, 1.5]$
- Excellent conditions (raw score ≈ 1.0) → $1.5 \times 1.0 = 1.5$ (50% leverage)
- Still bounded by ceiling for safety

**This allows the system to increase position size beyond baseline when conditions are favorable.**

### 7.3 Reachable Range

With $\alpha = 1.5$, $C_{\text{floor}} = 0.30$, $C_{\text{ceiling}} = 1.50$:

| $C_{\text{raw}}$ | $C_{\text{scaled}}$ | $C_{\text{final}}$ | Outcome |
|----------|-------------|-------------|---------|
| 0.0 | 0.0 | 0.30 | Floor |
| 0.2 | 0.3 | 0.30 | Floor |
| 0.3 | 0.45 | 0.45 | Scaled |
| 0.5 | 0.75 | 0.75 | Scaled |
| 0.7 | 1.05 | 1.05 | Scaled |
| 0.9 | 1.35 | 1.35 | Scaled |
| 1.0 | 1.5 | 1.50 | Ceiling |

**Conclusion:** The full range $[0.30, 1.50]$ **is reachable** with $\alpha = 1.5$.

---

## 8. Trade Filtering (Skip Logic)

### 8.1 Skip Threshold (Corrected)

**Critical Fix:** Check raw score **BEFORE** applying floor.

$$\text{Skip} = \begin{cases}
\text{True} & \text{if } C_{\text{raw}} < C_{\text{skip}} \\
\text{False} & \text{otherwise}
\end{cases}$$

where $C_{\text{skip}} = 0.05$ (skip threshold).

### 8.2 Why Check Raw Score?

**Bug in v1.0:** Checked final multiplier after flooring
- $C_{\text{floor}} = 0.30$, $C_{\text{skip}} = 0.05$
- After flooring, minimum value is 0.30
- Skip threshold 0.05 can **never** be reached
- Skip logic was **dead code**

**Fix in v2.0:** Check raw score before scaling/flooring
- $C_{\text{raw}} = 0.03$ → Skip (before floor applies)
- $C_{\text{raw}} = 0.08$ → Trade (above threshold, then floor to 0.30)

### 8.3 Example

| $C_{\text{raw}}$ | Skip? | Reason |
|----------|-------|--------|
| 0.02 | Yes | Raw < 0.05 |
| 0.04 | Yes | Raw < 0.05 |
| 0.08 | No | Raw ≥ 0.05 (multiplier floors to 0.30) |
| 0.20 | No | Raw ≥ 0.05 (multiplier floors to 0.30) |
| 0.50 | No | Raw ≥ 0.05 (multiplier = 0.75) |

---

## 9. Parameter Configuration

### 9.1 Default Parameters

| Category | Parameter | Symbol | Default | Range | Description |
|----------|-----------|--------|---------|-------|-------------|
| **Volatility** | ATR period | $n$ | 14 | 10-20 | Bars for ATR |
| | ATR lookback | $N_{\text{hist}}$ | 250 | 100-500 | Historical comparison |
| **Performance** | Lookback days | $W$ | 30 | 15-60 | Rolling Sharpe window |
| | Min trades | $N_{\text{min}}$ | 10 | 5-20 | Minimum for Sharpe |
| | Max Sharpe | $S_{\text{max}}$ | 2.0 | 1.5-3.0 | Normalization cap |
| **Bayesian** | Prior Sharpe | $S_{\text{prior}}$ | 0.5 | 0.3-1.0 | Expected baseline |
| | Prior weight | $N_{\text{prior}}$ | 5 | 3-10 | Prior strength |
| **Smoothing** | EMA alpha | $\beta$ | 0.3 | 0.1-0.5 | Smoothing rate |
| **Sizing** | Scaling constant | $\alpha$ | 1.5 | 1.0-2.0 | Leverage factor |
| | Floor | $C_{\text{floor}}$ | 0.30 | 0.1-0.5 | Minimum multiplier |
| | Ceiling | $C_{\text{ceiling}}$ | 1.50 | 1.0-2.0 | Maximum multiplier |
| | Skip threshold | $C_{\text{skip}}$ | 0.05 | 0.01-0.20 | Skip if raw < this |
| **Cold Start** | Use graduated ceiling | - | true | true/false | Enable warm-up caps |
| | Very low threshold | $N_{\text{very-low}}$ | 5 | 3-7 | N < this: ceiling = 0.50 |
| | Low threshold | $N_{\text{low}}$ | 10 | 8-15 | N < this: ceiling = 0.75 |
| | Ceiling (very low) | $C_{\text{ceiling}}^{\text{N<5}}$ | 0.50 | 0.3-0.7 | Max when N < 5 |
| | Ceiling (low) | $C_{\text{ceiling}}^{\text{5≤N<10}}$ | 0.75 | 0.5-1.0 | Max when 5 ≤ N < 10 |

### 9.2 Configuration File

```json
{
  "position_sizing": {
    "max_sharpe_for_scaling": 2.0,
    "scaling_alpha": 1.5,
    "limits": {
      "min_multiplier": 0.30,
      "max_multiplier": 1.50,
      "skip_trade_threshold": 0.05
    },
    "bayesian_shrinkage": {
      "bayesian_prior_sharpe": 0.5,
      "bayesian_prior_weight": 5
    },
    "smoothing": {
      "sharpe_smoothing_alpha": 0.3
    }
  }
}
```

---

## 10. Complete Algorithm

### 10.1 Pseudocode

```python
def calculate_position_size(symbol, strategy, capital, max_pct, bars, trades):
    """
    Calculate adaptive position size.

    Args:
        symbol: Stock symbol
        strategy: Strategy name
        capital: Total capital
        max_pct: Max position % (e.g., 0.10 = 10%)
        bars: Historical price bars
        trades: Recent trade history

    Returns:
        position_size, should_skip
    """
    # Base position (before adaptation)
    base_size = capital * max_pct

    # === VOLATILITY ===
    # Calculate current ATR
    atr_current = calculate_atr(bars[-15:], period=14)

    # Calculate historical ATRs (rolling window over last 250 bars)
    atr_hist = []
    for i in range(len(bars) - 250, len(bars) - 14):
        window = bars[i:i+15]
        atr_hist.append(calculate_atr(window, period=14))

    # Percentile rank
    atr_pct = percentile_rank(atr_current, atr_hist) * 100
    V_score = (100 - atr_pct) / 100  # Invert

    # === PERFORMANCE ===
    # 1. Sample Sharpe and Bayesian shrinkage
    recent = filter_trades(trades, lookback_days=30)
    N = len(recent)

    if N == 0:
        # No trades yet - use prior directly (no shrinkage needed)
        sharpe_shrunk = 0.5  # S_prior
    else:
        # Calculate sample Sharpe
        returns = [t.return_pct for t in recent]
        sharpe_sample = (mean(returns) / std(returns)) * sqrt(252)

        # Apply Bayesian shrinkage
        sharpe_shrunk = (N * sharpe_sample + 5 * 0.5) / (N + 5)

    # 3. Exponential smoothing
    sharpe_prev = get_smoothed_history(symbol, strategy)
    if sharpe_prev is None:
        sharpe_smoothed = sharpe_shrunk
    else:
        sharpe_smoothed = 0.3 * sharpe_shrunk + 0.7 * sharpe_prev
    update_smoothed_history(symbol, strategy, sharpe_smoothed)

    # 4. Normalize
    P_score = max(0, min(1, sharpe_smoothed / 2.0))

    # === AGGREGATION ===
    C_raw = sqrt(V_score * P_score)

    # === SKIP CHECK ===
    if C_raw < 0.05:
        return None, True  # Skip trade

    # === SCALING & BOUNDS ===
    C_scaled = 1.5 * C_raw

    # Graduated ceiling for cold start
    if N < 5:
        C_ceiling_effective = 0.50  # Very cold start
    elif N < 10:
        C_ceiling_effective = 0.75  # Warming up
    else:
        C_ceiling_effective = 1.50  # Mature

    C_final = clamp(C_scaled, 0.30, C_ceiling_effective)

    # === FINAL SIZE ===
    position_size = base_size * C_final

    return position_size, False
```

### 10.2 Data Flow Diagram

```
Market Data (bars)
    ↓
calculate_atr(bars) → atr_current
    ↓
historical_atrs(bars, lookback=250) → atr_hist
    ↓
percentile_rank(atr_current, atr_hist) → atr_pct ∈ [0, 100]
    ↓
V_score = (100 - atr_pct) / 100 ∈ [0, 1]
    ↓
    ╔═══════════════════════════════╗
    ║   Geometric Mean Aggregation  ║
    ║   C_raw = √(V_score × P_score)║
    ╚═══════════════════════════════╝
    ↓
Trade History (recent_trades)
    ↓
filter_by_time(trades, 30 days) → recent
    ↓
extract_returns(recent) → returns[]
    ↓
sharpe_sample = (mean/std) * √252
    ↓
bayesian_shrinkage(sharpe_sample, n) → sharpe_shrunk
    ↓
exponential_smoothing(sharpe_shrunk) → sharpe_smoothed
    ↓
P_score = normalize(sharpe_smoothed, max=2.0) ∈ [0, 1]
    ↓

C_raw ∈ [0, 1]
    ↓
if C_raw < 0.05: SKIP TRADE
    ↓
C_scaled = α × C_raw (α = 1.5)
    ↓
C_final = clamp(C_scaled, 0.30, 1.50)
    ↓
Position = Capital × MaxPct × C_final
```

---

## 11. Worked Examples

### Example 1: Excellent Conditions

**Inputs:**
- Capital = $100,000
- Max % = 10%
- ATR percentile = 15% (low volatility)
- Sample Sharpe = 2.2 (from 25 trades)
- Previous smoothed Sharpe = 1.8

**Step-by-Step:**

1. **Volatility Score:**
   $$V_{\text{score}} = \frac{100 - 15}{100} = 0.85$$

2. **Bayesian Shrinkage:**
   $$S_{\text{shrunk}} = \frac{25 \times 2.2 + 5 \times 0.5}{25 + 5} = \frac{57.5}{30} = 1.92$$

3. **Exponential Smoothing:**
   $$S_{\text{smoothed}} = 0.3 \times 1.92 + 0.7 \times 1.8 = 0.576 + 1.26 = 1.836$$

4. **Performance Score:**
   $$P_{\text{score}} = \frac{1.836}{2.0} = 0.918$$

5. **Raw Combined Score:**
   $$C_{\text{raw}} = \sqrt{0.85 \times 0.918} = \sqrt{0.780} = 0.883$$

6. **Skip Check:**
   $0.883 > 0.05$ → **Trade**

7. **Scaling:**
   $$C_{\text{scaled}} = 1.5 \times 0.883 = 1.325$$

8. **Bounds:**
   $C_{\text{final}} = 1.325$ (within [0.30, 1.50])

9. **Position Size:**
   $$\text{Position} = 100000 \times 0.10 \times 1.325 = \$13,250$$

**Result:** 132.5% of base position (32.5% leverage on favorable conditions)

---

### Example 2: Poor Conditions

**Inputs:**
- Capital = $100,000
- Max % = 10%
- ATR percentile = 85% (high volatility)
- Sample Sharpe = 0.3 (from 12 trades)
- Previous smoothed Sharpe = 0.6

**Step-by-Step:**

1. **Volatility Score:**
   $$V_{\text{score}} = \frac{100 - 85}{100} = 0.15$$

2. **Bayesian Shrinkage:**
   $$S_{\text{shrunk}} = \frac{12 \times 0.3 + 5 \times 0.5}{12 + 5} = \frac{6.1}{17} = 0.359$$

3. **Exponential Smoothing:**
   $$S_{\text{smoothed}} = 0.3 \times 0.359 + 0.7 \times 0.6 = 0.108 + 0.42 = 0.528$$

4. **Performance Score:**
   $$P_{\text{score}} = \frac{0.528}{2.0} = 0.264$$

5. **Raw Combined Score:**
   $$C_{\text{raw}} = \sqrt{0.15 \times 0.264} = \sqrt{0.0396} = 0.199$$

6. **Skip Check:**
   $0.199 > 0.05$ → **Trade** (but small size)

7. **Scaling:**
   $$C_{\text{scaled}} = 1.5 \times 0.199 = 0.299$$

8. **Bounds:**
   $C_{\text{final}} = 0.30$ (floored)

9. **Position Size:**
   $$\text{Position} = 100000 \times 0.10 \times 0.30 = \$3,000$$

**Result:** 30% of base position (70% size reduction due to unfavorable conditions)

---

### Example 3: Skip Trade

**Inputs:**
- Capital = $100,000
- Max % = 10%
- ATR percentile = 92% (very high volatility)
- Sample Sharpe = -0.5 (negative from 15 trades)
- Previous smoothed Sharpe = 0.2

**Step-by-Step:**

1. **Volatility Score:**
   $$V_{\text{score}} = \frac{100 - 92}{100} = 0.08$$

2. **Bayesian Shrinkage:**
   $$S_{\text{shrunk}} = \frac{15 \times (-0.5) + 5 \times 0.5}{15 + 5} = \frac{-5.0}{20} = -0.25$$

3. **Exponential Smoothing:**
   $$S_{\text{smoothed}} = 0.3 \times (-0.25) + 0.7 \times 0.2 = -0.075 + 0.14 = 0.065$$

4. **Performance Score:**
   $$P_{\text{score}} = \frac{0.065}{2.0} = 0.033$$

5. **Raw Combined Score:**
   $$C_{\text{raw}} = \sqrt{0.08 \times 0.033} = \sqrt{0.00264} = 0.051$$

6. **Skip Check:**
   $0.051 > 0.05$ → **Barely trades** (on the edge)

   Alternative with slightly worse performance:
   If $P_{\text{score}} = 0.025$:
   $$C_{\text{raw}} = \sqrt{0.08 \times 0.025} = 0.045 < 0.05$$
   → **SKIP TRADE**

**Result:** Trade skipped due to extremely unfavorable conditions

---

## 12. Mathematical Properties

### 12.1 Continuity

**Theorem:** The sizing function $C_{\text{final}}(V, P)$ is continuous on the active trading region.

**Proof:**
- $\sqrt{\cdot}$ is continuous on $[0, 1]$
- Multiplication is continuous
- Clamping preserves continuity except at boundaries (floor/ceiling)
- In the active region ($C_{\text{raw}} \in [C_{\text{floor}}/\alpha, C_{\text{ceiling}}/\alpha]$), no clamping occurs
∴ $C_{\text{final}}$ is smooth and gradual within the active region

**Implication:** Small changes in metrics → small changes in position size (no jumps within active region)

**Important Caveat:**

The **full execution policy** is **not globally continuous** due to:

1. **Skip threshold:** Discrete decision at $C_{\text{raw}} = 0.05$
   - Below: no trade
   - Above: trade (with floor applied)

2. **Clamping boundaries:** Discrete transitions at floor and ceiling

3. **Graduated ceiling transitions:** Step changes at $N = 5$ and $N = 10$ trade count thresholds

The sizing function itself is smooth, but the overall policy has piecewise behavior at these decision boundaries.

### 12.2 Symmetry

**Theorem:** $C_{\text{raw}}$ is symmetric in $V$ and $P$:

$$\sqrt{V \times P} = \sqrt{P \times V}$$

**Implication:** Volatility and performance have equal weight in the multiplier.

### 12.3 Bounds

**Theorem:** $C_{\text{final}} \in [C_{\text{floor}}, C_{\text{ceiling}}]$ always.

**Proof:**
- $C_{\text{raw}} \in [0, 1]$ (since $V, P \in [0, 1]$ and geometric mean preserves range)
- $C_{\text{scaled}} = \alpha \cdot C_{\text{raw}} \in [0, \alpha]$
- $C_{\text{final}} = \text{clamp}(C_{\text{scaled}}, C_{\text{floor}}, C_{\text{ceiling}})$
- By definition of clamp: $C_{\text{final}} \in [C_{\text{floor}}, C_{\text{ceiling}}]$ ∎

### 12.4 Monotonicity

**Theorem:** $C_{\text{raw}}$ is monotonically increasing in both $V$ and $P$.

**Proof:**
- Fix $P$, vary $V$: $\frac{\partial}{\partial V}\sqrt{V \times P} = \frac{P}{2\sqrt{VP}} > 0$ for $V, P > 0$
- Fix $V$, vary $P$: $\frac{\partial}{\partial P}\sqrt{V \times P} = \frac{V}{2\sqrt{VP}} > 0$ for $V, P > 0$
∴ Improving either metric increases position size (before clamping)

### 12.5 Conservatism (Geometric Mean Property)

**Theorem:** Geometric mean ≤ Arithmetic mean

$$\sqrt{V \times P} \leq \frac{V + P}{2}$$

with equality iff $V = P$.

**Implication:** The system is more conservative than simple averaging. Imbalanced metrics (one high, one low) result in lower confidence than balanced metrics.

**Example:**
- Balanced: $V = 0.6, P = 0.6$ → Geometric = 0.60, Arithmetic = 0.60
- Imbalanced: $V = 0.9, P = 0.3$ → Geometric = 0.52, Arithmetic = 0.60

---

## 13. Implementation Notes

### 13.1 Cold Start Handling

**Problem:** No trade history at start → cannot compute reliable Sharpe.

**Solution: Graduated Ceiling with Prior + Shrinkage**

Instead of skipping trades entirely when $N < N_{\text{min}}$, the system uses:

1. **Bayesian shrinkage** to blend prior belief with limited sample data
2. **Graduated ceiling** to cap exposure during warm-up phase

**Policy:**

$$C_{\text{ceiling}}^{\text{effective}} = \begin{cases}
0.50 & \text{if } N < 5 \text{ (very cold start)} \\
0.75 & \text{if } 5 \leq N < 10 \text{ (warming up)} \\
C_{\text{ceiling}} & \text{if } N \geq 10 \text{ (mature, full ceiling = 1.50)}
\end{cases}$$

**Behavior:**

| Trade Count $N$ | Sharpe Estimate | Ceiling | Max Multiplier | Strategy |
|-----------------|-----------------|---------|----------------|----------|
| 0-4 | Prior (0.5) | 0.50 | 0.50 | Very conservative, still participates |
| 5-9 | Shrunk (67% sample, 33% prior) | 0.75 | 0.75 | Moderate caution |
| 10+ | Shrunk (86% sample, 14% prior) | 1.50 | 1.50 | Full sizing enabled |

**Why not skip entirely?**
- Allows system to learn from real trades
- Prior + shrinkage provides conservative estimate
- Graduated ceiling limits downside risk
- Smooth transition as confidence grows

**Role of $N_{\text{min}}$:**

$N_{\text{min}} = 10$ is now a **reliability threshold**, not a hard execution wall:
- Below $N_{\text{min}}$: Bayesian shrinkage dominates (heavy prior weight)
- Above $N_{\text{min}}$: Sample data dominates (light prior weight)
- Trading is allowed at all values of $N$ (via graduated ceiling)

### 13.2 State Management

The system maintains state for exponential smoothing:

**Smoothed Sharpe History:**
- **Key:** `(symbol, strategy)` tuple
- **Value:** $S_{\text{smoothed}}$ (last smoothed Sharpe value)
- **Update cadence:** On each trade close (not intra-bar)
- **Persistence:** Must be saved to disk/database for recovery after restart

**Initialization behavior:**
```python
# First call for (symbol, strategy) pair
if (symbol, strategy) not in smoothed_history:
    # No previous value - use current shrunk value directly
    smoothed_sharpe = shrunk_sharpe
else:
    # Apply EMA
    prev_smoothed = smoothed_history[(symbol, strategy)]
    smoothed_sharpe = alpha * shrunk_sharpe + (1 - alpha) * prev_smoothed

# Store for next time
smoothed_history[(symbol, strategy)] = smoothed_sharpe
```

**Missing/corrupted state:**
- If history file is corrupted or missing after restart
- System treats as first call (no EMA applied)
- Converges to correct smoothed value within ~5-10 trades

**Thread safety:**
- Current implementation is single-threaded
- If using multi-threaded execution, add locking around history updates

### 13.3 Numerical Stability

**Division by Zero:**
- If $\sigma_R = 0$ (all trades same return): set Sharpe = 0
- If $V \times P = 0$: $C_{\text{raw}} = 0$ → skip or floor

**Overflow/Underflow:**
- All calculations use bounded ranges
- No risk of overflow

### 13.4 Performance

**Complexity:**
- ATR calculation: $O(n)$ where $n = 14$ (constant)
- ATR percentile: $O(N_{\text{hist}})$ where $N_{\text{hist}} = 250$ (constant)
- Sharpe calculation: $O(N)$ where $N =$ recent trades ($\approx 30$)
- Overall: $O(N)$ per calculation, very efficient

### 13.5 Testing

**Unit Tests:**
- Verify geometric mean calculation
- Verify Bayesian shrinkage blending
- Verify EMA smoothing
- Verify skip logic (raw score < threshold)
- Verify floor/ceiling clamping

**Integration Tests:**
- Run backtest comparing:
  - Baseline (no adaptation)
  - Continuous sizing (this system)
  - Measure: Sharpe improvement, drawdown reduction

---

## 14. Comparison with Kelly Criterion

### 14.1 Kelly Formula

$$f^* = \frac{p(b+1) - 1}{b}$$

where:
- $p$ = win probability
- $b$ = win/loss ratio

### 14.2 Our System vs Kelly

| Aspect | Kelly | Our System |
|--------|-------|------------|
| **Basis** | Historical win rate & ratio | Volatility + Sharpe |
| **Adaptation** | Static (recompute periodically) | Continuous (every trade) |
| **Risk** | Volatility-blind | Explicitly considers ATR |
| **Smoothing** | None | Bayesian + EMA |
| **Bounds** | Often requires fractional Kelly | Built-in floor/ceiling |

**Conclusion:** Our system is a **volatility-aware, smoothed alternative** to Kelly that adapts to changing market conditions.

---

## 15. Future Enhancements

### 15.1 Possible Improvements

1. **Regime-Dependent Parameters:**
   - Bull market: higher $\alpha$
   - Bear market: lower ceiling

2. **Correlation Adjustment:**
   - Reduce multiplier if correlated positions exist
   - $C_{\text{adj}} = C_{\text{final}} \times (1 - \rho_{\text{portfolio}})$

3. **Tail Risk Adjustment:**
   - Penalize strategies with fat-tailed returns
   - Use CVaR instead of Sharpe

4. **Machine Learning:**
   - Learn optimal $\alpha, C_{\text{floor}}, C_{\text{ceiling}}$ from backtest
   - Adaptive parameter tuning

### 15.2 Known Limitations

1. **Lookback Bias:** Sharpe based on recent past, not necessarily predictive
2. **Non-stationarity:** Market regimes change, historical Sharpe may not persist
3. **Overfitting Risk:** Too many parameters can lead to backtest overfitting

**Mitigation:** Robust defaults, out-of-sample validation, parameter stability analysis

---

## Appendix A: Notation Summary

| Symbol | Description | Range | Units |
|--------|-------------|-------|-------|
| $\text{ATR}_n$ | Average True Range | $[0, \infty)$ | Price units |
| $\text{ATR}_{\text{pct}}$ | ATR percentile rank | $[0, 100]$ | Percent |
| $V_{\text{score}}$ | Volatility score (inverted) | $[0, 1]$ | Unitless |
| $S_{\text{sample}}$ | Sample Sharpe ratio | $(-\infty, \infty)$ | Unitless |
| $S_{\text{shrunk}}$ | Bayesian shrunk Sharpe | $(-\infty, \infty)$ | Unitless |
| $S_{\text{smoothed}}$ | EMA smoothed Sharpe | $(-\infty, \infty)$ | Unitless |
| $P_{\text{score}}$ | Performance score | $[0, 1]$ | Unitless |
| $C_{\text{raw}}$ | Raw combined score | $[0, 1]$ | Unitless |
| $C_{\text{scaled}}$ | Scaled score | $[0, \alpha]$ | Unitless |
| $C_{\text{final}}$ | Final multiplier | $[C_{\text{floor}}, C_{\text{ceiling}}]$ | Unitless |
| $\alpha$ | Scaling constant | $[1.0, 2.0]$ | Unitless |
| $N$ | Number of recent trades | $\mathbb{N}$ | Count |
| $N_{\text{prior}}$ | Prior weight | $\mathbb{N}$ | Count |
| $\beta$ | EMA smoothing parameter | $[0, 1]$ | Unitless |

---

## Appendix B: Configuration Template

```json
{
  "_description": "Continuous Adaptive Position Sizing Configuration",
  "_version": "2.1.0",

  "metrics": {
    "atr_period": 14,
    "atr_lookback": 250,
    "sharpe_lookback_days": 30,
    "min_trades_for_sharpe": 10
  },

  "position_sizing": {
    "use_continuous_formula": true,
    "max_sharpe_for_scaling": 2.0,
    "scaling_alpha": 1.5,

    "limits": {
      "min_multiplier": 0.30,
      "max_multiplier": 1.50,
      "skip_trade_threshold": 0.05
    },

    "cold_start": {
      "_comment": "Graduated ceiling reduces max multiplier during warm-up",
      "use_graduated_ceiling": true,
      "cold_start_very_low_threshold": 5,
      "cold_start_low_threshold": 10,
      "ceiling_very_low_trades": 0.50,
      "ceiling_low_trades": 0.75
    },

    "bayesian_shrinkage": {
      "bayesian_prior_sharpe": 0.5,
      "bayesian_prior_weight": 5
    },

    "smoothing": {
      "sharpe_smoothing_alpha": 0.3
    }
  },

  "per_symbol_overrides": {
    "TSLA": {
      "min_multiplier": 0.40,
      "max_multiplier": 1.30
    }
  }
}
```

---

## Appendix C: References

1. **Sharpe Ratio:**
   - Sharpe, W. F. (1966). "Mutual Fund Performance". *Journal of Business*.

2. **Bayesian Shrinkage:**
   - Stein, C. (1956). "Inadmissibility of the Usual Estimator for the Mean of a Multivariate Normal Distribution".
   - James, W., & Stein, C. (1961). "Estimation with Quadratic Loss".

3. **Exponential Smoothing:**
   - Brown, R. G. (1959). *Statistical Forecasting for Inventory Control*.
   - Holt, C. C. (1957). "Forecasting Trends and Seasonals by Exponentially Weighted Averages".

4. **Kelly Criterion:**
   - Kelly, J. L. (1956). "A New Interpretation of Information Rate". *Bell System Technical Journal*.

5. **ATR (Average True Range):**
   - Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*.

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | March 2026 | Initial mathematical formulation |
| 2.0.0 | March 2026 | **Major revision:** Added Bayesian shrinkage, exponential smoothing, scaling constant, fixed skip logic bug, corrected formula inconsistencies |
| 2.1.0 | March 2026 | **Production readiness:** Fixed cold-start policy conflict (graduated ceiling), clarified scope as "sizing component", added formal assumptions section, improved continuity claim precision, enhanced state management documentation |

---

**End of Document**
