# Analyst Alpha Index Framework v1.1

## Overview

The Analyst Alpha Index measures each analyst's stock-picking skill by tracking the cumulative performance of their rating calls relative to the VnIndex benchmark.

**Core Concept:**
- Each analyst starts with an index value of **100**
- Index goes **UP** when calls are correct (stocks move in predicted direction vs index)
- Index goes **DOWN** when calls are wrong
- Daily compounding shows cumulative skill over time

---

## Rating Definitions

| Rating | Direction Weight | Meaning |
|--------|-----------------|---------|
| **OPF** (Outperform) | **+1.0** | Bullish - expect stock to BEAT VnIndex |
| **UPF** (Underperform) | **-1.0** | Bearish - expect stock to TRAIL VnIndex |
| **MPF** (Market Perform) | **-0.3** | Light bearish - upside unclear, price may be stretched |

### Why MPF = -0.3?

MPF at Dragon Capital typically means "upside not clear" or "price stretched but not aggressive on downside" - this is a **soft bearish** view, not truly neutral. The -0.3 weight:
- Rewards analysts when MPF stocks underperform (correct call)
- Penalizes when MPF stocks outperform (missed opportunity)
- Stakes are small (30% of UPF) matching the low conviction level

---

## Calculation Formula

### Step 1: Calculate Excess Return

For each stock on each day:

```
excess_return = stock_daily_return - vnindex_daily_return
```

Where:
- `stock_daily_return = (price_today / price_yesterday - 1) × 100`
- `vnindex_daily_return = (vnindex_today / vnindex_yesterday - 1) × 100`

### Step 2: Calculate Alpha Contribution

For each rated stock:

```
alpha_contribution = direction_weight × excess_return
```

| Rating | Formula | Example (excess = +5%) | Example (excess = -5%) |
|--------|---------|------------------------|------------------------|
| OPF | +1.0 × excess | +5.0% ✓ | -5.0% ✗ |
| UPF | -1.0 × excess | -5.0% ✗ | +5.0% ✓ |
| MPF | -0.3 × excess | -1.5% ✗ | +1.5% ✓ |

### Step 3: Calculate Daily Alpha

Average contribution across all rated stocks:

```
daily_alpha = Σ(alpha_contributions) / number_of_stocks
```

### Step 4: Update Index

Compound the daily alpha:

```
new_index = previous_index × (1 + daily_alpha / 100)
```

---

## Complete Example

### Analyst Coverage

| Ticker | Rating | Direction |
|--------|--------|-----------|
| VNM | OPF | +1.0 |
| FPT | OPF | +1.0 |
| MWG | OPF | +1.0 |
| NVL | UPF | -1.0 |
| PDR | UPF | -1.0 |
| TCB | MPF | -0.3 |
| MBB | MPF | -0.3 |
| VCB | MPF | -0.3 |
| HPG | MPF | -0.3 |
| ACB | MPF | -0.3 |

### Day X Calculation

**VnIndex Return:** +0.50%

| Ticker | Rating | Direction | Stock Return | Excess Return | Alpha Contribution |
|--------|--------|-----------|--------------|---------------|-------------------|
| VNM | OPF | +1.0 | +1.20% | +0.70% | +0.70% ✓ |
| FPT | OPF | +1.0 | +0.30% | -0.20% | -0.20% ✗ |
| MWG | OPF | +1.0 | +0.80% | +0.30% | +0.30% ✓ |
| NVL | UPF | -1.0 | -1.00% | -1.50% | +1.50% ✓ |
| PDR | UPF | -1.0 | +0.20% | -0.30% | +0.30% ✓ |
| TCB | MPF | -0.3 | +0.60% | +0.10% | -0.03% ✗ |
| MBB | MPF | -0.3 | -0.50% | -1.00% | +0.30% ✓ |
| VCB | MPF | -0.3 | +2.00% | +1.50% | -0.45% ✗ |
| HPG | MPF | -0.3 | -0.80% | -1.30% | +0.39% ✓ |
| ACB | MPF | -0.3 | +0.40% | -0.10% | +0.03% ✓ |

**Sum of contributions:** +2.84%

**Daily Alpha:** 2.84% / 10 = **+0.284%**

**Index Update:** 
- Previous: 102.50
- New: 102.50 × (1 + 0.284/100) = **102.79**

---

## Interpretation Guide

### Index Value Meaning

| Index Value | Interpretation |
|-------------|----------------|
| > 100 | Analyst generates positive alpha (good calls) |
| = 100 | Break even (no better than random) |
| < 100 | Analyst destroys alpha (poor calls) |
| 105 | Cumulative +5% alpha generated |
| 95 | Cumulative -5% alpha destroyed |

### Rating Outcome Matrix

| Rating | Stock vs Index | Result | Contribution |
|--------|----------------|--------|--------------|
| OPF | Beats index | ✓ Correct | Positive |
| OPF | Trails index | ✗ Wrong | Negative |
| UPF | Trails index | ✓ Correct | Positive |
| UPF | Beats index | ✗ Wrong | Negative |
| MPF | Trails index | ✓ Correct (soft bearish) | Small positive |
| MPF | Beats index | ✗ Wrong (missed upside) | Small negative |

---

## Cross-Analyst Comparison Framework

### Why Alpha Index Alone Is Not Enough

The Alpha Index is useful but **not perfectly fair** for comparing analysts due to:

| Issue | Impact on Comparison |
|-------|---------------------|
| **Coverage Size** | Analyst with 5 stocks has more volatile index than one with 15 stocks |
| **Stock Volatility** | High-beta coverage (VHM, NVL) swings more than defensive (VNM, SAB) |
| **Sector Exposure** | Banking rally vs Real Estate crash affects analysts differently |
| **Conviction Level** | Mostly OPF/UPF analyst has bigger swings than mostly MPF analyst |

### Additional Metrics for Fair Comparison

To enable fair cross-analyst comparison, calculate these additional metrics:

#### 1. Hit Rate (Skill Consistency)

Measures what percentage of calls were directionally correct.

```
Hit Rate = Correct Calls / Total Calls × 100%
```

**Definition of "Correct":**
| Rating | Correct When |
|--------|--------------|
| OPF | excess_return > 0 (stock beat index) |
| UPF | excess_return < 0 (stock trailed index) |
| MPF | excess_return < 0 (stock trailed index, soft bearish) |

**Why It Matters:**
- NOT affected by stock volatility
- NOT affected by coverage size
- Pure measure of directional accuracy
- Identifies "skilled but cautious" vs "lucky but volatile"

#### 2. Conviction Ratio (Aggressiveness)

Measures how much the analyst takes strong views vs neutral positions.

```
Conviction Ratio = (OPF_count + UPF_count) / Total_Stocks × 100%
```

Or weighted version:

```
Weighted Conviction = (OPF×1.0 + UPF×1.0 + MPF×0.3) / Total_Stocks
```

**Interpretation:**
| Conviction | Meaning |
|------------|---------|
| > 80% | High conviction - takes strong views |
| 50-80% | Moderate - balanced approach |
| < 50% | Low conviction - many MPF ratings |

#### 3. Information Ratio (Risk-Adjusted Performance)

Measures alpha generated per unit of volatility.

```
Information Ratio = Mean(Daily Alpha) / StdDev(Daily Alpha)
```

**Interpretation:**
| IR Value | Meaning |
|----------|---------|
| > 0.5 | Excellent - consistent alpha generation |
| 0.3 - 0.5 | Good - solid risk-adjusted performance |
| 0.1 - 0.3 | Average |
| < 0.1 | Poor - inconsistent or negative |

**Why It Matters:**
- Normalizes for different volatility exposures
- Analyst covering volatile stocks isn't unfairly advantaged/disadvantaged
- Standard metric used in fund management

#### 4. Coverage Count (Context)

Simply the number of stocks the analyst rates.

**Why It Matters:**
- Provides context for interpreting other metrics
- Fewer stocks = more concentrated risk
- More stocks = more diversified, harder to generate high alpha

---

## Analyst Assessment Dashboard

### Recommended Scorecard View

Display these 6 metrics for each analyst:

| Metric | Description | Primary Use |
|--------|-------------|-------------|
| **Alpha Index** | Cumulative index value | Main performance score |
| **YTD Alpha** | Index - 100 | Easy-to-read return |
| **Hit Rate** | % calls correct | Skill consistency |
| **Information Ratio** | Risk-adjusted alpha | Quality of alpha |
| **Conviction** | % OPF+UPF | Aggressiveness level |
| **Coverage** | Stock count | Context |

### Example Dashboard

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        ANALYST ALPHA SCORECARD - YTD 2025                           │
├─────────────┬───────────┬───────────┬──────────┬──────────┬────────────┬────────────┤
│ Analyst     │ Alpha Idx │ YTD Alpha │ Hit Rate │ Info Rat │ Conviction │ Coverage   │
├─────────────┼───────────┼───────────┼──────────┼──────────┼────────────┼────────────┤
│ Analyst A   │ 112.5     │ +12.5%    │ 62%      │ 0.45     │ 85%        │ 12         │
│ Analyst B   │ 108.2     │ +8.2%     │ 68%      │ 0.52     │ 45%        │ 15         │
│ Analyst C   │ 115.0     │ +15.0%    │ 55%      │ 0.28     │ 95%        │ 6          │
│ Analyst D   │ 103.5     │ +3.5%     │ 58%      │ 0.35     │ 60%        │ 10         │
│ Analyst E   │ 98.5      │ -1.5%     │ 48%      │ 0.15     │ 70%        │ 8          │
│ Analyst F   │ 95.2      │ -4.8%     │ 45%      │ 0.10     │ 80%        │ 11         │
├─────────────┼───────────┼───────────┼──────────┼──────────┼────────────┼────────────┤
│ TEAM AVG    │ 105.5     │ +5.5%     │ 56%      │ 0.31     │ 72%        │ 10.3       │
└─────────────┴───────────┴───────────┴──────────┴──────────┴────────────┴────────────┘
```

### How to Read the Scorecard

| Profile | Alpha Index | Hit Rate | Conviction | Interpretation |
|---------|-------------|----------|------------|----------------|
| **Star Performer** | High (>110) | High (>60%) | Any | Skilled AND generating alpha |
| **Skilled but Cautious** | Medium | High (>60%) | Low (<50%) | Good picks, needs more conviction |
| **Lucky Gambler** | High (>110) | Low (<55%) | High (>80%) | Volatile, may revert |
| **Consistent Average** | Medium (~105) | Medium (~55%) | Medium | Solid, room to improve |
| **Needs Coaching** | Low (<100) | Low (<50%) | Any | Review process and picks |

### Analyst Profiles from Example

- **Analyst A**: Star performer - high alpha with good hit rate and conviction
- **Analyst B**: Most skilled - highest hit rate & IR, but cautious (many MPF)
- **Analyst C**: ⚠️ Lucky gambler? - highest alpha but lowest hit rate, only 6 stocks
- **Analyst D**: Consistent average - solid across all metrics
- **Analyst E**: Below water - needs review, hit rate below 50%
- **Analyst F**: Worst performer - low alpha despite high conviction = wrong calls

---

## Ranking Methodology

### Option 1: Simple Ranking (Recommended to Start)

Rank by **Alpha Index** only.

**Pros:** Simple, intuitive, investors care about raw alpha
**Cons:** Not perfectly fair across different coverage types

### Option 2: Composite Score

Create weighted score from multiple metrics:

```
Composite Score = 0.4 × Alpha_Percentile 
                + 0.4 × HitRate_Percentile 
                + 0.2 × IR_Percentile
```

**Pros:** Balances raw performance with skill consistency
**Cons:** More complex, harder to explain

### Option 3: Tiered Rankings

Create separate rankings by category:

```
TIER 1: Alpha Champions (Top alpha generators)
TIER 2: Skill Leaders (Top hit rate)
TIER 3: Consistency Kings (Top information ratio)
```

**Pros:** Recognizes different strengths
**Cons:** No single "winner"

### Recommendation

**Start with Option 1** (Alpha Index ranking) but **display all metrics** so managers can see the full picture. Over time, consider Option 2 for bonus/evaluation purposes.

---

## Fairness Considerations

### What We Accept

1. **Volatility differences** - Analysts covering high-beta stocks will have more volatile indices
2. **Sector exposure** - Sector tailwinds/headwinds affect all analysts in that sector
3. **Coverage size** - Concentrated portfolios are inherently riskier

### What We Mitigate

1. **Show Hit Rate** - Identifies skill vs luck
2. **Show Information Ratio** - Normalizes for volatility
3. **Show Conviction** - Explains alpha magnitude
4. **Show Coverage** - Provides context

### Long-Term Fairness

Over **6-12 months**, these differences tend to even out:
- Lucky streaks don't last forever
- Good analysts stay above 100 consistently
- Poor analysts stay below 100 consistently
- The pattern becomes clear

---

## Data Requirements

### Input Data

1. **Analyst Ratings** (new collection to create)
   - analyst_id / analyst_email
   - ticker
   - rating (OPF / UPF / MPF)
   - rating_date
   - is_active (boolean)

2. **Stock Prices** (existing: Market_Data table)
   - TICKER
   - TRADE_DATE
   - PX_LAST

3. **VnIndex** (existing: MarketIndex table)
   - TRADINGDATE
   - INDEXVALUE (where COMGROUPCODE = 'VNINDEX')

### Output Data

1. **Daily Alpha Log** (new collection)
   - analyst_id
   - date
   - daily_alpha
   - index_value
   - hit_count (for hit rate calculation)
   - total_count

2. **Analyst Summary** (new collection, updated daily)
   - analyst_id
   - date
   - index_value
   - ytd_alpha
   - hit_rate
   - information_ratio
   - conviction_ratio
   - coverage_count
   - opf_count
   - upf_count
   - mpf_count

---

## Implementation Pseudocode

```python
# Configuration
DIRECTION_WEIGHTS = {
    'OPF': +1.0,
    'OUTPERFORM': +1.0,
    'BUY': +1.0,
    'UPF': -1.0,
    'UNDERPERFORM': -1.0,
    'SELL': -1.0,
    'MPF': -0.3,
    'MARKET-PERFORM': -0.3,
    'HOLD': -0.3,
}

def calculate_daily_metrics(analyst_id, trade_date):
    """Calculate daily alpha and hit rate for one analyst."""
    
    # 1. Get analyst's active ratings
    ratings = get_active_ratings(analyst_id)
    
    # 2. Get VnIndex return for the day
    vnindex_return = get_vnindex_return(trade_date)
    
    # 3. Calculate contribution for each stock
    contributions = []
    hits = 0
    
    for rating in ratings:
        stock_return = get_stock_return(rating.ticker, trade_date)
        excess_return = stock_return - vnindex_return
        direction = DIRECTION_WEIGHTS[rating.rating]
        contribution = direction * excess_return
        contributions.append(contribution)
        
        # Check if call was correct
        if rating.rating in ['OPF', 'OUTPERFORM', 'BUY']:
            if excess_return > 0:
                hits += 1
        elif rating.rating in ['UPF', 'UNDERPERFORM', 'SELL', 'MPF', 'MARKET-PERFORM', 'HOLD']:
            if excess_return < 0:
                hits += 1
    
    # 4. Calculate daily alpha (average)
    daily_alpha = sum(contributions) / len(contributions)
    hit_rate = hits / len(contributions) * 100
    
    return {
        'daily_alpha': daily_alpha,
        'hits': hits,
        'total': len(contributions),
        'hit_rate': hit_rate
    }


def update_analyst_index(analyst_id, trade_date):
    """Update the analyst's alpha index and all metrics."""
    
    # Get previous values
    prev_index = get_previous_index(analyst_id)  # Default 100 if first day
    prev_daily_alphas = get_ytd_daily_alphas(analyst_id)  # For IR calculation
    
    # Calculate today's metrics
    metrics = calculate_daily_metrics(analyst_id, trade_date)
    
    # Compound the index
    new_index = prev_index * (1 + metrics['daily_alpha'] / 100)
    
    # Calculate Information Ratio (rolling)
    all_alphas = prev_daily_alphas + [metrics['daily_alpha']]
    if len(all_alphas) >= 20:  # Need enough data points
        ir = np.mean(all_alphas) / np.std(all_alphas)
    else:
        ir = None
    
    # Get rating distribution
    ratings = get_active_ratings(analyst_id)
    opf_count = sum(1 for r in ratings if r.rating in ['OPF', 'OUTPERFORM', 'BUY'])
    upf_count = sum(1 for r in ratings if r.rating in ['UPF', 'UNDERPERFORM', 'SELL'])
    mpf_count = sum(1 for r in ratings if r.rating in ['MPF', 'MARKET-PERFORM', 'HOLD'])
    total_count = len(ratings)
    
    conviction = (opf_count + upf_count) / total_count * 100
    
    # Save to database
    save_analyst_summary(
        analyst_id=analyst_id,
        date=trade_date,
        index_value=new_index,
        ytd_alpha=new_index - 100,
        daily_alpha=metrics['daily_alpha'],
        hit_rate=metrics['hit_rate'],
        information_ratio=ir,
        conviction_ratio=conviction,
        coverage_count=total_count,
        opf_count=opf_count,
        upf_count=upf_count,
        mpf_count=mpf_count
    )
    
    return new_index


def run_daily_job(trade_date):
    """Run for all analysts after market close."""
    
    analysts = get_all_analysts()
    
    for analyst in analysts:
        update_analyst_index(analyst.id, trade_date)
    
    print(f"Updated {len(analysts)} analysts for {trade_date}")


def generate_scorecard(as_of_date):
    """Generate the analyst scorecard dashboard data."""
    
    analysts = get_all_analysts()
    scorecard = []
    
    for analyst in analysts:
        summary = get_analyst_summary(analyst.id, as_of_date)
        scorecard.append({
            'analyst_name': analyst.name,
            'alpha_index': summary.index_value,
            'ytd_alpha': summary.ytd_alpha,
            'hit_rate': summary.hit_rate,
            'information_ratio': summary.information_ratio,
            'conviction': summary.conviction_ratio,
            'coverage': summary.coverage_count
        })
    
    # Sort by alpha index descending
    scorecard.sort(key=lambda x: x['alpha_index'], reverse=True)
    
    # Add rank
    for i, row in enumerate(scorecard):
        row['rank'] = i + 1
    
    return scorecard
```

---

## Database Schema

### MongoDB: AnalystRatings

```javascript
{
  "_id": ObjectId,
  "analyst_email": "analyst@dragoncapital.com",
  "analyst_name": "Nguyen Van A",
  "ticker": "VNM",
  "rating": "OPF",                    // OPF, UPF, MPF
  "direction_weight": 1.0,           // +1.0, -1.0, -0.3
  "rating_date": ISODate("2025-01-01"),
  "end_date": null,                  // null if active, date if changed
  "is_active": true,
  "created_at": ISODate,
  "updated_at": ISODate
}

// Indexes
{ "analyst_email": 1, "ticker": 1, "is_active": 1 }
{ "ticker": 1, "is_active": 1 }
{ "rating_date": -1 }
```

### MongoDB: AnalystAlphaIndex (Daily Log)

```javascript
{
  "_id": ObjectId,
  "analyst_email": "analyst@dragoncapital.com",
  "date": ISODate("2025-01-15"),
  "index_value": 103.45,
  "daily_alpha": 0.284,
  "hits": 6,
  "total": 10,
  "created_at": ISODate
}

// Indexes
{ "analyst_email": 1, "date": -1 }
{ "date": -1 }
```

### MongoDB: AnalystSummary (Scorecard Data)

```javascript
{
  "_id": ObjectId,
  "analyst_email": "analyst@dragoncapital.com",
  "analyst_name": "Nguyen Van A",
  "date": ISODate("2025-01-15"),
  "index_value": 103.45,
  "ytd_alpha": 3.45,
  "daily_alpha": 0.284,
  "hit_rate": 60.0,                  // percentage
  "information_ratio": 0.42,
  "conviction_ratio": 70.0,          // percentage
  "coverage_count": 10,
  "opf_count": 5,
  "upf_count": 2,
  "mpf_count": 3,
  "created_at": ISODate
}

// Indexes
{ "analyst_email": 1, "date": -1 }
{ "date": -1, "index_value": -1 }
```

---

## SQL Queries

### Get Stock Daily Return

```sql
SELECT 
    TICKER,
    TRADE_DATE,
    PX_LAST,
    (PX_LAST / LAG(PX_LAST) OVER (PARTITION BY TICKER ORDER BY TRADE_DATE) - 1) * 100 
        AS daily_return
FROM Market_Data
WHERE TICKER IN ('VNM', 'FPT', 'MWG', ...)
  AND TRADE_DATE BETWEEN @start_date AND @end_date
ORDER BY TICKER, TRADE_DATE
```

### Get VnIndex Daily Return

```sql
SELECT 
    TRADINGDATE,
    INDEXVALUE,
    PERCENTINDEXCHANGE as daily_return
FROM MarketIndex
WHERE COMGROUPCODE = 'VNINDEX'
  AND TRADINGDATE = @trade_date
```

---

## Reset Rules

1. **Annual Reset**: Index resets to 100 on January 1st each year
2. **Rating Change**: When rating changes, use new rating from change date forward
3. **New Stock**: First day of rating doesn't contribute (no previous price)
4. **Stock Suspension**: Skip days where stock doesn't trade
5. **Information Ratio**: Requires minimum 20 trading days to calculate

---

## Summary

### Core Formula

```
contribution = direction × excess_return

Direction weights:
  OPF  = +1.0
  UPF  = -1.0
  MPF  = -0.3

daily_alpha = Σ(contributions) / n_stocks
new_index = old_index × (1 + daily_alpha/100)
```

### Assessment Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| Alpha Index | Compounded daily alpha | Main performance score |
| Hit Rate | Correct calls / Total calls | Skill consistency |
| Information Ratio | Mean(alpha) / StdDev(alpha) | Risk-adjusted quality |
| Conviction Ratio | (OPF + UPF) / Total | Aggressiveness level |

### Key Insights

1. **Alpha Index** tells you WHO is generating alpha
2. **Hit Rate** tells you WHO is skilled (vs lucky)
3. **Information Ratio** tells you WHO is consistent
4. **Conviction** tells you WHO takes strong views

Use ALL metrics together for fair analyst assessment.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-15 | Initial framework with OPF(+1), UPF(-1), MPF(-0.3) |
| 1.1 | 2025-01-15 | Added cross-analyst comparison framework, hit rate, IR, conviction metrics |

---

## Contact

For questions or changes to this framework, contact the Research Team.
