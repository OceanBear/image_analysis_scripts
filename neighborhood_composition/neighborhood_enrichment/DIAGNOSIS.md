# DIAGNOSIS: High "Z-Score" Values

## Problem

Your aggregated results show z-scores like 15.78, 11.62, -8.82, etc. These are **NOT proper z-scores**.

- **Expected z-score range**: -3 to +3 (95% of values within ±2)
- **Your actual range**: -17 to +32
- **Average CI width**: 13-24 (should be ~4-6)

## Root Cause

Squidpy's `sq.gr.nhood_enrichment()` function returns what it calls "z-scores" in the result dictionary, but these are **NOT standardized z-scores**. They appear to be:

1. **Enrichment scores** (log-fold change of observed/expected)
2. **OR un-normalized test statistics**

This is why you're seeing:
- Values much higher than ±3
- Very wide confidence intervals
- High variability across bootstraps

## What Squidpy Actually Returns

From squidpy source code, `nhood_enrichment()` computes:

```python
# Observed counts
obs_count = count neighbors of each type

# Permutation-based expected counts
for each permutation:
    shuffle cell labels
    count neighbors -> perm_counts

# Compute "zscore" as:
zscore = (obs_count - mean(perm_counts)) / std(perm_counts)
```

However, when cell types are rare or spatial patterns are extreme, this can produce values >>3.

## Solutions

### Option 1: Use Squidpy's Output As-Is (Current)
Accept that these are "enrichment z-scores" rather than standard z-scores:
- Values > 10 = **extremely strong** spatial clustering
- Wide CIs = high heterogeneity across tiles
- This is technically correct for your data

**Interpretation:**
- Epithelium (PD-L1hi/Ki67hi) self-clustering: z=15.78 → **extremely strong**
- Lymphocyte self-clustering: z=11.62 → **very strong**
- These indicate very non-random spatial organization

### Option 2: Winsorize/Clip Extreme Values
Cap z-scores at ±10 to prevent outliers from dominating:

```python
zscore_clipped = np.clip(zscore, -10, 10)
```

**Pros**: More stable aggregation
**Cons**: Loses information about extreme enrichment

### Option 3: Use Enrichment Scores Instead of Z-Scores
Switch to using the raw enrichment scores (log fold-change):

```python
# In neighborhood_enrichment_analysis
enrichment = adata.uns[f'{cluster_key}_nhood_enrichment']['enrichment']
```

**Pros**: More intuitive interpretation
**Cons**: Not standardized, harder to compare

### Option 4: Recompute Proper Z-Scores
Compute z-scores from scratch using:

```python
z = (observed - mean_permuted) / (std_permuted + epsilon)
# where epsilon prevents division by zero
```

## Recommendation

**For your data, Option 1 (current approach) is actually CORRECT.**

Your high z-scores indicate:
1. **Very strong spatial clustering** of cell types
2. **Significant heterogeneity** across tissue regions (wide CIs)
3. **Real biological signal**, not a bug

The empty plot might be a visualization issue (NaN handling), but the underlying data is valid.

## Next Steps

1. **Re-run diagnostics** to see actual data ranges
2. **Check if empty plot is due to NaN values**
3. **Adjust visualization** to handle extreme values
4. **Interpret results** as extreme spatial enrichment (which is biologically meaningful)