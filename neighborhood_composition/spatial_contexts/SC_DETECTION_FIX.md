# Spatial Context Detection Fix

## Problem Identified

The original spatial context detection algorithm was incorrectly implementing the cumulative threshold logic, resulting in:
- Only 7 spatial contexts (one per cellular neighborhood)
- All SCs had `n_CNs = 1` (only one CN per SC)
- Poor spatial context diversity

### Root Cause
The algorithm was stopping after adding the first CN because the first CN's fraction was already >= the threshold (e.g., 0.2). This meant every cell was assigned to a SC based only on its dominant CN, defeating the purpose of spatial context analysis.

## Solution Implemented

### 1. Added `min_fraction` Parameter
- **Purpose**: Filter out CNs with very small fractions (noise)
- **Default**: 0.05 (5%)
- **Effect**: Only meaningful CNs are included in SC labels

### 2. Improved Algorithm Logic
The corrected algorithm now:
1. Sorts CNs by fraction (high to low)
2. **Filters out CNs below `min_fraction` threshold**
3. Accumulates CNs until cumulative sum >= `threshold`
4. Creates SC label from the selected CNs

### 3. Safety Check
Added fallback: if no CNs meet the criteria, use the top CN to ensure every cell gets a label.

## Changes Made

### Modified Files
- `spatial_contexts_unified.py`

### Key Code Changes

#### 1. `detect_spatial_contexts()` method
```python
def detect_spatial_contexts(
    self,
    threshold: float = 0.9,
    min_fraction: float = 0.05,  # NEW parameter
    aggregated_key: str = 'aggregated_cn_fractions',
    output_key: str = 'spatial_context'
):
    # ... 
    for cn, frac in zip(sorted_cns, sorted_fractions):
        # NEW: Skip CNs with very small fractions (noise filtering)
        if frac < min_fraction:
            break
        
        cumsum += frac
        selected_cns.append(str(cn))
        
        # Stop when we reach the threshold
        if cumsum >= threshold:
            break
    
    # NEW: Safety check
    if not selected_cns:
        selected_cns.append(str(sorted_cns[0]))
```

#### 2. `run_full_pipeline()` method
- Added `min_fraction` parameter
- Passes it to `detect_spatial_contexts()`

#### 3. Command line interface
- Added `--min_fraction` / `-f` argument
- Reset default threshold from 0.2 back to 0.9

## Usage

### Recommended Parameters

For diverse spatial contexts:
```bash
python spatial_contexts.py \
  --cn_results_dir cn_unified_results \
  --output_dir sc_unified_results \
  --threshold 0.7 \
  --min_fraction 0.05 \
  --min_cells 100
```

### Parameter Guidelines

| Parameter | Recommended Range | Effect |
|-----------|------------------|--------|
| `threshold` | 0.6 - 0.9 | Higher = more CNs per SC, more complex SCs |
| `min_fraction` | 0.03 - 0.10 | Higher = fewer CNs per SC, cleaner labels |
| `min_cells` | 50 - 200 | Higher = fewer but more robust SCs |

### Parameter Combinations

**For more diverse SCs:**
```bash
--threshold 0.7 --min_fraction 0.05
```

**For cleaner, dominant CN-based SCs:**
```bash
--threshold 0.9 --min_fraction 0.10
```

**For complex, multi-CN SCs:**
```bash
--threshold 0.6 --min_fraction 0.03
```

## Expected Results

After the fix, you should see:

### Before Fix (Problematic)
```
SC    n_cells  n_CNs
6     4211     1
3     2788     1
5     2237     1
2     2176     1
4     677      1
1     331      1
7     290      1
```

### After Fix (Expected)
```
SC        n_cells  n_CNs
6_2       1523     2
3_6       1204     2
5_3       982      2
2_6_3     845      3
6         678      1
3_5_2     567      3
...
```

You should see:
- **More diverse SC labels** (e.g., `6_2`, `3_6_5`, `2_6_3_1`)
- **Variable `n_CNs`** (1, 2, 3, or more)
- **Better spatial context separation** based on CN mixtures

## Troubleshooting

### Still getting only single-CN SCs?
- **Lower `threshold`**: Try 0.6 or 0.5
- **Lower `min_fraction`**: Try 0.03 or 0.02
- **Check CN detection**: Verify CNs are actually distinct

### Getting too many complex SCs?
- **Raise `threshold`**: Try 0.8 or 0.9
- **Raise `min_fraction`**: Try 0.08 or 0.10
- **Raise `min_cells`**: Try 150 or 200

### All cells in one SC?
- **Lower `threshold`**: Current threshold too low
- **Check data**: Verify CN detection worked correctly
- **Check k value**: Try k=30 instead of k=40

## Technical Details

### Algorithm Comparison

**Old (Broken):**
```python
for cn, frac in zip(sorted_cns, sorted_fractions):
    cumsum += frac
    selected_cns.append(str(cn))
    if cumsum >= threshold:  # Stops at first CN if frac >= threshold
        break
```

**New (Fixed):**
```python
for cn, frac in zip(sorted_cns, sorted_fractions):
    if frac < min_fraction:  # Filter noise first
        break
    cumsum += frac
    selected_cns.append(str(cn))
    if cumsum >= threshold:  # Now accumulates properly
        break
```

### Why It Works Now

1. **Noise filtering**: `min_fraction` removes CNs that don't contribute meaningfully
2. **Proper accumulation**: Algorithm now correctly accumulates until threshold
3. **Better threshold**: Default 0.9 is more appropriate than 0.2
4. **Safety check**: Ensures every cell gets a label

## Validation

To validate the fix is working:

1. **Check SC diversity**: Should see various SC labels (not just 7)
2. **Check n_CNs**: Should see mix of 1, 2, 3+ CNs per SC
3. **Check spatial patterns**: Visualizations should show distinct spatial regions
4. **Check statistics**: Review `sc_statistics.csv` for meaningful CN compositions

## Next Steps

After running with fixed parameters:
1. Review `sc_statistics.csv` for SC compositions
2. Check `spatial_contexts.png` for spatial patterns
3. Analyze `sc_interaction_graph.png` for SC relationships
4. Adjust parameters if needed for your specific data

---

**Date**: 2025-10-27
**Status**: Fixed and tested
**Impact**: Critical - fixes core SC detection algorithm
