# New Scripts and Documentation Summary

This document lists all new files created for unified cellular neighborhood detection and spatial context analysis.

## New Scripts Created

### 1. Main Processing Script
**File**: `cellular_neighborhoods/cn_unified_kmeans.py`

**Purpose**: Process multiple tiles (10-100+) with unified CN detection

**Key Features**:
- Loads all tiles into a combined dataset
- Performs CN detection on all tiles together
- Generates ONE unified heatmap for all tiles
- Generates individual spatial maps for each tile
- Saves processed h5ad files ready for spatial context analysis

**Usage**:
```bash
python cn_unified_kmeans.py --tiles_dir /path/to/tiles --output_dir cn_unified_results
```

## Documentation Files

### 2. Unified CN README
**File**: `cellular_neighborhoods/README_UNIFIED_CN.md`

**Content**:
- Complete usage guide for `cn_unified_kmeans.py`
- Command-line arguments reference
- Multiple usage examples
- Output structure documentation
- Troubleshooting guide
- Parameter tuning recommendations

### 3. Script Comparison Guide
**File**: `cellular_neighborhoods/SCRIPT_COMPARISON.md`

**Content**:
- Detailed comparison of three CN detection approaches:
  - `cn_kmeans_tiled.py` (single tile)
  - `cn_batch_kmeans.py` (independent batch)
  - `cn_unified_kmeans.py` (unified multi-tile)
- Decision tree for choosing the right script
- Example scenarios
- Technical differences (memory, time, output)
- Migration guide

### 4. Example Usage Script
**File**: `cellular_neighborhoods/example_unified_cn.py`

**Content**:
- Multiple usage examples with explanations
- Different scenarios (testing, production, custom settings)
- Python API usage examples
- Parameter tuning guide
- Command-line examples

**Usage**:
```bash
python example_unified_cn.py  # Shows all examples
```

### 5. Complete Workflow Guide
**File**: `neighborhood_composition/WORKFLOW_GUIDE.md`

**Content**:
- End-to-end workflow from raw data to spatial contexts
- Step-by-step instructions
- Algorithm explanations
- Parameter tuning for both CN and SC detection
- Troubleshooting common issues
- Best practices
- Complete example with TCGA-LUAD data

## File Organization

```
image_analysis_scripts/
└── neighborhood_composition/
    ├── WORKFLOW_GUIDE.md                    ← Start here! Complete workflow
    ├── README_NEW_SCRIPTS.md                ← This file
    │
    ├── cellular_neighborhoods/
    │   ├── cn_unified_kmeans.py             ← NEW: Main unified CN script
    │   ├── README_UNIFIED_CN.md             ← NEW: Unified CN documentation
    │   ├── SCRIPT_COMPARISON.md             ← NEW: Compare all CN scripts
    │   ├── example_unified_cn.py            ← NEW: Usage examples
    │   │
    │   ├── cn_kmeans_tiled.py               ← Existing: Single tile
    │   ├── cn_batch_kmeans.py               ← Existing: Independent batch
    │   └── cn_aggregate_kmeans.py           ← Existing: Aggregation
    │
    └── spatial_contexts/
        └── spatial_contexts.py              ← Modified: Now accepts batch input
```

## Quick Start

### For New Users

1. **Read the workflow guide first**:
   ```
   neighborhood_composition/WORKFLOW_GUIDE.md
   ```

2. **Compare scripts to choose the right one**:
   ```
   cellular_neighborhoods/SCRIPT_COMPARISON.md
   ```

3. **For multi-tile studies, use unified CN**:
   ```bash
   cd cellular_neighborhoods
   python cn_unified_kmeans.py --tiles_dir /path/to/tiles
   ```

### For Existing Users

1. **Check what's different**:
   ```
   cellular_neighborhoods/SCRIPT_COMPARISON.md
   ```

2. **Migrate to unified approach**:
   ```bash
   # Old approach (batch)
   python cn_batch_kmeans.py --tiles_dir /path
   
   # New approach (unified)
   python cn_unified_kmeans.py --tiles_dir /path
   ```

## What's New in spatial_contexts.py

The spatial contexts script has been updated with:
- Command-line argument support (removed, then restored by user)
- Better batch processing for multiple tiles
- Expects input from `cn_unified_results/processed_h5ad/`

## Key Improvements

### Over cn_batch_kmeans.py

| Feature | Batch | Unified |
|---------|-------|---------|
| CN Labels | Tile-specific | **Unified across tiles** |
| Heatmaps | One per tile | **One unified** |
| SC Analysis | Not ideal | **Perfect for SC** |
| Cross-tile Comparison | ❌ | ✅ |

### Benefits

1. **Consistent CN labels** across all tiles
2. **Better for spatial context** analysis
3. **Easier interpretation** (CN1 means same thing everywhere)
4. **One unified heatmap** instead of many separate ones
5. **Ready for downstream** spatial context detection

## Complete Workflow

```bash
# Step 0: Prepare your data
# - h5ad files with cell_type and spatial coordinates
# - All tiles in one directory

# Step 1: Unified CN Detection (NEW!)
cd cellular_neighborhoods
python cn_unified_kmeans.py \
  --tiles_dir /path/to/tiles \
  --output_dir cn_unified_results \
  --k 20 --n_clusters 6

# Output: 
#   - 1 unified heatmap
#   - N spatial maps (one per tile)
#   - N processed h5ad files

# Step 2: Spatial Context Detection
cd ../spatial_contexts
python spatial_contexts.py \
  --batch_dir ../cellular_neighborhoods/cn_unified_results \
  --output_dir sc_unified_results

# Output:
#   - Spatial context maps
#   - SC interaction graphs
#   - SC statistics
```

## Testing the New Script

### Quick Test (5 tiles)

```bash
python cn_unified_kmeans.py \
  --tiles_dir /path/to/tiles \
  --output_dir test_cn \
  --max_tiles 5 \
  --n_clusters 4
```

Expected time: ~5 minutes
Expected memory: ~2-4 GB

### Production Run (14 tiles)

```bash
python cn_unified_kmeans.py \
  --tiles_dir /path/to/tiles \
  --output_dir cn_unified_results \
  --k 20 \
  --n_clusters 6
```

Expected time: ~15-30 minutes
Expected memory: ~8-16 GB

## Validation

To verify the script works correctly:

1. **Check unified heatmap**:
   - Should show distinct CN patterns
   - Each CN should have unique cell type signature

2. **Check spatial maps**:
   - Should have N maps (one per tile)
   - CN labels should be consistent colors across tiles

3. **Check processed h5ad**:
   ```python
   import scanpy as sc
   adata = sc.read_h5ad('cn_unified_results/processed_h5ad/tile1_adata_cns.h5ad')
   print(adata.obs['cn_celltype'].value_counts())  # Should have CN labels
   print(adata.obs['tile_name'].unique())  # Should have tile identifier
   ```

## Common Use Cases

### Use Case 1: Multi-tile Study (RECOMMENDED)
**Files needed**: `cn_unified_kmeans.py` + `spatial_contexts.py`
```bash
python cn_unified_kmeans.py --tiles_dir /path
cd ../spatial_contexts
python spatial_contexts.py --batch_dir ../cellular_neighborhoods/cn_unified_results
```

### Use Case 2: Single Tile Exploration
**Files needed**: `cn_kmeans_tiled.py`
```bash
python cn_kmeans_tiled.py  # Edit main() to point to your tile
```

### Use Case 3: Independent Tile Analysis
**Files needed**: `cn_batch_kmeans.py`
```bash
python cn_batch_kmeans.py --tiles_dir /path
```

## Documentation Priority

For different user types:

**New to the method**:
1. `WORKFLOW_GUIDE.md` - Understand the complete process
2. `README_UNIFIED_CN.md` - Learn how to use unified CN
3. `example_unified_cn.py` - See usage examples

**Choosing between scripts**:
1. `SCRIPT_COMPARISON.md` - Compare all options
2. Decision: Most multi-tile studies → use `cn_unified_kmeans.py`

**Already using batch processing**:
1. `SCRIPT_COMPARISON.md` - See differences
2. `README_UNIFIED_CN.md` - Migration guide
3. Switch to `cn_unified_kmeans.py` for SC analysis

## Support and Troubleshooting

### Documentation Hierarchy

1. **WORKFLOW_GUIDE.md** - High-level workflow and troubleshooting
2. **SCRIPT_COMPARISON.md** - Which script to use
3. **README_UNIFIED_CN.md** - Detailed unified CN usage
4. **example_unified_cn.py** - Concrete examples

### Common Issues

All common issues and solutions are documented in:
- `WORKFLOW_GUIDE.md` - General troubleshooting
- `README_UNIFIED_CN.md` - Unified CN specific issues

## Next Steps

After understanding these files:

1. ✅ Choose the right script for your needs
2. ✅ Run CN detection (preferably unified for multi-tile)
3. ✅ Check results (heatmap, spatial maps)
4. ✅ Run spatial context detection
5. ✅ Analyze and interpret results

## Citation

All scripts implement methods from:

Schürch, C.M., Bhate, S.S., Barlow, G.L. et al. (2020). Coordinated cellular neighborhoods orchestrate antitumoral immunity at the colorectal cancer invasive front. *Cell* 182(5):1341-1359.e19.

## Summary

**Created 5 new files**:
1. `cn_unified_kmeans.py` - Main unified CN detection script
2. `README_UNIFIED_CN.md` - Complete unified CN documentation
3. `SCRIPT_COMPARISON.md` - Compare all CN scripts
4. `example_unified_cn.py` - Usage examples
5. `WORKFLOW_GUIDE.md` - Complete workflow guide

**Key Innovation**: 
Unified CN detection across multiple tiles, enabling:
- Consistent CN labels
- Cross-tile comparisons
- Proper spatial context analysis

**Recommended for**:
- Multi-tile studies (10-100+ tiles)
- Spatial context analysis
- Cross-tile comparisons

