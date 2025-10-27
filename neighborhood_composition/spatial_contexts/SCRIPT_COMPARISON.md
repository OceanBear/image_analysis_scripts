# Cellular Neighborhood Scripts Comparison

This document compares the three main scripts for cellular neighborhood detection and helps you choose the right one for your needs.

## Quick Comparison Table

| Feature | `cn_kmeans_tiled.py` | `cn_batch_kmeans.py` | `cn_unified_kmeans.py` |
|---------|---------------------|---------------------|----------------------|
| **Input** | Single tile | Multiple tiles | Multiple tiles |
| **Processing** | One tile at a time | Each tile independently | All tiles together |
| **CN Labels** | Tile-specific | Tile-specific | **Unified across tiles** |
| **Output: Heatmaps** | 1 per tile | 1 per tile | **1 unified + 1 per tile for spatial** |
| **Output: Spatial Maps** | 1 per tile | 1 per tile | 1 per tile |
| **Output: h5ad files** | Optional | Always saved | Always saved |
| **Cross-tile Comparison** | ❌ No | ❌ No | ✅ **Yes** |
| **Spatial Context Ready** | ⚠️ Single tile only | ⚠️ Not recommended | ✅ **Yes, ideal** |
| **Processing Time** | Fast (1-5 min/tile) | Medium (batch) | Medium (all at once) |
| **Memory Usage** | Low | Low-Medium | **Medium-High** |
| **Best For** | Single tile analysis | Exploratory per-tile | **Multi-tile studies** |

## Detailed Comparison

### 1. `cn_kmeans_tiled.py` - Single Tile Processing

**Purpose**: Process one tile at a time for exploratory analysis.

**How it works**:
```python
from cn_kmeans_tiled import CellularNeighborhoodDetector

adata = sc.read_h5ad('tile1.h5ad')
detector = CellularNeighborhoodDetector(adata)
detector.run_full_pipeline(
    k=20, n_clusters=6,
    celltype_key='cell_type',
    img_id_key='tile_name',
    output_dir='cn_tile1'
)
```

**Outputs**:
```
cn_tile1/
├── spatial_cns.png                # Spatial map
├── cn_composition_heatmap.png     # Composition heatmap
├── cn_composition.csv             # Composition data
└── tile1_adata_cns.h5ad          # Optional: processed data
```

**CN Labels**: 
- CN labels are specific to this tile only
- CN1 in tile A has no relationship to CN1 in tile B

**Use When**:
- ✅ You have only one tile
- ✅ Quick exploratory analysis
- ✅ Testing parameters
- ✅ Learning the method
- ❌ You need to compare tiles
- ❌ You want spatial context analysis across tiles

---

### 2. `cn_batch_kmeans.py` - Independent Batch Processing

**Purpose**: Process multiple tiles independently, each with its own CN labels.

**How it works**:

```python
from cn_batch_kmeans import BatchCellularNeighborhoodProcessor

processor = BatchCellularNeighborhoodProcessor(
    tiles_directory='/path/to/tiles',
    output_base_dir='../cellular_neighborhoods/cn_batch_results'
)

results = processor.process_all_tiles(
    k=20, n_clusters=6,
    celltype_key='cell_type',
    save_adata=True
)
```

**Outputs**:
```
cn_batch_results/
├── individual_tiles/
│   ├── tile1/
│   │   ├── spatial_cns.png
│   │   └── cn_composition_heatmap.png
│   ├── tile2/
│   │   ├── spatial_cns.png
│   │   └── cn_composition_heatmap.png
│   └── ...
├── processed_h5ad/
│   ├── tile1_adata_cns.h5ad
│   ├── tile2_adata_cns.h5ad
│   └── ...
└── batch_processing_summary.csv
```

**CN Labels**:
- Each tile gets its own independent CN labels
- CN1 in tile A is different from CN1 in tile B
- No consistency across tiles

**Use When**:
- ✅ You want to explore each tile independently
- ✅ Tiles are from different experiments/tissues
- ✅ You don't need cross-tile comparisons
- ✅ You want to process tiles in parallel
- ❌ You need consistent CN definitions
- ❌ You want to do spatial context analysis across tiles

---

### 3. `cn_unified_kmeans.py` - Unified Multi-Tile Processing ⭐

**Purpose**: Process multiple tiles with **shared CN labels** for cross-tile analysis.

**How it works**:
```python
from cn_unified_kmeans import UnifiedCellularNeighborhoodDetector

detector = UnifiedCellularNeighborhoodDetector(
    tiles_directory='/path/to/tiles',
    output_dir='cn_unified_results'
)

tile_files = detector.discover_tiles()
detector.run_full_pipeline(
    tile_files=tile_files,
    k=20, n_clusters=6,
    celltype_key='cell_type'
)
```

**Outputs**:
```
cn_unified_results/
├── unified_analysis/
│   ├── unified_cn_composition_heatmap.png   # ONE heatmap for ALL tiles ⭐
│   ├── unified_cn_composition.csv
│   └── unified_cn_summary.json
├── individual_tiles/
│   ├── tile1/spatial_cns.png               # Spatial maps per tile
│   ├── tile2/spatial_cns.png
│   └── ...
└── processed_h5ad/
    ├── tile1_adata_cns.h5ad                # Ready for spatial context ⭐
    ├── tile2_adata_cns.h5ad
    └── ...
```

**CN Labels**:
- **All tiles share the same CN definitions**
- CN1 means the same thing in all tiles
- CN labels are based on neighborhood composition across entire dataset

**Use When**:
- ✅ **You have multiple tiles from the same experiment** ⭐
- ✅ **You want to compare CNs across tiles** ⭐
- ✅ **You plan to do spatial context analysis** ⭐
- ✅ You want consistent CN definitions
- ✅ Tiles are from similar tissue types
- ❌ Memory is very limited (< 8 GB RAM)
- ❌ Tiles are from completely different experiments

---

## Decision Tree

```
Do you have multiple tiles?
├─ No → Use cn_kmeans_tiled.py
└─ Yes → Do the tiles come from the same experiment/tissue?
    ├─ No → Use cn_batch_kmeans.py
    └─ Yes → Do you need consistent CN labels across tiles?
        ├─ No → Use cn_batch_kmeans.py (faster, less memory)
        └─ Yes → Use cn_unified_kmeans.py ⭐
            └─ Plan to do spatial context analysis?
                └─ Yes → Definitely use cn_unified_kmeans.py ⭐⭐⭐
```

## Example Scenarios

### Scenario 1: Exploratory Analysis (Single Tile)
**Goal**: Quickly explore one tile to understand CN patterns

**Script**: `cn_kmeans_tiled.py`

**Command**:
```bash
python cn_kmeans_tiled.py
# Edit main() to point to your tile
```

---

### Scenario 2: Multiple Samples, Independent Analysis
**Goal**: Process 20 tiles from different patients, analyze each separately

**Script**: `cn_batch_kmeans.py`

**Command**:
```bash
python cn_batch_kmeans.py \
  --tiles_dir /path/to/tiles \
  --output_dir cn_batch_results
```

**Why**: Each patient/sample should have its own CN definitions

---

### Scenario 3: Same Experiment, Cross-Tile Analysis ⭐
**Goal**: Process 14 tiles from same tumor, compare CNs across tiles

**Script**: `cn_unified_kmeans.py`

**Command**:
```bash
python cn_unified_kmeans.py \
  --tiles_dir /path/to/tiles \
  --output_dir cn_unified_results
```

**Why**: You want CN1 to mean the same thing in all tiles

---

### Scenario 4: Spatial Context Analysis Workflow ⭐⭐⭐
**Goal**: Detect CNs, then detect spatial contexts based on CN mixtures

**Script**: `cn_unified_kmeans.py` → `spatial_contexts.py`

**Commands**:
```bash
# Step 1: Unified CN detection
python cn_unified_kmeans.py \
  --tiles_dir /path/to/tiles \
  --output_dir cn_unified_results

# Step 2: Spatial context analysis
cd ../spatial_contexts
python spatial_contexts.py \
  --batch_dir ../cellular_neighborhoods/cn_unified_results \
  --output_dir sc_unified_results
```

**Why**: Spatial context analysis requires consistent CN labels

---

## Technical Differences

### Memory Usage

| Script | Memory per Tile | Memory for 14 Tiles |
|--------|----------------|---------------------|
| `cn_kmeans_tiled.py` | ~200-500 MB | N/A (one at a time) |
| `cn_batch_kmeans.py` | ~200-500 MB | ~200-500 MB (sequential) |
| `cn_unified_kmeans.py` | ~200-500 MB | **~8-16 GB (all at once)** |

### Processing Time

Assuming ~10,000 cells per tile:

| Script | Time per Tile | Time for 14 Tiles |
|--------|--------------|-------------------|
| `cn_kmeans_tiled.py` | 1-5 min | N/A |
| `cn_batch_kmeans.py` | 1-5 min | 15-70 min (sequential) |
| `cn_unified_kmeans.py` | N/A | **10-30 min (all together)** |

### CN Detection Method

**cn_kmeans_tiled.py & cn_batch_kmeans.py**:
1. Load tile → Build k-NN → Aggregate → K-means (per tile)
2. Each tile gets independent clustering
3. CN labels are not comparable across tiles

**cn_unified_kmeans.py**:
1. Load all tiles → Combine → Build k-NN → Aggregate → K-means (unified)
2. All tiles clustered together
3. CN labels are consistent and comparable

---

## Migration Guide

### From Single Tile to Unified

**Before** (single tile):
```python
# cn_kmeans_tiled.py approach
adata = sc.read_h5ad('tile1.h5ad')
detector = CellularNeighborhoodDetector(adata)
detector.run_full_pipeline(...)
```

**After** (unified):
```bash
# cn_unified_kmeans.py approach
python cn_unified_kmeans.py --tiles_dir /path/to/tiles
```

### From Batch to Unified

**Before** (batch processing):
```bash
python cn_batch_kmeans.py \
  --tiles_dir /path/to/tiles \
  --output_dir cn_batch_results
```

**After** (unified):
```bash
python cn_unified_kmeans.py \
  --tiles_dir /path/to/tiles \
  --output_dir cn_unified_results
```

**Key Difference**: Unified produces one shared set of CN labels!

---

## Output Comparison

### Heatmaps

**Batch Processing** (`cn_batch_kmeans.py`):
- 14 tiles → **14 separate heatmaps**
- Each heatmap shows CN composition for that tile only
- CN1 in different heatmaps might represent different neighborhoods

**Unified Processing** (`cn_unified_kmeans.py`):
- 14 tiles → **1 unified heatmap**
- Heatmap shows CN composition across all tiles
- CN1 represents the same neighborhood in all tiles

### Spatial Maps

**Both scripts**:
- Generate one spatial map per tile
- Show where each CN is located spatially

**Difference**:
- Batch: CN colors/labels are tile-specific
- Unified: CN colors/labels are consistent across tiles

---

## Common Questions

### Q: Which script should I use for spatial context analysis?

**A**: `cn_unified_kmeans.py` ⭐

Spatial context analysis (Figure 19) requires consistent CN labels across all cells. If you use batch processing, each tile will have different CN labels, making spatial context analysis meaningless.

### Q: Can I use batch processing outputs for spatial contexts?

**A**: Not recommended ⚠️

Batch processing gives each tile independent CN labels. When you try to do spatial context analysis, the CN labels won't be consistent, and the results won't be interpretable.

### Q: I have 100 tiles and not enough memory for unified processing. What do I do?

**A**: Process in groups:
1. Group tiles by region/patient/sample
2. Run unified processing on each group separately
3. Each group gets consistent CN labels within the group

```bash
# Group 1: Tumor core (20 tiles)
python cn_unified_kmeans.py --tiles_dir group1/ --output_dir cn_group1

# Group 2: Invasive front (30 tiles)
python cn_unified_kmeans.py --tiles_dir group2/ --output_dir cn_group2

# Group 3: Normal tissue (25 tiles)
python cn_unified_kmeans.py --tiles_dir group3/ --output_dir cn_group3
```

### Q: How do I know if my CNs are meaningful?

**A**: Check the unified heatmap:
1. Each CN should have a distinct cell type composition
2. CNs should be clearly separated (distinct colors in heatmap)
3. If CNs look very similar, reduce `--n_clusters`
4. If CNs are very heterogeneous within, increase `--n_clusters`

### Q: Can I combine results from batch and unified processing?

**A**: No, the CN labels are incompatible

Batch processing CN labels are tile-specific, while unified labels are global. You cannot meaningfully compare or combine them.

---

## Summary

### Use `cn_kmeans_tiled.py` when:
- Single tile analysis
- Quick exploration
- Testing parameters

### Use `cn_batch_kmeans.py` when:
- Multiple independent samples
- Don't need cross-tile comparison
- Limited memory

### Use `cn_unified_kmeans.py` when: ⭐
- Same experiment/tissue
- Need consistent CN labels
- Plan spatial context analysis
- Want cross-tile comparisons

**For most multi-tile studies preparing for spatial context analysis, use `cn_unified_kmeans.py`!**

