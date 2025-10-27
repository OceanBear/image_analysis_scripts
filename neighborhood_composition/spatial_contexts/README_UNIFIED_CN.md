# Unified Cellular Neighborhood Detection

This document explains how to use the **unified CN detection** approach for processing multiple tiles with a shared neighborhood composition.

## Overview

The `cn_unified_kmeans.py` script processes multiple tiles (10-100+) simultaneously and assigns **the same set of Cellular Neighborhood (CN) labels** across all tiles. This is essential for:

1. **Consistent CN definitions** across all tiles
2. **Cross-tile comparisons** and analysis
3. **Downstream spatial context (SC) analysis** that requires unified CN labels

## Key Differences from Batch Processing

| Feature | Batch Processing (`cn_batch_kmeans.py`) | Unified Processing (`cn_unified_kmeans.py`) |
|---------|----------------------------------------|-------------------------------------------|
| **CN Detection** | Each tile gets its own CNs | All tiles share the same CNs |
| **CN Labels** | Tile-specific (CN1 in tile A ≠ CN1 in tile B) | Unified (CN1 means the same across all tiles) |
| **Heatmap** | One per tile | ONE unified heatmap for all tiles |
| **Spatial Maps** | One per tile | One per tile (with unified CN labels) |
| **Use Case** | Exploratory analysis per tile | Cross-tile analysis and spatial contexts |

## Usage

### Basic Command

```bash
python cn_unified_kmeans.py --tiles_dir /path/to/tiles --output_dir cn_unified_results
```

### Full Command with Options

```bash
python cn_unified_kmeans.py \
  --tiles_dir /path/to/tiles \
  --output_dir cn_unified_results \
  --k 20 \
  --n_clusters 6 \
  --celltype_key cell_type \
  --pattern "*.h5ad" \
  --max_tiles 14
```

### Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--tiles_dir` | `-t` | (path) | Directory containing h5ad tile files |
| `--output_dir` | `-o` | `cn_unified_results` | Output directory for results |
| `--k` | | `20` | Number of nearest neighbors for CN detection |
| `--n_clusters` | `-n` | `6` | Number of cellular neighborhoods to detect |
| `--celltype_key` | `-c` | `cell_type` | Column name for cell types in adata.obs |
| `--max_tiles` | `-m` | `None` | Limit number of tiles (for testing) |
| `--pattern` | `-p` | `*.h5ad` | File pattern to match |
| `--no_offset` | | `False` | Disable spatial coordinate offsetting |

## Examples

### Example 1: Process 14 Tiles with Default Settings

```bash
python cn_unified_kmeans.py \
  --tiles_dir /mnt/g/TCGA-LUAD/tiles/h5ad \
  --output_dir cn_unified_results
```

This will:
- Load all 14 tiles
- Detect 6 unified CNs across all tiles
- Generate **1 unified heatmap**
- Generate **14 individual spatial maps**
- Save 14 processed h5ad files with CN annotations

### Example 2: Process First 5 Tiles (Testing)

```bash
python cn_unified_kmeans.py \
  --tiles_dir /path/to/tiles \
  --output_dir test_cn_results \
  --max_tiles 5 \
  --n_clusters 4
```

### Example 3: Custom Cell Type Column

```bash
python cn_unified_kmeans.py \
  --tiles_dir /path/to/tiles \
  --celltype_key celltype \
  --output_dir cn_unified_results
```

### Example 4: More CNs for Complex Tissue

```bash
python cn_unified_kmeans.py \
  --tiles_dir /path/to/tiles \
  --k 20 \
  --n_clusters 10 \
  --output_dir cn_unified_10clusters
```

## Output Structure

```
cn_unified_results/
├── unified_analysis/
│   ├── unified_cn_composition_heatmap.png    # ONE heatmap for all tiles
│   ├── unified_cn_composition.csv            # CN composition data
│   └── unified_cn_summary.json               # Summary statistics
├── individual_tiles/
│   ├── tile1/
│   │   └── spatial_cns.png                   # Spatial map for tile1
│   ├── tile2/
│   │   └── spatial_cns.png                   # Spatial map for tile2
│   └── ...
└── processed_h5ad/
    ├── tile1_adata_cns.h5ad                  # Processed data for tile1
    ├── tile2_adata_cns.h5ad                  # Processed data for tile2
    └── ...
```

## Output Files

### 1. Unified CN Composition Heatmap
- **Location**: `unified_analysis/unified_cn_composition_heatmap.png`
- **Description**: Single heatmap showing cell type composition for each CN across ALL tiles
- **Use**: Understand what each CN represents in terms of cell types

### 2. Individual Spatial Maps
- **Location**: `individual_tiles/{tile_name}/spatial_cns.png`
- **Description**: Spatial visualization of CN labels for each tile
- **Use**: See where each CN is located spatially in each tile

### 3. Processed h5ad Files
- **Location**: `processed_h5ad/{tile_name}_adata_cns.h5ad`
- **Description**: AnnData files with CN annotations (`cn_celltype` column)
- **Use**: Input for spatial context analysis (next step)

### 4. Summary Statistics
- **Location**: `unified_analysis/unified_cn_summary.json`
- **Description**: JSON file with CN distribution, parameters, and metadata
- **Use**: Track processing parameters and results

## Next Steps: Spatial Context Analysis

After running unified CN detection, use the processed h5ad files for spatial context (SC) analysis:

```bash
cd ../spatial_contexts
python spatial_contexts.py \
  --batch_dir ../cellular_neighborhoods/cn_unified_results \
  --output_dir sc_unified_results
```

The spatial context script expects:
- Input: `cn_unified_results/processed_h5ad/*.h5ad` files with `cn_celltype` column
- Process: Detect spatial contexts based on CN mixtures (k=40)
- Output: SC labels, interaction graphs, and visualizations

## Requirements

### Input Data Requirements
- **Format**: AnnData h5ad files
- **Required fields**:
  - `adata.obs[celltype_key]`: Cell type annotations (categorical)
  - `adata.obsm['spatial']`: Spatial coordinates (N x 2 array)
- **Tile identification**: Automatically added as `tile_name` column

### System Requirements
- **Memory**: ~8-16 GB RAM for 10-20 tiles
- **Time**: ~5-30 minutes depending on number of tiles and cells
- **Disk space**: ~2-5 GB for outputs

## Algorithm Details

### Step 1: Load and Combine Tiles
- Loads all tiles into memory
- Adds `tile_name` and `tile_id` columns
- Offsets spatial coordinates to prevent overlap
- Stores original coordinates for later use

### Step 2: Build k-NN Graph
- Constructs k=20 nearest neighbor graph across ALL cells
- Uses spatial coordinates (after offsetting)
- Creates unified connectivity matrix

### Step 3: Aggregate Neighbors
- For each cell, computes fractional composition of cell types in its k=20 neighborhood
- Results in a feature vector per cell (e.g., [0.4, 0.3, 0.2, 0.1] for 4 cell types)

### Step 4: Detect CNs
- Performs k-means clustering (k=6 by default) on aggregated neighbor vectors
- ALL cells from ALL tiles are clustered together
- Assigns unified CN labels (1-6) to all cells

### Step 5: Compute Composition
- Calculates cell type fractions within each CN across all tiles
- Z-score normalizes by cell type for visualization

### Step 6-9: Visualize and Save
- Creates unified heatmap
- Creates individual spatial maps
- Saves processed h5ad files
- Saves summary statistics

## Troubleshooting

### Issue: Out of Memory
**Solution**: Process fewer tiles at a time using `--max_tiles`:
```bash
python cn_unified_kmeans.py --tiles_dir /path --max_tiles 10
```

### Issue: Different Cell Type Column Name
**Solution**: Specify the correct column name:
```bash
python cn_unified_kmeans.py --tiles_dir /path --celltype_key celltype
```

### Issue: No Spatial Coordinates
**Error**: "Warning: No spatial coordinates found"
**Solution**: Ensure your h5ad files have `adata.obsm['spatial']` with x,y coordinates

### Issue: Too Many/Few CNs
**Solution**: Adjust `--n_clusters`:
```bash
# For simpler tissue
python cn_unified_kmeans.py --tiles_dir /path --n_clusters 4

# For complex tissue
python cn_unified_kmeans.py --tiles_dir /path --n_clusters 10
```

## Comparison with Alternatives

### When to Use Unified CN Detection
✅ You have multiple tiles from the same tissue/experiment
✅ You want to compare CNs across tiles
✅ You plan to do spatial context analysis
✅ You want consistent CN definitions

### When to Use Batch Processing
✅ You want to explore each tile independently
✅ Tiles are from different tissues/experiments
✅ You don't need cross-tile comparisons

### When to Use Single Tile Processing
✅ You only have one tile
✅ You want to test parameters quickly
✅ Exploratory analysis

## Parameter Tuning

### k (Number of Neighbors)
- **Default**: 20
- **Lower (k=10-15)**: More local, fine-grained CNs
- **Higher (k=30-40)**: More global, coarser CNs
- **Recommendation**: Start with 20, increase if CNs are too fragmented

### n_clusters (Number of CNs)
- **Default**: 6
- **Lower (3-4)**: Simpler tissue with few distinct regions
- **Higher (8-12)**: Complex tissue with many distinct microenvironments
- **Recommendation**: Look at heatmap; increase if CNs are very heterogeneous

## Citation

If you use this script, please cite the original paper:

Schürch et al. (2020) "Coordinated cellular neighborhoods orchestrate antitumoral immunity at the colorectal cancer invasive front" *Cell* 182(5):1341-1359.e19

## Support

For issues or questions:
1. Check this README
2. Check the main paper and supplementary methods
3. Review the code comments in `cn_unified_kmeans.py`
4. Check related scripts: `cn_batch_kmeans.py`, `cn_kmeans_tiled.py`
