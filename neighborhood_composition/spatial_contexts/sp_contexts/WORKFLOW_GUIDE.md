# Complete Workflow Guide: Cellular Neighborhoods → Spatial Contexts

This guide walks you through the complete workflow from raw h5ad tiles to spatial context analysis, following the methodology from Schürch et al. (2020) Cell paper.

## Overview

The analysis consists of two main steps:

1. **Cellular Neighborhoods (CN)**: Identify microenvironments based on cell type composition (Figure 18)
2. **Spatial Contexts (SC)**: Identify regions where CNs interact (Figure 19)

```
┌─────────────────┐
│  Raw h5ad Tiles │  (with cell types and spatial coordinates)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1: Cellular Neighborhood Detection                │
│  Script: cn_unified_kmeans.py                           │
│  Parameters: k=20, n_clusters=6                         │
│  Output: Tiles with CN labels                           │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2: Spatial Context Detection                      │
│  Script: spatial_contexts.py                            │
│  Parameters: k=40, threshold=0.9                        │
│  Output: Tiles with SC labels + interaction graphs      │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

### Required Data
Your h5ad files must contain:
- **Cell type annotations**: `adata.obs['cell_type']` (categorical)
- **Spatial coordinates**: `adata.obsm['spatial']` (N×2 array with x,y coordinates)
- **Gene expression** (optional for CN/SC detection)

### Required Software
```bash
pip install scanpy squidpy numpy pandas matplotlib seaborn networkx scikit-learn
```

## Step 1: Cellular Neighborhood Detection

### What are Cellular Neighborhoods?

Cellular neighborhoods (CNs) are microenvironments defined by the composition of cell types in local neighborhoods. Each cell is assigned a CN label based on what cell types surround it.

**Example CNs**:
- **CN1**: Tumor-rich (80% tumor cells, 10% fibroblasts, 10% immune)
- **CN2**: B-cell rich (60% B cells, 20% CD4 T cells, 20% tumor)
- **CN3**: Immunosuppressive (50% tumor, 30% macrophages, 20% regulatory T cells)

### Algorithm (Figure 18)

1. Build k=20 nearest neighbor graph
2. For each cell, count cell types in its 20 neighbors → feature vector
3. K-means clustering (k=6) on feature vectors
4. Assign CN labels (1-6) to all cells

### Choose the Right Script

See [SCRIPT_COMPARISON.md](spatial_contexts/SCRIPT_COMPARISON.md) for detailed comparison.

**Quick decision**:
- Multiple tiles from same experiment → **Use `cn_unified_kmeans.py`** ⭐
- Single tile or independent tiles → Use `cn_kmeans_tiled.py` or `cn_batch_kmeans.py`

### Running Unified CN Detection

```bash
cd neighborhood_composition/cellular_neighborhoods

# Basic command
python cn_unified_kmeans.py \
  --tiles_dir /path/to/tiles \
  --output_dir cn_unified_results \
  --k 20 \
  --n_clusters 6

# With custom settings
python cn_unified_kmeans.py \
  --tiles_dir /path/to/tiles \
  --output_dir cn_unified_results \
  --k 20 \
  --n_clusters 6 \
  --celltype_key cell_type \
  --max_tiles 14
```

### Expected Output

```
cn_unified_results/
├── unified_analysis/
│   ├── unified_cn_composition_heatmap.png   ← ONE heatmap for ALL tiles
│   ├── unified_cn_composition.csv
│   └── unified_cn_summary.json
├── individual_tiles/
│   ├── tile1/spatial_cns.png                ← Spatial map per tile
│   ├── tile2/spatial_cns.png
│   └── ... (one per tile)
└── processed_h5ad/
    ├── tile1_adata_cns.h5ad                 ← Ready for spatial context
    ├── tile2_adata_cns.h5ad
    └── ... (one per tile)
```

### Interpreting CN Results

1. **Check the unified heatmap**: 
   - Each row = one CN
   - Each column = one cell type
   - Colors = z-scored cell type fractions
   - Look for distinct patterns (each CN should be different)

2. **Check individual spatial maps**:
   - See where each CN is located
   - Look for spatial organization (e.g., CNs at tumor edge vs. center)

3. **Adjust parameters if needed**:
   - If CNs are too similar → Decrease `n_clusters`
   - If CNs are too heterogeneous → Increase `n_clusters`
   - If neighborhoods are too local → Increase `k`
   - If neighborhoods are too global → Decrease `k`

### Typical Processing Time

| Tiles | Cells per Tile | Total Cells | Processing Time |
|-------|---------------|-------------|-----------------|
| 5 | 10,000 | 50,000 | ~5 min |
| 14 | 10,000 | 140,000 | ~15 min |
| 50 | 10,000 | 500,000 | ~45 min |
| 100 | 10,000 | 1,000,000 | ~90 min |

## Step 2: Spatial Context Detection

### What are Spatial Contexts?

Spatial contexts (SCs) are regions where different CNs interact. While CNs describe local cell type composition, SCs describe the mixture of CNs in larger neighborhoods.

**Example SCs**:
- **SC "1"**: Homogeneous CN1 region (>90% CN1)
- **SC "1_2"**: Interface between CN1 and CN2
- **SC "2_4_5"**: Complex region with CN2, CN4, and CN5 mixing

### Algorithm (Figure 19)

1. Build k=40 nearest neighbor graph (larger than CN detection)
2. For each cell, compute fraction of each CN in its 40 neighbors
3. Sort CN fractions high→low, add until sum ≥ 90%
4. SC label = sorted CN IDs joined by "_" (e.g., "1_2")
5. Filter rare SCs (< 100 cells)

### Running Spatial Context Detection

```bash
cd neighborhood_composition/spatial_contexts

# Basic command (reads from CN results)
python spatial_contexts.py \
  --batch_dir ../cellular_neighborhoods/cn_unified_results \
  --output_dir sc_unified_results \
  --mode both

# The script will automatically:
# 1. Find processed h5ad files with CN labels
# 2. Detect spatial contexts
# 3. Generate visualizations
```

### Expected Output

```
sc_unified_results/
├── aggregated/
│   ├── spatial_contexts.png                 ← All tiles with SC labels
│   ├── sc_interaction_graph.png             ← SC network graph
│   ├── sc_statistics.csv                    ← SC composition stats
│   └── aggregated_adata_scs.h5ad           ← Combined data with SC labels
└── individual_tiles/
    ├── tile1/
    │   ├── spatial_contexts.png
    │   ├── sc_interaction_graph.png
    │   └── sc_statistics.csv
    └── ... (one per tile)
```

### Interpreting SC Results

1. **Spatial context map** (`spatial_contexts.png`):
   - Each color = one SC
   - Look for spatial organization
   - SCs with single CN ("1") = homogeneous regions
   - SCs with multiple CNs ("1_2") = interfaces/transition zones

2. **Interaction graph** (`sc_interaction_graph.png`):
   - Nodes = SCs
   - Edges = SCs that share neighborhoods
   - Node size = number of cells
   - Edge width = interaction frequency

3. **Statistics** (`sc_statistics.csv`):
   - Number of cells per SC
   - Dominant CNs in each SC
   - CN composition of each SC

## Complete Example Workflow

### Example: Analyzing 14 TCGA-LUAD Tiles

```bash
# Set up paths
TILES_DIR="/mnt/g/GDC-TCGA-LUAD/00a0b174-1eab-446a-ba8c-7c6e3acd7f0c/pred/h5ad"
CN_OUTPUT="cn_luad_14tiles"
SC_OUTPUT="sc_luad_14tiles"

# Step 1: Cellular Neighborhood Detection
cd neighborhood_composition/cellular_neighborhoods
python cn_unified_kmeans.py \
  --tiles_dir $TILES_DIR \
  --output_dir $CN_OUTPUT \
  --k 20 \
  --n_clusters 6 \
  --celltype_key cell_type

# Expected: 15-30 minutes for 14 tiles

# Step 2: Spatial Context Detection
cd ../spatial_contexts
python spatial_contexts.py \
  --batch_dir ../cellular_neighborhoods/$CN_OUTPUT \
  --output_dir $SC_OUTPUT \
  --mode both

# Expected: 20-40 minutes for 14 tiles

# Step 3: Explore results
echo "Results are in:"
echo "  CN results: neighborhood_composition/cellular_neighborhoods/$CN_OUTPUT/"
echo "  SC results: neighborhood_composition/spatial_contexts/$SC_OUTPUT/"
```

### What You Get

After running both steps, you will have:

**From CN Detection**:
- 1 unified heatmap showing 6 CN compositions
- 14 spatial maps showing CN locations per tile
- 14 h5ad files with CN annotations

**From SC Detection**:
- Spatial maps showing SC labels
- Interaction graphs showing how SCs connect
- Statistics on SC composition
- h5ad files with both CN and SC annotations

## Parameter Tuning

### Cellular Neighborhood Parameters

#### k (Number of Neighbors)
- **Default**: 20
- **Effect**: Size of local neighborhood
- **Tuning**:
  - Too low (k=10): Very local, fragmented CNs
  - Too high (k=40): Very global, coarse CNs
  - **Recommendation**: Start with 20, adjust based on tissue density

#### n_clusters (Number of CNs)
- **Default**: 6
- **Effect**: Number of distinct microenvironments
- **Tuning**:
  - Too low (n=3): Over-simplified, loses detail
  - Too high (n=12): Over-fragmented, hard to interpret
  - **Recommendation**: Look at heatmap; each CN should be distinct

### Spatial Context Parameters

#### k (Number of Neighbors)
- **Default**: 40
- **Effect**: Size of spatial context window
- **Note**: Should be larger than CN detection (20)
- **Recommendation**: Keep at 40 unless studying very large/small structures

#### threshold (Cumulative Fraction)
- **Default**: 0.9
- **Effect**: How many CNs to include in SC label
- **Tuning**:
  - Higher (0.95): Stricter, more CNs per SC
  - Lower (0.85): More lenient, fewer CNs per SC
  - **Recommendation**: Keep at 0.9 for most analyses

#### min_cells (Minimum Cells per SC)
- **Default**: 100
- **Effect**: Filters out rare SCs
- **Tuning**:
  - Higher (200): Fewer, more robust SCs
  - Lower (50): More SCs, including rare ones
  - **Recommendation**: Adjust based on total cell count

## Troubleshooting

### Problem: Out of Memory

**Solution 1**: Process fewer tiles
```bash
python cn_unified_kmeans.py --tiles_dir /path --max_tiles 10
```

**Solution 2**: Process in groups
```bash
# Group 1
python cn_unified_kmeans.py --tiles_dir group1/ --output_dir cn_group1

# Group 2  
python cn_unified_kmeans.py --tiles_dir group2/ --output_dir cn_group2
```

### Problem: CNs are not distinct

**Symptom**: All CNs look similar in heatmap

**Solution**: Decrease `n_clusters`
```bash
python cn_unified_kmeans.py --tiles_dir /path --n_clusters 4
```

### Problem: Too many SCs

**Symptom**: Hundreds of SCs, hard to interpret

**Solution**: Increase `min_cells` or `threshold`
```bash
python spatial_contexts.py --batch_dir /path \
  --threshold 0.95 \  # More stringent
  --min_cells 200     # Filter more aggressively
```

### Problem: No spatial coordinates found

**Symptom**: "Warning: No spatial coordinates found"

**Solution**: Check your h5ad files
```python
import scanpy as sc
adata = sc.read_h5ad('tile.h5ad')
print(adata.obsm.keys())  # Should include 'spatial'
print(adata.obsm['spatial'].shape)  # Should be (n_cells, 2)
```

## Advanced Usage

### Custom Analysis Pipeline

```python
from cn_unified_kmeans import UnifiedCellularNeighborhoodDetector
from spatial_contexts import SpatialContextDetector
import scanpy as sc

# Step 1: CN Detection
cn_detector = UnifiedCellularNeighborhoodDetector(
    tiles_directory='/path/to/tiles',
    output_dir='cn_results'
)
tile_files = cn_detector.discover_tiles()
cn_detector.run_full_pipeline(tile_files, k=20, n_clusters=6)

# Step 2: SC Detection on specific tile
adata = sc.read_h5ad('cn_results/processed_h5ad/tile1_adata_cns.h5ad')
sc_detector = SpatialContextDetector(adata, cn_key='cn_celltype')
sc_detector.run_full_pipeline(
    k=40, threshold=0.9, min_cells=100,
    img_id_key='tile_name',
    output_dir='sc_results'
)

# Access results
print(f"CNs detected: {cn_detector.combined_adata.obs['cn_celltype'].nunique()}")
print(f"SCs detected: {adata.obs['spatial_context_filtered'].nunique()}")
```

### Batch Processing with Custom Groups

```python
from pathlib import Path

# Define tile groups
groups = {
    'tumor_core': ['tile1', 'tile2', 'tile3'],
    'invasive_front': ['tile4', 'tile5', 'tile6'],
    'normal': ['tile7', 'tile8', 'tile9']
}

# Process each group
for group_name, tile_names in groups.items():
    # Filter tiles
    tile_files = [Path(f'/path/to/{t}.h5ad') for t in tile_names]
    
    # Run CN detection
    detector = UnifiedCellularNeighborhoodDetector(
        tiles_directory='/path/to/tiles',
        output_dir=f'cn_{group_name}'
    )
    detector.run_full_pipeline(tile_files, k=20, n_clusters=6)
```

## Best Practices

1. **Always use unified CN detection** for multi-tile studies that need spatial context analysis

2. **Check the heatmap** before proceeding to spatial contexts
   - Each CN should have a distinct cell type signature
   - Adjust `n_clusters` if needed

3. **Start with defaults** (k=20, n_clusters=6, threshold=0.9, min_cells=100)
   - Only adjust if results don't make biological sense

4. **Save intermediate results**
   - CN detection outputs are input for SC detection
   - Don't delete processed h5ad files

5. **Document your parameters**
   - Record k, n_clusters, threshold in your analysis notes
   - Summary JSON files track parameters automatically

## Citation

If you use this workflow, please cite:

Schürch, C.M., Bhate, S.S., Barlow, G.L. et al. (2020). Coordinated cellular neighborhoods orchestrate antitumoral immunity at the colorectal cancer invasive front. *Cell* 182(5):1341-1359.e19. https://doi.org/10.1016/j.cell.2020.07.005

## Additional Resources

- [Script Comparison](spatial_contexts/SCRIPT_COMPARISON.md) - Compare different CN detection scripts
- [Unified CN README](spatial_contexts/README_UNIFIED_CN.md) - Detailed unified CN documentation
- [Example Scripts](spatial_contexts/example_unified_cn.py) - Usage examples

## Support

For issues:
1. Check this guide and related documentation
2. Review the paper's supplementary methods
3. Check code comments in scripts
4. Verify input data format (cell types + spatial coordinates)

## Quick Command Reference

```bash
# CN Detection (Unified)
python cn_unified_kmeans.py \
  --tiles_dir /path/to/tiles \
  --output_dir cn_results \
  --k 20 --n_clusters 6

# SC Detection
python spatial_contexts.py \
  --batch_dir ../cellular_neighborhoods/cn_results \
  --output_dir sc_results \
  --mode both

# View Examples
python example_unified_cn.py
```

