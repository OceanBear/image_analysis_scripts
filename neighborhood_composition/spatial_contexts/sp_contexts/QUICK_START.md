# Quick Start Guide

**Goal**: Detect Cellular Neighborhoods and Spatial Contexts from multiple tiles

## TL;DR

```bash
# 1. Unified CN Detection (10-100 tiles → 1 shared CN composition)
cd cellular_neighborhoods
python cn_unified_kmeans.py --tiles_dir /path/to/tiles --output_dir cn_results

# 2. Spatial Context Detection (uses CN results)
cd ../spatial_contexts
python spatial_contexts.py --batch_dir ../cellular_neighborhoods/cn_results
```

## What You Need

✅ Multiple h5ad tiles with:
- Cell type annotations (`adata.obs['cell_type']`)
- Spatial coordinates (`adata.obsm['spatial']`)

## What You Get

**From Step 1 (CN Detection)**:
- 📊 **1 unified heatmap** showing 6 CN compositions across all tiles
- 🗺️ **N spatial maps** showing CN locations (one per tile)
- 💾 **N h5ad files** with CN labels ready for spatial context analysis

**From Step 2 (SC Detection)**:
- 🗺️ **Spatial context maps** showing where CNs interact
- 🕸️ **Interaction graphs** showing SC relationships
- 📈 **Statistics** on SC composition

## Visual Workflow

```
INPUT: Multiple h5ad Tiles
         ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
         │Tile1│ │Tile2│ │Tile3│ │ ... │
         └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
            └────────┴────────┴────────┘
                      │
                      ▼
         ╔═══════════════════════════╗
         ║  Step 1: CN Detection     ║
         ║  cn_unified_kmeans.py     ║
         ║                           ║
         ║  k=20, n_clusters=6       ║
         ╚═══════════════════════════╝
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐
   │ Unified │ │Spatial  │ │ H5AD    │
   │ Heatmap │ │  Maps   │ │ Files   │
   │ (1 file)│ │(N files)│ │(N files)│
   └─────────┘ └─────────┘ └────┬────┘
                                 │
                                 ▼
         ╔═══════════════════════════╗
         ║  Step 2: SC Detection     ║
         ║  spatial_contexts.py      ║
         ║                           ║
         ║  k=40, threshold=0.9      ║
         ╚═══════════════════════════╝
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐
   │   SC    │ │   SC    │ │   SC    │
   │  Maps   │ │ Graphs  │ │  Stats  │
   └─────────┘ └─────────┘ └─────────┘
```

## Step-by-Step

### Step 0: Check Your Data

```python
import scanpy as sc

# Load one tile to verify
adata = sc.read_h5ad('/path/to/tile1.h5ad')

# Check required fields
print("Cell types:", adata.obs['cell_type'].unique())  # Should work
print("Spatial shape:", adata.obsm['spatial'].shape)   # Should be (n_cells, 2)
```

### Step 1: Cellular Neighborhood Detection

```bash
cd cellular_neighborhoods

# Run unified CN detection
python cn_unified_kmeans.py \
  --tiles_dir /path/to/tiles \
  --output_dir cn_results \
  --k 20 \
  --n_clusters 6
```

⏱️ **Time**: ~15-30 min for 14 tiles  
💾 **Output**: `cn_results/`

**Check results**:
1. Open `cn_results/unified_analysis/unified_cn_composition_heatmap.png`
2. Each CN (row) should have distinct cell type composition
3. If CNs look too similar → decrease `n_clusters`
4. If CNs are too heterogeneous → increase `n_clusters`

### Step 2: Spatial Context Detection

```bash
cd ../spatial_contexts

# Run SC detection
python spatial_contexts.py \
  --batch_dir ../cellular_neighborhoods/cn_results \
  --output_dir sc_results
```

⏱️ **Time**: ~20-40 min for 14 tiles  
💾 **Output**: `sc_results/`

**Check results**:
1. Open `sc_results/aggregated/spatial_contexts.png`
2. Each color represents a spatial context
3. Look for spatial organization (interfaces, regions)

## Output Structure

```
cn_results/
├── unified_analysis/
│   └── unified_cn_composition_heatmap.png  ← Check this first!
├── individual_tiles/
│   └── tile{N}/spatial_cns.png             ← One per tile
└── processed_h5ad/
    └── tile{N}_adata_cns.h5ad              ← Input for Step 2

sc_results/
├── aggregated/
│   ├── spatial_contexts.png                ← SC map
│   └── sc_interaction_graph.png            ← SC network
└── individual_tiles/
    └── tile{N}/...                         ← Per-tile results
```

## Common Commands

### Test with 5 tiles first
```bash
python cn_unified_kmeans.py --tiles_dir /path --max_tiles 5 --output_dir test
```

### Use fewer CNs for simpler tissue
```bash
python cn_unified_kmeans.py --tiles_dir /path --n_clusters 4
```

### Use more CNs for complex tissue
```bash
python cn_unified_kmeans.py --tiles_dir /path --n_clusters 10
```

### Different cell type column name
```bash
python cn_unified_kmeans.py --tiles_dir /path --celltype_key celltype
```

## Troubleshooting

### ❌ "Out of memory"
```bash
# Process fewer tiles
python cn_unified_kmeans.py --tiles_dir /path --max_tiles 10
```

### ❌ "No spatial coordinates found"
```python
# Check your data
import scanpy as sc
adata = sc.read_h5ad('tile.h5ad')
print(adata.obsm.keys())  # Should include 'spatial'
```

### ❌ "Cell type column not found"
```bash
# Specify the correct column name
python cn_unified_kmeans.py --tiles_dir /path --celltype_key celltype
```

### ❌ CNs are not distinct
```bash
# Decrease number of clusters
python cn_unified_kmeans.py --tiles_dir /path --n_clusters 4
```

## Next Steps

After running both steps:

1. **Interpret CNs**: What cell types define each CN?
2. **Interpret SCs**: Where do CNs interact?
3. **Biological questions**:
   - Are certain CNs enriched in specific tissue regions?
   - Do certain SCs correlate with clinical outcomes?
   - How do CNs and SCs differ between samples?

## More Information

- 📖 **Complete workflow**: [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)
- 🔍 **Compare scripts**: [SCRIPT_COMPARISON.md](spatial_contexts/SCRIPT_COMPARISON.md)
- 📚 **Detailed docs**: [README_UNIFIED_CN.md](spatial_contexts/README_UNIFIED_CN.md)
- 💡 **Examples**: [example_unified_cn.py](spatial_contexts/example_unified_cn.py)

## Help

### I have...
- **1 tile** → Use `cn_kmeans_tiled.py` instead
- **Independent samples** → Use `cn_batch_kmeans.py` instead
- **Same experiment, multiple tiles** → ✅ Use `cn_unified_kmeans.py`

### I want to...
- **Compare CNs across tiles** → ✅ Use unified approach
- **Do spatial context analysis** → ✅ Use unified approach
- **Explore each tile separately** → Use batch approach

## Example: TCGA-LUAD (14 tiles)

```bash
# Real example with actual paths
TILES="/mnt/g/GDC-TCGA-LUAD/00a0b174-1eab-446a-ba8c-7c6e3acd7f0c/pred/h5ad"

# Step 1: CN Detection
python cn_unified_kmeans.py \
  --tiles_dir $TILES \
  --output_dir cn_luad_14tiles \
  --k 20 --n_clusters 6

# Step 2: SC Detection  
python spatial_contexts.py \
  --batch_dir cn_luad_14tiles \
  --output_dir sc_luad_14tiles
```

**Results**:
- 1 unified heatmap with 6 CNs
- 14 spatial CN maps
- 14 spatial SC maps
- SC interaction graph

## Citation

Schürch et al. (2020) "Coordinated cellular neighborhoods orchestrate antitumoral immunity at the colorectal cancer invasive front" *Cell* 182(5):1341-1359.e19

---

**Ready to start?** Run the commands above! 🚀

