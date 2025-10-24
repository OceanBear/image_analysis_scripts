# DBSCAN-based Cellular Neighborhood Detection

This document describes the new DBSCAN clustering method added to the `CellularNeighborhoodDetector` class as an alternative to K-means clustering.

## Overview

The original cellular neighborhood detection uses **K-nearest neighbors (KNN)** for building spatial graphs and **K-means** for clustering cells based on their neighborhood composition. We've now added **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** as an alternative clustering method.

## Why DBSCAN?

DBSCAN offers several advantages over K-means:

1. **Automatic cluster discovery**: No need to specify the number of clusters beforehand
2. **Arbitrary cluster shapes**: Can identify non-spherical clusters
3. **Noise detection**: Identifies outlier cells that don't fit into any cluster
4. **Density-based**: Clusters based on local density, which can be more biologically meaningful

## Key Differences

| Feature | K-means | DBSCAN |
|---------|---------|--------|
| Number of clusters | Must specify (`n_clusters`) | Automatically determined |
| Cluster shape | Spherical/globular | Arbitrary shapes |
| Outlier handling | Forces all points into clusters | Identifies noise points |
| Parameters | `n_clusters`, `random_state` | `eps`, `min_samples` |
| Deterministic | No (random initialization) | Yes |

## New Methods

### 1. `detect_cellular_neighborhoods_dbscan()`

The core DBSCAN clustering method:

```python
detector.detect_cn_dbscan(
    eps=0.5,  # Neighborhood radius
    min_samples=5,  # Minimum cluster size
    handle_noise='separate',  # How to handle noise points
    output_key='cn_celltype_dbscan'
)
```

**Parameters:**
- `eps` (float): The maximum distance between two samples for one to be considered neighbors. **Most important parameter**. Smaller = more clusters, larger = fewer clusters.
- `min_samples` (int): Minimum number of samples in a neighborhood for a point to be a core point. Larger = denser clusters.
- `handle_noise` (str): How to handle noise points:
  - `'separate'`: Keep noise points as separate cluster (label 0)
  - `'nearest'`: Assign noise points to nearest cluster centroid
- `output_key` (str): Key to store results in `adata.obs`

### 2. `run_full_pipeline_dbscan()`

Complete pipeline using DBSCAN:

```python
detector.run_full_pipeline_dbscan(
   k=20,  # Number of neighbors for graph
   eps=0.5,  # DBSCAN epsilon
   min_samples=5,  # DBSCAN min_samples
   handle_noise='separate',
   celltype_key='cell_type',
   img_id_key='tile_name',
   output_dir='cn_results_dbscan'
)
```

This runs the complete workflow:
1. Build KNN graph
2. Aggregate neighbor cell type fractions
3. Cluster with DBSCAN
4. Generate spatial visualizations
5. Generate composition heatmaps
6. Save all results

### 3. `compare_methods()`

Compare K-means and DBSCAN side-by-side:

```python
detector.compare_methods(
    k=20,
    n_clusters=6,  # For K-means
    eps=0.5,  # For DBSCAN
    min_samples=5,  # For DBSCAN
    celltype_key='cell_type',
    img_id_key='tile_name',
    output_dir='cn/cn_comparison'
)
```

This method:
- Runs both K-means and DBSCAN
- Creates side-by-side spatial visualizations
- Creates side-by-side composition heatmaps
- Prints comparison statistics
- Saves results for both methods

## Parameter Tuning Guide

### Finding the right `eps` value

1. **Start with 0.5** as a baseline
2. **If you get too many clusters**: Increase eps (try 0.6, 0.7, 0.8)
3. **If you get too few clusters**: Decrease eps (try 0.4, 0.3)
4. **If you get too much noise**: Increase eps or decrease min_samples

### Finding the right `min_samples` value

1. **Rule of thumb**: Start with `2 * number_of_dimensions`
   - For cellular neighborhoods, start with 5-10
2. **If clusters are too fragmented**: Increase min_samples
3. **If you want to detect small clusters**: Decrease min_samples

### Recommended parameter combinations

| Use Case | eps | min_samples | Description |
|----------|-----|-------------|-------------|
| Conservative | 0.7-0.8 | 8-10 | Fewer, larger, denser clusters |
| Balanced | 0.5-0.6 | 5-7 | Good starting point |
| Aggressive | 0.3-0.4 | 3-5 | More, smaller clusters |

## Usage Examples

### Example 1: Basic DBSCAN usage

```python
import scanpy as sc
from cellular_neighborhoods import CellularNeighborhoodDetector

# Load data
adata = sc.read_h5ad('your_data.h5ad')

# Initialize detector
detector = CellularNeighborhoodDetector(adata)

# Run DBSCAN pipeline
detector.run_full_pipeline_dbscan(
    k=20,
    eps=0.5,
    min_samples=5,
    celltype_key='cell_type',
    img_id_key='sample_id',
    output_dir='cn/cn_results_dbscan'
)

# Save results
adata.write('results_with_dbscan.h5ad')
```

### Example 2: Compare methods

```python
# Compare K-means and DBSCAN
detector.compare_methods(
    k=20,
    n_clusters=6,    # K-means
    eps=0.5,         # DBSCAN
    min_samples=5,   # DBSCAN
    celltype_key='cell_type',
    img_id_key='sample_id',
    output_dir='comparison'
)

# Results are saved in:
# - adata.obs['cn_celltype_kmeans']
# - adata.obs['cn_celltype_dbscan']
```

### Example 3: Manual workflow

```python
# Step-by-step for more control
detector.build_knn_graph(k=20)
detector.aggregate_neighbors(cluster_key='cell_type')
detector.detect_cn_dbscan(
    eps=0.5,
    min_samples=5,
    output_key='my_dbscan_labels'
)
```

## Output Format

The DBSCAN method generates the same output format as K-means:

1. **Cluster labels**: Stored in `adata.obs[output_key]` as categorical
   - Labels are 1-indexed (CN 1, CN 2, etc.)
   - If `handle_noise='separate'`, noise points are labeled as 0

2. **Visualizations**:
   - `spatial_cns_dbscan.png`: Spatial plot of cellular neighborhoods
   - `cn_composition_heatmap_dbscan.png`: Heatmap of cell type composition

3. **CSV files**:
   - `cn_composition_dbscan.csv`: Raw composition fractions
   - `cn_composition_zscore_dbscan.csv`: Z-score normalized composition

## When to Use DBSCAN vs K-means

**Use DBSCAN when:**
- You don't know how many clusters to expect
- You want to identify outlier/rare cell neighborhoods
- Your data might have non-spherical clusters
- You want more biologically meaningful clusters based on density

**Use K-means when:**
- You have a specific number of neighborhoods in mind
- You want all cells assigned to a cluster (no noise)
- You need reproducibility with a random seed
- You're following a specific protocol that uses K-means

**Use both (compare_methods) when:**
- You're exploring your data for the first time
- You want to validate clustering results
- You're tuning parameters
- You want to publish both for comparison

## Algorithm Details

The DBSCAN method follows this workflow:

1. **Build KNN graph** (same as K-means): Creates spatial neighbor relationships
2. **Aggregate neighbors** (same as K-means): Computes cell type fractions in each cell's neighborhood
3. **DBSCAN clustering**: Clusters cells based on their aggregated neighbor profiles
   - Uses Euclidean distance in the neighborhood composition space
   - Identifies core points (cells in dense regions)
   - Expands clusters from core points
   - Labels remaining points as noise
4. **Handle noise points**:
   - Option 1 (`'separate'`): Keep as separate cluster
   - Option 2 (`'nearest'`): Assign to nearest cluster centroid
5. **Generate visualizations and save results**

## Troubleshooting

### Problem: All cells are noise
**Solution**: Increase `eps` or decrease `min_samples`

### Problem: Only one cluster
**Solution**: Decrease `eps` or increase `min_samples`

### Problem: Too many small clusters
**Solution**: Increase `eps` or increase `min_samples`

### Problem: Results look very different from K-means
**Explanation**: This is normal! DBSCAN finds density-based clusters while K-means finds spherical clusters. Try adjusting parameters or use `compare_methods()` to evaluate both.

## References

1. Original cellular neighborhoods paper:
   - Schürch et al. (2020) "Coordinated cellular neighborhoods orchestrate antitumoral immunity at the colorectal cancer invasive front"

2. DBSCAN algorithm:
   - Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). "A density-based algorithm for discovering clusters in large spatial databases with noise"

3. For more on DBSCAN:
   - https://www.geeksforgeeks.org/machine-learning/dbscan-clustering-in-ml-density-based-clustering/

## Example Scripts

See `cn_eg_run.py` for detailed examples including:
- Basic DBSCAN usage
- K-means usage
- Method comparison
- Manual step-by-step workflow
- Parameter tuning guide

Generated with Claude Code - 2025-10-15