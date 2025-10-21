"""
Find Optimal Number of Cellular Neighborhoods Using Elbow Method

This script uses the elbow method to determine the optimal number of
cellular neighborhoods (clusters) for spatial analysis.

Method: Elbow Method (Kneedle Algorithm)
- Finds the point of maximum distance from the line connecting first and last points
- This represents the "elbow" where adding more clusters gives diminishing returns

Author: Generated with Claude Code
Date: 2025-10-21
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from typing import Tuple, List
import warnings

warnings.filterwarnings('ignore')


def compute_inertias(
    X: np.ndarray,
    k_range: List[int] = None,
    random_state: int = 220705
) -> pd.DataFrame:
    """
    Compute inertia (within-cluster sum of squares) for different numbers of clusters.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (cells × features)
    k_range : list, optional
        Range of cluster numbers to test. Default: 2 to 15
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    inertia_df : DataFrame
        DataFrame with inertia for each k value
    """
    if k_range is None:
        k_range = range(2, 16)

    print(f"Computing inertia for k = {min(k_range)} to {max(k_range)}...")

    results = []

    for k in k_range:
        print(f"  Testing k = {k}...")

        # Fit KMeans
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)

        results.append({
            'n_clusters': k,
            'inertia': kmeans.inertia_
        })

    inertia_df = pd.DataFrame(results)

    print("\nInertia computation complete!")
    return inertia_df


def detect_elbow_point(
    inertias: np.ndarray,
    k_values: np.ndarray
) -> int:
    """
    Detect elbow point using the "kneedle" algorithm.

    This finds the point of maximum distance from the line connecting
    first and last points on the inertia curve.

    Parameters:
    -----------
    inertias : np.ndarray
        Inertia values for different k
    k_values : np.ndarray
        Corresponding k values

    Returns:
    --------
    elbow_k : int
        Detected elbow point (optimal k)
    """
    # Normalize the curve to [0, 1]
    x = (k_values - k_values.min()) / (k_values.max() - k_values.min())
    y = (inertias - inertias.min()) / (inertias.max() - inertias.min())

    # Compute the distance from each point to the line connecting first and last points
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])

    # Vector from p1 to p2
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))

    # Calculate perpendicular distance for each point
    distances = []
    for i in range(len(x)):
        point = np.array([x[i], y[i]])
        vec_to_point = point - p1
        distance = np.abs(np.cross(line_vec_norm, vec_to_point))
        distances.append(distance)

    # Elbow is at maximum distance
    elbow_idx = np.argmax(distances)
    elbow_k = k_values[elbow_idx]

    return int(elbow_k)


def visualize_elbow_analysis(
    inertia_df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 6),
    save_path: str = None
):
    """
    Visualize elbow method.

    Parameters:
    -----------
    inertia_df : DataFrame
        DataFrame with inertia values
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    fig : Figure
        Matplotlib figure object
    elbow_k : int
        Optimal k from elbow method
    """
    print("\nGenerating elbow analysis plot...")

    k_values = inertia_df['n_clusters'].values
    inertias = inertia_df['inertia'].values

    # Detect elbow point
    elbow_k = detect_elbow_point(inertias, k_values)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot inertia curve
    ax.plot(k_values, inertias, 'o-', linewidth=2, markersize=8,
            color='steelblue', label='Inertia')

    # Mark elbow point
    elbow_idx = list(k_values).index(elbow_k)
    ax.scatter([elbow_k], [inertias[elbow_idx]],
              color='red', s=400, marker='*', zorder=5,
              label=f'Elbow at k={elbow_k}',
              edgecolor='darkred', linewidth=2)

    # Draw line from first to last point
    ax.plot([k_values[0], k_values[-1]],
            [inertias[0], inertias[-1]],
            'k--', alpha=0.3, linewidth=1, label='Reference line')

    # Add vertical line at elbow
    ax.axvline(x=elbow_k, color='red', linestyle=':', alpha=0.5, linewidth=1.5)

    ax.set_xlabel('Number of Clusters (n_clusters)', fontsize=13)
    ax.set_ylabel('Within-Cluster Sum of Squares (Inertia)', fontsize=13)
    ax.set_title('Elbow Method for Optimal Cluster Number',
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)
    ax.legend(loc='best', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved to: {save_path}")

    plt.show()

    return fig, elbow_k


def run_elbow_optimization(
    adata_path: str,
    k_range: List[int] = None,
    k_neighbors: int = 20,
    celltype_key: str = 'cell_type',
    output_dir: str = 'cluster_optimization',
    random_state: int = 220705
) -> Tuple[pd.DataFrame, int]:
    """
    Complete pipeline for finding optimal number of cellular neighborhoods using elbow method.

    Parameters:
    -----------
    adata_path : str
        Path to h5ad file
    k_range : list, optional
        Range of cluster numbers to test (default: 2-15)
    k_neighbors : int, default=20
        Number of nearest neighbors for CN detection
    celltype_key : str
        Column name for cell types
    output_dir : str
        Directory to save results
    random_state : int
        Random seed

    Returns:
    --------
    inertia_df : DataFrame
        DataFrame with inertia values
    elbow_k : int
        Optimal k from elbow method
    """
    import squidpy as sq

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("ELBOW METHOD: OPTIMAL CLUSTER NUMBER DETECTION")
    print("=" * 70)
    print(f"Output directory: {output_dir}")

    # Load data
    print(f"\nLoading data from: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    print(f"  - Loaded {adata.n_obs:,} cells")

    # Build KNN graph
    print(f"\nBuilding {k_neighbors}-nearest neighbor graph...")
    sq.gr.spatial_neighbors(
        adata,
        spatial_key='spatial',
        coord_type='generic',
        n_neighs=k_neighbors,
        radius=None
    )

    # Aggregate neighbors
    print(f"Aggregating neighbors by {celltype_key}...")
    cell_types = adata.obs[celltype_key].values
    unique_types = adata.obs[celltype_key].cat.categories
    connectivity = adata.obsp['spatial_connectivities']

    n_cells = adata.n_obs
    n_types = len(unique_types)
    aggregated = np.zeros((n_cells, n_types))

    for i in range(n_cells):
        neighbors_mask = connectivity[i].toarray().flatten() > 0
        if neighbors_mask.sum() > 0:
            neighbor_types = cell_types[neighbors_mask]
            for j, ct in enumerate(unique_types):
                aggregated[i, j] = (neighbor_types == ct).sum() / neighbors_mask.sum()

    print(f"  - Feature matrix shape: {aggregated.shape}")

    # Compute inertias
    inertia_df = compute_inertias(
        aggregated,
        k_range=k_range,
        random_state=random_state
    )

    # Save inertia values
    inertia_df.to_csv(output_dir / 'inertia_values.csv', index=False)
    print(f"\n  - Saved inertia values to: {output_dir / 'inertia_values.csv'}")

    # Visualize elbow analysis
    print("\n" + "=" * 70)
    print("ELBOW ANALYSIS")
    print("=" * 70)
    fig, elbow_k = visualize_elbow_analysis(
        inertia_df,
        save_path=output_dir / 'elbow_analysis.png'
    )

    # Print recommendations
    print("\n" + "=" * 70)
    print("OPTIMAL n_clusters RECOMMENDATION")
    print("=" * 70)
    print(f"\n✓ RECOMMENDED n_clusters = {elbow_k}")
    print(f"\n   → Use n_clusters = {elbow_k} in your cellular_neighborhoods.py")

    print(f"\n" + "=" * 70)
    print("HOW IT WORKS:")
    print("=" * 70)
    print("""
The elbow method finds the point where adding more clusters provides
diminishing returns:

1. Plot inertia (within-cluster sum of squares) vs. number of clusters
2. Inertia decreases as k increases (more clusters = tighter fit)
3. The "elbow" is where the curve bends most sharply
4. This is detected as the point of maximum distance from the line
   connecting the first and last points

The optimal k balances:
- Model complexity (number of clusters)
- Goodness of fit (inertia)

Beyond the elbow, adding more clusters doesn't significantly improve
the clustering quality.
    """)

    print(f"\n{'=' * 70}")
    print("OPTIMIZATION COMPLETE!")
    print(f"{'=' * 70}")
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - elbow_analysis.png: Visualization of elbow method")
    print(f"  - inertia_values.csv: Raw inertia data")

    return inertia_df, elbow_k


# Example usage
if __name__ == "__main__":
    # Configuration
    adata_path = '../tile_39520_7904.h5ad'  # Adjust to your file
    output_dir = 'cluster_optimization'

    # Run elbow optimization
    inertia_df, optimal_k = run_elbow_optimization(
        adata_path=adata_path,
        k_range=range(2, 16),      # Test n_clusters from 2 to 15
        k_neighbors=20,             # Number of neighbors for CN detection (fixed)
        celltype_key='cell_type',   # Adjust to your column name
        output_dir=output_dir,
        random_state=220705
    )

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print(f"""
1. Review the plot: {output_dir}/elbow_analysis.png
   - The elbow point is marked with a red star
   - This is where the curve bends most sharply

2. Update cellular_neighborhoods.py:
   - Set n_clusters = {optimal_k}

3. Parameters to update in cellular_neighborhoods.py:
   - k_neighbors = 20 (number of neighbors)
   - n_clusters = {optimal_k} (number of cellular neighborhoods)

4. If you want to test nearby values:
   - Try n_clusters = {max(2, optimal_k-1)} or {optimal_k+1}
   - Run cellular neighborhoods detection with each
   - Visualize spatial patterns
   - Choose the one that makes most biological sense
    """)