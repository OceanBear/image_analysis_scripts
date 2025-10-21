"""
Find Optimal Number of Neighbors (k) for Cellular Neighborhood Detection

This script finds the optimal number of neighbors (k) for a given number of
cellular neighborhoods (n_clusters) using the Calinski-Harabasz Index.

Method: Calinski-Harabasz Index Maximization
- Ratio of between-cluster to within-cluster variance
- Higher values indicate better-defined, more distinct clusters
- Less biased toward small k than silhouette score
- Better for capturing meaningful neighborhood structure

Author: Generated with Claude Code
Date: 2025-10-21
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from typing import Tuple, List
import warnings

warnings.filterwarnings('ignore')


def compute_calinski_harabasz_scores(
    adata: sc.AnnData,
    k_neighbors_range: List[int],
    n_clusters: int,
    celltype_key: str = 'cell_type',
    random_state: int = 220705
) -> pd.DataFrame:
    """
    Compute Calinski-Harabasz scores for different numbers of neighbors.

    Parameters:
    -----------
    adata : AnnData
        AnnData object with spatial coordinates
    k_neighbors_range : list
        Range of k neighbors to test
    n_clusters : int
        Fixed number of clusters
    celltype_key : str
        Column name for cell types
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    scores_df : DataFrame
        DataFrame with Calinski-Harabasz scores for each k_neighbors value
    """
    import squidpy as sq

    print(f"Computing Calinski-Harabasz scores for k = {min(k_neighbors_range)} to {max(k_neighbors_range)}...")
    print(f"  - Fixed n_clusters = {n_clusters}")

    results = []

    cell_types = adata.obs[celltype_key].values
    unique_types = adata.obs[celltype_key].cat.categories
    n_types = len(unique_types)

    for k in k_neighbors_range:
        print(f"  Testing k_neighbors = {k}...")

        # Build spatial graph with k neighbors
        sq.gr.spatial_neighbors(
            adata,
            spatial_key='spatial',
            coord_type='generic',
            n_neighs=k,
            radius=None
        )

        # Aggregate neighbors by cell type
        connectivity = adata.obsp['spatial_connectivities']
        n_cells = adata.n_obs
        aggregated = np.zeros((n_cells, n_types))

        for i in range(n_cells):
            neighbors_mask = connectivity[i].toarray().flatten() > 0
            if neighbors_mask.sum() > 0:
                neighbor_types = cell_types[neighbors_mask]
                for j, ct in enumerate(unique_types):
                    aggregated[i, j] = (neighbor_types == ct).sum() / neighbors_mask.sum()

        # Cluster with fixed n_clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(aggregated)

        # Compute Calinski-Harabasz score
        ch_score = calinski_harabasz_score(aggregated, labels)

        results.append({
            'k_neighbors': k,
            'calinski_harabasz_score': ch_score
        })

    scores_df = pd.DataFrame(results)

    print("\nCalinski-Harabasz score computation complete!")
    return scores_df


def visualize_calinski_analysis(
    scores_df: pd.DataFrame,
    n_clusters: int,
    figsize: Tuple[int, int] = (10, 6),
    save_path: str = None
):
    """
    Visualize Calinski-Harabasz scores vs. number of neighbors.

    Parameters:
    -----------
    scores_df : DataFrame
        DataFrame with Calinski-Harabasz scores
    n_clusters : int
        Number of clusters used
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    fig : Figure
        Matplotlib figure object
    optimal_k : int
        Optimal k_neighbors from Calinski-Harabasz score
    """
    print("\nGenerating Calinski-Harabasz analysis plot...")

    k_values = scores_df['k_neighbors'].values
    ch_scores = scores_df['calinski_harabasz_score'].values

    # Find optimal k (maximum Calinski-Harabasz score)
    optimal_idx = np.argmax(ch_scores)
    optimal_k = k_values[optimal_idx]
    optimal_score = ch_scores[optimal_idx]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot Calinski-Harabasz scores
    ax.plot(k_values, ch_scores, 'o-', linewidth=2, markersize=8,
            color='steelblue', label='Calinski-Harabasz Score')

    # Mark optimal point
    ax.scatter([optimal_k], [optimal_score],
              color='red', s=400, marker='*', zorder=5,
              label=f'Optimal k={optimal_k} (score={optimal_score:.1f})',
              edgecolor='darkred', linewidth=2)

    # Add vertical line at optimal k
    ax.axvline(x=optimal_k, color='red', linestyle=':', alpha=0.5, linewidth=1.5)

    ax.set_xlabel('Number of Neighbors (k)', fontsize=13)
    ax.set_ylabel('Calinski-Harabasz Index', fontsize=13)
    ax.set_title(f'Optimal k for n_clusters={n_clusters}\n(Calinski-Harabasz Index Maximization)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values[::2])  # Show every other tick to avoid crowding
    ax.legend(loc='best', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved to: {save_path}")

    plt.show()

    return fig, optimal_k


def run_k_neighbors_optimization(
    adata_path: str,
    n_clusters: int = 6,
    k_neighbors_range: List[int] = None,
    celltype_key: str = 'cell_type',
    output_dir: str = 'k_neighbors_optimization',
    random_state: int = 220705
) -> Tuple[pd.DataFrame, int]:
    """
    Complete pipeline for finding optimal number of neighbors (k) for cellular neighborhoods.

    Parameters:
    -----------
    adata_path : str
        Path to h5ad file
    n_clusters : int, default=6
        Fixed number of clusters
    k_neighbors_range : list, optional
        Range of k neighbors to test (default: 5-50)
    celltype_key : str
        Column name for cell types
    output_dir : str
        Directory to save results
    random_state : int
        Random seed

    Returns:
    --------
    scores_df : DataFrame
        DataFrame with silhouette scores
    optimal_k : int
        Optimal k_neighbors from silhouette score
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("OPTIMAL k (NEIGHBORS) DETECTION")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Fixed n_clusters: {n_clusters}")

    # Load data
    print(f"\nLoading data from: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    print(f"  - Loaded {adata.n_obs:,} cells")

    # Set default k_neighbors range if not provided
    if k_neighbors_range is None:
        k_neighbors_range = list(range(5, 51, 5))  # 5, 10, 15, ..., 50
        print(f"\nUsing default k_neighbors range: {k_neighbors_range}")

    # Compute Calinski-Harabasz scores
    scores_df = compute_calinski_harabasz_scores(
        adata,
        k_neighbors_range=k_neighbors_range,
        n_clusters=n_clusters,
        celltype_key=celltype_key,
        random_state=random_state
    )

    # Save scores
    scores_df.to_csv(output_dir / 'calinski_harabasz_scores.csv', index=False)
    print(f"\n  - Saved Calinski-Harabasz scores to: {output_dir / 'calinski_harabasz_scores.csv'}")

    # Visualize Calinski-Harabasz analysis
    print("\n" + "=" * 70)
    print("CALINSKI-HARABASZ ANALYSIS")
    print("=" * 70)
    fig, optimal_k = visualize_calinski_analysis(
        scores_df,
        n_clusters=n_clusters,
        save_path=output_dir / 'k_neighbors_analysis.png'
    )

    # Print recommendations
    optimal_row = scores_df[scores_df['k_neighbors'] == optimal_k].iloc[0]
    optimal_score = optimal_row['calinski_harabasz_score']

    print("\n" + "=" * 70)
    print("OPTIMAL k_neighbors RECOMMENDATION")
    print("=" * 70)
    print(f"\n✓ RECOMMENDED k_neighbors = {optimal_k}")
    print(f"  Calinski-Harabasz Score: {optimal_score:.1f}")
    print(f"\n   → Use k_neighbors = {optimal_k} in your cellular_neighborhoods.py")

    print(f"\n" + "=" * 70)
    print("HOW IT WORKS:")
    print("=" * 70)
    print("""
The Calinski-Harabasz Index measures cluster quality:

1. For each k_neighbors value:
   - Build spatial graph with k nearest neighbors
   - Aggregate neighbor cell type composition
   - Cluster cells into n_clusters groups using KMeans
   - Compute Calinski-Harabasz score

2. Calinski-Harabasz Index (Variance Ratio):
   - Ratio of between-cluster to within-cluster variance
   - Higher scores = better-defined, more distinct clusters
   - Score considers:
     • How separated clusters are from each other (between variance)
     • How tight/compact clusters are internally (within variance)

3. Why Calinski-Harabasz for k optimization:
   - Less biased toward small k than silhouette score
   - Better captures meaningful neighborhood structure
   - Balances cluster separation with cluster coherence

4. Optimal k maximizes the Calinski-Harabasz score:
   - Too few neighbors: Noisy features, poor cluster definition
   - Too many neighbors: Over-smoothed, loses local structure
   - Optimal k: Best variance ratio for distinct neighborhoods
    """)

    print(f"\n{'=' * 70}")
    print("OPTIMIZATION COMPLETE!")
    print(f"{'=' * 70}")
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - k_neighbors_analysis.png: Visualization of Calinski-Harabasz scores")
    print(f"  - calinski_harabasz_scores.csv: Raw score data")

    return scores_df, optimal_k


# Example usage
if __name__ == "__main__":
    # Configuration
    adata_path = '../tile_39520_7904.h5ad'  # Adjust to your file
    output_dir = 'k_neighbors_optimization'

    # Run k_neighbors optimization
    scores_df, optimal_k = run_k_neighbors_optimization(
        adata_path=adata_path,
        n_clusters=6,                           # Fixed number of clusters (from elbow method)
        k_neighbors_range=list(range(5, 51, 5)),  # Test k from 5 to 50 in steps of 5
        celltype_key='cell_type',               # Adjust to your column name
        output_dir=output_dir,
        random_state=220705
    )

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print(f"""
1. Review the plot: {output_dir}/k_neighbors_analysis.png
   - The optimal k is marked with a red star
   - Check if the silhouette score is > 0.5 (good) or > 0.7 (excellent)

2. Update cellular_neighborhoods.py:
   - Set k_neighbors = {optimal_k}
   - Keep n_clusters = 6 (from elbow method)

3. If you want to fine-tune:
   - If the peak is broad, any k nearby is acceptable
   - If the peak is sharp, stick to the optimal k
   - Consider testing k ± 2 around the optimal value

4. Final parameters for cellular_neighborhoods.py:
   - k_neighbors = {optimal_k} (number of neighbors)
   - n_clusters = 6 (number of cellular neighborhoods)
    """)