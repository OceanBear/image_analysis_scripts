"""
Find Optimal Number of Neighbors (k) for Cellular Neighborhood Detection

This script finds the optimal number of neighbors (k) for a given number of
cellular neighborhoods (n_clusters) using the Silhouette Score.

Method: Silhouette Score Maximization
- Measures how well-separated the clusters are
- Score ranges from -1 to 1 (higher is better)
- > 0.5: Good cluster separation
- > 0.7: Strong cluster separation

Author: Generated with Claude Code
Date: 2025-10-21
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple, List
import warnings

warnings.filterwarnings('ignore')


def compute_silhouette_scores(
    adata: sc.AnnData,
    k_neighbors_range: List[int],
    n_clusters: int,
    celltype_key: str = 'cell_type',
    random_state: int = 220705
) -> pd.DataFrame:
    """
    Compute silhouette scores for different numbers of neighbors.

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
        DataFrame with silhouette scores for each k_neighbors value
    """
    import squidpy as sq

    print(f"Computing silhouette scores for k = {min(k_neighbors_range)} to {max(k_neighbors_range)}...")
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

        # Compute silhouette score
        silhouette = silhouette_score(aggregated, labels)

        results.append({
            'k_neighbors': k,
            'silhouette_score': silhouette
        })

    scores_df = pd.DataFrame(results)

    print("\nSilhouette score computation complete!")
    return scores_df


def visualize_silhouette_analysis(
    scores_df: pd.DataFrame,
    n_clusters: int,
    figsize: Tuple[int, int] = (10, 6),
    save_path: str = None
):
    """
    Visualize silhouette scores vs. number of neighbors.

    Parameters:
    -----------
    scores_df : DataFrame
        DataFrame with silhouette scores
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
        Optimal k_neighbors from silhouette score
    """
    print("\nGenerating silhouette analysis plot...")

    k_values = scores_df['k_neighbors'].values
    silhouettes = scores_df['silhouette_score'].values

    # Find optimal k (maximum silhouette score)
    optimal_idx = np.argmax(silhouettes)
    optimal_k = k_values[optimal_idx]
    optimal_score = silhouettes[optimal_idx]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot silhouette scores
    ax.plot(k_values, silhouettes, 'o-', linewidth=2, markersize=8,
            color='steelblue', label='Silhouette Score')

    # Mark optimal point
    ax.scatter([optimal_k], [optimal_score],
              color='red', s=400, marker='*', zorder=5,
              label=f'Optimal k={optimal_k} (score={optimal_score:.3f})',
              edgecolor='darkred', linewidth=2)

    # Add vertical line at optimal k
    ax.axvline(x=optimal_k, color='red', linestyle=':', alpha=0.5, linewidth=1.5)

    # Add horizontal reference lines
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.3, linewidth=1,
               label='Good threshold (0.5)')
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.3, linewidth=1,
               label='Strong threshold (0.7)')

    ax.set_xlabel('Number of Neighbors (k)', fontsize=13)
    ax.set_ylabel('Silhouette Score', fontsize=13)
    ax.set_title(f'Optimal k for n_clusters={n_clusters}\n(Silhouette Score Maximization)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values[::2])  # Show every other tick to avoid crowding
    ax.legend(loc='best', fontsize=11)

    # Set y-axis limits for better visualization
    ax.set_ylim([max(0, silhouettes.min() - 0.05), min(1, silhouettes.max() + 0.05)])

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

    # Compute silhouette scores
    scores_df = compute_silhouette_scores(
        adata,
        k_neighbors_range=k_neighbors_range,
        n_clusters=n_clusters,
        celltype_key=celltype_key,
        random_state=random_state
    )

    # Save silhouette scores
    scores_df.to_csv(output_dir / 'silhouette_scores.csv', index=False)
    print(f"\n  - Saved silhouette scores to: {output_dir / 'silhouette_scores.csv'}")

    # Visualize silhouette analysis
    print("\n" + "=" * 70)
    print("SILHOUETTE ANALYSIS")
    print("=" * 70)
    fig, optimal_k = visualize_silhouette_analysis(
        scores_df,
        n_clusters=n_clusters,
        save_path=output_dir / 'k_neighbors_analysis.png'
    )

    # Print recommendations
    optimal_row = scores_df[scores_df['k_neighbors'] == optimal_k].iloc[0]
    optimal_score = optimal_row['silhouette_score']

    print("\n" + "=" * 70)
    print("OPTIMAL k_neighbors RECOMMENDATION")
    print("=" * 70)
    print(f"\n✓ RECOMMENDED k_neighbors = {optimal_k}")
    print(f"  Silhouette Score: {optimal_score:.3f}")

    # Interpret the score
    if optimal_score > 0.7:
        quality = "EXCELLENT - Strong cluster separation"
    elif optimal_score > 0.5:
        quality = "GOOD - Reasonable cluster separation"
    elif optimal_score > 0.25:
        quality = "FAIR - Weak cluster structure"
    else:
        quality = "POOR - No meaningful cluster structure"

    print(f"  Quality: {quality}")
    print(f"\n   → Use k_neighbors = {optimal_k} in your cellular_neighborhoods.py")

    print(f"\n" + "=" * 70)
    print("HOW IT WORKS:")
    print("=" * 70)
    print("""
The silhouette score measures how well-separated the clusters are:

1. For each k_neighbors value:
   - Build spatial graph with k nearest neighbors
   - Aggregate neighbor cell type composition
   - Cluster cells into n_clusters groups using KMeans
   - Compute silhouette score

2. Silhouette score interpretation:
   - Ranges from -1 to 1
   - Higher scores = better cluster separation
   - Score considers both:
     • How close cells are to their own cluster
     • How far cells are from other clusters

3. Optimal k maximizes the silhouette score:
   - Too few neighbors: Noisy, unstable clusters
   - Too many neighbors: Over-smoothed, loses local structure
   - Optimal k: Best balance for cluster separation
    """)

    print(f"\n{'=' * 70}")
    print("OPTIMIZATION COMPLETE!")
    print(f"{'=' * 70}")
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - k_neighbors_analysis.png: Visualization of silhouette scores")
    print(f"  - silhouette_scores.csv: Raw silhouette score data")

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