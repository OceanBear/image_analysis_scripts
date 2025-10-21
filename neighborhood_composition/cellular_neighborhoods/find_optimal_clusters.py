"""
Find Optimal Number of Cellular Neighborhoods

This script provides methods to determine the optimal number of cellular
neighborhoods (clusters) for spatial analysis using multiple metrics.

Methods:
1. Elbow Method (Within-cluster Sum of Squares)
2. Silhouette Score
3. Calinski-Harabasz Index (Variance Ratio)
4. Davies-Bouldin Index
5. Gap Statistic

Author: Generated with Claude Code
Date: 2025-10-21
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score
)
from scipy.spatial.distance import cdist
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings('ignore')


def compute_cluster_metrics(
    X: np.ndarray,
    k_range: List[int] = None,
    random_state: int = 220705
) -> pd.DataFrame:
    """
    Compute multiple clustering metrics for different numbers of clusters.

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
    metrics_df : DataFrame
        DataFrame with metrics for each k value
    """
    if k_range is None:
        k_range = range(2, 16)

    print(f"Computing clustering metrics for k = {min(k_range)} to {max(k_range)}...")

    results = []

    for k in k_range:
        print(f"  Testing k = {k}...")

        # Fit KMeans
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)

        # Compute metrics
        inertia = kmeans.inertia_  # Within-cluster sum of squares
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)

        results.append({
            'n_clusters': k,
            'inertia': inertia,
            'silhouette_score': silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies_bouldin
        })

    metrics_df = pd.DataFrame(results)

    print("\nMetrics computed successfully!")
    return metrics_df


def compute_gap_statistic(
    X: np.ndarray,
    k_range: List[int] = None,
    n_refs: int = 10,
    random_state: int = 220705
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Compute gap statistic for optimal cluster number.

    Gap statistic compares within-cluster dispersion to expected dispersion
    under null reference distribution.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    k_range : list, optional
        Range of cluster numbers to test
    n_refs : int, default=10
        Number of reference datasets to generate
    random_state : int
        Random seed

    Returns:
    --------
    gap_df : DataFrame
        Gap statistics and standard errors
    optimal_k : int
        Suggested optimal number of clusters
    """
    if k_range is None:
        k_range = range(2, 16)

    print(f"\nComputing gap statistic (this may take a while)...")
    print(f"  - Testing k = {min(k_range)} to {max(k_range)}")
    print(f"  - Using {n_refs} reference datasets")

    gaps = []
    log_wks = []
    log_wks_refs = []

    for k in k_range:
        print(f"  Computing gap for k = {k}...")

        # Fit to actual data
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_

        # Compute within-cluster dispersion
        wk = 0
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                wk += np.sum(cdist(cluster_points, [centers[i]], 'euclidean'))

        log_wk = np.log(wk)
        log_wks.append(log_wk)

        # Generate reference datasets (uniform random)
        ref_log_wks = []
        for _ in range(n_refs):
            # Generate uniform random data in same range as X
            X_ref = np.random.uniform(
                X.min(axis=0),
                X.max(axis=0),
                size=X.shape
            )

            # Fit to reference data
            kmeans_ref = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels_ref = kmeans_ref.fit_predict(X_ref)
            centers_ref = kmeans_ref.cluster_centers_

            # Compute within-cluster dispersion for reference
            wk_ref = 0
            for i in range(k):
                cluster_points_ref = X_ref[labels_ref == i]
                if len(cluster_points_ref) > 0:
                    wk_ref += np.sum(cdist(cluster_points_ref, [centers_ref[i]], 'euclidean'))

            ref_log_wks.append(np.log(wk_ref))

        # Compute gap
        mean_log_wk_ref = np.mean(ref_log_wks)
        gap = mean_log_wk_ref - log_wk
        gaps.append(gap)
        log_wks_refs.append(ref_log_wks)

    # Compute standard deviations and standard errors
    gaps = np.array(gaps)
    log_wks = np.array(log_wks)

    sks = []
    for ref_wks in log_wks_refs:
        sk = np.sqrt(1 + 1/n_refs) * np.std(ref_wks)
        sks.append(sk)
    sks = np.array(sks)

    # Find optimal k using "first k where Gap(k) >= Gap(k+1) - s(k+1)"
    optimal_k = None
    for i in range(len(gaps) - 1):
        if gaps[i] >= gaps[i+1] - sks[i+1]:
            optimal_k = list(k_range)[i]
            break

    if optimal_k is None:
        optimal_k = list(k_range)[np.argmax(gaps)]

    gap_df = pd.DataFrame({
        'n_clusters': list(k_range),
        'gap': gaps,
        'gap_se': sks,
        'log_wk': log_wks
    })

    print(f"\n  Gap statistic suggests optimal k = {optimal_k}")

    return gap_df, optimal_k


def visualize_cluster_metrics(
    metrics_df: pd.DataFrame,
    gap_df: pd.DataFrame = None,
    figsize: Tuple[int, int] = (16, 10),
    save_path: str = None
):
    """
    Visualize all clustering metrics in a comprehensive plot.

    Parameters:
    -----------
    metrics_df : DataFrame
        Basic clustering metrics
    gap_df : DataFrame, optional
        Gap statistic results
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    print("\nGenerating comprehensive metrics visualization...")

    n_plots = 5 if gap_df is not None else 4
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    k_values = metrics_df['n_clusters'].values

    # Plot 1: Elbow Method (Inertia)
    ax = axes[0]
    ax.plot(k_values, metrics_df['inertia'], 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax.set_ylabel('Within-Cluster Sum of Squares', fontsize=11)
    ax.set_title('Elbow Method\n(Look for "elbow" point)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)

    # Plot 2: Silhouette Score
    ax = axes[1]
    ax.plot(k_values, metrics_df['silhouette_score'], 'o-', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax.set_ylabel('Silhouette Score', fontsize=11)
    ax.set_title('Silhouette Analysis\n(Higher is better)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Good threshold (0.5)')
    ax.legend()

    # Highlight maximum
    max_idx = metrics_df['silhouette_score'].idxmax()
    max_k = metrics_df.loc[max_idx, 'n_clusters']
    max_score = metrics_df.loc[max_idx, 'silhouette_score']
    ax.scatter([max_k], [max_score], color='red', s=200, marker='*', zorder=5,
               label=f'Max at k={int(max_k)}')
    ax.legend()

    # Plot 3: Calinski-Harabasz Index
    ax = axes[2]
    ax.plot(k_values, metrics_df['calinski_harabasz'], 'o-', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax.set_ylabel('Calinski-Harabasz Index', fontsize=11)
    ax.set_title('Calinski-Harabasz Score\n(Higher is better)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)

    # Highlight maximum
    max_idx = metrics_df['calinski_harabasz'].idxmax()
    max_k = metrics_df.loc[max_idx, 'n_clusters']
    max_score = metrics_df.loc[max_idx, 'calinski_harabasz']
    ax.scatter([max_k], [max_score], color='red', s=200, marker='*', zorder=5,
               label=f'Max at k={int(max_k)}')
    ax.legend()

    # Plot 4: Davies-Bouldin Index
    ax = axes[3]
    ax.plot(k_values, metrics_df['davies_bouldin'], 'o-', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax.set_ylabel('Davies-Bouldin Index', fontsize=11)
    ax.set_title('Davies-Bouldin Score\n(Lower is better)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)

    # Highlight minimum
    min_idx = metrics_df['davies_bouldin'].idxmin()
    min_k = metrics_df.loc[min_idx, 'n_clusters']
    min_score = metrics_df.loc[min_idx, 'davies_bouldin']
    ax.scatter([min_k], [min_score], color='red', s=200, marker='*', zorder=5,
               label=f'Min at k={int(min_k)}')
    ax.legend()

    # Plot 5: Gap Statistic (if available)
    if gap_df is not None:
        ax = axes[4]
        gap_k = gap_df['n_clusters'].values
        ax.errorbar(gap_k, gap_df['gap'], yerr=gap_df['gap_se'],
                   fmt='o-', linewidth=2, markersize=8, color='teal', capsize=5)
        ax.set_xlabel('Number of Clusters (k)', fontsize=11)
        ax.set_ylabel('Gap Statistic', fontsize=11)
        ax.set_title('Gap Statistic\n(First peak often optimal)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(gap_k)

        # Highlight suggested optimal
        optimal_k = gap_k[np.argmax(gap_df['gap'].values)]
        ax.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7,
                  label=f'Suggested k={int(optimal_k)}')
        ax.legend()

    # Plot 6: Summary table
    ax = axes[5]
    ax.axis('off')

    # Create summary text
    summary_lines = [
        "RECOMMENDED k VALUES:",
        "",
        f"Silhouette Score: k = {int(metrics_df.loc[metrics_df['silhouette_score'].idxmax(), 'n_clusters'])}",
        f"Calinski-Harabasz: k = {int(metrics_df.loc[metrics_df['calinski_harabasz'].idxmax(), 'n_clusters'])}",
        f"Davies-Bouldin: k = {int(metrics_df.loc[metrics_df['davies_bouldin'].idxmin(), 'n_clusters'])}",
    ]

    if gap_df is not None:
        optimal_gap_k = gap_df['n_clusters'].values[np.argmax(gap_df['gap'].values)]
        summary_lines.append(f"Gap Statistic: k = {int(optimal_gap_k)}")

    summary_lines.extend([
        "",
        "INTERPRETATION:",
        "• Look for consensus across metrics",
        "• Consider biological interpretability",
        "• Test k ± 1 around suggested values",
        "• Visualize spatial patterns for each k"
    ])

    summary_text = "\n".join(summary_lines)
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Cellular Neighborhood Cluster Optimization',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved to: {save_path}")

    plt.show()

    return fig


def print_recommendations(metrics_df: pd.DataFrame, gap_df: pd.DataFrame = None):
    """
    Print recommended k values based on all metrics.

    Parameters:
    -----------
    metrics_df : DataFrame
        Basic clustering metrics
    gap_df : DataFrame, optional
        Gap statistic results
    """
    print("\n" + "=" * 70)
    print("CLUSTER NUMBER RECOMMENDATIONS")
    print("=" * 70)

    # Get recommendations from each metric
    silhouette_k = int(metrics_df.loc[metrics_df['silhouette_score'].idxmax(), 'n_clusters'])
    calinski_k = int(metrics_df.loc[metrics_df['calinski_harabasz'].idxmax(), 'n_clusters'])
    davies_k = int(metrics_df.loc[metrics_df['davies_bouldin'].idxmin(), 'n_clusters'])

    print(f"\nBased on individual metrics:")
    print(f"  • Silhouette Score:      k = {silhouette_k}")
    print(f"  • Calinski-Harabasz:     k = {calinski_k}")
    print(f"  • Davies-Bouldin:        k = {davies_k}")

    if gap_df is not None:
        gap_k = int(gap_df['n_clusters'].values[np.argmax(gap_df['gap'].values)])
        print(f"  • Gap Statistic:         k = {gap_k}")
        all_recommendations = [silhouette_k, calinski_k, davies_k, gap_k]
    else:
        all_recommendations = [silhouette_k, calinski_k, davies_k]

    # Find consensus
    from collections import Counter
    recommendation_counts = Counter(all_recommendations)
    most_common_k = recommendation_counts.most_common(1)[0][0]

    print(f"\n{'=' * 70}")
    print(f"CONSENSUS RECOMMENDATION: k = {most_common_k}")
    print(f"{'=' * 70}")

    print(f"\nSuggested range to test: k ∈ [{most_common_k - 1}, {most_common_k + 1}]")

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE:")
    print("=" * 70)
    print("""
1. SILHOUETTE SCORE (range: -1 to 1)
   - Measures how similar objects are to their own cluster vs. other clusters
   - > 0.7: Strong structure
   - > 0.5: Reasonable structure
   - < 0.25: Weak or artificial structure

2. CALINSKI-HARABASZ INDEX (higher is better)
   - Ratio of between-cluster to within-cluster variance
   - Higher values indicate better-defined clusters

3. DAVIES-BOULDIN INDEX (lower is better)
   - Average similarity between clusters
   - Lower values indicate better cluster separation

4. GAP STATISTIC
   - Compares data structure to random uniform distribution
   - First local maximum often indicates optimal k

5. ELBOW METHOD
   - Look for "elbow" in inertia plot where adding clusters gives diminishing returns
   - Often subjective but useful for visual inspection

RECOMMENDATIONS:
• Use consensus k as starting point
• Visualize spatial patterns for k-1, k, and k+1
• Consider biological interpretability
• Check cluster stability across different random seeds
    """)


def run_cluster_optimization_pipeline(
    adata_path: str,
    k_range: List[int] = None,
    k_neighbors: int = 20,
    celltype_key: str = 'cell_type',
    output_dir: str = 'cluster_optimization',
    compute_gap: bool = True,
    n_refs: int = 10,
    random_state: int = 220705
):
    """
    Complete pipeline for finding optimal number of cellular neighborhoods.

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
    compute_gap : bool, default=True
        Whether to compute gap statistic (slower)
    n_refs : int, default=10
        Number of reference datasets for gap statistic
    random_state : int
        Random seed
    """
    import squidpy as sq

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("CELLULAR NEIGHBORHOOD CLUSTER OPTIMIZATION")
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

    # Compute basic metrics
    metrics_df = compute_cluster_metrics(
        aggregated,
        k_range=k_range,
        random_state=random_state
    )

    # Save metrics
    metrics_df.to_csv(output_dir / 'clustering_metrics.csv', index=False)
    print(f"\n  - Saved metrics to: {output_dir / 'clustering_metrics.csv'}")

    # Compute gap statistic (optional, slower)
    gap_df = None
    if compute_gap:
        gap_df, optimal_gap_k = compute_gap_statistic(
            aggregated,
            k_range=k_range,
            n_refs=n_refs,
            random_state=random_state
        )
        gap_df.to_csv(output_dir / 'gap_statistic.csv', index=False)
        print(f"  - Saved gap statistic to: {output_dir / 'gap_statistic.csv'}")

    # Visualize all metrics
    visualize_cluster_metrics(
        metrics_df,
        gap_df=gap_df,
        save_path=output_dir / 'cluster_optimization_metrics.png'
    )

    # Print recommendations
    print_recommendations(metrics_df, gap_df)

    print(f"\n{'=' * 70}")
    print("OPTIMIZATION COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Results saved to: {output_dir}/")

    return metrics_df, gap_df


# Example usage
if __name__ == "__main__":
    # Configuration
    adata_path = '../tile_39520_7904.h5ad'  # Adjust to your file
    output_dir = 'cluster_optimization'

    # Run optimization pipeline
    metrics_df, gap_df = run_cluster_optimization_pipeline(
        adata_path=adata_path,
        k_range=range(2, 16),      # Test k from 2 to 15
        k_neighbors=20,             # Number of neighbors for CN detection
        celltype_key='cell_type',   # Adjust to your column name
        output_dir=output_dir,
        compute_gap=True,           # Set to False to skip gap statistic (faster)
        n_refs=10,                  # Number of reference datasets for gap statistic
        random_state=220705
    )

    print("\nNext steps:")
    print("  1. Review the plots in 'cluster_optimization/' folder")
    print("  2. Choose k based on consensus recommendation")
    print("  3. Test k-1, k, and k+1 to see which gives most interpretable results")
    print("  4. Update n_clusters in your cellular_neighborhoods.py accordingly")