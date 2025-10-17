import squidpy as sq
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_and_apply_cell_type_colors(adata, celltype_key='cell_type'):
    """Convert hex colors from h5ad to matplotlib RGB tuples."""
    # Colors already saved by data_preparation.py in hex format
    if f'{celltype_key}_colors' in adata.uns:
        colors = adata.uns[f'{celltype_key}_colors']
        # Convert hex strings to RGB tuples for matplotlib
        if isinstance(colors[0], str):
            adata.uns[f'{celltype_key}_colors'] = [
                tuple(int(c.lstrip('#')[i:i+2], 16)/255.0 for i in (0, 2, 4))
                for c in colors
            ]
        print(f"  - Using cell type colors from h5ad file")
    else:
        print("  - Warning: Colors not found in h5ad file")


def build_spatial_graph(adata, method='radius', radius=50, n_neighbors=6, coord_type='generic'):
    """
    Build spatial neighborhood graph for cells.

    Parameters:
    -----------
    adata : AnnData
        AnnData object with spatial coordinates
    method : str, default='radius'
        Method to build graph: 'radius' or 'knn'
        - 'radius': All cells within radius distance
        - 'knn': K nearest neighbors
    radius : float, default=50
        Radius for neighborhood definition (in pixels)
        Typical cell diameter is 10-30 pixels, so 50 captures immediate neighbors
    n_neighbors : int, default=6
        Number of neighbors for KNN method
    coord_type : str, default='generic'
        Coordinate type for Squidpy

    Returns:
    --------
    adata : AnnData
        Modified in place with spatial graph added
    """

    print(f"Building spatial graph using {method} method...")

    if method == 'radius':
        sq.gr.spatial_neighbors(
            adata,
            spatial_key='spatial',
            coord_type=coord_type,
            radius=radius,
            n_rings=1
            # n_neighs is omitted - radius method finds all neighbors within radius
        )
        print(f"  - Using radius: {radius} pixels")

    elif method == 'knn':
        sq.gr.spatial_neighbors(
            adata,
            spatial_key='spatial',
            coord_type=coord_type,
            n_neighs=n_neighbors,
            radius=None
        )
        print(f"  - Using K-nearest neighbors: {n_neighbors}")

    # Print connectivity statistics
    connectivity = adata.obsp['spatial_connectivities']
    avg_neighbors = connectivity.sum(axis=1).mean()
    print(f"  - Average neighbors per cell: {avg_neighbors:.2f}")
    print(f"  - Connectivity matrix shape: {connectivity.shape}")

    return adata


def neighborhood_enrichment_analysis(adata, cluster_key='cell_type', n_perms=1000, seed=42):
    """
    Perform neighborhood enrichment analysis to identify cell type interactions.

    Parameters:
    -----------
    adata : AnnData
        AnnData object with spatial graph
    cluster_key : str, default='cell_type'
        Key in adata.obs containing cell type labels
    n_perms : int, default=1000
        Number of permutations for statistical testing
    seed : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    adata : AnnData
        Modified in place with enrichment results
    """

    print(f"\nPerforming neighborhood enrichment analysis...")
    print(f"  - Cluster key: {cluster_key}")
    print(f"  - Number of permutations: {n_perms}")

    # Compute neighborhood enrichment
    sq.gr.nhood_enrichment(
        adata,
        cluster_key=cluster_key,
        n_perms=n_perms,
        seed=seed
    )

    print(f"  - Enrichment analysis complete!")
    print(f"  - Results stored in adata.uns['{cluster_key}_nhood_enrichment']")

    return adata


def compute_co_occurrence(adata, cluster_key='cell_type', n_splits=20):
    """
    Compute cell type co-occurrence scores.

    Parameters:
    -----------
    adata : AnnData
        AnnData object with spatial coordinates
    cluster_key : str, default='cell_type'
        Key in adata.obs containing cell type labels
    n_splits : int, default=20
        Number of splits for interval computation

    Returns:
    --------
    adata : AnnData
        Modified in place with co-occurrence results
    """

    print(f"\nComputing co-occurrence analysis...")

    sq.gr.co_occurrence(
        adata,
        cluster_key=cluster_key,
        spatial_key='spatial',
        n_splits=n_splits
    )

    print(f"  - Co-occurrence analysis complete!")
    print(f"  - Results stored in adata.uns['{cluster_key}_co_occurrence']")

    return adata


def compute_centrality_scores(adata, cluster_key='cell_type'):
    """
    Compute network centrality scores for each cell type.

    Parameters:
    -----------
    adata : AnnData
        AnnData object with spatial graph
    cluster_key : str, default='cell_type'
        Key in adata.obs containing cell type labels

    Returns:
    --------
    adata : AnnData
        Modified in place with centrality scores
    """

    print(f"\nComputing centrality scores...")

    sq.gr.centrality_scores(
        adata,
        cluster_key=cluster_key
    )

    # Print summary of centrality scores
    centrality_cols = [col for col in adata.obs.columns if 'centrality' in col]
    print(f"  - Computed centrality metrics: {centrality_cols}")

    return adata


def visualize_enrichment(adata, cluster_key='cell_type', figsize=(10, 8), save_path=None):
    """
    Visualize neighborhood enrichment as a heatmap.

    Parameters:
    -----------
    adata : AnnData
        AnnData object with enrichment results
    cluster_key : str, default='cell_type'
        Key for cell type labels
    figsize : tuple, default=(10, 8)
        Figure size
    save_path : str, optional
        Path to save figure
    """

    print(f"\nVisualizing neighborhood enrichment...")

    fig, ax = plt.subplots(figsize=figsize)

    sq.pl.nhood_enrichment(
        adata,
        cluster_key=cluster_key,
        method='ward',
        cmap='coolwarm',
        vmin=-3,
        vmax=3,
        ax=ax
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved to: {save_path}")

    plt.show()

    return fig


def visualize_spatial_distribution(adata, cluster_key='cell_type', figsize=(12, 10),
                                   size=3, save_path=None):
    """
    Visualize spatial distribution of cell types.

    Parameters:
    -----------
    adata : AnnData
        AnnData object with spatial coordinates
    cluster_key : str, default='cell_type'
        Key for cell type labels
    figsize : tuple, default=(12, 10)
        Figure size
    size : float, default=3
        Point size for scatter plot
    save_path : str, optional
        Path to save figure
    """

    print(f"\nVisualizing spatial distribution...")

    fig, ax = plt.subplots(figsize=figsize)

    # Get spatial coordinates
    coords = adata.obsm['spatial']

    # Get cell types and colors
    cell_types = adata.obs[cluster_key]

    # Use colors from adata.uns if available, otherwise use default palette
    if f'{cluster_key}_colors' in adata.uns:
        colors = adata.uns[f'{cluster_key}_colors']
        # Create a color map for each category
        unique_types = cell_types.cat.categories
        color_map = {ct: colors[i] for i, ct in enumerate(unique_types)}
    else:
        # Use default color palette
        unique_types = cell_types.cat.categories
        palette = sns.color_palette('tab10', n_colors=len(unique_types))
        color_map = {ct: palette[i] for i, ct in enumerate(unique_types)}

    # Plot each cell type separately for legend
    for cell_type in unique_types:
        mask = cell_types == cell_type
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[color_map[cell_type]],
                   label=cell_type,
                   s=size,
                   alpha=0.7)

    ax.set_xlabel('X coordinate (pixels)', fontsize=12)
    ax.set_ylabel('Y coordinate (pixels)', fontsize=12)
    ax.set_title('Spatial Distribution of Cell Types', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved to: {save_path}")

    plt.close()

    return fig


def summarize_interactions(adata, cluster_key='cell_type', threshold=2.0):
    """
    Summarize significant cell-cell interactions.

    Parameters:
    -----------
    adata : AnnData
        AnnData object with enrichment results
    cluster_key : str, default='cell_type'
        Key for cell type labels
    threshold : float, default=2.0
        Z-score threshold for significance

    Returns:
    --------
    interactions_df : DataFrame
        Summary of significant interactions
    """

    print(f"\nSummarizing cell-cell interactions (threshold: |z| > {threshold})...")

    # Get enrichment z-scores
    zscore = adata.uns[f'{cluster_key}_nhood_enrichment']['zscore']

    # Get cell type names from the categorical data
    cell_types = adata.obs[cluster_key].cat.categories.tolist()

    # Convert to numpy array if it's not already
    if isinstance(zscore, pd.DataFrame):
        zscore_array = zscore.values
    else:
        zscore_array = np.array(zscore)

    # Find significant interactions
    interactions = []
    for i, ct1 in enumerate(cell_types):
        for j, ct2 in enumerate(cell_types):
            z = zscore_array[i, j]
            if abs(z) > threshold:
                interaction_type = "Attraction" if z > 0 else "Avoidance"
                interactions.append({
                    'Cell Type 1': ct1,
                    'Cell Type 2': ct2,
                    'Z-score': float(z),
                    'Interaction': interaction_type
                })

    interactions_df = pd.DataFrame(interactions).sort_values('Z-score',
                                                             key=abs,
                                                             ascending=False)

    print(f"  - Found {len(interactions_df)} significant interactions")
    print(f"\nTop interactions:")
    if len(interactions_df) > 0:
        print(interactions_df.head(10).to_string(index=False))
    else:
        print("  No significant interactions found above threshold")

    return interactions_df


# Main analysis pipeline
def run_spatial_analysis_pipeline(adata_path, output_dir='spatial_analysis_results',
                                  radius=50, n_perms=1000):
    """
    Run complete spatial analysis pipeline.

    Parameters:
    -----------
    adata_path : str
        Path to h5ad file
    output_dir : str, default='spatial_analysis_results'
        Directory to save results
    radius : float, default=50
        Radius for spatial graph
    n_perms : int, default=1000
        Number of permutations for enrichment

    Returns:
    --------
    adata : AnnData
        AnnData object with all analysis results
    """

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("SPATIAL ANALYSIS PIPELINE")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    print(f"  - Loaded {adata.n_obs} cells")

    # Apply cell type colors
    load_and_apply_cell_type_colors(adata)

    # Step 1: Build spatial graph
    adata = build_spatial_graph(adata, method='radius', radius=radius)
    #adata = build_spatial_graph(adata, method='knn',n_neighbors=3)

    # Step 2: Neighborhood enrichment
    adata = neighborhood_enrichment_analysis(adata, n_perms=n_perms)

    # Step 3: Co-occurrence analysis
    adata = compute_co_occurrence(adata)

    # Step 4: Centrality scores
    adata = compute_centrality_scores(adata)

    # Step 5: Visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # Spatial distribution
    visualize_spatial_distribution(
        adata,
        save_path=output_dir / 'spatial_distribution.png'
    )

    # Enrichment heatmap
    visualize_enrichment(
        adata,
        save_path=output_dir / 'neighborhood_enrichment.png'
    )

    # Step 6: Summarize interactions
    interactions_df = summarize_interactions(adata)
    interactions_df.to_csv(output_dir / 'significant_interactions.csv', index=False)
    print(f"\n  - Saved interactions to: {output_dir / 'significant_interactions.csv'}")

    # Save processed data
    output_adata_path = output_dir / 'adata_with_spatial_analysis.h5ad'
    adata.write(output_adata_path)
    print(f"\n  - Saved processed AnnData to: {output_adata_path}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)

    return adata


# Example usage
if __name__ == "__main__":
    # Run pipeline on your data
    adata_path = 'tile_39520_7904.h5ad'

    adata = run_spatial_analysis_pipeline(
        adata_path=adata_path,
        output_dir='spatial_analysis_results',
        radius=20,  # Adjust based on your tissue/magnification
        n_perms=1000
    )

    print("\nYou can now explore the results:")
    print("  - Check 'spatial_analysis_results/' folder for figures")
    print("  - Load 'adata_with_spatial_analysis.h5ad' for further analysis")