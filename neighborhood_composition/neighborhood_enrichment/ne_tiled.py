import squidpy as sq
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# from dask.tests.test_config import no_read_permissions
# from networkx.algorithms.distance_measures import radius


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
        spatial_key='spatial',  # Modify to knn for large files
        n_splits=n_splits,
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


def visualize_enrichment(adata, cluster_key='cell_type', figsize=(10, 8), save_path=None, radius=None, n_perms=None):
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
    radius : float, optional
        Radius used for spatial graph (displayed in title)
    n_perms : int, optional
        Number of permutations used (displayed in title)
    """

    print(f"\nVisualizing neighborhood enrichment...")

    # Get z-scores to calculate dynamic scale
    zscore = adata.uns[f'{cluster_key}_nhood_enrichment']['zscore']
    if isinstance(zscore, pd.DataFrame):
        zscore_array = zscore.values
    else:
        zscore_array = np.array(zscore)

    # Calculate symmetric scale based on actual data range
    max_abs_z = np.abs(zscore_array).max()
    vmin, vmax = -max_abs_z, max_abs_z

    print(f"  - Z-score range: [{zscore_array.min():.2f}, {zscore_array.max():.2f}]")
    print(f"  - Color scale: [{vmin:.2f}, {vmax:.2f}]")

    fig, ax = plt.subplots(figsize=figsize)
    max_abs_value = max(abs(vmin), abs(vmax))

    # Get cell type names for labels
    cell_types = adata.obs[cluster_key].cat.categories.tolist()

    # Use seaborn directly to show annotations
    sns.heatmap(
        zscore,
        cmap='coolwarm',
        center=0,
        vmin=-np.ceil(max_abs_value),
        vmax=np.ceil(max_abs_value),
        annot=True,  # Show values in cells
        fmt='.2f',   # Format to 2 decimal places
        cbar_kws={'label': 'Z-score'},
        linewidths=0.5,
        linecolor='white',
        #xticklabels=cell_types,
        #yticklabels=cell_types,
        square=True,
        ax=ax
    )
    ax.set_xlabel('Cell Type', fontsize=12)
    ax.set_ylabel('Cell Type', fontsize=12)

    # Build title with optional radius and n_perms
    title = 'Neighborhood Enrichment Analysis\n(Mean Z-score)'
    if radius is not None or n_perms is not None:
        params = []
        if radius is not None:
            params.append(f'radius={radius}')
        if n_perms is not None:
            params.append(f'n_perms={n_perms}')
        title += f'\n({", ".join(params)})'

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Set tick labels
    ax.set_xticklabels(cell_types, rotation=45, ha='right')
    ax.set_yticklabels(cell_types, rotation=0)

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

    # Add text showing number of cells displayed
    text_str = f"Displaying {adata.n_obs:,} cells"
    ax.text(0.98, 0.02, text_str,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

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
                                  radius=50, n_perms=1000, save_adata=False,
                                  skip_cooccurrence=False, max_cells_for_cooccurrence=50000):
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
    save_adata : bool, default=False
        Whether to save the processed AnnData object with analysis results
    skip_cooccurrence : bool, default=False
        Whether to skip co-occurrence analysis (useful for large datasets)
    max_cells_for_cooccurrence : int, default=50000
        Maximum number of cells for co-occurrence analysis.
        If dataset is larger, co-occurrence will be automatically skipped.

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

    # Step 1: Build spatial graph (Choose radius or knn)
    adata = build_spatial_graph(adata, method='radius', radius=radius)
    #adata = build_spatial_graph(adata, method='knn',n_neighbors=6)

    # Step 2: Neighborhood enrichment
    adata = neighborhood_enrichment_analysis(adata, n_perms=n_perms)

    # Step 3: Co-occurrence analysis (skip for large datasets to avoid memory issues)
    if not skip_cooccurrence:
        if adata.n_obs > max_cells_for_cooccurrence:
            print(f"\n⚠️  Skipping co-occurrence analysis:")
            print(f"   Dataset has {adata.n_obs} cells (> {max_cells_for_cooccurrence} threshold)")
            print(f"   Co-occurrence requires ~{(adata.n_obs**2 * 4 / 1e9):.1f} GB of RAM")
            print(f"   Set skip_cooccurrence=False and increase max_cells_for_cooccurrence to force run")
        else:
            adata = compute_co_occurrence(adata)
    else:
        print("\nSkipping co-occurrence analysis (skip_cooccurrence=True)")

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
        save_path=output_dir / 'neighborhood_enrichment.png',
        radius=radius,
        n_perms=n_perms
    )

    # Step 6: Summarize interactions
    interactions_df = summarize_interactions(adata)
    interactions_df.to_csv(output_dir / 'significant_interactions.csv', index=False)
    print(f"\n  - Saved interactions to: {output_dir / 'significant_interactions.csv'}")

    # Save processed data (optional)
    if save_adata:
        output_adata_path = output_dir / 'adata_with_spatial_analysis.h5ad'
        adata.write(output_adata_path)
        print(f"\n  - Saved processed AnnData to: {output_adata_path}")
    else:
        print(f"\n  - AnnData not saved (set save_adata=True to save)")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)

    return adata


# Example usage
if __name__ == "__main__":
    # Run pipeline on your data
    # current_dir = os.path.dirname(__file__)
    # parent_dir = os.path.dirname(current_dir)
    # adata_path = os.path.join(parent_dir, "tile_39520_7904.h5ad")
    adata_path = '../tile_39520_7904.h5ad'

    adata = run_spatial_analysis_pipeline(
        adata_path=adata_path,
        output_dir='spatial_analysis_results',
        radius=50,  # Adjust based on your tissue/magnification
        n_perms=1000,
        save_adata=False,  # Set to True to save the h5ad file
        skip_cooccurrence=False,  # Set to True to skip co-occurrence for large datasets
        max_cells_for_cooccurrence=50000  # Auto-skip co-occurrence if more cells
    )

    print("\nYou can now explore the results:")
    print("  - Check 'spatial_analysis_results/' folder for figures")
    print("  - Set save_adata=True if you want to save 'adata_with_spatial_analysis.h5ad'")