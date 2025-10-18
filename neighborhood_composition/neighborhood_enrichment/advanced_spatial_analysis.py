import squidpy as sq
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial import distance_matrix
from scipy.stats import chi2_contingency


def compute_ripley_statistics(adata, cluster_key='cell_type', mode='L',
                              max_dist=200, n_steps=50):
    """
    Compute Ripley's K/L function for spatial point pattern analysis.
    Tests whether cell types are clustered, dispersed, or randomly distributed.

    Parameters:
    -----------
    adata : AnnData
        AnnData object with spatial coordinates
    cluster_key : str, default='cell_type'
        Key in adata.obs containing cell type labels
    mode : str, default='L'
        'K' for K-function, 'L' for L-function (variance-stabilized)
    max_dist : float, default=200
        Maximum distance to compute (in pixels)
    n_steps : int, default=50
        Number of distance steps

    Returns:
    --------
    adata : AnnData
        Modified in place with Ripley results
    """

    print(f"\nComputing Ripley's {mode}-function...")
    print(f"  - Maximum distance: {max_dist} pixels")
    print(f"  - Number of steps: {n_steps}")

    sq.gr.ripley(
        adata,
        cluster_key=cluster_key,
        spatial_key='spatial',
        mode=mode,
        max_dist=max_dist,
        n_steps=n_steps,
        copy=False
    )

    print(f"  - Ripley's {mode} analysis complete!")
    print(f"  - Results stored in adata.uns['{cluster_key}_ripley_{mode}']")

    return adata


def plot_ripley_statistics(adata, cluster_key='cell_type', mode='L',
                           figsize=(12, 8), save_path=None):
    """
    Visualize Ripley's statistics for each cell type.

    Parameters:
    -----------
    adata : AnnData
        AnnData object with Ripley results
    cluster_key : str, default='cell_type'
        Key for cell type labels
    mode : str, default='L'
        'K' or 'L' function
    figsize : tuple, default=(12, 8)
        Figure size
    save_path : str, optional
        Path to save figure
    """

    print(f"\nVisualizing Ripley's {mode}-function...")

    fig = sq.pl.ripley(
        adata,
        cluster_key=cluster_key,
        mode=mode,
        figsize=figsize
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved to: {save_path}")

    plt.close()

    return fig


def analyze_interaction_zones(adata, cluster_key='cell_type',
                              distance_bins=[0, 25, 50, 100, 200]):
    """
    Analyze cell type interactions at different distance ranges.

    Parameters:
    -----------
    adata : AnnData
        AnnData object with spatial coordinates
    cluster_key : str, default='cell_type'
        Key for cell type labels
    distance_bins : list, default=[0, 25, 50, 100, 200]
        Distance bins in pixels (e.g., immediate, near, medium, far)

    Returns:
    --------
    zone_interactions : dict
        Dictionary with interaction statistics for each zone
    """

    print(f"\nAnalyzing interaction zones...")
    print(f"  - Distance bins: {distance_bins}")

    coords = adata.obsm['spatial']
    cell_types = adata.obs[cluster_key].values
    unique_types = adata.obs[cluster_key].cat.categories

    # Compute pairwise distances
    print("  - Computing pairwise distances...")
    dist_matrix = distance_matrix(coords, coords)

    zone_interactions = {}

    for i in range(len(distance_bins) - 1):
        min_dist = distance_bins[i]
        max_dist = distance_bins[i + 1]
        zone_name = f"{min_dist}-{max_dist}px"

        print(f"  - Analyzing zone: {zone_name}")

        # Find pairs within this distance range
        mask = (dist_matrix > min_dist) & (dist_matrix <= max_dist)

        # Count interactions between cell types
        interaction_counts = pd.DataFrame(0,
                                          index=unique_types,
                                          columns=unique_types)

        for idx1 in range(len(cell_types)):
            neighbors_in_zone = mask[idx1]
            ct1 = cell_types[idx1]

            for idx2 in np.where(neighbors_in_zone)[0]:
                ct2 = cell_types[idx2]
                interaction_counts.loc[ct1, ct2] += 1

        zone_interactions[zone_name] = {
            'counts': interaction_counts,
            'total_interactions': interaction_counts.sum().sum(),
            'normalized': interaction_counts / interaction_counts.sum().sum()
        }

    return zone_interactions


def plot_interaction_zones(zone_interactions, figsize=(18, 6), save_path=None):
    """
    Visualize interaction patterns across different distance zones.

    Parameters:
    -----------
    zone_interactions : dict
        Output from analyze_interaction_zones()
    figsize : tuple, default=(18, 6)
        Figure size
    save_path : str, optional
        Path to save figure
    """

    print(f"\nVisualizing interaction zones...")

    n_zones = len(zone_interactions)
    fig, axes = plt.subplots(1, n_zones, figsize=figsize)

    if n_zones == 1:
        axes = [axes]

    for ax, (zone_name, data) in zip(axes, zone_interactions.items()):
        normalized = data['normalized']

        # Reduce annotation font size and format
        sns.heatmap(normalized, annot=True, fmt='.2f', cmap='YlOrRd',
                    annot_kws={'fontsize': 7},
                    ax=ax, cbar_kws={'label': 'Proportion'})
        ax.set_title(f'Zone: {zone_name}', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Cell Type 2', fontsize=9, labelpad=5)
        ax.set_ylabel('Cell Type 1', fontsize=9, labelpad=5)

        # Rotate and adjust labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=8)

    plt.tight_layout(pad=2.0)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved to: {save_path}")

    plt.close()

    return fig


def compute_local_cell_density(adata, cluster_key='cell_type', radius=100):
    """
    Compute local cell density and cell type composition for each cell.

    Parameters:
    -----------
    adata : AnnData
        AnnData object with spatial coordinates
    cluster_key : str, default='cell_type'
        Key for cell type labels
    radius : float, default=100
        Radius for local neighborhood (in pixels)

    Returns:
    --------
    adata : AnnData
        Modified in place with density metrics in adata.obs
    """

    print(f"\nComputing local cell density (radius={radius}px)...")

    coords = adata.obsm['spatial']
    cell_types = adata.obs[cluster_key].values
    unique_types = adata.obs[cluster_key].cat.categories

    # Compute pairwise distances
    dist_matrix = distance_matrix(coords, coords)

    # Initialize storage
    local_density = np.zeros(len(cell_types))
    local_composition = {ct: np.zeros(len(cell_types)) for ct in unique_types}

    for i in range(len(cell_types)):
        # Find neighbors within radius
        neighbors = dist_matrix[i] <= radius
        n_neighbors = neighbors.sum() - 1  # Exclude self

        local_density[i] = n_neighbors / (np.pi * radius ** 2)  # Density per unit area

        # Count each cell type in neighborhood
        for ct in unique_types:
            ct_count = (cell_types[neighbors] == ct).sum()
            if neighbors.sum() > 0:
                local_composition[ct][i] = ct_count / neighbors.sum()

    # Add to adata.obs
    adata.obs['local_density'] = local_density

    for ct in unique_types:
        safe_ct = ct.replace('/', '-').replace(' ', '_')
        adata.obs[f'local_frac_{safe_ct}'] = local_composition[ct]

    print(f"  - Added local density metrics to adata.obs")
    print(f"  - Mean local density: {local_density.mean():.4f} cells/px²")

    return adata


def identify_spatial_domains(adata, cluster_key='cell_type', resolution=0.5):
    """
    Identify spatial domains/niches using Leiden clustering on spatial graph.

    Parameters:
    -----------
    adata : AnnData
        AnnData object with spatial graph
    cluster_key : str, default='cell_type'
        Key for cell type labels
    resolution : float, default=0.5
        Resolution parameter for Leiden clustering

    Returns:
    --------
    adata : AnnData
        Modified in place with spatial domains in adata.obs
    """

    print(f"\nIdentifying spatial domains...")
    print(f"  - Resolution: {resolution}")

    # Run Leiden clustering on spatial graph
    sc.tl.leiden(
        adata,
        resolution=resolution,      #
        key_added='spatial_domain',
        adjacency=adata.obsp['spatial_connectivities']
    )

    n_domains = len(adata.obs['spatial_domain'].unique())   # change the number of domains
    print(f"  - Identified {n_domains} spatial domains")

    # Compute cell type composition for each domain
    domain_composition = pd.crosstab(
        adata.obs['spatial_domain'],
        adata.obs[cluster_key],
        normalize='index'   # modified
    )

    print("\nDomain composition:")
    print(domain_composition.round(3))

    return adata


def plot_spatial_domains(adata, figsize=(16, 6), save_path=None):
    """
    Visualize spatial domains and their cell type composition.

    Parameters:
    -----------
    adata : AnnData
        AnnData object with spatial domains
    figsize : tuple, default=(16, 6)
        Figure size
    save_path : str, optional
        Path to save figure
    """

    print(f"\nVisualizing spatial domains...")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Spatial domains
    coords = adata.obsm['spatial']
    domains = adata.obs['spatial_domain']

    scatter = axes[0].scatter(coords[:, 0], coords[:, 1],
                              c=domains.astype(int),
                              cmap='tab20',
                              s=5,
                              alpha=0.7)
    axes[0].set_xlabel('X coordinate (pixels)', fontsize=12)
    axes[0].set_ylabel('Y coordinate (pixels)', fontsize=12)
    axes[0].set_title('Spatial Domains', fontsize=14, fontweight='bold')
    axes[0].set_aspect('equal')
    cbar = plt.colorbar(scatter, ax=axes[0], label='Domain ID')
    cbar.ax.tick_params(labelsize=8)

    # Plot 2: Cell type composition by domain
    composition = pd.crosstab(
        adata.obs['spatial_domain'],
        adata.obs['cell_type'],
        normalize='columns'
    )

    # Limit to top 20 domains by cell count if there are too many
    n_domains = len(composition)
    if n_domains > 20:
        # Get top 20 domains by total cell count
        domain_sizes = adata.obs['spatial_domain'].value_counts()
        top_domains = domain_sizes.head(20).index
        composition_filtered = composition.loc[top_domains]
        title_suffix = f' (Top 20 of {n_domains} domains)'
    else:
        composition_filtered = composition
        title_suffix = ''

    # Plot without legend - too many domains make it unreadable
    composition_filtered.T.plot(kind='bar', stacked=True, ax=axes[1],
                                colormap='tab20', legend=False)
    axes[1].set_xlabel('Cell Type', fontsize=12)
    axes[1].set_ylabel('Proportion', fontsize=12)
    axes[1].set_title(f'Cell Type Composition by Domain{title_suffix}',
                      fontsize=14, fontweight='bold')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right', fontsize=10)

    # Add text annotation about number of domains
    axes[1].text(0.02, 0.98, f'Total domains: {n_domains}',
                 transform=axes[1].transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved to: {save_path}")

    plt.close()

    return fig


def compute_nearest_neighbor_distances(adata, cluster_key='cell_type'):
    """
    Compute nearest neighbor distances for each cell and between cell types.

    Parameters:
    -----------
    adata : AnnData
        AnnData object with spatial coordinates
    cluster_key : str, default='cell_type'
        Key for cell type labels

    Returns:
    --------
    nn_stats : dict
        Dictionary with nearest neighbor statistics
    """

    print(f"\nComputing nearest neighbor distances...")

    coords = adata.obsm['spatial']
    cell_types = adata.obs[cluster_key].values
    unique_types = adata.obs[cluster_key].cat.categories

    # Compute distance matrix
    dist_matrix = distance_matrix(coords, coords)
    # Set diagonal to infinity to exclude self
    np.fill_diagonal(dist_matrix, np.inf)

    # Overall nearest neighbor distances
    nn_distances = dist_matrix.min(axis=1)
    adata.obs['nn_distance'] = nn_distances

    # Nearest neighbor distances by cell type
    nn_stats = {
        'overall': {
            'mean': nn_distances.mean(),
            'median': np.median(nn_distances),
            'std': nn_distances.std()
        }
    }

    # For each cell type, find nearest neighbor of same and different types
    for ct in unique_types:
        ct_mask = cell_types == ct
        ct_indices = np.where(ct_mask)[0]

        if len(ct_indices) > 1:
            # Same type nearest neighbor
            same_type_nn = []
            for idx in ct_indices:
                same_type_dists = dist_matrix[idx, ct_mask]
                if len(same_type_dists) > 0:
                    same_type_nn.append(same_type_dists.min())

            # Different type nearest neighbor
            diff_type_nn = []
            for idx in ct_indices:
                diff_type_dists = dist_matrix[idx, ~ct_mask]
                if len(diff_type_dists) > 0:
                    diff_type_nn.append(diff_type_dists.min())

            nn_stats[ct] = {
                'same_type_mean': np.mean(same_type_nn) if same_type_nn else np.nan,
                'diff_type_mean': np.mean(diff_type_nn) if diff_type_nn else np.nan,
                'n_cells': len(ct_indices)
            }

    print("\nNearest neighbor statistics:")
    print(f"  Overall mean NN distance: {nn_stats['overall']['mean']:.2f} px")

    for ct in unique_types:
        if ct in nn_stats and not np.isnan(nn_stats[ct]['same_type_mean']):
            print(f"  {ct}:")
            print(f"    - Same type: {nn_stats[ct]['same_type_mean']:.2f} px")
            print(f"    - Different type: {nn_stats[ct]['diff_type_mean']:.2f} px")

    return nn_stats


# Main advanced analysis pipeline
def run_advanced_spatial_analysis(adata_path, output_dir='advanced_spatial_analysis',
                                  max_ripley_dist=200, local_density_radius=100,
                                  save_adata=False):
    """
    Run advanced spatial analysis pipeline.

    Parameters:
    -----------
    adata_path : str
        Path to h5ad file (with basic spatial analysis completed)
    output_dir : str, default='advanced_spatial_analysis'
        Directory to save results
    max_ripley_dist : float, default=200
        Maximum distance for Ripley's statistics
    local_density_radius : float, default=100
        Radius for local density computation
    save_adata : bool, default=False
        Whether to save the processed AnnData object with analysis results

    Returns:
    --------
    adata : AnnData
        AnnData object with all advanced analysis results
    results : dict
        Dictionary containing all analysis results
    """

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("ADVANCED SPATIAL ANALYSIS PIPELINE")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    print(f"  - Loaded {adata.n_obs} cells")

    results = {}

    # Step 1: Ripley's statistics
    adata = compute_ripley_statistics(adata, max_dist=max_ripley_dist)
    plot_ripley_statistics(adata, save_path=output_dir / 'ripley_statistics.png')

    # Step 2: Interaction zones
    zone_interactions = analyze_interaction_zones(adata)
    plot_interaction_zones(zone_interactions,
                           save_path=output_dir / 'interaction_zones.png')
    results['zone_interactions'] = zone_interactions

    # Step 3: Local cell density
    adata = compute_local_cell_density(adata, radius=local_density_radius)

    # Step 4: Spatial domains
    adata = identify_spatial_domains(adata)
    plot_spatial_domains(adata, save_path=output_dir / 'spatial_domains.png')

    # Step 5: Nearest neighbor analysis
    nn_stats = compute_nearest_neighbor_distances(adata)
    results['nn_stats'] = nn_stats

    # Save nearest neighbor stats
    nn_df = pd.DataFrame(nn_stats).T
    nn_df.to_csv(output_dir / 'nearest_neighbor_stats.csv')
    print(f"\n  - Saved NN stats to: {output_dir / 'nearest_neighbor_stats.csv'}")

    # Save processed data (optional)
    if save_adata:
        output_adata_path = output_dir / 'adata_with_advanced_analysis.h5ad'
        adata.write(output_adata_path)
        print(f"  - Saved processed AnnData to: {output_adata_path}")
    else:
        print(f"  - AnnData not saved (set save_adata=True to save)")

    print("\n" + "=" * 60)
    print("ADVANCED ANALYSIS COMPLETE!")
    print("=" * 60)

    return adata, results


# Example usage
if __name__ == "__main__":
    # Run advanced analysis on data with basic spatial analysis
    adata_path = 'spatial_analysis_results/adata_with_spatial_analysis.h5ad'

    adata, results = run_advanced_spatial_analysis(
        adata_path=adata_path,
        output_dir='advanced_spatial_analysis',
        max_ripley_dist=200,
        local_density_radius=100,
        save_adata=False  # Set to True to save the h5ad file
    )

    print("\nYou can now explore the advanced results:")
    print("  - Check 'advanced_spatial_analysis/' folder for figures")
    print("  - Set save_adata=True if you want to save 'adata_with_advanced_analysis.h5ad'")