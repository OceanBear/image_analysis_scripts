"""
Bootstrap Permutation Analysis for Tiled Images

This module implements hierarchical bootstrap-permutation testing for spatial
neighborhood enrichment analysis in tiled microscopy images. It addresses the
pseudo-replication problem where cells within tiles are not independent.

Key features:
- Tile-level bootstrap resampling (respects hierarchical structure)
- Within-bootstrap permutation testing for statistical significance
- Confidence intervals from bootstrap distribution
- Comparison with standard permutation results
- Robust uncertainty quantification

Author: Generated with Claude Code
Date: 2025-10-21
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from tqdm import tqdm
import warnings
import os
from pathlib import Path
# Set the working directory to the script's directory
os.chdir(Path(__file__).parent)

# Import reusable functions from existing modules
from ne_tiled import (
    load_and_apply_cell_type_colors,
    build_spatial_graph,
    neighborhood_enrichment_analysis,
    compute_centrality_scores,
    summarize_interactions
)

warnings.filterwarnings('ignore')


def handle_extreme_zscores(zscore_array, max_zscore=50.0):
    """
    Handle infinite and extreme z-scores by clipping and replacing with robust estimates.
    
    Parameters:
    -----------
    zscore_array : np.array
        Array of z-scores that may contain infinite or extreme values
    max_zscore : float, default=50.0
        Maximum allowed z-score value
        
    Returns:
    --------
    zscore_clean : np.array
        Array with extreme values handled
    """
    zscore_clean = zscore_array.copy()
    
    # Handle infinite values
    inf_mask = np.isinf(zscore_clean)
    if inf_mask.any():
        # Replace infinite values with clipped finite values
        finite_values = zscore_clean[np.isfinite(zscore_clean)]
        if len(finite_values) > 0:
            # Use the maximum finite value as replacement for +inf
            max_finite = np.max(finite_values)
            min_finite = np.min(finite_values)
            
            zscore_clean[zscore_clean == np.inf] = min(max_finite, max_zscore)
            zscore_clean[zscore_clean == -np.inf] = max(min_finite, -max_zscore)
        else:
            # If all values are infinite, replace with zeros
            zscore_clean[inf_mask] = 0.0
    
    # Handle NaN values
    nan_mask = np.isnan(zscore_clean)
    if nan_mask.any():
        zscore_clean[nan_mask] = 0.0
    
    # Clip extreme values
    zscore_clean = np.clip(zscore_clean, -max_zscore, max_zscore)
    
    return zscore_clean


def identify_tiles(adata, tile_key='tile_name'):
    """
    Identify unique tiles in the dataset.

    Parameters:
    -----------
    adata : AnnData
        AnnData object with tile information
    tile_key : str, default='tile_name'
        Key in adata.obs containing tile identifiers

    Returns:
    --------
    tiles : list
        List of unique tile identifiers
    tile_counts : dict
        Dictionary mapping tile IDs to cell counts
    """
    if tile_key not in adata.obs.columns:
        raise ValueError(f"Tile key '{tile_key}' not found in adata.obs. "
                        f"Available columns: {adata.obs.columns.tolist()}")

    tiles = adata.obs[tile_key].unique().tolist()
    tile_counts = adata.obs[tile_key].value_counts().to_dict()

    print(f"  - Found {len(tiles)} unique tiles")
    print(f"  - Cells per tile: min={min(tile_counts.values())}, "
          f"max={max(tile_counts.values())}, "
          f"mean={np.mean(list(tile_counts.values())):.1f}")

    return tiles, tile_counts


def bootstrap_resample_tiles(adata, tile_key='tile_name', seed=None):
    """
    Resample tiles with replacement (hierarchical bootstrap).

    This preserves spatial relationships within tiles while accounting
    for tile-level variability.

    Parameters:
    -----------
    adata : AnnData
        AnnData object with tile information
    tile_key : str, default='tile_name'
        Key in adata.obs containing tile identifiers
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    adata_bootstrap : AnnData
        Bootstrapped AnnData object (cells from resampled tiles)
    selected_tiles : list
        List of tiles selected in this bootstrap (with replacement)
    """
    if seed is not None:
        np.random.seed(seed)

    # Get unique tiles
    tiles = adata.obs[tile_key].unique()
    n_tiles = len(tiles)

    # Resample tiles with replacement
    selected_tiles = np.random.choice(tiles, size=n_tiles, replace=True)

    # Collect all cells from selected tiles
    cell_indices = []
    for tile_id in selected_tiles:
        tile_cells = adata.obs[tile_key] == tile_id
        cell_indices.extend(adata.obs_names[tile_cells].tolist())

    # Create bootstrapped dataset
    adata_bootstrap = adata[cell_indices].copy()

    return adata_bootstrap, selected_tiles.tolist()


def run_single_bootstrap_iteration(
    adata,
    tile_key='tile_name',
    method='knn',
    radius=50,
    n_neighbors=6,
    n_perms=100,
    cluster_key='cell_type',
    seed=None,
    verbose=False,
    max_zscore=50.0,
    min_cells_per_type=5
):
    """
    Run a single bootstrap iteration: resample tiles + enrichment analysis.

    Parameters:
    -----------
    adata : AnnData
        Original AnnData object
    tile_key : str
        Key for tile identifiers
    method : str, default='knn'
        Method for spatial graph: 'knn' or 'radius'
    radius : float, default=50
        Radius for spatial graph (used if method='radius')
    n_neighbors : int, default=6
        Number of neighbors for KNN (used if method='knn')
    n_perms : int
        Number of permutations for enrichment test
    cluster_key : str
        Key for cell type labels
    seed : int, optional
        Random seed
    verbose : bool
        Whether to print progress
    max_zscore : float, default=50.0
        Maximum z-score value (clips extreme values)
    min_cells_per_type : int, default=5
        Minimum cells per cell type for valid analysis

    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'zscore': enrichment z-scores matrix
        - 'selected_tiles': tiles included in this bootstrap
        - 'n_cells': number of cells in bootstrap sample
        - 'data_quality': quality metrics
    """
    if verbose:
        print(f"  Running bootstrap iteration (seed={seed})...")

    # Bootstrap resample tiles
    adata_boot, selected_tiles = bootstrap_resample_tiles(
        adata, tile_key=tile_key, seed=seed
    )

    # Check data quality before analysis
    cell_type_counts = adata_boot.obs[cluster_key].value_counts()
    min_count = cell_type_counts.min()
    
    if min_count < min_cells_per_type:
        if verbose:
            print(f"    Warning: Some cell types have < {min_cells_per_type} cells (min: {min_count})")
    
    # Build spatial graph
    adata_boot = build_spatial_graph(
        adata_boot, method=method, radius=radius, n_neighbors=n_neighbors
    )

    # Run enrichment analysis with permutation
    adata_boot = neighborhood_enrichment_analysis(
        adata_boot,
        cluster_key=cluster_key,
        n_perms=n_perms,
        seed=seed
    )

    # Extract z-scores
    zscore = adata_boot.uns[f'{cluster_key}_nhood_enrichment']['zscore']
    zscore_array = np.array(zscore)
    
    # Handle infinite and extreme values
    zscore_clean = handle_extreme_zscores(zscore_array, max_zscore=max_zscore)
    
    # Calculate data quality metrics
    n_inf = np.isinf(zscore_array).sum()
    n_nan = np.isnan(zscore_array).sum()
    n_extreme = (np.abs(zscore_array) > max_zscore).sum()
    max_abs_z = np.nanmax(np.abs(zscore_array))
    
    data_quality = {
        'n_inf': n_inf,
        'n_nan': n_nan,
        'n_extreme': n_extreme,
        'max_abs_z': max_abs_z,
        'min_cell_count': min_count,
        'n_cell_types': len(cell_type_counts)
    }

    results = {
        'zscore': zscore_clean,
        'selected_tiles': selected_tiles,
        'n_cells': adata_boot.n_obs,
        'data_quality': data_quality
    }

    return results


def run_bootstrap_permutation_analysis(
    adata,
    tile_key='tile_name',
    n_bootstrap=100,
    method='knn',
    radius=50,
    n_neighbors=6,
    n_perms=100,
    cluster_key='cell_type',
    seed=42,
    n_jobs=1,
    max_zscore=50.0,
    min_cells_per_type=5
):
    """
    Run full bootstrap-permutation analysis across multiple iterations.

    Parameters:
    -----------
    adata : AnnData
        AnnData object with tile information
    tile_key : str, default='tile_name'
        Key for tile identifiers
    n_bootstrap : int, default=100
        Number of bootstrap iterations
    method : str, default='knn'
        Method for spatial graph: 'knn' or 'radius'
    radius : float, default=50
        Radius for spatial graph (used if method='radius')
    n_neighbors : int, default=6
        Number of neighbors for KNN (used if method='knn')
    n_perms : int, default=100
        Number of permutations per bootstrap
    cluster_key : str, default='cell_type'
        Key for cell type labels
    seed : int, default=42
        Random seed for reproducibility
    n_jobs : int, default=1
        Number of parallel jobs (currently single-threaded)

    Returns:
    --------
    bootstrap_results : dict
        Dictionary containing:
        - 'zscores': array of z-scores for each bootstrap (shape: n_bootstrap x n_celltypes x n_celltypes)
        - 'mean_zscore': mean z-score across bootstraps
        - 'std_zscore': standard deviation across bootstraps
        - 'ci_lower': lower 95% confidence interval
        - 'ci_upper': upper 95% confidence interval
        - 'cell_types': list of cell type names
    """
    print("=" * 70)
    print("BOOTSTRAP-PERMUTATION ENRICHMENT ANALYSIS")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  - Number of bootstrap iterations: {n_bootstrap}")
    print(f"  - Permutations per bootstrap: {n_perms}")
    print(f"  - Spatial graph method: {method}")
    if method == 'knn':
        print(f"  - Number of neighbors (KNN): {n_neighbors}")
    else:
        print(f"  - Spatial radius: {radius} pixels")
    print(f"  - Tile key: {tile_key}")

    # Identify tiles
    tiles, tile_counts = identify_tiles(adata, tile_key=tile_key)

    # Get cell types
    cell_types = adata.obs[cluster_key].cat.categories.tolist()
    n_celltypes = len(cell_types)

    print(f"\n  - Cell types: {cell_types}")
    print(f"\nRunning {n_bootstrap} bootstrap iterations...")

    # Storage for bootstrap results
    zscores_list = []

    # Run bootstrap iterations with progress bar
    for i in tqdm(range(n_bootstrap), desc="Bootstrap iterations"):
        boot_seed = seed + i if seed is not None else None

        try:
            results = run_single_bootstrap_iteration(
                adata,
                tile_key=tile_key,
                method=method,
                radius=radius,
                n_neighbors=n_neighbors,
                n_perms=n_perms,
                cluster_key=cluster_key,
                seed=boot_seed,
                verbose=False,
                max_zscore=max_zscore,
                min_cells_per_type=min_cells_per_type
            )
            zscores_list.append(results['zscore'])
        except Exception as e:
            print(f"\n  Warning: Bootstrap iteration {i+1} failed: {e}")
            continue

    print(f"\n  - Successfully completed {len(zscores_list)}/{n_bootstrap} iterations")

    if len(zscores_list) == 0:
        raise RuntimeError("All bootstrap iterations failed!")

    # Stack results
    zscores_array = np.stack(zscores_list)  # shape: (n_bootstrap, n_celltypes, n_celltypes)

    # Calculate robust statistics
    mean_zscore = np.nanmean(zscores_array, axis=0)
    std_zscore = np.nanstd(zscores_array, axis=0)
    median_zscore = np.nanmedian(zscores_array, axis=0)

    # Calculate 95% confidence intervals using robust percentiles
    ci_lower = np.nanpercentile(zscores_array, 2.5, axis=0)
    ci_upper = np.nanpercentile(zscores_array, 97.5, axis=0)
    
    # Additional robust statistics
    trimmed_mean = np.array([
        np.nanmean(np.clip(zscores_array[:, i, j], 
                          np.nanpercentile(zscores_array[:, i, j], 5),
                          np.nanpercentile(zscores_array[:, i, j], 95)))
        for i in range(zscores_array.shape[1])
        for j in range(zscores_array.shape[2])
    ]).reshape(zscores_array.shape[1], zscores_array.shape[2])

    # Data quality checks
    n_inf_total = np.isinf(zscores_array).sum()
    n_nan_total = np.isnan(zscores_array).sum()
    n_extreme_total = (np.abs(zscores_array) > max_zscore).sum()
    
    # Prepare results
    bootstrap_results = {
        'zscores': zscores_array,
        'mean_zscore': mean_zscore,
        'std_zscore': std_zscore,
        'median_zscore': median_zscore,
        'trimmed_mean': trimmed_mean,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'cell_types': cell_types,
        'n_bootstrap': len(zscores_list),
        'data_quality': {
            'n_inf_total': n_inf_total,
            'n_nan_total': n_nan_total,
            'n_extreme_total': n_extreme_total,
            'max_zscore_used': max_zscore,
            'min_cells_per_type': min_cells_per_type
        },
        'parameters': {
            'n_perms': n_perms,
            'method': method,
            'radius': radius,
            'n_neighbors': n_neighbors,
            'tile_key': tile_key,
            'seed': seed,
            'max_zscore': max_zscore,
            'min_cells_per_type': min_cells_per_type
        }
    }

    print("\n" + "=" * 70)
    print("BOOTSTRAP SUMMARY")
    print("=" * 70)
    print(f"  - Mean std of z-scores: {std_zscore.mean():.3f}")
    print(f"  - Max std of z-scores: {std_zscore.max():.3f}")
    print(f"  - Mean CI width: {(ci_upper - ci_lower).mean():.3f}")
    
    # Data quality summary
    print(f"\n  Data Quality:")
    print(f"  - Infinite values handled: {n_inf_total}")
    print(f"  - NaN values handled: {n_nan_total}")
    print(f"  - Extreme values clipped: {n_extreme_total}")
    print(f"  - Max z-score limit: {max_zscore}")
    print(f"  - Min cells per type: {min_cells_per_type}")

    return bootstrap_results


def compare_with_standard_permutation(
    adata,
    bootstrap_results,
    method='knn',
    radius=50,
    n_neighbors=6,
    n_perms=1000,
    cluster_key='cell_type'
):
    """
    Run standard permutation analysis for comparison with bootstrap results.

    Parameters:
    -----------
    adata : AnnData
        Full AnnData object (all tiles)
    bootstrap_results : dict
        Results from bootstrap analysis
    method : str, default='knn'
        Method for spatial graph: 'knn' or 'radius'
    radius : float, default=50
        Radius for spatial graph (used if method='radius')
    n_neighbors : int, default=6
        Number of neighbors for KNN (used if method='knn')
    n_perms : int
        Number of permutations
    cluster_key : str
        Key for cell type labels

    Returns:
    --------
    adata : AnnData
        AnnData with standard enrichment results
    comparison : dict
        Comparison metrics between bootstrap and standard methods
    """
    print("\n" + "=" * 70)
    print("STANDARD PERMUTATION ANALYSIS (for comparison)")
    print("=" * 70)

    # Run standard analysis on all tiles
    adata_std = adata.copy()
    adata_std = build_spatial_graph(adata_std, method=method, radius=radius, n_neighbors=n_neighbors)
    adata_std = neighborhood_enrichment_analysis(
        adata_std,
        cluster_key=cluster_key,
        n_perms=n_perms
    )

    # Extract z-scores
    standard_zscore = np.array(adata_std.uns[f'{cluster_key}_nhood_enrichment']['zscore'])

    # Compare with bootstrap mean
    bootstrap_mean = bootstrap_results['mean_zscore']

    # Calculate differences
    diff = standard_zscore - bootstrap_mean
    abs_diff = np.abs(diff)

    comparison = {
        'standard_zscore': standard_zscore,
        'mean_absolute_difference': abs_diff.mean(),
        'max_absolute_difference': abs_diff.max(),
        'correlation': np.corrcoef(standard_zscore.flatten(), bootstrap_mean.flatten())[0, 1],
        'differences': diff
    }

    print(f"\nComparison with Bootstrap Results:")
    print(f"  - Mean absolute difference: {comparison['mean_absolute_difference']:.3f}")
    print(f"  - Max absolute difference: {comparison['max_absolute_difference']:.3f}")
    print(f"  - Correlation: {comparison['correlation']:.3f}")

    if comparison['correlation'] > 0.95:
        print("   EXCELLENT: Bootstrap and standard results highly consistent")
    elif comparison['correlation'] > 0.85:
        print("   GOOD: Bootstrap and standard results reasonably consistent")
    elif comparison['correlation'] > 0.70:
        print("  ⚠ FAIR: Some discrepancy between methods")
    else:
        print("   WARNING: Substantial discrepancy - tile variability is high!")

    return adata_std, comparison


def visualize_bootstrap_enrichment(
    bootstrap_results,
    cluster_key='cell_type',
    figsize=(14, 10),
    cmap='coolwarm',
    vmin=None,
    vmax=None,
    save_path=None
):
    """
    Visualize bootstrap enrichment results with confidence intervals.

    Creates a heatmap showing:
    - Mean z-scores (color)
    - Confidence intervals (annotations)
    - Significance indicators

    Parameters:
    -----------
    bootstrap_results : dict
        Results from bootstrap analysis
    cluster_key : str
        Key for cell type labels
    figsize : tuple
        Figure size
    cmap : str
        Colormap
    vmin, vmax : float, optional
        Color scale limits. If None, automatically determined from data
    save_path : str, optional
        Path to save figure
    """
    print("\nVisualizing bootstrap enrichment results with confidence intervals...")

    mean_zscore = bootstrap_results['mean_zscore']
    ci_lower = bootstrap_results['ci_lower']
    ci_upper = bootstrap_results['ci_upper']
    cell_types = bootstrap_results['cell_types']

    # Calculate dynamic color scale if not provided
    if vmin is None or vmax is None:
        # Use symmetrical scale around 0 for coolwarm colormap
        max_abs_value = max(abs(mean_zscore.min()), abs(mean_zscore.max()))
        vmin = -max_abs_value
        vmax = max_abs_value
        print(f"  - Dynamic color scale: [{vmin:.2f}, {vmax:.2f}]")

    # Create annotations with CI
    annotations = []
    for i in range(len(cell_types)):
        row = []
        for j in range(len(cell_types)):
            mean = mean_zscore[i, j]
            lower = ci_lower[i, j]
            upper = ci_upper[i, j]

            # Check if CI excludes zero (significant)
            significant = (lower > 0 and upper > 0) or (lower < 0 and upper < 0)
            sig_marker = "***" if significant else ""

            # Format: mean ± CI width
            ci_width = (upper - lower) / 2
            row.append(f"{mean:.2f}{sig_marker}\n±{ci_width:.2f}")
        annotations.append(row)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        mean_zscore,
        cmap=cmap,
        center=0,
        vmin=vmin,
        vmax=vmax,
        annot=annotations,
        fmt='',
        cbar_kws={'label': 'Mean Z-score (Bootstrap)'},
        linewidths=0.5,
        linecolor='white',
        square=True,
        ax=ax,
        #annot_kws={'fontsize': 8}
    )

    ax.set_xlabel('Cell Type', fontsize=12)
    ax.set_ylabel('Cell Type', fontsize=12)
    ax.set_title('Neighborhood Enrichment: Bootstrap-Permutation Analysis\n'
                 '(Mean Z-score at 95% CI, *** = CI excludes zero)',
                 fontsize=14, fontweight='bold', pad=20)

    # Set tick labels
    ax.set_xticklabels(cell_types, rotation=45, ha='right')
    ax.set_yticklabels(cell_types, rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved to: {save_path}")

    plt.show()

    return fig


def visualize_bootstrap_comparison(
    bootstrap_results,
    standard_zscore,
    cell_types,
    figsize=(10, 8),
    save_path=None
):
    """
    Visualize bootstrap uncertainty (standard deviation across iterations).

    Parameters:
    -----------
    bootstrap_results : dict
        Results from bootstrap analysis
    standard_zscore : array
        Z-scores from standard permutation (unused, kept for compatibility)
    cell_types : list
        List of cell type names
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    print("\nVisualizing bootstrap uncertainty...")

    std_zscore = bootstrap_results['std_zscore']

    # Calculate auto-scale based on extreme values
    max_value = std_zscore.max()
    vmin, vmax = 0, np.ceil(max_value)

    print(f"  - Std Dev range: [0.00, {max_value:.2f}]")
    print(f"  - Color scale: [0, {vmax:.0f}]")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap with annotations
    sns.heatmap(
        std_zscore,
        cmap='YlOrRd',
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Standard Deviation'},
        annot=True,  # Show values in cells
        fmt='.2f',   # Format to 2 decimal places
        linewidths=0.5,
        linecolor='white',
        xticklabels=cell_types,
        yticklabels=cell_types,
        square=True,
        ax=ax
    )

    ax.set_xlabel('Cell Type', fontsize=12)
    ax.set_ylabel('Cell Type', fontsize=12)
    ax.set_title('Bootstrap Uncertainty\n(Std Dev across iterations)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  - Saved to: {save_path}")

    plt.show()

    return fig


def summarize_bootstrap_interactions(
    bootstrap_results,
    threshold_zscore=2.0,
    threshold_ci=True
):
    """
    Summarize significant interactions from bootstrap analysis.

    Parameters:
    -----------
    bootstrap_results : dict
        Results from bootstrap analysis
    threshold_zscore : float, default=2.0
        Z-score threshold for significance
    threshold_ci : bool, default=True
        If True, require 95% CI to exclude zero for significance

    Returns:
    --------
    interactions_df : DataFrame
        Summary of significant interactions with confidence intervals
    """
    print("\n" + "=" * 70)
    print("SIGNIFICANT INTERACTIONS (Bootstrap-Permutation)")
    print("=" * 70)

    mean_zscore = bootstrap_results['mean_zscore']
    ci_lower = bootstrap_results['ci_lower']
    ci_upper = bootstrap_results['ci_upper']
    cell_types = bootstrap_results['cell_types']

    interactions = []

    for i, ct1 in enumerate(cell_types):
        for j, ct2 in enumerate(cell_types):
            mean_z = mean_zscore[i, j]
            lower = ci_lower[i, j]
            upper = ci_upper[i, j]

            # Check significance
            abs_z = abs(mean_z)
            ci_excludes_zero = (lower > 0 and upper > 0) or (lower < 0 and upper < 0)

            if threshold_ci:
                # Require both z-score threshold AND CI excludes zero
                is_significant = (abs_z > threshold_zscore) and ci_excludes_zero
            else:
                # Only require z-score threshold
                is_significant = abs_z > threshold_zscore

            if is_significant:
                interaction_type = "Attraction" if mean_z > 0 else "Avoidance"
                interactions.append({
                    'Cell Type 1': ct1,
                    'Cell Type 2': ct2,
                    'Mean Z-score': float(mean_z),
                    'CI Lower': float(lower),
                    'CI Upper': float(upper),
                    'CI Width': float(upper - lower),
                    'Interaction': interaction_type,
                    'CI Excludes Zero': ci_excludes_zero
                })

    interactions_df = pd.DataFrame(interactions).sort_values(
        'Mean Z-score', key=abs, ascending=False
    )

    print(f"\nFound {len(interactions_df)} significant interactions")
    print(f"(|Z| > {threshold_zscore}, CI excludes zero: {threshold_ci})")

    if len(interactions_df) > 0:
        print("\nTop 10 interactions:")
        print(interactions_df.head(10).to_string(index=False))
    else:
        print("No significant interactions found above threshold")

    return interactions_df


def save_bootstrap_intermediate_results(bootstrap_results, output_dir, tile_name=None):
    """
    Save bootstrap intermediate results for efficient aggregation later.

    This saves all bootstrap iterations' zscores and metadata to disk, avoiding
    the need to keep large bootstrap arrays in memory during batch processing.

    Parameters:
    -----------
    bootstrap_results : dict
        Dictionary from run_bootstrap_permutation_analysis() containing:
        - 'zscores': array of shape (n_bootstrap, n_celltypes, n_celltypes)
        - 'mean_zscore', 'std_zscore', 'ci_lower', 'ci_upper'
        - 'cell_types', 'n_bootstrap', 'parameters'
    output_dir : str or Path
        Directory to save intermediate results
    tile_name : str, optional
        Name of the tile (for prefixing files). If None, no prefix used.

    Returns:
    --------
    saved_files : dict
        Dictionary with paths to saved files
    """
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get prefix for files
    prefix = f'{tile_name}_' if tile_name else ''

    # Extract bootstrap data
    zscores_array = bootstrap_results['zscores']  # (n_bootstrap, n_celltypes, n_celltypes)
    mean_zscore = bootstrap_results['mean_zscore']
    std_zscore = bootstrap_results['std_zscore']
    ci_lower = bootstrap_results['ci_lower']
    ci_upper = bootstrap_results['ci_upper']
    cell_types = bootstrap_results['cell_types']

    # Save all bootstrap zscores as numpy binary (fast and compact)
    zscores_path = output_dir / f'{prefix}bootstrap_zscores.npy'
    np.save(zscores_path, zscores_array)

    # Save aggregated statistics
    mean_path = output_dir / f'{prefix}bootstrap_mean.npy'
    np.save(mean_path, mean_zscore)

    std_path = output_dir / f'{prefix}bootstrap_std.npy'
    np.save(std_path, std_zscore)

    ci_lower_path = output_dir / f'{prefix}bootstrap_ci_lower.npy'
    np.save(ci_lower_path, ci_lower)

    ci_upper_path = output_dir / f'{prefix}bootstrap_ci_upper.npy'
    np.save(ci_upper_path, ci_upper)

    # Save metadata as JSON
    metadata = {
        'tile_name': tile_name,
        'n_bootstrap': int(bootstrap_results['n_bootstrap']),
        'cell_types': cell_types,
        'zscores_shape': list(zscores_array.shape),
        'mean_abs_zscore': float(np.abs(mean_zscore).mean()),
        'max_abs_zscore': float(np.abs(mean_zscore).max()),
        'mean_std': float(std_zscore.mean()),
        'max_std': float(std_zscore.max()),
        'parameters': bootstrap_results['parameters']
    }

    metadata_path = output_dir / f'{prefix}bootstrap_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  - Saved bootstrap intermediate results:")
    print(f"    • {zscores_path.name}")
    print(f"    • {mean_path.name}")
    print(f"    • {std_path.name}")
    print(f"    • {ci_lower_path.name}")
    print(f"    • {ci_upper_path.name}")
    print(f"    • {metadata_path.name}")

    return {
        'zscores_path': zscores_path,
        'mean_path': mean_path,
        'std_path': std_path,
        'ci_lower_path': ci_lower_path,
        'ci_upper_path': ci_upper_path,
        'metadata_path': metadata_path,
        'zscores': zscores_array,
        'mean_zscore': mean_zscore,
        'std_zscore': std_zscore,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'metadata': metadata
    }


def load_bootstrap_intermediate_results(output_dir, tile_name=None):
    """
    Load bootstrap intermediate results saved by save_bootstrap_intermediate_results().

    Parameters:
    -----------
    output_dir : str or Path
        Directory containing saved results
    tile_name : str, optional
        Name of the tile (for prefixing files). If None, no prefix used.

    Returns:
    --------
    results : dict
        Dictionary containing bootstrap results:
        - 'zscores': full bootstrap array (n_bootstrap, n_celltypes, n_celltypes)
        - 'mean_zscore', 'std_zscore', 'ci_lower', 'ci_upper'
        - 'cell_types', 'metadata'
    """
    import json

    output_dir = Path(output_dir)
    prefix = f'{tile_name}_' if tile_name else ''

    # Load all bootstrap zscores
    zscores_path = output_dir / f'{prefix}bootstrap_zscores.npy'
    if not zscores_path.exists():
        raise FileNotFoundError(f"Bootstrap zscores file not found: {zscores_path}")
    zscores = np.load(zscores_path)

    # Load aggregated statistics
    mean_path = output_dir / f'{prefix}bootstrap_mean.npy'
    if not mean_path.exists():
        raise FileNotFoundError(f"Bootstrap mean file not found: {mean_path}")
    mean_zscore = np.load(mean_path)

    std_path = output_dir / f'{prefix}bootstrap_std.npy'
    if not std_path.exists():
        raise FileNotFoundError(f"Bootstrap std file not found: {std_path}")
    std_zscore = np.load(std_path)

    ci_lower_path = output_dir / f'{prefix}bootstrap_ci_lower.npy'
    if not ci_lower_path.exists():
        raise FileNotFoundError(f"Bootstrap CI lower file not found: {ci_lower_path}")
    ci_lower = np.load(ci_lower_path)

    ci_upper_path = output_dir / f'{prefix}bootstrap_ci_upper.npy'
    if not ci_upper_path.exists():
        raise FileNotFoundError(f"Bootstrap CI upper file not found: {ci_upper_path}")
    ci_upper = np.load(ci_upper_path)

    # Load metadata
    metadata_path = output_dir / f'{prefix}bootstrap_metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return {
        'zscores': zscores,
        'mean_zscore': mean_zscore,
        'std_zscore': std_zscore,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'metadata': metadata,
        'cell_types': metadata['cell_types'],
        'n_bootstrap': metadata['n_bootstrap'],
        'tile_name': metadata.get('tile_name'),
        'parameters': metadata.get('parameters', {})
    }


def run_bootstrap_pipeline(
    adata_path,
    tile_key='tile_name',
    output_dir='bootstrap_spatial_analysis',
    n_bootstrap=100,
    radius=50,
    n_perms_bootstrap=100,
    n_perms_standard=1000,
    cluster_key='cell_type',
    save_adata=False,
    save_matrix_csvs=False,
    seed=42,
    max_zscore=50.0,
    min_cells_per_type=5
):
    """
    Run complete bootstrap-permutation analysis pipeline for tiled images.

    This pipeline:
    1. Loads tiled image data
    2. Runs hierarchical bootstrap-permutation analysis
    3. Runs standard permutation for comparison
    4. Generates visualizations with confidence intervals
    5. Summarizes significant interactions

    Parameters:
    -----------
    adata_path : str
        Path to h5ad file with tiled image data
    tile_key : str, default='tile_name'
        Key in adata.obs containing tile identifiers
    output_dir : str, default='bootstrap_spatial_analysis'
        Directory to save results
    n_bootstrap : int, default=100
        Number of bootstrap iterations
    radius : float, default=50
        Radius for spatial graph (pixels)
    n_perms_bootstrap : int, default=100
        Permutations per bootstrap iteration (can be lower since we do many bootstraps)
    n_perms_standard : int, default=1000
        Permutations for standard analysis
    cluster_key : str, default='cell_type'
        Key for cell type labels
    save_adata : bool, default=False
        Whether to save processed data
    save_matrix_csvs : bool, default=False
        Whether to save full matrix CSVs (mean, std, CI). If False, only saves
        bootstrap_significant_interactions.csv which contains the key results.
    seed : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    results : dict
        Dictionary containing all analysis results
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("BOOTSTRAP-PERMUTATION SPATIAL ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")

    # Load data
    print(f"\nLoading data from: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    print(f"  - Loaded {adata.n_obs:,} cells")

    # Apply cell type colors
    load_and_apply_cell_type_colors(adata, celltype_key=cluster_key)

    # Verify tile information exists
    if tile_key not in adata.obs.columns:
        raise ValueError(f"Tile key '{tile_key}' not found. Cannot perform tile-level bootstrap!")

    # Step 1: Bootstrap-Permutation Analysis
    print("\n" + "=" * 70)
    print("STEP 1: BOOTSTRAP-PERMUTATION ANALYSIS")
    print("=" * 70)

    bootstrap_results = run_bootstrap_permutation_analysis(
        adata,
        tile_key=tile_key,
        n_bootstrap=n_bootstrap,
        radius=radius,
        n_perms=n_perms_bootstrap,
        cluster_key=cluster_key,
        seed=seed,
        max_zscore=max_zscore,
        min_cells_per_type=min_cells_per_type
    )

    # Save intermediate results for later aggregation
    print("\n  - Saving intermediate bootstrap results for aggregation...")
    tile_name = Path(adata_path).stem
    save_bootstrap_intermediate_results(
        bootstrap_results,
        output_dir=output_dir,
        tile_name=tile_name
    )

    # Step 2: Standard Permutation Analysis (for comparison)
    print("\n" + "=" * 70)
    print("STEP 2: STANDARD PERMUTATION ANALYSIS")
    print("=" * 70)

    adata_standard, comparison = compare_with_standard_permutation(
        adata,
        bootstrap_results,
        radius=radius,
        n_perms=n_perms_standard,
        cluster_key=cluster_key
    )

    # Step 3: Visualizations
    print("\n" + "=" * 70)
    print("STEP 3: GENERATING VISUALIZATIONS")
    print("=" * 70)

    # Bootstrap enrichment with CI
    visualize_bootstrap_enrichment(
        bootstrap_results,
        cluster_key=cluster_key,
        save_path=output_dir / 'bootstrap_enrichment_with_ci.png'
    )

    # Bootstrap uncertainty plot
    visualize_bootstrap_comparison(
        bootstrap_results,
        comparison['standard_zscore'],
        bootstrap_results['cell_types'],
        save_path=output_dir / 'bootstrap_uncertainty.png'
    )

    # Step 4: Summarize Interactions
    print("\n" + "=" * 70)
    print("STEP 4: SUMMARIZING INTERACTIONS")
    print("=" * 70)

    # Bootstrap interactions
    bootstrap_interactions = summarize_bootstrap_interactions(
        bootstrap_results,
        threshold_zscore=2.0,
        threshold_ci=True
    )
    bootstrap_interactions.to_csv(
        output_dir / 'bootstrap_significant_interactions.csv',
        index=False
    )
    print(f"\n  - Saved bootstrap interactions to: {output_dir / 'bootstrap_significant_interactions.csv'}")

    # Standard interactions (for comparison)
    print("\n" + "=" * 70)
    print("STANDARD PERMUTATION INTERACTIONS (for comparison)")
    print("=" * 70)
    standard_interactions = summarize_interactions(adata_standard, cluster_key=cluster_key)
    standard_interactions.to_csv(
        output_dir / 'standard_significant_interactions.csv',
        index=False
    )
    print(f"\n  - Saved standard interactions to: {output_dir / 'standard_significant_interactions.csv'}")

    # Step 5: Save Results
    print("\n" + "=" * 70)
    print("STEP 5: SAVING RESULTS")
    print("=" * 70)

    # Save bootstrap statistics (optional, contains duplicate info with interactions CSV)
    if save_matrix_csvs:
        cell_types = bootstrap_results['cell_types']

        print("\n  - Saving full matrix CSVs...")

        # Mean z-scores
        mean_df = pd.DataFrame(
            bootstrap_results['mean_zscore'],
            index=cell_types,
            columns=cell_types
        )
        mean_df.to_csv(output_dir / 'bootstrap_mean_zscore.csv')

        # Standard deviations
        std_df = pd.DataFrame(
            bootstrap_results['std_zscore'],
            index=cell_types,
            columns=cell_types
        )
        std_df.to_csv(output_dir / 'bootstrap_std_zscore.csv')

        # Confidence intervals
        ci_lower_df = pd.DataFrame(
            bootstrap_results['ci_lower'],
            index=cell_types,
            columns=cell_types
        )
        ci_lower_df.to_csv(output_dir / 'bootstrap_ci_lower.csv')

        ci_upper_df = pd.DataFrame(
            bootstrap_results['ci_upper'],
            index=cell_types,
            columns=cell_types
        )
        ci_upper_df.to_csv(output_dir / 'bootstrap_ci_upper.csv')

        print(f"    • bootstrap_mean_zscore.csv")
        print(f"    • bootstrap_std_zscore.csv")
        print(f"    • bootstrap_ci_lower.csv")
        print(f"    • bootstrap_ci_upper.csv")
    else:
        print("\n  - Skipping matrix CSVs (set save_matrix_csvs=True to enable)")
        print("    Key results are in bootstrap_significant_interactions.csv")

    # Comparison metrics (always save)
    comparison_df = pd.DataFrame([{
        'mean_absolute_difference': comparison['mean_absolute_difference'],
        'max_absolute_difference': comparison['max_absolute_difference'],
        'correlation': comparison['correlation']
    }])
    comparison_df.to_csv(output_dir / 'bootstrap_vs_standard_comparison.csv', index=False)

    print(f"\n  - Saved results to: {output_dir}/")

    # Save AnnData if requested
    if save_adata:
        output_adata_path = output_dir / 'adata_with_bootstrap_analysis.h5ad'
        adata_standard.write(output_adata_path)
        print(f"  - Saved AnnData to: {output_adata_path}")

    # Compile results
    results = {
        'bootstrap': bootstrap_results,
        'standard': {
            'zscore': comparison['standard_zscore'],
            'adata': adata_standard
        },
        'comparison': comparison,
        'interactions': {
            'bootstrap': bootstrap_interactions,
            'standard': standard_interactions
        }
    }

    # Final summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nBootstrap Summary:")
    print(f"  - Bootstrap iterations: {bootstrap_results['n_bootstrap']}")
    print(f"  - Permutations per bootstrap: {n_perms_bootstrap}")
    print(f"  - Mean uncertainty (std): {bootstrap_results['std_zscore'].mean():.3f}")
    print(f"\nComparison with Standard Permutation:")
    print(f"  - Correlation: {comparison['correlation']:.3f}")
    print(f"  - Mean absolute difference: {comparison['mean_absolute_difference']:.3f}")
    print(f"\nSignificant Interactions:")
    print(f"  - Bootstrap method: {len(bootstrap_interactions)}")
    print(f"  - Standard method: {len(standard_interactions)}")
    print(f"\nAll results saved to: {output_dir}/")
    print(f"\nNote: Run neighborhood_enrichment_tiled.py for spatial distribution visualization")

    return results


# Example usage
if __name__ == "__main__":
    # Configuration
    adata_path = '../tile_39520_7904.h5ad'  # Replace with your tiled image data
    output_dir = 'bootstrap_spatial_analysis'

    # Run bootstrap pipeline
    results = run_bootstrap_pipeline(
        adata_path=adata_path,
        tile_key='tile_name',           # Adjust to your tile identifier column
        output_dir=output_dir,
        n_bootstrap=100,                # More bootstraps = better uncertainty estimates
        radius=50,                      # Adjust based on your tissue/magnification
        n_perms_bootstrap=100,          # Can be lower since we do many bootstraps
        n_perms_standard=1000,          # Higher for single standard analysis
        cluster_key='cell_type',        # Adjust to your cell type column
        save_adata=False,               # Set to True to save processed data
        seed=42,
        max_zscore=50.0,               # Clip extreme z-scores
        min_cells_per_type=5           # Minimum cells per type for valid analysis
    )

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
Bootstrap-Permutation Results Interpretation:

1. MEAN Z-SCORES: Average enrichment across bootstrap iterations
   - Positive = attraction between cell types
   - Negative = avoidance between cell types
   - Same interpretation as standard permutation

2. CONFIDENCE INTERVALS (CI):
   - Narrow CI = consistent pattern across tiles (robust finding)
   - Wide CI = pattern varies across tiles (tile-specific or unstable)
   - If CI excludes zero → strong evidence for interaction

3. COMPARISON WITH STANDARD:
   - High correlation (>0.95) → tile variability is low, standard method OK
   - Low correlation (<0.85) → tile variability is high, bootstrap method essential
   - Large differences → some interactions are tile-specific

4. SIGNIFICANCE CRITERIA:
   - Bootstrap method: Mean |Z| > 2 AND CI excludes zero
   - More stringent than standard permutation
   - Accounts for both spatial randomness AND tile sampling variability

5. UNCERTAINTY (STD DEV):
   - Shows which interactions are most variable across tiles
   - High std = interaction strength depends on tissue region
   - Low std = interaction is consistent across all tiles

RECOMMENDATION:
- If bootstrap and standard results agree → both methods valid
- If they disagree → trust bootstrap (accounts for hierarchical structure)
- Always report confidence intervals for transparency
""")