"""
Bootstrap Permutation Analysis for Multiple Tiles - Batch Processing and Aggregation

This module processes multiple tiled h5ad files in batch, running hierarchical
bootstrap-permutation analysis on each tile, and aggregates the results to
provide robust uncertainty estimates across the entire tissue sample.

Key features:
- Batch processing of multiple tiles
- Memory-efficient: processes one tile at a time
- Saves intermediate results for each tile
- Aggregates bootstrap iterations across all tiles
- Generates comprehensive visualizations with confidence intervals
- Compares aggregated bootstrap vs. standard permutation results

Workflow:
1. Process each tile: bootstrap-permutation analysis
2. Save intermediate results (.npy files)
3. Aggregate all bootstrap iterations across tiles
4. Generate final visualizations and statistics

Author: Generated with Claude Code
Date: 2025-10-23
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm
import warnings

# Import functions from ne_bootstrap_tiled
from ne_bootstrap_tiled import (
    run_bootstrap_permutation_analysis,
    save_bootstrap_intermediate_results,
    load_bootstrap_intermediate_results,
    visualize_bootstrap_enrichment,
    visualize_bootstrap_comparison,
    summarize_bootstrap_interactions
)

# Import functions from ne_tiled
from ne_tiled import (
    load_and_apply_cell_type_colors,
    build_spatial_graph,
    neighborhood_enrichment_analysis,
    summarize_interactions
)

warnings.filterwarnings('ignore')


def find_h5ad_files(data_dir, pattern='*.h5ad'):
    """
    Find all h5ad files in a directory.

    Parameters:
    -----------
    data_dir : str or Path
        Directory to search for h5ad files
    pattern : str, default='*.h5ad'
        Pattern to match files

    Returns:
    --------
    h5ad_files : list of Path
        List of paths to h5ad files
    """
    data_dir = Path(data_dir)
    h5ad_files = sorted(data_dir.glob(pattern))

    print(f"Found {len(h5ad_files)} h5ad files in {data_dir}")

    return h5ad_files


def process_single_tile_bootstrap(
    h5ad_path,
    output_dir,
    tile_key='tile_name',
    n_bootstrap=100,
    method='knn',
    radius=50,
    n_neighbors=6,
    n_perms=100,
    cluster_key='cell_type',
    seed=42
):
    """
    Process a single tile: run bootstrap analysis and save intermediate results.

    Parameters:
    -----------
    h5ad_path : str or Path
        Path to h5ad file
    output_dir : str or Path
        Directory to save intermediate results
    tile_key : str, default='tile_name'
        Key for tile identifiers
    n_bootstrap : int, default=100
        Number of bootstrap iterations
    method : str, default='knn'
        Method for spatial graph: 'knn' or 'radius'
    radius : float, default=50
        Radius for spatial graph (if method='radius')
    n_neighbors : int, default=6
        Number of neighbors for KNN (if method='knn')
    n_perms : int, default=100
        Number of permutations per bootstrap
    cluster_key : str, default='cell_type'
        Key for cell type labels
    seed : int, default=42
        Random seed

    Returns:
    --------
    tile_name : str
        Name of the processed tile
    success : bool
        Whether processing was successful
    """
    h5ad_path = Path(h5ad_path)
    output_dir = Path(output_dir)
    tile_name = h5ad_path.stem

    print(f"\n{'='*70}")
    print(f"Processing: {tile_name}")
    print(f"{'='*70}")

    try:
        # Load data
        print(f"Loading {h5ad_path.name}...")
        adata = sc.read_h5ad(h5ad_path)
        print(f"  - Loaded {adata.n_obs:,} cells")

        # Apply cell type colors
        load_and_apply_cell_type_colors(adata, celltype_key=cluster_key)

        # Run bootstrap-permutation analysis
        bootstrap_results = run_bootstrap_permutation_analysis(
            adata,
            tile_key=tile_key,
            n_bootstrap=n_bootstrap,
            method=method,
            radius=radius,
            n_neighbors=n_neighbors,
            n_perms=n_perms,
            cluster_key=cluster_key,
            seed=seed
        )

        # Save intermediate results
        print(f"\nSaving intermediate results for {tile_name}...")
        tile_output_dir = output_dir / tile_name
        save_bootstrap_intermediate_results(
            bootstrap_results,
            output_dir=tile_output_dir,
            tile_name=tile_name
        )

        print(f"✓ Successfully processed {tile_name}")
        return tile_name, True

    except Exception as e:
        print(f"✗ Failed to process {tile_name}: {e}")
        return tile_name, False


def process_multiple_tiles_bootstrap(
    h5ad_files,
    output_dir,
    tile_key='tile_name',
    n_bootstrap=100,
    method='knn',
    radius=50,
    n_neighbors=6,
    n_perms=100,
    cluster_key='cell_type',
    seed=42
):
    """
    Process multiple tiles in batch: run bootstrap analysis on each.

    Parameters:
    -----------
    h5ad_files : list of str/Path
        List of paths to h5ad files
    output_dir : str or Path
        Directory to save intermediate results
    tile_key : str, default='tile_name'
        Key for tile identifiers
    n_bootstrap : int, default=100
        Number of bootstrap iterations per tile
    method : str, default='knn'
        Method for spatial graph
    radius : float, default=50
        Radius for spatial graph (if method='radius')
    n_neighbors : int, default=6
        Number of neighbors for KNN (if method='knn')
    n_perms : int, default=100
        Number of permutations per bootstrap
    cluster_key : str, default='cell_type'
        Key for cell type labels
    seed : int, default=42
        Random seed

    Returns:
    --------
    results_summary : dict
        Summary of processing results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "="*70)
    print("BATCH PROCESSING: BOOTSTRAP-PERMUTATION ANALYSIS")
    print("="*70)
    print(f"\nProcessing {len(h5ad_files)} tiles...")
    print(f"Bootstrap iterations per tile: {n_bootstrap}")
    print(f"Permutations per bootstrap: {n_perms}")
    print(f"Spatial graph method: {method}")
    if method == 'knn':
        print(f"Number of neighbors: {n_neighbors}")
    else:
        print(f"Radius: {radius} pixels")

    # Track results
    successful_tiles = []
    failed_tiles = []

    # Process each tile
    for i, h5ad_path in enumerate(h5ad_files, 1):
        print(f"\n[{i}/{len(h5ad_files)}] {'-'*60}")

        tile_name, success = process_single_tile_bootstrap(
            h5ad_path,
            output_dir,
            tile_key=tile_key,
            n_bootstrap=n_bootstrap,
            method=method,
            radius=radius,
            n_neighbors=n_neighbors,
            n_perms=n_perms,
            cluster_key=cluster_key,
            seed=seed + i  # Different seed for each tile
        )

        if success:
            successful_tiles.append(tile_name)
        else:
            failed_tiles.append(tile_name)

    # Print summary
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"Successfully processed: {len(successful_tiles)}/{len(h5ad_files)} tiles")
    if failed_tiles:
        print(f"Failed tiles: {', '.join(failed_tiles)}")

    return {
        'successful_tiles': successful_tiles,
        'failed_tiles': failed_tiles,
        'n_total': len(h5ad_files),
        'n_success': len(successful_tiles),
        'n_failed': len(failed_tiles)
    }


def aggregate_bootstrap_results_from_tiles(
    tile_dirs,
    output_dir,
    tile_names=None
):
    """
    Aggregate bootstrap results from multiple tiles.

    This function loads bootstrap iterations from all tiles and pools them
    together to compute aggregated statistics across the entire tissue sample.

    Parameters:
    -----------
    tile_dirs : list of str/Path
        List of directories containing intermediate results (one per tile)
    output_dir : str or Path
        Directory to save aggregated results
    tile_names : list of str, optional
        List of tile names. If None, extracts from directory names.

    Returns:
    --------
    aggregated : dict
        Dictionary containing aggregated bootstrap statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "="*70)
    print("AGGREGATING BOOTSTRAP RESULTS FROM ALL TILES")
    print("="*70)
    print(f"Loading from {len(tile_dirs)} tile directories...")

    # Load all bootstrap results
    all_zscores = []
    all_metadata = []
    actual_tile_names = []

    for i, tile_dir in enumerate(tile_dirs):
        tile_dir = Path(tile_dir)
        tile_name = tile_names[i] if tile_names and i < len(tile_names) else tile_dir.name

        try:
            # Load intermediate results
            results = load_bootstrap_intermediate_results(tile_dir, tile_name=tile_name)

            # Get the full bootstrap array (n_bootstrap, n_celltypes, n_celltypes)
            all_zscores.append(results['zscores'])
            all_metadata.append(results['metadata'])
            actual_tile_names.append(results['tile_name'] or tile_dir.name)

            print(f"  [{i+1}/{len(tile_dirs)}] Loaded: {results['tile_name'] or tile_dir.name} "
                  f"({results['n_bootstrap']} bootstrap iterations)")

        except Exception as e:
            print(f"  [!] Warning: Could not load {tile_dir}: {e}")
            continue

    if len(all_zscores) == 0:
        raise ValueError("No valid results found to aggregate!")

    # Pool all bootstrap iterations from all tiles
    print(f"\nPooling bootstrap iterations from {len(all_zscores)} tiles...")

    # Concatenate all bootstrap iterations along the first axis
    # Each tile has shape (n_bootstrap_per_tile, n_celltypes, n_celltypes)
    # Result: (n_tiles * n_bootstrap_per_tile, n_celltypes, n_celltypes)
    pooled_zscores = np.concatenate(all_zscores, axis=0)

    n_total_bootstrap = pooled_zscores.shape[0]
    n_celltypes = pooled_zscores.shape[1]

    print(f"  - Total bootstrap iterations pooled: {n_total_bootstrap}")
    print(f"  - From {len(all_zscores)} tiles")
    print(f"  - Cell types: {n_celltypes}")

    # Compute aggregated statistics
    print(f"\nComputing aggregated statistics...")
    mean_zscore = pooled_zscores.mean(axis=0)
    std_zscore = pooled_zscores.std(axis=0)
    median_zscore = np.median(pooled_zscores, axis=0)

    # Compute 95% confidence intervals
    ci_lower = np.percentile(pooled_zscores, 2.5, axis=0)
    ci_upper = np.percentile(pooled_zscores, 97.5, axis=0)

    # Get cell types from first tile
    cell_types = all_metadata[0]['cell_types']

    print(f"  - Mean z-score range: [{mean_zscore.min():.2f}, {mean_zscore.max():.2f}]")
    print(f"  - Mean std across cell type pairs: {std_zscore.mean():.3f}")
    print(f"  - Mean CI width: {(ci_upper - ci_lower).mean():.3f}")

    # Save aggregated statistics
    print(f"\nSaving aggregated results to {output_dir}/...")

    # Save as CSV
    mean_df = pd.DataFrame(mean_zscore, index=cell_types, columns=cell_types)
    mean_df.to_csv(output_dir / 'aggregated_bootstrap_mean_zscore.csv')

    std_df = pd.DataFrame(std_zscore, index=cell_types, columns=cell_types)
    std_df.to_csv(output_dir / 'aggregated_bootstrap_std_zscore.csv')

    median_df = pd.DataFrame(median_zscore, index=cell_types, columns=cell_types)
    median_df.to_csv(output_dir / 'aggregated_bootstrap_median_zscore.csv')

    ci_lower_df = pd.DataFrame(ci_lower, index=cell_types, columns=cell_types)
    ci_lower_df.to_csv(output_dir / 'aggregated_bootstrap_ci_lower.csv')

    ci_upper_df = pd.DataFrame(ci_upper, index=cell_types, columns=cell_types)
    ci_upper_df.to_csv(output_dir / 'aggregated_bootstrap_ci_upper.csv')

    # Prepare aggregated results dictionary
    aggregated = {
        'mean_zscore': mean_zscore,
        'std_zscore': std_zscore,
        'median_zscore': median_zscore,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'cell_types': cell_types,
        'n_tiles': len(all_zscores),
        'n_total_bootstrap': n_total_bootstrap,
        'tile_names': actual_tile_names,
        'metadata_list': all_metadata,
        'pooled_zscores': pooled_zscores  # Keep for additional analysis if needed
    }

    print(f"  - Saved aggregated_bootstrap_mean_zscore.csv")
    print(f"  - Saved aggregated_bootstrap_std_zscore.csv")
    print(f"  - Saved aggregated_bootstrap_median_zscore.csv")
    print(f"  - Saved aggregated_bootstrap_ci_lower.csv")
    print(f"  - Saved aggregated_bootstrap_ci_upper.csv")

    print("\n" + "="*70)
    print("AGGREGATION COMPLETE!")
    print("="*70)

    return aggregated


def visualize_aggregated_bootstrap_enrichment(
    aggregated_results,
    output_dir,
    figsize=(14, 10),
    cmap='coolwarm'
):
    """
    Visualize aggregated bootstrap enrichment with confidence intervals.

    Parameters:
    -----------
    aggregated_results : dict
        Aggregated results from aggregate_bootstrap_results_from_tiles()
    output_dir : str or Path
        Directory to save visualizations
    figsize : tuple, default=(14, 10)
        Figure size
    cmap : str, default='coolwarm'
        Colormap

    Returns:
    --------
    fig : matplotlib Figure
        The generated figure
    """
    output_dir = Path(output_dir)

    print("\nVisualizing aggregated bootstrap enrichment...")

    mean_zscore = aggregated_results['mean_zscore']
    ci_lower = aggregated_results['ci_lower']
    ci_upper = aggregated_results['ci_upper']
    cell_types = aggregated_results['cell_types']
    n_tiles = aggregated_results['n_tiles']
    n_total_bootstrap = aggregated_results['n_total_bootstrap']

    # Calculate dynamic color scale
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
        cbar_kws={'label': 'Mean Z-score (Aggregated Bootstrap)'},
        linewidths=0.5,
        linecolor='white',
        square=True,
        ax=ax
    )

    ax.set_xlabel('Cell Type', fontsize=12)
    ax.set_ylabel('Cell Type', fontsize=12)
    ax.set_title(f'Aggregated Bootstrap Neighborhood Enrichment\n'
                 f'({n_tiles} tiles, {n_total_bootstrap} total bootstrap iterations)\n'
                 f'(Mean Z-score with 95% CI, *** = CI excludes zero)',
                 fontsize=14, fontweight='bold', pad=20)

    # Set tick labels
    ax.set_xticklabels(cell_types, rotation=45, ha='right')
    ax.set_yticklabels(cell_types, rotation=0)

    plt.tight_layout()

    save_path = output_dir / 'aggregated_bootstrap_enrichment.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved to: {save_path}")

    plt.close()

    return fig


def visualize_aggregated_uncertainty(
    aggregated_results,
    output_dir,
    figsize=(10, 8)
):
    """
    Visualize uncertainty (standard deviation) across bootstrap iterations.

    Parameters:
    -----------
    aggregated_results : dict
        Aggregated results
    output_dir : str or Path
        Directory to save visualizations
    figsize : tuple, default=(10, 8)
        Figure size

    Returns:
    --------
    fig : matplotlib Figure
    """
    output_dir = Path(output_dir)

    print("\nVisualizing aggregated bootstrap uncertainty...")

    std_zscore = aggregated_results['std_zscore']
    cell_types = aggregated_results['cell_types']
    n_tiles = aggregated_results['n_tiles']

    # Calculate auto-scale
    max_value = std_zscore.max()
    vmin, vmax = 0, np.ceil(max_value)

    print(f"  - Std Dev range: [0.00, {max_value:.2f}]")
    print(f"  - Color scale: [0, {vmax:.0f}]")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        std_zscore,
        cmap='YlOrRd',
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Standard Deviation'},
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        linecolor='white',
        xticklabels=cell_types,
        yticklabels=cell_types,
        square=True,
        ax=ax
    )

    ax.set_xlabel('Cell Type', fontsize=12)
    ax.set_ylabel('Cell Type', fontsize=12)
    ax.set_title(f'Bootstrap Uncertainty Across {n_tiles} Tiles\n'
                 f'(Std Dev of Z-scores)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    save_path = output_dir / 'aggregated_bootstrap_uncertainty.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  - Saved to: {save_path}")

    plt.close()

    return fig


def summarize_aggregated_interactions(
    aggregated_results,
    output_dir,
    threshold_zscore=2.0,
    threshold_ci=True
):
    """
    Summarize significant interactions from aggregated bootstrap results.

    Parameters:
    -----------
    aggregated_results : dict
        Aggregated results
    output_dir : str or Path
        Directory to save summary
    threshold_zscore : float, default=2.0
        Z-score threshold for significance
    threshold_ci : bool, default=True
        Require 95% CI to exclude zero

    Returns:
    --------
    interactions_df : DataFrame
        Significant interactions
    """
    output_dir = Path(output_dir)

    print("\n" + "="*70)
    print("SIGNIFICANT INTERACTIONS (Aggregated Bootstrap)")
    print("="*70)

    mean_zscore = aggregated_results['mean_zscore']
    ci_lower = aggregated_results['ci_lower']
    ci_upper = aggregated_results['ci_upper']
    cell_types = aggregated_results['cell_types']

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
                is_significant = (abs_z > threshold_zscore) and ci_excludes_zero
            else:
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

    # Save to CSV
    interactions_df.to_csv(
        output_dir / 'aggregated_bootstrap_significant_interactions.csv',
        index=False
    )
    print(f"\n  - Saved to: {output_dir / 'aggregated_bootstrap_significant_interactions.csv'}")

    return interactions_df


def run_bootstrap_multiple_pipeline(
    data_dir,
    output_dir='bootstrap_multiple_analysis',
    pattern='*.h5ad',
    tile_key='tile_name',
    n_bootstrap=100,
    method='knn',
    radius=50,
    n_neighbors=6,
    n_perms=100,
    cluster_key='cell_type',
    seed=42
):
    """
    Complete pipeline: batch process tiles and aggregate bootstrap results.

    This pipeline:
    1. Finds all h5ad files in data_dir
    2. Runs bootstrap-permutation analysis on each tile
    3. Saves intermediate results for each tile
    4. Aggregates all bootstrap iterations
    5. Generates visualizations and summary statistics

    Parameters:
    -----------
    data_dir : str or Path
        Directory containing h5ad files
    output_dir : str or Path, default='bootstrap_multiple_analysis'
        Directory to save all results
    pattern : str, default='*.h5ad'
        Pattern to match h5ad files
    tile_key : str, default='tile_name'
        Key for tile identifiers in adata.obs
    n_bootstrap : int, default=100
        Number of bootstrap iterations per tile
    method : str, default='knn'
        Spatial graph method: 'knn' or 'radius'
    radius : float, default=50
        Radius for spatial graph (if method='radius')
    n_neighbors : int, default=6
        Number of neighbors for KNN (if method='knn')
    n_perms : int, default=100
        Number of permutations per bootstrap
    cluster_key : str, default='cell_type'
        Key for cell type labels
    seed : int, default=42
        Random seed

    Returns:
    --------
    results : dict
        Dictionary containing all analysis results
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "="*70)
    print("BOOTSTRAP MULTIPLE TILES ANALYSIS PIPELINE")
    print("="*70)
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Step 1: Find all h5ad files
    print("\n" + "="*70)
    print("STEP 1: FINDING H5AD FILES")
    print("="*70)
    h5ad_files = find_h5ad_files(data_dir, pattern=pattern)

    if len(h5ad_files) == 0:
        raise ValueError(f"No h5ad files found in {data_dir} with pattern '{pattern}'")

    # Step 2: Process all tiles in batch
    print("\n" + "="*70)
    print("STEP 2: BATCH PROCESSING TILES")
    print("="*70)

    processing_summary = process_multiple_tiles_bootstrap(
        h5ad_files,
        output_dir=output_dir,
        tile_key=tile_key,
        n_bootstrap=n_bootstrap,
        method=method,
        radius=radius,
        n_neighbors=n_neighbors,
        n_perms=n_perms,
        cluster_key=cluster_key,
        seed=seed
    )

    # Step 3: Aggregate results from all tiles
    print("\n" + "="*70)
    print("STEP 3: AGGREGATING BOOTSTRAP RESULTS")
    print("="*70)

    # Get list of tile directories with intermediate results
    tile_dirs = [output_dir / tile_name for tile_name in processing_summary['successful_tiles']]

    aggregated_dir = output_dir / 'aggregated_results'
    aggregated_results = aggregate_bootstrap_results_from_tiles(
        tile_dirs,
        output_dir=aggregated_dir,
        tile_names=processing_summary['successful_tiles']
    )

    # Step 4: Visualizations
    print("\n" + "="*70)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("="*70)

    visualize_aggregated_bootstrap_enrichment(
        aggregated_results,
        output_dir=aggregated_dir
    )

    visualize_aggregated_uncertainty(
        aggregated_results,
        output_dir=aggregated_dir
    )

    # Step 5: Summarize interactions
    print("\n" + "="*70)
    print("STEP 5: SUMMARIZING SIGNIFICANT INTERACTIONS")
    print("="*70)

    interactions_df = summarize_aggregated_interactions(
        aggregated_results,
        output_dir=aggregated_dir,
        threshold_zscore=2.0,
        threshold_ci=True
    )

    # Compile final results
    results = {
        'processing_summary': processing_summary,
        'aggregated_results': aggregated_results,
        'interactions': interactions_df,
        'output_dir': output_dir,
        'aggregated_dir': aggregated_dir
    }

    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nProcessing Summary:")
    print(f"  - Total tiles: {processing_summary['n_total']}")
    print(f"  - Successfully processed: {processing_summary['n_success']}")
    print(f"  - Failed: {processing_summary['n_failed']}")
    print(f"\nAggregated Bootstrap Analysis:")
    print(f"  - Tiles included: {aggregated_results['n_tiles']}")
    print(f"  - Total bootstrap iterations: {aggregated_results['n_total_bootstrap']}")
    print(f"  - Significant interactions: {len(interactions_df)}")
    print(f"\nAll results saved to: {output_dir}/")
    print(f"Aggregated results in: {aggregated_dir}/")

    return results


# Example usage
if __name__ == "__main__":
    # Configuration
    data_dir = Path('../')  # Directory containing h5ad files
    output_dir = Path('bootstrap_multiple_analysis')

    # Run complete pipeline
    results = run_bootstrap_multiple_pipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        pattern='tile_*.h5ad',          # Pattern to match tile files
        tile_key='tile_name',            # Column in adata.obs with tile IDs
        n_bootstrap=100,                 # Bootstrap iterations per tile
        method='knn',                    # Spatial graph method
        n_neighbors=6,                   # KNN neighbors
        n_perms=100,                     # Permutations per bootstrap
        cluster_key='cell_type',         # Cell type column
        seed=42
    )

    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("""
Aggregated Bootstrap-Permutation Results:

1. POOLED BOOTSTRAP ITERATIONS:
   - All bootstrap iterations from all tiles are pooled together
   - If you process 10 tiles with 100 bootstraps each = 1,000 total iterations
   - This provides robust uncertainty estimates across the entire tissue

2. MEAN Z-SCORES:
   - Average enrichment across ALL bootstrap iterations from ALL tiles
   - Represents the overall pattern across the entire sample
   - More robust than analyzing single tiles

3. CONFIDENCE INTERVALS (CI):
   - 95% CI from pooled bootstrap distribution
   - Narrow CI = consistent pattern across entire sample
   - Wide CI = high variability between regions/tiles
   - CI excludes zero = strong evidence for interaction

4. SIGNIFICANCE CRITERIA:
   - Bootstrap method: Mean |Z| > 2 AND CI excludes zero
   - More stringent than standard permutation
   - Accounts for both spatial randomness AND tile sampling variability

5. UNCERTAINTY (STD DEV):
   - Standard deviation across all bootstrap iterations
   - Shows robustness of each cell type interaction
   - Low std = consistent across all tiles and iterations
   - High std = variable pattern (tile-specific or sampling variation)

ADVANTAGES OF AGGREGATED BOOTSTRAP:
- Pools data from multiple tissue regions
- Robust to tile-specific effects
- Proper uncertainty quantification
- Accounts for hierarchical structure (cells within tiles)

FILES GENERATED:
- aggregated_bootstrap_mean_zscore.csv: Mean z-scores
- aggregated_bootstrap_ci_lower/upper.csv: Confidence intervals
- aggregated_bootstrap_enrichment.png: Main visualization
- aggregated_bootstrap_uncertainty.png: Variability heatmap
- aggregated_bootstrap_significant_interactions.csv: Significant pairs
""")