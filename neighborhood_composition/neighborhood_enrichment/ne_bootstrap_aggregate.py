"""
Bootstrap Permutation Analysis - Aggregation Only

This module aggregates intermediate bootstrap results from multiple tiles
that have already been processed by ne_bootstrap_multiple.py or ne_bootstrap_tiled.py.

Key features:
- Loads intermediate results from multiple tiles
- Pools all bootstrap iterations across tiles
- Generates aggregated visualizations with confidence intervals
- Summarizes significant interactions across the entire sample
- Memory-efficient: loads one tile at a time

Workflow:
1. Find directories containing intermediate bootstrap results
2. Load and pool bootstrap iterations from all tiles
3. Compute aggregated statistics (mean, std, CI)
4. Generate visualizations
5. Identify and save significant interactions

Author: Generated with Claude Code
Date: 2025-10-23
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict
import warnings
from pathlib import Path
import os
# Set the working directory to the script's directory
os.chdir(Path(__file__).parent)

# Import functions from ne_bootstrap_tiled
from ne_bootstrap_tiled import (
    load_bootstrap_intermediate_results
)

warnings.filterwarnings('ignore')


def handle_extreme_values_in_aggregation(zscores_array, max_zscore=50.0):
    """
    Handle infinite and extreme values during aggregation.
    
    Parameters:
    -----------
    zscores_array : np.array
        Array of z-scores that may contain infinite or extreme values
    max_zscore : float, default=50.0
        Maximum allowed z-score value
        
    Returns:
    --------
    zscores_clean : np.array
        Array with extreme values handled
    """
    zscores_clean = zscores_array.copy()
    
    # Handle infinite values
    inf_mask = np.isinf(zscores_clean)
    if inf_mask.any():
        # Replace infinite values with clipped finite values
        finite_values = zscores_clean[np.isfinite(zscores_clean)]
        if len(finite_values) > 0:
            max_finite = np.max(finite_values)
            min_finite = np.min(finite_values)
            
            zscores_clean[zscores_clean == np.inf] = min(max_finite, max_zscore)
            zscores_clean[zscores_clean == -np.inf] = max(min_finite, -max_zscore)
        else:
            zscores_clean[inf_mask] = 0.0
    
    # Handle NaN values
    nan_mask = np.isnan(zscores_clean)
    if nan_mask.any():
        zscores_clean[nan_mask] = 0.0
    
    # Clip extreme values
    zscores_clean = np.clip(zscores_clean, -max_zscore, max_zscore)
    
    return zscores_clean


def find_bootstrap_result_dirs(parent_dir, pattern='tile_*'):
    """
    Find all directories containing bootstrap intermediate results.

    Parameters:
    -----------
    parent_dir : str or Path
        Parent directory containing tile subdirectories
    pattern : str, default='tile_*'
        Pattern to match tile directory names

    Returns:
    --------
    result_dirs : list of Path
        List of directories containing bootstrap results
    """
    parent_dir = Path(parent_dir)

    # Find all matching directories
    all_dirs = sorted(parent_dir.glob(pattern))

    # Filter to only those with bootstrap results
    result_dirs = []
    for tile_dir in all_dirs:
        if tile_dir.is_dir():
            # Check if it has bootstrap metadata file
            metadata_file = list(tile_dir.glob('*_bootstrap_metadata.json'))
            if metadata_file:
                result_dirs.append(tile_dir)

    print(f"Found {len(result_dirs)} directories with bootstrap results in {parent_dir}")

    return result_dirs


def aggregate_bootstrap_results_from_tiles(
    tile_dirs,
    output_dir,
    tile_names=None,
    save_matrix_csvs=False,
    max_zscore=50.0
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
    save_matrix_csvs : bool, default=False
        Whether to save full matrix CSVs. If False, only saves interactions CSV.
    max_zscore : float, default=50.0
        Maximum z-score value for handling extreme values

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
            zscores = results['zscores']
            metadata = results['metadata']

            # Validate data
            print(f"  [{i+1}/{len(tile_dirs)}] Loaded: {results['tile_name'] or tile_dir.name} "
                  f"({results['n_bootstrap']} bootstrap iterations)")
            print(f"      Shape: {zscores.shape}, "
                  f"Z-score range: [{np.nanmin(zscores):.2f}, {np.nanmax(zscores):.2f}]")

            # Check for compatibility with first tile
            if len(all_metadata) > 0:
                first_celltypes = set(all_metadata[0]['cell_types'])
                current_celltypes = set(metadata['cell_types'])
                if first_celltypes != current_celltypes:
                    missing = first_celltypes - current_celltypes
                    extra = current_celltypes - first_celltypes
                    print(f"      [!] Warning: Cell type mismatch with first tile!")
                    if missing:
                        print(f"          Missing: {missing}")
                    if extra:
                        print(f"          Extra: {extra}")

            all_zscores.append(zscores)
            all_metadata.append(metadata)
            actual_tile_names.append(results['tile_name'] or tile_dir.name)

        except Exception as e:
            print(f"  [!] Warning: Could not load {tile_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_zscores) == 0:
        raise ValueError("No valid results found to aggregate!")

    # Pool all bootstrap iterations from all tiles
    print(f"\nPooling bootstrap iterations from {len(all_zscores)} tiles...")

    # Concatenate all bootstrap iterations along the first axis
    # Each tile has shape (n_bootstrap_per_tile, n_celltypes, n_celltypes)
    # Result: (n_tiles * n_bootstrap_per_tile, n_celltypes, n_celltypes)
    pooled_zscores = np.concatenate(all_zscores, axis=0)
    
    # Handle extreme values in pooled data
    print(f"  - Handling extreme values in pooled data...")
    pooled_zscores_clean = handle_extreme_values_in_aggregation(pooled_zscores, max_zscore=max_zscore)

    n_total_bootstrap = pooled_zscores.shape[0]
    n_celltypes = pooled_zscores.shape[1]

    print(f"  - Total bootstrap iterations pooled: {n_total_bootstrap}")
    print(f"  - From {len(all_zscores)} tiles")
    print(f"  - Cell types: {n_celltypes}")

    # Compute aggregated statistics using cleaned data
    print(f"\nComputing aggregated statistics...")
    mean_zscore = np.nanmean(pooled_zscores_clean, axis=0)
    std_zscore = np.nanstd(pooled_zscores_clean, axis=0)
    median_zscore = np.nanmedian(pooled_zscores_clean, axis=0)

    # Compute 95% confidence intervals using cleaned data
    ci_lower = np.nanpercentile(pooled_zscores_clean, 2.5, axis=0)
    ci_upper = np.nanpercentile(pooled_zscores_clean, 97.5, axis=0)
    
    # Additional robust statistics
    trimmed_mean = np.array([
        np.nanmean(np.clip(pooled_zscores_clean[:, i, j], 
                          np.nanpercentile(pooled_zscores_clean[:, i, j], 5),
                          np.nanpercentile(pooled_zscores_clean[:, i, j], 95)))
        for i in range(pooled_zscores_clean.shape[1])
        for j in range(pooled_zscores_clean.shape[2])
    ]).reshape(pooled_zscores_clean.shape[1], pooled_zscores_clean.shape[2])

    # Get cell types from first tile
    cell_types = all_metadata[0]['cell_types']

    # Data quality checks
    print(f"\n  Data Quality Checks:")
    print(f"  - Mean z-score range: [{mean_zscore.min():.2f}, {mean_zscore.max():.2f}]")
    print(f"  - Mean z-score: {np.nanmean(mean_zscore):.3f}")
    print(f"  - Std range: [{std_zscore.min():.2f}, {std_zscore.max():.2f}]")
    print(f"  - Mean std across cell type pairs: {np.nanmean(std_zscore):.3f}")
    print(f"  - Mean CI width: {np.nanmean(ci_upper - ci_lower):.3f}")

    # Check for extreme values in original data
    n_inf_original = np.isinf(pooled_zscores).sum()
    n_nan_original = np.isnan(pooled_zscores).sum()
    n_extreme_original = (np.abs(pooled_zscores) > max_zscore).sum()
    
    print(f"  - Extreme values in original data:")
    print(f"    • Infinite values: {n_inf_original}")
    print(f"    • NaN values: {n_nan_original}")
    print(f"    • Extreme values (> {max_zscore}): {n_extreme_original}")

    # Check for NaN or Inf values in aggregated results
    n_nan_mean = np.isnan(mean_zscore).sum()
    n_inf_mean = np.isinf(mean_zscore).sum()
    n_nan_std = np.isnan(std_zscore).sum()
    n_inf_std = np.isinf(std_zscore).sum()

    if n_nan_mean > 0 or n_inf_mean > 0:
        print(f"  [!] Warning: Mean z-scores contain {n_nan_mean} NaN and {n_inf_mean} Inf values")
    if n_nan_std > 0 or n_inf_std > 0:
        print(f"  [!] Warning: Std deviations contain {n_nan_std} NaN and {n_inf_std} Inf values")

    # Check for unusually high std values
    high_std_mask = std_zscore > 3.0
    n_high_std = high_std_mask.sum()
    if n_high_std > 0:
        print(f"  [!] Warning: {n_high_std} cell type pairs have std > 3.0")
        print(f"      Max std: {std_zscore.max():.2f}")
        # Find which pairs have high std
        high_std_indices = np.argwhere(high_std_mask)
        if len(high_std_indices) > 0:
            print(f"      Examples of high-std pairs:")
            for idx in high_std_indices[:5]:
                i, j = idx
                print(f"        {cell_types[i]} - {cell_types[j]}: std={std_zscore[i,j]:.2f}")

    # Save aggregated statistics (optional)
    if save_matrix_csvs:
        print(f"\nSaving aggregated matrix CSVs to {output_dir}/...")

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

        print(f"  - Saved aggregated_bootstrap_mean_zscore.csv")
        print(f"  - Saved aggregated_bootstrap_std_zscore.csv")
        print(f"  - Saved aggregated_bootstrap_median_zscore.csv")
        print(f"  - Saved aggregated_bootstrap_ci_lower.csv")
        print(f"  - Saved aggregated_bootstrap_ci_upper.csv")
    else:
        print(f"\nSkipping matrix CSVs (set save_matrix_csvs=True to enable)")
        print("  Key results will be in aggregated_bootstrap_significant_interactions.csv")

    # Prepare aggregated results dictionary
    aggregated = {
        'mean_zscore': mean_zscore,
        'std_zscore': std_zscore,
        'median_zscore': median_zscore,
        'trimmed_mean': trimmed_mean,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'cell_types': cell_types,
        'n_tiles': len(all_zscores),
        'n_total_bootstrap': n_total_bootstrap,
        'tile_names': actual_tile_names,
        'metadata_list': all_metadata,
        'pooled_zscores': pooled_zscores,  # Keep original for reference
        'pooled_zscores_clean': pooled_zscores_clean,  # Cleaned version
        'data_quality': {
            'n_inf_original': n_inf_original,
            'n_nan_original': n_nan_original,
            'n_extreme_original': n_extreme_original,
            'max_zscore_used': max_zscore,
            'n_nan_mean': n_nan_mean,
            'n_inf_mean': n_inf_mean,
            'n_nan_std': n_nan_std,
            'n_inf_std': n_inf_std
        }
    }

    print("\n" + "="*70)
    print("AGGREGATION COMPLETE!")
    print("="*70)

    return aggregated


def visualize_aggregated_bootstrap_enrichment(
    aggregated_results,
    output_dir,
    figsize=(14, 10),
    cmap='coolwarm',
    max_zscore=50.0
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
    max_zscore : float, default=50.0
        Maximum z-score value for color scale

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

    # Calculate dynamic color scale (ignoring NaN and extreme values)
    finite_values = mean_zscore[np.isfinite(mean_zscore)]
    if len(finite_values) > 0:
        max_abs_value = max(abs(np.nanmin(finite_values)), abs(np.nanmax(finite_values)))
        vmin = -min(max_abs_value, max_zscore)
        vmax = min(max_abs_value, max_zscore)
    else:
        vmin = -max_zscore
        vmax = max_zscore

    print(f"  - Dynamic color scale: [{vmin:.2f}, {vmax:.2f}]")

    # Check for NaN values that might cause empty plots
    n_nan = np.isnan(mean_zscore).sum()
    if n_nan > 0:
        print(f"  [!] Warning: {n_nan} NaN values in mean z-scores - these will appear as white/empty in the plot")

    # Create annotations with CI
    annotations = []
    for i in range(len(cell_types)):
        row = []
        for j in range(len(cell_types)):
            mean = mean_zscore[i, j]
            lower = ci_lower[i, j]
            upper = ci_upper[i, j]

            # Handle NaN values
            if np.isnan(mean) or np.isnan(lower) or np.isnan(upper):
                row.append("N/A")
                continue

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


def run_aggregation_pipeline(
    results_dir,
    output_dir='aggregated_results',
    tile_pattern='tile_*',
    save_matrix_csvs=False,
    threshold_zscore=2.0,
    threshold_ci=True,
    max_zscore=50.0
):
    """
    Complete aggregation pipeline: load intermediate results and aggregate.

    This pipeline:
    1. Finds all directories with intermediate bootstrap results
    2. Loads and pools bootstrap iterations from all tiles
    3. Computes aggregated statistics
    4. Generates visualizations
    5. Identifies and saves significant interactions

    Parameters:
    -----------
    results_dir : str or Path
        Directory containing tile subdirectories with intermediate results
    output_dir : str or Path, default='aggregated_results'
        Directory to save aggregated results
    tile_pattern : str, default='tile_*'
        Pattern to match tile directory names
    save_matrix_csvs : bool, default=False
        Whether to save full matrix CSVs
    threshold_zscore : float, default=2.0
        Z-score threshold for significance
    threshold_ci : bool, default=True
        Require 95% CI to exclude zero for significance
    max_zscore : float, default=50.0
        Maximum z-score value for handling extreme values

    Returns:
    --------
    results : dict
        Dictionary containing aggregated results and interactions
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "="*70)
    print("BOOTSTRAP AGGREGATION PIPELINE")
    print("="*70)
    print(f"\nResults directory: {results_dir}")
    print(f"Output directory: {output_dir}")

    # Step 1: Find intermediate result directories
    print("\n" + "="*70)
    print("STEP 1: FINDING INTERMEDIATE RESULTS")
    print("="*70)

    tile_dirs = find_bootstrap_result_dirs(results_dir, pattern=tile_pattern)

    if len(tile_dirs) == 0:
        raise ValueError(f"No intermediate results found in {results_dir} with pattern '{tile_pattern}'")

    # Step 2: Aggregate results
    print("\n" + "="*70)
    print("STEP 2: AGGREGATING BOOTSTRAP RESULTS")
    print("="*70)

    tile_names = [tile_dir.name for tile_dir in tile_dirs]
    aggregated_results = aggregate_bootstrap_results_from_tiles(
        tile_dirs,
        output_dir=output_dir,
        tile_names=tile_names,
        save_matrix_csvs=save_matrix_csvs,
        max_zscore=max_zscore
    )

    # Step 3: Visualizations
    print("\n" + "="*70)
    print("STEP 3: GENERATING VISUALIZATIONS")
    print("="*70)

    visualize_aggregated_bootstrap_enrichment(
        aggregated_results,
        output_dir=output_dir,
        max_zscore=max_zscore
    )

    visualize_aggregated_uncertainty(
        aggregated_results,
        output_dir=output_dir
    )

    # Step 4: Summarize interactions
    print("\n" + "="*70)
    print("STEP 4: SUMMARIZING SIGNIFICANT INTERACTIONS")
    print("="*70)

    interactions_df = summarize_aggregated_interactions(
        aggregated_results,
        output_dir=output_dir,
        threshold_zscore=threshold_zscore,
        threshold_ci=threshold_ci
    )

    # Compile results
    results = {
        'aggregated_results': aggregated_results,
        'interactions': interactions_df,
        'output_dir': output_dir
    }

    # Final summary
    print("\n" + "="*70)
    print("AGGREGATION PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nAggregated Bootstrap Analysis:")
    print(f"  - Tiles included: {aggregated_results['n_tiles']}")
    print(f"  - Total bootstrap iterations: {aggregated_results['n_total_bootstrap']}")
    print(f"  - Significant interactions: {len(interactions_df)}")
    print(f"\nAll results saved to: {output_dir}/")

    return results


# Example usage
if __name__ == "__main__":
    # Configuration
    results_dir = Path('bootstrap_multiple_analysis')  # Directory with tile subdirectories
    output_dir = Path('bootstrap_multiple_analysis/aggregated_results')

    # Run aggregation pipeline
    results = run_aggregation_pipeline(
        results_dir=results_dir,
        output_dir=output_dir,
        tile_pattern='tile_*',           # Pattern to match tile directories
        save_matrix_csvs=False,          # Set to True to save full matrices
        threshold_zscore=2.0,            # Z-score threshold
        threshold_ci=True,               # Require CI to exclude zero
        max_zscore=50.0                  # Handle extreme values
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
- aggregated_bootstrap_enrichment.png: Main visualization with CI
- aggregated_bootstrap_uncertainty.png: Variability heatmap
- aggregated_bootstrap_significant_interactions.csv: Significant pairs

OPTIONAL FILES (if save_matrix_csvs=True):
- aggregated_bootstrap_mean_zscore.csv: Full mean z-score matrix
- aggregated_bootstrap_std_zscore.csv: Full std deviation matrix
- aggregated_bootstrap_ci_lower/upper.csv: Full confidence interval matrices
""")