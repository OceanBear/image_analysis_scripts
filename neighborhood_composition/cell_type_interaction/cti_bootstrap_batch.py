"""
Bootstrap Permutation Analysis for Multiple Tiles - Batch Processing Only

This module processes multiple tiled h5ad files in batch, running hierarchical
bootstrap-permutation analysis on each tile and saving intermediate results.

For aggregation of results across tiles, use cti_bootstrap_aggregate.py

Key features:
- Batch processing of multiple tiles
- Memory-efficient: processes one tile at a time
- Saves intermediate results for each tile (.npy and .json files)
- Detects and skips already-processed tiles
- Resumable: can continue interrupted batch jobs

Workflow:
1. Find all h5ad files in directory
2. Process each tile: bootstrap-permutation analysis
3. Save intermediate results (.npy files + metadata.json)
4. Use cti_bootstrap_aggregate.py to aggregate results

Author: Generated with Claude Code
Date: 2025-10-23
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Dict
import warnings
from pathlib import Path
import os
# Set the working directory to the script's directory
os.chdir(Path(__file__).parent)

# Import functions from ne_bootstrap_tiled
from cti_bootstrap_tiled import (
    run_bootstrap_permutation_analysis,
    save_bootstrap_intermediate_results,
    visualize_bootstrap_enrichment,
    visualize_bootstrap_comparison
)

# Import functions from ne_tiled
from cti_tiled import (
    load_and_apply_cell_type_colors
)

warnings.filterwarnings('ignore')


def visualize_bootstrap_enrichment_silent(bootstrap_results, cluster_key='cell_type', save_path=None):
    """
    Silent version of visualize_bootstrap_enrichment that doesn't show figures.
    """
    # Temporarily redirect stdout to suppress print statements
    import sys
    from io import StringIO
    
    # Capture print output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        # Call the original function
        fig = visualize_bootstrap_enrichment(
            bootstrap_results,
            cluster_key=cluster_key,
            save_path=save_path
        )
        return fig
    finally:
        # Restore stdout
        sys.stdout = old_stdout


def visualize_bootstrap_comparison_silent(bootstrap_results, standard_zscore, cell_types, save_path=None):
    """
    Silent version of visualize_bootstrap_comparison that doesn't show figures.
    """
    # Temporarily redirect stdout to suppress print statements
    import sys
    from io import StringIO
    
    # Capture print output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        # Call the original function
        fig = visualize_bootstrap_comparison(
            bootstrap_results,
            standard_zscore=standard_zscore,
            cell_types=cell_types,
            save_path=save_path
        )
        return fig
    finally:
        # Restore stdout
        sys.stdout = old_stdout


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


def save_bootstrap_results_with_figures(
    bootstrap_results,
    output_dir,
    tile_name,
    save_figures=True,
    figure_format='png',
    figure_dpi=300
):
    """
    Save bootstrap intermediate results and generate figures for a tile.

    Parameters:
    -----------
    bootstrap_results : dict
        Results from bootstrap analysis
    output_dir : str or Path
        Directory to save results
    tile_name : str
        Name of the tile
    save_figures : bool, default=True
        Whether to generate and save figures
    figure_format : str, default='png'
        Format for saved figures ('png', 'pdf', 'svg')
    figure_dpi : int, default=300
        DPI for saved figures

    Returns:
    --------
    saved_files : dict
        Dictionary with paths to saved files
    """
    output_dir = Path(output_dir)
    tile_output_dir = output_dir / tile_name
    tile_output_dir.mkdir(exist_ok=True, parents=True)

    # Save intermediate results (data files)
    saved_files = save_bootstrap_intermediate_results(
        bootstrap_results,
        output_dir=tile_output_dir,
        tile_name=tile_name
    )

    # Generate and save figures if requested
    if save_figures:
        print(f"  - Generating figures for {tile_name}...")
        
        # Temporarily disable interactive mode to prevent popup windows
        original_backend = matplotlib.get_backend()
        matplotlib.use('Agg')
        
        try:
            # Figure 1: Bootstrap enrichment with confidence intervals
            fig1_path = tile_output_dir / f'{tile_name}_bootstrap_enrichment.{figure_format}'
            try:
                fig1 = visualize_bootstrap_enrichment_silent(
                    bootstrap_results,
                    cluster_key=bootstrap_results.get('parameters', {}).get('cluster_key', 'cell_type'),
                    save_path=fig1_path
                )
                plt.close(fig1)  # Close to free memory
                print(f"    • Saved enrichment figure: {fig1_path.name}")
            except Exception as e:
                print(f"    ✗ Failed to generate enrichment figure: {e}")

            # Figure 2: Bootstrap uncertainty (standard deviation)
            fig2_path = tile_output_dir / f'{tile_name}_bootstrap_uncertainty.{figure_format}'
            try:
                fig2 = visualize_bootstrap_comparison_silent(
                    bootstrap_results,
                    standard_zscore=None,  # Not needed for uncertainty plot
                    cell_types=bootstrap_results['cell_types'],
                    save_path=fig2_path
                )
                plt.close(fig2)  # Close to free memory
                print(f"    • Saved uncertainty figure: {fig2_path.name}")
            except Exception as e:
                print(f"    ✗ Failed to generate uncertainty figure: {e}")

            # Add figure paths to saved files
            saved_files['enrichment_figure'] = fig1_path
            saved_files['uncertainty_figure'] = fig2_path
            
        finally:
            # Restore original backend
            matplotlib.use(original_backend)

    return saved_files


def check_intermediate_results_exist(tile_name, output_dir, include_figures=True):
    """
    Check if intermediate bootstrap results already exist for a tile.

    Parameters:
    -----------
    tile_name : str
        Name of the tile
    output_dir : Path
        Directory where intermediate results are saved
    include_figures : bool, default=True
        Whether to check for figure files as well

    Returns:
    --------
    exists : bool
        True if all required intermediate files exist
    """
    tile_output_dir = output_dir / tile_name

    # List of required data files
    required_files = [
        f'{tile_name}_bootstrap_zscores.npy',
        f'{tile_name}_bootstrap_mean.npy',
        f'{tile_name}_bootstrap_std.npy',
        f'{tile_name}_bootstrap_ci_lower.npy',
        f'{tile_name}_bootstrap_ci_upper.npy',
        f'{tile_name}_bootstrap_metadata.json'
    ]

    # Add figure files if requested
    if include_figures:
        required_files.extend([
            f'{tile_name}_bootstrap_enrichment.png',
            f'{tile_name}_bootstrap_uncertainty.png'
        ])

    # Check if all files exist
    all_exist = all((tile_output_dir / fname).exists() for fname in required_files)

    return all_exist


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
    seed=42,
    force_reprocess=False,
    save_figures=True,
    figure_format='png',
    figure_dpi=300,
    max_zscore=50.0,
    min_cells_per_type=5
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
    force_reprocess : bool, default=False
        If True, reprocess even if results exist. If False, skip existing tiles.
    save_figures : bool, default=True
        Whether to generate and save figures
    figure_format : str, default='png'
        Format for saved figures ('png', 'pdf', 'svg')
    figure_dpi : int, default=300
        DPI for saved figures
    max_zscore : float, default=50.0
        Maximum z-score value (clips extreme values)
    min_cells_per_type : int, default=5
        Minimum cells per cell type for valid analysis

    Returns:
    --------
    tile_name : str
        Name of the processed tile
    success : bool
        Whether processing was successful
    skipped : bool
        Whether processing was skipped (results already exist)
    """
    h5ad_path = Path(h5ad_path)
    output_dir = Path(output_dir)
    tile_name = h5ad_path.stem

    print(f"\n{'='*70}")
    print(f"Processing: {tile_name}")
    print(f"{'='*70}")

    # Check if results already exist (including figures if save_figures=True)
    if not force_reprocess and check_intermediate_results_exist(tile_name, output_dir, include_figures=save_figures):
        print(f"⊙ Results already exist for {tile_name} - skipping")
        print(f"  (Set force_reprocess=True to reprocess)")
        return tile_name, True, True  # success=True, skipped=True

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
            seed=seed,
            max_zscore=max_zscore,
            min_cells_per_type=min_cells_per_type
        )

        # Save intermediate results and generate figures
        print(f"\nSaving intermediate results for {tile_name}...")
        saved_files = save_bootstrap_results_with_figures(
            bootstrap_results,
            output_dir=output_dir,
            tile_name=tile_name,
            save_figures=save_figures,
            figure_format=figure_format,
            figure_dpi=figure_dpi
        )

        print(f"✓ Successfully processed {tile_name}")
        return tile_name, True, False  # success=True, skipped=False

    except Exception as e:
        print(f"✗ Failed to process {tile_name}: {e}")
        return tile_name, False, False  # success=False, skipped=False


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
    seed=42,
    save_figures=True,
    figure_format='png',
    figure_dpi=300,
    max_zscore=50.0,
    min_cells_per_type=5
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
    save_figures : bool, default=True
        Whether to generate and save figures for each tile
    figure_format : str, default='png'
        Format for saved figures ('png', 'pdf', 'svg')
    figure_dpi : int, default=300
        DPI for saved figures
    max_zscore : float, default=50.0
        Maximum z-score value (clips extreme values)
    min_cells_per_type : int, default=5
        Minimum cells per cell type for valid analysis

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
    skipped_tiles = []

    # Process each tile
    for i, h5ad_path in enumerate(h5ad_files, 1):
        print(f"\n[{i}/{len(h5ad_files)}] {'-'*60}")

        tile_name, success, skipped = process_single_tile_bootstrap(
            h5ad_path,
            output_dir,
            tile_key=tile_key,
            n_bootstrap=n_bootstrap,
            method=method,
            radius=radius,
            n_neighbors=n_neighbors,
            n_perms=n_perms,
            cluster_key=cluster_key,
            seed=seed + i,  # Different seed for each tile
            save_figures=save_figures,
            figure_format=figure_format,
            figure_dpi=figure_dpi,
            max_zscore=max_zscore,
            min_cells_per_type=min_cells_per_type
        )

        if success:
            successful_tiles.append(tile_name)
            if skipped:
                skipped_tiles.append(tile_name)
        else:
            failed_tiles.append(tile_name)

    # Print summary
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"Total tiles: {len(h5ad_files)}")
    print(f"  - Newly processed: {len(successful_tiles) - len(skipped_tiles)}")
    print(f"  - Skipped (already exist): {len(skipped_tiles)}")
    print(f"  - Failed: {len(failed_tiles)}")
    if skipped_tiles:
        print(f"\nSkipped tiles (set force_reprocess=True to reprocess):")
        for tile in skipped_tiles[:10]:  # Show first 10
            print(f"  • {tile}")
        if len(skipped_tiles) > 10:
            print(f"  ... and {len(skipped_tiles) - 10} more")
    if failed_tiles:
        print(f"\nFailed tiles:")
        for tile in failed_tiles:
            print(f"  • {tile}")

    return {
        'successful_tiles': successful_tiles,
        'failed_tiles': failed_tiles,
        'skipped_tiles': skipped_tiles,
        'n_total': len(h5ad_files),
        'n_success': len(successful_tiles),
        'n_newly_processed': len(successful_tiles) - len(skipped_tiles),
        'n_skipped': len(skipped_tiles),
        'n_failed': len(failed_tiles)
    }


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
    seed=42,
    save_figures=True,
    figure_format='png',
    figure_dpi=300,
    max_zscore=50.0,
    min_cells_per_type=5
):
    """
    Batch process multiple tiles: run bootstrap analysis and save intermediate results.

    This pipeline:
    1. Finds all h5ad files in data_dir
    2. Runs bootstrap-permutation analysis on each tile
    3. Saves intermediate results for each tile (.npy and .json files)
    4. Generates figures for each tile (if save_figures=True)

    For aggregation of results across tiles, use cti_bootstrap_aggregate.py

    Parameters:
    -----------
    data_dir : str or Path
        Directory containing h5ad files
    output_dir : str or Path, default='bootstrap_multiple_analysis'
        Directory to save intermediate results
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
    save_figures : bool, default=True
        Whether to generate and save figures for each tile
    figure_format : str, default='png'
        Format for saved figures ('png', 'pdf', 'svg')
    figure_dpi : int, default=300
        DPI for saved figures
    max_zscore : float, default=50.0
        Maximum z-score value (clips extreme values)
    min_cells_per_type : int, default=5
        Minimum cells per cell type for valid analysis

    Returns:
    --------
    processing_summary : dict
        Dictionary containing processing summary
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "="*70)
    print("BOOTSTRAP MULTIPLE TILES - BATCH PROCESSING")
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
        seed=seed,
        save_figures=save_figures,
        figure_format=figure_format,
        figure_dpi=figure_dpi,
        max_zscore=max_zscore,
        min_cells_per_type=min_cells_per_type
    )

    # Final summary
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE!")
    print("="*70)
    print(f"\nProcessing Summary:")
    print(f"  - Total tiles: {processing_summary['n_total']}")
    print(f"  - Successfully processed: {processing_summary['n_success']}")
    print(f"  - Newly processed: {processing_summary['n_newly_processed']}")
    print(f"  - Skipped (already exist): {processing_summary['n_skipped']}")
    print(f"  - Failed: {processing_summary['n_failed']}")
    print(f"\nIntermediate results saved to: {output_dir}/")
    print(f"\nNext step: Use cti_bootstrap_aggregate.py to aggregate results across tiles")

    return processing_summary


# Example usage
if __name__ == "__main__":
    # Configuration
    data_dir = Path('/mnt/g/GDC-TCGA-LUAD/00a0b174-1eab-446a-ba8c-7c6e3acd7f0c/pred/h5ad')  # Directory containing h5ad files
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
        seed=42,
        save_figures=True,               # Generate figures for each tile
        figure_format='png',             # Figure format
        figure_dpi=300,                  # Figure DPI
        max_zscore=50.0,                # Clip extreme z-scores
        min_cells_per_type=5            # Minimum cells per type
    )

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("""
Batch Processing Complete!

WHAT WAS GENERATED:
- Intermediate bootstrap results for each tile (.npy files)
- Metadata for each tile (.json files)
- Figures for each tile (if save_figures=True)
- Each tile directory contains:
  * {tile_name}_bootstrap_zscores.npy: Full bootstrap z-score array
  * {tile_name}_bootstrap_mean.npy: Mean z-scores
  * {tile_name}_bootstrap_std.npy: Standard deviations
  * {tile_name}_bootstrap_ci_lower.npy: Lower confidence interval
  * {tile_name}_bootstrap_ci_upper.npy: Upper confidence interval
  * {tile_name}_bootstrap_metadata.json: Metadata (cell types, parameters, etc.)
  * {tile_name}_bootstrap_enrichment.png: Enrichment heatmap with confidence intervals
  * {tile_name}_bootstrap_uncertainty.png: Bootstrap uncertainty (standard deviation) heatmap

NEXT STEP - AGGREGATION:
To aggregate results across all tiles and generate final visualizations:

    from ne_bootstrap_aggregated import run_aggregation_pipeline

    results = run_aggregation_pipeline(
        results_dir='bootstrap_multiple_analysis',  # Where your tile results are
        output_dir='bootstrap_multiple_analysis/aggregated_results',
        tile_pattern='tile_*',
        save_matrix_csvs=False,  # Set True to save full matrices
        threshold_zscore=2.0,
        threshold_ci=True
    )

This will:
1. Find all tile directories with intermediate results
2. Load and pool bootstrap iterations from all tiles
3. Compute aggregated statistics (mean, std, CI)
4. Generate visualizations with confidence intervals
5. Identify and save significant interactions

See cti_bootstrap_aggregate.py for more details.
""")