"""
Bootstrap Permutation Analysis for Whole Slide Images (WSI)

This module implements subsample-level bootstrap-permutation testing for spatial
neighborhood enrichment analysis in whole slide images with millions of cells.

Key features:
- Subsample-level bootstrap resampling (each bootstrap = different random subsample)
- Within-bootstrap permutation testing for statistical significance
- Confidence intervals from bootstrap distribution
- Comparison with standard single-subsample analysis
- Robust uncertainty quantification for large datasets

Differences from tiled bootstrap:
- No hierarchical tile structure (or tiles too large for tile-level bootstrap)
- Each bootstrap iteration creates a new random subsample from full WSI
- Accounts for sampling variability in addition to spatial randomness

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
from cti_tiled import (
    load_and_apply_cell_type_colors,
    build_spatial_graph,
    neighborhood_enrichment_analysis,
    compute_centrality_scores,
    summarize_interactions
)

from cti_wsi import (
    subsample_adata,
    validate_subsampling
)

# Import visualization functions from bootstrap_tiled
from cti_bootstrap_tiled import (
    visualize_bootstrap_enrichment,
    visualize_bootstrap_comparison,
    summarize_bootstrap_interactions
)

warnings.filterwarnings('ignore')


def run_single_bootstrap_iteration_wsi(
    adata_full,
    n_cells,
    method='knn',
    radius=50,
    n_neighbors=20,
    n_perms=100,
    cluster_key='cell_type',
    seed=None,
    verbose=False,
    max_zscore=50.0,
    min_cells_per_type=5
):
    """
    Run a single bootstrap iteration for WSI: create random subsample + enrichment analysis.

    Parameters:
    -----------
    adata_full : AnnData
        Full WSI dataset
    n_cells : int
        Number of cells to subsample
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
        - 'n_cells': number of cells in subsample
    """
    if verbose:
        print(f"  Running bootstrap iteration (seed={seed})...")

    # Create random subsample
    adata_sub = subsample_adata(adata_full, n_cells=n_cells, seed=seed)

    # Build spatial graph
    adata_sub = build_spatial_graph(
        adata_sub, method=method, radius=radius, n_neighbors=n_neighbors
    )

    # Run enrichment analysis with permutation
    adata_sub = neighborhood_enrichment_analysis(
        adata_sub,
        cluster_key=cluster_key,
        n_perms=n_perms,
        seed=seed
    )

    # Extract z-scores
    zscore = adata_sub.uns[f'{cluster_key}_nhood_enrichment']['zscore']

    results = {
        'zscore': np.array(zscore),
        'n_cells': adata_sub.n_obs
    }

    return results


def run_bootstrap_permutation_analysis_wsi(
    adata_full,
    n_cells_per_bootstrap=50000,
    n_bootstrap=100,
    method='knn',
    radius=50,
    n_neighbors=20,
    n_perms=100,
    cluster_key='cell_type',
    seed=42,
    n_jobs=1,
    max_zscore=50.0,
    min_cells_per_type=5
):
    """
    Run full bootstrap-permutation analysis for WSI across multiple subsamples.

    Each bootstrap iteration creates a new random subsample from the full WSI.
    This accounts for both spatial randomness (via permutation) and sampling
    variability (via bootstrap).

    Parameters:
    -----------
    adata_full : AnnData
        Full WSI AnnData object
    n_cells_per_bootstrap : int, default=50000
        Number of cells to use per bootstrap subsample
    n_bootstrap : int, default=100
        Number of bootstrap iterations (random subsamples)
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
    max_zscore : float, default=50.0
        Maximum z-score value (clips extreme values)
    min_cells_per_type : int, default=5
        Minimum cells per cell type for valid analysis

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
    print("BOOTSTRAP-PERMUTATION ENRICHMENT ANALYSIS (WSI)")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  - Total cells in WSI: {adata_full.n_obs:,}")
    print(f"  - Cells per bootstrap subsample: {n_cells_per_bootstrap:,}")
    print(f"  - Number of bootstrap iterations: {n_bootstrap}")
    print(f"  - Permutations per bootstrap: {n_perms}")
    print(f"  - Spatial graph method: {method}")
    if method == 'knn':
        print(f"  - Number of neighbors (KNN): {n_neighbors}")
    else:
        print(f"  - Spatial radius: {radius} pixels")

    # Get cell types
    cell_types = adata_full.obs[cluster_key].cat.categories.tolist()
    n_celltypes = len(cell_types)

    print(f"\n  - Cell types: {cell_types}")
    print(f"\nRunning {n_bootstrap} bootstrap iterations (each with different random subsample)...")

    # Storage for bootstrap results
    zscores_list = []

    # Run bootstrap iterations with progress bar
    for i in tqdm(range(n_bootstrap), desc="Bootstrap iterations"):
        boot_seed = seed + i if seed is not None else None

        try:
            results = run_single_bootstrap_iteration_wsi(
                adata_full,
                n_cells=n_cells_per_bootstrap,
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

    # Calculate statistics
    mean_zscore = zscores_array.mean(axis=0)
    std_zscore = zscores_array.std(axis=0)

    # Calculate 95% confidence intervals
    ci_lower = np.percentile(zscores_array, 2.5, axis=0)
    ci_upper = np.percentile(zscores_array, 97.5, axis=0)

    # Prepare results
    bootstrap_results = {
        'zscores': zscores_array,
        'mean_zscore': mean_zscore,
        'std_zscore': std_zscore,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'cell_types': cell_types,
        'n_bootstrap': len(zscores_list),
        'parameters': {
            'n_bootstrap': len(zscores_list),
            'n_cells_per_bootstrap': n_cells_per_bootstrap,
            'n_perms': n_perms,
            'method': method,
            'radius': radius,
            'n_neighbors': n_neighbors,
            'seed': seed,
            'total_cells': adata_full.n_obs,
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

    return bootstrap_results


def compare_with_standard_single_subsample(
    adata_full,
    bootstrap_results,
    n_cells,
    method='knn',
    radius=50,
    n_neighbors=20,
    n_perms=1000,
    cluster_key='cell_type',
    seed=42
):
    """
    Run standard permutation analysis on a single subsample for comparison.

    Parameters:
    -----------
    adata_full : AnnData
        Full WSI dataset
    bootstrap_results : dict
        Results from bootstrap analysis
    n_cells : int
        Number of cells for standard subsample
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
    seed : int
        Random seed

    Returns:
    --------
    adata_std : AnnData
        AnnData with standard enrichment results
    comparison : dict
        Comparison metrics between bootstrap and standard methods
    """
    print("\n" + "=" * 70)
    print("STANDARD PERMUTATION ANALYSIS (single subsample, for comparison)")
    print("=" * 70)

    # Create single subsample
    print(f"  - Creating subsample of {n_cells:,} cells (seed={seed})...")
    adata_std = subsample_adata(adata_full, n_cells=n_cells, seed=seed)

    # Build graph and run enrichment
    adata_std = build_spatial_graph(adata_std, method=method, radius=radius, n_neighbors=n_neighbors)
    adata_std = neighborhood_enrichment_analysis(
        adata_std,
        cluster_key=cluster_key,
        n_perms=n_perms,
        seed=seed
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
        print("  ✓ EXCELLENT: Bootstrap and single-subsample results highly consistent")
    elif comparison['correlation'] > 0.85:
        print("  ✓ GOOD: Bootstrap and single-subsample results reasonably consistent")
    elif comparison['correlation'] > 0.70:
        print("  ⚠ FAIR: Some discrepancy between methods")
    else:
        print("  ✗ WARNING: Substantial discrepancy - sampling variability is high!")

    return adata_std, comparison


def run_bootstrap_wsi_pipeline(
    adata_path,
    output_dir='bootstrap_spatial_analysis_wsi',
    n_cells_per_bootstrap=50000,
    n_cells_visualization=10000,
    n_bootstrap=100,
    method='knn',
    radius=50,
    n_neighbors=20,
    n_perms_bootstrap=100,
    n_perms_standard=1000,
    cluster_key='cell_type',
    save_adata=False,
    seed=42,
    max_zscore=50.0,
    min_cells_per_type=5
):
    """
    Run complete bootstrap-permutation analysis pipeline for whole slide images (WSI).

    This pipeline:
    1. Loads full WSI data
    2. Runs subsample-level bootstrap-permutation analysis
    3. Runs standard permutation on single subsample for comparison
    4. Validates that subsamples represent the full dataset
    5. Generates visualizations with confidence intervals
    6. Summarizes significant interactions

    Parameters:
    -----------
    adata_path : str
        Path to h5ad file from WSI
    output_dir : str, default='bootstrap_spatial_analysis_wsi'
        Directory to save results
    n_cells_per_bootstrap : int, default=50000
        Number of cells per bootstrap subsample
    n_cells_visualization : int, default=10000
        Number of cells for spatial distribution plots
    n_bootstrap : int, default=100
        Number of bootstrap iterations
    method : str, default='knn'
        Method for spatial graph: 'knn' or 'radius'
    radius : float, default=50
        Radius for spatial graph (pixels, used if method='radius')
    n_neighbors : int, default=6
        Number of neighbors for KNN (used if method='knn')
    n_perms_bootstrap : int, default=100
        Permutations per bootstrap iteration (can be lower)
    n_perms_standard : int, default=1000
        Permutations for standard analysis
    cluster_key : str, default='cell_type'
        Key for cell type labels
    save_adata : bool, default=False
        Whether to save processed data
    seed : int, default=42
        Random seed for reproducibility
    max_zscore : float, default=50.0
        Maximum z-score value (clips extreme values)
    min_cells_per_type : int, default=5
        Minimum cells per cell type for valid analysis

    Returns:
    --------
    results : dict
        Dictionary containing all analysis results
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("BOOTSTRAP-PERMUTATION SPATIAL ANALYSIS PIPELINE (WSI)")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")

    # Load full WSI data
    print(f"\nLoading full WSI data from: {adata_path}")
    adata_full = sc.read_h5ad(adata_path)
    print(f"  - Loaded {adata_full.n_obs:,} cells from WSI")

    # Apply cell type colors
    load_and_apply_cell_type_colors(adata_full, celltype_key=cluster_key)

    # Step 1: Bootstrap-Permutation Analysis
    print("\n" + "=" * 70)
    print("STEP 1: BOOTSTRAP-PERMUTATION ANALYSIS")
    print("=" * 70)

    bootstrap_results = run_bootstrap_permutation_analysis_wsi(
        adata_full,
        n_cells_per_bootstrap=n_cells_per_bootstrap,
        n_bootstrap=n_bootstrap,
        method=method,
        radius=radius,
        n_neighbors=n_neighbors,
        n_perms=n_perms_bootstrap,
        cluster_key=cluster_key,
        seed=seed,
        max_zscore=max_zscore,
        min_cells_per_type=min_cells_per_type
    )

    # Step 2: Validate Subsampling (use first bootstrap subsample)
    print("\n" + "=" * 70)
    print("STEP 2: VALIDATING SUBSAMPLE REPRESENTATIVENESS")
    print("=" * 70)

    adata_validation = subsample_adata(adata_full, n_cells=n_cells_per_bootstrap, seed=seed)
    validate_subsampling(adata_full, adata_validation, output_dir, cluster_key=cluster_key)

    # Step 3: Standard Permutation Analysis (for comparison)
    print("\n" + "=" * 70)
    print("STEP 3: STANDARD PERMUTATION ANALYSIS")
    print("=" * 70)

    adata_standard, comparison = compare_with_standard_single_subsample(
        adata_full,
        bootstrap_results,
        n_cells=n_cells_per_bootstrap,
        method=method,
        radius=radius,
        n_neighbors=n_neighbors,
        n_perms=n_perms_standard,
        cluster_key=cluster_key,
        seed=seed
    )

    # Step 4: Visualizations
    print("\n" + "=" * 70)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("=" * 70)

    # Bootstrap enrichment with CI
    visualize_bootstrap_enrichment(
        bootstrap_results,
        cluster_key=cluster_key,
        save_path=output_dir / 'bootstrap_enrichment_with_ci_wsi.png',
        n_bootstrap=n_bootstrap,
        n_perms_bootstrap=n_perms_bootstrap,
        n_perms_standard=n_perms_standard,
        n_neighbors=n_neighbors,
        max_zscore=max_zscore,
        min_cells_per_type=min_cells_per_type
    )

    # Bootstrap uncertainty plot
    visualize_bootstrap_comparison(
        bootstrap_results,
        comparison['standard_zscore'],
        bootstrap_results['cell_types'],
        save_path=output_dir / 'bootstrap_uncertainty_wsi.png',
        n_bootstrap=n_bootstrap,
        n_perms_bootstrap=n_perms_bootstrap,
        n_perms_standard=n_perms_standard,
        n_neighbors=n_neighbors,
        max_zscore=max_zscore,
        min_cells_per_type=min_cells_per_type
    )

    # Spatial distribution (using separate visualization subsample)
    print(f"\nGenerating spatial distribution plot ({n_cells_visualization:,} cells)...")
    from cti_tiled import visualize_spatial_distribution
    adata_vis = subsample_adata(adata_full, n_cells=n_cells_visualization, seed=seed)
    if f'{cluster_key}_colors' in adata_full.uns:
        adata_vis.uns[f'{cluster_key}_colors'] = adata_full.uns[f'{cluster_key}_colors']

    visualize_spatial_distribution(
        adata_vis,
        cluster_key=cluster_key,
        save_path=output_dir / 'spatial_distribution_wsi.png',
        size=1,
        figsize=(16, 14)
    )

    # Step 5: Summarize Interactions
    print("\n" + "=" * 70)
    print("STEP 5: SUMMARIZING INTERACTIONS")
    print("=" * 70)

    # Bootstrap interactions
    bootstrap_interactions = summarize_bootstrap_interactions(
        bootstrap_results,
        threshold_zscore=2.0,
        threshold_ci=True
    )
    bootstrap_interactions.to_csv(
        output_dir / 'bootstrap_significant_interactions_wsi.csv',
        index=False
    )
    print(f"\n  - Saved bootstrap interactions to: {output_dir / 'bootstrap_significant_interactions_wsi.csv'}")

    # Standard interactions (for comparison)
    print("\n" + "=" * 70)
    print("STANDARD PERMUTATION INTERACTIONS (for comparison)")
    print("=" * 70)
    standard_interactions = summarize_interactions(adata_standard, cluster_key=cluster_key)
    standard_interactions.to_csv(
        output_dir / 'standard_significant_interactions_wsi.csv',
        index=False
    )
    print(f"\n  - Saved standard interactions to: {output_dir / 'standard_significant_interactions_wsi.csv'}")

    # Step 6: Save Results
    print("\n" + "=" * 70)
    print("STEP 6: SAVING RESULTS")
    print("=" * 70)

    # Save bootstrap statistics
    cell_types = bootstrap_results['cell_types']

    # Mean z-scores
    mean_df = pd.DataFrame(
        bootstrap_results['mean_zscore'],
        index=cell_types,
        columns=cell_types
    )
    mean_df.to_csv(output_dir / 'bootstrap_mean_zscore_wsi.csv')

    # Standard deviations
    std_df = pd.DataFrame(
        bootstrap_results['std_zscore'],
        index=cell_types,
        columns=cell_types
    )
    std_df.to_csv(output_dir / 'bootstrap_std_zscore_wsi.csv')

    # Confidence intervals
    ci_lower_df = pd.DataFrame(
        bootstrap_results['ci_lower'],
        index=cell_types,
        columns=cell_types
    )
    ci_lower_df.to_csv(output_dir / 'bootstrap_ci_lower_wsi.csv')

    ci_upper_df = pd.DataFrame(
        bootstrap_results['ci_upper'],
        index=cell_types,
        columns=cell_types
    )
    ci_upper_df.to_csv(output_dir / 'bootstrap_ci_upper_wsi.csv')

    # Comparison metrics
    comparison_df = pd.DataFrame([{
        'mean_absolute_difference': comparison['mean_absolute_difference'],
        'max_absolute_difference': comparison['max_absolute_difference'],
        'correlation': comparison['correlation']
    }])
    comparison_df.to_csv(output_dir / 'bootstrap_vs_standard_comparison_wsi.csv', index=False)

    print(f"\n  - Saved all results to: {output_dir}/")

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
    print(f"\nDataset Summary:")
    print(f"  - Total WSI cells: {adata_full.n_obs:,}")
    print(f"  - Cells per bootstrap subsample: {n_cells_per_bootstrap:,}")
    print(f"  - Bootstrap iterations: {bootstrap_results['n_bootstrap']}")
    print(f"  - Permutations per bootstrap: {n_perms_bootstrap}")

    print(f"\nBootstrap Uncertainty:")
    print(f"  - Mean std of z-scores: {bootstrap_results['std_zscore'].mean():.3f}")
    print(f"  - Mean CI width: {(bootstrap_results['ci_upper'] - bootstrap_results['ci_lower']).mean():.3f}")

    print(f"\nComparison with Standard Single-Subsample:")
    print(f"  - Correlation: {comparison['correlation']:.3f}")
    print(f"  - Mean absolute difference: {comparison['mean_absolute_difference']:.3f}")

    print(f"\nSignificant Interactions:")
    print(f"  - Bootstrap method: {len(bootstrap_interactions)}")
    print(f"  - Standard method: {len(standard_interactions)}")

    print(f"\nAll results saved to: {output_dir}/")

    return results


# Example usage
if __name__ == "__main__":
    # Configuration
    adata_path = '../TCGA-MN-A4N4-01Z-00-DX2.9550732D-8FB1-43D9-B094-7C0CD310E9C0.h5ad'  # Replace with your WSI file
    output_dir = 'bootstrap_spatial_analysis_wsi'

    # Run bootstrap pipeline for WSI
    results = run_bootstrap_wsi_pipeline(
        adata_path=adata_path,
        output_dir=output_dir,
        n_cells_per_bootstrap=50000,    # Cells per bootstrap subsample
        n_cells_visualization=20000,     # Cells for visualization
        n_bootstrap=100,                 # Number of bootstrap iterations
        method='knn',
        n_neighbors=20,
        max_zscore=50.0,
        min_cells_per_type=5,
        radius=50,                       # Adjust based on tissue/magnification
        n_perms_bootstrap=100,           # Permutations per bootstrap (can be lower)
        n_perms_standard=1000,           # Permutations for standard analysis
        cluster_key='cell_type',         # Adjust to your cell type column
        save_adata=False,                # Set to True to save processed data
        seed=42
    )

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
Bootstrap-Permutation Results for WSI:

1. SUBSAMPLE-LEVEL BOOTSTRAP:
   - Each bootstrap iteration creates a NEW random subsample from full WSI
   - Accounts for sampling variability + spatial randomness
   - CI width reflects consistency across different tissue regions

2. MEAN Z-SCORES: Average enrichment across all subsamples
   - Positive = attraction between cell types
   - Negative = avoidance between cell types
   - More robust than single subsample estimate

3. CONFIDENCE INTERVALS (CI):
   - Narrow CI = pattern consistent across tissue (robust finding)
   - Wide CI = pattern varies by region (region-specific or unstable)
   - If CI excludes zero → strong evidence for interaction

4. COMPARISON WITH STANDARD SINGLE-SUBSAMPLE:
   - High correlation (>0.95) → single subsample is representative
   - Low correlation (<0.85) → sampling variability is high
   - Bootstrap provides more reliable estimates for heterogeneous WSI

5. SIGNIFICANCE CRITERIA:
   - Bootstrap: Mean |Z| > 2 AND CI excludes zero
   - More stringent than standard single-subsample analysis
   - Accounts for both spatial randomness AND sampling uncertainty

6. UNCERTAINTY (STD DEV):
   - Shows which interactions vary most across tissue regions
   - High std = interaction strength depends on sampled region
   - Low std = interaction is consistent across entire WSI

WHEN TO USE BOOTSTRAP FOR WSI:
✓ Always recommended for heterogeneous tissues
✓ Essential when single-subsample results need confidence bounds
✓ Critical when making statistical claims about entire WSI

VALIDATION:
- Check validation_celltype_distribution_wsi.png for subsample quality
- If subsampling is poor, increase n_cells_per_bootstrap
- If bootstrap CI are very wide, tissue may be highly heterogeneous
""")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print(f"""
1. Review validation plot: {output_dir}/validation_celltype_distribution_wsi.png
   - Ensure subsamples represent full WSI well

2. Check bootstrap enrichment: {output_dir}/bootstrap_enrichment_with_ci_wsi.png
   - Look for interactions where CI excludes zero

3. Compare methods: {output_dir}/bootstrap_vs_standard_comparison_wsi.png
   - Left: Single subsample result (what you'd get from standard analysis)
   - Middle: Bootstrap mean (more robust estimate)
   - Right: Bootstrap uncertainty (shows regional variability)

4. Significant interactions:
   - {output_dir}/bootstrap_significant_interactions_wsi.csv (with CI)
   - {output_dir}/standard_significant_interactions_wsi.csv (for comparison)

5. Adjust parameters if needed:
   - Increase n_cells_per_bootstrap if subsamples are not representative
   - Increase n_bootstrap (>100) for tighter confidence intervals
   - Adjust radius based on your tissue characteristics
""")