import os
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from neighborhood_enrichment_tiled import (
    load_and_apply_cell_type_colors,
    build_spatial_graph,
    neighborhood_enrichment_analysis,
    compute_centrality_scores,
    visualize_enrichment,
    visualize_spatial_distribution,
    summarize_interactions
)


def validate_subsampling(adata_full, adata_sub, output_dir, cluster_key='cell_type'):
    """
    Validate that subsampled data represents the full dataset well.

    Parameters:
    -----------
    adata_full : AnnData
        Full dataset
    adata_sub : AnnData
        Subsampled dataset
    output_dir : Path
        Directory to save validation plots
    cluster_key : str, default='cell_type'
        Key for cell type labels

    Returns:
    --------
    validation_results : dict
        Dictionary with validation metrics
    """
    print("\n" + "=" * 70)
    print("VALIDATION: Checking if subsample represents full WSI")
    print("=" * 70)

    validation_results = {}

    # 1. Compare cell type proportions
    print("\n1. Cell Type Distribution Comparison:")
    full_counts = adata_full.obs[cluster_key].value_counts(normalize=True).sort_index()
    sub_counts = adata_sub.obs[cluster_key].value_counts(normalize=True).sort_index()

    comparison_df = pd.DataFrame({
        'Full_WSI_%': full_counts * 100,
        'Subsample_%': sub_counts * 100
    })
    comparison_df['Difference_%'] = comparison_df['Subsample_%'] - comparison_df['Full_WSI_%']
    comparison_df['Abs_Difference_%'] = comparison_df['Difference_%'].abs()

    print(comparison_df.round(2))

    max_diff = comparison_df['Abs_Difference_%'].max()
    mean_diff = comparison_df['Abs_Difference_%'].mean()

    validation_results['max_celltype_diff'] = max_diff
    validation_results['mean_celltype_diff'] = mean_diff

    print(f"\n  Max difference: {max_diff:.2f}%")
    print(f"  Mean difference: {mean_diff:.2f}%")

    if mean_diff < 1.0:
        print("  ✓ EXCELLENT: Cell type proportions well preserved")
    elif mean_diff < 2.0:
        print("  ✓ GOOD: Cell type proportions reasonably preserved")
    elif mean_diff < 5.0:
        print("  ⚠ FAIR: Some deviation in cell type proportions")
    else:
        print("  ✗ WARNING: Significant deviation in cell type proportions")

    # Save comparison table
    comparison_df.to_csv(output_dir / 'validation_celltype_proportions.csv')

    # 2. Visualize distribution comparison
    print("\n2. Creating distribution comparison plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    x_pos = np.arange(len(comparison_df))
    width = 0.35

    axes[0].bar(x_pos - width/2, comparison_df['Full_WSI_%'], width,
                label='Full WSI', alpha=0.8, color='steelblue')
    axes[0].bar(x_pos + width/2, comparison_df['Subsample_%'], width,
                label='Subsample', alpha=0.8, color='coral')
    axes[0].set_xlabel('Cell Type', fontsize=11)
    axes[0].set_ylabel('Percentage (%)', fontsize=11)
    axes[0].set_title('Cell Type Distribution: Full vs Subsample', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(comparison_df.index, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Difference plot
    colors = ['red' if x > 2 else 'orange' if x > 1 else 'green'
              for x in comparison_df['Abs_Difference_%']]
    axes[1].bar(x_pos, comparison_df['Difference_%'], color=colors, alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[1].axhline(y=2, color='orange', linestyle='--', linewidth=0.8, alpha=0.5, label='±2% threshold')
    axes[1].axhline(y=-2, color='orange', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel('Cell Type', fontsize=11)
    axes[1].set_ylabel('Difference (Subsample - Full) %', fontsize=11)
    axes[1].set_title('Cell Type Proportion Differences', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(comparison_df.index, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'validation_celltype_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  - Saved: {output_dir / 'validation_celltype_distribution.png'}")

    # 3. Spatial coverage check
    print("\n3. Spatial Coverage Analysis:")

    # Get spatial coordinates
    coords_full = adata_full.obsm['spatial']
    coords_sub = adata_sub.obsm['spatial']

    # Check spatial bounds
    full_x_range = coords_full[:, 0].max() - coords_full[:, 0].min()
    full_y_range = coords_full[:, 1].max() - coords_full[:, 1].min()
    sub_x_range = coords_sub[:, 0].max() - coords_sub[:, 0].min()
    sub_y_range = coords_sub[:, 1].max() - coords_sub[:, 1].min()

    x_coverage = (sub_x_range / full_x_range) * 100
    y_coverage = (sub_y_range / full_y_range) * 100

    validation_results['x_coverage_%'] = x_coverage
    validation_results['y_coverage_%'] = y_coverage

    print(f"  X-axis coverage: {x_coverage:.1f}%")
    print(f"  Y-axis coverage: {y_coverage:.1f}%")

    if x_coverage > 95 and y_coverage > 95:
        print("  ✓ EXCELLENT: Subsample spans the full tissue area")
    elif x_coverage > 85 and y_coverage > 85:
        print("  ✓ GOOD: Subsample covers most of the tissue area")
    else:
        print("  ⚠ WARNING: Subsample may not cover full tissue extent")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Cell type distribution:")
    print(f"  - Mean absolute difference: {mean_diff:.2f}%")
    print(f"  - Max absolute difference: {max_diff:.2f}%")
    print(f"\nSpatial coverage:")
    print(f"  - X-axis: {x_coverage:.1f}%")
    print(f"  - Y-axis: {y_coverage:.1f}%")
    print(f"\n✓ Check validation plots in: {output_dir}/")

    # Save validation metrics
    validation_df = pd.DataFrame([validation_results])
    validation_df.to_csv(output_dir / 'validation_metrics.csv', index=False)

    return validation_results


def compare_multiple_subsamples(adata_full, n_cells, n_iterations=5, cluster_key='cell_type'):
    """
    Compare enrichment results across multiple random subsamples to check robustness.

    Parameters:
    -----------
    adata_full : AnnData
        Full dataset
    n_cells : int
        Number of cells per subsample
    n_iterations : int, default=5
        Number of different random subsamples to test
    cluster_key : str, default='cell_type'
        Key for cell type labels

    Returns:
    --------
    results : dict
        Enrichment z-scores for each iteration
    """
    print("\n" + "=" * 70)
    print(f"ROBUSTNESS CHECK: Testing {n_iterations} different random subsamples")
    print("=" * 70)

    results = []

    for i in range(n_iterations):
        seed = 42 + i
        print(f"\nIteration {i+1}/{n_iterations} (seed={seed})...")

        # Subsample
        adata_sub = subsample_adata(adata_full, n_cells=n_cells, seed=seed)

        # Build graph and run enrichment
        adata_sub = build_spatial_graph(adata_sub, method='radius', radius=50)
        adata_sub = neighborhood_enrichment_analysis(adata_sub, n_perms=100, seed=seed)

        # Extract z-scores
        zscore = adata_sub.uns[f'{cluster_key}_nhood_enrichment']['zscore']
        results.append(zscore)

    # Calculate mean and std across iterations
    zscores_stack = np.stack([np.array(z) for z in results])
    mean_zscore = zscores_stack.mean(axis=0)
    std_zscore = zscores_stack.std(axis=0)

    print("\n" + "=" * 70)
    print("ROBUSTNESS RESULTS")
    print("=" * 70)
    print(f"Average std of z-scores across iterations: {std_zscore.mean():.3f}")
    print(f"Max std of z-scores: {std_zscore.max():.3f}")

    if std_zscore.mean() < 0.5:
        print("✓ EXCELLENT: Results very consistent across subsamples")
    elif std_zscore.mean() < 1.0:
        print("✓ GOOD: Results reasonably consistent across subsamples")
    else:
        print("⚠ WARNING: Results vary across subsamples - consider larger subsample size")

    return {
        'mean_zscore': mean_zscore,
        'std_zscore': std_zscore,
        'individual_zscores': results
    }


def subsample_adata(adata, n_cells, seed=42):
    """
    Randomly subsample cells from AnnData object.

    Parameters:
    -----------
    adata : AnnData
        AnnData object to subsample
    n_cells : int
        Number of cells to keep after subsampling
    seed : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    adata_sub : AnnData
        Subsampled AnnData object
    """
    if adata.n_obs <= n_cells:
        print(f"  - Dataset has {adata.n_obs} cells (<= {n_cells}), keeping all cells")
        return adata.copy()

    print(f"  - Subsampling from {adata.n_obs} to {n_cells} cells")

    # Set random seed
    np.random.seed(seed)

    # Randomly select cell indices
    indices = np.random.choice(adata.n_obs, size=n_cells, replace=False)
    indices = np.sort(indices)  # Sort to maintain some spatial structure

    # Subsample
    adata_sub = adata[indices].copy()

    print(f"  - Subsampled to {adata_sub.n_obs} cells")

    return adata_sub


def run_wsi_spatial_analysis_pipeline(
    adata_path,
    output_dir='spatial_analysis_results_wsi',
    n_cells_analysis=50000,
    n_cells_visualization=10000,
    radius=50,
    n_perms=1000,
    save_adata=False,
    skip_cooccurrence=True,
    seed=42
):
    """
    Run spatial analysis pipeline optimized for whole slide images (WSI) with large cell counts.

    This pipeline handles datasets with 1M+ cells by:
    1. Subsampling to a manageable size for analysis
    2. Further subsampling for visualization
    3. Reusing validated functions from neighborhood_enrichment_tiled.py

    Parameters:
    -----------
    adata_path : str
        Path to h5ad file from WSI
    output_dir : str, default='spatial_analysis_results_wsi'
        Directory to save results
    n_cells_analysis : int, default=50000
        Number of cells to use for spatial analysis (graph, enrichment, etc.)
        Larger values give more representative results but take longer
    n_cells_visualization : int, default=9999
        Number of cells to display in spatial distribution plot (1000-9999 recommended)
    radius : float, default=50
        Radius for spatial graph (in pixels)
    n_perms : int, default=1000
        Number of permutations for enrichment analysis
    save_adata : bool, default=False
        Whether to save the processed AnnData object with analysis results
    skip_cooccurrence : bool, default=True
        Whether to skip co-occurrence analysis (recommended for large datasets)
    seed : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    adata_analysis : AnnData
        AnnData object with all analysis results (subsampled)
    """

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("WHOLE SLIDE IMAGE (WSI) SPATIAL ANALYSIS PIPELINE")
    print("=" * 70)

    # Load full dataset
    print(f"\nLoading data from: {adata_path}")
    adata_full = sc.read_h5ad(adata_path)
    print(f"  - Loaded {adata_full.n_obs:,} cells from WSI")

    # Apply cell type colors
    load_and_apply_cell_type_colors(adata_full)

    # ==================== STEP 1: Subsample for Analysis ====================
    print("\n" + "=" * 70)
    print("STEP 1: SUBSAMPLING FOR ANALYSIS")
    print("=" * 70)

    adata_analysis = subsample_adata(adata_full, n_cells=n_cells_analysis, seed=seed)

    # Validate subsampling
    validate_subsampling(adata_full, adata_analysis, output_dir)

    # ==================== STEP 2: Build Spatial Graph ====================
    print("\n" + "=" * 70)
    print("STEP 2: BUILDING SPATIAL GRAPH")
    print("=" * 70)

    adata_analysis = build_spatial_graph(adata_analysis, method='radius', radius=radius)

    # ==================== STEP 3: Neighborhood Enrichment ====================
    print("\n" + "=" * 70)
    print("STEP 3: NEIGHBORHOOD ENRICHMENT ANALYSIS")
    print("=" * 70)

    adata_analysis = neighborhood_enrichment_analysis(adata_analysis, n_perms=n_perms, seed=seed)

    # ==================== STEP 4: Co-occurrence (Optional) ====================
    if not skip_cooccurrence:
        print("\n" + "=" * 70)
        print("STEP 4: CO-OCCURRENCE ANALYSIS")
        print("=" * 70)
        print("  - Note: This step is memory-intensive for large datasets")
        print(f"  - Estimated RAM required: ~{(adata_analysis.n_obs**2 * 4 / 1e9):.1f} GB")
        # User can implement this if they have sufficient memory
        print("  - Skipped by default. Set skip_cooccurrence=False to run.")
    else:
        print("\n" + "=" * 70)
        print("STEP 4: CO-OCCURRENCE ANALYSIS - SKIPPED")
        print("=" * 70)
        print("  - Skipped for large WSI datasets (memory constraints)")

    # ==================== STEP 5: Centrality Scores ====================
    print("\n" + "=" * 70)
    print("STEP 5: COMPUTING CENTRALITY SCORES")
    print("=" * 70)

    adata_analysis = compute_centrality_scores(adata_analysis)

    # ==================== STEP 6: Visualizations ====================
    print("\n" + "=" * 70)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("=" * 70)

    # Subsample for visualization
    print(f"\nSubsampling for visualization (target: {n_cells_visualization} cells):")
    adata_vis = subsample_adata(adata_full, n_cells=n_cells_visualization, seed=seed)

    # Copy over cell type colors
    if 'cell_type_colors' in adata_analysis.uns:
        adata_vis.uns['cell_type_colors'] = adata_analysis.uns['cell_type_colors']

    # Spatial distribution (using visualization subsample)
    print(f"\nGenerating spatial distribution plot ({adata_vis.n_obs:,} cells)...")
    visualize_spatial_distribution(
        adata_vis,
        save_path=output_dir / 'spatial_distribution.png',
        size=1,  # Smaller point size for better visualization with many cells
        figsize=(16, 14)
    )

    # Enrichment heatmap (using analysis results)
    print("\nGenerating neighborhood enrichment heatmap...")
    visualize_enrichment(
        adata_analysis,
        save_path=output_dir / 'neighborhood_enrichment.png'
    )

    # ==================== STEP 7: Summarize Interactions ====================
    print("\n" + "=" * 70)
    print("STEP 7: SUMMARIZING CELL-CELL INTERACTIONS")
    print("=" * 70)

    interactions_df = summarize_interactions(adata_analysis)
    interactions_df.to_csv(output_dir / 'significant_interactions.csv', index=False)
    print(f"\n  - Saved interactions to: {output_dir / 'significant_interactions.csv'}")

    # ==================== STEP 8: Save Results ====================
    if save_adata:
        print("\n" + "=" * 70)
        print("STEP 8: SAVING PROCESSED DATA")
        print("=" * 70)

        output_adata_path = output_dir / 'adata_wsi_with_spatial_analysis.h5ad'
        adata_analysis.write(output_adata_path)
        print(f"  - Saved processed AnnData to: {output_adata_path}")
        print(f"  - Contains {adata_analysis.n_obs:,} cells (subsampled for analysis)")
    else:
        print(f"\n  - Processed AnnData not saved (set save_adata=True to save)")

    # ==================== Summary ====================
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nDataset Summary:")
    print(f"  - Original WSI cells: {adata_full.n_obs:,}")
    print(f"  - Cells used for analysis: {adata_analysis.n_obs:,}")
    print(f"  - Cells shown in visualization: {adata_vis.n_obs:,}")
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - spatial_distribution.png")
    print(f"  - neighborhood_enrichment.png")
    print(f"  - significant_interactions.csv")
    if save_adata:
        print(f"  - adata_wsi_with_spatial_analysis.h5ad")

    return adata_analysis


# Example usage
if __name__ == "__main__":
    # WSI file path
    # current_dir = os.path.dirname(__file__)
    # parent_dir = os.path.dirname(current_dir)
    # adata_path = os.path.join(parent_dir, "TCGA-MN-A4N4-01Z-00-DX2.9550732D-8FB1-43D9-B094-7C0CD310E9C0.h5ad")
    adata_path = '../TCGA-MN-A4N4-01Z-00-DX2.9550732D-8FB1-43D9-B094-7C0CD310E9C0.h5ad'

    # Run WSI-optimized pipeline
    adata = run_wsi_spatial_analysis_pipeline(
        adata_path=adata_path,
        output_dir='spatial_analysis_results_wsi',
        n_cells_analysis=50000,        # Number of cells for analysis (adjust based on memory)
        n_cells_visualization=20000,    # Number of cells to display (1000-10000 recommended)
        radius=50,                     # Adjust based on your tissue/magnification
        n_perms=1000,                  # Permutations for statistical testing
        save_adata=False,              # Set to True to save the subsampled h5ad file
        skip_cooccurrence=True,        # Keep True for large datasets
        seed=42                        # Random seed for reproducibility
    )

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\nYou can now:")
    print("  1. Check 'spatial_analysis_results_wsi/' folder for results")
    print("  2. Review validation plot: validation_celltype_distribution.png")
    print("  3. Adjust n_cells_analysis (50k default) for more/less detail")
    print("  4. Adjust n_cells_visualization (20k default) for plot density")
    print("  5. Modify radius parameter based on tissue characteristics")
    print("  6. Set save_adata=True to save processed data for further analysis")

    # Optional: Robustness check with multiple subsamples
    # Uncomment below to test consistency across different random subsamples

    print("\n" + "=" * 70)
    print("OPTIONAL: ROBUSTNESS TESTING")
    print("=" * 70)
    print("\nTesting consistency across multiple random subsamples...")

    robustness_results = compare_multiple_subsamples(
        adata_full=sc.read_h5ad(adata_path),
        n_cells=50000,
        n_iterations=5
    )

    print("\nIf results are consistent, your neighborhood enrichment")
    print("analysis is robust and representative of the full WSI.")
