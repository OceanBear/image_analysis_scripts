"""
Neighborhood Enrichment Analysis for Multiple Tiled Images

This script processes multiple tiled images from a directory and performs
spatial neighborhood enrichment analysis on each tile individually.

Features:
- Batch processing of multiple h5ad files
- Individual analysis results for each tile
- Summary statistics across all tiles
- Consolidated results and visualizations

Author: Generated with Claude Code
Date: 2025-10-22
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

# Import functions from ne_tiled.py
from ne_tiled import (
    load_and_apply_cell_type_colors,
    build_spatial_graph,
    neighborhood_enrichment_analysis,
    compute_centrality_scores,
    visualize_enrichment,
    visualize_spatial_distribution,
    summarize_interactions
)

warnings.filterwarnings('ignore')


def find_h5ad_files(directory, pattern='*.h5ad'):
    """
    Find all h5ad files in a directory.

    Parameters:
    -----------
    directory : str or Path
        Directory to search for h5ad files
    pattern : str, default='*.h5ad'
        File pattern to match

    Returns:
    --------
    h5ad_files : list
        List of Path objects for h5ad files
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    h5ad_files = sorted(directory.glob(pattern))

    if len(h5ad_files) == 0:
        raise FileNotFoundError(f"No h5ad files found in {directory}")

    print(f"Found {len(h5ad_files)} h5ad files in {directory}")
    for i, file in enumerate(h5ad_files[:10]):  # Show first 10
        print(f"  {i+1}. {file.name}")
    if len(h5ad_files) > 10:
        print(f"  ... and {len(h5ad_files) - 10} more files")

    return h5ad_files


def is_tile_processed(output_dir, tile_name):
    """
    Check if a tile has been fully processed by verifying all output files exist.

    Parameters:
    -----------
    output_dir : str or Path
        Directory where tile results should be saved
    tile_name : str
        Name of the tile (used as prefix for output files)

    Returns:
    --------
    is_complete : bool
        True if all expected output files exist, False otherwise
    """
    output_dir = Path(output_dir)

    # Check for all expected output files with tile name prefix
    expected_files = [
        f'{tile_name}_spatial_distribution.png',
        f'{tile_name}_neighborhood_enrichment.png',
        f'{tile_name}_significant_interactions.csv'
    ]

    for filename in expected_files:
        if not (output_dir / filename).exists():
            return False

    return True


def process_single_tile(
    adata_path,
    output_dir,
    radius=50,
    n_perms=1000,
    cluster_key='cell_type',
    save_adata=False,
    n_neighbors=6,
    skip_cooccurrence=True,
    max_cells_for_cooccurrence=50000
):
    """
    Process a single tile using the standard spatial analysis pipeline.

    Parameters:
    -----------
    adata_path : str or Path
        Path to h5ad file
    output_dir : str or Path
        Directory to save results
    radius : float, default=50
        Radius for spatial graph
    n_perms : int, default=1000
        Number of permutations
    cluster_key : str, default='cell_type'
        Key for cell type labels
    save_adata : bool, default=False
        Whether to save processed AnnData
    skip_cooccurrence : bool, default=True
        Whether to skip co-occurrence analysis
    max_cells_for_cooccurrence : int, default=50000
        Max cells for co-occurrence

    Returns:
    --------
    results : dict
        Dictionary containing analysis results
    """
    adata_path = Path(adata_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get tile name from file path (without .h5ad extension)
    tile_name = adata_path.stem

    # Load data
    adata = sc.read_h5ad(adata_path)

    # Apply cell type colors
    load_and_apply_cell_type_colors(adata, celltype_key=cluster_key)

    # Build spatial graph
    adata = build_spatial_graph(adata, method='knn', n_neighbors=n_neighbors)

    # Neighborhood enrichment
    adata = neighborhood_enrichment_analysis(
        adata,
        cluster_key=cluster_key,
        n_perms=n_perms
    )

    # Centrality scores
    adata = compute_centrality_scores(adata, cluster_key=cluster_key)

    # Visualizations with tile name prefix
    visualize_spatial_distribution(
        adata,
        cluster_key=cluster_key,
        save_path=output_dir / f'{tile_name}_spatial_distribution.png'
    )

    visualize_enrichment(
        adata,
        cluster_key=cluster_key,
        n_perms=n_perms,
        n_neighbors=n_neighbors,
        save_path=output_dir / f'{tile_name}_neighborhood_enrichment.png'
    )

    # Summarize interactions
    interactions_df = summarize_interactions(adata, cluster_key=cluster_key)
    interactions_df.to_csv(output_dir / f'{tile_name}_significant_interactions.csv', index=False)

    # Save processed data if requested
    if save_adata:
        output_adata_path = output_dir / f'{tile_name}_adata_with_spatial_analysis.h5ad'
        adata.write(output_adata_path)

    # Extract results
    zscore = adata.uns[f'{cluster_key}_nhood_enrichment']['zscore']
    cell_types = adata.obs[cluster_key].cat.categories.tolist()

    results = {
        'adata': adata,
        'zscore': np.array(zscore),
        'cell_types': cell_types,
        'interactions': interactions_df,
        'n_cells': adata.n_obs
    }

    return results


def aggregate_results_across_tiles(all_results, output_dir, cluster_key='cell_type'):
    """
    Aggregate and summarize results across all tiles.

    Parameters:
    -----------
    all_results : dict
        Dictionary mapping tile names to their results
    output_dir : Path
        Output directory for aggregated results
    cluster_key : str
        Key for cell type labels

    Returns:
    --------
    aggregated : dict
        Aggregated statistics across tiles
    """
    print("\n" + "=" * 70)
    print("AGGREGATING RESULTS ACROSS ALL TILES")
    print("=" * 70)

    tile_names = list(all_results.keys())
    n_tiles = len(tile_names)

    # Get cell types from first tile
    cell_types = all_results[tile_names[0]]['cell_types']
    n_celltypes = len(cell_types)

    # Collect z-scores from all tiles
    zscores_list = []
    for tile_name in tile_names:
        zscores_list.append(all_results[tile_name]['zscore'])

    zscores_array = np.stack(zscores_list)  # shape: (n_tiles, n_celltypes, n_celltypes)

    # Calculate statistics
    mean_zscore = zscores_array.mean(axis=0)
    std_zscore = zscores_array.std(axis=0)
    median_zscore = np.median(zscores_array, axis=0)
    min_zscore = zscores_array.min(axis=0)
    max_zscore = zscores_array.max(axis=0)

    print(f"\nAggregated statistics computed from {n_tiles} tiles:")
    print(f"  - Mean z-score range: [{mean_zscore.min():.2f}, {mean_zscore.max():.2f}]")
    print(f"  - Mean std across tiles: {std_zscore.mean():.3f}")
    print(f"  - Max std across tiles: {std_zscore.max():.3f}")

    # Visualize aggregated results
    print("\nGenerating aggregated visualizations...")

    # Mean enrichment heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    max_abs_value = max(abs(mean_zscore.min()), abs(mean_zscore.max()))

    sns.heatmap(
        mean_zscore,
        cmap='coolwarm',
        center=0,
        vmin=-np.ceil(max_abs_value),
        vmax=np.ceil(max_abs_value),
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Mean Z-score'},
        linewidths=0.5,
        linecolor='white',
        xticklabels=cell_types,
        yticklabels=cell_types,
        square=True,
        ax=ax
    )
    ax.set_xlabel('Cell Type', fontsize=12)
    ax.set_ylabel('Cell Type', fontsize=12)
    ax.set_title(f'Aggregated Neighborhood Enrichment\n(Mean Z-score across {n_tiles} tiles)',
                 fontsize=14, fontweight='bold', pad=20)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'aggregated_mean_enrichment.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Variability heatmap (std dev)
    fig, ax = plt.subplots(figsize=(10, 8))
    max_std = std_zscore.max()

    sns.heatmap(
        std_zscore,
        cmap='YlOrRd',
        vmin=0,
        vmax=np.ceil(max_std),
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Standard Deviation'},
        linewidths=0.5,
        linecolor='white',
        xticklabels=cell_types,
        yticklabels=cell_types,
        square=True,
        ax=ax
    )
    ax.set_xlabel('Cell Type', fontsize=12)
    ax.set_ylabel('Cell Type', fontsize=12)
    ax.set_title(f'Variability Across Tiles\n(Std Dev of Z-scores, {n_tiles} tiles)',
                 fontsize=14, fontweight='bold', pad=20)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'aggregated_variability.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  - Saved aggregated_mean_enrichment.png")
    print(f"  - Saved aggregated_variability.png")

    # Save aggregated statistics as CSV
    mean_df = pd.DataFrame(mean_zscore, index=cell_types, columns=cell_types)
    mean_df.to_csv(output_dir / 'aggregated_mean_zscore.csv')

    std_df = pd.DataFrame(std_zscore, index=cell_types, columns=cell_types)
    std_df.to_csv(output_dir / 'aggregated_std_zscore.csv')

    median_df = pd.DataFrame(median_zscore, index=cell_types, columns=cell_types)
    median_df.to_csv(output_dir / 'aggregated_median_zscore.csv')

    print(f"  - Saved aggregated statistics CSVs")

    # Aggregate interactions
    print("\nAggregating significant interactions across tiles...")
    all_interactions = []
    for tile_name, results in all_results.items():
        tile_interactions = results['interactions'].copy()
        tile_interactions['Tile'] = tile_name
        all_interactions.append(tile_interactions)

    combined_interactions = pd.concat(all_interactions, ignore_index=True)
    combined_interactions.to_csv(output_dir / 'all_tiles_interactions.csv', index=False)
    print(f"  - Saved all_tiles_interactions.csv ({len(combined_interactions)} total interactions)")

    # Summary of consistent interactions
    interaction_counts = combined_interactions.groupby(['Cell Type 1', 'Cell Type 2', 'Interaction']).size()
    interaction_counts = interaction_counts.reset_index(name='Count')
    interaction_counts['Frequency'] = interaction_counts['Count'] / n_tiles
    interaction_counts = interaction_counts.sort_values('Count', ascending=False)
    interaction_counts.to_csv(output_dir / 'interaction_consistency.csv', index=False)

    print(f"  - Saved interaction_consistency.csv")
    print(f"\nMost consistent interactions (present in multiple tiles):")
    print(interaction_counts.head(10).to_string(index=False))

    aggregated = {
        'mean_zscore': mean_zscore,
        'std_zscore': std_zscore,
        'median_zscore': median_zscore,
        'min_zscore': min_zscore,
        'max_zscore': max_zscore,
        'cell_types': cell_types,
        'n_tiles': n_tiles,
        'combined_interactions': combined_interactions,
        'interaction_consistency': interaction_counts
    }

    return aggregated


def run_multiple_tiles_pipeline(
    tiles_directory,
    output_dir='multiple_tiles_analysis',
    radius=50,
    n_perms=1000,
    cluster_key='cell_type',
    save_adata=False,
    n_neighbors=6,
    skip_cooccurrence=True,
    max_cells_for_cooccurrence=50000,
    file_pattern='*.h5ad'
):
    """
    Run spatial analysis pipeline on multiple tiled images.

    Parameters:
    -----------
    tiles_directory : str
        Directory containing h5ad files
    output_dir : str, default='multiple_tiles_analysis'
        Directory to save results
    radius : float, default=50
        Radius for spatial graph
    n_perms : int, default=1000
        Number of permutations
    cluster_key : str, default='cell_type'
        Key for cell type labels
    save_adata : bool, default=False
        Whether to save processed AnnData objects
    skip_cooccurrence : bool, default=True
        Whether to skip co-occurrence analysis
    max_cells_for_cooccurrence : int, default=50000
        Max cells for co-occurrence
    file_pattern : str, default='*.h5ad'
        Pattern to match h5ad files

    Returns:
    --------
    results : dict
        Dictionary containing all results
    """
    print("=" * 70)
    print("MULTIPLE TILES SPATIAL ANALYSIS PIPELINE")
    print("=" * 70)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nOutput directory: {output_dir}")

    # Find all h5ad files
    print(f"\nSearching for h5ad files in: {tiles_directory}")
    h5ad_files = find_h5ad_files(tiles_directory, pattern=file_pattern)
    n_tiles = len(h5ad_files)

    print(f"\n" + "=" * 70)
    print(f"PROCESSING {n_tiles} TILES")
    print("=" * 70)

    # Process each tile
    all_results = {}
    failed_tiles = []
    skipped_tiles = []

    for i, h5ad_path in enumerate(tqdm(h5ad_files, desc="Processing tiles")):
        tile_name = h5ad_path.stem
        tile_output_dir = output_dir / tile_name

        print(f"\n[{i+1}/{n_tiles}] Processing: {tile_name}")

        # Check if tile is already processed
        if is_tile_processed(tile_output_dir, tile_name):
            print(f"  ⊙ Skipped: Already processed (found all output files)")
            skipped_tiles.append(tile_name)

            # Load existing results for aggregation
            try:
                # Read the existing results with tile name prefix
                interactions_df = pd.read_csv(tile_output_dir / f'{tile_name}_significant_interactions.csv')
                adata = sc.read_h5ad(h5ad_path)
                zscore = adata.uns[f'{cluster_key}_nhood_enrichment']['zscore'] if f'{cluster_key}_nhood_enrichment' in adata.uns else None

                # If zscore not in original file, we need to recompute (skip for now)
                if zscore is None:
                    print(f"  ⚠ Warning: Could not load zscore from existing data, will reprocess")
                    # Remove from skipped and continue to process
                    skipped_tiles.remove(tile_name)
                else:
                    cell_types = adata.obs[cluster_key].cat.categories.tolist()
                    results = {
                        'adata': adata,
                        'zscore': np.array(zscore),
                        'cell_types': cell_types,
                        'interactions': interactions_df,
                        'n_cells': adata.n_obs
                    }
                    all_results[tile_name] = results
                    continue
            except Exception as e:
                print(f"  ⚠ Warning: Could not load existing results ({e}), will reprocess")
                skipped_tiles.remove(tile_name)

        try:
            results = process_single_tile(
                adata_path=h5ad_path,
                output_dir=tile_output_dir,
                radius=radius,
                n_perms=n_perms,
                n_neighbors=n_neighbors,
                cluster_key=cluster_key,
                save_adata=save_adata,
                skip_cooccurrence=skip_cooccurrence,
                max_cells_for_cooccurrence=max_cells_for_cooccurrence
            )
            all_results[tile_name] = results
            print(f"  ✓ Success: {results['n_cells']} cells, {len(results['interactions'])} significant interactions")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed_tiles.append((tile_name, str(e)))
            continue

    # Summary
    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"  - Total tiles: {n_tiles}")
    print(f"  - Successfully processed: {len(all_results)}")
    print(f"  - Skipped (already processed): {len(skipped_tiles)}")
    print(f"  - Failed: {len(failed_tiles)}")

    if skipped_tiles:
        print(f"\nSkipped tiles (already processed):")
        for tile_name in skipped_tiles[:10]:  # Show first 10
            print(f"  - {tile_name}")
        if len(skipped_tiles) > 10:
            print(f"  ... and {len(skipped_tiles) - 10} more")

    if failed_tiles:
        print("\nFailed tiles:")
        for tile_name, error in failed_tiles:
            print(f"  - {tile_name}: {error}")

    if len(all_results) == 0:
        raise RuntimeError("All tiles failed to process!")

    # Aggregate results
    aggregated = aggregate_results_across_tiles(
        all_results,
        output_dir,
        cluster_key=cluster_key
    )

    # Create summary report
    print("\n" + "=" * 70)
    print("CREATING SUMMARY REPORT")
    print("=" * 70)

    summary_data = []
    for tile_name, results in all_results.items():
        summary_data.append({
            'Tile': tile_name,
            'N_Cells': results['n_cells'],
            'N_Significant_Interactions': len(results['interactions']),
            'Mean_Abs_Zscore': np.abs(results['zscore']).mean(),
            'Max_Abs_Zscore': np.abs(results['zscore']).max()
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('N_Cells', ascending=False)
    summary_df.to_csv(output_dir / 'tiles_summary.csv', index=False)

    print(f"\nTiles summary:")
    print(summary_df.to_string(index=False))
    print(f"\n  - Saved tiles_summary.csv")

    # Final results
    results = {
        'all_results': all_results,
        'aggregated': aggregated,
        'summary': summary_df,
        'failed_tiles': failed_tiles,
        'parameters': {
            'n_tiles': n_tiles,
            'radius': radius,
            'n_perms': n_perms,
            'cluster_key': cluster_key
        }
    }

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}/")
    print(f"\nKey outputs:")
    print(f"  - Individual tile results: {output_dir}/<tile_name>/")
    print(f"  - Aggregated mean enrichment: aggregated_mean_enrichment.png")
    print(f"  - Variability across tiles: aggregated_variability.png")
    print(f"  - All interactions: all_tiles_interactions.csv")
    print(f"  - Interaction consistency: interaction_consistency.csv")
    print(f"  - Tiles summary: tiles_summary.csv")

    return results


# Example usage
if __name__ == "__main__":
    # Configuration
    tiles_directory = '/mnt/g/GDC-TCGA-LUAD/00a0b174-1eab-446a-ba8c-7c6e3acd7f0c/pred/h5ad'
    output_dir = 'multiple_tiles_analysis'

    # Run pipeline on multiple tiles
    results = run_multiple_tiles_pipeline(
        tiles_directory=tiles_directory,
        output_dir=output_dir,
        radius=50,                      # Adjust based on your tissue/magnification
        n_perms=1000,                   # Number of permutations
        n_neighbors=6,
        cluster_key='cell_type',        # Adjust to your cell type column
        save_adata=False,               # Set to True to save processed h5ad files
        skip_cooccurrence=True,         # Skip co-occurrence for faster processing
        max_cells_for_cooccurrence=50000,
        file_pattern='*.h5ad'           # Pattern to match files
    )

    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
Multiple Tiles Analysis Results:

1. INDIVIDUAL TILE RESULTS:
   - Each tile processed independently
   - Results saved in separate subdirectories
   - Allows comparison of spatial patterns across tiles

2. AGGREGATED MEAN ENRICHMENT:
   - Average z-scores across all tiles
   - Shows consistent spatial patterns
   - More robust than single tile analysis

3. VARIABILITY ACROSS TILES:
   - Standard deviation of z-scores
   - High variability = interaction varies by tile/region
   - Low variability = consistent pattern across all tiles

4. INTERACTION CONSISTENCY:
   - Shows which interactions appear in multiple tiles
   - Frequency = proportion of tiles with this interaction
   - High frequency = robust, reproducible pattern

5. TILES SUMMARY:
   - Overview of all processed tiles
   - Cell counts and interaction statistics
   - Use to identify outlier tiles

RECOMMENDATIONS:
- Focus on interactions with high consistency across tiles
- High variability suggests heterogeneous tissue regions
- Compare individual tiles to identify region-specific patterns
- Use aggregated results for overall tissue-level conclusions
""")