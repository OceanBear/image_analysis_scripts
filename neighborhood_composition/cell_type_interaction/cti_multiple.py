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
import os
from pathlib import Path
# Set the working directory to the script's directory
os.chdir(Path(__file__).parent)

# Import functions from cti_tiled.py
from cti_tiled import (
    load_and_apply_cell_type_colors,
    build_spatial_graph,
    neighborhood_enrichment_analysis,
    compute_centrality_scores,
    visualize_enrichment,
    visualize_spatial_distribution,
    summarize_interactions,
    save_intermediate_results,
    load_intermediate_results,
    aggregate_from_saved_results
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
        f'{tile_name}_significant_interactions.csv',
        f'{tile_name}_zscore.npy',         # Intermediate file for aggregation
        f'{tile_name}_metadata.json'       # Intermediate metadata
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

    # Save intermediate results for file-based aggregation (STEP 1)
    save_intermediate_results(
        adata=adata,
        output_dir=output_dir,
        tile_name=tile_name,
        cluster_key=cluster_key
    )

    # Save processed data if requested
    if save_adata:
        output_adata_path = output_dir / f'{tile_name}_adata_with_spatial_analysis.h5ad'
        adata.write(output_adata_path)

    # Return minimal summary (don't keep full adata in memory)
    results = {
        'tile_name': tile_name,
        'n_cells': adata.n_obs,
        'n_interactions': len(interactions_df)
    }

    return results


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

    # Process each tile (memory-efficient - don't store in RAM)
    successful_tiles = []
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
            successful_tiles.append(tile_name)  # Count as successful for aggregation
            continue

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
            successful_tiles.append(tile_name)
            print(f"  ✓ Success: {results['n_cells']} cells, {results['n_interactions']} significant interactions")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed_tiles.append((tile_name, str(e)))
            continue

    # Summary
    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"  - Total tiles: {n_tiles}")
    print(f"  - Successfully processed: {len(successful_tiles)}")
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

    if len(successful_tiles) == 0:
        raise RuntimeError("All tiles failed to process!")

    # STEP 2: Aggregate results from disk (memory-efficient)
    print("\n" + "=" * 70)
    print("FILE-BASED AGGREGATION (STEP 2)")
    print("=" * 70)

    # Collect tile directories for aggregation
    tile_dirs = [output_dir / tile_name for tile_name in successful_tiles]

    aggregated = aggregate_from_saved_results(
        tile_dirs=tile_dirs,
        output_dir=output_dir,
        tile_names=successful_tiles
    )

    # Create summary report from saved metadata
    print("\n" + "=" * 70)
    print("CREATING SUMMARY REPORT")
    print("=" * 70)

    summary_data = []
    for tile_name in successful_tiles:
        tile_dir = output_dir / tile_name
        try:
            # Load metadata from saved files
            metadata_result = load_intermediate_results(tile_dir, tile_name=tile_name)
            interactions_df = pd.read_csv(tile_dir / f'{tile_name}_significant_interactions.csv')

            summary_data.append({
                'Tile': tile_name,
                'N_Cells': metadata_result['n_cells'],
                'N_Significant_Interactions': len(interactions_df),
                'Mean_Abs_Zscore': metadata_result['metadata']['mean_abs_zscore'],
                'Max_Abs_Zscore': metadata_result['metadata']['max_abs_zscore']
            })
        except Exception as e:
            print(f"  ⚠ Warning: Could not load summary for {tile_name}: {e}")

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('N_Cells', ascending=False)
    summary_df.to_csv(output_dir / 'tiles_summary.csv', index=False)

    print(f"\nTiles summary:")
    print(summary_df.to_string(index=False))
    print(f"\n  - Saved tiles_summary.csv")

    # Aggregate interactions from CSVs
    print("\nAggregating interaction CSVs...")
    all_interactions = []
    for tile_name in successful_tiles:
        tile_dir = output_dir / tile_name
        interactions_csv = tile_dir / f'{tile_name}_significant_interactions.csv'
        if interactions_csv.exists():
            tile_interactions = pd.read_csv(interactions_csv)
            tile_interactions['Tile'] = tile_name
            all_interactions.append(tile_interactions)

    if all_interactions:
        combined_interactions = pd.concat(all_interactions, ignore_index=True)
        combined_interactions.to_csv(output_dir / 'all_tiles_interactions.csv', index=False)

        # Interaction consistency
        interaction_counts = combined_interactions.groupby(['Cell Type 1', 'Cell Type 2', 'Interaction']).size()
        interaction_counts = interaction_counts.reset_index(name='Count')
        interaction_counts['Frequency'] = interaction_counts['Count'] / len(successful_tiles)
        interaction_counts = interaction_counts.sort_values('Count', ascending=False)
        interaction_counts.to_csv(output_dir / 'interaction_consistency.csv', index=False)

        print(f"  - Saved all_tiles_interactions.csv ({len(combined_interactions)} total interactions)")
        print(f"  - Saved interaction_consistency.csv")
        print(f"\nMost consistent interactions (present in multiple tiles):")
        print(interaction_counts.head(10).to_string(index=False))

    # Final results
    results = {
        'aggregated': aggregated,
        'summary': summary_df,
        'failed_tiles': failed_tiles,
        'successful_tiles': successful_tiles,
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