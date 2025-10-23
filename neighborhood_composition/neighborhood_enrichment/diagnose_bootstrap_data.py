"""
Diagnostic script to check bootstrap intermediate results.

This helps identify data quality issues before aggregation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import os
from pathlib import Path
# Set the working directory to the script's directory
os.chdir(Path(__file__).parent)

def diagnose_single_tile(tile_dir):
    """Diagnose a single tile's bootstrap results."""
    tile_dir = Path(tile_dir)
    tile_name = tile_dir.name

    print(f"\n{'='*70}")
    print(f"Tile: {tile_name}")
    print(f"{'='*70}")

    # Load metadata
    metadata_files = list(tile_dir.glob('*_bootstrap_metadata.json'))
    if not metadata_files:
        print("  [!] No metadata file found")
        return None

    with open(metadata_files[0], 'r') as f:
        metadata = json.load(f)

    # Load z-scores
    zscore_files = list(tile_dir.glob('*_bootstrap_zscores.npy'))
    if not zscore_files:
        print("  [!] No zscores file found")
        return None

    zscores = np.load(zscore_files[0])

    print(f"  Shape: {zscores.shape}")
    print(f"  Cell types: {metadata['cell_types']}")
    print(f"\n  Data Quality:")
    print(f"    - Min: {np.nanmin(zscores):.2f}")
    print(f"    - Max: {np.nanmax(zscores):.2f}")
    print(f"    - Mean: {np.nanmean(zscores):.2f}")
    print(f"    - Median: {np.nanmedian(zscores):.2f}")
    print(f"    - Std: {np.nanstd(zscores):.2f}")
    print(f"    - NaN count: {np.isnan(zscores).sum()}/{zscores.size} ({100*np.isnan(zscores).sum()/zscores.size:.1f}%)")
    print(f"    - Inf count: {np.isinf(zscores).sum()}")

    # Check for extreme values
    extreme_threshold = 10.0
    n_extreme = (np.abs(zscores) > extreme_threshold).sum()
    if n_extreme > 0:
        print(f"    - Values > {extreme_threshold}: {n_extreme} ({100*n_extreme/zscores.size:.1f}%)")

    # Check mean across bootstrap iterations
    mean_per_bootstrap = zscores.mean(axis=(1,2))
    print(f"\n  Bootstrap Consistency:")
    print(f"    - Mean z-score per iteration: {mean_per_bootstrap.mean():.2f} ± {mean_per_bootstrap.std():.2f}")
    print(f"    - Range: [{mean_per_bootstrap.min():.2f}, {mean_per_bootstrap.max():.2f}]")

    # Check if any cell type pairs are consistently NaN or extreme
    mean_zscore = np.nanmean(zscores, axis=0)
    std_zscore = np.nanstd(zscores, axis=0)

    cell_types = metadata['cell_types']

    print(f"\n  Problematic Cell Type Pairs:")
    for i, ct1 in enumerate(cell_types):
        for j, ct2 in enumerate(cell_types):
            mean_val = mean_zscore[i, j]
            std_val = std_zscore[i, j]

            if np.isnan(mean_val):
                print(f"    - {ct1} - {ct2}: NaN")
            elif np.abs(mean_val) > 10:
                print(f"    - {ct1} - {ct2}: {mean_val:.2f} ± {std_val:.2f} (EXTREME)")
            elif std_val > 5:
                print(f"    - {ct1} - {ct2}: {mean_val:.2f} ± {std_val:.2f} (HIGH VARIABILITY)")

    return {
        'tile_name': tile_name,
        'shape': zscores.shape,
        'n_nan': np.isnan(zscores).sum(),
        'n_inf': np.isinf(zscores).sum(),
        'n_extreme': n_extreme,
        'min': np.nanmin(zscores),
        'max': np.nanmax(zscores),
        'mean': np.nanmean(zscores),
        'std': np.nanstd(zscores),
        'metadata': metadata
    }


def diagnose_all_tiles(results_dir, tile_pattern='tile_*'):
    """Diagnose all tiles in a directory."""
    results_dir = Path(results_dir)

    # Find all tile directories
    tile_dirs = sorted([d for d in results_dir.glob(tile_pattern) if d.is_dir()])

    print(f"\n{'='*70}")
    print(f"DIAGNOSING {len(tile_dirs)} TILES")
    print(f"{'='*70}")

    summaries = []
    for tile_dir in tile_dirs:
        summary = diagnose_single_tile(tile_dir)
        if summary:
            summaries.append(summary)

    # Overall summary
    print(f"\n{'='*70}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*70}")

    if summaries:
        total_nan = sum(s['n_nan'] for s in summaries)
        total_inf = sum(s['n_inf'] for s in summaries)
        total_extreme = sum(s['n_extreme'] for s in summaries)

        print(f"  Total tiles analyzed: {len(summaries)}")
        print(f"  Total NaN values: {total_nan}")
        print(f"  Total Inf values: {total_inf}")
        print(f"  Total extreme values (|z|>10): {total_extreme}")

        all_means = [s['mean'] for s in summaries]
        all_stds = [s['std'] for s in summaries]

        print(f"\n  Mean z-score across tiles: {np.mean(all_means):.2f} ± {np.std(all_means):.2f}")
        print(f"  Mean std across tiles: {np.mean(all_stds):.2f} ± {np.std(all_stds):.2f}")

        # Check cell type consistency
        cell_type_sets = [set(s['metadata']['cell_types']) for s in summaries]
        if len(set(map(tuple, map(sorted, cell_type_sets)))) > 1:
            print(f"\n  [!] WARNING: Cell types are NOT consistent across tiles!")
            all_cell_types = sorted(set.union(*cell_type_sets))
            print(f"      All cell types across tiles: {all_cell_types}")
        else:
            print(f"\n  Cell types are consistent across all tiles")

    return summaries


if __name__ == "__main__":
    # Example usage - UPDATE THIS PATH TO YOUR ACTUAL RESULTS DIRECTORY
    import sys

    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        # Default paths to try
        possible_paths = [
            Path('multiple_tiles_analysis'),
            Path('bootstrap_multiple_analysis'),
            Path('neighborhood_composition/neighborhood_enrichment/multiple_tiles_analysis'),
        ]

        results_dir = None
        for path in possible_paths:
            if path.exists():
                results_dir = path
                print(f"Using results directory: {results_dir}")
                break

        if results_dir is None:
            print("ERROR: Could not find results directory!")
            print("Please specify the path as a command line argument:")
            print("  python diagnose_bootstrap_data.py /path/to/results")
            print("\nOr update the script with your actual path")
            sys.exit(1)

    # Run diagnostic
    diagnose_all_tiles(results_dir)