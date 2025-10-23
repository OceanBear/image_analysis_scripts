"""
Quick script to check what squidpy nhood_enrichment actually returns
"""

import numpy as np
import scanpy as sc
import squidpy as sq
from pathlib import Path

# Load one of your processed tiles
tile_dirs = sorted(Path('neighborhood_composition/neighborhood_enrichment/multiple_tiles_analysis').glob('tile_*'))

if len(tile_dirs) == 0:
    print("No tile directories found!")
    print("Please update the path to your actual results directory")
else:
    print(f"Found {len(tile_dirs)} tile directories")
    print(f"First tile: {tile_dirs[0]}")

    # Check what's in the first tile
    first_tile = tile_dirs[0]
    zscore_file = list(first_tile.glob('*_bootstrap_zscores.npy'))[0]

    zscores = np.load(zscore_file)
    print(f"\nBootstrap zscores shape: {zscores.shape}")
    print(f"Min: {np.nanmin(zscores):.2f}")
    print(f"Max: {np.nanmax(zscores):.2f}")
    print(f"Mean: {np.nanmean(zscores):.2f}")
    print(f"Std: {np.nanstd(zscores):.2f}")

    # Sample some values
    print(f"\nSample from first bootstrap iteration:")
    print(zscores[0, :3, :3])

    print("\n" + "="*70)
    print("DIAGNOSIS:")
    print("="*70)

    if np.nanmean(np.abs(zscores)) > 5:
        print("⚠ VALUES ARE TOO HIGH FOR Z-SCORES!")
        print(f"  Average |z-score|: {np.nanmean(np.abs(zscores)):.2f}")
        print(f"  Expected range for z-scores: -3 to +3")
        print(f"  Actual range: {np.nanmin(zscores):.2f} to {np.nanmax(zscores):.2f}")
        print("\nPossible causes:")
        print("1. Squidpy is returning enrichment scores, not standardized z-scores")
        print("2. Permutation normalization is not working correctly")
        print("3. Very strong spatial clustering in your data")

        # Check if these look like log-fold enrichments
        if np.nanmin(zscores) > 0:
            print("\n⚠ All values are positive - these might be log-fold enrichments!")
        elif np.nanmean(zscores) > 2:
            print("\n⚠ Mean is very high - likely enrichment ratios, not z-scores!")
    else:
        print("✓ Values look reasonable for z-scores")
        print(f"  Average |z-score|: {np.nanmean(np.abs(zscores)):.2f}")