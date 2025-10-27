#!/usr/bin/env python3
"""
Example script demonstrating how to use the unified CN detection.

This script shows different usage patterns for cn_unified_kmeans.py
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print the result."""
    print("\n" + "=" * 80)
    print(f"EXAMPLE: {description}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    # Note: This is just showing the command, not actually running it
    # To actually run it, uncomment the next lines:
    # result = subprocess.run(cmd, capture_output=True, text=True)
    # print(result.stdout)
    # if result.stderr:
    #     print("STDERR:", result.stderr)
    # return result.returncode == 0
    
    return True


def main():
    """Show example usage patterns."""
    
    print("=" * 80)
    print("UNIFIED CELLULAR NEIGHBORHOOD DETECTION - EXAMPLE USAGE")
    print("=" * 80)
    
    # Example 1: Basic usage with all defaults
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Usage (All Defaults)")
    print("=" * 80)
    print("Process all tiles in directory with default parameters")
    print("\nCommand:")
    print("python cn_unified_kmeans.py \\")
    print("  --tiles_dir /path/to/tiles \\")
    print("  --output_dir cn_unified_results")
    print("\nThis will:")
    print("  - Process ALL h5ad files in the directory")
    print("  - Use k=20 neighbors for CN detection")
    print("  - Detect 6 cellular neighborhoods")
    print("  - Generate 1 unified heatmap + 1 spatial map per tile")
    
    # Example 2: Testing with limited tiles
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Testing Mode (Limited Tiles)")
    print("=" * 80)
    print("Process only first 5 tiles to test parameters")
    print("\nCommand:")
    print("python cn_unified_kmeans.py \\")
    print("  --tiles_dir /path/to/tiles \\")
    print("  --output_dir test_results \\")
    print("  --max_tiles 5 \\")
    print("  --n_clusters 4")
    print("\nThis will:")
    print("  - Process only the FIRST 5 tiles")
    print("  - Detect 4 cellular neighborhoods (fewer for testing)")
    print("  - Generate outputs in test_results/")
    
    # Example 3: Custom cell type column
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Custom Cell Type Column")
    print("=" * 80)
    print("If your h5ad files use 'celltype' instead of 'cell_type'")
    print("\nCommand:")
    print("python cn_unified_kmeans.py \\")
    print("  --tiles_dir /path/to/tiles \\")
    print("  --output_dir cn_unified_results \\")
    print("  --celltype_key celltype")
    
    # Example 4: Complex tissue with more CNs
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Complex Tissue (More CNs)")
    print("=" * 80)
    print("For tissues with many distinct microenvironments")
    print("\nCommand:")
    print("python cn_unified_kmeans.py \\")
    print("  --tiles_dir /path/to/tiles \\")
    print("  --output_dir cn_unified_10clusters \\")
    print("  --n_clusters 10 \\")
    print("  --k 25")
    print("\nThis will:")
    print("  - Detect 10 cellular neighborhoods (more granular)")
    print("  - Use k=25 neighbors (slightly larger neighborhood)")
    
    # Example 5: Specific file pattern
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Specific File Pattern")
    print("=" * 80)
    print("Process only files matching a specific pattern")
    print("\nCommand:")
    print("python cn_unified_kmeans.py \\")
    print("  --tiles_dir /path/to/tiles \\")
    print("  --output_dir cn_unified_results \\")
    print("  --pattern 'tile_*_adata.h5ad'")
    
    # Example 6: Real-world usage
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Real-World Usage (14 Tiles)")
    print("=" * 80)
    print("Process 14 tiles from TCGA-LUAD dataset")
    print("\nCommand:")
    print("python cn_unified_kmeans.py \\")
    print("  --tiles_dir /mnt/g/GDC-TCGA-LUAD/00a0b174-1eab-446a-ba8c-7c6e3acd7f0c/pred/h5ad \\")
    print("  --output_dir cn_unified_luad \\")
    print("  --k 20 \\")
    print("  --n_clusters 6")
    print("\nExpected output:")
    print("  cn_unified_luad/")
    print("  ├── unified_analysis/")
    print("  │   ├── unified_cn_composition_heatmap.png  (1 file)")
    print("  │   ├── unified_cn_composition.csv")
    print("  │   └── unified_cn_summary.json")
    print("  ├── individual_tiles/")
    print("  │   ├── tile1/spatial_cns.png")
    print("  │   ├── tile2/spatial_cns.png")
    print("  │   └── ... (14 tiles total)")
    print("  └── processed_h5ad/")
    print("      ├── tile1_adata_cns.h5ad")
    print("      ├── tile2_adata_cns.h5ad")
    print("      └── ... (14 files total)")
    
    # Workflow example
    print("\n" + "=" * 80)
    print("COMPLETE WORKFLOW: CN Detection → Spatial Context Analysis")
    print("=" * 80)
    print("\nStep 1: Unified CN Detection")
    print("python cn_unified_kmeans.py \\")
    print("  --tiles_dir /path/to/tiles \\")
    print("  --output_dir cn_unified_results")
    print("\nStep 2: Spatial Context Analysis (uses CN results)")
    print("cd ../spatial_contexts")
    print("python spatial_contexts.py \\")
    print("  --batch_dir ../cellular_neighborhoods/cn_unified_results \\")
    print("  --output_dir sc_unified_results")
    
    # Python API usage
    print("\n" + "=" * 80)
    print("PYTHON API USAGE")
    print("=" * 80)
    print("\nYou can also use the detector programmatically:")
    print("""
from cn_unified_kmeans import UnifiedCellularNeighborhoodDetector

# Initialize detector
detector = UnifiedCellularNeighborhoodDetector(
    tiles_directory='/path/to/tiles',
    output_dir='cn_unified_results'
)

# Discover tiles
tile_files = detector.discover_tiles(pattern='*.h5ad', max_tiles=None)

# Run full pipeline
detector.run_full_pipeline(
    tile_files=tile_files,
    k=20,
    n_clusters=6,
    celltype_key='cell_type',
    random_state=220705,
    coord_offset=True
)

# Access results
print(f"Processed {len(detector.tile_list)} tiles")
print(f"Total cells: {detector.combined_adata.n_obs}")
print(f"CN distribution: {detector.combined_adata.obs['cn_celltype'].value_counts()}")
""")
    
    # Parameter tuning guide
    print("\n" + "=" * 80)
    print("PARAMETER TUNING GUIDE")
    print("=" * 80)
    print("\n1. k (Number of Neighbors)")
    print("   - Default: 20")
    print("   - Lower (10-15): More local, fine-grained neighborhoods")
    print("   - Higher (30-40): More global, coarser neighborhoods")
    print("   - Start with 20, adjust based on results")
    print("\n2. n_clusters (Number of CNs)")
    print("   - Default: 6")
    print("   - Lower (3-4): Simpler tissues")
    print("   - Higher (8-12): Complex tissues with many microenvironments")
    print("   - Look at heatmap to decide if you need more/fewer")
    print("\n3. celltype_key")
    print("   - Default: 'cell_type'")
    print("   - Change if your data uses different column name")
    print("   - Check with: import scanpy as sc; adata = sc.read_h5ad('file.h5ad'); print(adata.obs.columns)")
    
    print("\n" + "=" * 80)
    print("For more information, see README_UNIFIED_CN.md")
    print("=" * 80)


if __name__ == '__main__':
    main()

