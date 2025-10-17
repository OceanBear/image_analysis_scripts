"""
Example script demonstrating DBSCAN-based cellular neighborhood detection.

This script shows three different ways to use the DBSCANCellularNeighborhoodDetector:
1. Run only DBSCAN clustering
2. Run only K-means clustering (original method)
3. Compare both methods side-by-side

Author: Generated with Claude Code
Date: 2025-10-15
"""

import scanpy as sc
from dbscan_cellular_neighborhoods import DBSCANCellularNeighborhoodDetector


def example_dbscan_only(save_adata=False):
    """
    Example 1: Run DBSCAN clustering only.

    DBSCAN parameters guide:
    - eps: Controls the neighborhood radius. Smaller = more clusters
           Typical range: 0.3 - 1.0 for normalized data
    - min_samples: Minimum points to form a cluster. Larger = denser clusters
                   Typical range: 3 - 10

    Parameters:
    -----------
    save_adata : bool, default=False
        Whether to save the h5ad file with results
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: DBSCAN CLUSTERING ONLY")
    print("=" * 80)

    # Load your data
    input_file = 'tile_39520_7904.h5ad'  # Replace with your file
    adata = sc.read_h5ad(input_file)

    # Initialize detector
    detector = DBSCANCellularNeighborhoodDetector(adata)

    # Run DBSCAN pipeline
    detector.run_full_pipeline_dbscan(
        k=20,                      # Number of neighbors for graph construction
        eps=0.1,                   # DBSCAN epsilon (neighborhood radius)
        min_samples=5,             # DBSCAN min_samples
        handle_noise='separate',   # 'separate' or 'nearest'
        celltype_key='cell_type',  # Adjust to your column name
        img_id_key='tile_name',    # Adjust to your column name
        output_dir='cn_results_dbscan'
    )

    # Save results (optional)
    if save_adata:
        adata.write('cn_results_dbscan/adata_with_dbscan_cns.h5ad')
        print("\n  - Saved processed AnnData to: cn_results_dbscan/adata_with_dbscan_cns.h5ad")
    else:
        print("\n  - AnnData not saved (set save_adata=True to save)")

    print("\nDBSCAN analysis complete!")


def example_kmeans_only(save_adata=False):
    """
    Example 2: Run K-means clustering only (original method).

    K-means parameters guide:
    - n_clusters: Number of cellular neighborhoods to detect
                  You need to specify this beforehand

    Parameters:
    -----------
    save_adata : bool, default=False
        Whether to save the h5ad file with results
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: K-MEANS CLUSTERING ONLY")
    print("=" * 80)

    # Load your data
    input_file = 'tile_39520_7904.h5ad'
    adata = sc.read_h5ad(input_file)

    # Initialize detector (can use either class, both support K-means)
    detector = DBSCANCellularNeighborhoodDetector(adata)

    # Run K-means pipeline (original method)
    detector.run_full_pipeline(
        k=20,                      # Number of neighbors
        n_clusters=6,              # Number of clusters to detect
        celltype_key='cell_type',
        img_id_key='tile_name',
        output_dir='cn_results_kmeans',
        random_state=220705
    )

    # Save results (optional)
    if save_adata:
        adata.write('cn_results_kmeans/adata_with_kmeans_cns.h5ad')
        print("\n  - Saved processed AnnData to: cn_results_kmeans/adata_with_kmeans_cns.h5ad")
    else:
        print("\n  - AnnData not saved (set save_adata=True to save)")

    print("\nK-means analysis complete!")


def example_compare_methods(save_adata=False):
    """
    Example 3: Compare K-means and DBSCAN side-by-side.

    This is useful for:
    - Evaluating which method works better for your data
    - Understanding differences between methods
    - Parameter tuning

    Parameters:
    -----------
    save_adata : bool, default=False
        Whether to save the h5ad file with results
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: COMPARE K-MEANS vs DBSCAN")
    print("=" * 80)

    # Load your data
    input_file = 'tile_39520_7904.h5ad'
    adata = sc.read_h5ad(input_file)

    # Initialize detector
    detector = DBSCANCellularNeighborhoodDetector(adata)

    # Compare both methods
    detector.compare_methods(
        k=20,                      # Number of neighbors (shared)
        n_clusters=6,              # K-means: number of clusters
        eps=0.1,                   # DBSCAN: epsilon
        min_samples=5,             # DBSCAN: min_samples
        celltype_key='cell_type',
        img_id_key='tile_name',
        output_dir='cn_comparison',
        random_state=220705
    )

    # The adata now contains both:
    # - adata.obs['cn_celltype_kmeans']: K-means labels
    # - adata.obs['cn_celltype_dbscan']: DBSCAN labels

    # Save results (optional)
    if save_adata:
        adata.write('cn_comparison/adata_with_both_methods.h5ad')
        print("\n  - Saved processed AnnData to: cn_comparison/adata_with_both_methods.h5ad")
    else:
        print("\n  - AnnData not saved (set save_adata=True to save)")

    print("\nComparison complete!")


def example_manual_workflow(save_adata=False):
    """
    Example 4: Manual step-by-step workflow for more control.

    This gives you full control over each step and allows
    you to customize parameters or add custom analysis.

    Parameters:
    -----------
    save_adata : bool, default=False
        Whether to save the h5ad file with results
    """
    import os

    print("\n" + "=" * 80)
    print("EXAMPLE 4: MANUAL STEP-BY-STEP WORKFLOW")
    print("=" * 80)

    # Create output directory
    os.makedirs('manual_workflow', exist_ok=True)

    # Load your data
    input_file = 'tile_39520_7904.h5ad'
    adata = sc.read_h5ad(input_file)

    # Initialize detector
    detector = DBSCANCellularNeighborhoodDetector(adata)

    # Step 1: Build KNN graph
    print("\nStep 1: Building KNN graph...")
    detector.build_knn_graph(k=20)

    # Step 2: Aggregate neighbors
    print("\nStep 2: Aggregating neighbors...")
    detector.aggregate_neighbors(cluster_key='cell_type')

    # Step 3a: Detect CNs with DBSCAN
    print("\nStep 3a: Detecting CNs with DBSCAN...")
    detector.detect_cellular_neighborhoods_dbscan(
        eps=0.1,
        min_samples=5,
        output_key='cn_dbscan'
    )

    # Step 3b: Detect CNs with K-means for comparison
    print("\nStep 3b: Detecting CNs with K-means...")
    detector.detect_cellular_neighborhoods(
        n_clusters=6,
        output_key='cn_kmeans'
    )

    # Step 4: Visualize DBSCAN results
    print("\nStep 4: Visualizing results...")
    detector.visualize_spatial_cns(
        cn_key='cn_dbscan',
        img_id_key='tile_name',
        save_path='manual_workflow/spatial_cns_dbscan.png'
    )

    # Step 5: Compute and visualize composition
    composition_dbscan = detector.compute_cn_composition(
        cn_key='cn_dbscan',
        celltype_key='cell_type'
    )
    print("\nDBSCAN CN Composition:")
    print(composition_dbscan)

    # Save CSV results
    composition_dbscan.to_csv('manual_workflow/composition_dbscan.csv')

    # Save h5ad results (optional)
    if save_adata:
        adata.write('manual_workflow/adata_manual.h5ad')
        print("\n  - Saved processed AnnData to: manual_workflow/adata_manual.h5ad")
    else:
        print("\n  - AnnData not saved (set save_adata=True to save)")

    print("\nManual workflow complete!")


def parameter_tuning_guide():
    """
    Guide for tuning DBSCAN parameters.
    """
    print("\n" + "=" * 80)
    print("DBSCAN PARAMETER TUNING GUIDE")
    print("=" * 80)

    print("""
DBSCAN has two main parameters that need tuning:

1. EPS (epsilon) - Neighborhood radius:
   - Smaller eps → More clusters, more noise points
   - Larger eps → Fewer clusters, less noise
   - Start with 0.5 and adjust based on results
   - Try range: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]

2. MIN_SAMPLES - Minimum cluster size:
   - Smaller value → More small clusters
   - Larger value → Fewer, denser clusters
   - Typical range: 3-10
   - Rule of thumb: 2 * number_of_features (dimensions)

3. HANDLE_NOISE - How to deal with outliers:
   - 'separate': Keeps noise as a separate cluster (CN 0)
   - 'nearest': Assigns noise to nearest cluster centroid

TIPS:
- If you get too many clusters → increase eps
- If you get too few clusters → decrease eps
- If you get too much noise → decrease min_samples or increase eps
- If clusters are too fragmented → increase min_samples
- Try the compare_methods() function to compare with K-means

EXAMPLE PARAMETER COMBINATIONS:
- Conservative (fewer, denser clusters): eps=0.7, min_samples=10
- Balanced: eps=0.5, min_samples=5
- Aggressive (more, smaller clusters): eps=0.3, min_samples=3
""")


def main():
    """
    Main function to run examples.
    Uncomment the example you want to run.

    Note: Set save_adata=True to save h5ad files with results.
    """

    # Uncomment the example you want to run:
    
    SAVE_ADATA_ORNOT = False
    example_dbscan_only(save_adata=SAVE_ADATA_ORNOT)       # Set to True to save h5ad
    example_kmeans_only(save_adata=SAVE_ADATA_ORNOT)       # Set to True to save h5ad
    example_compare_methods(save_adata=SAVE_ADATA_ORNOT)   # Set to True to save h5ad
    example_manual_workflow(save_adata=SAVE_ADATA_ORNOT)   # Set to True to save h5ad
    parameter_tuning_guide()

    # print("\n" + "=" * 80)
    # print("To run an example, uncomment the desired function in main()")
    # print("=" * 80)


if __name__ == '__main__':
    main()