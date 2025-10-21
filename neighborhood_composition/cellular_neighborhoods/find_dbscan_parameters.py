"""
Script to find optimal DBSCAN parameters for your data.

Run this first before running the full DBSCAN pipeline to get
appropriate parameter suggestions.

Author: Generated with Claude Code
Date: 2025-10-15
"""

import scanpy as sc
from cn_dbscan import DBSCANCellularNeighborhoodDetector


def find_parameters():
    """
    Find optimal DBSCAN parameters for your data.
    """
    print("\n" + "=" * 80)
    print("FINDING OPTIMAL DBSCAN PARAMETERS")
    print("=" * 80)

    # Load your data
    input_file = '../tile_39520_7904.h5ad'
    print(f"\nLoading data from: {input_file}")
    adata = sc.read_h5ad(input_file)
    print(f"Loaded {adata.n_obs} cells")

    # Initialize detector
    detector = DBSCANCellularNeighborhoodDetector(adata)

    # Build graph and aggregate neighbors (required before parameter suggestion)
    print("\nPreparing data...")
    detector.build_knn_graph(k=20)
    detector.aggregate_neighbors(cluster_key='cell_type')

    # Get parameter suggestions
    print("\nAnalyzing neighborhood composition distances...")
    suggestions = detector.suggest_dbscan_parameters(
        k_neighbors=5,  # This will be your min_samples
        plot=True,
        save_path='dbscan_parameter_suggestion.png'
    )

    # Test the suggested parameters
    print("\n" + "=" * 80)
    print("TESTING SUGGESTED PARAMETERS")
    print("=" * 80)

    # Try balanced suggestion
    eps_balanced = suggestions['suggested_eps_balanced']
    min_samples = suggestions['suggested_min_samples']

    print(f"\nTesting: eps={eps_balanced:.4f}, min_samples={min_samples}")
    detector.detect_cn_dbscan(
        eps=eps_balanced,
        min_samples=min_samples,
        output_key='cn_test'
    )

    # Check results
    n_clusters = len(adata.obs['cn_test'].unique())
    cluster_sizes = adata.obs['cn_test'].value_counts().to_dict()

    print(f"\nResults with balanced parameters:")
    print(f"  - Number of clusters: {n_clusters}")
    print(f"  - Cluster sizes: {cluster_sizes}")

    # Give recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if n_clusters == 1:
        print("\n⚠️  Only 1 cluster found - eps is TOO LARGE")
        print(f"   Try decreasing eps to: {suggestions['suggested_eps_aggressive']:.4f}")
        print(f"   Or even smaller: {suggestions['distance_percentiles']['25%'] * 0.8:.4f}")
    elif n_clusters > 15:
        print("\n⚠️  Too many clusters found - eps is TOO SMALL")
        print(f"   Try increasing eps to: {suggestions['suggested_eps_conservative']:.4f}")
        print(f"   Or even larger: {suggestions['distance_percentiles']['75%'] * 1.2:.4f}")
    else:
        print("\n✓ Good number of clusters!")
        print(f"  Use these parameters in your analysis:")
        print(f"  - eps={eps_balanced:.4f}")
        print(f"  - min_samples={min_samples}")

    print("\nNext steps:")
    print("1. Review the k-distance plot saved to 'dbscan_parameter_suggestion.png'")
    print("2. Adjust eps based on the recommendations above")
    print("3. Run the full pipeline with your chosen parameters")

    return suggestions


if __name__ == '__main__':
    suggestions = find_parameters()