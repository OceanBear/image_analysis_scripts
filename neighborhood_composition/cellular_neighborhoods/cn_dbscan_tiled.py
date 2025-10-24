"""
DBSCAN-based Cellular Neighborhood Detection

This module extends the CellularNeighborhoodDetector class with DBSCAN clustering
as an alternative to K-means for detecting cellular neighborhoods.

Author: Generated with Claude Code
Date: 2025-10-15
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from scipy.spatial import distance_matrix
from typing import Optional
from cn_kmeans_tiled import CellularNeighborhoodDetector
import os
from pathlib import Path
# Set the working directory to the script's directory
os.chdir(Path(__file__).parent)

class DBSCANCellularNeighborhoodDetector(CellularNeighborhoodDetector):
    """
    Extends CellularNeighborhoodDetector with DBSCAN clustering capabilities.

    This class adds DBSCAN-based methods while maintaining all original K-means
    functionality from the parent class.
    """

    def detect_cn_dbscan(
            self,
            eps: float = 0.5,
            min_samples: int = 5,
            aggregated_key: str = 'aggregated_neighbors',
            output_key: str = 'cn_celltype_dbscan',
            handle_noise: str = 'separate'
    ):
        """
        Cluster cells based on their neighborhood composition using DBSCAN.

        DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
        is an alternative to K-means that:
        - Automatically determines the number of clusters
        - Can identify arbitrarily shaped clusters
        - Identifies outliers/noise points
        - Does not require specifying number of clusters beforehand

        Parameters:
        -----------
        eps : float, default=0.5
            The maximum distance between two samples for one to be considered
            in the neighborhood of the other. This is the most important DBSCAN
            parameter. Smaller values create more clusters, larger values create
            fewer clusters.
        min_samples : int, default=5
            The number of samples in a neighborhood for a point to be considered
            as a core point. Larger values result in denser clusters.
        aggregated_key : str, default='aggregated_neighbors'
            Key in adata.obsm containing aggregated neighbor fractions
        output_key : str, default='cn_celltype_dbscan'
            Key to store CN labels in adata.obs
        handle_noise : str, default='separate'
            How to handle noise points (labeled as -1 by DBSCAN):
            - 'separate': Keep noise points as a separate cluster (label 1)
            - 'nearest': Assign noise points to nearest cluster centroid

        Returns:
        --------
        self : DBSCANCellularNeighborhoodDetector
            Returns self for method chaining
        """
        print(f"Detecting cellular neighborhoods using DBSCAN...")
        print(f"  - eps (neighborhood radius): {eps}")
        print(f"  - min_samples: {min_samples}")
        print(f"  - noise handling: {handle_noise}")

        # Get aggregated neighbor fractions
        aggregated = self.adata.obsm[aggregated_key]

        # Perform DBSCAN clustering
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='euclidean'
        )

        cn_labels = dbscan.fit_predict(aggregated)

        # Handle noise points (labeled as -1)
        n_noise = (cn_labels == -1).sum()
        n_clusters_found = len(set(cn_labels)) - (1 if -1 in cn_labels else 0)

        print(f"  - Clusters found: {n_clusters_found}")
        print(f"  - Noise points: {n_noise} ({100*n_noise/len(cn_labels):.2f}%)")

        if n_noise > 0:
            if handle_noise == 'separate':
                # Assign noise points to CN 1, shift other labels to start from 2
                # (to match 1-based indexing like k-means)
                cn_labels_adjusted = np.where(cn_labels == -1, 1, cn_labels + 2)
                print(f"  - Noise points assigned to CN 1 (separate cluster)")
            elif handle_noise == 'nearest':
                # Compute cluster centroids
                unique_labels = set(cn_labels)
                unique_labels.discard(-1)  # Remove noise label

                centroids = np.array([
                    aggregated[cn_labels == label].mean(axis=0)
                    for label in sorted(unique_labels)
                ])

                # Assign noise points to nearest centroid
                noise_mask = cn_labels == -1
                noise_points = aggregated[noise_mask]

                if len(noise_points) > 0:
                    # Compute distances to all centroids
                    distances = distance_matrix(noise_points, centroids)
                    nearest_cluster = distances.argmin(axis=1)

                    # Assign noise points to nearest cluster
                    cn_labels_adjusted = cn_labels.copy()
                    noise_indices = np.where(noise_mask)[0]
                    for i, cluster in enumerate(nearest_cluster):
                        cn_labels_adjusted[noise_indices[i]] = cluster

                    # Shift to 1-based indexing
                    cn_labels_adjusted = cn_labels_adjusted + 1
                    print(f"  - Noise points assigned to nearest cluster centroids")
            else:
                raise ValueError(f"Unknown handle_noise option: {handle_noise}. "
                                 "Choose 'separate' or 'nearest'")
        else:
            # No noise points, just shift to 1-based indexing
            cn_labels_adjusted = cn_labels + 1

        # Store CN labels
        self.cn_labels = cn_labels_adjusted
        self.adata.obs[output_key] = pd.Categorical(self.cn_labels)

        # Print CN sizes
        cn_counts = pd.Series(self.cn_labels).value_counts().sort_index()
        print(f"  - CN sizes:")
        for cn, count in cn_counts.items():
            label = f"{cn} (Noise)" if cn == 1 and n_noise > 0 and handle_noise == 'separate' else cn
            print(f"    CN {label}: {count} cells")

        return self

    def run_full_pipeline_dbscan(
            self,
            k: int = 20,
            eps: float = 0.5,
            min_samples: int = 5,
            handle_noise: str = 'separate',
            celltype_key: str = 'celltype',
            img_id_key: str = 'sample_id',
            output_dir: str = 'cn_results_dbscan',
            output_key: str = 'cn_celltype_dbscan'
    ):
        """
        Run the complete CN detection pipeline using DBSCAN.

        Parameters:
        -----------
        k : int, default=20
            Number of nearest neighbors for graph construction
        eps : float, default=0.5
            DBSCAN epsilon parameter (neighborhood radius)
        min_samples : int, default=5
            DBSCAN minimum samples parameter
        handle_noise : str, default='separate'
            How to handle noise points ('separate' or 'nearest')
        celltype_key : str
            Key in adata.obs containing cell type labels
        img_id_key : str
            Key in adata.obs containing image identifiers
        output_dir : str
            Directory to save results
        output_key : str
            Key to store DBSCAN CN labels
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 60)
        print("CELLULAR NEIGHBORHOOD DETECTION PIPELINE (DBSCAN)")
        print("=" * 60)

        # Step 1: Build KNN graph
        self.build_knn_graph(k=k)

        # Step 2: Aggregate neighbors
        self.aggregate_neighbors(cluster_key=celltype_key)

        # Step 3: Detect CNs using DBSCAN
        self.detect_cn_dbscan(
            eps=eps,
            min_samples=min_samples,
            handle_noise=handle_noise,
            output_key=output_key
        )

        # Step 4: Visualize spatial CNs
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)

        self.visualize_spatial_cns(
            cn_key=output_key,
            img_id_key=img_id_key,
            save_path=f'{output_dir}/spatial_cns_dbscan_{eps}_{min_samples}.png',
            k=k,
            eps=eps,
            min_samples=min_samples
        )

        # Step 5: Visualize CN composition
        fig, composition, composition_zscore = self.visualize_cn_composition(
            cn_key=output_key,
            celltype_key=celltype_key,
            save_path=f'{output_dir}/cn_composition_heatmap_dbscan_{eps}_{min_samples}.png',
            k=k,
            eps=eps,
            min_samples=min_samples,
            show_values=True
        )

        # Step 6: Save results
        composition.to_csv(f'{output_dir}/cn_composition_dbscan_{eps}_{min_samples}.csv')
        composition_zscore.to_csv(f'{output_dir}/cn_composition_zscore_dbscan_{eps}_{min_samples}.csv')

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"Results saved to: {output_dir}/")

        return self

    def suggest_dbscan_parameters(
            self,
            aggregated_key: str = 'aggregated_neighbors',
            k_neighbors: int = 5,
            plot: bool = True,
            save_path: Optional[str] = None
    ):
        """
        Suggest DBSCAN parameters by analyzing the k-nearest neighbor distances.

        This method helps you find appropriate eps and min_samples values by:
        1. Computing k-nearest neighbor distances in the aggregated neighbor space
        2. Plotting the sorted distances (k-distance plot)
        3. Suggesting eps as the "elbow point" in the k-distance curve

        Parameters:
        -----------
        aggregated_key : str, default='aggregated_neighbors'
            Key in adata.obsm containing aggregated neighbor fractions
        k_neighbors : int, default=5
            Number of neighbors to use for distance calculation.
            This should match your intended min_samples parameter.
        plot : bool, default=True
            Whether to create a k-distance plot
        save_path : str, optional
            Path to save the plot

        Returns:
        --------
        suggestions : dict
            Dictionary with suggested parameters and distance statistics
        """
        print("=" * 60)
        print("DBSCAN PARAMETER SUGGESTION")
        print("=" * 60)

        # Get aggregated neighbor fractions
        if aggregated_key not in self.adata.obsm:
            print(f"Error: '{aggregated_key}' not found in adata.obsm")
            print("Please run build_knn_graph() and aggregate_neighbors() first")
            return None

        aggregated = self.adata.obsm[aggregated_key]
        n_samples = aggregated.shape[0]

        print(f"\nAnalyzing {n_samples} cells...")
        print(f"Computing {k_neighbors}-nearest neighbor distances...")

        # Compute pairwise distances
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=k_neighbors + 1)
        neighbors.fit(aggregated)
        distances, _ = neighbors.kneighbors(aggregated)

        # Get k-nearest neighbor distance (exclude self at index 0)
        k_distances = distances[:, k_neighbors].copy()
        k_distances_sorted = np.sort(k_distances)

        # Calculate statistics
        suggestions = {
            'k_neighbors': k_neighbors,
            'distance_min': k_distances.min(),
            'distance_max': k_distances.max(),
            'distance_mean': k_distances.mean(),
            'distance_median': np.median(k_distances),
            'distance_std': k_distances.std(),
            'distance_percentiles': {
                '25%': np.percentile(k_distances, 25),
                '50%': np.percentile(k_distances, 50),
                '75%': np.percentile(k_distances, 75),
                '90%': np.percentile(k_distances, 90),
                '95%': np.percentile(k_distances, 95),
            }
        }

        # Suggest eps values
        suggested_eps_conservative = suggestions['distance_percentiles']['75%']
        suggested_eps_balanced = suggestions['distance_percentiles']['50%']
        suggested_eps_aggressive = suggestions['distance_percentiles']['25%']

        suggestions['suggested_eps_conservative'] = suggested_eps_conservative
        suggestions['suggested_eps_balanced'] = suggested_eps_balanced
        suggestions['suggested_eps_aggressive'] = suggested_eps_aggressive
        suggestions['suggested_min_samples'] = k_neighbors

        # Print suggestions
        print("\n" + "=" * 60)
        print("DISTANCE STATISTICS")
        print("=" * 60)
        print(f"Min distance:    {suggestions['distance_min']:.4f}")
        print(f"Mean distance:   {suggestions['distance_mean']:.4f}")
        print(f"Median distance: {suggestions['distance_median']:.4f}")
        print(f"Max distance:    {suggestions['distance_max']:.4f}")
        print(f"\nPercentiles:")
        for pct, val in suggestions['distance_percentiles'].items():
            print(f"  {pct:>4}: {val:.4f}")

        print("\n" + "=" * 60)
        print("SUGGESTED PARAMETERS")
        print("=" * 60)
        print(f"\n1. Conservative (fewer, larger clusters):")
        print(f"   eps={suggested_eps_conservative:.4f}, min_samples={k_neighbors}")

        print(f"\n2. Balanced (moderate number of clusters):")
        print(f"   eps={suggested_eps_balanced:.4f}, min_samples={k_neighbors}")

        print(f"\n3. Aggressive (more, smaller clusters):")
        print(f"   eps={suggested_eps_aggressive:.4f}, min_samples={k_neighbors}")

        print(f"\nNote: Start with the balanced suggestion and adjust based on results.")
        print(f"      If you get only 1 cluster, decrease eps.")
        print(f"      If you get too many clusters or noise, increase eps.")

        # Plot k-distance graph
        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(range(len(k_distances_sorted)), k_distances_sorted, 'b-', linewidth=1)

            # Mark suggested eps values
            ax.axhline(y=suggested_eps_conservative, color='green', linestyle='--',
                      label=f'Conservative eps={suggested_eps_conservative:.4f}', alpha=0.7)
            ax.axhline(y=suggested_eps_balanced, color='orange', linestyle='--',
                      label=f'Balanced eps={suggested_eps_balanced:.4f}', alpha=0.7)
            ax.axhline(y=suggested_eps_aggressive, color='red', linestyle='--',
                      label=f'Aggressive eps={suggested_eps_aggressive:.4f}', alpha=0.7)

            ax.set_xlabel('Points sorted by distance', fontsize=12)
            ax.set_ylabel(f'{k_neighbors}-nearest neighbor distance', fontsize=12)
            ax.set_title('K-distance Plot for DBSCAN Parameter Selection\n'
                        'The "elbow point" suggests a good eps value',
                        fontsize=14, fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"\nPlot saved to: {save_path}")

            plt.show()

        return suggestions