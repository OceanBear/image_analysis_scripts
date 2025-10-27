"""
Spatial Context Detection for Unified CN Results

This script processes the results from cn_unified_kmeans.py to detect spatial contexts (SCs).
It reads the processed h5ad files with CN annotations and performs spatial context analysis.

Key Features:
- Reads processed h5ad files from cn_unified_kmeans.py output
- Detects spatial contexts based on CN mixtures (k=40, threshold=0.9)
- Generates spatial context maps and interaction graphs
- Optimized for unified CN results

Author: Generated with Claude Code
Date: 2025-10-27
"""

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import squidpy as sq
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
from scipy.sparse import csr_matrix
import warnings
import glob
import time

# Set the working directory to the script's directory
os.chdir(Path(__file__).parent)
warnings.filterwarnings('ignore')


def discover_unified_cn_files(cn_results_dir: str) -> List[str]:
    """
    Discover processed h5ad files from cn_unified_kmeans.py results.
    
    Parameters:
    -----------
    cn_results_dir : str
        Path to cn_unified_kmeans.py results directory
        
    Returns:
    --------
    file_paths : List[str]
        List of paths to processed h5ad files
    """
    processed_h5ad_dir = Path(cn_results_dir) / 'processed_h5ad'
    
    if not processed_h5ad_dir.exists():
        print(f"Warning: Directory {processed_h5ad_dir} does not exist")
        return []
    
    # Find all h5ad files with CN annotations
    pattern = str(processed_h5ad_dir / '*_adata_cns.h5ad')
    file_paths = glob.glob(pattern)
    
    print(f"Found {len(file_paths)} processed h5ad files from unified CN detection:")
    for i, path in enumerate(file_paths, 1):
        filename = Path(path).name
        print(f"  {i}. {filename}")
    
    return sorted(file_paths)


def load_unified_cn_data(file_paths: List[str], coord_offset: bool = False) -> ad.AnnData:
    """
    Load and combine processed h5ad files from unified CN detection.
    
    Parameters:
    -----------
    file_paths : List[str]
        List of paths to h5ad files
    coord_offset : bool
        Whether to offset spatial coordinates to avoid overlap
        
    Returns:
    --------
    combined_adata : AnnData
        Combined AnnData object with all tiles
    """
    print(f"Loading and combining {len(file_paths)} tiles from unified CN results...")
    
    adata_list = []
    coord_offset_x = 0
    coord_offset_y = 0
    
    for i, file_path in enumerate(file_paths, 1):
        tile_name = Path(file_path).stem.replace('_adata_cns', '')
        print(f"  [{i}/{len(file_paths)}] Loading: {tile_name}")
        
        try:
            adata = sc.read_h5ad(file_path)
            
            # Add tile identifier if not present
            if 'tile_name' not in adata.obs.columns:
                adata.obs['tile_name'] = tile_name
            
            # Verify required columns
            if 'cn_celltype' not in adata.obs.columns:
                print(f"    Warning: No 'cn_celltype' column found in {file_path}")
                continue
            
            # Offset spatial coordinates if requested
            if coord_offset and 'spatial' in adata.obsm:
                coords = adata.obsm['spatial'].copy()
                coords[:, 0] += coord_offset_x
                coords[:, 1] += coord_offset_y
                adata.obsm['spatial'] = coords
                
                # Update offset for next tile
                max_x = coords[:, 0].max()
                coord_offset_x = max_x + 500  # 500 pixel gap between tiles
            
            adata_list.append(adata)
            print(f"    ✓ Loaded {adata.n_obs} cells, {adata.n_vars} genes")
            
        except Exception as e:
            print(f"    ✗ Error loading {file_path}: {str(e)}")
            continue
    
    if not adata_list:
        raise ValueError("No valid tiles could be loaded")
    
    # Combine all tiles
    print("\nCombining tiles into single dataset...")
    combined_adata = ad.concat(adata_list, join='outer', index_unique='-')
    
    print(f"✓ Combined dataset: {combined_adata.n_obs} cells, {combined_adata.n_vars} genes")
    print(f"  Tiles: {combined_adata.obs['tile_name'].nunique()}")
    print(f"  CNs: {combined_adata.obs['cn_celltype'].nunique()}")
    
    return combined_adata


class SpatialContextDetector:
    """
    Detects Spatial Contexts (SCs) based on cellular neighborhood (CN) composition
    in larger neighborhoods, similar to Figure 19 from the paper.

    Spatial Contexts represent regions where different CNs interact and are detected by:
    1. Building a larger k-NN graph (k=40) to capture broader spatial interactions
    2. Computing CN fraction composition in each cell's neighborhood
    3. Assigning minimal CN combinations that exceed a threshold (default: 0.9)
    4. Filtering rare SCs and visualizing spatial patterns

    Reference:
    Schürch et al. (2020) "Coordinated cellular neighborhoods orchestrate
    antitumoral immunity at the colorectal cancer invasive front"
    """

    def __init__(self, adata: ad.AnnData, cn_key: str = 'cn_celltype'):
        """
        Initialize SC detector.

        Parameters:
        -----------
        adata : AnnData
            AnnData object with spatial coordinates and CN annotations
        cn_key : str, default='cn_celltype'
            Key in adata.obs containing CN labels
        """
        self.adata = adata
        self.cn_key = cn_key
        self.sc_labels = None
        self.aggregated_cn_fractions = None
        self.sc_graph = None

        # Verify CN labels exist
        if cn_key not in adata.obs.columns:
            raise ValueError(f"CN key '{cn_key}' not found in adata.obs. "
                           f"Please run unified CN detection first.")

    def build_knn_graph(
        self,
        k: int = 40,
        coord_key: str = 'spatial',
        key_added: str = 'spatial_connectivities_sc'
    ):
        """
        Build k-nearest neighbor graph for spatial context detection.
        Uses a larger k than CN detection to capture broader spatial interactions.

        Parameters:
        -----------
        k : int, default=40
            Number of nearest neighbors (larger than CN detection)
        coord_key : str, default='spatial'
            Key in adata.obsm containing spatial coordinates
        key_added : str
            Key to store the connectivity matrix
        """
        print(f"Building {k}-nearest neighbor graph for SC detection...")

        # Build spatial neighbors graph
        sq.gr.spatial_neighbors(
            self.adata,
            spatial_key=coord_key,
            coord_type='generic',
            n_neighs=k,
            radius=None  # Use KNN, not radius
        )

        # Squidpy uses fixed key names, rename if needed
        actual_key = 'spatial_connectivities'
        if key_added != actual_key and actual_key in self.adata.obsp:
            self.adata.obsp[key_added] = self.adata.obsp[actual_key]

        # Print statistics
        connectivity = self.adata.obsp[key_added]
        avg_neighbors = connectivity.sum(axis=1).mean()
        print(f"  - Average neighbors per cell: {avg_neighbors:.2f}")
        print(f"  - Connectivity matrix shape: {connectivity.shape}")

        return self

    def aggregate_cn_fractions(
        self,
        connectivity_key: str = 'spatial_connectivities_sc',
        output_key: str = 'aggregated_cn_fractions'
    ):
        """
        For each cell, compute the fraction of each CN type in its neighborhood.
        This is analogous to aggregating cell types for CN detection, but now
        we aggregate CN labels instead.

        Parameters:
        -----------
        connectivity_key : str
            Key in adata.obsp containing spatial connectivity
        output_key : str
            Key to store aggregated CN fractions
        """
        print(f"Aggregating CN fractions in neighborhoods...")

        # Get CN labels
        cn_labels = self.adata.obs[self.cn_key].values
        unique_cns = sorted(self.adata.obs[self.cn_key].cat.categories)

        # Get connectivity matrix
        connectivity = self.adata.obsp[connectivity_key]

        # Initialize aggregated CN fractions matrix
        n_cells = self.adata.n_obs
        n_cns = len(unique_cns)
        aggregated = np.zeros((n_cells, n_cns))

        print(f"  Processing {n_cells} cells...")

        # For each cell, compute CN fractions in neighborhood
        for i in range(n_cells):
            if i % 10000 == 0 and i > 0:
                print(f"    Processed {i:,}/{n_cells:,} cells ({100*i/n_cells:.1f}%)")
            
            # Get neighbors (including self)
            neighbors_mask = connectivity[i].toarray().flatten() > 0

            if neighbors_mask.sum() > 0:
                # Get CN labels of neighbors
                neighbor_cns = cn_labels[neighbors_mask]

                # Compute fractions for each CN
                for j, cn in enumerate(unique_cns):
                    aggregated[i, j] = (neighbor_cns == cn).sum() / neighbors_mask.sum()

        # Store in adata
        self.aggregated_cn_fractions = pd.DataFrame(
            aggregated,
            columns=[f'CN_{cn}' for cn in unique_cns],
            index=self.adata.obs_names
        )

        self.adata.obsm[output_key] = aggregated

        print(f"  - Aggregated CN fractions shape: {aggregated.shape}")
        print(f"  - CN types: {unique_cns}")

        return self

    def detect_spatial_contexts(
        self,
        threshold: float = 0.9,
        min_fraction: float = 0.05,
        aggregated_key: str = 'aggregated_cn_fractions',
        output_key: str = 'spatial_context'
    ):
        """
        Assign spatial context labels based on minimal CN combinations that
        exceed the threshold.

        Algorithm:
        1. For each cell, sort CN fractions from high to low
        2. Add CNs cumulatively until sum >= threshold
        3. Only include CNs with fraction >= min_fraction
        4. SC label = sorted CN IDs joined by "_" (e.g., "1_2", "2_4_5")

        Parameters:
        -----------
        threshold : float, default=0.9
            Cumulative fraction threshold for SC assignment
        min_fraction : float, default=0.05
            Minimum individual CN fraction to be included (filters out noise)
        aggregated_key : str
            Key in adata.obsm containing aggregated CN fractions
        output_key : str
            Key to store SC labels in adata.obs
        """
        print(f"Detecting spatial contexts (threshold={threshold}, min_fraction={min_fraction})...")

        # Get aggregated CN fractions
        aggregated = self.adata.obsm[aggregated_key]
        unique_cns = sorted(self.adata.obs[self.cn_key].cat.categories)

        # Initialize SC labels
        sc_labels = []

        # For each cell, determine minimal CN combination
        for i in range(len(aggregated)):
            cn_fractions = aggregated[i]

            # Sort CNs by fraction (high to low)
            sorted_indices = np.argsort(cn_fractions)[::-1]
            sorted_cns = [unique_cns[idx] for idx in sorted_indices]
            sorted_fractions = cn_fractions[sorted_indices]

            # Add CNs until cumulative sum >= threshold
            cumsum = 0.0
            selected_cns = []

            for cn, frac in zip(sorted_cns, sorted_fractions):
                # Skip CNs with very small fractions (noise filtering)
                if frac < min_fraction:
                    break
                
                cumsum += frac
                selected_cns.append(str(cn))

                # Stop when we reach the threshold
                if cumsum >= threshold:
                    break

            # If no CNs meet the criteria, use the top CN
            if not selected_cns:
                selected_cns.append(str(sorted_cns[0]))

            # Create SC label by joining CN IDs (sorted by fraction)
            sc_label = '_'.join(selected_cns)
            sc_labels.append(sc_label)

        # Store SC labels
        self.sc_labels = pd.Categorical(sc_labels)
        self.adata.obs[output_key] = self.sc_labels

        # Print SC statistics
        sc_counts = pd.Series(sc_labels).value_counts()
        print(f"  - Detected {len(sc_counts)} unique spatial contexts")
        print(f"  - Top 10 SCs by cell count:")
        for sc, count in sc_counts.head(10).items():
            print(f"    SC '{sc}': {count} cells")

        return self

    def filter_spatial_contexts(
        self,
        min_cells: int = 100,
        min_groups: Optional[int] = None,
        group_key: Optional[str] = None,
        sc_key: str = 'spatial_context',
        filtered_key: str = 'spatial_context_filtered'
    ):
        """
        Filter out rare spatial contexts based on cell count and group occurrence.

        Parameters:
        -----------
        min_cells : int, default=100
            Minimum number of cells for an SC to be retained
        min_groups : int, optional
            Minimum number of groups (e.g., tiles) an SC must appear in
        group_key : str, optional
            Key in adata.obs containing group identifiers (e.g., 'tile_name')
        sc_key : str
            Key in adata.obs containing SC labels
        filtered_key : str
            Key to store filtered SC labels
        """
        print(f"Filtering spatial contexts...")
        print(f"  - Minimum cells: {min_cells}")
        if min_groups is not None and group_key is not None:
            print(f"  - Minimum groups ({group_key}): {min_groups}")

        sc_labels = self.adata.obs[sc_key].values
        sc_counts = pd.Series(sc_labels).value_counts()

        # Filter by cell count
        valid_scs = set(sc_counts[sc_counts >= min_cells].index)
        print(f"  - SCs passing cell count filter: {len(valid_scs)}")

        # Filter by group occurrence if specified
        if min_groups is not None and group_key is not None:
            if group_key not in self.adata.obs.columns:
                print(f"  WARNING: group_key '{group_key}' not found, skipping group filter")
            else:
                # Count groups for each SC
                sc_group_counts = self.adata.obs.groupby(sc_key)[group_key].nunique()
                valid_scs = set(sc_group_counts[sc_group_counts >= min_groups].index)
                print(f"  - SCs passing group filter: {len(valid_scs)}")

        # Create filtered labels (set rare SCs to 'Other')
        filtered_labels = [sc if sc in valid_scs else 'Other' for sc in sc_labels]
        self.adata.obs[filtered_key] = pd.Categorical(filtered_labels)

        n_other = sum([1 for sc in filtered_labels if sc == 'Other'])
        print(f"  - Cells assigned to 'Other': {n_other}")
        print(f"  - Final number of SCs: {len(valid_scs)}")

        return self

    def visualize_spatial_contexts(
        self,
        sc_key: str = 'spatial_context_filtered',
        img_id_key: str = 'tile_name',
        coord_key: str = 'spatial',
        point_size: float = 2.0,
        palette: str = 'tab20',
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize spatial contexts spatially (similar to Fig 19a).

        Parameters:
        -----------
        sc_key : str
            Key in adata.obs containing SC labels
        img_id_key : str
            Key in adata.obs containing image/sample identifiers
        coord_key : str
            Key in adata.obsm containing spatial coordinates
        point_size : float
            Size of points in scatter plot
        palette : str
            Color palette name
        figsize : tuple, optional
            Figure size (width, height)
        save_path : str, optional
            Path to save figure
        """
        print("Visualizing spatial contexts...")

        # Get unique images
        images = self.adata.obs[img_id_key].unique()
        n_images = len(images)

        # Calculate grid dimensions
        n_cols = min(4, n_images)
        n_rows = int(np.ceil(n_images / n_cols))

        if figsize is None:
            figsize = (n_cols * 4, n_rows * 4)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        if n_images == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Get coordinates
        coords = self.adata.obsm[coord_key]

        # Get SC labels
        sc_labels = self.adata.obs[sc_key].values
        unique_scs = sorted([sc for sc in self.adata.obs[sc_key].cat.categories if sc != 'Other'])

        # Add 'Other' to end if it exists
        if 'Other' in self.adata.obs[sc_key].cat.categories:
            unique_scs.append('Other')

        # Get colors
        n_scs = len(unique_scs)
        if n_scs <= 20:
            colors_palette = sns.color_palette(palette, n_scs)
        else:
            colors_palette = sns.color_palette('husl', n_scs)

        color_map = {sc: colors_palette[i] for i, sc in enumerate(unique_scs)}

        # Plot each image
        for idx, img in enumerate(images):
            ax = axes[idx]

            # Get cells from this image
            mask = self.adata.obs[img_id_key] == img
            img_coords = coords[mask]
            img_scs = sc_labels[mask]

            # Plot each SC
            for sc in unique_scs:
                sc_mask = img_scs == sc
                if sc_mask.sum() > 0:
                    ax.scatter(
                        img_coords[sc_mask, 0],
                        img_coords[sc_mask, 1],
                        c=[color_map[sc]],
                        s=point_size,
                        alpha=0.7,
                        label=f'SC {sc}'
                    )

            ax.set_title(img, fontsize=10)
            ax.set_xlabel('X coordinate (pixels)')
            ax.set_ylabel('Y coordinate (pixels)')
            ax.set_aspect('equal')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Remove empty subplots
        for idx in range(n_images, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  - Saved to: {save_path}")

        plt.close()

        return fig

    def build_sc_interaction_graph(
        self,
        sc_key: str = 'spatial_context_filtered',
        connectivity_key: str = 'spatial_connectivities_sc'
    ) -> nx.Graph:
        """
        Build interaction graph between spatial contexts.
        Nodes = SCs, edges = SCs that share neighbors in the k-NN graph.

        Parameters:
        -----------
        sc_key : str
            Key in adata.obs containing SC labels
        connectivity_key : str
            Key in adata.obsp containing spatial connectivity

        Returns:
        --------
        G : networkx.Graph
            SC interaction graph
        """
        print("Building SC interaction graph...")

        # Get SC labels and connectivity
        sc_labels = self.adata.obs[sc_key].values
        connectivity = self.adata.obsp[connectivity_key]

        # Initialize graph
        G = nx.Graph()

        # Count cells in each SC
        sc_counts = pd.Series(sc_labels).value_counts().to_dict()

        # Add nodes (SCs) with cell counts
        for sc, count in sc_counts.items():
            if sc != 'Other':  # Skip 'Other' category
                G.add_node(sc, size=count)

        # Count interactions between SCs
        interaction_counts = {}

        for i in range(len(sc_labels)):
            sc_i = sc_labels[i]
            if sc_i == 'Other':
                continue

            # Get neighbors
            neighbors = connectivity[i].toarray().flatten() > 0
            neighbor_scs = sc_labels[neighbors]

            # Count interactions with other SCs
            for sc_j in neighbor_scs:
                if sc_j == 'Other' or sc_j == sc_i:
                    continue

                # Create edge key (sorted to avoid duplicates)
                edge = tuple(sorted([sc_i, sc_j]))

                if edge not in interaction_counts:
                    interaction_counts[edge] = 0
                interaction_counts[edge] += 1

        # Add edges with weights
        for (sc1, sc2), count in interaction_counts.items():
            G.add_edge(sc1, sc2, weight=count)

        self.sc_graph = G

        print(f"  - Graph nodes (SCs): {G.number_of_nodes()}")
        print(f"  - Graph edges (interactions): {G.number_of_edges()}")

        return G

    def plot_sc_graph(
        self,
        sc_key: str = 'spatial_context_filtered',
        layout: str = 'spring',
        figsize: Tuple[int, int] = (12, 10),
        node_size_scale: float = 0.5,
        edge_width_scale: float = 0.01,
        save_path: Optional[str] = None
    ):
        """
        Visualize SC interaction graph (similar to Fig 19b).

        Parameters:
        -----------
        sc_key : str
            Key in adata.obs containing SC labels
        layout : str, default='spring'
            Graph layout algorithm ('spring', 'kamada_kawai', or 'circular')
        figsize : tuple
            Figure size (width, height)
        node_size_scale : float
            Scaling factor for node sizes
        edge_width_scale : float
            Scaling factor for edge widths
        save_path : str, optional
            Path to save figure
        """
        print("Visualizing SC interaction graph...")

        # Build graph if not already done
        if self.sc_graph is None:
            self.build_sc_interaction_graph(sc_key=sc_key)

        G = self.sc_graph

        # Compute layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            raise ValueError(f"Unknown layout: {layout}")

        # Prepare node sizes based on cell counts
        node_sizes = [G.nodes[node]['size'] * node_size_scale for node in G.nodes()]

        # Prepare edge widths based on interaction counts
        edge_widths = [G.edges[edge]['weight'] * edge_width_scale for edge in G.edges()]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Draw graph
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color='lightblue',
            alpha=0.7,
            ax=ax
        )

        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            alpha=0.5,
            edge_color='gray',
            ax=ax
        )

        nx.draw_networkx_labels(
            G, pos,
            font_size=9,
            font_weight='bold',
            ax=ax
        )

        ax.set_title('Spatial Context Interaction Network',
                     fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')

        # Add legend for node/edge meaning
        legend_text = (
            f"Nodes: Spatial Contexts (size ∝ cell count)\n"
            f"Edges: Shared neighborhoods (width ∝ interaction frequency)\n"
            f"Total SCs: {G.number_of_nodes()}, Interactions: {G.number_of_edges()}"
        )
        ax.text(0.02, 0.98, legend_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  - Saved to: {save_path}")

        plt.close()

        return fig

    def compute_sc_statistics(
        self,
        sc_key: str = 'spatial_context_filtered',
        cn_key: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compute statistics for each spatial context.

        Parameters:
        -----------
        sc_key : str
            Key in adata.obs containing SC labels
        cn_key : str, optional
            Key in adata.obs containing CN labels (for composition analysis)

        Returns:
        --------
        stats_df : DataFrame
            Statistics for each SC
        """
        print("Computing SC statistics...")

        if cn_key is None:
            cn_key = self.cn_key

        # Get SC labels
        sc_labels = self.adata.obs[sc_key]
        unique_scs = [sc for sc in sc_labels.cat.categories if sc != 'Other']

        stats = []

        for sc in unique_scs:
            sc_mask = sc_labels == sc
            n_cells = sc_mask.sum()

            # CN composition
            cn_composition = self.adata.obs[cn_key][sc_mask].value_counts()
            cn_composition_str = ', '.join([f'CN{cn}:{count}'
                                           for cn, count in cn_composition.items()])

            # Dominant CN
            dominant_cn = cn_composition.idxmax() if len(cn_composition) > 0 else None

            stats.append({
                'SC': sc,
                'n_cells': n_cells,
                'dominant_CN': dominant_cn,
                'CN_composition': cn_composition_str,
                'n_CNs': len([x for x in sc.split('_')])
            })

        stats_df = pd.DataFrame(stats).sort_values('n_cells', ascending=False)

        print(f"  - Computed statistics for {len(stats_df)} SCs")

        return stats_df

    def run_full_pipeline(
        self,
        k: int = 40,
        threshold: float = 0.9,
        min_fraction: float = 0.05,
        min_cells: int = 100,
        min_groups: Optional[int] = None,
        group_key: Optional[str] = None,
        img_id_key: str = 'tile_name',
        output_dir: str = 'spatial_contexts',
        graph_layout: str = 'spring'
    ):
        """
        Run the complete spatial context detection pipeline.

        Parameters:
        -----------
        k : int, default=40
            Number of nearest neighbors for SC detection
        threshold : float, default=0.9
            Cumulative CN fraction threshold for SC assignment
        min_fraction : float, default=0.05
            Minimum individual CN fraction to be included
        min_cells : int, default=100
            Minimum cells for SC to be retained
        min_groups : int, optional
            Minimum groups (e.g., tiles) for SC to be retained
        group_key : str, optional
            Key in adata.obs for group identifiers
        img_id_key : str
            Key in adata.obs containing image identifiers
        output_dir : str
            Directory to save results
        graph_layout : str
            Layout algorithm for SC graph ('spring', 'kamada_kawai', 'circular')
        """
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 60)
        print("SPATIAL CONTEXT DETECTION PIPELINE")
        print("=" * 60)

        # Step 1: Build larger KNN graph
        self.build_knn_graph(k=k)

        # Step 2: Aggregate CN fractions
        self.aggregate_cn_fractions()

        # Step 3: Detect spatial contexts
        self.detect_spatial_contexts(threshold=threshold, min_fraction=min_fraction)

        # Step 4: Filter rare SCs
        self.filter_spatial_contexts(
            min_cells=min_cells,
            min_groups=min_groups,
            group_key=group_key
        )

        # Step 5: Compute statistics
        stats_df = self.compute_sc_statistics()
        stats_df.to_csv(f'{output_dir}/sc_statistics.csv', index=False)
        print(f"\n  - Saved statistics to: {output_dir}/sc_statistics.csv")

        # Step 6: Visualizations
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)

        # Spatial visualization
        self.visualize_spatial_contexts(
            img_id_key=img_id_key,
            save_path=f'{output_dir}/spatial_contexts.png'
        )

        # SC interaction graph
        self.build_sc_interaction_graph()
        self.plot_sc_graph(
            layout=graph_layout,
            save_path=f'{output_dir}/sc_interaction_graph.png'
        )

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"Results saved to: {output_dir}/")

        return self


def main():
    """
    Main function to run spatial context detection on unified CN results.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Spatial Context Detection for Unified CN Results'
    )
    parser.add_argument(
        '--cn_results_dir', '-c',
        default='cn_unified_results',
        help='Directory containing cn_unified_kmeans.py results'
    )
    parser.add_argument(
        '--output_dir', '-o',
        default='sc_unified_results',
        help='Output directory for spatial context results'
    )
    parser.add_argument(
        '--k', type=int, default=40,
        help='Number of nearest neighbors for SC detection (default: 40)'
    )
    parser.add_argument(
        '--threshold', '-t', type=float, default=0.6,
        help='Cumulative CN fraction threshold for SC assignment (default: 0.9)'
    )
    parser.add_argument(
        '--min_fraction', '-f', type=float, default=0.05,
        help='Minimum individual CN fraction to be included (default: 0.05)'
    )
    parser.add_argument(
        '--min_cells', '-m', type=int, default=100,
        help='Minimum cells for SC to be retained (default: 100)'
    )
    parser.add_argument(
        '--min_groups', type=int, default=None,
        help='Minimum number of tiles an SC must appear in (default: None)'
    )
    parser.add_argument(
        '--coord_offset', action='store_true',
        help='Offset spatial coordinates between tiles for visualization'
    )
    parser.add_argument(
        '--graph_layout', '-g',
        choices=['spring', 'kamada_kawai', 'circular'],
        default='spring',
        help='Layout algorithm for SC interaction graph (default: spring)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SPATIAL CONTEXT DETECTION FOR UNIFIED CN RESULTS")
    print("=" * 80)
    print(f"CN results directory: {args.cn_results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Parameters: k={args.k}, threshold={args.threshold}, min_fraction={args.min_fraction}, min_cells={args.min_cells}")
    if args.min_groups:
        print(f"Min groups: {args.min_groups}")
    print("=" * 80)

    try:
        # Discover processed files
        file_paths = discover_unified_cn_files(args.cn_results_dir)
        
        if not file_paths:
            print("No processed h5ad files found! Please run cn_unified_kmeans.py first.")
            return

        # Load and combine data
        combined_adata = load_unified_cn_data(file_paths, coord_offset=args.coord_offset)

        # Initialize detector
        detector = SpatialContextDetector(combined_adata, cn_key='cn_celltype')

        # Run full pipeline
        detector.run_full_pipeline(
            k=args.k,
            threshold=args.threshold,
            min_fraction=args.min_fraction,
            min_cells=args.min_cells,
            min_groups=args.min_groups,
            group_key='tile_name' if args.min_groups else None,
            img_id_key='tile_name',
            output_dir=args.output_dir,
            graph_layout=args.graph_layout
        )

        print(f"\nSpatial context detection completed successfully!")
        print(f"Check the results in: {args.output_dir}/")

    except Exception as e:
        print(f"Error in spatial context detection: {str(e)}")
        return


if __name__ == '__main__':
    main()
