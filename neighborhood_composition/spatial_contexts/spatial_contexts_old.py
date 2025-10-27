# spatial_contexts.py
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import squidpy as sq
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
from scipy.sparse import csr_matrix
import warnings
import glob
matplotlib.use('Agg')  # Use non-interactive backend

from pathlib import Path
# Set the working directory to the script's directory
os.chdir(Path(__file__).parent)
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore')


def discover_processed_files(batch_results_dir: str) -> List[str]:
    """
    Discover all processed h5ad files with CN annotations.
    
    Parameters:
    -----------
    batch_results_dir : str
        Path to batch results directory containing processed_h5ad subdirectory
        
    Returns:
    --------
    file_paths : List[str]
        List of paths to processed h5ad files
    """
    processed_h5ad_dir = Path(batch_results_dir) / 'processed_h5ad'
    
    if not processed_h5ad_dir.exists():
        print(f"Warning: Directory {processed_h5ad_dir} does not exist")
        return []
    
    # Find all h5ad files with CN annotations
    pattern = str(processed_h5ad_dir / '*_adata_cns.h5ad')
    file_paths = glob.glob(pattern)
    
    print(f"Found {len(file_paths)} processed h5ad files:")
    for i, path in enumerate(file_paths, 1):
        filename = Path(path).name
        print(f"  {i}. {filename}")
    
    return sorted(file_paths)


def load_multiple_tiles_individual(file_paths: List[str]) -> List[ad.AnnData]:
    """
    Load multiple tiles as individual AnnData objects.
    
    Parameters:
    -----------
    file_paths : List[str]
        List of paths to h5ad files
        
    Returns:
    --------
    adata_list : List[AnnData]
        List of AnnData objects, one per tile
    """
    print(f"Loading {len(file_paths)} tiles individually...")
    
    adata_list = []
    for i, file_path in enumerate(file_paths, 1):
        print(f"  Loading tile {i}/{len(file_paths)}: {Path(file_path).name}")
        
        try:
            adata = sc.read_h5ad(file_path)
            
            # Add tile identifier if not present
            if 'tile_name' not in adata.obs.columns:
                tile_name = Path(file_path).stem.replace('_adata_cns', '')
                adata.obs['tile_name'] = tile_name
            
            # Verify required columns
            if 'cn_celltype' not in adata.obs.columns:
                print(f"    Warning: No 'cn_celltype' column found in {file_path}")
                continue
                
            adata_list.append(adata)
            print(f"    - Loaded {adata.n_obs} cells, {adata.n_vars} genes")
            
        except Exception as e:
            print(f"    Error loading {file_path}: {str(e)}")
            continue
    
    print(f"Successfully loaded {len(adata_list)} tiles")
    return adata_list


def load_multiple_tiles_aggregated(file_paths: List[str], 
                                  coord_offset: bool = True) -> ad.AnnData:
    """
    Load multiple tiles and combine into a single AnnData object.
    
    Parameters:
    -----------
    file_paths : List[str]
        List of paths to h5ad files
    coord_offset : bool, default=True
        Whether to offset spatial coordinates to avoid overlap
        
    Returns:
    --------
    combined_adata : AnnData
        Combined AnnData object with all tiles
    """
    print(f"Loading and combining {len(file_paths)} tiles...")
    
    adata_list = []
    coord_offset_x = 0
    coord_offset_y = 0
    
    for i, file_path in enumerate(file_paths, 1):
        print(f"  Loading tile {i}/{len(file_paths)}: {Path(file_path).name}")
        
        try:
            adata = sc.read_h5ad(file_path)
            
            # Add tile identifier
            tile_name = Path(file_path).stem.replace('_adata_cns', '')
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
                if coord_offset:
                    coord_offset_x += coords[:, 0].max() + 100  # 100 pixel gap
                    coord_offset_y = 0  # Reset Y offset for now
            
            adata_list.append(adata)
            print(f"    - Loaded {adata.n_obs} cells, {adata.n_vars} genes")
            
        except Exception as e:
            print(f"    Error loading {file_path}: {str(e)}")
            continue
    
    if not adata_list:
        raise ValueError("No valid tiles could be loaded")
    
    # Combine all tiles
    print("Combining tiles into single AnnData object...")
    combined_adata = ad.concat(adata_list, join='outer', index_unique='-')
    
    # Ensure spatial coordinates are properly set
    if 'spatial' not in combined_adata.obsm:
        print("Warning: No spatial coordinates found in combined data")
    
    print(f"Combined dataset: {combined_adata.n_obs} cells, {combined_adata.n_vars} genes")
    print(f"Tiles: {combined_adata.obs['tile_name'].unique()}")
    
    return combined_adata


def load_cell_type_colors(type_info_path: str = '../type_info.json') -> Dict[str, tuple]:
    """
    Load cell type colors from type_info.json file.

    Parameters:
    -----------
    type_info_path : str
        Path to type_info.json file

    Returns:
    --------
    color_dict : dict
        Dictionary mapping cell type IDs to normalized RGB tuples
    """
    # Try multiple possible paths
    possible_paths = [
        type_info_path,
        'type_info.json',
        '../type_info.json',
        '../../type_info.json',
        os.path.join(os.path.dirname(__file__), '../type_info.json'),
        'C:\\ProgramData\\github_repo\\image_analysis_scripts\\type_info.json'
    ]

    type_info = None
    used_path = None

    for path in possible_paths:
        try:
            with open(path, 'r') as f:
                type_info = json.load(f)
                used_path = path
                break
        except (FileNotFoundError, json.JSONDecodeError):
            continue

    if type_info is None:
        print(f"Warning: Could not find type_info.json at any expected location")
        return {}

    print(f"  - Loaded cell type colors from: {used_path}")

    # Convert RGB values from 0-255 to 0-1 for matplotlib
    color_dict = {}
    for type_id, (name, rgb) in type_info.items():
        normalized_rgb = tuple(c / 255.0 for c in rgb)
        color_dict[int(type_id)] = normalized_rgb

    return color_dict


def apply_cell_type_colors_to_adata(adata: ad.AnnData,
                                     celltype_key: str = 'cell_type',
                                     celltype_id_key: str = 'cell_type_id'):
    """
    Apply cell type colors from type_info.json to AnnData object.

    Parameters:
    -----------
    adata : AnnData
        AnnData object to apply colors to
    celltype_key : str
        Key in adata.obs containing cell type labels
    celltype_id_key : str
        Key in adata.obs containing cell type IDs (numeric)
    """
    if celltype_key not in adata.obs.columns:
        print(f"  - Warning: '{celltype_key}' not found in adata.obs, skipping color setup")
        return

    # Load colors from JSON
    color_dict = load_cell_type_colors()
    if not color_dict:
        return

    # Get unique cell types in order
    if celltype_id_key in adata.obs.columns:
        # Use cell_type_id for ordering if available
        cell_type_order = adata.obs[[celltype_key, celltype_id_key]].drop_duplicates()
        cell_type_order = cell_type_order.sort_values(celltype_id_key)
        cell_types_ordered = cell_type_order[celltype_key].tolist()
        cell_type_ids = cell_type_order[celltype_id_key].tolist()
    else:
        # Fallback: use categorical order
        cell_types_ordered = list(adata.obs[celltype_key].cat.categories)
        cell_type_ids = list(range(len(cell_types_ordered)))

    # Create color list matching cell type order
    colors = []
    for ct_id in cell_type_ids:
        if ct_id in color_dict:
            colors.append(color_dict[ct_id])
        else:
            # Fallback to gray if ID not found
            colors.append((0.5, 0.5, 0.5))

    # Store colors in adata
    adata.uns[f'{celltype_key}_colors'] = colors
    print(f"  - Applied {len(colors)} cell type colors from type_info.json")


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
    Sch�rch et al. (2020) "Coordinated cellular neighborhoods orchestrate
    antitumoral immunity at the colorectal cancer invasive front"
    """

    def __init__(self, adata: ad.AnnData, cn_key: str = 'cn_celltype',
                 celltype_key: str = 'cell_type', apply_colors: bool = True):
        """
        Initialize SC detector.

        Parameters:
        -----------
        adata : AnnData
            AnnData object with spatial coordinates and CN annotations
        cn_key : str, default='cn_celltype'
            Key in adata.obs containing CN labels
        celltype_key : str, default='cell_type'
            Key in adata.obs containing cell type labels
        apply_colors : bool, default=True
            Whether to automatically apply colors from type_info.json
        """
        self.adata = adata
        self.cn_key = cn_key
        self.sc_labels = None
        self.aggregated_cn_fractions = None
        self.sc_graph = None

        # Verify CN labels exist
        if cn_key not in adata.obs.columns:
            raise ValueError(f"CN key '{cn_key}' not found in adata.obs. "
                           f"Please run cellular neighborhood detection first.")

        # Apply cell type colors from type_info.json
        if apply_colors:
            print("\nApplying cell type colors from type_info.json...")
            apply_cell_type_colors_to_adata(self.adata, celltype_key=celltype_key)

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

        # For each cell, compute CN fractions in neighborhood
        for i in range(n_cells):
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
        aggregated_key: str = 'aggregated_cn_fractions',
        output_key: str = 'spatial_context'
    ):
        """
        Assign spatial context labels based on minimal CN combinations that
        exceed the threshold.

        Algorithm:
        1. For each cell, sort CN fractions from high to low
        2. Add CNs cumulatively until sum >= threshold
        3. SC label = sorted CN IDs joined by "_" (e.g., "1_2", "2_4_5")

        Parameters:
        -----------
        threshold : float, default=0.9
            Cumulative fraction threshold for SC assignment
        aggregated_key : str
            Key in adata.obsm containing aggregated CN fractions
        output_key : str
            Key to store SC labels in adata.obs
        """
        print(f"Detecting spatial contexts (threshold={threshold})...")

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
                cumsum += frac
                selected_cns.append(str(cn))

                if cumsum >= threshold:
                    break

            # Create SC label by joining CN IDs
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
            Minimum number of groups (e.g., patients) an SC must appear in
        group_key : str, optional
            Key in adata.obs containing group identifiers (e.g., 'patient_id')
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
        img_id_key: str = 'sample_id',
        coord_key: str = 'spatial',
        point_size: float = 0.5,
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

        plt.show()

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
            f"Nodes: Spatial Contexts (size  cell count)\n"
            f"Edges: Shared neighborhoods (width  interaction frequency)\n"
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

        plt.show()

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
        min_cells: int = 100,
        min_groups: Optional[int] = None,
        group_key: Optional[str] = None,
        img_id_key: str = 'sample_id',
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
        min_cells : int, default=100
            Minimum cells for SC to be retained
        min_groups : int, optional
            Minimum groups (e.g., patients) for SC to be retained
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
        self.detect_spatial_contexts(threshold=threshold)

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


class BatchSpatialContextProcessor:
    """
    Process multiple tiles for spatial context detection.
    Supports both individual tile processing and aggregated processing.
    """
    
    def __init__(self, batch_results_dir: str, output_dir: str = 'batch_spatial_contexts'):
        """
        Initialize batch processor.
        
        Parameters:
        -----------
        batch_results_dir : str
            Path to batch results directory
        output_dir : str
            Directory to save results
        """
        self.batch_results_dir = Path(batch_results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Discover processed files
        self.file_paths = discover_processed_files(batch_results_dir)
        if not self.file_paths:
            raise ValueError(f"No processed h5ad files found in {batch_results_dir}")
    
    def process_individual_tiles(self, 
                                k: int = 40,
                                threshold: float = 0.9,
                                min_cells: int = 100,
                                img_id_key: str = 'tile_name',
                                graph_layout: str = 'spring') -> Dict[str, Dict]:
        """
        Process each tile individually for spatial context detection.
        
        Parameters:
        -----------
        k : int, default=40
            Number of nearest neighbors for SC detection
        threshold : float, default=0.9
            Cumulative CN fraction threshold for SC assignment
        min_cells : int, default=100
            Minimum cells for SC to be retained
        img_id_key : str
            Key in adata.obs containing tile identifiers
        graph_layout : str
            Layout algorithm for SC graph
            
        Returns:
        --------
        results : Dict[str, Dict]
            Results for each tile
        """
        print("=" * 80)
        print("INDIVIDUAL TILE SPATIAL CONTEXT PROCESSING")
        print("=" * 80)
        
        # Load tiles individually
        adata_list = load_multiple_tiles_individual(self.file_paths)
        
        results = {}
        tile_output_dir = self.output_dir / 'individual_tiles'
        tile_output_dir.mkdir(exist_ok=True)
        
        for i, adata in enumerate(adata_list, 1):
            tile_name = adata.obs['tile_name'].iloc[0]
            print(f"\n{'='*60}")
            print(f"PROCESSING TILE {i}/{len(adata_list)}: {tile_name}")
            print(f"{'='*60}")
            
            try:
                # Initialize detector
                detector = SpatialContextDetector(adata, cn_key='cn_celltype', celltype_key='cell_type')
                
                # Run pipeline
                detector.run_full_pipeline(
                    k=k,
                    threshold=threshold,
                    min_cells=min_cells,
                    img_id_key=img_id_key,
                    output_dir=str(tile_output_dir / tile_name),
                    graph_layout=graph_layout
                )
                
                # Save annotated data
                output_path = tile_output_dir / f'{tile_name}_adata_scs.h5ad'
                adata.write(output_path)
                
                # Store results
                results[tile_name] = {
                    'n_cells': adata.n_obs,
                    'n_scs': len(adata.obs['spatial_context_filtered'].cat.categories),
                    'output_path': str(output_path)
                }
                
                print(f"Tile {tile_name} completed successfully")
                
            except Exception as e:
                print(f"Error processing tile {tile_name}: {str(e)}")
                results[tile_name] = {'error': str(e)}
                continue
        
        # Create summary
        self._create_individual_summary(results, tile_output_dir)
        
        return results
    
    def process_aggregated_tiles(self,
                                 k: int = 40,
                                 threshold: float = 0.9,
                                 min_cells: int = 100,
                                 min_groups: Optional[int] = None,
                                 group_key: Optional[str] = None,
                                 img_id_key: str = 'tile_name',
                                 graph_layout: str = 'spring',
                                 coord_offset: bool = True) -> Dict:
        """
        Process all tiles as a combined dataset for spatial context detection.
        
        Parameters:
        -----------
        k : int, default=40
            Number of nearest neighbors for SC detection
        threshold : float, default=0.9
            Cumulative CN fraction threshold for SC assignment
        min_cells : int, default=100
            Minimum cells for SC to be retained
        min_groups : int, optional
            Minimum number of tiles an SC must appear in
        group_key : str, optional
            Key for group filtering (use 'tile_name' for tile-based filtering)
        img_id_key : str
            Key in adata.obs containing tile identifiers
        graph_layout : str
            Layout algorithm for SC graph
        coord_offset : bool
            Whether to offset spatial coordinates to avoid overlap
            
        Returns:
        --------
        results : Dict
            Aggregated processing results
        """
        print("=" * 80)
        print("AGGREGATED TILE SPATIAL CONTEXT PROCESSING")
        print("=" * 80)
        
        # Load and combine all tiles
        combined_adata = load_multiple_tiles_aggregated(self.file_paths, coord_offset=coord_offset)
        
        # Initialize detector
        detector = SpatialContextDetector(combined_adata, cn_key='cn_celltype', celltype_key='cell_type')
        
        # Run pipeline
        detector.run_full_pipeline(
            k=k,
            threshold=threshold,
            min_cells=min_cells,
            min_groups=min_groups,
            group_key=group_key or 'tile_name',
            img_id_key=img_id_key,
            output_dir=str(self.output_dir / 'aggregated'),
            graph_layout=graph_layout
        )
        
        # Save annotated data
        output_path = self.output_dir / 'aggregated_adata_scs.h5ad'
        combined_adata.write(output_path)
        
        # Create tile-specific summaries
        self._create_aggregated_summary(combined_adata, self.output_dir)
        
        results = {
            'n_tiles': len(combined_adata.obs['tile_name'].unique()),
            'n_cells': combined_adata.n_obs,
            'n_scs': len(combined_adata.obs['spatial_context_filtered'].cat.categories),
            'output_path': str(output_path)
        }
        
        return results
    
    def _create_individual_summary(self, results: Dict[str, Dict], output_dir: Path):
        """Create summary statistics for individual tile processing."""
        print("\n" + "=" * 60)
        print("CREATING INDIVIDUAL TILE SUMMARY")
        print("=" * 60)
        
        # Collect statistics
        summary_data = []
        for tile_name, tile_results in results.items():
            if 'error' not in tile_results:
                summary_data.append({
                    'tile_name': tile_name,
                    'n_cells': tile_results['n_cells'],
                    'n_scs': tile_results['n_scs'],
                    'status': 'success'
                })
            else:
                summary_data.append({
                    'tile_name': tile_name,
                    'n_cells': 0,
                    'n_scs': 0,
                    'status': f"error: {tile_results['error']}"
                })
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        summary_path = output_dir / 'individual_tile_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        # Print summary
        successful_tiles = summary_df[summary_df['status'] == 'success']
        print(f"Successfully processed: {len(successful_tiles)}/{len(summary_df)} tiles")
        if len(successful_tiles) > 0:
            print(f"Average cells per tile: {successful_tiles['n_cells'].mean():.0f}")
            print(f"Average SCs per tile: {successful_tiles['n_scs'].mean():.1f}")
        
        print(f"Summary saved to: {summary_path}")
    
    def _create_aggregated_summary(self, adata: ad.AnnData, output_dir: Path):
        """Create summary statistics for aggregated processing."""
        print("\n" + "=" * 60)
        print("CREATING AGGREGATED SUMMARY")
        print("=" * 60)
        
        # Tile-level statistics
        tile_stats = []
        for tile_name in adata.obs['tile_name'].unique():
            tile_mask = adata.obs['tile_name'] == tile_name
            tile_cells = tile_mask.sum()
            tile_scs = adata.obs['spatial_context_filtered'][tile_mask].nunique()
            
            tile_stats.append({
                'tile_name': tile_name,
                'n_cells': tile_cells,
                'n_scs': tile_scs
            })
        
        tile_stats_df = pd.DataFrame(tile_stats)
        tile_stats_path = output_dir / 'aggregated_tile_statistics.csv'
        tile_stats_df.to_csv(tile_stats_path, index=False)
        
        # SC-level statistics
        sc_stats = adata.obs['spatial_context_filtered'].value_counts()
        sc_stats_path = output_dir / 'aggregated_sc_statistics.csv'
        sc_stats.to_csv(sc_stats_path)
        
        print(f"Total tiles: {len(tile_stats_df)}")
        print(f"Total cells: {adata.n_obs}")
        print(f"Total SCs: {len(sc_stats)}")
        print(f"Tile statistics saved to: {tile_stats_path}")
        print(f"SC statistics saved to: {sc_stats_path}")


# Example usage function
def main():
    """
    Example usage of SpatialContextDetector for multiple files.

    Prerequisites:
    - Run cn_batch_kmeans.py to detect CNs in multiple tiles
    - Processed h5ad files should be in cn_batch_results/processed_h5ad/
    """
    # Configuration
    batch_results_dir = '/mnt/c/ProgramData/github_repo/image_analysis_scripts/neighborhood_composition/spatial_contexts/selected_cn_batch_results'
    output_dir = 'batch_spatial_contexts'
    
    print("=" * 80)
    print("BATCH SPATIAL CONTEXT DETECTION")
    print("=" * 80)
    print(f"Batch results directory: {batch_results_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    try:
        # Initialize batch processor
        processor = BatchSpatialContextProcessor(
            batch_results_dir=batch_results_dir,
            output_dir=output_dir
        )
        
        print(f"Found {len(processor.file_paths)} processed tiles")
        
        # Choose processing mode
        print("\nChoose processing mode:")
        print("1. Individual tile processing (each tile processed separately)")
        print("2. Aggregated processing (all tiles combined)")
        
        # For this example, we'll do both
        print("\n" + "="*60)
        print("RUNNING INDIVIDUAL TILE PROCESSING")
        print("="*60)
        
        individual_results = processor.process_individual_tiles(
            k=40,
            threshold=0.9,
            min_cells=100,
            img_id_key='tile_name',
            graph_layout='spring'
        )
        
        print("\n" + "="*60)
        print("RUNNING AGGREGATED PROCESSING")
        print("="*60)
        
        aggregated_results = processor.process_aggregated_tiles(
            k=40,
            threshold=0.9,
            min_cells=100,
            min_groups=2,  # SC must appear in at least 2 tiles
            group_key='tile_name',
            img_id_key='tile_name',
            graph_layout='spring',
            coord_offset=True
        )
        
        print("\n" + "="*80)
        print("BATCH PROCESSING COMPLETE!")
        print("="*80)
        print(f"Individual processing: {len([r for r in individual_results.values() if 'error' not in r])} tiles successful")
        print(f"Aggregated processing: {aggregated_results['n_tiles']} tiles, {aggregated_results['n_cells']} cells, {aggregated_results['n_scs']} SCs")
        print(f"Results saved to: {output_dir}/")
        
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        return


def main_single_file():
    """
    Example usage for single file processing (original functionality).
    """
    # Input file configuration
    input_file = '../cellular_neighborhoods/cn/tile_39520_7904_adata_cns.h5ad'
    output_dir = 'sp_contexts'

    # Load data with CN annotations
    print("Loading data with CN annotations...")
    adata = sc.read_h5ad(input_file)

    # Verify required columns exist
    required_cols = ['cn_celltype', 'tile_name']
    for col in required_cols:
        if col not in adata.obs.columns:
            print(f"WARNING: Required column '{col}' not found in adata.obs")
            print(f"Available columns: {adata.obs.columns.tolist()}")
            return

    # Initialize detector (colors will be automatically loaded from type_info.json)
    detector = SpatialContextDetector(adata, cn_key='cn_celltype', celltype_key='cell_type')

    # Run full pipeline
    detector.run_full_pipeline(
        k=40,  # Larger neighborhood for SC detection
        threshold=0.9,  # 90% cumulative CN fraction
        min_cells=100,  # Minimum cells per SC
        min_groups=None,  # Set to e.g., 3 if you have patient_id column
        group_key=None,  # Set to 'patient_id' if available
        img_id_key='tile_name',  # Adjust to your column name
        output_dir=output_dir,
        graph_layout='spring'  # Try 'kamada_kawai' or 'circular' too
    )

    # Save annotated data
    output_filename = Path(input_file).stem.replace('_adata_cns', '_adata_scs.h5ad')
    output_path = os.path.join(output_dir, output_filename)
    adata.write(output_path)
    print(f"\nAnnotated data with SCs saved to: {output_path}")


if __name__ == '__main__':
    main()