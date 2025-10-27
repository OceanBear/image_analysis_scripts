"""
Unified Cellular Neighborhood Detection Across Multiple Tiles

This script loads multiple tiles at once and performs CN detection on the combined
dataset, ensuring all tiles share the same CN composition. This is crucial for
downstream spatial context analysis.

Key Features:
- Loads 10-100 tiles into a unified dataset
- Performs k-means clustering on all cells together (default k=6 CNs)
- Generates ONE unified CN composition heatmap based on all tiles
- Generates individual spatial CN maps for EACH tile
- Saves processed h5ad files with CN annotations for spatial context analysis

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
import os
import glob
import argparse
from pathlib import Path
from sklearn.cluster import KMeans
from typing import Optional, Tuple, List, Dict
import warnings
import time
import json

# Set the working directory to the script's directory
os.chdir(Path(__file__).parent)
warnings.filterwarnings('ignore')


class UnifiedCellularNeighborhoodDetector:
    """
    Detects cellular neighborhoods across multiple tiles using a unified approach.
    All tiles share the same CN composition, enabling cross-tile comparisons.
    """

    def __init__(self, tiles_directory: str, output_dir: str = 'cn_unified_results'):
        """
        Initialize unified CN detector.

        Parameters:
        -----------
        tiles_directory : str
            Directory containing h5ad tile files
        output_dir : str
            Base directory for all outputs
        """
        self.tiles_directory = Path(tiles_directory)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'individual_tiles').mkdir(exist_ok=True)
        (self.output_dir / 'processed_h5ad').mkdir(exist_ok=True)
        (self.output_dir / 'unified_analysis').mkdir(exist_ok=True)
        
        # Also create the direct path for spatial CN figures
        Path('cn_unified_results/individual_tiles').mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.combined_adata = None
        self.tile_list = []
        self.cn_labels = None
        self.aggregated_neighbors = None

    def discover_tiles(self, pattern: str = "*.h5ad", max_tiles: Optional[int] = None) -> List[Path]:
        """
        Discover h5ad files in the tiles directory.

        Parameters:
        -----------
        pattern : str
            File pattern to match (default: "*.h5ad")
        max_tiles : int, optional
            Maximum number of tiles to process (for testing)

        Returns:
        --------
        tile_files : List[Path]
            List of discovered tile files
        """
        print(f"Discovering tiles in: {self.tiles_directory}")
        
        tile_files = sorted(list(self.tiles_directory.glob(pattern)))
        
        if not tile_files:
            print(f"Warning: No {pattern} files found in {self.tiles_directory}")
            return []
        
        if max_tiles:
            tile_files = tile_files[:max_tiles]
            print(f"Found {len(tile_files)} tile files (limited to {max_tiles})")
        else:
            print(f"Found {len(tile_files)} tile files")
        
        return tile_files

    def load_and_combine_tiles(
        self,
        tile_files: List[Path],
        celltype_key: str = 'cell_type',
        coord_offset: bool = True
    ) -> ad.AnnData:
        """
        Load multiple tiles and combine them into a single AnnData object.

        Parameters:
        -----------
        tile_files : List[Path]
            List of paths to h5ad files
        celltype_key : str
            Key in adata.obs containing cell type labels
        coord_offset : bool
            Whether to offset spatial coordinates to avoid overlap between tiles

        Returns:
        --------
        combined_adata : AnnData
            Combined AnnData object with all tiles
        """
        print(f"\nLoading and combining {len(tile_files)} tiles...")
        
        adata_list = []
        coord_offset_x = 0
        coord_offset_y = 0
        
        for i, tile_path in enumerate(tile_files, 1):
            tile_name = tile_path.stem
            print(f"  [{i}/{len(tile_files)}] Loading: {tile_name}")
            
            try:
                adata = sc.read_h5ad(tile_path)
                
                # Add tile identifier
                adata.obs['tile_name'] = tile_name
                adata.obs['tile_id'] = i - 1  # 0-based tile ID
                
                # Auto-detect cell type column
                if celltype_key not in adata.obs.columns:
                    alternatives = ['celltype', 'cell_type', 'CellType', 'Cell_Type']
                    for alt in alternatives:
                        if alt in adata.obs.columns:
                            celltype_key = alt
                            break
                
                # Verify required columns
                if celltype_key not in adata.obs.columns:
                    print(f"    Warning: No cell type column found, skipping tile")
                    continue
                
                # Ensure cell types are categorical
                if not pd.api.types.is_categorical_dtype(adata.obs[celltype_key]):
                    adata.obs[celltype_key] = pd.Categorical(adata.obs[celltype_key])
                
                # Offset spatial coordinates if requested
                if coord_offset and 'spatial' in adata.obsm:
                    coords = adata.obsm['spatial'].copy()
                    coords[:, 0] += coord_offset_x
                    coords[:, 1] += coord_offset_y
                    adata.obsm['spatial'] = coords
                    
                    # Store original coordinates before offset
                    adata.obsm['spatial_original'] = adata.obsm['spatial'] - np.array([coord_offset_x, coord_offset_y])
                    
                    # Update offset for next tile (arrange tiles horizontally)
                    max_x = coords[:, 0].max()
                    coord_offset_x = max_x + 500  # 500 pixel gap between tiles
                
                adata_list.append(adata)
                self.tile_list.append(tile_name)
                print(f"    ✓ Loaded {adata.n_obs} cells, {adata.n_vars} genes")
                
            except Exception as e:
                print(f"    ✗ Error loading {tile_path}: {str(e)}")
                continue
        
        if not adata_list:
            raise ValueError("No valid tiles could be loaded")
        
        # Combine all tiles
        print("\nCombining tiles into single dataset...")
        combined_adata = ad.concat(adata_list, join='outer', index_unique='-')
        
        # Ensure spatial coordinates are properly set
        if 'spatial' not in combined_adata.obsm:
            print("  Warning: No spatial coordinates found in combined data")
        
        print(f"✓ Combined dataset: {combined_adata.n_obs} cells, {combined_adata.n_vars} genes")
        print(f"  Tiles: {combined_adata.obs['tile_name'].nunique()}")
        print(f"  Cell types: {combined_adata.obs[celltype_key].nunique()}")
        
        self.combined_adata = combined_adata
        return combined_adata

    def build_knn_graph(
        self,
        k: int = 20,
        coord_key: str = 'spatial',
        key_added: str = 'spatial_connectivities_knn'
    ):
        """
        Build k-nearest neighbor graph on the combined dataset.

        Parameters:
        -----------
        k : int, default=20
            Number of nearest neighbors
        coord_key : str
            Key in adata.obsm containing spatial coordinates
        key_added : str
            Key to store the connectivity matrix
        """
        print(f"\nBuilding unified {k}-NN graph across all tiles...")

        # Build spatial neighbors graph on combined data
        sq.gr.spatial_neighbors(
            self.combined_adata,
            spatial_key=coord_key,
            coord_type='generic',
            n_neighs=k,
            radius=None  # Use KNN, not radius
        )

        # Squidpy uses fixed key names, rename if needed
        actual_key = 'spatial_connectivities'
        if key_added != actual_key and actual_key in self.combined_adata.obsp:
            self.combined_adata.obsp[key_added] = self.combined_adata.obsp[actual_key]

        # Print statistics
        connectivity = self.combined_adata.obsp[key_added]
        avg_neighbors = connectivity.sum(axis=1).mean()
        print(f"  ✓ Average neighbors per cell: {avg_neighbors:.2f}")
        print(f"  ✓ Connectivity matrix shape: {connectivity.shape}")

        return self

    def aggregate_neighbors(
        self,
        celltype_key: str = 'cell_type',
        connectivity_key: str = 'spatial_connectivities_knn',
        output_key: str = 'aggregated_neighbors'
    ):
        """
        For each cell, compute the fraction of each cell phenotype in its neighborhood.

        Parameters:
        -----------
        celltype_key : str
            Key in adata.obs containing cell type labels
        connectivity_key : str
            Key in adata.obsp containing spatial connectivity
        output_key : str
            Key to store aggregated neighbor fractions
        """
        print(f"\nAggregating neighbors by {celltype_key}...")

        # Get cell types
        cell_types = self.combined_adata.obs[celltype_key].values
        unique_types = self.combined_adata.obs[celltype_key].cat.categories

        # Get connectivity matrix
        connectivity = self.combined_adata.obsp[connectivity_key]

        # Initialize aggregated neighbors matrix
        n_cells = self.combined_adata.n_obs
        n_types = len(unique_types)
        aggregated = np.zeros((n_cells, n_types))

        print(f"  Processing {n_cells} cells...")
        
        # For each cell, compute cell type fractions in neighborhood
        for i in range(n_cells):
            if i % 10000 == 0 and i > 0:
                print(f"    Processed {i:,}/{n_cells:,} cells ({100*i/n_cells:.1f}%)")
            
            # Get neighbors (including self)
            neighbors_mask = connectivity[i].toarray().flatten() > 0

            if neighbors_mask.sum() > 0:
                # Get cell types of neighbors
                neighbor_types = cell_types[neighbors_mask]

                # Compute fractions
                for j, ct in enumerate(unique_types):
                    aggregated[i, j] = (neighbor_types == ct).sum() / neighbors_mask.sum()

        # Store in adata
        self.aggregated_neighbors = pd.DataFrame(
            aggregated,
            columns=unique_types,
            index=self.combined_adata.obs_names
        )

        self.combined_adata.obsm[output_key] = aggregated

        print(f"  ✓ Aggregated neighbor fractions shape: {aggregated.shape}")

        return self

    def detect_cellular_neighborhoods(
        self,
        n_clusters: int = 6,    #  numbers of CNs to detect
        random_state: int = 220705,
        aggregated_key: str = 'aggregated_neighbors',
        output_key: str = 'cn_celltype'
    ):
        """
        Cluster cells based on their neighborhood composition using k-means.
        This creates a UNIFIED set of CNs across all tiles.

        Parameters:
        -----------
        n_clusters : int, default=6
            Number of cellular neighborhoods to detect
        random_state : int
            Random seed for reproducibility
        aggregated_key : str
            Key in adata.obsm containing aggregated neighbor fractions
        output_key : str
            Key to store CN labels in adata.obs
        """
        print(f"\nDetecting {n_clusters} unified cellular neighborhoods using k-means...")

        # Get aggregated neighbor fractions
        aggregated = self.combined_adata.obsm[aggregated_key]

        # Perform k-means clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )

        cn_labels = kmeans.fit_predict(aggregated)

        # Store CN labels (add 1 to match R's 1-based indexing)
        self.cn_labels = cn_labels + 1
        self.combined_adata.obs[output_key] = pd.Categorical(self.cn_labels)

        # Print CN sizes overall and per tile
        cn_counts = pd.Series(self.cn_labels).value_counts().sort_index()
        print(f"\n  ✓ Unified CN sizes (across all tiles):")
        for cn, count in cn_counts.items():
            percentage = 100 * count / len(self.cn_labels)
            print(f"    CN {cn}: {count:,} cells ({percentage:.1f}%)")
        
        # CN distribution per tile
        print(f"\n  CN distribution per tile:")
        for tile_name in self.combined_adata.obs['tile_name'].unique():
            tile_mask = self.combined_adata.obs['tile_name'] == tile_name
            tile_cns = self.combined_adata.obs[output_key][tile_mask]
            print(f"    {tile_name}: {tile_cns.value_counts().sort_index().to_dict()}")

        return self

    def compute_unified_cn_composition(
        self,
        cn_key: str = 'cn_celltype',
        celltype_key: str = 'cell_type'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute unified cell phenotype fractions in each CN across ALL tiles.

        Parameters:
        -----------
        cn_key : str
            Key in adata.obs containing CN labels
        celltype_key : str
            Key in adata.obs containing cell type labels

        Returns:
        --------
        composition : DataFrame
            CN composition matrix (rows=CNs, columns=cell types)
        composition_zscore : DataFrame
            Z-score normalized composition matrix
        """
        print("\nComputing unified CN composition across all tiles...")

        # Create contingency table
        composition = pd.crosstab(
            self.combined_adata.obs[cn_key],
            self.combined_adata.obs[celltype_key],
            normalize='index'  # Normalize by CN (rows)
        )

        # Z-score by column (cell type)
        composition_zscore = composition.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

        print(f"  ✓ Composition matrix shape: {composition.shape}")
        
        return composition, composition_zscore

    def visualize_unified_cn_composition(
        self,
        composition_zscore: pd.DataFrame,
        k: int,
        n_clusters: int,
        figsize: Tuple[int, int] = (12, 8),
        cmap: str = 'coolwarm',
        vmin: float = -2,
        vmax: float = 2,
        save_path: Optional[str] = None,
        show_values: bool = True
    ):
        """
        Visualize unified CN composition as heatmap across ALL tiles.

        Parameters:
        -----------
        composition_zscore : DataFrame
            Z-score normalized composition matrix
        k : int
            Number of nearest neighbors used
        n_clusters : int
            Number of CNs detected
        figsize : tuple
            Figure size (width, height)
        cmap : str
            Colormap name
        vmin, vmax : float
            Color scale limits for z-scores
        save_path : str, optional
            Path to save figure
        show_values : bool
            Whether to show values in heatmap cells
        """
        print("\nVisualizing unified CN composition heatmap...")

        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        sns.heatmap(
            composition_zscore,
            cmap=cmap,
            center=0,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={'label': 'Z-score'},
            linewidths=0.5,
            linecolor='white',
            ax=ax,
            annot=show_values,
            fmt='.2f' if show_values else '',
            annot_kws={'size': 8}
        )

        ax.set_xlabel('Cell Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cellular Neighborhood', fontsize=12, fontweight='bold')
        
        n_tiles = self.combined_adata.obs['tile_name'].nunique()
        n_cells = self.combined_adata.n_obs
        title = (f'Unified Cell Type Composition by Cellular Neighborhood\n'
                f'(k={k}, n_clusters={n_clusters}, {n_tiles} tiles, {n_cells:,} cells)\n'
                f'Z-score scaled by column')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved unified heatmap to: {save_path}")

        plt.close()

        return fig

    def visualize_individual_tile_cns(
        self,
        cn_key: str = 'cn_celltype',
        coord_key: str = 'spatial_original',
        point_size: float = 10.0,  # Increased to 5.0 for much better cell visibility
        palette: str = 'Set3',
        k: Optional[int] = None,
        n_clusters: Optional[int] = None
    ):
        """
        Visualize cellular neighborhoods spatially for EACH tile individually.
        
        Saves figures as spatial_cns_{tile_name}.png in cn_unified_results/individual_tiles/

        Parameters:
        -----------
        cn_key : str
            Key in adata.obs containing CN labels
        coord_key : str
            Key in adata.obsm containing spatial coordinates
        point_size : float
            Size of points in scatter plot (default: 5.0 for much better cell visibility)
        palette : str
            Color palette name
        k : int, optional
            Number of nearest neighbors used
        n_clusters : int, optional
            Number of CNs detected
        """
        print(f"\nGenerating individual spatial CN maps for each tile...")

        # Get CN labels and colors
        n_cns = len(self.combined_adata.obs[cn_key].cat.categories)
        colors_palette = sns.color_palette(palette, n_cns)

        # Process each tile
        for tile_idx, tile_name in enumerate(self.tile_list, 1):
            print(f"  [{tile_idx}/{len(self.tile_list)}] Plotting {tile_name}...")
            
            # Get cells from this tile
            tile_mask = self.combined_adata.obs['tile_name'] == tile_name
            tile_adata = self.combined_adata[tile_mask]
            
            # Use original coordinates if available, otherwise use spatial
            if coord_key in tile_adata.obsm:
                coords = tile_adata.obsm[coord_key]
            elif 'spatial' in tile_adata.obsm:
                coords = tile_adata.obsm['spatial']
            else:
                print(f"    Warning: No spatial coordinates found for {tile_name}, skipping...")
                continue
            
            cn_labels = tile_adata.obs[cn_key].values

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 10))

            # Plot each CN
            for cn_id in np.unique(cn_labels):
                cn_mask = cn_labels == cn_id
                cn_idx = int(cn_id) - 1  # Convert to 0-based for color indexing

                ax.scatter(
                    coords[cn_mask, 0],
                    coords[cn_mask, 1],
                    c=[colors_palette[cn_idx]],
                    s=point_size,
                    alpha=0.7,
                    label=f'CN {cn_id}'
                )

            ax.set_xlabel('X coordinate (pixels)', fontsize=12)
            ax.set_ylabel('Y coordinate (pixels)', fontsize=12)
            ax.set_aspect('equal')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Add title with parameters
            title = f'Cellular Neighborhoods: {tile_name}'
            if k is not None and n_clusters is not None:
                title += f'\n(k={k}, n_clusters={n_clusters}, {tile_adata.n_obs:,} cells)'
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

            # Add legend
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

            plt.tight_layout()

            # Save figure directly to the specified directory with tile-specific naming
            # Create the target directory: image_analysis_scripts/neighborhood_composition/spatial_contexts/cn_unified_results/individual_tiles
            target_dir = Path('cn_unified_results') / 'individual_tiles'
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Name the file as spatial_cns_{tile_name}.png
            save_path = target_dir / f'spatial_cns_{tile_name}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"    ✓ Saved to: {save_path}")

        print(f"  ✓ Generated {len(self.tile_list)} spatial CN maps")

    def save_processed_data(
        self,
        cn_key: str = 'cn_celltype'
    ):
        """
        Save processed h5ad files with CN annotations for each tile.
        These files are needed for spatial context analysis.

        Parameters:
        -----------
        cn_key : str
            Key in adata.obs containing CN labels
        """
        print(f"\nSaving processed h5ad files for spatial context analysis...")

        for tile_idx, tile_name in enumerate(self.tile_list, 1):
            print(f"  [{tile_idx}/{len(self.tile_list)}] Saving {tile_name}...")
            
            # Extract tile data
            tile_mask = self.combined_adata.obs['tile_name'] == tile_name
            tile_adata = self.combined_adata[tile_mask].copy()
            
            # Restore original spatial coordinates if available
            if 'spatial_original' in tile_adata.obsm:
                tile_adata.obsm['spatial'] = tile_adata.obsm['spatial_original']
                del tile_adata.obsm['spatial_original']
            
            # Save processed h5ad file
            output_path = self.output_dir / 'processed_h5ad' / f'{tile_name}_adata_cns.h5ad'
            tile_adata.write(output_path)
            print(f"    ✓ Saved to: {output_path}")

        print(f"  ✓ Saved {len(self.tile_list)} processed h5ad files")

    def save_summary_statistics(
        self,
        k: int,
        n_clusters: int,
        celltype_key: str,
        composition: pd.DataFrame
    ):
        """
        Save summary statistics for the unified CN analysis.

        Parameters:
        -----------
        k : int
            Number of nearest neighbors used
        n_clusters : int
            Number of CNs detected
        celltype_key : str
            Cell type key used
        composition : DataFrame
            CN composition matrix
        """
        print("\nSaving summary statistics...")

        summary = {
            'analysis_type': 'Unified Cellular Neighborhoods',
            'n_tiles': len(self.tile_list),
            'tile_names': self.tile_list,
            'total_cells': int(self.combined_adata.n_obs),
            'total_genes': int(self.combined_adata.n_vars),
            'parameters': {
                'k_neighbors': k,
                'n_clusters': n_clusters,
                'random_state': 220705,
                'celltype_key': celltype_key
            },
            'cn_distribution': self.combined_adata.obs['cn_celltype'].value_counts().to_dict(),
            'cell_type_distribution': self.combined_adata.obs[celltype_key].value_counts().to_dict(),
            'cn_composition': composition.to_dict()
        }

        # Convert any numpy types to native Python types
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj

        summary = convert_to_native(summary)

        # Save summary
        summary_path = self.output_dir / 'unified_analysis' / 'unified_cn_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Saved summary to: {summary_path}")

        # Save composition matrices
        comp_path = self.output_dir / 'unified_analysis' / 'unified_cn_composition.csv'
        composition.to_csv(comp_path)
        print(f"  ✓ Saved composition to: {comp_path}")

    def run_full_pipeline(
        self,
        tile_files: List[Path],
        k: int = 20,
        n_clusters: int = 6,    # number of CNs to detect
        celltype_key: str = 'cell_type',
        random_state: int = 220705,
        coord_offset: bool = True
    ):
        """
        Run the complete unified CN detection pipeline.

        Parameters:
        -----------
        tile_files : List[Path]
            List of paths to h5ad tile files
        k : int, default=20
            Number of nearest neighbors
        n_clusters : int, default=6
            Number of CNs to detect
        celltype_key : str
            Key in adata.obs containing cell type labels
        random_state : int
            Random seed
        coord_offset : bool
            Whether to offset spatial coordinates between tiles
        """
        print("=" * 80)
        print("UNIFIED CELLULAR NEIGHBORHOOD DETECTION PIPELINE")
        print("=" * 80)
        print(f"Processing {len(tile_files)} tiles with unified CN detection")
        print(f"Parameters: k={k}, n_clusters={n_clusters}")
        print("=" * 80)

        start_time = time.time()

        # Step 1: Load and combine tiles
        self.load_and_combine_tiles(tile_files, celltype_key, coord_offset)

        # Step 2: Build KNN graph
        self.build_knn_graph(k=k)

        # Step 3: Aggregate neighbors
        self.aggregate_neighbors(celltype_key=celltype_key)

        # Step 4: Detect CNs
        self.detect_cellular_neighborhoods(n_clusters=n_clusters, random_state=random_state)

        # Step 5: Compute composition
        composition, composition_zscore = self.compute_unified_cn_composition(celltype_key=celltype_key)

        # Step 6: Visualize unified heatmap
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        heatmap_path = self.output_dir / 'unified_analysis' / 'unified_cn_composition_heatmap.png'
        self.visualize_unified_cn_composition(
            composition_zscore,
            k=k,
            n_clusters=n_clusters,
            save_path=str(heatmap_path),
            show_values=True
        )

        # Step 7: Visualize individual tile maps
        self.visualize_individual_tile_cns(k=k, n_clusters=n_clusters)

        # Step 8: Save processed data
        self.save_processed_data()

        # Step 9: Save summary statistics
        self.save_summary_statistics(k, n_clusters, celltype_key, composition)

        total_time = time.time() - start_time

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE!")
        print("=" * 80)
        print(f"Total processing time: {total_time/60:.1f} minutes")
        print(f"Results saved to: {self.output_dir}/")
        print(f"  - Unified heatmap: {self.output_dir}/unified_analysis/")
        print(f"  - Individual tile maps: cn_unified_results/individual_tiles/")
        print(f"  - Processed h5ad files: {self.output_dir}/processed_h5ad/")
        print("\nProcessed h5ad files are ready for spatial context analysis!")

        return self


def main():
    """
    Main function to run unified CN detection.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Unified Cellular Neighborhood Detection Across Multiple Tiles'
    )
    parser.add_argument(
        '--tiles_dir', '-t',
        default='selected_h5ad_tiles',  #/mnt/c/ProgramData/github_repo/image_analysis_scripts/neighborhood_composition/spatial_contexts/selected_h5ad_tiles
        help='Directory containing h5ad tile files'
    )
    parser.add_argument(
        '--output_dir', '-o',
        default='cn_unified_results',
        help='Output directory for results (spatial CN figures saved to cn_unified_results/individual_tiles/)'
    )
    parser.add_argument(
        '--k', type=int, default=20,
        help='Number of nearest neighbors (default: 20)'
    )
    parser.add_argument(
        '--n_clusters', '-n', type=int, default=7,
        help='Number of cellular neighborhoods (default: 6)'
    )
    parser.add_argument(
        '--celltype_key', '-c',
        default='cell_type',
        help='Column name for cell types (default: cell_type)'
    )
    parser.add_argument(
        '--max_tiles', '-m', type=int, default=None,
        help='Maximum number of tiles to process (for testing)'
    )
    parser.add_argument(
        '--pattern', '-p',
        default='*.h5ad',
        help='File pattern to match (default: *.h5ad)'
    )
    parser.add_argument(
        '--no_offset', action='store_true',
        help='Disable spatial coordinate offsetting between tiles'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("UNIFIED CELLULAR NEIGHBORHOOD DETECTION")
    print("=" * 80)
    print(f"Tiles directory: {args.tiles_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Parameters: k={args.k}, n_clusters={args.n_clusters}")
    print(f"Cell type key: {args.celltype_key}")
    if args.max_tiles:
        print(f"Max tiles: {args.max_tiles} (testing mode)")
    print("=" * 80)

    # Initialize detector
    detector = UnifiedCellularNeighborhoodDetector(
        tiles_directory=args.tiles_dir,
        output_dir=args.output_dir
    )

    # Discover tiles
    tile_files = detector.discover_tiles(pattern=args.pattern, max_tiles=args.max_tiles)

    if not tile_files:
        print("No tiles found! Exiting...")
        return

    # Run full pipeline
    detector.run_full_pipeline(
        tile_files=tile_files,
        k=args.k,
        n_clusters=args.n_clusters,
        celltype_key=args.celltype_key,
        random_state=220705,
        coord_offset=not args.no_offset
    )

    print(f"\nUnified CN detection completed successfully!")
    print(f"Check the results in: {args.output_dir}/")


if __name__ == '__main__':
    main()

