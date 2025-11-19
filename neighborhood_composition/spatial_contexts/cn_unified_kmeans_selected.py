"""
Group-based Cellular Neighborhood Analysis

This script analyzes pre-computed unified CN results by groups (adjacent_tissue, center, margin).
It reads processed h5ad files from unified CN detection and generates group-specific visualizations.

Author: Generated with Claude Code
Date: 2025-11-19
"""

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
import json
from pathlib import Path
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class GroupCNAnalyzer:
    """Analyzes cellular neighborhoods by predefined groups."""
    
    def __init__(self, processed_h5ad_dir: str, categories_json: str, output_dir: str):
        """
        Initialize group CN analyzer.
        
        Parameters:
        -----------
        processed_h5ad_dir : str
            Directory containing processed h5ad files with CN annotations
        categories_json : str
            Path to JSON file with tile categorization
        output_dir : str
            Output directory for group-specific results
        """
        self.processed_h5ad_dir = Path(processed_h5ad_dir)
        self.categories_json = Path(categories_json)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load categorization
        with open(self.categories_json, 'r') as f:
            self.categories = json.load(f)
        
        print(f"✓ Loaded tile categories from: {self.categories_json}")
        print(f"  Groups: {list(self.categories.keys() - {'metadata'})}")
        
    def load_group_data(
        self,
        group_name: str,
        cn_key: str = 'cn_celltype',
        celltype_key: str = 'cell_type'
    ) -> ad.AnnData:
        """
        Load and combine h5ad files for a specific group.
        
        Parameters:
        -----------
        group_name : str
            Group name (e.g., 'adjacent_tissue', 'center', 'margin')
        cn_key : str
            Key in adata.obs containing CN labels
        celltype_key : str
            Key in adata.obs containing cell type labels
            
        Returns:
        --------
        combined_adata : AnnData
            Combined AnnData for the group
        """
        if group_name not in self.categories:
            raise ValueError(f"Group '{group_name}' not found in categories")
        
        tile_names = self.categories[group_name]
        print(f"\nLoading {len(tile_names)} tiles for group: {group_name}")
        
        adata_list = []
        for i, tile_name in enumerate(tile_names, 1):
            h5ad_file = self.processed_h5ad_dir / f'{tile_name}_adata_cns.h5ad'
            
            if not h5ad_file.exists():
                print(f"  [{i}/{len(tile_names)}] Warning: {h5ad_file.name} not found, skipping")
                continue
            
            adata = ad.read_h5ad(h5ad_file)
            
            # Ensure cn_key exists
            if cn_key not in adata.obs.columns:
                print(f"  [{i}/{len(tile_names)}] Warning: {cn_key} not in {tile_name}, skipping")
                continue
            
            print(f"  [{i}/{len(tile_names)}] Loaded {tile_name}: {adata.n_obs} cells")
            adata_list.append(adata)
        
        if not adata_list:
            raise ValueError(f"No valid h5ad files found for group: {group_name}")
        
        # Combine
        combined_adata = ad.concat(adata_list, join='outer', index_unique='-')
        print(f"✓ Combined {len(adata_list)} tiles: {combined_adata.n_obs:,} cells")
        
        return combined_adata
    
    def compute_cn_composition(
        self,
        adata: ad.AnnData,
        cn_key: str = 'cn_celltype',
        celltype_key: str = 'cell_type'
    ):
        """Compute CN composition for a group."""
        composition = pd.crosstab(
            adata.obs[cn_key],
            adata.obs[celltype_key],
            normalize='index'
        )
        composition_zscore = composition.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        return composition, composition_zscore
    
    def visualize_cn_composition_heatmap(
        self,
        composition_zscore: pd.DataFrame,
        group_name: str,
        n_cells: int,
        figsize=(12, 8),
        save_path: Optional[str] = None
    ):
        """Visualize CN composition heatmap for a group."""
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            composition_zscore,
            cmap='coolwarm',
            center=0,
            vmin=-2,
            vmax=2,
            cbar_kws={'label': 'Z-score'},
            linewidths=0.5,
            linecolor='white',
            ax=ax,
            annot=True,
            fmt='.2f',
            annot_kws={'size': 8}
        )
        
        ax.set_xlabel('Cell Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cellular Neighborhood', fontsize=12, fontweight='bold')
        
        title = (f'Cell Type Composition by Cellular Neighborhood\n'
                f'Group: {group_name} ({n_cells:,} cells)\n'
                f'Z-score scaled by column')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved heatmap to: {save_path}")
        
        plt.close(fig)
    
    def calculate_neighborhood_frequency(
        self,
        adata: ad.AnnData,
        cn_key: str = 'cn_celltype'
    ) -> pd.DataFrame:
        """Calculate neighborhood frequency for a group."""
        cn_counts = adata.obs[cn_key].value_counts().sort_index()
        total_cells = len(adata.obs)
        cn_percentages = (cn_counts / total_cells * 100).round(2)
        
        frequency_df = pd.DataFrame({
            'Count': cn_counts,
            'Percentage': cn_percentages
        })
        frequency_df.index.name = 'Cellular_Neighborhood'
        frequency_df = frequency_df.reset_index()
        
        return frequency_df
    
    def visualize_neighborhood_frequency(
        self,
        frequency_df: pd.DataFrame,
        group_name: str,
        figsize=(10, 6),
        save_path: Optional[str] = None,
        color_palette: str = 'Set2'
    ):
        """Visualize neighborhood frequency for a group."""
        plt.close('all')
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Sort by CN ID
        frequency_df_sorted = frequency_df.sort_values('Cellular_Neighborhood')
        cn_ids = [int(cn_id) for cn_id in frequency_df_sorted['Cellular_Neighborhood']]
        
        # Get colors
        n_cns = len(cn_ids)
        colors_palette = sns.color_palette(color_palette, max(cn_ids))
        colors_for_bars = [colors_palette[int(cn_id) - 1] for cn_id in cn_ids]
        
        # Create bars
        bars = ax.bar(
            frequency_df_sorted['Cellular_Neighborhood'].astype(str),
            frequency_df_sorted['Count'],
            color=colors_for_bars
        )
        
        ax.set_xlabel('Cellular Neighborhood', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cell Count', fontsize=12, fontweight='bold')
        ax.set_title(f'CN Frequency (Count) - {group_name}',
                    fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add labels with outline
        text_outline = [path_effects.withStroke(linewidth=3, foreground='white')]
        max_count = max(frequency_df_sorted['Count'])
        
        for bar, count, pct in zip(bars,
                                   frequency_df_sorted['Count'],
                                   frequency_df_sorted['Percentage']):
            height = bar.get_height()
            # Count above bar
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count):,}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   color='black', path_effects=text_outline)
            # Percentage in middle
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'{pct:.1f}%',
                   ha='center', va='center', fontsize=10,
                   color='black', fontweight='bold', path_effects=text_outline)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved frequency graph to: {save_path}")
        
        plt.close(fig)
    
    def visualize_per_tile_frequency_highlighted(
        self,
        all_tiles_h5ad_dir: Path,
        group_name: str,
        cn_key: str = 'cn_celltype',
        figsize=(14, 8),
        save_path: Optional[str] = None,
        color_palette: str = 'Set2'
    ):
        """
        Visualize per-tile frequency with highlighted group tiles.
        
        Parameters:
        -----------
        all_tiles_h5ad_dir : Path
            Directory containing ALL processed h5ad files
        group_name : str
            Group name to highlight
        """
        print(f"\nGenerating per-tile frequency with {group_name} highlighted...")
        
        # Load ALL tiles
        all_h5ad_files = sorted(all_tiles_h5ad_dir.glob('*_adata_cns.h5ad'))
        
        tile_data = {}
        for h5ad_file in all_h5ad_files:
            tile_name = h5ad_file.stem.replace('_adata_cns', '')
            adata = ad.read_h5ad(h5ad_file)
            if cn_key in adata.obs.columns:
                tile_data[tile_name] = adata
        
        # Create frequency dataframe for all tiles
        tile_names = []
        cn_frequencies = []
        
        for tile_name, adata in tile_data.items():
            tile_names.append(tile_name)
            cn_counts = adata.obs[cn_key].value_counts()
            total = len(adata.obs)
            cn_freq = cn_counts / total
            cn_frequencies.append(cn_freq)
        
        # Combine into dataframe
        frequency_df = pd.DataFrame(cn_frequencies, index=tile_names)
        frequency_df = frequency_df.fillna(0)
        
        # Sort columns by CN ID
        cn_ids = sorted([int(col) for col in frequency_df.columns])
        col_mapping = {int(col): col for col in frequency_df.columns}
        sorted_cols = [col_mapping[cn_id] for cn_id in cn_ids]
        frequency_df_sorted = frequency_df[sorted_cols]
        
        # Get colors
        colors_palette = sns.color_palette(color_palette, max(cn_ids))
        color_map = {cn_id: colors_palette[int(cn_id) - 1] for cn_id in cn_ids}
        colors_sorted = [color_map[cn_id] for cn_id in cn_ids]
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        frequency_df_sorted.plot(kind='bar', stacked=True, ax=ax,
                                color=colors_sorted, width=0.8)
        
        ax.set_xlabel('Tile', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency (Proportion)', fontsize=12, fontweight='bold')
        ax.set_title(f'Cellular Neighborhood Frequency by Tile\n(Group: {group_name} highlighted)',
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(title='Cellular Neighborhood', bbox_to_anchor=(1.05, 1),
                 loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Highlight group tiles
        group_tiles = self.categories[group_name]
        x_labels = ax.get_xticklabels()
        
        for i, label in enumerate(x_labels):
            tile_name = label.get_text()
            if tile_name in group_tiles:
                # Highlight with bold and different color
                label.set_weight('bold')
                label.set_color('red')
                label.set_fontsize(10)
        
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved per-tile frequency (highlighted) to: {save_path}")
        
        plt.close(fig)
    
    def analyze_group(
        self,
        group_name: str,
        cn_key: str = 'cn_celltype',
        celltype_key: str = 'cell_type',
        color_palette: str = 'Set2'
    ):
        """Run complete analysis for a single group."""
        print(f"\n{'='*80}")
        print(f"ANALYZING GROUP: {group_name.upper()}")
        print(f"{'='*80}")
        
        # Load data
        adata = self.load_group_data(group_name, cn_key, celltype_key)
        
        # Compute composition
        print("\nComputing CN composition...")
        composition, composition_zscore = self.compute_cn_composition(
            adata, cn_key, celltype_key
        )
        
        # Save composition CSV
        csv_path = self.output_dir / f'unified_cn_composition_{group_name}.csv'
        composition.to_csv(csv_path)
        print(f"  ✓ Saved composition CSV to: {csv_path}")
        
        # Visualize heatmap
        print("\nGenerating CN composition heatmap...")
        heatmap_path = self.output_dir / f'unified_cn_composition_heatmap_{group_name}.png'
        self.visualize_cn_composition_heatmap(
            composition_zscore,
            group_name,
            adata.n_obs,
            save_path=str(heatmap_path)
        )
        
        # Calculate frequency
        print("\nCalculating neighborhood frequency...")
        frequency_df = self.calculate_neighborhood_frequency(adata, cn_key)
        
        # Visualize frequency
        print("\nGenerating neighborhood frequency graph...")
        freq_path = self.output_dir / f'neighborhood_frequency_{group_name}.png'
        self.visualize_neighborhood_frequency(
            frequency_df,
            group_name,
            save_path=str(freq_path),
            color_palette=color_palette
        )
        
        # Visualize per-tile frequency with highlighting
        per_tile_path = self.output_dir / f'neighborhood_frequency_per_tile_{group_name}.png'
        self.visualize_per_tile_frequency_highlighted(
            self.processed_h5ad_dir,
            group_name,
            cn_key=cn_key,
            save_path=str(per_tile_path),
            color_palette=color_palette
        )
        
        print(f"\n✓ Group analysis complete for: {group_name}")
        print(f"  Total cells: {adata.n_obs:,}")
        print(f"  Tiles: {len(self.categories[group_name])}")
    
    def analyze_all_groups(
        self,
        cn_key: str = 'cn_celltype',
        celltype_key: str = 'cell_type',
        color_palette: str = 'Set2'
    ):
        """Run analysis for all groups."""
        banner = "=" * 80
        print(f"\n{banner}")
        print("GROUP-BASED CELLULAR NEIGHBORHOOD ANALYSIS")
        print(f"{banner}")
        print(f"Processed h5ad directory: {self.processed_h5ad_dir}")
        print(f"Categories JSON: {self.categories_json}")
        print(f"Output directory: {self.output_dir}")
        print(f"{banner}\n")
        
        groups = [key for key in self.categories.keys() if key != 'metadata']
        
        for group in groups:
            self.analyze_group(group, cn_key, celltype_key, color_palette)
        
        print(f"\n{banner}")
        print("ALL GROUP ANALYSES COMPLETE!")
        print(f"{banner}")
        print(f"\nResults saved to: {self.output_dir}/")
        print(f"\nGenerated files for each group:")
        print(f"  - unified_cn_composition_heatmap_{{group}}.png")
        print(f"  - unified_cn_composition_{{group}}.csv")
        print(f"  - neighborhood_frequency_{{group}}.png")
        print(f"  - neighborhood_frequency_per_tile_{{group}}.png (with highlighted tiles)")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Group-based Cellular Neighborhood Analysis'
    )
    parser.add_argument(
        '--processed_h5ad_dir',
        default='/mnt/c/ProgramData/github_repo/image_analysis_scripts/neighborhood_composition/spatial_contexts/cn_unified_results/2mm_all_17_clusters=7/processed_h5ad',
        help='Directory containing processed h5ad files with CN annotations'
    )
    parser.add_argument(
        '--categories_json',
        default='/mnt/c/ProgramData/github_repo/image_analysis_scripts/neighborhood_composition/spatial_contexts/cn_unified_results/2mm_all_17_clusters=7/tile_categories.json',
        help='Path to tile categories JSON file'
    )
    parser.add_argument(
        '--output_dir',
        default='cn_unified_results_selected',
        help='Output directory for group-specific results'
    )
    parser.add_argument(
        '--cn_key',
        default='cn_celltype',
        help='Column name for CN labels (default: cn_celltype)'
    )
    parser.add_argument(
        '--celltype_key',
        default='cell_type',
        help='Column name for cell types (default: cell_type)'
    )
    parser.add_argument(
        '--color_palette',
        default='Set2',
        help='Color palette (default: Set2)'
    )
    parser.add_argument(
        '--group',
        default=None,
        help='Analyze specific group only (default: all groups)'
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = GroupCNAnalyzer(
        processed_h5ad_dir=args.processed_h5ad_dir,
        categories_json=args.categories_json,
        output_dir=args.output_dir
    )
    
    # Run analysis
    if args.group:
        analyzer.analyze_group(
            args.group,
            cn_key=args.cn_key,
            celltype_key=args.celltype_key,
            color_palette=args.color_palette
        )
    else:
        analyzer.analyze_all_groups(
            cn_key=args.cn_key,
            celltype_key=args.celltype_key,
            color_palette=args.color_palette
        )


if __name__ == '__main__':
    main()

