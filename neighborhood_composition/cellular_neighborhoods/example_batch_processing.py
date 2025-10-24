"""
Example script for batch processing cellular neighborhoods.

This script demonstrates how to use the BatchCellularNeighborhoodProcessor
to process multiple tiles with different configurations.
"""

from cn_batch_processing import BatchCellularNeighborhoodProcessor


def example_basic_processing():
    """
    Example 1: Basic batch processing with default parameters.
    """
    print("=" * 80)
    print("EXAMPLE 1: BASIC BATCH PROCESSING")
    print("=" * 80)
    
    # Configuration
    tiles_directory = '/mnt/g/GDC-TCGA-LUAD/00a0b174-1eab-446a-ba8c-7c6e3acd7f0c/pred/h5ad'
    output_base_dir = 'cn_batch_results_basic'
    
    # Initialize processor
    processor = BatchCellularNeighborhoodProcessor(
        tiles_directory=tiles_directory,
        output_base_dir=output_base_dir
    )
    
    # Process all tiles with default parameters
    results = processor.process_all_tiles(
        k=20,
        n_clusters=6,
        celltype_key='cell_type',
        img_id_key='tile_name',
        random_state=220705,
        save_adata=True
    )
    
    print(f"Processed {len(results)} tiles")
    return results


def example_test_processing():
    """
    Example 2: Test processing with limited tiles.
    """
    print("=" * 80)
    print("EXAMPLE 2: TEST PROCESSING (FIRST 3 TILES)")
    print("=" * 80)
    
    # Configuration
    tiles_directory = '/mnt/g/GDC-TCGA-LUAD/00a0b174-1eab-446a-ba8c-7c6e3acd7f0c/pred/h5ad'
    output_base_dir = 'cn_batch_results_test'
    
    # Initialize processor
    processor = BatchCellularNeighborhoodProcessor(
        tiles_directory=tiles_directory,
        output_base_dir=output_base_dir
    )
    
    # Process only first 3 tiles for testing
    results = processor.process_all_tiles(
        k=20,
        n_clusters=6,
        celltype_key='cell_type',
        img_id_key='tile_name',
        random_state=220705,
        save_adata=True,
        max_tiles=3  # Only process first 3 tiles
    )
    
    print(f"Test processed {len(results)} tiles")
    return results


def example_custom_parameters():
    """
    Example 3: Custom parameters for different clustering.
    """
    print("=" * 80)
    print("EXAMPLE 3: CUSTOM PARAMETERS")
    print("=" * 80)
    
    # Configuration
    tiles_directory = '/mnt/g/GDC-TCGA-LUAD/00a0b174-1eab-446a-ba8c-7c6e3acd7f0c/pred/h5ad'
    output_base_dir = 'cn_batch_results_custom'
    
    # Initialize processor
    processor = BatchCellularNeighborhoodProcessor(
        tiles_directory=tiles_directory,
        output_base_dir=output_base_dir
    )
    
    # Process with custom parameters
    results = processor.process_all_tiles(
        k=15,           # Fewer neighbors
        n_clusters=8,   # More clusters
        celltype_key='cell_type',
        img_id_key='tile_name',
        random_state=42,  # Different random seed
        save_adata=True,
        max_tiles=5     # Process first 5 tiles
    )
    
    print(f"Custom processed {len(results)} tiles")
    return results


def main():
    """
    Run examples. Uncomment the example you want to run.
    """
    print("Batch Cellular Neighborhood Processing Examples")
    print("=" * 80)
    
    # Uncomment the example you want to run:
    
    # Example 1: Basic processing (all tiles)
    # results = example_basic_processing()
    
    # Example 2: Test processing (first 3 tiles only)
    results = example_test_processing()
    
    # Example 3: Custom parameters
    # results = example_custom_parameters()
    
    print("\nExample completed!")
    print("Check the output directories for results:")
    print("- Individual tile results: individual_tiles/")
    print("- Intermediate data: intermediate_data/")
    print("- Processed h5ad files: processed_h5ad/")
    print("- Batch summary: batch_processing_summary.csv")


if __name__ == '__main__':
    main()
