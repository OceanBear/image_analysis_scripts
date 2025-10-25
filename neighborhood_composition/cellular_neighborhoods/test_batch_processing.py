"""
Test script for batch processing with skip functionality.

This script demonstrates the new skip functionality and auto-detection
of column names in the batch processing script.
"""

from cn_batch_kmeans import BatchCellularNeighborhoodProcessor


def test_batch_processing_with_skip():
    """
    Test batch processing with skip functionality enabled.
    """
    print("=" * 80)
    print("TESTING BATCH PROCESSING WITH SKIP FUNCTIONALITY")
    print("=" * 80)
    
    # Configuration
    tiles_directory = '/mnt/g/GDC-TCGA-LUAD/00a0b174-1eab-446a-ba8c-7c6e3acd7f0c/pred/h5ad'
    output_base_dir = 'cn_batch_results_test'
    
    # Initialize processor
    processor = BatchCellularNeighborhoodProcessor(
        tiles_directory=tiles_directory,
        output_base_dir=output_base_dir
    )
    
    # Process with skip functionality enabled
    results = processor.process_all_tiles(
        k=20,
        n_clusters=6,
        celltype_key='cell_type',  # Will auto-detect if not found
        img_id_key='tile_name',    # Will auto-detect if not found
        random_state=220705,
        save_adata=True,
        max_tiles=5,               # Test with first 5 tiles only
        skip_processed=True        # Skip already processed tiles
    )
    
    print(f"\nTest completed!")
    print(f"Results: {len(results)} tiles processed")
    
    # Show results summary
    for result in results:
        status = result.get('status', 'unknown')
        tile_name = result.get('tile_name', 'unknown')
        print(f"  - {tile_name}: {status}")
    
    return results


def test_column_detection():
    """
    Test the column name auto-detection functionality.
    """
    print("=" * 80)
    print("TESTING COLUMN NAME AUTO-DETECTION")
    print("=" * 80)
    
    # This will test the auto-detection by trying different column names
    tiles_directory = '/mnt/g/GDC-TCGA-LUAD/00a0b174-1eab-446a-ba8c-7c6e3acd7f0c/pred/h5ad'
    output_base_dir = 'cn_batch_results_detection_test'
    
    processor = BatchCellularNeighborhoodProcessor(
        tiles_directory=tiles_directory,
        output_base_dir=output_base_dir
    )
    
    # Try with wrong column names - should auto-detect correct ones
    results = processor.process_all_tiles(
        k=20,
        n_clusters=6,
        celltype_key='wrong_column',  # Will be auto-detected
        img_id_key='wrong_column',    # Will be auto-detected
        random_state=220705,
        save_adata=True,
        max_tiles=2,                  # Test with first 2 tiles only
        skip_processed=True
    )
    
    print(f"\nColumn detection test completed!")
    return results


def main():
    """
    Run the tests.
    """
    print("Batch Processing Test Suite")
    print("=" * 80)
    
    # Test 1: Skip functionality
    print("\n1. Testing skip functionality...")
    results1 = test_batch_processing_with_skip()
    
    # Test 2: Column detection
    print("\n2. Testing column auto-detection...")
    results2 = test_column_detection()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
    print("Check the output directories for results:")
    print("- cn_batch_results_test/")
    print("- cn_batch_results_detection_test/")


if __name__ == '__main__':
    main()
