"""
Example script for aggregating cellular neighborhood batch results.

This script demonstrates how to use the CNResultsAggregator to combine
all processed tiles into a single comprehensive dataset.
"""

from cn_aggregate_results import CNResultsAggregator


def example_basic_aggregation():
    """
    Example 1: Basic aggregation of all batch results.
    """
    print("=" * 80)
    print("EXAMPLE 1: BASIC AGGREGATION")
    print("=" * 80)
    
    # Configuration
    batch_results_dir = 'cn_batch_results'
    output_dir = 'aggregated'
    
    # Initialize aggregator
    aggregator = CNResultsAggregator(
        batch_results_dir=batch_results_dir,
        output_dir=output_dir
    )
    
    # Run aggregation
    results = aggregator.aggregate_all_results()
    
    if results:
        print(f"\nBasic aggregation completed!")
        print(f"Results saved in: {batch_results_dir}/{output_dir}/")
        
        # Show summary
        summary = results['summary_statistics']
        print(f"\nSummary:")
        print(f"  - Total tiles: {summary['total_tiles']}")
        print(f"  - Total cells: {summary['total_cells']}")
        print(f"  - Total genes: {summary['total_genes']}")
        
        if 'cn_statistics' in summary and summary['cn_statistics']:
            print(f"  - Total CNs: {summary['cn_statistics']['total_cns']}")
        
        if 'cell_type_statistics' in summary and summary['cell_type_statistics']:
            print(f"  - Total cell types: {summary['cell_type_statistics']['total_cell_types']}")
    
    return results


def example_custom_aggregation():
    """
    Example 2: Custom aggregation with specific output directory.
    """
    print("=" * 80)
    print("EXAMPLE 2: CUSTOM AGGREGATION")
    print("=" * 80)
    
    # Configuration
    batch_results_dir = 'cn_batch_results'
    output_dir = 'custom_aggregated'
    
    # Initialize aggregator
    aggregator = CNResultsAggregator(
        batch_results_dir=batch_results_dir,
        output_dir=output_dir
    )
    
    # Run aggregation
    results = aggregator.aggregate_all_results()
    
    if results:
        print(f"\nCustom aggregation completed!")
        print(f"Results saved in: {batch_results_dir}/{output_dir}/")
    
    return results


def main():
    """
    Run aggregation examples.
    """
    print("Cellular Neighborhood Results Aggregation Examples")
    print("=" * 80)
    
    # Example 1: Basic aggregation
    print("\n1. Running basic aggregation...")
    results1 = example_basic_aggregation()
    
    # Example 2: Custom aggregation (uncomment if needed)
    # print("\n2. Running custom aggregation...")
    # results2 = example_custom_aggregation()
    
    print("\n" + "=" * 80)
    print("AGGREGATION EXAMPLES COMPLETED")
    print("=" * 80)
    print("Check the output directories for aggregated results:")
    print("- cn_batch_results/aggregated/")
    print("- cn_batch_results/custom_aggregated/ (if run)")


if __name__ == '__main__':
    main()
