from spatial_analysis import run_spatial_analysis_pipeline

adata = run_spatial_analysis_pipeline(
    adata_path='tile_39520_7904.h5ad',
    radius=50,  # Adjust for your tissue
    n_perms=1000
)

