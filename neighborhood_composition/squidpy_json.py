# 1. Convert JSON to Spatial Data Structure
import json
import pandas as pd
import numpy as np
from pathlib import Path
import anndata as ad
import squidpy as sq
import matplotlib.pyplot as plt

# Cell type mapping
CELL_TYPE_DICT = {
    0: "Undefined",
    1: "Epithelium (PD-L1lo/Ki67lo)",
    2: "Epithelium (PD-L1hi/Ki67hi)",
    3: "Macrophage",
    4: "Lymphocyte",
    5: "Vascular",
    6: "Fibroblast/Stroma"
}

# Color mapping for visualization (converted from RGB to hex)
CELL_TYPE_COLORS = {
    0: "#000000",  # Black (RGB: 0, 0, 0) - Undefined
    1: "#387F39",  # Dark Green (RGB: 56, 127, 57) - Epithelium low
    2: "#00FF00",  # Bright Green (RGB: 0, 255, 0) - Epithelium high
    3: "#FC8D62",  # Coral/Salmon (RGB: 252, 141, 98) - Macrophage
    4: "#FFD92F",  # Yellow (RGB: 255, 217, 47) - Lymphocyte
    5: "#4535C1",  # Blue/Purple (RGB: 69, 53, 193) - Vascular
    6: "#17BECF"   # Cyan (RGB: 23, 190, 207) - Fibroblast/Stroma
}


# Load your JSON
with open('/mnt/g/GDC-TCGA-LUAD/00a0b174-1eab-446a-ba8c-7c6e3acd7f0c/pred_00a0b174-1eab-446a-ba8c-7c6e3acd7f0c/json/tile_39520_7904.json', 'r') as f:
    data = json.load(f)

# Extract nuclei information
nuclei_data = []
for nuc_id, nuc_info in data['nuc'].items():
    nuclei_data.append({
        'cell_id': int(nuc_id),
        'x': nuc_info['centroid'][0],
        'y': nuc_info['centroid'][1],
        'cell_type': nuc_info['type'],
        'type_prob': nuc_info['type_prob'],
        'bbox_x_min': nuc_info['bbox'][0][1],
        'bbox_y_min': nuc_info['bbox'][0][0],
        'bbox_x_max': nuc_info['bbox'][1][1],
        'bbox_y_max': nuc_info['bbox'][1][0]
    })

df = pd.DataFrame(nuclei_data)

# ===== Clean the data =====
# Check for missing values
print(f"Total nuclei: {len(df)}")
print(f"Nuclei with NaN cell_type: {df['cell_type'].isna().sum()}")
print(f"Nuclei with NaN type_prob: {df['type_prob'].isna().sum()}")

# Check for unique cell_type values
print(f"Unique cell_type values: {sorted(df['cell_type'].dropna().unique())}")

# Remove rows with missing cell_type (None, NaN)
df_clean = df.dropna(subset=['cell_type']).copy()

# Also filter out string 'nan' values if they exist
if df_clean['cell_type'].dtype == object:
    df_clean = df_clean[df_clean['cell_type'] != 'nan'].copy()

print(f"Nuclei after removing NaN: {len(df_clean)}")

# Ensure cell_type is numeric (convert if needed)
# This handles cases where JSON might have stored numbers as strings
df_clean['cell_type'] = pd.to_numeric(df_clean['cell_type'], errors='coerce')

# Drop any rows where conversion to numeric failed
df_clean = df_clean.dropna(subset=['cell_type']).copy()

# Convert to integer type
df_clean['cell_type'] = df_clean['cell_type'].astype(int)

print(f"Cell type distribution:\n{df_clean['cell_type'].value_counts().sort_index()}")

# Map cell type numbers to labels
df_clean['cell_type_label'] = df_clean['cell_type'].map(CELL_TYPE_DICT)

# Check for unmapped values (values not in CELL_TYPE_DICT)
unmapped_mask = df_clean['cell_type_label'].isna()
if unmapped_mask.sum() > 0:
    print(f"\nWarning: Found {unmapped_mask.sum()} cells with unmapped cell types:")
    print(f"Unmapped cell_type values: {df_clean.loc[unmapped_mask, 'cell_type'].unique()}")
    # Remove rows with unmapped cell types
    df_clean = df_clean[~unmapped_mask].copy()

print(f"Nuclei after cleaning and mapping: {len(df_clean)}")
print(f"\nCell type distribution (labeled):\n{df_clean['cell_type_label'].value_counts()}")
# =====================================

# 2. Create AnnData Object for Squidpy
# Use df_clean instead of df
adata = ad.AnnData(
    X=np.zeros((len(df_clean), 1)),  # Placeholder for expression data
    obs=df_clean.set_index('cell_id')
)

# Store spatial coordinates
adata.obsm['spatial'] = df_clean[['x', 'y']].values

# Store cell type information (use labels for better visualization)
# IMPORTANT: Use .values to avoid index mismatch issues
# df_clean has default integer index, but adata.obs has cell_id index
adata.obs['cell_type'] = pd.Categorical(df_clean['cell_type_label'].values)
adata.obs['cell_type_numeric'] = df_clean['cell_type'].values
adata.obs['type_confidence'] = df_clean['type_prob'].values

# Verify no NaN values in cell_type
print(f"\nVerifying adata.obs['cell_type']:")
print(f"  Total cells: {len(adata.obs)}")
print(f"  NaN values: {adata.obs['cell_type'].isna().sum()}")
print(f"  Unique values: {adata.obs['cell_type'].unique()}")

# Create color palette for cell types (ordered by category)
cell_type_categories = sorted(df_clean['cell_type'].unique())
color_palette = [CELL_TYPE_COLORS[ct] for ct in cell_type_categories]

# 3. Compute Spatial Metrics with Squidpy
# Compute spatial neighbors
sq.gr.spatial_neighbors(adata, coord_type='generic', spatial_key='spatial')

# Compute neighborhood enrichment (cell type interactions)
sq.gr.nhood_enrichment(adata, cluster_key='cell_type')  # err

# Compute centrality scores
sq.gr.centrality_scores(adata, cluster_key='cell_type')

# Compute co-occurrence across spatial dimensions
sq.gr.co_occurrence(adata, cluster_key='cell_type')

# Ripley's statistics for spatial patterns
sq.gr.ripley(adata, cluster_key='cell_type', mode='L')

# 4. Visualization
# Plot spatial distribution with custom colors
# Create a simple scatter plot using matplotlib instead of sq.pl.spatial_scatter
# since we don't have tissue image metadata
fig, ax = plt.subplots(figsize=(10, 10))

# Get coordinates
coords = adata.obsm['spatial']

# Create color mapping
cell_type_to_color = {ct_label: CELL_TYPE_COLORS[ct_num]
                      for ct_num, ct_label in CELL_TYPE_DICT.items()}

# Plot each cell type
for cell_type in adata.obs['cell_type'].cat.categories:
    mask = adata.obs['cell_type'] == cell_type
    ax.scatter(coords[mask, 0], coords[mask, 1],
               c=cell_type_to_color[cell_type],
               label=cell_type,
               s=2,
               alpha=0.8)

ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_title('Spatial Distribution of Cell Types')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=3)
ax.set_aspect('equal')
plt.tight_layout()

# Plot neighborhood enrichment
fig2, ax2 = plt.subplots(figsize=(8, 8))
sq.pl.nhood_enrichment(adata, cluster_key='cell_type', ax=ax2)
plt.tight_layout()

# Plot co-occurrence - let squidpy create its own figure
fig3 = plt.figure(figsize=(10, 6))
sq.pl.co_occurrence(
    adata,
    cluster_key='cell_type'
)
plt.tight_layout()

# Plot Ripley's statistics if computed
fig4 = plt.figure(figsize=(10, 6))
sq.pl.ripley(adata, cluster_key='cell_type')
plt.tight_layout()

plt.show()  # Display all plots