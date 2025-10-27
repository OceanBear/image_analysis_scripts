# Spatial Context Interaction Graph Improvements

## Overview

The SC interaction graph visualization has been significantly improved to match the aesthetic quality of published research figures.

## Key Improvements

### 1. **Enhanced Node Visualization**
- **Color Gradient**: Nodes now use a viridis colormap based on cell count
  - Low cell count: Purple/dark colors
  - High cell count: Yellow/bright colors
- **Size Scaling**: Better node size range (300-3000) for clear differentiation
- **Edge Styling**: Black borders with 1.5pt width for definition
- **Alpha**: Set to 0.9 for solid appearance

### 2. **Improved Edge Rendering**
- **Dynamic Width**: Edge width scales with interaction frequency (0.5-5pt)
- **Transparency**: 40% alpha for cleaner look
- **Color**: Black edges for clear contrast
- **Z-order**: Edges drawn first (behind nodes)

### 3. **Better Layout Options**
Added new **'hierarchical'** layout option that organizes nodes by complexity:
- Layer 1: Single-CN spatial contexts (e.g., `1`, `2`, `3`)
- Layer 2: Two-CN spatial contexts (e.g., `1_2`, `3_5`)
- Layer 3: Three-CN spatial contexts (e.g., `2_4_6`)
- Layer N: Complex multi-CN spatial contexts

Other layouts available:
- `'spring'`: Force-directed (default, improved with k=3, iterations=100)
- `'kamada_kawai'`: Energy-based minimization
- `'circular'`: Circular arrangement

### 4. **Enhanced Legends and Annotations**
- **Colorbar**: Shows n_cells gradient on the right side
- **n_group Legend**: Shows node size by number of CNs
  - Size increases with complexity (1 CN < 2 CNs < 3 CNs < 4+ CNs)
- **White Background**: Clean, publication-ready appearance

### 5. **Typography Improvements**
- **Labels**: Bold, white text for high contrast
- **Font Size**: Optimized at 10pt for readability
- **Legend**: Styled with shadow and frame for professional look

## Usage

### Basic Usage (Default Spring Layout)
```bash
python spatial_contexts.py \
  --cn_results_dir cn_unified_results \
  --output_dir sc_unified_results \
  --threshold 0.6 \
  --min_fraction 0.05
```

### Hierarchical Layout (Best for Layered Structure)
```python
from spatial_contexts import SpatialContextDetector

# ... load data ...
detector = SpatialContextDetector(adata, cn_key='cn_celltype')

# Run pipeline with hierarchical layout
detector.run_full_pipeline(
    k=40,
    threshold=0.6,
    min_fraction=0.05,
    output_dir='sc_results',
    graph_layout='hierarchical'  # Use hierarchical layout
)
```

### Custom Visualization Parameters
```python
# After running the pipeline, customize the graph
detector.plot_sc_graph(
    layout='hierarchical',      # or 'spring', 'kamada_kawai', 'circular'
    figsize=(16, 14),           # Larger figure
    node_size_scale=2.0,        # Bigger nodes
    save_path='custom_graph.png'
)
```

## Layout Comparison

| Layout | Best For | Pros | Cons |
|--------|----------|------|------|
| **spring** | General use, moderate complexity | Natural clustering, good spacing | Can be messy with many nodes |
| **hierarchical** | Showing CN complexity layers | Clear organization by complexity | Fixed structure |
| **kamada_kawai** | Small-medium graphs | Optimal edge crossing | Slow for large graphs |
| **circular** | Equal emphasis on all SCs | Simple, symmetric | Poor for showing relationships |

## Visual Elements Explained

### Node Properties
- **Size**: Proportional to number of cells in that SC
- **Color**: Gradient from purple (few cells) to yellow (many cells)
- **Border**: Black edge for definition
- **Label**: SC identifier (e.g., `1_6`, `2_4_5`)

### Edge Properties
- **Width**: Thicker edges = more spatial interactions between SCs
- **Color**: Black for clear visibility
- **Transparency**: 40% to avoid visual clutter

### Color Scale (Viridis)
```
Purple/Dark Blue → Teal → Green → Yellow
  Few cells              →        Many cells
```

## Command Line Options

### Standard Run
```bash
python spatial_contexts.py \
  --cn_results_dir cn_unified_results \
  --graph_layout spring
```

### Available Layouts
```bash
--graph_layout spring         # Default, force-directed
--graph_layout hierarchical   # Layered by CN complexity
--graph_layout kamada_kawai   # Energy minimization
--graph_layout circular       # Circular arrangement
```

## Technical Details

### Node Size Calculation
```python
# Range: 300-3000 (base) * node_size_scale (1.5)
normalized = (cell_count - min_cells) / (max_cells - min_cells)
size = (300 + normalized * 2700) * 1.5
```

### Edge Width Calculation
```python
# Range: 0.5-5.0
normalized = (interactions - min_int) / (max_int - min_int)
width = 0.5 + normalized * 4.5
```

### Color Mapping
```python
# Viridis colormap: 0 (purple) to 1 (yellow)
normalized_value = (cell_count - min_cells) / (max_cells - min_cells)
color = viridis_colormap(normalized_value)
```

## Example Output

The improved graph will show:
1. **Clear node differentiation** by size and color
2. **Hierarchical structure** (if using hierarchical layout)
3. **Interaction patterns** through edge thickness
4. **Publication-quality** appearance with legends and labels

## Troubleshooting

### Nodes too small/large?
Adjust `node_size_scale` parameter:
```python
detector.plot_sc_graph(node_size_scale=2.5)  # Larger nodes
```

### Graph too cluttered?
Try hierarchical layout or increase figure size:
```python
detector.plot_sc_graph(
    layout='hierarchical',
    figsize=(18, 16)
)
```

### Need different colors?
Modify the colormap in the code:
```python
# In plot_sc_graph method, change:
cmap='viridis'  # to 'plasma', 'inferno', 'magma', etc.
```

### Labels overlapping?
Increase figure size or reduce font size:
```python
# In the code, modify:
font_size=8  # Smaller font
figsize=(16, 14)  # Larger canvas
```

## Comparison: Before vs After

### Before
- Simple light blue nodes
- Uniform node sizes
- Gray edges with uniform width
- Text box legend
- No colorbar

### After
- ✓ Color gradient based on cell count
- ✓ Proportional node sizing (300-3000 range)
- ✓ Variable edge widths based on interactions
- ✓ Professional legend with n_group sizes
- ✓ Colorbar showing n_cells scale
- ✓ White background for publication
- ✓ Hierarchical layout option
- ✓ Better spacing (k=3, iterations=100)

## Related Files

- `spatial_contexts_unified.py` - Main script with improved visualization
- `SC_DETECTION_FIX.md` - Information about SC detection algorithm fix
- `sc_interaction_graph.png` - Generated output

---

**Date**: 2025-10-27
**Version**: 2.0
**Status**: Production-ready
