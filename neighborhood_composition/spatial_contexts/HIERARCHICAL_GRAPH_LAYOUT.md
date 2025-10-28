# Hierarchical Row-Based Graph Layout

## Overview

The spatial context interaction graph now uses a **clean hierarchical row-based layout** that organizes SCs by complexity, exactly matching your reference image pattern.

## 🎨 Layout Structure

### Row Organization

SCs are organized into horizontal rows based on the number of CNs:

```
Row 1:  [1]  [2]  [3]  [4]  [5]  [6]  [7]        ← Single-CN contexts
         │    │    │    │    │    │    │
         └────┴────┴────┴────┴────┴────┘
              (connections to Row 2 only)
              
Row 2:  [1_6]  [2_4]  [2_5]  [3_5]  [3_6]  ...  ← Two-CN contexts
         │      │      │      │      │
         ├──────┴──────┴──────┴──────┤
         │    (to Row 1 and Row 3)   │
         
Row 3:  [1_4_6]  [2_4_5]  [3_5_6]  ...           ← Three-CN contexts
         │        │        │
         └────────┴────────┘
         (connections to Row 2 and Row 4 only)
         
Row 4:  [1_4_5_6]  [2_4_5_6]  ...                ← Four+ CN contexts
         (connections to Row 3 only)
```

### **Key Connection Rules** 🔗

✅ **Adjacent rows only**: Nodes connect ONLY to adjacent rows
- Row 1 ↔ Row 2
- Row 2 ↔ Row 1 and Row 3
- Row 3 ↔ Row 2 and Row 4
- etc.

❌ **No same-row connections**: Nodes in the same row do NOT connect
- `1` does NOT connect to `2`, `3`, `4`, etc.

❌ **No row-skipping**: Nodes cannot skip rows
- Row 1 does NOT connect directly to Row 3
- Row 2 does NOT connect directly to Row 4

### Why This Connection Pattern?

This design reflects the **biological meaning** of spatial contexts:

1. **Same-row nodes** represent **alternative states** at the same complexity level
   - They don't spatially interact (they're mutually exclusive)
   - Example: `1` and `2` are both single-CN "pure" neighborhoods

2. **Adjacent-row connections** show **evolutionary transitions**
   - How simple contexts evolve into complex ones
   - Example: `1` → `1_6` shows CN1 gaining CN6 influence

3. **No row-skipping** because transitions are **gradual**
   - Contexts don't jump from 1 CN to 3 CNs directly
   - They transition through 2-CN intermediate states

### Key Features

✅ **Hierarchical Organization**
- Top row: Simplest contexts (1 CN)
- Bottom row: Most complex contexts (4+ CNs)
- Clear visual hierarchy

✅ **Clean Label Placement**
- Labels positioned **above** each circle
- White background boxes for readability
- No text inside circles

✅ **Even Spacing**
- Nodes evenly distributed across each row
- 8% margins on left and right
- Proper vertical spacing (0.9 to 0.1)

✅ **Color-Coded by Cell Count**
- Viridis colormap (purple → yellow)
- Darker = fewer cells
- Brighter = more cells

✅ **Filtered Edges**
- Only adjacent-row connections shown
- Cleaner, more meaningful visualization
- Easier to understand interaction patterns

## Visual Elements

### Node Appearance

| Element | Description |
|---------|-------------|
| **Shape** | Circles with black borders (2pt width) |
| **Size** | Proportional to cell count (500-2500 range) |
| **Color** | Viridis gradient based on n_cells |
| **Border** | Black, 2pt linewidth for definition |
| **Opacity** | 85% for slight transparency |

### Label Appearance

| Element | Description |
|---------|-------------|
| **Position** | Above each node with 0.08 offset |
| **Background** | White rounded box with black border |
| **Font** | Bold, 9pt, black text |
| **Alignment** | Center-aligned, bottom-anchored |

### Edge Appearance

| Element | Description |
|---------|-------------|
| **Width** | Variable (0.5-4pt) based on interactions |
| **Color** | Black with 30% transparency |
| **Style** | Solid lines |
| **Z-order** | Behind nodes (drawn first) |

## Example Structure

For a dataset with 7 CNs and typical complexity:

```
═══════════════════════════════════════════════════════════
    [1]        [2]        [3]        [4]        [5]    [6]    [7]
     ●          ●          ●          ●          ●      ●      ●
─────────────────────────────────────────────────────────────
  [1_6]    [2_4]    [2_5]    [2_6]    [3_5]    [3_6]    [4_5]
    ●        ●        ●        ●        ●        ●        ●
─────────────────────────────────────────────────────────────
 [1_4_6]  [1_5_6]  [2_4_5]  [2_4_6]  [2_5_6]  [3_4_5]
    ●        ●        ●        ●        ●        ●
─────────────────────────────────────────────────────────────
[1_4_5_6] [2_4_5_6]
    ●        ●
═══════════════════════════════════════════════════════════
```

## Usage

### Default (Automatic Hierarchical Layout)

```bash
python spatial_contexts.py \
  --cn_results_dir cn_unified_results \
  --output_dir sc_unified_results \
  --threshold 0.6 \
  --min_fraction 0.05
```

The script now **automatically uses hierarchical layout** by default.

### Custom Parameters

```python
from spatial_contexts import SpatialContextDetector

# ... load data ...
detector = SpatialContextDetector(adata)
detector.run_full_pipeline(
    k=40,
    threshold=0.6,
    min_fraction=0.05,
    output_dir='sc_results'
)

# Or customize the graph separately
detector.plot_sc_graph(
    figsize=(20, 12),        # Larger canvas for more nodes
    node_size_scale=1.2,     # Slightly larger nodes
    save_path='custom_sc_graph.png'
)
```

## Layout Algorithm

### Step 1: Group by Complexity
```python
layers = {
    1: ['1', '2', '3', '4', '5', '6', '7'],
    2: ['1_6', '2_4', '2_5', '3_5', '3_6', '4_5'],
    3: ['1_4_6', '2_4_5', '2_4_6', '3_4_5'],
    4: ['1_4_5_6', '2_4_5_6']
}
```

### Step 2: Calculate Y Positions
```python
n_layers = len(layers)
y_spacing = 2.0 / (n_layers + 1)

for layer_idx, (n_cns, nodes) in enumerate(sorted(layers.items())):
    y = 1.0 - (layer_idx + 1) * y_spacing
```

### Step 3: Calculate X Positions
```python
n_nodes = len(nodes)
margin = 0.1
x_spacing = (1.0 - 2 * margin) / (n_nodes - 1)
x_positions = [margin + i * x_spacing for i in range(n_nodes)]
```

### Step 4: Position Labels
```python
for node in nodes:
    x, y = pos[node]
    label_y = y + 0.08  # Position above node
```

## Comparison: Before vs After

### Before (Chaotic)
- ❌ Random spring layout
- ❌ Labels inside circles
- ❌ Overlapping nodes
- ❌ No clear organization
- ❌ Hard to read

### After (Clean)
- ✅ Organized rows by complexity
- ✅ Labels above circles in white boxes
- ✅ Even spacing, no overlap
- ✅ Clear hierarchical structure
- ✅ Publication-ready

## Benefits

### 1. **Instant Understanding**
- Row position indicates complexity
- Quick identification of simple vs complex SCs
- Easy to find specific SC by CN composition

### 2. **Clean Aesthetics**
- Professional, publication-ready appearance
- No cluttered labels
- Clear node boundaries

### 3. **Scalability**
- Works with any number of SCs
- Automatically adjusts spacing
- Handles 5-50+ nodes gracefully

### 4. **Pattern Recognition**
- Easy to see which rows have more/fewer SCs
- Clear visualization of complexity distribution
- Interaction patterns between rows visible

## Customization Options

### Adjust Figure Size
```python
# For many nodes (wide figure)
detector.plot_sc_graph(figsize=(24, 10))

# For fewer nodes (compact)
detector.plot_sc_graph(figsize=(14, 8))
```

### Adjust Node Sizes
```python
# Larger nodes (better for presentations)
detector.plot_sc_graph(node_size_scale=1.5)

# Smaller nodes (more compact)
detector.plot_sc_graph(node_size_scale=0.8)
```

### Adjust Label Position
If labels overlap with edges, modify the offset in the code:
```python
# In plot_sc_graph method, line ~758:
offset = 0.10  # Increase for more space above nodes
```

### Change Colormap
```python
# In plot_sc_graph method, line ~743:
cmap='plasma'  # or 'inferno', 'magma', 'cividis'
```

## Technical Details

### Position Calculation
- **X-axis**: 0.0 to 1.0 (with 10% margins → 0.1 to 0.9)
- **Y-axis**: 1.0 (top) to 0.0 (bottom)
- **Spacing**: Dynamically calculated based on number of layers and nodes

### Size Normalization
```python
# Cell count → normalized value → scaled size
normalized = (cell_count - min_cells) / (max_cells - min_cells)
size = 500 + normalized * 2000  # Range: 500-2500
```

### Color Normalization
```python
# Same normalization for viridis colormap
normalized = (cell_count - min_cells) / (max_cells - min_cells)
color = viridis(normalized)  # 0=purple, 1=yellow
```

## Troubleshooting

### Labels Overlap with Nodes Below
**Solution**: Increase figure height or adjust offset
```python
figsize=(16, 12)  # Taller figure
# or modify offset in code: offset = 0.10
```

### Too Many Nodes in One Row
**Solution**: Increase figure width
```python
figsize=(24, 10)  # Wider figure
```

### Edges Too Thick/Thin
**Solution**: Adjust edge width calculation
```python
# In code, modify line ~720:
edge_width = 0.3 + normalized * 2.5  # Thinner edges
```

### Nodes Too Small/Large
**Solution**: Adjust size scale
```python
detector.plot_sc_graph(node_size_scale=1.5)  # Larger
detector.plot_sc_graph(node_size_scale=0.7)  # Smaller
```

## Related Files

- `spatial_contexts_unified.py` - Main implementation
- `GRAPH_VISUALIZATION_IMPROVEMENTS.md` - Feature overview
- `SC_DETECTION_FIX.md` - SC detection algorithm improvements

---

**Date**: 2025-10-27  
**Version**: 3.0  
**Status**: Production-ready  
**Layout**: Hierarchical row-based (default)
