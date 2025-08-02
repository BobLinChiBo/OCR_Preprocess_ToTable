# Tools Directory

Visualization and analysis tools for the OCR Table Extraction Pipeline. These tools help analyze pipeline performance and debug processing issues.

## 🆕 Version 2 Architecture

The visualization tools now use a new processor-based architecture (v2) that provides better maintainability and consistency. The v2 scripts centralize parameter handling and reduce code duplication.

### Using V2 Scripts

```bash
# Use v2 scripts via run_visualizations.py
python tools/run_visualizations.py all --test-images --use-v2

# Or call v2 scripts directly
python tools/visualize_page_split_v2.py image.jpg
python tools/visualize_margin_removal_v2.py image.jpg --compare
python tools/visualize_deskew_v2.py image.jpg
python tools/visualize_table_lines_v2.py image.jpg
python tools/visualize_table_crop_v2.py image.jpg
```

### Migration from V1 to V2

⚠️ **Deprecation Notice**: V1 scripts are deprecated and will be removed in a future version. Please migrate to v2 scripts.

**Key differences in V2:**
- Centralized parameter mapping in `processor_wrappers.py`
- Unified configuration loading via `config_utils.py`
- Better consistency across all visualization scripts
- Reduced maintenance burden when utils.py changes

**Migration examples:**
```bash
# Old (V1)
python tools/visualize_margin_removal.py image.jpg
python tools/visualize_margin_removal_fast.py image.jpg
python tools/visualize_margin_removal_bbox.py image.jpg

# New (V2) - single script with method selection
python tools/visualize_margin_removal_v2.py image.jpg --method aggressive
python tools/visualize_margin_removal_v2.py image.jpg --method bounding_box --use-optimized
python tools/visualize_margin_removal_v2.py image.jpg --compare  # Compare all methods
```

## Available Tools

### Visualization Suite

#### Complete Analysis
```bash
# Run all visualizations for comprehensive analysis (V2 recommended)
python tools/run_visualizations.py all --pipeline image.jpg --use-v2

# Process test images with v2 scripts
python tools/run_visualizations.py all --test-images --use-v2

# Run pipeline mode with v2 scripts
python tools/run_visualizations.py page-split margin-removal deskew table-lines table-crop --test-images --pipeline --use-v2
```

#### Individual Visualization Scripts

**Page Splitting Analysis**
```bash
# Analyze gutter detection and page separation
python tools/visualize_page_split.py image.jpg
python tools/visualize_page_split.py image.jpg --save-debug
```

**Deskewing Analysis**
```bash
# Analyze rotation detection and correction
python tools/visualize_deskew.py image.jpg
python tools/visualize_deskew.py image.jpg --save-intermediates
```

**ROI Detection Visualization**
```bash
# Analyze region of interest detection (margin removal)
python tools/visualize_roi.py image.jpg
python tools/visualize_roi.py image.jpg --save-debug
```

**Table Line Detection**
```bash
# Analyze table structure detection
python tools/visualize_table_lines.py image.jpg
python tools/visualize_table_lines.py image.jpg --save-debug
```

**Table Cropping Visualization**
```bash
# Analyze final table extraction
python tools/visualize_table_crop.py image.jpg
python tools/visualize_table_crop.py image.jpg --show-boundaries
```

## Utilities

### Results Management
```bash
# List all available results
python tools/check_results.py list

# Clean up old results
python tools/check_results.py cleanup --older-than 7d

# View specific results
python tools/check_results.py view latest
```

## Tool Categories

### Visualization and Analysis

#### V2 Scripts (Recommended)
- `run_visualizations.py --use-v2` - Master visualization runner with v2 architecture
- `visualize_page_split_v2.py` - Page splitting with PageSplitProcessor
- `visualize_margin_removal_v2.py` - Unified margin removal (all methods)
- `visualize_deskew_v2.py` - Deskewing with DeskewProcessor
- `visualize_table_lines_v2.py` - Table lines with TableLineProcessor
- `visualize_table_crop_v2.py` - Table cropping with TableCropProcessor

#### V1 Scripts (Deprecated)
- `visualize_page_split.py` - ⚠️ Deprecated - use v2
- `visualize_deskew.py` - ⚠️ Deprecated - use v2
- `visualize_margin_removal.py` - ⚠️ Deprecated - use v2
- `visualize_margin_removal_fast.py` - ⚠️ Deprecated - use v2 with --use-optimized
- `visualize_margin_removal_bbox.py` - ⚠️ Deprecated - use v2 with --method bounding_box
- `visualize_table_lines.py` - ⚠️ Deprecated - use v2
- `visualize_table_crop.py` - ⚠️ Deprecated - use v2

### Results Management
- `check_results.py` - Results exploration and cleanup utilities
- `output_manager.py` - Output directory management utilities

## Usage Patterns

### New Document Analysis
```bash
# Complete analysis workflow for new document type
python tools/run_visualizations.py all --pipeline representative_image.jpg
python tools/visualize_page_split.py representative_image.jpg --save-debug
python tools/visualize_deskew.py representative_image.jpg --save-intermediates
```

### Troubleshooting Poor Results
```bash
# Analyze each stage individually
python tools/visualize_page_split.py problematic_image.jpg --save-debug
python tools/visualize_deskew.py problematic_image.jpg
python tools/visualize_roi.py problematic_image.jpg
python tools/visualize_table_lines.py problematic_image.jpg --save-debug
```

### Batch Analysis
```bash
# Run visualizations on multiple representative images
python tools/run_visualizations.py all --pipeline image1.jpg
python tools/run_visualizations.py all --pipeline image2.jpg
```

## Output Organization

All tools organize outputs in structured directories:

```
data/output/
├── visualization/             # Visualization outputs
│   ├── page_split/           # Page splitting analysis
│   ├── deskew/               # Deskewing analysis  
│   ├── roi/                  # ROI detection analysis
│   ├── table_lines/          # Line detection analysis
│   └── table_crop/           # Table cropping analysis
└── debug/                    # Debug outputs when --save-debug used
```

## Integration with Main Pipeline

The visualization tools integrate seamlessly with the main pipeline:

- **Parameter Analysis**: Visualizations help understand how current parameters affect processing
- **Debugging**: Debug outputs help identify where processing issues occur
- **Quality Assessment**: Visual outputs help assess pipeline performance on different document types

## Best Practices

### Visualization Strategy

1. **Start with complete analysis** for comprehensive understanding
2. **Focus on problem areas** with individual visualization tools
3. **Save intermediate results** when debugging specific issues
4. **Use representative samples** for document type analysis

### Performance Optimization

1. **Use selective visualization** - only run needed analysis tools
2. **Use debug mode selectively** - generates large files
3. **Clean up outputs regularly** to manage disk space
4. **Process representative samples** before full datasets

### Common Workflows

**New Document Type Analysis:**
```bash
python tools/run_visualizations.py all --pipeline sample_image.jpg
# Review outputs to understand document characteristics
# Focus on problematic stages with individual tools
```

**Performance Debugging:**
```bash
python tools/visualize_page_split.py problem_image.jpg --save-debug
python tools/visualize_deskew.py problem_image.jpg
# Continue through pipeline stages as needed
```

**Results Management:**
```bash
python tools/check_results.py list
python tools/check_results.py cleanup --older-than 3d
```

## Requirements

- All visualization tools require the main pipeline dependencies
- matplotlib and other visualization libraries (automatically installed with main requirements)
- Sufficient disk space for debug outputs when using --save-debug options

For detailed pipeline usage, see the main project [README.md](../README.md) and [docs/CLAUDE.md](../docs/CLAUDE.md).