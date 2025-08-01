# Tools Directory

A comprehensive suite of parameter tuning, visualization, and analysis tools for the OCR Table Extraction Pipeline. These tools help optimize pipeline performance for your specific document types and provide detailed insights into processing results.

## Quick Start

### Interactive Guided Tuning (Recommended)

```bash
# Start the interactive tuning process
python tools/quick_start_tuning.py
```

This script guides you through the complete parameter optimization process with prompts and assistance at each stage.

### Manual Step-by-Step Tuning

For more control over the tuning process:

```bash
# 1. Initial setup (one-time)
python tools/setup_tuning.py

# 2. Tune each stage individually
python tools/tune_page_splitting.py      # Stage 1: Page separation
python tools/tune_deskewing.py           # Stage 2: Rotation correction
python tools/tune_roi_detection.py       # Stage 3: Content area detection
python tools/tune_line_detection.py      # Stage 4: Table line detection

# 3. Test optimized parameters
python tools/run_tuned_pipeline.py data/input/ --verbose
```

## Visualization Tools

### Complete Analysis Suite

```bash
# Run all visualizations for comprehensive analysis
python tools/run_visualizations.py all --pipeline image.jpg --save-intermediates

# Run specific visualization categories
python tools/run_visualizations.py preprocessing image.jpg
python tools/run_visualizations.py detection image.jpg
python tools/run_visualizations.py extraction image.jpg
```

### Individual Visualization Scripts

#### Page Splitting Analysis
```bash
# Analyze gutter detection and page separation
python tools/visualize_page_split.py image.jpg --gutter-start 0.35 --gutter-end 0.65
python tools/visualize_page_split.py image.jpg --show-histogram --save-debug
```

#### Deskewing Analysis
```bash
# Analyze rotation detection and correction
python tools/visualize_deskew.py image.jpg --angle-range 30
python tools/visualize_deskew.py image.jpg --method histogram --save-intermediates
python tools/visualize_deskew.py image.jpg --show-line-detection
```

#### ROI Detection Visualization
```bash
# Analyze region of interest detection
python tools/visualize_roi.py image.jpg --method gabor --threshold 150
python tools/visualize_roi.py image.jpg --method canny_sobel --save-debug
python tools/visualize_roi.py image.jpg --show-projections
```

#### Table Line Detection
```bash
# Analyze table structure detection
python tools/visualize_table_lines.py image.jpg --min-line-length 40
python tools/visualize_table_lines.py image.jpg --method probabilistic --save-debug
python tools/visualize_table_lines.py image.jpg --show-preprocessing
```

#### Table Cropping Visualization
```bash
# Analyze final table extraction
python tools/visualize_table_crop.py image.jpg --margin 10
python tools/visualize_table_crop.py image.jpg --show-boundaries
```

## Parameter Tuning Tools

### Page Splitting Optimization
Tests different gutter detection parameters for double-page document separation:

```bash
python tools/tune_page_splitting.py
```

**Key Parameters:**
- `gutter_search_start`: Start of gutter search range (0.35-0.45)
- `gutter_search_end`: End of gutter search range (0.55-0.65)
- `min_gutter_width`: Minimum gutter width threshold

### Deskewing Optimization
Optimizes rotation correction parameters:

```bash
python tools/tune_deskewing.py
```

**Key Parameters:**
- `angle_range`: Maximum rotation angle to test (5-20°)
- `angle_step`: Angle increment precision (0.1-0.5°)
- `min_angle_correction`: Minimum angle threshold for correction

### ROI Detection Optimization
Tunes content area detection algorithms:

```bash
python tools/tune_roi_detection.py
```

**Key Parameters:**
- `method`: Detection algorithm (`gabor`, `canny_sobel`, `adaptive_threshold`)
- `gabor_binary_threshold`: Gabor filter threshold (90-180)
- `roi_min_cut_strength`: Minimum boundary strength (10-30)

### Line Detection Optimization
Optimizes table structure detection:

```bash
python tools/tune_line_detection.py
```

**Key Parameters:**
- `min_line_length`: Minimum line length to detect (20-80 pixels)
- `max_line_gap`: Maximum gap to bridge in lines (5-25 pixels)
- `line_detection_method`: Algorithm choice (`hough`, `probabilistic`)

## Results Management Tools

### Results Analysis
```bash
# List all available results
python tools/check_results.py list

# View specific results
python tools/check_results.py view latest
python tools/check_results.py view "2024-01-15_14:30:45"

# Clean up old results
python tools/check_results.py cleanup --older-than 7d
```

### Results Comparison
```bash
# Compare different parameter sets
python tools/compare_results.py

# Compare specific runs
python tools/compare_results.py --run1 "2024-01-15_14:30:45" --run2 "2024-01-15_15:45:22"

# Generate comparison report
python tools/compare_results.py --export-report comparison_report.json
```

## Configuration Management

### Test Configuration
```bash
# Validate configuration files
python tools/test_config_loading.py

# Test custom configurations
python tools/test_config_loading.py --config configs/my_custom_config.json
```

### Tuned Pipeline Execution
```bash
# Run pipeline with optimized parameters
python tools/run_tuned_pipeline.py data/input/ --verbose

# Use specific tuned configuration
python tools/run_tuned_pipeline.py data/input/ --config tuned_params.json
```

## Tool Categories

### Setup and Initialization
- `setup_tuning.py` - One-time setup for parameter tuning
- `setup_tuning_windows.py` - Windows-specific setup script

### Parameter Optimization
- `tune_page_splitting.py` - Page separation parameter tuning
- `tune_deskewing.py` - Rotation correction optimization
- `tune_roi_detection.py` - Content area detection tuning
- `tune_line_detection.py` - Table line detection optimization

### Visualization and Analysis
- `run_visualizations.py` - Master visualization runner
- `visualize_page_split.py` - Page splitting analysis
- `visualize_deskew.py` - Deskewing analysis
- `visualize_roi.py` - ROI detection visualization
- `visualize_table_lines.py` - Table line detection analysis
- `visualize_table_crop.py` - Final table extraction visualization

### Results Management
- `check_results.py` - Results exploration and cleanup
- `compare_results.py` - Parameter comparison and analysis
- `output_manager.py` - Output directory management utilities

### Pipeline Execution
- `quick_start_tuning.py` - Interactive guided tuning
- `run_tuned_pipeline.py` - Execute pipeline with optimized parameters
- `test_config_loading.py` - Configuration validation

## Best Practices

### Parameter Tuning Workflow

1. **Start with representative test images** (6 diverse samples)
2. **Use interactive tuning** for first-time users
3. **Tune stages sequentially** - each stage depends on previous ones
4. **Visually inspect results** at each parameter setting
5. **Document successful parameter combinations** for future use

### Visualization Strategy

1. **Use complete analysis suite** for comprehensive understanding
2. **Focus on problem areas** with individual visualization tools
3. **Save intermediate results** when debugging issues
4. **Compare different parameter sets** side-by-side

### Performance Optimization

1. **Test on small sample sets** before processing large datasets
2. **Use debug mode selectively** - generates large files
3. **Clean up tuning outputs regularly** to manage disk space
4. **Document optimal parameters** for different document types

## Common Use Cases

### New Document Type
```bash
# Complete parameter optimization for new document type
python tools/quick_start_tuning.py
python tools/run_visualizations.py all --pipeline representative_image.jpg
```

### Troubleshooting Poor Results
```bash
# Analyze each stage individually
python tools/visualize_page_split.py problematic_image.jpg --save-debug
python tools/visualize_deskew.py problematic_image.jpg --show-line-detection
python tools/visualize_roi.py problematic_image.jpg --method gabor --show-projections
```

### Performance Comparison
```bash
# Compare different parameter sets
python tools/tune_roi_detection.py  # Try different ROI methods
python tools/compare_results.py     # Compare results quantitatively
```

### Batch Processing Preparation
```bash
# Optimize on test set, then process full dataset
python tools/run_tuned_pipeline.py data/input/test_images/ --verbose
python tools/run_tuned_pipeline.py data/input/raw_images/ --config optimized_params.json
```

## Output Organization

All tools organize outputs in structured directories:

```
data/output/
├── tuning/                    # Parameter optimization results
│   ├── 01_split_pages/       # Page splitting tests
│   ├── 02_deskewed/          # Deskewing tests
│   ├── 03_roi_detection/     # ROI detection tests
│   └── 04_line_detection/    # Line detection tests
├── visualization/             # Visualization outputs
│   └── pipeline_YYYYMMDD_HHMMSS/
└── comparison/               # Results comparison data
```

## Integration with Main Pipeline

The tools directory integrates seamlessly with the main pipeline:

- **Parameter files** generated by tuning tools are compatible with main pipeline configs
- **Visualization outputs** help validate main pipeline results
- **Optimized parameters** can be directly used in production processing

For detailed pipeline usage, see the main project [README.md](../README.md) and [docs/CLAUDE.md](../docs/CLAUDE.md).