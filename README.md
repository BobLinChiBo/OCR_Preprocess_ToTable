# OCR Table Extraction Pipeline

A professional two-stage OCR pipeline for extracting tables from scanned document images with high accuracy. Uses advanced computer vision techniques for page splitting, deskewing, ROI detection, and table line detection.

## Features

- **Two-Stage Processing**: Initial aggressive processing followed by precision refinement
- **Automatic Page Splitting**: Intelligently separates double-page scanned documents
- **Advanced Deskewing**: Corrects document rotation with sub-degree precision
- **ROI Detection**: Multiple algorithms for content area identification
- **Table Line Detection**: Robust detection of horizontal and vertical table structures
- **Comprehensive Tuning Tools**: Interactive parameter optimization system
- **Visualization Suite**: Complete analysis and debugging capabilities
- **Publication-Ready Output**: Final tables optimized for academic and professional use

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/BobLinChiBo/OCR_Preprocess_ToTable.git
cd OCR_Preprocess_ToTable

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Process all images in a directory (complete two-stage pipeline)
python scripts/run_complete.py data/input/ --verbose

# Process a single image
python scripts/run_complete.py image.jpg --verbose

# Stage 1 only (initial processing)
python scripts/run_stage1.py data/input/ --verbose

# Stage 2 only (refinement)
python scripts/run_stage2.py --verbose
```

### Interactive Parameter Tuning

For best results, tune parameters on your specific document types:

```bash
# Quick interactive tuning (recommended)
python tools/quick_start_tuning.py

# Manual step-by-step tuning
python tools/setup_tuning.py
python tools/tune_page_splitting.py
python tools/tune_deskewing.py
python tools/tune_roi_detection.py
python tools/tune_line_detection.py
```

## Pipeline Overview

### Stage 1: Initial Processing
Aggressive parameters for robust extraction from challenging scanned documents:

1. **Page Splitting** - Separate double-page scans using gutter detection
2. **Deskewing** - Correct rotation up to ±10° with 0.2° precision
3. **ROI Detection** - Identify content areas using Gabor filters or edge detection
4. **Line Detection** - Find table structures with permissive parameters
5. **Table Cropping** - Extract table regions for Stage 2

**Output**: `data/output/stage1_initial_processing/05_cropped_tables/`

### Stage 2: Refinement Processing
Precise parameters for publication-quality table extraction:

1. **Re-deskewing** - Fine-tune rotation on cropped tables
2. **Refined Line Detection** - Precise detection with stricter parameters
3. **Table Reconstruction** - Advanced structure analysis
4. **Table Fitting** - Final optimization for clean output

**Final Output**: `data/output/stage2_refinement/04_fitted_tables/`

## Directory Structure

```
OCR_Preprocess_ToTable/
├── src/ocr_pipeline/          # Core pipeline modules
│   ├── pipeline.py            # Main pipeline classes
│   ├── config.py              # Configuration management
│   └── utils.py               # Image processing utilities
├── scripts/                   # CLI entry points
│   ├── run_complete.py        # Two-stage pipeline
│   ├── run_stage1.py          # Stage 1 only
│   └── run_stage2.py          # Stage 2 only
├── tools/                     # Parameter tuning and visualization
│   ├── quick_start_tuning.py  # Interactive tuning guide
│   ├── tune_*.py             # Individual parameter tuning
│   ├── visualize_*.py        # Analysis and debugging tools
│   └── run_visualizations.py # Complete visualization suite
├── configs/                   # Configuration files
│   ├── stage1_default.json   # Stage 1 default parameters
│   └── stage2_default.json   # Stage 2 default parameters
├── data/
│   ├── input/                # Input images
│   └── output/               # Processing results
└── docs/                     # Documentation
```

## Configuration

The pipeline uses JSON configuration files with hierarchical parameter structures:

### Key Parameters

- **Page Splitting**: `gutter_search_start` (0.35-0.45), `gutter_search_end` (0.55-0.65)
- **Deskewing**: `angle_range` (5-20°), `min_angle_correction` (0.1-1.0°)
- **ROI Detection**: `gabor_binary_threshold` (90-180), `roi_min_cut_strength` (10-30)
- **Line Detection**: `min_line_length` (20-80), `max_line_gap` (5-25)

### ROI Detection Methods

- `gabor`: Gabor filter-based edge detection (default)
- `canny_sobel`: Combined Canny and Sobel edge detection
- `adaptive_threshold`: Adaptive thresholding with edge enhancement

## Advanced Usage

### Custom Configuration

```bash
# Use custom configuration files
python scripts/run_complete.py data/input/ --stage1-config configs/my_stage1.json

# Override specific parameters
python scripts/run_complete.py data/input/ --s1-angle-range 15 --s1-min-line-length 50
```

### Visualization and Analysis

```bash
# Complete pipeline analysis
python tools/run_visualizations.py all --pipeline image.jpg --save-intermediates

# Individual step analysis
python tools/visualize_deskew.py image.jpg --angle-range 30
python tools/visualize_roi.py image.jpg --gabor-threshold 150
python tools/visualize_table_lines.py image.jpg --min-line-length 40

# Results management
python tools/check_results.py list
python tools/compare_results.py
```

### Development

```bash
# Run tests
python -m pytest tests/ -v

# Code formatting (optional dev dependencies)
python -m black src/ scripts/ tools/
python -m flake8 src/ scripts/ tools/
python -m mypy src/
```

## Best Practices

1. **Always tune parameters** on representative test images before processing large datasets
2. **Visually inspect** intermediate results at each stage
3. **Use debug mode** (`--debug`) for troubleshooting
4. **Start with test images** - process 6 representative samples first
5. **Check intermediate outputs** if final results are poor

## Troubleshooting

### Common Issues

- **Poor page splitting**: Adjust gutter search parameters based on document layout
- **Excessive rotation**: Increase `min_angle_correction` for more conservative deskewing
- **Over-cropping**: Increase `roi_min_cut_strength` for less aggressive ROI detection
- **Missing table lines**: Decrease `min_line_length` or increase `max_line_gap`

### Getting Help

- Use `--verbose` flag for detailed processing information
- Check `docs/CLAUDE.md` for comprehensive development guidance
- Refer to `tools/README.md` for detailed tuning instructions
- Examine intermediate outputs in debug mode

## Requirements

- Python 3.8+
- OpenCV 4.5.0+
- NumPy 1.20.0+
- Pillow 8.0.0+

See `requirements.txt` for complete dependency list.

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Academic Use

This pipeline is designed for academic and research applications. When using in publications, please consider citing this repository and any relevant computer vision techniques employed.