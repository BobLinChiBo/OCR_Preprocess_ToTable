# OCR Table Extraction Pipeline

A professional two-stage OCR pipeline for extracting tables from scanned document images with high accuracy. Uses advanced computer vision techniques for page splitting, deskewing, margin removal, and table line detection.

## Features

- **Two-Stage Processing**: Initial aggressive processing followed by precision refinement
- **Automatic Page Splitting**: Intelligently separates double-page scanned documents
- **Advanced Deskewing**: Corrects document rotation with sub-degree precision
- **Smart Margin Removal**: Automatically removes document margins and background noise
- **Table Line Detection**: Connected components method for robust table structure detection
- **Visualization Tools**: Analysis and debugging capabilities for each processing stage
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

### Visualization and Analysis

Analyze pipeline performance and debug issues:

```bash
# Complete pipeline analysis
python tools/run_visualizations.py all --pipeline image.jpg

# Individual step analysis
python tools/visualize_page_split.py image.jpg
python tools/visualize_deskew.py image.jpg
python tools/visualize_roi.py image.jpg
python tools/visualize_table_lines.py image.jpg
```

## Pipeline Overview

### Stage 1: Initial Processing
Robust extraction from challenging scanned documents:

1. **Page Splitting** - Separate double-page scans using gutter detection
2. **Deskewing** - Correct rotation with sub-degree precision
3. **Margin Removal** - Remove document margins and background noise
4. **Line Detection** - Find table structures using connected components method
5. **Table Cropping** - Extract table regions for Stage 2

**Output**: `data/output/stage1_initial_processing/05_cropped_tables/`

### Stage 2: Refinement Processing
Precise refinement for publication-quality table extraction:

1. **Re-deskewing** - Fine-tune rotation on cropped tables
2. **Refined Line Detection** - Precise detection with optimized parameters
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
├── tools/                     # Visualization and analysis tools
│   ├── visualize_*.py        # Analysis and debugging tools
│   ├── run_visualizations.py # Complete visualization suite
│   └── setup_tuning.py       # Setup utilities
├── configs/                   # Configuration files
│   ├── stage1_default.json   # Stage 1 default parameters
│   └── stage2_default.json   # Stage 2 default parameters
├── data/
│   ├── input/                # Input images
│   └── output/               # Processing results
└── docs/                     # Documentation
```

## Configuration

The pipeline uses JSON configuration files with simplified parameter structures:

### Key Parameters

- **Page Splitting**: `gutter_search_start` (0.4), `gutter_search_end` (0.6)
- **Deskewing**: `angle_range` (5°), `min_angle_correction` (0.1°)
- **Margin Removal**: `black_threshold` (50), `content_threshold` (200)
- **Line Detection**: `threshold` (40), `horizontal_kernel_size` (10), `vertical_kernel_size` (10)

## Advanced Usage

### Custom Configuration

```bash
# Use custom configuration files
python scripts/run_complete.py data/input/ --config configs/my_config.json

# Run with debug mode for detailed output
python scripts/run_complete.py data/input/ --debug --verbose
```

### Visualization and Analysis

```bash
# Complete pipeline analysis
python tools/run_visualizations.py all --pipeline image.jpg --save-intermediates

# Individual step analysis
python tools/visualize_deskew.py image.jpg
python tools/visualize_roi.py image.jpg
python tools/visualize_table_lines.py image.jpg

# Results management
python tools/check_results.py list
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

1. **Test on representative samples** before processing large datasets
2. **Visually inspect** intermediate results at each stage
3. **Use debug mode** (`--debug`) for troubleshooting
4. **Use visualization tools** to understand pipeline behavior
5. **Check intermediate outputs** if final results are poor

## Troubleshooting

### Common Issues

- **Poor page splitting**: Adjust `gutter_search_start` and `gutter_search_end` based on document layout
- **Excessive rotation**: Increase `min_angle_correction` for more conservative deskewing  
- **Over-cropping**: Adjust `content_threshold` and `black_threshold` for margin removal
- **Missing table lines**: Adjust `threshold` and kernel sizes for line detection

### Getting Help

- Use `--verbose` flag for detailed processing information
- Check `docs/CLAUDE.md` for comprehensive development guidance
- Refer to `tools/README.md` for visualization tool instructions
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