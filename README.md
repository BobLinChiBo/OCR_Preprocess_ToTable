# OCR Table Extraction Pipeline

A professional two-stage OCR preprocessing and table extraction pipeline for scanned document images. This pipeline is specifically designed to handle historical documents, books, and manuscripts containing tabular data.

## Features

- **Two-Stage Processing**: Table cropping followed by structure recovery and column extraction
- **Advanced Preprocessing**: Mark removal, margin detection, and intelligent deskewing
- **Robust Table Detection**: Multi-angle line detection with preprocessing and stroke enhancement
- **Intelligent Structure Recovery**: Reconstructs complete table grid from detected lines
- **Vertical Strip Cutting**: Extracts individual columns ready for OCR processing
- **Parallel Processing**: Efficient batch processing with multiprocessing support
- **Memory Optimization**: Flexible memory modes for different system configurations
- **Debug Visualization**: Comprehensive debug images for each processing step

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/BobLinChiBo/OCR_Preprocess_ToTable.git
cd OCR_Preprocess_ToTable

# Install required packages
pip install -r requirements.txt
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```bash
# Run the complete pipeline with default settings
python scripts/run_pipeline.py

# Process specific input directory
python scripts/run_pipeline.py data/input/raw_images

# Specify custom output directory
python scripts/run_pipeline.py -o custom_output
```

### Advanced Usage

```bash
# Run with parallel processing (faster for multiple images)
python scripts/run_parallel.py

# Run Stage 1 only (table cropping)
python scripts/run_stage1.py

# Run Stage 2 only (structure recovery and column cutting)
python scripts/run_stage2.py

# Enable debug mode with visualization
python scripts/run_pipeline.py --debug --verbose
```

## Pipeline Overview

### Stage 1: Table Cropping and Isolation
**Purpose**: Crop out all irrelevant content and focus on the main table region

1. **Mark Removal** (Optional): Removes watermarks, stamps, and artifacts
2. **Margin Removal** (Optional): Detects and removes paper edges and black backgrounds  
3. **Page Splitting** (Optional): Identifies and splits two-page spreads
4. **Tag Removal** (Optional): Removes headers, footers, and page numbers
5. **Initial Deskewing**: Corrects image rotation using Radon transform
6. **Table Line Detection**: Detects table lines with preprocessing and stroke enhancement
7. **Table Structure Detection**: Identifies grid structure from detected lines
8. **Table Cropping**: Extracts individual table regions, removing all non-table content

### Stage 2: Structure Recovery and Column Extraction  
**Purpose**: Recover the actual table structure and cut by vertical lines for OCR

1. **Fine Deskewing**: Precise rotation correction for cropped table images
2. **Refined Table Line Detection**: Enhanced detection with preprocessing optimized for cropped tables
3. **Table Structure Recovery**: Reconstructs complete table grid, filling missing lines
4. **Vertical Strip Cutting**: Cuts table into individual columns based on vertical grid lines
5. **Final Binarization** (Optional): Prepares strips for OCR processing
6. **Final Deskewing** (Optional): Last-stage rotation correction

## Configuration

The pipeline uses a single JSON configuration file for each stage:

- `configs/stage1_default.json`: Stage 1 table cropping parameters
- `configs/stage2_default.json`: Stage 2 structure recovery parameters

### Key Configuration Sections

#### Table Line Detection with Preprocessing
```json
{
  "table_line_detection": {
    "line_detection_use_preprocessing": true,
    "line_detection_binarization_method": "adaptive",
    "line_detection_stroke_enhancement": true,
    "line_detection_binarization_denoise": true,
    "threshold": 20,
    "horizontal_kernel_size": 80,
    "vertical_kernel_size": 40,
    "skew_tolerance": 2
  }
}
```

#### Processing Options
```json
{
  "margin_removal": {
    "enable": false,
    "use_gradient_detection": true
  },
  "optimization": {
    "parallel_processing": true,
    "memory_mode": false,
    "max_workers": 6
  },
  "save_debug_images": false
}
```

### Preprocessing Features

Both stages support advanced preprocessing for better table line detection:

- **Adaptive Binarization**: Smart thresholding for varying image quality
- **Stroke Enhancement**: Morphological operations to enhance thin/faded lines
- **Denoising**: Removes image artifacts that interfere with line detection
- **Multi-angle Detection**: Handles skewed tables up to configurable tolerance
- **Search Regions**: Focus detection on specific image areas

### Override Configuration via CLI

```bash
# Enable debug images
python scripts/run_pipeline.py --save-debug-images

# Disable parallel processing
python scripts/run_pipeline.py --no-parallel

# Set specific number of workers
python scripts/run_pipeline.py --max-workers 4
```

## Project Structure

```
OCR_Preprocess_ToTable/
├── configs/                 # Configuration files
│   ├── stage1_default.json
│   └── stage2_default.json
├── data/                    # Data directories
│   ├── input/              # Input images
│   ├── output/             # Processing results
│   └── debug/              # Debug visualizations
├── docs/                    # Documentation
│   ├── CLAUDE.md           # Development guide
│   └── parameter_tuning_guide.md  # Parameter tuning guide
├── scripts/                 # Entry point scripts
│   ├── run_pipeline.py     # Main pipeline
│   ├── run_parallel.py     # Parallel processing
│   └── run_stage[1|2].py   # Individual stages
├── src/ocr_pipeline/        # Core pipeline code
│   ├── processors/         # Image processing modules
│   ├── pipeline.py         # Pipeline orchestration
│   └── config.py           # Configuration management
└── tests/                   # Test suite
```

## Input/Output

### Supported Input Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tif, .tiff)
- BMP (.bmp)

### Output Structure

```
output/
├── stage1/                    # Table Cropping Stage
│   ├── 01_mark_removed/         # Watermarks/stamps removed
│   ├── 02_margin_removed/       # Margins removed
│   ├── 03_deskewed/             # Rotation corrected
│   ├── 04_split_pages/          # Split pages (if applicable)
│   ├── 05_table_lines/          # Detected table lines
│   ├── 06_table_structure/      # Table grid structure
│   └── 07_border_cropped/       # Cropped table regions
└── stage2/                    # Structure Recovery Stage
    ├── 01_refined_deskewed/     # Fine-tuned rotation
    ├── 02_table_lines/          # Refined line detection
    ├── 03_table_structure/      # Final structure
    ├── 04_table_recovered/      # Structure recovery data
    ├── 05_vertical_strips/      # Individual columns
    ├── 06_binarized/           # OCR-ready images
    └── 07_final_deskewed/      # Final rotation correction
```

## Performance Optimization

### Parallel Processing
Enable parallel processing for batch operations:
```bash
python scripts/run_parallel.py --max-workers 8
```

### Memory Modes
- **Memory Mode**: Keeps intermediate results in RAM (faster)
- **Disk Mode** (default): Saves intermediate results to disk (uses less RAM)

```bash
# Use memory mode for faster processing
python scripts/run_pipeline.py --memory-mode
```

## Debug Mode

Enable comprehensive debug visualization:

```bash
# Save debug images for all processing steps
python scripts/run_pipeline.py --save-debug-images --debug-dir data/debug

# Enable verbose output
python scripts/run_pipeline.py --verbose
```

Debug images show:
- Preprocessing steps (binarization, stroke enhancement, denoising)
- Line detection at each stage (morphological operations, connected components)
- Table structure reconstruction
- Before/after comparisons for each processing step

## Documentation

- [Development Guide](docs/CLAUDE.md) - Architecture and development guidelines
- [Parameter Tuning Guide](docs/parameter_tuning_guide.md) - Detailed parameter documentation

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=src/ocr_pipeline tests/

# Run specific test
python -m pytest tests/test_pipeline.py::test_stage1_processing
```

## Code Quality

```bash
# Format code with black
python -m black src/ scripts/ tests/

# Lint with flake8
python -m flake8 src/ scripts/ tests/

# Type checking with mypy
python -m mypy src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please use the [GitHub Issues](https://github.com/BobLinChiBo/OCR_Preprocess_ToTable/issues) page.

## Acknowledgments

This pipeline incorporates advanced OCR preprocessing techniques and is optimized for historical document processing with robust table structure recovery.