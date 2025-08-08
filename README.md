# OCR Table Extraction Pipeline

A professional two-stage OCR preprocessing and table extraction pipeline for scanned document images. This pipeline is specifically designed to handle historical documents, books, and manuscripts containing tabular data.

## Features

- **Two-Stage Processing**: Comprehensive preprocessing followed by refined table extraction
- **Mark Removal**: Automatic removal of watermarks, stamps, and artifacts
- **Intelligent Page Processing**: Automatic detection and splitting of two-page spreads
- **Advanced Deskewing**: Multi-method rotation correction using Radon transform
- **Table Detection & Recovery**: Robust table line detection and structure recovery
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

# Run Stage 1 only (preprocessing)
python scripts/run_stage1.py

# Run Stage 2 only (table refinement)
python scripts/run_stage2.py

# Enable debug mode with visualization
python scripts/run_pipeline.py --debug --verbose
```

## Pipeline Overview

### Stage 1: Preprocessing
1. **Mark Removal** (Optional): Removes watermarks, stamps, and artifacts
2. **Margin Removal**: Detects and removes paper edges and black backgrounds
3. **Page Splitting**: Identifies and splits two-page spreads
4. **Tag Removal**: Removes headers, footers, and page numbers
5. **Initial Deskewing**: Corrects image rotation
6. **Table Detection**: Identifies table regions
7. **Table Cropping**: Extracts individual table regions

### Stage 2: Table Refinement
1. **Fine Deskewing**: Precise rotation correction for cropped tables
2. **Table Line Detection**: Enhanced detection of table lines
3. **Structure Recovery**: Reconstructs table grid structure
4. **Vertical Strip Cutting**: Extracts individual columns for OCR

## Configuration

The pipeline uses JSON configuration files located in the `configs/` directory:

- `stage1_default.json`: Stage 1 preprocessing parameters
- `stage2_default.json`: Stage 2 refinement parameters

### Key Configuration Options

```json
{
  "mark_removal": {
    "enable": true,
    "dilate_iter": 2,
    "protect_table_lines": true
  },
  "optimization": {
    "parallel_processing": true,
    "memory_mode": true,
    "max_workers": null
  },
  "save_debug_images": false
}
```

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
│   └── PARAMETER_TUNING_GUIDE.md  # Parameter tuning guide
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
├── stage1/
│   ├── 02_margin_removed/     # Margins removed
│   ├── 03_deskewed/           # Rotation corrected
│   ├── 04_split_pages/        # Split pages
│   ├── 05_table_lines/        # Detected lines
│   ├── 06_table_structure/    # Table structure
│   └── 07_border_cropped/     # Cropped tables
└── stage2/
    ├── 01_refined_deskewed/   # Fine-tuned rotation
    ├── 02_table_lines/        # Refined lines
    ├── 03_table_structure/    # Final structure
    ├── 04_table_recovered/    # Recovery data
    └── 05_vertical_strips/    # Column strips
```

## Performance Optimization

### Parallel Processing
Enable parallel processing for batch operations:
```bash
python scripts/run_parallel.py --max-workers 8
```

### Memory Modes
- **Memory Mode** (default): Keeps intermediate results in RAM
- **Disk Mode**: Saves intermediate results to disk (slower but uses less RAM)

```bash
# Use disk mode for large datasets
python scripts/run_pipeline.py --no-memory-mode
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
- Input/output at each step
- Detected features (lines, contours, masks)
- Processing decisions and thresholds
- Before/after comparisons

## Documentation

- [Development Guide](docs/CLAUDE.md) - Architecture and development guidelines
- [Parameter Tuning Guide](docs/PARAMETER_TUNING_GUIDE.md) - Detailed parameter documentation

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

This pipeline incorporates techniques from various OCR preprocessing research and is optimized for historical document processing.