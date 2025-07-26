# OCR Table Extraction Pipeline

A professional two-stage pipeline for extracting table structures from scanned document images. This system handles both straight and curved table lines using advanced morphological processing, contour detection, and polynomial fitting algorithms.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Stage 1: Initial Processing (raw scanned images → cropped tables)
python run_stage1_initial_processing.py

# Stage 2: Refinement (cropped tables → publication-ready results)
python run_stage2_refinement.py

# Or use Make shortcuts
make pipeline  # Run both stages
```

## 📁 Project Structure

```
OCR_Preprocess_ToTable/
├── run_stage1_initial_processing.py    # Legacy Stage 1 workflow
├── run_stage2_refinement.py            # Legacy Stage 2 workflow
├── src/ocr_pipeline/                   # Modern modular architecture
│   ├── config/                         # Configuration system
│   │   ├── loader.py                   # Multi-format config loader
│   │   ├── models.py                   # Pydantic config models
│   │   ├── default_stage1.yaml         # Default Stage 1 config
│   │   └── default_stage2.yaml         # Default Stage 2 config
│   ├── processors/                     # Processing modules
│   │   └── base.py                     # Base processor class
│   ├── utils/                          # Utility functions
│   │   ├── file_utils.py               # File operations
│   │   ├── image_utils.py              # Image processing
│   │   └── logging_utils.py            # Logging utilities
│   └── exceptions.py                   # Custom exceptions
├── configs/                            # Legacy JSON configurations
│   ├── stage1_config.json              # Stage 1 parameters
│   ├── stage2_config.json              # Stage 2 parameters
│   ├── CONFIG_GUIDE.md                 # Configuration guide
│   └── PARAMETER_REFERENCE.md          # Parameter reference
├── input/                              # Input directory
│   └── raw_images/                     # Place raw scanned images here
├── output/                             # Output directory (auto-created)
│   ├── stage1_initial_processing/      # Stage 1 results
│   └── stage2_refinement/              # Stage 2 results
├── debug/                              # Debug output (auto-created)
│   ├── stage1_debug/                   # Stage 1 debug images
│   └── stage2_debug/                   # Stage 2 debug images
└── tests/                              # Test suite
    ├── unit/                           # Unit tests
    ├── integration/                    # Integration tests
    └── fixtures/                       # Test fixtures
```

## 🔄 Two-Stage Processing Workflow

### Stage 1: Initial Processing
Transforms raw scanned images into cropped table regions:

1. **Page Splitting** - Separates double-page scans using gutter detection
2. **Deskewing** - Corrects rotation using variance-based angle optimization
3. **Edge Detection** - Detects content boundaries using Gabor filters (optional)
4. **Curved Line Detection** - Advanced table line detection with:
   - OpenCV morphological operations for line isolation
   - Hough transform for straight line detection
   - Contour-based polynomial fitting for curved lines
   - ROI-aware processing with different margins for verso/recto pages
5. **Table Reconstruction** - Builds complete table grid structures
6. **Table Cropping** - Extracts table regions for Stage 2 processing

### Stage 2: Refinement
Polishes cropped table images to publication quality:

1. **Re-deskewing** - Fine-tunes rotation on cropped table images
2. **Refined Line Detection** - Zero-margin processing with optimized parameters
3. **Final Table Reconstruction** - Precise grid generation
4. **Table Fitting** - Cell structure optimization for publication-ready output

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+ (supports 3.8-3.12)
- Git for version control

### Installation

```bash
# Clone the repository
git clone https://github.com/BobLinChiBo/OCR_Preprocess_ToTable.git
cd OCR_Preprocess_ToTable

# Install in development mode with all dependencies
make install-dev

# Or manually with pip
pip install -e ".[dev,test,docs]"

# Set up pre-commit hooks (optional but recommended)
make setup-dev
```

### Dependencies
Core dependencies include:
- **OpenCV** (>=4.8.0) - Computer vision operations
- **NumPy** (>=1.21.0) - Numerical computing
- **SciPy** (>=1.7.0) - Scientific computing
- **Matplotlib** (>=3.5.0) - Plotting and visualization
- **Pydantic** (>=2.0.0) - Data validation
- **PyYAML** (>=6.0.0) - YAML configuration support

## 📊 Input/Output

### Input Requirements
- **Formats**: JPG, PNG, TIFF scanned images
- **Content**: Academic papers, technical documents, books with tabular data
- **Layout**: Single-page or double-page spreads
- **Quality**: Sufficient contrast for line detection (parameters adjustable)

### Output Structure
- **Stage 1**: Cropped table regions with processing metadata
- **Stage 2**: Publication-ready table images with cell structure data
- **Formats**: 
  - Images: High-quality JPG files
  - Metadata: JSON files with line coordinates, cell boundaries, processing statistics

## 🎯 Key Features

- **Intelligent Page Splitting** - Automatic detection of double-page layouts with gutter detection
- **Curved Line Support** - Handles slightly curved table lines using polynomial fitting algorithms
- **ROI-Aware Processing** - Different processing parameters for recto/verso pages
- **Multi-Format Configuration** - Supports JSON, YAML, and TOML configuration files
- **Comprehensive Logging** - Detailed progress tracking with configurable log levels
- **Debug Visualization** - Optional debug image generation for parameter tuning
- **Modular Architecture** - Clean separation between legacy scripts and modern modules
- **Type Safety** - Full type hints with mypy checking
- **Professional Testing** - Comprehensive test suite with pytest

## ⚙️ Configuration

The pipeline supports multiple configuration formats with validation:

### Configuration Files
- **Modern**: YAML files in `src/ocr_pipeline/config/` (recommended)
- **Legacy**: JSON files in `configs/` (maintained for compatibility)

### Configuration Loading
The system automatically detects configuration format:
```bash
# YAML (recommended)
python run_stage1_initial_processing.py my_config.yaml

# JSON (legacy)
python run_stage1_initial_processing.py my_config.json

# TOML (supported)
python run_stage1_initial_processing.py my_config.toml
```

### Parameter Categories
- **Morphological Operations**: Kernel sizes as ratios of image dimensions
- **Hough Transform**: Threshold, minimum line length, maximum gap ratios
- **Line Clustering**: Distance-based grouping with progressive filtering
- **Curved Line Detection**: Contour length ratios and aspect ratio thresholds

For detailed configuration guidance:
- [`configs/CONFIG_GUIDE.md`](configs/CONFIG_GUIDE.md) - Comprehensive guide
- [`configs/PARAMETER_REFERENCE.md`](configs/PARAMETER_REFERENCE.md) - Parameter reference

## 🔧 Development

### Development Commands
```bash
# Code quality
make format          # Format with black and isort
make lint           # Run ruff, flake8, bandit
make type-check     # Run mypy type checking

# Testing
make test           # Run full test suite
make test-fast      # Skip slow integration tests
make test-cov       # Generate coverage report

# Pipeline execution
make stage1         # Run Stage 1 only
make stage2         # Run Stage 2 only
make pipeline       # Run complete pipeline

# Quick development check
make quick          # format + lint + test-fast
```

### Testing
The project includes comprehensive testing:
```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m "not slow"        # Skip slow tests

# With coverage
pytest --cov=ocr_pipeline
```

## 📝 Usage Examples

### Basic Usage
```bash
# Process with default configurations
python run_stage1_initial_processing.py
python run_stage2_refinement.py
```

### Custom Configuration
```bash
# Use custom configuration files
python run_stage1_initial_processing.py custom_stage1.yaml
python run_stage2_refinement.py custom_stage2.json
```

### Debug Mode
Enable debug image generation in configuration:
```yaml
line_detection:
  save_debug_images: true
  debug_output_dir: "debug/custom_debug"
```

### Development Workflow
```bash
# Set up development environment
make setup-dev

# Make changes, then run quality checks
make quick

# Run full test suite before committing
make test

# Check security
make security
```

## 🐛 Troubleshooting

### Common Issues

1. **No images found**: Verify images are in `input/raw_images/` directory
2. **Stage 2 fails**: Ensure Stage 1 completed successfully and check `output/stage1_initial_processing/05_cropped_tables/`
3. **Poor line detection**: Enable debug images and adjust morphological parameters
4. **Memory issues**: Process smaller batches or reduce image resolution in config

### Debug Information

Enable debug output in configuration:
```yaml
logging:
  level: DEBUG
  
line_detection:
  save_debug_images: true
```

Debug images are saved to `debug/stage1_debug/line_detection/` or `debug/stage2_debug/line_detection/` showing:
- Binary masked images
- Morphological operations
- Line detection stages
- Final clustered results

## 📚 Documentation

- **Configuration**: [`configs/CONFIG_GUIDE.md`](configs/CONFIG_GUIDE.md)
- **Parameters**: [`configs/PARAMETER_REFERENCE.md`](configs/PARAMETER_REFERENCE.md)
- **Development**: [`CLAUDE.md`](CLAUDE.md) - Claude Code guidance
- **API Documentation**: Comprehensive docstrings in source code

## 🤝 Contributing

The codebase is organized for easy contribution:

1. **Modern Architecture** - Modular components in `src/ocr_pipeline/`
2. **Configuration** - Multi-format support with validation
3. **Testing** - Comprehensive test suite with fixtures
4. **Quality Tools** - Pre-commit hooks, linters, type checking

### Development Setup
```bash
git clone <repository-url>
cd OCR_Preprocess_ToTable
make setup-dev  # Installs dependencies and pre-commit hooks
```

## 🙏 Acknowledgments

Built for robust OCR table extraction from academic papers and technical documents. Features advanced morphological processing, polynomial fitting for curved lines, and comprehensive parameter tuning capabilities.

**Key Technologies:**
- OpenCV for computer vision operations
- SciPy for scientific computing and optimization
- Pydantic for configuration validation
- Modern Python packaging with pyproject.toml