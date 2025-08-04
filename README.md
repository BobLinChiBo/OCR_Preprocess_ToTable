# OCR Table Extraction Pipeline

A professional two-stage OCR pipeline for extracting tables from scanned document images with high accuracy. Features advanced computer vision techniques including page splitting, deskewing, intelligent margin removal, and robust table line detection.

## 🚀 Features

- **Two-Stage Processing Architecture**: Initial processing followed by precision refinement
- **Automatic Page Splitting**: Intelligently separates double-page scanned documents
- **Advanced Deskewing**: Sub-degree rotation correction for perfect alignment
- **Intelligent Margin Removal**: Multiple methods including the new inscribed rectangle algorithm
- **Robust Table Line Detection**: Connected components method for accurate structure detection
- **Table Structure Analysis**: Advanced cell and grid detection for complex tables
- **Border-Based Table Cropping**: Precise extraction of table regions
- **V2 Visualization Architecture**: Enhanced debugging and analysis tools with processor wrappers
- **Publication-Ready Output**: Final tables optimized for academic and professional use

## 📋 Requirements

- Python 3.8+
- OpenCV 4.5.0+
- NumPy 1.20.0+
- Pillow 8.0.0+

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/OCR_Preprocess_ToTable.git
cd OCR_Preprocess_ToTable

# Install dependencies
pip install -r requirements.txt

# Optional: Install with development dependencies
pip install -e ".[dev]"
```

## 🚀 Quick Start

### Basic Usage

```bash
# Process all images in a directory (complete two-stage pipeline)
python scripts/run_complete.py data/input/raw_images/ --verbose

# Process a single image
python scripts/run_complete.py image.jpg --verbose

# Process test images
python scripts/run_complete.py --test-images --verbose

# Stage 1 only (initial processing)
python scripts/run_stage1.py data/input/raw_images/ --verbose

# Stage 2 only (refinement) - requires Stage 1 output
python scripts/run_stage2.py --verbose
```

### Visualization and Analysis

The pipeline includes comprehensive V2 visualization tools for debugging and parameter tuning:

```bash
# Complete pipeline analysis with all steps
python tools/run_visualizations.py all --pipeline image.jpg --save-intermediates

# Individual step analysis (V2 tools - recommended)
python tools/visualize_page_split_v2.py image.jpg
python tools/visualize_margin_removal_v2.py image.jpg --method inscribed
python tools/visualize_deskew_v2.py image.jpg
python tools/visualize_table_lines_v2.py image.jpg
python tools/visualize_table_structure_v2.py image.jpg
python tools/visualize_table_crop_v2.py image.jpg

# Compare margin removal methods
python tools/visualize_margin_removal_v2.py image.jpg --compare
```

## 🔄 Pipeline Architecture

### Two-Stage Processing Flow

The pipeline uses a sophisticated two-stage approach for optimal results:

```
Input Images → Stage 1 (Initial Processing) → Stage 2 (Refinement) → Final Output
```

### Stage 1: Initial Processing
Robust extraction from challenging scanned documents:

1. **Page Splitting** (`01_split_pages/`)
   - Separates double-page scans using advanced gutter detection
   - Handles various binding types and scan qualities

2. **Margin Removal** (`02_margin_removed/`)
   - NEW: Inscribed rectangle method (default) - paper mask detection
   - Multiple algorithms available: aggressive, bounding box, smart, gradient
   - Removes document margins and background noise

3. **Deskewing** (`03_deskewed/`)
   - Corrects rotation with sub-degree precision
   - Projection profile optimization for accurate angle detection

4. **Table Line Detection** (`04_table_lines/`)
   - Connected components method for robust line detection
   - Handles broken lines and various table styles
   - Exports line data as JSON for further processing

5. **Table Structure Analysis** (`05_table_structure/`)
   - Detects table cells and grid structure
   - Analyzes table layout and boundaries
   - Exports structure data as JSON

6. **Border-Based Cropping** (`06_border_cropped/`)
   - Extracts table regions using detected structure
   - Applies configurable padding for clean extraction

### Stage 2: Refinement Processing
Precision refinement for publication-quality output:

1. **Re-deskewing** (`01_deskewed/`)
   - Fine-tunes rotation on cropped tables
   - Optimized parameters for table-specific content

2. **Margin Refinement** (`02_margin_removed/`)
   - Further cleans table edges if needed
   - Removes any remaining artifacts

3. **Refined Line Detection** (`03_table_lines/`)
   - Enhanced detection with optimized parameters
   - Better handling of thin or faint lines

4. **Final Table Fitting** (`04_fitted_tables/`)
   - Produces publication-ready table images
   - Optimal for OCR and further processing

## 📁 Project Structure

```
OCR_Preprocess_ToTable/
├── src/ocr_pipeline/              # Core pipeline modules
│   ├── __init__.py
│   ├── pipeline.py                # Main pipeline classes
│   ├── config.py                  # Configuration management
│   ├── utils.py                   # Image processing utilities
│   ├── utils_optimized.py         # Performance-optimized utilities
│   ├── processor_wrappers.py      # V2 processor wrapper architecture
│   └── processors/                # Additional processors
├── scripts/                       # CLI entry points
│   ├── run_complete.py           # Complete two-stage pipeline
│   ├── run_stage1.py             # Stage 1 only
│   └── run_stage2.py             # Stage 2 only
├── tools/                         # Visualization and analysis
│   ├── run_visualizations.py      # Comprehensive pipeline analysis
│   ├── visualize_*_v2.py         # V2 visualization tools (recommended)
│   ├── debug_margin_analysis.py   # Advanced debugging
│   ├── check_results.py          # Results management
│   ├── output_manager.py         # Output organization
│   └── config_utils.py           # Configuration utilities
├── configs/                       # Configuration files
│   ├── stage1_default.json       # Stage 1 parameters
│   └── stage2_default.json       # Stage 2 parameters
├── data/
│   ├── input/                    # Input images
│   │   ├── raw_images/          # Full dataset
│   │   └── test_images/         # Test subset
│   ├── output/                   # Processing results
│   │   ├── stage1_initial_processing/
│   │   └── stage2_refinement/
│   └── debug/                    # Debug outputs
├── docs/                         # Documentation
│   ├── CLAUDE.md                # AI assistant guidance
│   ├── PARAMETER_REFERENCE.md   # Detailed parameter guide
│   └── V2_MIGRATION_GUIDE.md    # V2 architecture guide
├── tests/                        # Test suite
├── requirements.txt              # Python dependencies
└── setup.py                      # Package setup
```

## ⚙️ Configuration

The pipeline uses JSON configuration files for flexible parameter management:

### Key Parameters

**Page Splitting**
- `search_ratio`: Gutter detection sensitivity (default: 0.3)
- `blur_k`: Blur kernel for noise reduction (default: 21)
- `width_min`: Minimum gutter width (default: 20)

**Margin Removal (Inscribed Rectangle Method - Default)**
- `inscribed_blur_ksize`: Blur for mask detection (default: 7)
- `inscribed_close_ksize`: Morphology kernel size (default: 30)
- `inscribed_close_iter`: Morphology iterations (default: 3)

**Deskewing**
- `angle_range`: Maximum rotation to detect ±degrees (default: 5)
- `angle_step`: Detection precision (default: 0.2)
- `min_angle_correction`: Minimum angle to correct (default: 0.2)

**Table Line Detection**
- `threshold`: Binary threshold (default: 30)
- `horizontal_kernel_size`: Horizontal line detection kernel (default: 20)
- `vertical_kernel_size`: Vertical line detection kernel (default: 20)
- `close_line_distance`: Distance for merging close lines (default: 45)

**Table Structure**
- `table_detection_eps`: Clustering epsilon (default: 10)
- `table_crop_padding`: Padding around detected tables (default: 20)

### Custom Configuration

```bash
# Create custom config file (see configs/ for examples)
# Then use it:
python scripts/run_complete.py input/ --stage1-config my_config.json

# Or override specific parameters:
python scripts/run_complete.py input/ --s1-angle-range 10 --s1-min-line-length 30
```

## 🔍 Advanced Usage

### Debugging and Analysis

```bash
# Enable debug mode for detailed output
python scripts/run_complete.py input/ --debug --verbose

# Analyze specific processing steps
python tools/debug_margin_analysis.py image.jpg

# Check processing results
python tools/check_results.py list
python tools/check_results.py view latest

# Clean up output directories
python tools/check_results.py cleanup
```

### Batch Processing

```bash
# Process large datasets efficiently
python scripts/run_complete.py /path/to/images/ -o /path/to/output/ --verbose

# Process with custom parameters for specific document types
python scripts/run_complete.py scholarly_docs/ --stage1-config configs/academic.json
```

### Development

```bash
# Run tests
python -m pytest tests/ -v

# Code quality checks
python -m black src/ scripts/ tools/
python -m flake8 src/ scripts/ tools/
python -m mypy src/

# Build distribution
python -m build
```

## 🎯 Best Practices

1. **Start with Test Images**: Use the provided test images for parameter tuning
2. **Visual Inspection**: Always check intermediate outputs when optimizing
3. **Parameter Tuning**: Adjust one parameter at a time for systematic optimization
4. **Use Debug Mode**: Enable `--debug` when troubleshooting issues
5. **Check Documentation**: Refer to `docs/PARAMETER_REFERENCE.md` for detailed parameter guidance

## 🐛 Troubleshooting

### Common Issues and Solutions

**Poor Page Splitting**
- Adjust `search_ratio` and `blur_k` based on gutter characteristics
- Use visualization tool: `python tools/visualize_page_split_v2.py image.jpg`

**Over/Under Rotation**
- Fine-tune `angle_range` and `min_angle_correction`
- Visualize: `python tools/visualize_deskew_v2.py image.jpg`

**Margin Removal Issues**
- Try different methods: `--method inscribed`, `--method gradient`, etc.
- Compare methods: `python tools/visualize_margin_removal_v2.py image.jpg --compare`

**Missing Table Lines**
- Lower `threshold` value for faint lines
- Adjust kernel sizes for different line thicknesses
- Debug: `python tools/visualize_table_lines_v2.py image.jpg --save-debug`

**Poor Final Quality**
- Ensure Stage 1 output is good before Stage 2
- Check each intermediate step in the pipeline
- Use `--save-intermediates` flag for detailed analysis

## 📚 Documentation

- **[PARAMETER_REFERENCE.md](docs/PARAMETER_REFERENCE.md)**: Complete parameter documentation
- **[V2_MIGRATION_GUIDE.md](docs/V2_MIGRATION_GUIDE.md)**: Guide to V2 visualization architecture
- **[CLAUDE.md](docs/CLAUDE.md)**: AI assistant integration guide
- **[tools/README.md](tools/README.md)**: Visualization tools documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and test thoroughly
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Computer vision techniques inspired by academic research in document analysis
- Built with OpenCV and the Python scientific computing ecosystem
- Designed for academic and professional document processing needs

## 📧 Contact

For questions, issues, or contributions, please use the GitHub issue tracker or submit a pull request.