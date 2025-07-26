# OCR Table Extraction Pipeline

A comprehensive two-stage pipeline for extracting table structures from scanned document images. This system handles both straight and slightly curved table lines using advanced morphological processing and contour detection.

## 🚀 Quick Start

```bash
# Stage 1: Initial Processing (raw scanned images → cropped tables)
python run_stage1_initial_processing.py

# Stage 2: Refinement (cropped tables → publication-ready results)
python run_stage2_refinement.py
```

## 📁 Project Structure

```
OCR_Preprocess_ToTable/
├── run_stage1_initial_processing.py    # Main Stage 1 workflow
├── run_stage2_refinement.py            # Main Stage 2 workflow
├── configs/                            # Configuration files
│   ├── stage1_config.json             # Stage 1 parameters
│   ├── stage2_config.json             # Stage 2 parameters
│   ├── CONFIG_GUIDE.md               # Configuration guide
│   └── PARAMETER_REFERENCE.md        # Parameter reference
├── src/                              # Source code
│   ├── core/                         # Core processing modules
│   └── utils/                        # Utility functions
├── input/                            # Input directory
│   └── raw_images/                   # Place raw scanned images here
├── output/                           # Output directory (git-ignored)
│   ├── stage1_initial_processing/    # Stage 1 results
│   └── stage2_refinement/            # Stage 2 results
└── debug/                            # Debug output (git-ignored)
```

## 🔄 Two-Stage Processing Workflow

### Stage 1: Initial Processing
Processes raw scanned images through complete initial workflow:

1. **Page Splitting** - Separates double-page scans into individual pages
2. **Deskewing** - Corrects image rotation and skewing
3. **Edge Detection** - Detects content boundaries (optional)
4. **Line Detection** - Finds table lines using curved line detection
5. **Table Reconstruction** - Builds complete table structures
6. **Table Cropping** - Extracts table regions for refinement

### Stage 2: Refinement
Refines cropped table images for higher precision:

1. **Re-deskewing** - Fine-tunes rotation on cropped tables  
2. **Refined Line Detection** - Optimized parameters for table content
3. **Final Table Reconstruction** - Precise grid creation
4. **Table Fitting** - Publication-ready cell structures

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7+
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- SciPy (`pip install scipy`)
- Matplotlib (`pip install matplotlib`)

### Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt` (if available)
3. Place raw scanned images in `input/raw_images/`
4. Run the pipeline: `python run_stage1_initial_processing.py`

## 📊 Input/Output

### Input
- **Raw scanned images** (JPG, PNG, TIFF)
- Supports both single-page and double-page scans
- Academic papers, books, technical documents

### Output
- **Stage 1**: Cropped table regions ready for refinement
- **Stage 2**: Publication-ready table images with cell structure data
- **Formats**: Images (JPG) + metadata (JSON)

## 🎯 Key Features

- **Intelligent Page Splitting** - Automatically detects double-page layouts
- **Curved Line Support** - Handles slightly curved table lines using polynomial fitting
- **ROI-Aware Processing** - Different parameters for recto/verso pages
- **Organized Output** - Clean directory structure with numbered stages
- **Comprehensive Logging** - Detailed progress tracking and error reporting
- **Modular Architecture** - Well-organized, maintainable codebase

## ⚙️ Configuration

The pipeline uses JSON configuration files with extensive parameter control:

- **Stage 1 Config**: `configs/stage1_config.json` - Initial processing parameters
- **Stage 2 Config**: `configs/stage2_config.json` - Refinement parameters

For detailed configuration guidance, see:
- [`configs/CONFIG_GUIDE.md`](configs/CONFIG_GUIDE.md) - Comprehensive configuration guide
- [`configs/PARAMETER_REFERENCE.md`](configs/PARAMETER_REFERENCE.md) - Parameter reference

## 📈 Performance & Scalability

- **Batch Processing** - Handles large document collections
- **Debug Support** - Optional debug image generation for parameter tuning
- **Progress Tracking** - Real-time processing statistics
- **Error Recovery** - Graceful handling of problematic images

## 🔧 Customization

The modular architecture allows easy customization:

- **Core Modules** (`src/core/`) - Processing algorithms
- **Utilities** (`src/utils/`) - Shared functionality
- **Configuration** - Extensive parameter control
- **Extension Points** - Easy to add new processing steps

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
python run_stage1_initial_processing.py my_stage1_config.json
python run_stage2_refinement.py my_stage2_config.json
```

### Debug Mode
Enable debug image generation in configuration:
```json
{
  "line_detection": {
    "SAVE_DEBUG_IMAGES": true
  }
}
```

## 🐛 Troubleshooting

### Common Issues

1. **No images found**: Check `input/raw_images/` directory
2. **Stage 2 fails**: Ensure Stage 1 completed successfully
3. **Poor line detection**: Adjust morphological parameters in config
4. **Memory issues**: Process smaller batches or reduce image resolution

### Debug Information

Enable debug logging in configurations:
```json
{
  "debug_enabled": true,
  "log_level": "DEBUG"
}
```

## 📚 Documentation

- **Configuration Guide**: [`configs/CONFIG_GUIDE.md`](configs/CONFIG_GUIDE.md)
- **Parameter Reference**: [`configs/PARAMETER_REFERENCE.md`](configs/PARAMETER_REFERENCE.md)
- **API Documentation**: See docstrings in source code

## 🤝 Contributing

The codebase is organized for easy contribution:
1. Core processing logic in `src/core/`
2. Shared utilities in `src/utils/`
3. Configuration management in `configs/`
4. Main workflows at project root

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

Built for robust OCR table extraction from academic papers and technical documents.
Supports both straight and curved table lines with advanced morphological processing.