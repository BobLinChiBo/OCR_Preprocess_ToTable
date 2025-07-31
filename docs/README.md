# OCR Table Extraction Pipeline

A **simple and clean** OCR preprocessing pipeline for extracting table structures from scanned documents.

## 🚀 Quick Start

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/OCR_Preprocess_ToTable.git
cd OCR_Preprocess_ToTable

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Basic Usage

```bash
# Process all images in input directory
python -m ocr.pipeline

# Process specific directory
python -m ocr.pipeline /path/to/images -o /path/to/output

# Process single file
python -m ocr.pipeline image.jpg -o output/

# With verbose output
python -m ocr.pipeline --verbose

# Using the installed command
ocr-pipeline input/ -o output/ --verbose
```

## 📁 Project Structure

```
OCR_Preprocess_ToTable/
├── README.md
├── requirements.txt        # Simple dependencies
├── setup.py               # Minimal setup
├── ocr/                   # Main package
│   ├── __init__.py
│   ├── config.py         # Simple configuration
│   ├── pipeline.py       # Main pipeline logic
│   ├── utils.py          # Core utilities
│   └── processors/       # Processing modules
│       └── __init__.py
├── tests/                # Basic tests
│   ├── __init__.py
│   └── test_pipeline.py
├── visualization/        # Debug and visualization tools
│   ├── visualize_*.py   # Stage-specific visualizers
│   └── run_visualizations.py
├── input/                # Input images (create this)
├── output/               # Output results (auto-created)
└── examples/             # Usage examples
```

## 🛠️ How It Works

The pipeline performs these steps on each scanned image:

1. **Page Splitting** - Detects two-page spreads and splits them
2. **Deskewing** - Corrects rotation using line detection
3. **ROI Detection** - Uses Gabor filters to find content regions (optional)
4. **Table Detection** - Finds horizontal and vertical lines
5. **Table Extraction** - Crops to the table region

## 📋 Configuration

You can customize the pipeline behavior by modifying the configuration:

```python
from ocr.config import Config
from ocr.pipeline import OCRPipeline

# Create custom configuration
config = Config(
    input_dir="my_images",
    output_dir="results", 
    verbose=True,
    min_line_length=50,
    gutter_search_start=0.3,
    gutter_search_end=0.7,
    enable_roi_detection=True
)

# Run pipeline with custom config
pipeline = OCRPipeline(config)
pipeline.process_directory()
```

### Key Configuration Parameters

- **Page Splitting**: `gutter_search_start/end` (0.4-0.6 default)
- **Line Detection**: `min_line_length` (100px default)
- **Deskewing**: `angle_range/step` (±45° range, 0.5° step)
- **ROI Detection**: `enable_roi_detection`, Gabor filter parameters
- **Debug**: `save_debug_images` for troubleshooting

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_pipeline.py -v

# Run tests with coverage (optional)
pip install pytest-cov
python -m pytest tests/ --cov=ocr
```

## 📊 Visualization and Debugging

The project includes comprehensive visualization tools to help debug and understand each pipeline stage:

```bash
# Run all visualizations
python tools/run_visualizations.py

# Visualize specific stages
python tools/visualize_page_split.py
python tools/visualize_deskew.py
python tools/visualize_roi.py
python tools/visualize_table_lines.py
python tools/visualize_table_crop.py

# Check results
python tools/check_results.py
```

## 📦 Dependencies

- **OpenCV** - Image processing
- **NumPy** - Numerical operations  
- **Pillow** - Image I/O support

## 🎯 Features

- ✅ **Simple setup** - Just `pip install -r requirements.txt`
- ✅ **Clean code** - No complex configurations or tooling
- ✅ **Cross-platform** - Works on Windows, macOS, and Linux
- ✅ **Command line** - Easy to use from terminal
- ✅ **Python API** - Integrate into other projects
- ✅ **Lightweight** - Minimal dependencies
- ✅ **Visualization tools** - Debug each pipeline stage
- ✅ **ROI Detection** - Advanced content region detection

## 💻 Development

```bash
# Install in development mode
pip install -e .

# Run single test
python tests/test_pipeline.py

# Format code (optional)
python -m black src/ tests/
```

## 📖 Examples

### Processing a Directory

```python
from src.ocr_pipeline.pipeline import TwoStageOCRPipeline
from src.ocr_pipeline.config import Stage1Config, Stage2Config

# Configure two-stage pipeline
stage1_config = Stage1Config(input_dir="scanned_documents/")
stage2_config = Stage2Config()
pipeline = TwoStageOCRPipeline(stage1_config, stage2_config)

# Run complete pipeline
results = pipeline.run_complete_pipeline()
print(f"Created {len(results)} table extractions")
```

### Custom Processing with ROI Detection

```python
from pathlib import Path
from src.ocr_pipeline.config import Config
from src.ocr_pipeline.utils import load_image, detect_roi_for_image, crop_to_roi

# Load image and detect ROI
image = load_image(Path("scan.jpg"))
config = Config(enable_roi_detection=True)
roi_coords = detect_roi_for_image(image, config)

# Crop to detected region
cropped = crop_to_roi(image, roi_coords)
print(f"ROI: ({roi_coords['roi_left']}, {roi_coords['roi_top']}) to "
      f"({roi_coords['roi_right']}, {roi_coords['roi_bottom']})")
```

### Manual Pipeline Steps

```python
from pathlib import Path
from src.ocr_pipeline.utils import (
    load_image, split_two_page_image, deskew_image, 
    detect_table_lines, crop_table_region
)

# Load and split a two-page scan
image = load_image(Path("scan.jpg"))
left_page, right_page = split_two_page_image(image)

# Deskew and detect table structure
deskewed = deskew_image(left_page)
h_lines, v_lines = detect_table_lines(deskewed)
table_crop = crop_table_region(deskewed, h_lines, v_lines)

print(f"Found {len(h_lines)} horizontal and {len(v_lines)} vertical lines")
```

## 🔧 Command Line Options

```bash
# Two-stage pipeline (recommended)
python scripts/run_complete.py [input] [-o OUTPUT] [-v] [--debug]

# Individual stages
python scripts/run_stage1.py [input] [-o OUTPUT] [-v] [--debug]
python scripts/run_stage2.py [input] [-o OUTPUT] [-v] [--debug]

Arguments:
  input                 Input directory or file

Options:
  -o, --output OUTPUT   Output directory (default: data/output)
  -v, --verbose         Verbose output
  --debug              Save debug images
```

## 🏗️ Architecture

The pipeline is built with a modular design:

- **Config**: Dataclass-based configuration with JSON support and validation
- **Pipeline**: Two-stage orchestration with TwoStageOCRPipeline class
- **Utils**: Core image processing functions using OpenCV
- **Tools**: Comprehensive tuning and visualization utilities
- **Tests**: Test suite with real document examples

## 📝 License

MIT License - see LICENSE file for details.

---

**Note**: This is a simplified rewrite focused on core functionality and clean code. For advanced features and customization, see the configuration options and visualization tools.