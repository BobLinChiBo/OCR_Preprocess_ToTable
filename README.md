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
├── input/                # Input images (create this)
├── output/               # Output results (auto-created)
└── examples/             # Usage examples
```

## 🛠️ How It Works

The pipeline performs these steps on each scanned image:

1. **Page Splitting** - Detects two-page spreads and splits them
2. **Deskewing** - Corrects rotation using line detection
3. **Table Detection** - Finds horizontal and vertical lines
4. **Table Extraction** - Crops to the table region

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
    gutter_search_end=0.7
)

# Run pipeline with custom config
pipeline = OCRPipeline(config)
pipeline.process_directory()
```

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

## 💻 Development

```bash
# Install in development mode
pip install -e .

# Format code (optional)
python -m black ocr/ tests/

# Run tests
python -m pytest
```

## 📖 Examples

### Processing a Directory

```python
from ocr.pipeline import OCRPipeline

pipeline = OCRPipeline()
output_files = pipeline.process_directory("scanned_documents/")
print(f"Created {len(output_files)} table extractions")
```

### Custom Processing

```python
from pathlib import Path
from ocr.utils import load_image, split_two_page_image, detect_table_lines

# Load and split a two-page scan
image = load_image(Path("scan.jpg"))
left_page, right_page = split_two_page_image(image)

# Detect table structure
h_lines, v_lines = detect_table_lines(left_page)
print(f"Found {len(h_lines)} horizontal and {len(v_lines)} vertical lines")
```

## 🔧 Command Line Options

```bash
python -m ocr.pipeline [input] [-o OUTPUT] [-v] [--debug]

Arguments:
  input                 Input directory or file (default: input)

Options:
  -o, --output OUTPUT   Output directory (default: output)
  -v, --verbose         Verbose output
  --debug              Save debug images
```

## 📝 License

MIT License - see LICENSE file for details.

---

**Note**: This is a simplified rewrite focused on core functionality. For advanced features, see the previous version history.