# OCR Table Extraction Pipeline

A professional two-stage OCR pipeline for extracting table structures from scanned documents.

## Features

- **Two-Stage Processing**: Initial processing and refinement for optimal results
- **Flexible Configuration**: Stage-specific parameters for different use cases
- **Modern Architecture**: Clean separation of source code, scripts, and tools
- **Comprehensive Tooling**: Visualization and debugging utilities

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Run complete two-stage pipeline
python scripts/run_complete.py data/input/ --verbose

# Or run stages individually
python scripts/run_stage1.py data/input/ --verbose
python scripts/run_stage2.py --verbose
```

## Project Structure

```
OCR_Preprocess_ToTable/
├── src/ocr_pipeline/     # Core library code
├── scripts/              # CLI entry points
├── tools/                # Debugging and visualization utilities
├── docs/                 # Documentation
├── data/                 # Input, output, and debug data
├── configs/              # Configuration files
├── tests/                # Test suite
└── examples/             # Usage examples
```

## Documentation

See the [`docs/`](docs/) directory for detailed documentation:

- [API Documentation](docs/API.md)
- [Developer Guide](docs/CLAUDE.md)
- [Configuration Guide](configs/README.md)
- [Unicode Display Setup](docs/UNICODE_SETUP.md) - Fix display issues on Windows

## License

MIT License - see LICENSE file for details.