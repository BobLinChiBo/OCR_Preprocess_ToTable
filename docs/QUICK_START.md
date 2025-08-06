# Quick Start Guide

Get up and running with the OCR Table Extraction Pipeline in minutes.

## Prerequisites

- Python 3.8+ installed
- Git installed  
- Basic familiarity with command line

## 5-Minute Setup

### 1. Install the Pipeline

```bash
# Clone repository
git clone https://github.com/yourusername/OCR_Preprocess_ToTable.git
cd OCR_Preprocess_ToTable

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Test with Sample Images

```bash
# Process test images (if available)
python scripts/run_complete.py --test-images --verbose

# Or process a single image
python scripts/run_complete.py path/to/your/image.jpg --verbose
```

### 3. Check Results

```bash
# View processed outputs
ls data/output/

# Stage 1 results (initial processing)
ls data/output/stage1/

# Stage 2 results (refined tables)
ls data/output/stage2/
```

🎉 **That's it!** Your images are now processed and ready for OCR.

---

## Processing Your First Image

### Single Image Processing

```bash
# Complete two-stage pipeline
python scripts/run_complete.py my_document.jpg --verbose

# Results will be in:
# data/output/stage1/07_border_cropped/  (initial processing)
# data/output/stage2/06_binarized/       (final tables)
```

### Batch Processing

```bash
# Process entire directory
python scripts/run_complete.py /path/to/images/ --verbose

# With custom output location
python scripts/run_complete.py /path/to/images/ -o /custom/output/ --verbose
```

### Stage-by-Stage Processing

```bash
# Stage 1 only (initial processing)
python scripts/run_stage1.py /path/to/images/ --verbose

# Stage 2 only (refinement) - requires Stage 1 output
python scripts/run_stage2.py --verbose
```

## Understanding the Output

### Stage 1: Initial Processing
```
data/output/stage1/
├── 02_margin_removed/     # Margins removed
├── 04_deskewed/          # Rotation corrected  
├── 05_table_lines/       # Table structure detected
├── 06_table_structure/   # Table analysis
└── 07_border_cropped/    # Tables extracted
```

### Stage 2: Refinement
```
data/output/stage2/
├── 01_deskewed/          # Fine-tuned rotation
├── 02_table_lines/       # Enhanced line detection
├── 04_table_recovered/   # Advanced table recovery
├── 05_vertical_strips/   # Column extraction
└── 06_binarized/         # Final publication-ready output ⭐
```

**💡 Final Output**: The `06_binarized/` directory contains your publication-ready table images.

## Visual Analysis

### Quick Visualization

```bash
# Analyze processing results for a single image
python tools/run_visualizations.py all --pipeline image.jpg

# Individual step analysis
python tools/visualize_deskew_v2.py image.jpg
python tools/visualize_table_lines_v2.py image.jpg
```

### Debug Mode

```bash
# Enable debug mode for detailed analysis
python scripts/run_complete.py image.jpg --debug --verbose

# Debug outputs saved to: data/debug/
```

## Common First-Time Adjustments

### If Pages Aren't Splitting Correctly

```bash
# Test page splitting
python tools/visualize_page_split_v2.py double_page.jpg

# Adjust configuration
# Edit configs/stage1_default.json:
# "search_ratio": 0.7  (increase for wider search)
```

### If Tables Aren't Detected

```bash
# Visualize table detection
python tools/visualize_table_lines_v2.py image.jpg

# Lower threshold for faint lines
# Edit configs/stage1_default.json:
# "threshold": 25  (decrease from default 40)
```

### If Images Are Over-Rotated

```bash
# Check deskewing
python tools/visualize_deskew_v2.py image.jpg

# Be more conservative
# Edit configs/stage1_default.json:
# "min_angle_correction": 0.5  (increase from 0.1)
```

## Customization Examples

### Custom Configuration

```bash
# Create custom config
cp configs/stage1_default.json my_config.json
# Edit my_config.json as needed

# Use custom config
python scripts/run_complete.py images/ --stage1-config my_config.json
```

### Processing Options

```bash
# Skip certain steps
python scripts/run_complete.py images/ --verbose
# Edit config to disable steps: "enable": false

# Different output formats
python scripts/run_complete.py images/ -o custom_output/ --verbose
```

## What's Next?

### Learn More
- **[Parameter Reference](PARAMETER_REFERENCE.md)** - Understand all configuration options
- **[Tools Documentation](../tools/README.md)** - Master the visualization tools
- **[Debug Mode Guide](DEBUG_MODE_GUIDE.md)** - Advanced debugging techniques

### Optimize Performance
- **[Configuration Guide](../configs/README.md)** - Fine-tune for your document types
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Solve common issues

### For Developers
- **[API Reference](API_REFERENCE.md)** - Use the pipeline in your code
- **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - Understand the architecture

## Troubleshooting Quick Fixes

### Installation Issues
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Check OpenCV installation  
python -c "import cv2; print(cv2.__version__)"
```

### Memory Issues
```bash
# Process smaller batches
python scripts/run_complete.py large_images/ --verbose
# Consider resizing very large images first
```

### Permission Issues
```bash
# Ensure data directories are writable
chmod -R 755 data/
mkdir -p data/output data/debug
```

### Poor Results
```bash
# Enable debug mode to see what's happening
python scripts/run_complete.py problem_image.jpg --debug --verbose

# Use visualization tools to understand the issue
python tools/visualize_table_lines_v2.py problem_image.jpg
```

## Getting Help

- **Quick Issues**: Check [Troubleshooting Guide](TROUBLESHOOTING.md)  
- **Parameter Questions**: See [Parameter Reference](PARAMETER_REFERENCE.md)
- **Bug Reports**: Use GitHub Issues
- **Feature Requests**: Use GitHub Issues with "enhancement" label

---

**Navigation**: [← Installation Guide](INSTALLATION.md) | [Documentation Index](README.md) | [Parameter Reference →](PARAMETER_REFERENCE.md)

**Ready to dive deeper?** Continue with the [Parameter Reference](PARAMETER_REFERENCE.md) to optimize processing for your specific document types.