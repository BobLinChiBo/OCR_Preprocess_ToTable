# Visualization and Analysis Tools

Complete toolkit for analyzing, debugging, and optimizing the OCR Table Extraction Pipeline. These tools provide interactive analysis, parameter tuning, and comprehensive debugging capabilities.

## 🚀 Quick Start

```bash
# Complete pipeline analysis for any image
python tools/run_visualizations.py all --pipeline your_image.jpg

# Individual step analysis (V2 tools - recommended)
python tools/visualize_deskew_v2.py your_image.jpg
python tools/visualize_table_lines_v2.py your_image.jpg  
python tools/visualize_margin_removal_v2.py your_image.jpg --compare

# Interactive parameter testing
python tools/visualize_deskew_v2.py your_image.jpg --angle-range 15 --angle-step 0.2
```

## 📋 Tool Categories

### 🎯 Master Analysis Tool
- **`run_visualizations.py`** - One-stop analysis runner for complete pipeline visualization

### 🔍 V2 Interactive Analysis Tools (Recommended)
- **`visualize_page_split_v2.py`** - Page splitting analysis with V2 gutter detection  
- **`visualize_margin_removal_v2.py`** - Unified margin removal with method comparison
- **`visualize_deskew_v2.py`** - Rotation detection and correction analysis
- **`visualize_table_lines_v2.py`** - Table structure detection with connected components
- **`visualize_table_crop_v2.py`** - Table extraction and cropping visualization

### 🛠️ Specialized Tools
- **`debug_margin_analysis.py`** - Advanced margin removal debugging
- **`check_results.py`** - Results management and cleanup utilities
- **`config_utils.py`** - Configuration loading and validation utilities

## 🆕 V2 Architecture Advantages

The V2 visualization tools use a processor-based architecture that offers significant improvements:

### Benefits
- **🔧 Centralized Parameters**: Single update point when core algorithms change
- **🎛️ Consistent Interface**: All tools follow the same usage patterns  
- **🧪 Enhanced Analysis**: Built-in debug image management and analysis return
- **🚀 Better Performance**: Optimized parameter handling and memory management
- **📊 Unified Output**: Standardized visualization formats across all tools

### V1 to V2 Migration

```bash
# V1 Approach (Multiple separate scripts)
python tools/visualize_margin_removal.py image.jpg
python tools/visualize_margin_removal_fast.py image.jpg  
python tools/visualize_margin_removal_bbox.py image.jpg

# V2 Approach (Unified script with method selection)
python tools/visualize_margin_removal_v2.py image.jpg --method aggressive
python tools/visualize_margin_removal_v2.py image.jpg --method bounding_box
python tools/visualize_margin_removal_v2.py image.jpg --compare  # Compare all methods
```

> **🔄 Complete Migration Guide**: See [V2 Migration Guide](../docs/V2_MIGRATION_GUIDE.md) for detailed migration instructions and feature comparisons.

## 🔍 Detailed Tool Documentation

### Master Visualization Runner

#### `run_visualizations.py`
**Purpose**: One-stop tool for comprehensive pipeline analysis

```bash
# Complete analysis of single image
python tools/run_visualizations.py all --pipeline image.jpg

# Analyze specific steps
python tools/run_visualizations.py page-split deskew table-lines --pipeline image.jpg

# Process test images with all tools
python tools/run_visualizations.py all --test-images

# Save detailed intermediate outputs
python tools/run_visualizations.py all --pipeline image.jpg --save-intermediates
```

**Key Features**:
- **Batch Processing**: Analyze multiple images or test sets
- **Step Selection**: Choose specific processing steps to analyze
- **Pipeline Mode**: Maintain processing state between steps
- **Comprehensive Output**: Organized results in `data/output/visualization/`

### Individual Analysis Tools

#### `visualize_page_split_v2.py`
**Purpose**: Analyze V2 page splitting algorithm with gutter detection

```bash
# Basic page splitting analysis
python tools/visualize_page_split_v2.py double_page_image.jpg

# Adjust parameters for different document layouts
python tools/visualize_page_split_v2.py image.jpg --search-ratio 0.8 --peak-threshold 0.4

# Enable debug output for detailed analysis
python tools/visualize_page_split_v2.py image.jpg --save-debug
```

**Output**:
- Gutter search region visualization
- Vertical line detection results  
- Peak detection analysis
- Split position marking
- Debug images in `data/output/visualization/page_split_v2/debug/`

#### `visualize_margin_removal_v2.py`
**Purpose**: Unified margin removal analysis with method comparison

```bash
# Compare all margin removal methods
python tools/visualize_margin_removal_v2.py image.jpg --compare

# Test specific method
python tools/visualize_margin_removal_v2.py image.jpg --method inscribed
python tools/visualize_margin_removal_v2.py image.jpg --method gradient --gradient-threshold 25

# Use optimized utilities  
python tools/visualize_margin_removal_v2.py image.jpg --use-optimized
```

**Available Methods**:
- **`inscribed`**: NEW default method with paper mask + inscribed rectangle
- **`gradient`**: Gradient magnitude detection for subtle margins
- **`bounding_box`**: Simple bounding box approach
- **`aggressive`**: Legacy aggressive removal method

**Output**:
- Side-by-side method comparison (when using `--compare`)
- Detailed mask visualization and analysis
- Performance metrics for each method

#### `visualize_deskew_v2.py`
**Purpose**: Rotation detection and correction analysis

```bash
# Standard deskewing analysis
python tools/visualize_deskew_v2.py image.jpg

# Wide angle range for severely skewed documents
python tools/visualize_deskew_v2.py image.jpg --angle-range 20

# High precision detection
python tools/visualize_deskew_v2.py image.jpg --angle-step 0.05
```

**Output**:
- Angle histogram visualization
- Edge detection analysis
- Before/after rotation comparison
- Detected angle and confidence metrics

#### `visualize_table_lines_v2.py`
**Purpose**: Table structure detection with connected components method

```bash
# Standard table line analysis
python tools/visualize_table_lines_v2.py image.jpg

# Adjust sensitivity for faint lines
python tools/visualize_table_lines_v2.py image.jpg --threshold 25

# Modify kernel sizes for different line thicknesses
python tools/visualize_table_lines_v2.py image.jpg --horizontal-kernel 15 --vertical-kernel 15
```

**Output**:
- Binary threshold visualization
- Horizontal and vertical morphology results
- Connected components analysis
- Filtered line detection results
- Comprehensive table structure overlay

#### `visualize_table_crop_v2.py`
**Purpose**: Table extraction and boundary visualization

```bash
# Basic table cropping analysis
python tools/visualize_table_crop_v2.py image.jpg

# Show detailed boundary information
python tools/visualize_table_crop_v2.py image.jpg --show-boundaries

# Custom padding analysis
python tools/visualize_table_crop_v2.py image.jpg --padding 30
```

**Output**:
- Detected table boundaries
- Crop region with padding visualization
- Before/after extraction comparison

### Specialized Tools

#### `debug_margin_analysis.py`
**Purpose**: Advanced margin removal debugging and analysis

```bash
# Comprehensive margin analysis
python tools/debug_margin_analysis.py image.jpg

# Focus on specific margin detection issues
python tools/debug_margin_analysis.py problematic_image.jpg --detailed-analysis
```

#### `check_results.py`
**Purpose**: Results management and cleanup utilities

```bash
# List all available visualization results
python tools/check_results.py list

# View latest processing results
python tools/check_results.py view latest

# Clean up old results (older than 7 days)
python tools/check_results.py cleanup --older-than 7d

# Remove specific result set
python tools/check_results.py remove run_2025-01-15_10-30-45
```

#### `config_utils.py`
**Purpose**: Configuration loading and validation utilities (used by other tools)

```python
# Example usage in custom tools
from tools.config_utils import load_config
from src.ocr_pipeline.config import Stage1Config

config, source = load_config(args, Stage1Config, 'deskew')
print(f"Configuration loaded from: {source}")
```

## 🎯 Usage Workflows

### Parameter Optimization Workflow

```bash
# 1. Analyze your document type
python tools/visualize_table_lines_v2.py sample_document.jpg
python tools/visualize_margin_removal_v2.py sample_document.jpg --compare

# 2. Test parameter changes interactively  
python tools/visualize_deskew_v2.py sample_document.jpg --angle-range 15

# 3. Compare methods for margin removal
python tools/visualize_margin_removal_v2.py sample_document.jpg --compare

# 4. Validate with full pipeline
python scripts/run_complete.py sample_document.jpg --stage1-config my_optimized.json --debug
```

### Troubleshooting Workflow

```bash
# 1. Identify the problem stage
python tools/run_visualizations.py all --pipeline problem_image.jpg

# 2. Deep dive into the problematic stage  
python tools/visualize_table_lines_v2.py problem_image.jpg --threshold 25
python tools/visualize_deskew_v2.py problem_image.jpg --angle-range 20

# 3. Test fixes with debug mode
python scripts/run_complete.py problem_image.jpg --debug --verbose

# 4. Validate with representative batch
python tools/run_visualizations.py table-lines --test-images
```

### New Document Type Analysis

```bash
# 1. Complete analysis to understand document characteristics
python tools/run_visualizations.py all --pipeline representative_sample.jpg

# 2. Focus on challenging aspects identified in step 1
python tools/visualize_page_split_v2.py double_page_sample.jpg
python tools/debug_margin_analysis.py challenging_margin_sample.jpg

# 3. Create optimized configuration based on findings
# Edit configs/my_document_type.json based on analysis results

# 4. Validate optimized settings
python scripts/run_complete.py validation_set/ --stage1-config configs/my_document_type.json
```

## 📁 Output Organization

### Visualization Output Structure

```
data/output/visualization/
├── page_split_v2/                    # Page splitting analysis
│   ├── {image_name}_analysis.jpg     # Main visualization
│   ├── parameters.json               # Parameters used
│   └── debug/                        # Debug images (if --save-debug)
├── margin_removal_v2/                # Margin removal analysis  
│   ├── {image_name}_comparison.jpg   # Method comparison
│   ├── {image_name}_method_{name}.jpg # Individual method results
│   └── performance_metrics.json      # Method performance data
├── deskew_v2/                        # Deskewing analysis
│   ├── {image_name}_angle_analysis.jpg
│   ├── {image_name}_rotation_comparison.jpg
│   └── angle_detection_data.json
├── table_lines_v2/                   # Table structure analysis
│   ├── {image_name}_line_detection.jpg
│   ├── {image_name}_components.jpg
│   └── table_structure_data.json
└── table_crop_v2/                    # Table extraction analysis
    ├── {image_name}_boundaries.jpg
    ├── {image_name}_extraction.jpg
    └── crop_analysis.json
```

## 🚀 Performance and Best Practices

### Tool Performance

| Tool | Typical Runtime | Memory Usage | Output Size |
|------|----------------|--------------|-------------|
| `visualize_deskew_v2.py` | 5-15 seconds | ~200MB | ~2MB |
| `visualize_table_lines_v2.py` | 10-30 seconds | ~300MB | ~5MB |
| `visualize_margin_removal_v2.py --compare` | 15-45 seconds | ~400MB | ~10MB |
| `run_visualizations.py all` | 1-3 minutes | ~500MB | ~20MB |

### Best Practices

#### Analysis Strategy
1. **🎯 Start Targeted**: Use specific tools for known problem areas
2. **📊 Compare Methods**: Use `--compare` flags to understand trade-offs  
3. **🔍 Debug Selectively**: Enable debug output only when needed (large files)
4. **📈 Iterate Systematically**: Change one parameter at a time

#### Resource Management  
1. **💾 Clean Up Regularly**: Use `check_results.py cleanup` to manage disk space
2. **🖥️ Monitor Memory**: Tools can use significant RAM with large images
3. **⏱️ Batch Efficiently**: Process representative samples before full datasets
4. **📁 Organize Output**: Use meaningful names for custom analysis runs

#### Integration with Pipeline
- **Parameter Discovery**: Use tools to find optimal parameters, then update configs
- **Quality Validation**: Visual confirmation before batch processing
- **Debug Integration**: Tools complement pipeline `--debug` mode

## 🛡️ Troubleshooting Tools

### Common Tool Issues

#### Memory Issues with Large Images
```bash
# Resize large images before analysis
python -c "
from PIL import Image
img = Image.open('large_image.jpg')
if max(img.size) > 3000:
    img.thumbnail((3000, 3000), Image.LANCZOS)
    img.save('resized_for_analysis.jpg', quality=90)
"

# Then analyze the resized version
python tools/visualize_table_lines_v2.py resized_for_analysis.jpg
```

#### Tool Not Found Errors  
```bash
# Ensure you're in the project root directory
cd /path/to/OCR_Preprocess_ToTable

# Verify Python path includes the project  
python -c "import sys; print(sys.path)"

# Run tools with explicit Python module path
python -m tools.visualize_deskew_v2 image.jpg
```

#### Output Directory Issues
```bash
# Create output directories if they don't exist
mkdir -p data/output/visualization
chmod 755 data/output/visualization

# Check disk space for visualization outputs
df -h data/output/
```

## 📚 Additional Resources

- **[Debug Mode Guide](../docs/DEBUG_MODE_GUIDE.md)**: Comprehensive debugging with pipeline debug mode
- **[Parameter Reference](../docs/PARAMETER_REFERENCE.md)**: Complete parameter documentation  
- **[V2 Migration Guide](../docs/V2_MIGRATION_GUIDE.md)**: Detailed V1 to V2 migration guide
- **[Troubleshooting Guide](../docs/TROUBLESHOOTING.md)**: Solutions to common visualization issues

---

**Navigation**: [← Main README](../README.md) | [Configuration Guide](../configs/README.md) | [Documentation Index](../docs/README.md)

> **💡 Tip**: Start with `run_visualizations.py all --pipeline your_image.jpg` for a complete overview, then use individual tools to dive deep into specific processing steps.