# Configuration Guide

Complete guide to configuring and customizing the OCR Table Extraction Pipeline for optimal results on your document types.

## 📋 Quick Reference

| Configuration File | Purpose | Key Features |
|-------------------|---------|-------------|
| **stage1_default.json** | Initial processing parameters | Page splitting, deskewing, margin removal, table detection |
| **stage2_default.json** | Refinement parameters | Fine-tuning, table recovery, column extraction |

## 🔧 Stage 1 Configuration (Initial Processing)

### Processing Pipeline Order

```
Input → Mark Removal → Margin Removal → Page Splitting → Deskewing → Table Detection → Table Cropping → Output
```

#### Step-by-Step Processing

1. **Mark Removal** (`mark_removal`)
   - **Purpose**: Remove watermarks, stamps, and document artifacts
   - **Method**: Otsu thresholding with content protection
   - **Key Parameter**: `dilate_iter` (2) - Controls mark detection sensitivity

2. **Margin Removal** (`margin_removal`) - **NEW: Inscribed Rectangle Method (Default)**
   - **Purpose**: Remove document margins and background
   - **Method**: Paper mask detection + largest inscribed rectangle
   - **Alternative**: Gradient detection method available
   - **Key Parameters**: 
     - `blur_ksize` (5) - Noise reduction
     - `close_ksize` (25) - Mask refinement
     - `close_iter` (2) - Morphology iterations

3. **Page Splitting** (`page_splitting`) - **NEW: V2 Algorithm**  
   - **Purpose**: Separate double-page scanned documents
   - **Method**: Enhanced vertical line detection with peak analysis
   - **Key Parameters**:
     - `search_ratio` (0.5) - Search area (center 50%)
     - `line_len_frac` (0.3) - Minimum line length
     - `peak_thr` (0.3) - Detection sensitivity

4. **Deskewing** (`deskewing`)
   - **Purpose**: Correct image rotation/skew
   - **Method**: Hough line transform with angle histogram
   - **Key Parameters**:
     - `angle_range` (5) - Maximum rotation to detect (±degrees)
     - `min_angle_correction` (0.1) - Minimum angle to apply

5. **Table Detection** (`table_detection`) - **Connected Components Method**
   - **Purpose**: Detect table structure and boundaries
   - **Method**: Morphological operations + component analysis  
   - **Key Parameters**:
     - `threshold` (40) - Binary threshold for line detection
     - `horizontal_kernel_size` (10) - Horizontal line kernel
     - `vertical_kernel_size` (10) - Vertical line kernel

6. **Table Cropping** (`table_detection.enable_table_cropping`)
   - **Purpose**: Extract table regions with padding
   - **Method**: Border-based extraction using detected bounds

## 🚀 Usage

### Command Line Usage

```bash
# Use default configurations (recommended for most cases)
python scripts/run_complete.py your_images/ --verbose

# Use custom Stage 1 configuration
python scripts/run_complete.py your_images/ --stage1-config my_custom.json --verbose

# Use custom configurations for both stages
python scripts/run_complete.py your_images/ \
    --stage1-config custom_stage1.json \
    --stage2-config custom_stage2.json \
    --verbose

# Debug mode with custom config
python scripts/run_complete.py your_images/ \
    --stage1-config academic_papers.json \
    --debug --verbose
```

### Programmatic Usage

```python
from src.ocr_pipeline import TwoStageOCRPipeline
from src.ocr_pipeline.config import Stage1Config, Stage2Config

# Load configurations
stage1_config = Stage1Config.from_json("configs/stage1_default.json")
stage2_config = Stage2Config.from_json("configs/stage2_default.json")

# Customize for your document type
stage1_config.angle_range = 15  # Wider rotation range
stage1_config.threshold = 25    # More sensitive line detection

# Initialize and run pipeline
pipeline = TwoStageOCRPipeline(stage1_config, stage2_config)
result = pipeline.process_complete("document.jpg", "output/")
```

## ⚡ Optimization Settings (NEW)

### Performance Optimization Configuration

The pipeline now supports advanced optimization settings for batch processing. Add these to your JSON configuration files:

```json
"optimization": {
  "parallel_processing": true,   // Enable parallel processing for batch jobs
  "max_workers": null,           // Number of workers (null = auto-detect CPU count - 1)
  "batch_size": 4,               // Images per batch for memory management
  "memory_mode": true            // Process in memory to reduce disk I/O
}
```

#### Optimization Modes

| Setting | Description | Best For | Performance Gain |
|---------|-------------|----------|------------------|
| **parallel_processing** | Process multiple images simultaneously | Batch processing (>5 images) | 3-10x speedup |
| **memory_mode** | Keep intermediate results in memory | All scenarios | 30-50% faster |
| **max_workers** | Control CPU usage | Server environments | Scales with cores |
| **batch_size** | Memory management | Limited RAM systems | Prevents OOM errors |

#### Recommended Configurations

**Fast Processing (Maximum Speed)**
```json
"optimization": {
  "parallel_processing": true,
  "max_workers": null,  // Use all available cores
  "batch_size": 4,
  "memory_mode": true
}
```

**Balanced (Speed + Stability)**
```json
"optimization": {
  "parallel_processing": true,
  "max_workers": 4,     // Limit to 4 workers
  "batch_size": 2,
  "memory_mode": true
}
```

**Conservative (Low Memory)**
```json
"optimization": {
  "parallel_processing": false,
  "max_workers": 1,
  "batch_size": 1,
  "memory_mode": false   // Write to disk between steps
}
```

## 🎛️ Key Configuration Parameters

### Essential Parameters by Document Type

#### Academic Papers (Clean Scans)
```json
{
  "angle_range": 3,           // Conservative rotation range
  "threshold": 45,            // Higher threshold for clean lines  
  "search_ratio": 0.3,        // Narrow search for centered binding
  "min_angle_correction": 0.2 // Less sensitive rotation
}
```

#### Historical Documents (Poor Quality)
```json
{
  "angle_range": 20,          // Wide rotation range
  "threshold": 20,            // Lower threshold for faint lines
  "search_ratio": 0.8,        // Wide search for varied layouts
  "min_angle_correction": 0.05 // More sensitive rotation
}
```

#### Mixed Document Collections  
```json
{
  "angle_range": 5,           // Moderate rotation range
  "threshold": 40,            // Standard threshold
  "search_ratio": 0.5,        // Standard search area
  "min_angle_correction": 0.1 // Standard sensitivity
}
```

## 🔄 Margin Removal Methods

The pipeline offers multiple margin removal approaches:

### Inscribed Rectangle Method (Default - Recommended)
```json
"margin_removal": {
  "enable": true,
  "use_gradient_detection": false,  // Use inscribed method
  "blur_ksize": 5,                  // Noise reduction
  "close_ksize": 25,                // Mask refinement  
  "close_iter": 2,                  // Morphology iterations
  "erode_after_close": 0            // Additional erosion
}
```

**Best for**: Most document types, especially those with clear paper boundaries
**Algorithm**: Paper mask detection + largest inscribed rectangle extraction

### Gradient Detection Method (Alternative)
```json  
"margin_removal": {
  "enable": true,
  "use_gradient_detection": true,   // Switch to gradient method
  "gradient_threshold": 30          // Edge sensitivity (20-40 range)
}
```

**Best for**: Documents with subtle or thin margins
**Algorithm**: Gradient magnitude analysis for margin boundary detection

## 📝 Document Type-Specific Configurations

### Creating Custom Configurations

```bash
# Start with default configuration
cp configs/stage1_default.json configs/my_document_type.json

# Edit for your specific needs
# - Adjust parameters based on visualization tool analysis
# - Test with representative samples
# - Document your changes

# Use your custom configuration  
python scripts/run_complete.py images/ --stage1-config configs/my_document_type.json
```

### Configuration Templates

#### High-Quality Scans (Aggressive Processing)
```json
{
  "verbose": true,
  "mark_removal": {"enable": true, "dilate_iter": 3},
  "margin_removal": {"enable": true, "close_ksize": 35},
  "deskewing": {"angle_range": 2, "min_angle_correction": 0.3},
  "table_detection": {"threshold": 50}
}
```

#### Poor Quality Scans (Sensitive Processing)  
```json
{
  "verbose": true,
  "margin_removal": {"enable": true, "blur_ksize": 9, "close_ksize": 20},
  "deskewing": {"angle_range": 15, "min_angle_correction": 0.05},
  "table_detection": {"threshold": 25, "horizontal_kernel_size": 15}
}
```

#### Fast Processing (Speed Optimized)
```json
{
  "verbose": false,
  "save_debug_images": false,
  "deskewing": {"angle_step": 0.5},
  "table_detection": {"horizontal_kernel_size": 15, "vertical_kernel_size": 15}
}
```

## 🧪 Parameter Optimization Workflow

### 1. Analyze Your Documents
```bash
# Use visualization tools to understand your document characteristics
python tools/visualize_page_split_v2.py sample_document.jpg
python tools/visualize_table_lines_v2.py sample_document.jpg  
python tools/visualize_margin_removal_v2.py sample_document.jpg --compare
```

### 2. Create Test Configuration
```bash
# Copy default and modify based on analysis
cp configs/stage1_default.json configs/test_config.json
# Edit test_config.json with your parameter adjustments
```

### 3. Test and Iterate
```bash
# Test with single representative image
python scripts/run_complete.py sample_image.jpg --stage1-config configs/test_config.json --debug --verbose

# Check debug outputs in data/debug/ to verify improvements
# Iterate on parameters until satisfied
```

### 4. Validate with Batch
```bash
# Test on small batch of representative images
python scripts/run_complete.py test_batch/ --stage1-config configs/test_config.json --verbose

# If results are good, use for full dataset
```

## ⚙️ Advanced Configuration Options

### Enable/Disable Processing Steps

Most processing steps can be enabled or disabled:

```json
{
  "mark_removal": {"enable": false},      // Skip mark removal
  "page_splitting": {"enable": false},    // Process as single page  
  "deskewing": {"enable": false},         // Skip rotation correction
  "table_detection": {
    "enable_table_cropping": false       // Detect but don't crop tables
  }
}
```

> **📖 Complete Step Control**: See [Enable/Disable Steps Guide](../docs/enable_disable_steps.md) for detailed information about which steps can be customized.

### Debug and Development Options

```json
{
  "verbose": true,                        // Detailed console output
  "save_debug_images": true,              // Save intermediate images  
  "debug_dir": "custom_debug_location"    // Custom debug directory
}
```

## 📚 Additional Resources

- **[Parameter Reference](../docs/PARAMETER_REFERENCE.md)**: Complete documentation of all 50+ parameters
- **[Quick Start Guide](../docs/QUICK_START.md)**: Step-by-step setup and first processing
- **[Troubleshooting Guide](../docs/TROUBLESHOOTING.md)**: Solutions to common configuration issues
- **[Tools Documentation](../tools/README.md)**: Visualization tools for parameter optimization

## 🎯 Best Practices

### Configuration Management
1. **Start with Defaults**: Use built-in configurations for initial testing
2. **Document Changes**: Keep notes on what works for different document types  
3. **Version Control**: Track configuration files with your project
4. **Test Systematically**: Change one parameter at a time when optimizing

### Parameter Tuning Strategy
1. **Visual First**: Use visualization tools to understand your documents
2. **Representative Samples**: Test with diverse, representative images
3. **Incremental Changes**: Make small parameter adjustments and measure impact
4. **Batch Validation**: Confirm parameter choices work across your full dataset

---

**Navigation**: [← Main README](../README.md) | [Documentation Index](../docs/README.md) | [Parameter Reference →](../docs/PARAMETER_REFERENCE.md)