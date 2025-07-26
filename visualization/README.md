# OCR Visualization Scripts

This directory contains visualization scripts for analyzing and debugging the OCR pipeline. The scripts now support both organized timestamp-based output structure and traditional flat output.

## 🚀 Quick Start

### View Latest Results
```bash
# View latest deskew results in HTML
python visualization/check_results.py latest deskew --view

# List all recent runs
python visualization/check_results.py list
```

### Run Single Visualization
```bash
# Run with organized output (default)
python visualization/visualize_deskew.py image.jpg

# Run with flat output (old style)
python visualization/visualize_deskew.py image.jpg --flat-output --output-dir my_results/
```

### Run Multiple Visualizations
```bash
# Run multiple scripts at once
python visualization/run_visualizations.py deskew page-split roi image.jpg

# Run all visualization scripts
python visualization/run_visualizations.py all image.jpg
```

## 📁 Output Organization

### New Organized Structure (Default)
```
visualization_outputs/
├── deskew_20240726_143022/
│   ├── images/           # All image outputs
│   ├── analysis/         # JSON analysis files
│   ├── comparisons/      # Overlay and comparison images
│   ├── run_metadata.json
│   └── deskew_results.html
├── page_split_20240726_143105/
└── ...
```

### Traditional Flat Structure (with --flat-output)
```
deskew_visualization/
├── image1_original.jpg
├── image1_deskewed.jpg
├── image1_comparison.jpg
└── deskew_visualization_summary.json
```

## 📊 Available Scripts

| Script | Description | Default Output Folder |
|--------|-------------|----------------------|
| `visualize_deskew.py` | Deskewing analysis | `deskew_visualization` |
| `visualize_page_split.py` | Page splitting | `page_split_visualization` |
| `visualize_roi.py` | ROI detection | `roi_visualization` |
| `visualize_table_lines.py` | Table line detection | `table_lines_visualization` |
| `visualize_table_crop.py` | Table cropping | `table_crop_visualization` |
| `visualize_pipeline.py` | Complete pipeline | `pipeline_visualization` |

## 🛠 Management Commands

### Results Checker
```bash
# List recent runs
python visualization/check_results.py list

# Show details of a specific run
python visualization/check_results.py show 0

# Open HTML viewer for a run
python visualization/check_results.py view 0

# Compare runs of the same script
python visualization/check_results.py compare deskew

# Clean up old runs (keep latest 5 per script)
python visualization/check_results.py cleanup --keep 5
```

### Visualization Runner
```bash
# List available scripts
python visualization/run_visualizations.py --list

# Run multiple scripts with custom args
python visualization/run_visualizations.py deskew page-split image.jpg \
  --deskew-args --angle-range 30 \
  --page-split-args --gutter-start 0.3

# Save execution report
python visualization/run_visualizations.py all image.jpg --save-report
```

## 🎯 Individual Step Tuning
To focus on specific processing steps:

```bash
# Page splitting parameters
python visualization/visualize_page_split.py your_image.jpg --gutter-start 0.35 --gutter-end 0.65

# Deskewing parameters  
python visualization/visualize_deskew.py your_image.jpg --angle-range 30 --angle-step 1.0

# Table line detection
python visualization/visualize_table_lines.py your_image.jpg --min-line-length 80 --kernel-h-size 50

# Table cropping
python visualization/visualize_table_crop.py your_image.jpg --crop-padding 15

# ROI detection
python visualization/visualize_roi.py your_image.jpg --gabor-threshold 150 --cut-strength 15.0
```

## 📊 Available Tools

### 1. Complete Pipeline (`visualize_pipeline.py`)

Runs your entire OCR pipeline with visualization at each step.

**Features:**
- Step-by-step processing visualization
- Complete workflow assessment
- Parameter override capabilities
- Intermediate step saving

**Usage:**
```bash
# Basic pipeline visualization
python visualization/visualize_pipeline.py input/raw_images/Wang2017_Page_001.jpg

# With parameter overrides
python visualization/visualize_pipeline.py your_image.jpg \
    --enable-roi --min-line-length 120 --save-intermediates
```

### 2. Page Split Visualization (`visualize_page_split.py`)

Visualizes two-page document splitting and gutter detection.

**Usage:**
```bash
# Basic page split visualization
python visualization/visualize_page_split.py input/raw_images/Wang2017_Page_001.jpg

# Test different gutter detection parameters
python visualization/visualize_page_split.py your_image.jpg \
    --gutter-start 0.35 --gutter-end 0.65 --min-gutter-width 40
```

**What it creates:**
- `*_split_comparison.jpg` - Complete split analysis view
- `*_split_overlay.jpg` - Original with gutter detection overlay
- `*_left_page.jpg` / `*_right_page.jpg` - Split page results
- `*_gutter_analysis.jpg` - Vertical projection analysis
- `*_split_analysis.json` - Gutter detection metrics

### 3. Deskew Visualization (`visualize_deskew.py`)

Analyzes and visualizes skew detection and rotation correction.

**Usage:**
```bash
# Basic deskew analysis
python visualization/visualize_deskew.py input/raw_images/Wang2017_Page_001.jpg

# Test different angle detection parameters
python visualization/visualize_deskew.py your_image.jpg \
    --angle-range 30 --angle-step 1.0 --min-angle-correction 1.0
```

**What it creates:**
- `*_deskew_comparison.jpg` - Complete deskew analysis
- `*_line_detection.jpg` - Detected lines overlay
- `*_edges.jpg` - Edge detection visualization
- `*_angle_analysis.jpg` - Angle distribution plot
- `*_deskew_analysis.json` - Skew detection metrics

### 4. Table Lines Visualization (`visualize_table_lines.py`)

Visualizes table line detection using morphological operations.

**Usage:**
```bash
# Basic line detection
python visualization/visualize_table_lines.py input/raw_images/Wang2017_Page_001.jpg

# Test different morphological parameters
python visualization/visualize_table_lines.py your_image.jpg \
    --min-line-length 80 --kernel-h-size 50 --kernel-v-size 50 --hough-threshold 60
```

**What it creates:**
- `*_table_lines_comparison.jpg` - Complete line detection analysis
- `*_table_lines.jpg` - Detected lines overlay
- `*_morphology.jpg` - Morphological operations visualization
- `*_line_stats.jpg` - Line length distribution plots
- `*_table_lines_analysis.json` - Line detection metrics

### 5. Table Crop Visualization (`visualize_table_crop.py`)

Visualizes final table region cropping based on detected lines.

**Usage:**
```bash
# Basic crop visualization
python visualization/visualize_table_crop.py input/raw_images/Wang2017_Page_001.jpg

# Test different cropping parameters
python visualization/visualize_table_crop.py your_image.jpg \
    --min-line-length 100 --crop-padding 15
```

**What it creates:**
- `*_crop_comparison.jpg` - Complete cropping analysis
- `*_crop_overlay.jpg` - Crop boundaries overlay
- `*_cropped.jpg` - Final cropped table region
- `*_spacing_analysis.jpg` - Line spacing analysis
- `*_crop_analysis.json` - Cropping metrics

### 6. ROI Visualization (`visualize_roi.py`)

Visualizes ROI detection results to help you assess parameter quality.

**Usage:**
```bash
# Basic ROI visualization
python visualization/visualize_roi.py input/raw_images/Wang2017_Page_001.jpg

# Test specific ROI parameters
python visualization/visualize_roi.py your_image.jpg \
    --gabor-threshold 150 --cut-strength 15.0 --show-gabor --save-debug
```

**What it creates:**
- `*_comparison.jpg` - Side-by-side original vs ROI detection
- `*_roi_overlay.jpg` - Original with ROI boundaries overlaid
- `*_roi_cropped.jpg` - Just the detected table region
- `*_roi_coords.json` - ROI coordinates and metrics
- Debug images showing Gabor filter steps (with --save-debug)

## 🎯 Parameter Guide

### Page Splitting Parameters

| Parameter | Purpose | Typical Range | Effect |
|-----------|---------|---------------|----------|
| `gutter_search_start` | Gutter search start position | 0.3-0.5 | Where to start looking for page split |
| `gutter_search_end` | Gutter search end position | 0.5-0.7 | Where to stop looking for page split |
| `min_gutter_width` | Minimum gutter width | 20-100px | Validates gutter detection quality |

### Deskewing Parameters

| Parameter | Purpose | Typical Range | Effect |
|-----------|---------|---------------|----------|
| `angle_range` | Max rotation to detect | 15-60° | Larger = detects more extreme skew |
| `angle_step` | Detection precision | 0.1-2.0° | Smaller = more precise but slower |
| `min_angle_correction` | Rotation threshold | 0.1-2.0° | Minimum angle to trigger rotation |

### Table Line Detection Parameters

| Parameter | Purpose | Typical Range | Effect |
|-----------|---------|---------------|----------|
| `min_line_length` | Minimum line length | 50-200px | Shorter = detects more lines |
| `max_line_gap` | Maximum gap in lines | 5-20px | Larger = connects broken lines |
| `kernel_h_size` | Horizontal morphology kernel | 20-80px | Larger = detects thicker lines |
| `kernel_v_size` | Vertical morphology kernel | 20-80px | Larger = detects thicker lines |
| `hough_threshold` | Line detection threshold | 30-100 | Lower = more sensitive detection |

### ROI Detection Parameters

| Parameter | Purpose | Typical Range | Effect |
|-----------|---------|---------------|----------|
| `gabor_binary_threshold` | Edge detection sensitivity | 90-180 | Lower = more sensitive edge detection |
| `roi_min_cut_strength` | Content boundary detection | 5.0-30.0 | Higher = more aggressive cropping |
| `roi_min_confidence_threshold` | Cut reliability | 2.0-15.0 | Higher = more conservative cuts |
| `gabor_kernel_size` | Edge filter size | 21-51 | Larger = detects thicker edges |
| `gabor_sigma` | Gaussian width | 2.0-8.0 | Affects edge detection smoothness |

### Table Cropping Parameters

| Parameter | Purpose | Typical Range | Effect |
|-----------|---------|---------------|----------|
| `crop_padding` | Boundary padding | 5-25px | Extra space around detected table |

## 📋 Recommended Workflow

### For New Document Types:

1. **Start with complete pipeline**:
   ```bash
   python visualization/visualize_pipeline.py your_sample_image.jpg --save-intermediates
   ```

2. **Identify problem steps** by reviewing the comparison images

3. **Tune specific steps**:
   ```bash
   # If page splitting is problematic
   python visualization/visualize_page_split.py your_image.jpg --gutter-start 0.35
   
   # If deskewing is problematic  
   python visualization/visualize_deskew.py your_image.jpg --angle-range 30
   
   # If line detection is problematic
   python visualization/visualize_table_lines.py your_image.jpg --min-line-length 80
   
   # If final cropping is problematic
   python visualization/visualize_table_crop.py your_image.jpg --crop-padding 20
   
   # If ROI detection is problematic
   python visualization/visualize_roi.py your_image.jpg --gabor-threshold 140
   ```

4. **Test complete pipeline with optimized parameters**:
   ```bash
   python visualization/visualize_pipeline.py your_image.jpg \
       --gutter-start 0.35 --min-line-length 80 --enable-roi
   ```

### For Production Use:

Update your config with optimized parameters:

```python
from ocr.config import Config

# Use recommended parameters from visualization testing
config = Config(
    # Page splitting
    gutter_search_start=0.35,        # From page split tuning
    gutter_search_end=0.65,          # From page split tuning
    min_gutter_width=40,             # From page split tuning
    
    # Deskewing
    angle_range=30,                  # From deskew tuning
    angle_step=1.0,                  # From deskew tuning
    min_angle_correction=1.0,        # From deskew tuning
    
    # Line detection
    min_line_length=80,              # From line detection tuning
    max_line_gap=15,                 # From line detection tuning
    
    # ROI detection (if used)
    enable_roi_detection=True,       # Enable/disable as needed
    gabor_binary_threshold=150,      # From ROI tuning
    roi_min_cut_strength=15.0,       # From ROI tuning
    roi_min_confidence_threshold=5.0, # From ROI tuning
    
    verbose=True
)
```

## 🔍 Interpreting Results

### Good Page Splitting:
- ✅ Gutter detected in the binding area between pages
- ✅ Pages split cleanly without cutting through text
- ✅ Left and right pages are roughly equal in width
- ✅ Gutter strength indicates clear boundary detection

### Good Deskewing:
- ✅ Text lines appear horizontal after rotation
- ✅ Rotation angle is reasonable for document condition
- ✅ High confidence score indicates reliable detection
- ✅ Edge detection shows clear line structures

### Good Line Detection:
- ✅ Table grid lines clearly detected in red (horizontal) and blue (vertical)
- ✅ Minimal noise lines outside table area
- ✅ Line count matches expected table structure
- ✅ Morphological operations show clean line enhancement

### Good Table Cropping:
- ✅ Crop boundary tightly encompasses entire table
- ✅ Minimal non-table content included
- ✅ No table content cut off at edges
- ✅ Good coverage ratio (typically 40-80% of original image)

### Good ROI Detection:
- ✅ ROI box tightly encompasses the entire table
- ✅ Minimal non-table content included (margins, headers)
- ✅ No table content cut off at edges
- ✅ Consistent performance across similar document layouts

### Signs to Adjust Parameters:

**Page splitting issues:**
- Wrong split location: Adjust `gutter_search_start/end` range
- No gutter detected: Decrease `min_gutter_width` requirement
- Multiple gutters: Narrow the search range

**Deskewing issues:**
- No rotation when needed: Decrease `min_angle_correction`
- Over-rotation: Increase `min_angle_correction` or check `angle_range`
- Poor line detection: Image may need preprocessing

**Line detection issues:**
- Too many noise lines: Increase `min_line_length` or `hough_threshold`
- Missing table lines: Decrease thresholds or adjust kernel sizes
- Broken line detection: Increase `max_line_gap`

**Cropping issues:**
- Crop too tight: Increase `crop_padding`
- Crop includes noise: Improve line detection parameters first
- No crop region: Check if table lines are being detected

**ROI issues:**
- ROI too large: Increase `roi_min_cut_strength` or `gabor_binary_threshold`
- ROI too small: Decrease `roi_min_cut_strength` or `roi_min_confidence_threshold`
- Inconsistent detection: Adjust `gabor_kernel_size` or increase confidence thresholds

## 📁 Output Files

Each visualization script creates organized output directories:

### Complete Pipeline Output:
```
pipeline_visualization/
├── *_pipeline_comparison.jpg      # Complete workflow overview
├── *_pipeline_summary.json        # Detailed processing metrics
├── *_00_original.jpg              # Original document (with --save-intermediates)
├── *_01_split_overlay.jpg          # Page splitting visualization
├── *_02_left_page.jpg              # Left page result
├── *_03_right_page.jpg             # Right page result
├── *_04_1_deskewed.jpg             # Page 1 after deskewing
├── *_05_1_deskew_overlay.jpg       # Page 1 deskew analysis
├── *_06_1_roi_processed.jpg        # Page 1 after ROI (if enabled)
├── *_07_1_roi_overlay.jpg          # Page 1 ROI visualization
├── *_08_1_lines_overlay.jpg        # Page 1 line detection
├── *_09_1_crop_overlay.jpg         # Page 1 crop analysis
├── *_10_1_final_result.jpg         # Page 1 final output
└── ... (similar files for page 2)
```

### Individual Step Outputs:
```
step_visualization/
├── *_comparison.jpg               # Step-specific comparison view
├── *_overlay.jpg                  # Processing overlay
├── *_analysis.json                # Step metrics and parameters
└── *_summary.json                 # Summary statistics
```

## 🛠 Advanced Usage

### Batch Processing

Process multiple images efficiently:

```bash
# Process all images in a directory
python visualization/visualize_pipeline.py input/raw_images/*.jpg --save-intermediates

# Test specific step on multiple images
python visualization/visualize_deskew.py input/raw_images/*.jpg --angle-range 45

# Compare different parameter settings
python visualization/visualize_roi.py input/raw_images/*.jpg --gabor-threshold 120
python visualization/visualize_roi.py input/raw_images/*.jpg --gabor-threshold 160
```

### Configuration File Usage

Use JSON configuration files for complex parameter sets:

```bash
# Create a config file
echo '{
  "gutter_search_start": 0.35,
  "gutter_search_end": 0.65,
  "enable_roi_detection": true,
  "gabor_binary_threshold": 150,
  "min_line_length": 80
}' > my_config.json

# Use with pipeline
python visualization/visualize_pipeline.py your_image.jpg --config-file my_config.json
```

### Parameter Sensitivity Testing

Systematically test parameter ranges:

```bash
# Test different thresholds
for threshold in 100 120 140 160; do
  python visualization/visualize_roi.py your_image.jpg \
    --gabor-threshold $threshold --output-dir "roi_thresh_$threshold"
done

# Test different line lengths
for length in 60 80 100 120; do
  python visualization/visualize_table_lines.py your_image.jpg \
    --min-line-length $length --output-dir "lines_len_$length"
done
```

## 🚨 Troubleshooting

**Error: "No valid images found"**
- Check that image paths are correct
- Ensure images are in supported formats (.jpg, .png, .tiff, .bmp)
- Use absolute paths or verify working directory

**Poor page splitting**
- Check if document is actually two-page or single-page
- Adjust gutter search range for your document layout
- Verify image resolution is sufficient for gutter detection

**Inconsistent deskewing**
- Some documents may not have enough straight lines for detection
- Try adjusting angle range and step size
- Check if preprocessing (noise reduction) is needed

**No table lines detected**
- Verify document actually contains table structures
- Try different morphological kernel sizes
- Check if image resolution/quality is sufficient
- Consider preprocessing steps (denoising, contrast enhancement)

**Pipeline crashes or hangs**
- Start with single images before batch processing
- Check available memory for large images
- Use `--verbose` flag to identify problem steps
- Try processing smaller sections of large documents

**Visualization images appear blank or incorrect**
- Check OpenCV installation and image codec support
- Verify input images are not corrupted
- Some image formats may need additional dependencies

**Memory issues with large images**
- Scripts automatically resize images for display efficiency
- Large source images are processed at full resolution for accuracy
- Consider testing on smaller representative samples first
- Use `--save-intermediates` selectively for large batches

**Parameter changes have no visible effect**
- Ensure parameter values are within reasonable ranges
- Check if step is actually being applied (e.g., ROI detection enabled)
- Some parameters only affect edge cases - try more extreme values for testing
- Review the JSON analysis files for numerical changes even if visual differences are subtle

## 🎓 Tips for Best Results

### Document Quality
- Higher resolution images (300+ DPI) generally work better
- Good contrast between text/lines and background is essential
- Minimize noise, shadows, and artifacts when possible

### Parameter Tuning Strategy
1. **Start with defaults** - Use default parameters as baseline
2. **One step at a time** - Focus on one problematic step rather than changing everything
3. **Visual validation** - Always review comparison images, don't rely only on metrics
4. **Test variety** - Use representative samples of your document types
5. **Document changes** - Keep notes on what parameters work for which document types

### Performance Optimization
- Use `visualize_pipeline.py` for overview, individual scripts for detailed tuning
- Test on small image sets first, then scale up
- Save successful parameter combinations in configuration files
- Consider preprocessing steps if document quality is poor