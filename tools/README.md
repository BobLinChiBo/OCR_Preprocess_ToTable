# OCR Pipeline Tools

Comprehensive visualization and parameter tuning tools for optimizing OCR pipeline performance on your specific document types.

## Quick Start

### Visualization (Debugging & Analysis)
```bash
# Complete pipeline analysis
python tools/run_visualizations.py all --pipeline image.jpg --save-intermediates

# Individual step analysis
python tools/visualize_deskew.py image.jpg
python tools/visualize_page_split.py image.jpg
python tools/visualize_roi.py image.jpg
```

### Parameter Tuning (Optimization)
```bash
# Interactive guided tuning
python tools/quick_start_tuning.py

# Manual step-by-step tuning
python tools/setup_tuning.py                 # One-time setup
python tools/tune_page_splitting.py          # Stage 1
python tools/tune_deskewing.py               # Stage 2  
python tools/tune_roi_detection.py           # Stage 3
python tools/tune_line_detection.py          # Stage 4
python tools/run_tuned_pipeline.py input/    # Final test
```

### Results Management
```bash
python tools/check_results.py list           # View recent runs
python tools/check_results.py latest deskew --view  # Open HTML report
python tools/compare_results.py              # Compare parameter sets
```

## Visualization Tools

### Individual Step Analysis

| Script | Purpose | Key Parameters |
|--------|---------|----------------|
| `run_visualizations.py all --pipeline` | Complete workflow analysis | `--save-intermediates` |
| `visualize_deskew.py` | Skew detection/correction | `--angle-range`, `--angle-step` |
| `visualize_page_split.py` | Two-page document splitting | `--gutter-start`, `--gutter-end` |
| `visualize_roi.py` | Region of interest detection | `--gabor-threshold`, `--cut-strength` |
| `visualize_table_lines.py` | Table line detection | `--min-line-length`, `--kernel-h-size` |
| `visualize_table_crop.py` | Final table cropping | `--crop-padding` |

### Basic Usage Examples
```bash
# Test different page splitting parameters
python tools/visualize_page_split.py image.jpg --gutter-start 0.35 --gutter-end 0.65

# Analyze deskewing with custom angle range
python tools/visualize_deskew.py image.jpg --angle-range 30 --angle-step 1.0

# ROI detection with debug output
python tools/visualize_roi.py image.jpg --gabor-threshold 150 --save-debug

# Batch visualization
python tools/run_visualizations.py deskew page-split roi image.jpg
```

## Parameter Tuning Tools

### Core Tuning Scripts

| Script | Stage | Parameters Tested | Output Directory |
|--------|-------|------------------|------------------|
| `tune_page_splitting.py` | 1 | Gutter search range | `01_split_pages/` |
| `tune_deskewing.py` | 2 | Angle detection | `02_deskewed/` |
| `tune_roi_detection.py` | 3 | Edge detection & cropping | `03_roi_detection/` |
| `tune_line_detection.py` | 4 | Line detection sensitivity | `04_line_detection/` |

### Support Scripts

| Script | Purpose |
|--------|---------|
| `setup_tuning.py` | Initialize directory structure and test setup |
| `quick_start_tuning.py` | Interactive guided tuning process |
| `run_tuned_pipeline.py` | Test optimized parameters on new images |
| `compare_results.py` | Compare different parameter combinations |

### Tuning Workflow

1. **Setup** (one-time): `python tools/setup_tuning.py`
2. **Stage 1 - Page Splitting**: `python tools/tune_page_splitting.py`
   - Review results in `data/output/tuning/01_split_pages/`
   - Copy best results to `02_deskewed_input/`
3. **Stage 2 - Deskewing**: `python tools/tune_deskewing.py`
   - Review results in `data/output/tuning/02_deskewed/`
   - Manually copy best results to `03_roi_input/`
4. **Stage 3 - ROI Detection**: `python tools/tune_roi_detection.py`
   - Review results in `data/output/tuning/03_roi_detection/`
   - Manually copy best results to `04_line_input/`
5. **Stage 4 - Line Detection**: `python tools/tune_line_detection.py`
   - Review results in `data/output/tuning/04_line_detection/`
   - Note optimal parameters
6. **Final Test**: Update parameters in `run_tuned_pipeline.py` and test

> **📖 Detailed Guide**: See `TUNING_GUIDE.md` for comprehensive parameter tuning instructions.

## Management Commands

### Results Browser
```bash
python tools/check_results.py list                    # List all recent runs
python tools/check_results.py show 0                  # Show details for run #0  
python tools/check_results.py view 0                  # Open HTML viewer
python tools/check_results.py compare deskew          # Compare deskew runs
python tools/check_results.py cleanup --keep 5        # Keep only latest 5 runs
```

### Batch Operations  
```bash
python tools/run_visualizations.py --list             # List available scripts
python tools/run_visualizations.py all image.jpg      # Run all visualizations
python tools/run_visualizations.py deskew roi image.jpg \
  --deskew-args --angle-range 30 \
  --roi-args --gabor-threshold 150
```

## Parameter Reference

### Key Parameters by Stage

**Page Splitting**
- `--gutter-start 0.3-0.5` - Where to start looking for page boundary
- `--gutter-end 0.5-0.7` - Where to stop looking for page boundary
- `--min-gutter-width 20-100` - Minimum gap width (pixels)

**Deskewing**  
- `--angle-range 5-30` - Maximum rotation to detect (degrees)
- `--angle-step 0.1-1.0` - Detection precision (degrees)
- `--min-angle-correction 0.1-2.0` - Minimum angle to trigger rotation

**ROI Detection**
- `--gabor-threshold 90-180` - Edge detection sensitivity
- `--cut-strength 5.0-30.0` - Content boundary detection strength
- `--gabor-kernel-size 21-51` - Edge filter size

**Line Detection**
- `--min-line-length 20-80` - Minimum line length (pixels)
- `--max-line-gap 5-25` - Maximum gap to bridge in broken lines
- `--kernel-h-size 20-80` - Horizontal morphology kernel size

### Interpreting Results

**✅ Good Results**
- **Page Split**: Clean boundary, equal page widths, no text cutting
- **Deskew**: Horizontal text lines, reasonable rotation angle
- **ROI**: Precise content focus, appropriate margin removal
- **Lines**: Clear table grid detection, minimal noise

**❌ Needs Adjustment**
- **Split Issues**: Adjust gutter search range or width threshold
- **Deskew Issues**: Modify angle range or correction threshold  
- **ROI Issues**: Adjust threshold or cut strength parameters
- **Line Issues**: Change line length or morphology parameters

## Output Organization

### Visualization Outputs (Default: Organized Structure)
```
visualization_outputs/
├── deskew_20240726_143022/
│   ├── images/           # Visual outputs
│   ├── analysis/         # JSON metrics  
│   ├── comparisons/      # Overlay images
│   └── results.html      # Summary report
└── page_split_20240726_143105/
    └── ...
```

### Tuning Outputs
```
data/output/tuning/
├── 01_split_pages/
│   ├── start0.4_end0.6_width50/     # Parameter combinations
│   └── parameter_test_summary.txt
├── 02_deskewed/ 
├── 03_roi_detection/
└── 04_line_detection/
```

## Advanced Usage

### Configuration Files
```bash
# Create parameter config
echo '{
  "gutter_search_start": 0.35,
  "enable_roi_detection": true,
  "gabor_binary_threshold": 150,
  "min_line_length": 80
}' > config.json

# Use with pipeline (note: config files not yet supported with run_visualizations.py)
python tools/run_visualizations.py all --pipeline image.jpg
```

### Batch Processing
```bash
# Test multiple images
python tools/run_visualizations.py all --pipeline input/*.jpg --save-intermediates

# Parameter sensitivity testing
for t in 100 120 140 160; do
  python tools/visualize_roi.py image.jpg --gabor-threshold $t --output-dir "test_$t"
done
```

## Troubleshooting

**Setup Issues**
- Ensure test images exist in `data/input/test_images/`
- Check that output directories have write permissions
- Verify OpenCV and dependencies are installed

**Poor Results**
- Use higher resolution images (300+ DPI recommended)
- Ensure good contrast between text and background
- Try preprocessing for noisy or low-quality documents

**Performance Issues**
- Start with smaller test sets before scaling up
- Use `--verbose` flag to identify bottlenecks
- Consider resizing very large images for tuning

**Memory Issues**
- Process images individually before batch operations
- Clean up intermediate results if disk space is limited
- Use `--save-intermediates` selectively

## Workflow Recommendations

### For New Document Types
1. **Quick Assessment**: `python tools/run_visualizations.py all --pipeline sample.jpg --save-intermediates`
2. **Identify Problems**: Review comparison images to find issues
3. **Systematic Tuning**: Use `python tools/quick_start_tuning.py` for guided optimization
4. **Validation**: Test optimized parameters on larger dataset with `run_tuned_pipeline.py`

### For Production Use  
1. **Document Parameters**: Save optimal settings in configuration files
2. **Version Control**: Track parameter changes over time
3. **Monitoring**: Regularly validate performance on new document batches
4. **Re-tuning**: Re-evaluate parameters as document types evolve

---

**📖 Additional Resources**
- `TUNING_GUIDE.md` - Comprehensive parameter tuning guide
- `tuning_progress.md` - Progress tracking template
- Individual script help: `python tools/[script_name].py --help`