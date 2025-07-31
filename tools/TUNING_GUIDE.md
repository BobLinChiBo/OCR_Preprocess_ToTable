# OCR Parameter Tuning Guide

A comprehensive guide for systematically tuning the OCR pipeline parameters to optimize performance for your specific document types.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Detailed Tuning Process](#detailed-tuning-process)
5. [Parameter Reference](#parameter-reference)
6. [Evaluation Guidelines](#evaluation-guidelines)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## 🎯 Overview

The OCR pipeline consists of four main stages that can be independently tuned:

1. **Page Splitting** - Separating double-page scans into individual pages
2. **Deskewing** - Correcting image rotation and alignment
3. **ROI Detection** - Identifying and cropping the content area
4. **Line Detection** - Finding table lines for structure extraction

Each stage builds on the previous one, allowing systematic optimization of the entire pipeline.

## ✅ Prerequisites

### Required Setup
```bash
# 1. Install dependencies
pip install opencv-python numpy

# 2. Set up directory structure
python tools/setup_tuning.py

# 3. Verify test images exist
dir "data/input/test_images/"
```

### Test Image Requirements
- **Minimum**: 3-6 representative test images
- **Format**: Common image formats (JPG, PNG, TIFF)
- **Content**: Should represent your typical document types
- **Quality**: Various quality levels to test robustness

## 🚀 Quick Start

```cmd
REM 1. Setup (one-time)
python tools\setup_tuning.py

REM 2. Run each tuning stage in sequence
python tools\tune_page_splitting.py
REM → Evaluate results, copy best to data\output\tuning\02_deskewed_input\

copy "data\output\tuning\01_split_pages\start0.4_end0.6_width50\*" "data\output\tuning\02_deskewed_input\"

python tools\tune_deskewing.py  
REM → Evaluate results, copy best to data\output\tuning\03_roi_input\

copy "data\output\tuning\02_deskewed\range5_step0.1_min0.1\*"  "data\output\tuning\03_roi_input\"

python tools\tune_roi_detection.py
REM → Evaluate results, copy best to data\output\tuning\04_line_input\

python tools\tune_line_detection.py
REM → Note optimal parameters

REM 3. Update parameters and run final pipeline
REM Edit TUNED_PARAMETERS in tools\run_tuned_pipeline.py
python tools\run_tuned_pipeline.py data\input\test_images\ --verbose
```

## 🔧 Detailed Tuning Process

### Stage 1: Page Splitting

**Purpose**: Split double-page scanned documents into separate pages

**Script**: `python tools/tune_page_splitting.py`

**Parameters Tested**:
- `gutter_search_start` (0.35, 0.4, 0.45) - Where to start looking for the page gutter
- `gutter_search_end` (0.55, 0.6, 0.65) - Where to stop looking for the page gutter

**Evaluation Criteria**:
- ✅ Clean separation at the book spine/gutter
- ✅ No content cut off from either page
- ✅ Consistent splitting across similar document types
- ❌ Content bleeding between pages
- ❌ Uneven page sizes when they should be equal

**Expected Results**:
- 9 parameter combination folders in `data/output/tuning/01_split_pages/`
- Each contains split page images for your test set
- Summary report with parameter details

**Next Step**:
```cmd
REM Copy the best results (replace 'best_folder' with actual folder name)
copy "data\output\tuning\01_split_pages\best_folder\*.jpg" "data\output\tuning\02_deskewed_input\"
```

### Stage 2: Deskewing

**Purpose**: Correct image rotation to make text horizontal and tables aligned

**Script**: `python tools/tune_deskewing.py`

**Parameters Tested**:
- `angle_range` (5°, 10°, 15°, 20°) - Maximum rotation angle to consider
- `angle_step` (0.1°, 0.2°, 0.5°) - Precision of angle detection (not used in basic implementation)
- `min_angle_correction` (0.1°, 0.2°, 0.5°, 1.0°) - Minimum angle before applying correction

**Evaluation Criteria**:
- ✅ Text lines are horizontal
- ✅ Table borders are straight (horizontal/vertical)
- ✅ No over-correction of already straight images
- ✅ Consistent correction across similar skew levels
- ❌ Wobbly or slanted text after correction
- ❌ Unnecessary rotation of straight images

**Expected Results**:
- 48 parameter combination folders in `data/output/tuning/02_deskewed/`
- Each contains deskewed images and angle analysis reports
- Look for balance between sensitivity and stability

**Interpretation**:
- **Low min_angle_correction**: More sensitive, corrects small angles
- **High min_angle_correction**: More conservative, only corrects obvious skew
- **High angle_range**: Handles severely rotated images
- **Low angle_range**: More stable for mostly straight images

**Next Step**:
```cmd
copy "data\output\tuning\02_deskewed\best_folder\*.jpg" "data\output\tuning\03_roi_input\"
```

### Stage 3: ROI Detection

**Purpose**: Detect and crop the content area, removing margins and noise

**Script**: `python tools/tune_roi_detection.py`

**Parameters Tested**:
- `gabor_kernel_size` (21, 31, 41) - Size of edge detection filter
- `gabor_sigma` (3.0, 4.0, 5.0) - Edge detection sensitivity  
- `gabor_lambda` (6.0, 8.0, 10.0) - Edge detection wavelength
- `roi_min_cut_strength` (10.0, 20.0, 30.0) - Threshold for boundary detection
- `roi_min_confidence_threshold` (3.0, 5.0, 7.0) - Confidence required for cropping

**Evaluation Criteria**:
- ✅ Focuses on table/content area
- ✅ Removes headers, footers, and wide margins
- ✅ Preserves all important content
- ✅ Consistent cropping across similar layouts
- ❌ Important content cut off
- ❌ Too much background noise preserved
- ❌ Inconsistent cropping behavior

**Area Ratio Guidelines**:
- **90-100%**: Very conservative, minimal cropping
- **70-90%**: Moderate cropping, good for mixed content  
- **50-70%**: Aggressive cropping, good for clean table extraction
- **<50%**: Very aggressive, check for over-cropping

**Expected Results**:
- ~20 parameter combination folders in `data/output/tuning/03_roi_detection/`
- Each contains ROI-cropped images (`*_roi.jpg`) and Gabor visualizations (`*_gabor.jpg`)
- Review area ratios and visual quality

**Next Step**:
```cmd
copy "data\output\tuning\03_roi_detection\best_folder\*_roi.jpg" "data\output\tuning\04_line_input\"
```

### Stage 4: Line Detection

**Purpose**: Detect table lines for structure extraction and final cropping

**Script**: `python tools/tune_line_detection.py`

**Parameters Tested**:
- `min_line_length` (20, 30, 40, 50, 60, 80) - Minimum length for valid table lines
- `max_line_gap` (5, 10, 15, 20, 25) - Maximum gap to bridge in broken lines

**Evaluation Criteria**:
- ✅ Detects main table structure lines
- ✅ Handles broken or faint lines appropriately
- ✅ Filters out noise and irrelevant marks
- ✅ Consistent detection across similar table types
- ❌ Missing obvious table lines
- ❌ Detecting too much noise as lines
- ❌ Inconsistent detection quality

**Parameter Effects**:
- **Lower min_line_length**: More sensitive, detects shorter segments
- **Higher min_line_length**: More selective, reduces noise
- **Lower max_line_gap**: Requires continuous lines
- **Higher max_line_gap**: Bridges gaps, connects broken lines

**Expected Results**:
- 30 parameter combination folders in `data/output/tuning/04_line_detection/`
- Each contains line visualizations (`*_lines.jpg`) and final table crops (`*_table.jpg`)
- Look for optimal detection rate and quality

## 📊 Parameter Reference

### Complete Parameter Set

```python
TUNED_PARAMETERS = {
    'page_splitting': {
        'gutter_search_start': 0.4,     # 0.3-0.5 typical range
        'gutter_search_end': 0.6,       # 0.5-0.7 typical range
    },
    'deskewing': {
        'angle_range': 10,              # 5-20° typical range
        'angle_step': 0.2,              # 0.1-0.5° typical range
        'min_angle_correction': 0.2,    # 0.1-1.0° typical range
    },
    'roi_detection': {
        'gabor_kernel_size': 31,        # 21-41 typical range
        'gabor_sigma': 4.0,             # 3.0-5.0 typical range
        'gabor_lambda': 8.0,            # 6.0-10.0 typical range
        'roi_min_cut_strength': 20.0,   # 10.0-30.0 typical range
        'roi_min_confidence_threshold': 5.0,  # 3.0-7.0 typical range
    },
    'line_detection': {
        'stage1': {
            'min_line_length': 40,      # 20-80 typical range
            'max_line_gap': 15,         # 5-25 typical range
        },
        'stage2': {
            'min_line_length': 30,      # 20-60 typical range  
            'max_line_gap': 5,          # 3-15 typical range
        }
    }
}
```

## 📏 Evaluation Guidelines

### Visual Inspection Checklist

**Page Splitting**:
- [ ] Clean separation at book spine
- [ ] No content bleeding between pages
- [ ] Consistent page sizes where expected
- [ ] Gutter correctly identified across test set

**Deskewing**:
- [ ] Text lines are horizontal
- [ ] Table borders are straight
- [ ] No unnecessary rotation of straight images
- [ ] Consistent correction quality

**ROI Detection**:
- [ ] Content area properly focused
- [ ] Headers/footers removed appropriately
- [ ] No important content cut off
- [ ] Reasonable area preservation ratio

**Line Detection**:
- [ ] Major table lines detected
- [ ] Minimal noise/false detections
- [ ] Broken lines handled well
- [ ] Final table crops look complete

### Quality Metrics

**Detection Rate**: Percentage of images where lines are successfully detected
- **Good**: >80% for clean documents, >60% for poor quality
- **Acceptable**: >70% for clean documents, >50% for poor quality
- **Poor**: <60% for clean documents, <40% for poor quality

**Area Preservation**: Ratio of cropped area to original
- **Conservative**: 85-95% (good for mixed content)
- **Moderate**: 70-85% (balanced approach)
- **Aggressive**: 50-70% (clean table extraction)

## 🐛 Troubleshooting

### Common Issues and Solutions

**Page Splitting Issues**:
- **Pages not split properly**: Adjust gutter search range
- **Content cut at edges**: Widen gutter search range
- **Multiple splits found**: Narrow gutter search range

**Deskewing Issues**:
- **Over-rotation**: Increase `min_angle_correction`
- **Under-rotation**: Decrease `min_angle_correction`
- **No rotation applied**: Check if angle detection is working

**ROI Detection Issues**:
- **Too much cropping**: Increase `roi_min_cut_strength`
- **Too little cropping**: Decrease `roi_min_cut_strength`
- **Inconsistent results**: Adjust Gabor filter parameters

**Line Detection Issues**:
- **Missing lines**: Decrease `min_line_length` or increase `max_line_gap`
- **Too much noise**: Increase `min_line_length` or decrease `max_line_gap`
- **No lines detected**: Check if ROI stage left enough content

### Performance Optimization

**Speed vs Quality Trade-offs**:
- Larger parameter ranges = better quality but slower tuning
- Fewer test images = faster but less representative
- More parameter combinations = more thorough but time-consuming

**Memory Usage**:
- Large images may require more memory
- Consider resizing very large images for tuning
- Clean up intermediate results if disk space is limited

## 💡 Best Practices

### General Guidelines

1. **Start Conservative**: Begin with moderate parameter values
2. **One Stage at a Time**: Complete each stage before moving to the next
3. **Document Everything**: Keep notes on what works and what doesn't
4. **Test Variety**: Use diverse test images representing your use cases
5. **Validate Results**: Always visually inspect parameter combinations

### Parameter Selection Strategy

1. **Identify Your Document Type**:
   - Clean scans vs poor quality
   - Consistent layouts vs varied
   - Simple tables vs complex structures

2. **Prioritize Stages**:
   - Focus more effort on stages that affect your specific use case
   - ROI detection is often most critical for mixed-content documents
   - Line detection is crucial for table extraction quality

3. **Balance Trade-offs**:
   - Sensitivity vs stability
   - Speed vs accuracy  
   - Automation vs manual control

### Production Recommendations

1. **Save Optimal Parameters**: Document your final parameter set
2. **Version Control**: Track parameter changes over time
3. **Batch Testing**: Test on larger datasets before production use
4. **Performance Monitoring**: Track success rates in production
5. **Regular Re-tuning**: Re-evaluate parameters as document types evolve

## 📁 File Organization

After tuning, your directory structure will look like:

```
data/output/tuning/
├── 01_split_pages/
│   ├── start0.4_end0.6_width50/     # Parameter combination folders
│   ├── parameter_test_summary.txt   # Summary report
│   └── ...
├── 02_deskewed/
│   ├── range10_step0.2_min0.2/      # Parameter combination folders  
│   ├── deskewing_test_summary.txt   # Summary report
│   └── ...
├── 03_roi_detection/
│   ├── k31_s4.0_l8.0_cs20.0_ct5.0/  # Parameter combination folders
│   ├── roi_detection_test_summary.txt
│   └── ...
├── 04_line_detection/
│   ├── minlen40_maxgap15/           # Parameter combination folders
│   ├── line_detection_test_summary.txt
│   └── ...
└── [input directories]/             # Intermediate staging areas
```

Each parameter folder contains:
- Processed images for that parameter set
- Analysis files (JSON, TXT)
- Visualization images where applicable

## 🎯 Success Criteria

Your tuning is successful when:

- [ ] Page splitting cleanly separates pages without content loss
- [ ] Deskewing straightens images without over-correction  
- [ ] ROI detection focuses on content while preserving important information
- [ ] Line detection reliably finds table structure
- [ ] Overall pipeline produces better results than default parameters
- [ ] Parameters work consistently across your document types

Remember: The goal is optimization for **your specific document types**, not universal perfection.