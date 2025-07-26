# Configuration Guide

This guide explains how to configure the OCR Table Extraction Pipeline for optimal results with your specific document types and requirements.

## 📋 Table of Contents

- [Overview](#overview)
- [Configuration Files](#configuration-files)
- [Directory Structure Configuration](#directory-structure-configuration)
- [Stage 1 Configuration](#stage-1-configuration)
- [Stage 2 Configuration](#stage-2-configuration)
- [Parameter Tuning Guide](#parameter-tuning-guide)
- [Troubleshooting](#troubleshooting)

## 🔍 Overview

The pipeline uses two JSON configuration files:
- **`stage1_config.json`** - Controls initial processing of raw scanned images
- **`stage2_config.json`** - Controls refinement of cropped table images

Each configuration file contains multiple sections controlling different aspects of processing.

## 📁 Configuration Files

### Stage 1 Configuration (`stage1_config.json`)
Used by `run_stage1_initial_processing.py` for processing raw scanned images.

### Stage 2 Configuration (`stage2_config.json`)  
Used by `run_stage2_refinement.py` for refining cropped table images.

## 🗂️ Directory Structure Configuration

Both configuration files start with a `directories` section that defines input and output paths:

```json
{
  "directories": {
    "raw_images": "input/raw_images",
    "splited_images": "output/stage1_initial_processing/01_split_pages",
    "deskewed_images": "output/stage1_initial_processing/02_deskewed",
    "lines_images": "output/stage1_initial_processing/03_line_detection",
    "table_images": "output/stage1_initial_processing/04_table_reconstruction",
    "table_fit_images": "output/stage1_initial_processing/05_cropped_tables",
    "debug_output_dir": "debug/stage1_debug/line_detection"
  }
}
```

### Key Directory Roles:
- **`raw_images`** - Input directory for original scanned images (Stage 1 only)
- **`splited_images`** - Individual pages from page splitting (Stage 1) or cropped tables (Stage 2 input)
- **`debug_output_dir`** - Debug images for parameter tuning (if enabled)

## 🚀 Stage 1 Configuration

Stage 1 processes raw scanned images through the complete initial workflow.

### Page Splitting Section
Controls separation of double-page scans:

```json
{
  "page_splitting": {
    "GUTTER_SEARCH_START_PERCENT": 0.4,
    "GUTTER_SEARCH_END_PERCENT": 0.6,
    "SPLIT_THRESHOLD": 0.8,
    "LEFT_PAGE_SUFFIX": "_page_2.jpg",
    "RIGHT_PAGE_SUFFIX": "_page_1.jpg"
  }
}
```

**Key Parameters:**
- **Gutter Search Region**: `0.4` to `0.6` means search for the binding in the central 20% of image width
- **Split Threshold**: `0.8` is the confidence threshold for detecting double-page layout
- **Page Suffixes**: Naming convention for split pages (verso = page_2, recto = page_1)

### Deskewing Section
Controls rotation correction:

```json
{
  "deskewing": {
    "ANGLE_RANGE": 10.0,
    "ANGLE_STEP": 0.2,
    "MIN_ANGLE_FOR_CORRECTION": 0.2
  }
}
```

**Key Parameters:**
- **Angle Range**: Search ±10 degrees for optimal rotation
- **Angle Step**: 0.2-degree precision in angle detection
- **Minimum Correction**: Only correct if detected skew > 0.2 degrees

### Line Detection Section
The most complex section controlling table line detection:

```json
{
  "line_detection": {
    "SAVE_DEBUG_IMAGES": true,
    "ROI_MARGINS_PAGE_1": { "top": 120, "bottom": 120, "left": 0, "right": 100 },
    "ROI_MARGINS_PAGE_2": { "top": 120, "bottom": 120, "left": 60, "right": 5 },
    "ROI_MARGINS_DEFAULT": { "top": 60, "bottom": 60, "left": 5, "right": 5 }
  }
}
```

**ROI Margins** exclude page headers, footers, and margins:
- **PAGE_1** (recto): Wider right margin due to binding
- **PAGE_2** (verso): Wider left margin due to binding  
- **DEFAULT**: For single pages or unknown types

#### Vertical Line Parameters (`V_PARAMS`)
Controls detection of vertical table lines:

```json
{
  "V_PARAMS": {
    "morph_open_kernel_ratio": 0.0166,
    "morph_close_kernel_ratio": 0.0166,
    "hough_threshold": 5,
    "hough_min_line_length": 40,
    "hough_max_line_gap_ratio": 0.001,
    "cluster_distance_threshold": 15,
    "qualify_length_ratio": 0.5,
    "final_selection_ratio": 0.5,
    "solid_check_std_threshold": 30.0,
    "contour_min_length_ratio": 0.5,
    "contour_aspect_ratio_threshold": 5.0
  }
}
```

**Morphological Operations:**
- **Open/Close Kernel Ratios**: Size relative to image dimensions (1.66% of height)
- Used to isolate vertical structures and repair broken lines

**Hough Transform Parameters:**
- **Threshold**: Minimum intersections for line detection (lower = more sensitive)
- **Min Line Length**: Minimum length in pixels for valid lines
- **Max Line Gap**: Maximum gap to bridge (as ratio of image height)

**Clustering and Filtering:**
- **Cluster Distance**: Group lines within 15 pixels
- **Quality Ratios**: Progressive filtering of detected lines
- **Solid Check**: Remove solid regions with low variance (std < 30)

**Curved Line Detection:**
- **Min Length Ratio**: Minimum contour length (50% of image height)
- **Aspect Ratio**: Height/width ratio for vertical lines (> 5.0)

#### Horizontal Line Parameters (`H_PARAMS`)
Similar structure but optimized for horizontal lines:
- Larger morphological kernels (width-based)
- Higher Hough thresholds (horizontal lines often more noisy)
- Tighter clustering for precision
- More selective final filtering (80% vs 50%)

## 🎯 Stage 2 Configuration

Stage 2 refines cropped table images with optimized parameters.

### Key Differences from Stage 1:

#### ROI Margins - All Zero
```json
{
  "ROI_MARGINS_PAGE_1": { "top": 0, "bottom": 0, "left": 0, "right": 0 },
  "ROI_MARGINS_PAGE_2": { "top": 0, "bottom": 0, "left": 0, "right": 0 },
  "ROI_MARGINS_DEFAULT": { "top": 0, "bottom": 0, "left": 0, "right": 0 }
}
```
No margins needed since working with pre-cropped table regions.

#### Refined Line Detection Parameters
- Same morphological operations (work well for tables)
- More selective final filtering for horizontal lines (80% vs 70%)
- Enhanced curved line detection with higher requirements

#### Input/Output Flow
- **Stage 2 Input**: `splited_images` points to Stage 1's cropped tables
- **Stage 2 Output**: Organized refinement directories

## 🔧 Parameter Tuning Guide

### Common Adjustments

#### For Poor Line Detection:
1. **Enable Debug Images**: Set `"SAVE_DEBUG_IMAGES": true`
2. **Check ROI Margins**: Ensure table content isn't excluded
3. **Adjust Morphological Kernels**: 
   - Increase ratios for thicker lines
   - Decrease ratios for thinner lines
4. **Tune Hough Parameters**:
   - Lower threshold for more sensitivity
   - Adjust min line length for document type

#### For Noisy Results:
1. **Increase Quality Thresholds**:
   - Higher `qualify_length_ratio`
   - Higher `final_selection_ratio`
2. **Tighter Clustering**: Lower `cluster_distance_threshold`
3. **Stricter Solid Check**: Higher `solid_check_std_threshold`

#### For Curved Lines:
1. **Adjust Contour Parameters**:
   - Lower `contour_min_length_ratio` for shorter curves
   - Adjust `contour_aspect_ratio_threshold` for curve tolerance

### Document-Specific Tuning

#### Academic Papers:
- Use provided Stage 1 defaults
- May need wider ROI margins for headers/footers

#### Historical Documents:
- Increase angle range for deskewing
- Lower Hough thresholds for faded lines
- Adjust solid check for aged paper

#### Technical Manuals:
- May need different page splitting thresholds
- Adjust morphological kernels for line thickness

## 📊 Configuration Testing Workflow

1. **Start with Defaults**: Use provided configurations
2. **Enable Debug Images**: Set `"SAVE_DEBUG_IMAGES": true`
3. **Process Small Sample**: Test on 2-3 representative images
4. **Analyze Debug Output**: Check intermediate processing steps
5. **Adjust Parameters**: Make targeted parameter changes
6. **Iterate**: Repeat until satisfactory results
7. **Scale Up**: Process full document set

## 🐛 Troubleshooting

### Common Issues and Solutions

#### No Lines Detected:
- Check ROI margins aren't excluding table content
- Lower Hough thresholds
- Verify image quality and contrast

#### Too Many False Lines:
- Increase quality ratio thresholds
- Tighter clustering parameters
- Higher solid check threshold

#### Poor Page Splitting:
- Adjust gutter search region
- Modify split threshold
- Check for unusual binding styles

#### Skewed Results After Deskewing:
- Increase angle range
- Decrease minimum correction threshold
- Check for complex page layouts

### Debug Image Analysis

When `SAVE_DEBUG_IMAGES` is enabled, examine these key outputs:

1. **01_binary_masked.jpg** - ROI application and thresholding
2. **02_morph_repaired.jpg** - Morphological line isolation
3. **04_repaired_filtered.jpg** - Solid region removal
4. **05_canny_edges.jpg** - Edge detection input
6. **07_final_clustered_lines.jpg** - Final detected lines

Each stage should show progressive improvement in line isolation and detection.

## 💡 Best Practices

1. **Incremental Tuning**: Change one parameter at a time
2. **Document Sampling**: Test on representative document subset
3. **Debug Verification**: Always check debug images when tuning
4. **Parameter Documentation**: Keep notes on successful parameter sets
5. **Version Control**: Back up working configurations
6. **Batch Testing**: Validate changes across document collection

---

For detailed parameter meanings and ranges, see [PARAMETER_REFERENCE.md](PARAMETER_REFERENCE.md).