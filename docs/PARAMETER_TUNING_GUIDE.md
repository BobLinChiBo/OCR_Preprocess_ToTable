# Parameter Tuning Guide

This guide provides detailed information about all configurable parameters in the OCR Table Extraction Pipeline, including their purpose, default values, and tuning recommendations for the consolidated single-config approach.

## Table of Contents

- [Overview](#overview)
- [Stage 1 Parameters](#stage-1-parameters)
  - [Preprocessing Options](#preprocessing-options)
  - [Table Line Detection with Preprocessing](#table-line-detection-with-preprocessing)
  - [Table Structure Detection](#table-structure-detection)
- [Stage 2 Parameters](#stage-2-parameters)
  - [Refined Processing](#refined-processing)
  - [Structure Recovery](#structure-recovery)
  - [Column Extraction](#column-extraction)
- [Optimization Parameters](#optimization-parameters)
- [Debug Parameters](#debug-parameters)
- [Common Tuning Scenarios](#common-tuning-scenarios)

---

## Overview

The pipeline now uses a **single configuration file per stage** with all preprocessing parameters included. Both stages support advanced preprocessing for robust table line detection.

**Key Changes:**
- Single config approach - no separate preprocessing files
- All `line_detection_*` parameters included in main config  
- Consistent parameter naming across stages
- Easy to toggle preprocessing features on/off

---

## Stage 1 Parameters

**Purpose**: Crop out all irrelevant content and focus on the main table region

### Preprocessing Options

#### Mark Removal
Removes watermarks, stamps, and artifacts while preserving text and table lines.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `mark_removal.enable` | `false` | boolean | Enable/disable mark removal |
| `mark_removal.dilate_iter` | `2` | 1-5 | Dilation iterations for text protection |
| `mark_removal.kernel_size` | `1` | 1-5 | Morphological kernel size |
| `mark_removal.protect_table_lines` | `false` | boolean | Preserve detected table lines |
| `mark_removal.table_line_thickness` | `5` | 3-10 | Thickness of protected table lines |

#### Margin Removal
Detects and removes paper edges, black backgrounds, and scanning artifacts.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `margin_removal.enable` | `false` | boolean | Enable/disable margin removal |
| `margin_removal.use_gradient_detection` | `true` | boolean | Use gradient-based edge detection |
| `margin_removal.gradient_threshold` | `50` | 20-100 | Gradient detection sensitivity |
| `margin_removal.blur_ksize` | `9` | 3-25 | Gaussian blur kernel size |
| `margin_removal.close_ksize` | `20` | 10-50 | Morphological closing kernel size |
| `margin_removal.close_iter` | `3` | 1-5 | Closing iterations |

#### Page Splitting
Detects and splits two-page book spreads into individual pages.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `page_splitting.enable` | `false` | boolean | Enable/disable page splitting |
| `page_splitting.search_ratio` | `0.3` | 0.1-0.5 | Width fraction to search for gutter |
| `page_splitting.line_len_frac` | `0.3` | 0.2-0.5 | Vertical line detection kernel height |
| `page_splitting.line_thick` | `3` | 1-5 | Detection kernel width |
| `page_splitting.peak_thr` | `0.3` | 0.1-0.5 | Peak detection threshold |

#### Tag Removal
Removes headers, footers, page numbers, and other marginal annotations.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `tag_removal.enable` | `false` | boolean | Enable/disable tag removal |
| `tag_removal.method` | `"auto_whitefill_broadmask"` | string | Tag removal method |
| `tag_removal.thresh_dark` | `110` | 100-180 | Dark pixel threshold |
| `tag_removal.min_dark` | `0.52` | 0.3-0.7 | Minimum dark pixel ratio |
| `tag_removal.min_score` | `0.55` | 0.3-0.8 | Minimum detection score |

#### Deskewing
Initial rotation correction for entire page images.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `deskewing.enable` | `true` | boolean | Enable/disable deskewing |
| `deskewing.method` | `"radon"` | string | Method: "radon", "histogram_variance" |
| `deskewing.use_binarization` | `true` | boolean | Binarize before angle detection |
| `deskewing.coarse_range` | `5` | 2-15 | Coarse search angle range (degrees) |
| `deskewing.coarse_step` | `0.5` | 0.2-1.0 | Coarse search step size |
| `deskewing.fine_range` | `1.0` | 0.5-2.0 | Fine search range around coarse result |
| `deskewing.fine_step` | `0.2` | 0.05-0.5 | Fine search step size |
| `deskewing.min_angle_correction` | `0.1` | 0.05-0.5 | Minimum angle to apply correction |

### Table Line Detection with Preprocessing

Advanced table line detection with built-in preprocessing for robust line extraction.

#### Core Detection Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `table_line_detection.threshold` | `20` | 10-100 | Binary threshold for line detection |
| `table_line_detection.horizontal_kernel_size` | `80` | 20-200 | Horizontal morphology kernel width |
| `table_line_detection.vertical_kernel_size` | `40` | 20-200 | Vertical morphology kernel height |
| `table_line_detection.alignment_threshold` | `3` | 1-10 | Line alignment clustering threshold |
| `table_line_detection.min_aspect_ratio` | `2` | 2-10 | Minimum aspect ratio for lines |

#### Length Filtering

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `table_line_detection.h_min_length_image_ratio` | `0.10` | 0.05-0.5 | Min horizontal line length (image ratio) |
| `table_line_detection.h_min_length_relative_ratio` | `0.3` | 0.2-0.7 | Min horizontal line length (relative) |
| `table_line_detection.v_min_length_image_ratio` | `0.2` | 0.1-0.5 | Min vertical line length (image ratio) |
| `table_line_detection.v_min_length_relative_ratio` | `0.3` | 0.2-0.7 | Min vertical line length (relative) |

#### Post-processing

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `table_line_detection.max_h_length_ratio` | `1.0` | 0.5-1.0 | Max horizontal line length ratio (1.0 = disable) |
| `table_line_detection.max_v_length_ratio` | `1.0` | 0.5-1.0 | Max vertical line length ratio (1.0 = disable) |
| `table_line_detection.close_line_distance` | `45` | 20-100 | Distance to merge parallel lines |

#### Search Regions

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `table_line_detection.search_region_top` | `30` | 0-200 | Pixels to ignore from top |
| `table_line_detection.search_region_bottom` | `30` | 0-200 | Pixels to ignore from bottom |
| `table_line_detection.search_region_left` | `5` | 0-100 | Pixels to ignore from left |
| `table_line_detection.search_region_right` | `5` | 0-100 | Pixels to ignore from right |

#### Skew Tolerance

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `table_line_detection.skew_tolerance` | `2` | 0-10 | Maximum skew angle tolerance (degrees) |
| `table_line_detection.skew_angle_step` | `0.2` | 0.1-0.5 | Step size for angle search |

#### **Preprocessing Parameters** (NEW)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `table_line_detection.line_detection_use_preprocessing` | `true` | boolean | **Enable preprocessing for better detection** |
| `table_line_detection.line_detection_binarization_method` | `"adaptive"` | string | **Binarization method: "adaptive", "otsu", "fixed"** |
| `table_line_detection.line_detection_binarization_threshold` | `127` | 0-255 | **Fixed threshold value** |
| `table_line_detection.line_detection_adaptive_block_size` | `21` | 3-99 | **Adaptive method block size (odd numbers)** |
| `table_line_detection.line_detection_adaptive_c` | `7` | -10-10 | **Adaptive method constant** |
| `table_line_detection.line_detection_binarization_invert` | `false` | boolean | **Invert black/white** |
| `table_line_detection.line_detection_binarization_denoise` | `true` | boolean | **Apply denoising** |
| `table_line_detection.line_detection_stroke_enhancement` | `true` | boolean | **Enable stroke enhancement** |
| `table_line_detection.line_detection_stroke_kernel_size` | `2` | 1-5 | **Enhancement kernel size** |
| `table_line_detection.line_detection_stroke_iterations` | `2` | 1-5 | **Enhancement iterations** |
| `table_line_detection.line_detection_stroke_kernel_shape` | `"cross"` | string | **Kernel shape: "ellipse", "rect", "cross"** |

**Preprocessing Tuning Tips:**
- **Poor quality/faded lines**: Enable stroke enhancement, adjust adaptive parameters
- **High quality images**: Disable preprocessing or use minimal settings  
- **Noisy images**: Enable denoising, increase adaptive block size
- **Varying lighting**: Use adaptive binarization with tuned parameters
- **Thick lines**: Decrease stroke kernel size and iterations
- **Thin lines**: Increase stroke enhancement parameters

### Table Structure Detection

Identifies and crops table regions from the page.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `table_detection.table_detection_eps` | `10` | 5-30 | Clustering tolerance in pixels |
| `table_detection.table_detection_kernel_ratio` | `0.05` | 0.02-0.1 | Morphology kernel size ratio |
| `table_detection.table_crop_padding` | `25` | 10-50 | Padding around detected tables |
| `table_detection.enable_table_cropping` | `true` | boolean | Enable automatic table cropping |

---

## Stage 2 Parameters  

**Purpose**: Recover the actual table structure and cut by vertical lines for OCR

### Refined Processing

#### Deskewing (Fine-tuning)
Fine-tuned rotation correction for cropped table images.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `deskewing.enable` | `true` | boolean | Enable/disable fine deskewing |
| `deskewing.method` | `"radon"` | string | Same methods as Stage 1 |
| `deskewing.coarse_range` | `5` | 1-10 | Typically smaller than Stage 1 |
| `deskewing.fine_step` | `0.2` | 0.05-0.3 | Fine search step |
| `deskewing.min_angle_correction` | `0.1` | 0.05-0.3 | Minimum correction angle |

#### Table Line Detection (Refined)
Enhanced table line detection with preprocessing optimized for cropped tables.

**Core Parameters:**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `table_line_detection.threshold` | `30` | 10-100 | Binary threshold |
| `table_line_detection.horizontal_kernel_size` | `50` | 10-100 | Smaller than Stage 1 |
| `table_line_detection.vertical_kernel_size` | `40` | 20-100 | Adjusted for cropped region |

**Preprocessing (optimized for cropped tables):**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `table_line_detection.line_detection_use_preprocessing` | `true` | boolean | **Enhanced preprocessing enabled** |
| `table_line_detection.line_detection_binarization_threshold` | `180` | 100-255 | **Higher threshold for refined detection** |
| `table_line_detection.line_detection_adaptive_block_size` | `11` | 3-21 | **Smaller block size for precision** |
| `table_line_detection.line_detection_adaptive_c` | `5` | 2-10 | **Adaptive constant** |
| `table_line_detection.line_detection_stroke_enhancement` | `true` | boolean | **Enabled by default** |

**Length Filtering (stricter):**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `table_line_detection.h_min_length_relative_ratio` | `0.5` | 0.4-0.8 | **Stricter relative filtering** |
| `table_line_detection.v_min_length_image_ratio` | `0.15` | 0.1-0.4 | **Higher for column detection** |
| `table_line_detection.v_min_length_relative_ratio` | `0.5` | 0.4-0.8 | **Stricter filtering** |

**Search Regions (none - working on cropped tables):**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `table_line_detection.search_region_*` | `0` | 0-50 | **No search regions needed** |
| `table_line_detection.skew_tolerance` | `2` | 0-5 | **Lower after Stage 1 deskewing** |

### Structure Recovery

#### Table Recovery
Reconstructs complete table structure from detected lines.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `table_recovery.coverage_ratio` | `0.5` | 0.1-0.8 | Minimum coverage for valid recovery |

#### Inscribed Margin Removal
Removes internal table margins and artifacts.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `inscribed_margin_removal.inscribed_blur_ksize` | `3` | 3-15 | Blur kernel size |
| `inscribed_margin_removal.inscribed_close_ksize` | `15` | 5-30 | Closing kernel size |
| `inscribed_margin_removal.inscribed_close_iter` | `1` | 1-3 | Closing iterations |

### Column Extraction

#### Vertical Strip Cutting
Extracts individual columns from tables for OCR processing.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `vertical_strip_cutting.enable` | `true` | boolean | Enable/disable column extraction |
| `vertical_strip_cutting.padding_left` | `5` | 0-50 | Left padding for columns |
| `vertical_strip_cutting.padding_right` | `5` | 0-50 | Right padding for columns |
| `vertical_strip_cutting.min_width` | `1` | 1-50 | Minimum column width |
| `vertical_strip_cutting.use_longest_lines_only` | `true` | boolean | Use only longest vertical lines |
| `vertical_strip_cutting.min_length_ratio` | `0.9` | 0.7-1.0 | Minimum line length ratio |

#### Final Binarization (Optional)
Converts images to black and white for OCR engines.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `binarization.enable` | `false` | boolean | Enable/disable final binarization |
| `binarization.method` | `"otsu"` | string | Method: "otsu", "adaptive", "fixed" |
| `binarization.adaptive_block_size` | `11` | 3-99 | Adaptive method block size |
| `binarization.adaptive_c` | `2` | -10-10 | Adaptive method constant |

#### Final Deskewing (Optional)
Last-stage rotation correction.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `final_deskew.enable` | `true` | boolean | Enable/disable final deskewing |
| `final_deskew.coarse_range` | `2` | 1-5 | Small correction range |
| `final_deskew.fine_step` | `0.2` | 0.05-0.5 | Fine step size |

---

## Optimization Parameters

Control processing performance and resource usage.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `optimization.parallel_processing` | `true` | boolean | Enable parallel processing |
| `optimization.max_workers` | `6` | 1-32 | Worker processes |
| `optimization.batch_size` | `null` | 1-100 | Images per batch |
| `optimization.memory_mode` | `false` | boolean | Keep intermediates in memory |

**Performance Tuning:**
- **Large datasets**: Enable `parallel_processing`, increase `max_workers`
- **Limited RAM**: Keep `memory_mode: false`, reduce `batch_size`
- **Fast processing**: Enable `memory_mode: true`, increase `batch_size`

---

## Debug Parameters

Control debug output and visualization.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `save_debug_images` | `false` | boolean | Save debug visualizations |
| `debug_dir` | `"data/debug"` | string | Debug output directory |
| `verbose` | `true` | boolean | Enable verbose console output |

**Debug Mode Features:**
- Shows preprocessing steps (binarization, stroke enhancement, denoising)
- Visualizes line detection (morphological operations, connected components)
- Displays table structure reconstruction
- Saves before/after comparisons for each processing step

---

## Common Tuning Scenarios

### Poor Quality/Faded Documents
```json
{
  "table_line_detection": {
    "line_detection_use_preprocessing": true,
    "line_detection_stroke_enhancement": true,
    "line_detection_stroke_kernel_size": 3,
    "line_detection_stroke_iterations": 3,
    "line_detection_binarization_denoise": true,
    "line_detection_adaptive_block_size": 25,
    "line_detection_adaptive_c": 10,
    "threshold": 15
  }
}
```

### High Quality Modern Scans
```json
{
  "table_line_detection": {
    "line_detection_use_preprocessing": false,
    "threshold": 40
  },
  "margin_removal": {"enable": false},
  "mark_removal": {"enable": false}
}
```

### Historical Documents with Artifacts
```json
{
  "mark_removal": {"enable": true, "dilate_iter": 3},
  "table_line_detection": {
    "line_detection_use_preprocessing": true,
    "line_detection_stroke_enhancement": true,
    "line_detection_binarization_denoise": true,
    "threshold": 20
  }
}
```

### Skewed Tables
```json
{
  "deskewing": {"coarse_range": 10, "fine_step": 0.1},
  "table_line_detection": {
    "skew_tolerance": 5,
    "skew_angle_step": 0.1
  }
}
```

### Speed Optimization
```json
{
  "table_line_detection": {
    "line_detection_use_preprocessing": false
  },
  "deskewing": {"coarse_step": 1.0, "fine_step": 0.5},
  "optimization": {
    "parallel_processing": true,
    "max_workers": 16,
    "memory_mode": true
  }
}
```

### Maximum Quality Processing
```json
{
  "table_line_detection": {
    "line_detection_use_preprocessing": true,
    "line_detection_stroke_enhancement": true,
    "line_detection_binarization_method": "adaptive",
    "line_detection_adaptive_block_size": 19,
    "line_detection_adaptive_c": 8,
    "skew_tolerance": 3,
    "skew_angle_step": 0.1,
    "alignment_threshold": 2
  },
  "deskewing": {"fine_step": 0.05},
  "optimization": {"memory_mode": true}
}
```

---

## Tips for Parameter Tuning

### Single Config Approach
1. **One config per stage**: All parameters in `stage1_default.json` and `stage2_default.json`
2. **Toggle preprocessing**: Set `line_detection_use_preprocessing: false` to disable all preprocessing
3. **Incremental tuning**: Adjust preprocessing parameters based on image quality
4. **Consistent naming**: All preprocessing parameters start with `line_detection_*`

### Preprocessing Strategy
1. **Start with defaults**: Preprocessing enabled with adaptive binarization and stroke enhancement
2. **Quality assessment**: 
   - High quality → Disable preprocessing
   - Poor quality → Tune adaptive parameters  
   - Faded lines → Increase stroke enhancement
   - Noisy images → Enable denoising
3. **Debug visualization**: Use `save_debug_images: true` to see preprocessing effects

### Performance vs Quality
1. **Speed priority**: Disable preprocessing, use larger step sizes
2. **Quality priority**: Enable all preprocessing features, use fine step sizes
3. **Balanced approach**: Use defaults with selective preprocessing tuning

### Troubleshooting
- **Missing lines**: Lower threshold, enable stroke enhancement
- **False detections**: Increase threshold, disable preprocessing  
- **Poor structure**: Tune length filtering ratios
- **Slow processing**: Disable preprocessing, increase parallel workers

## Advanced Configuration

### Custom Preprocessing Profiles
Create document-type specific preprocessing settings:

**Books/Manuscripts:**
```json
"line_detection_stroke_enhancement": true,
"line_detection_binarization_denoise": true,
"line_detection_adaptive_block_size": 21
```

**Technical Documents:**
```json
"line_detection_use_preprocessing": false,
"search_region_top": 50,
"search_region_bottom": 50
```

**Mixed Quality Scans:**
```json
"line_detection_binarization_method": "adaptive",
"line_detection_adaptive_c": 8,
"skew_tolerance": 3
```