# Parameter Tuning Guide

This guide provides detailed information about all configurable parameters in the OCR Table Extraction Pipeline, including their purpose, default values, and tuning recommendations.

## Table of Contents

- [Stage 1 Parameters](#stage-1-parameters)
  - [Mark Removal](#mark-removal)
  - [Margin Removal](#margin-removal)
  - [Page Splitting](#page-splitting)
  - [Tag Removal](#tag-removal)
  - [Deskewing](#deskewing-stage-1)
  - [Table Line Detection](#table-line-detection-stage-1)
  - [Table Detection](#table-detection)
- [Stage 2 Parameters](#stage-2-parameters)
  - [Deskewing](#deskewing-stage-2)
  - [Table Line Detection](#table-line-detection-stage-2)
  - [Table Recovery](#table-recovery)
  - [Vertical Strip Cutting](#vertical-strip-cutting)
  - [Binarization](#binarization)
- [Optimization Parameters](#optimization-parameters)
- [Debug Parameters](#debug-parameters)

---

## Stage 1 Parameters

Stage 1 focuses on initial preprocessing and table region extraction from raw scanned images.

### Mark Removal

Removes watermarks, stamps, and artifacts while preserving text and table lines.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `mark_removal.enable` | `false` | boolean | Enable/disable mark removal |
| `mark_removal.dilate_iter` | `2` | 1-5 | Dilation iterations for text protection |
| `mark_removal.kernel_size` | `1` | 1-5 | Morphological kernel size |
| `mark_removal.protect_table_lines` | `false` | boolean | Preserve detected table lines |
| `mark_removal.table_line_thickness` | `5` | 3-10 | Thickness of protected table lines |

**When to Enable:**
- Documents with watermarks or stamps
- Scans with coffee stains or other artifacts
- Historical documents with age spots

**Tuning Tips:**
- Increase `dilate_iter` if thin text strokes are being removed
- Enable `protect_table_lines` for documents where table lines are faint
- Larger `kernel_size` provides more aggressive cleaning but may remove fine details

### Margin Removal

Detects and removes paper edges, black backgrounds, and scanning artifacts.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `margin_removal.enable` | `true` | boolean | Enable/disable margin removal |
| `margin_removal.use_gradient_detection` | `false` | boolean | Use gradient-based edge detection |
| `margin_removal.gradient_threshold` | `50` | 20-100 | Gradient detection sensitivity |
| `margin_removal.blur_ksize` | `9` | 3-25 | Gaussian blur kernel size |
| `margin_removal.close_ksize` | `20` | 10-50 | Morphological closing kernel size |
| `margin_removal.close_iter` | `3` | 1-5 | Closing iterations |
| `margin_removal.erode_after_close` | `0` | 0-10 | Erosion after closing |

**When to Adjust:**
- Black scanner backgrounds: Increase `close_ksize` and `close_iter`
- Curved book pages: Enable `use_gradient_detection`
- Noisy edges: Increase `blur_ksize`
- Over-cropping: Decrease `erode_after_close`

### Page Splitting

Detects and splits two-page book spreads into individual pages.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `page_splitting.enable` | `true` | boolean | Enable/disable page splitting |
| `page_splitting.search_ratio` | `0.3` | 0.1-0.5 | Width fraction to search for gutter |
| `page_splitting.line_len_frac` | `0.3` | 0.2-0.5 | Vertical line detection kernel height |
| `page_splitting.line_thick` | `3` | 1-5 | Detection kernel width |
| `page_splitting.peak_thr` | `0.3` | 0.1-0.5 | Peak detection threshold |

**When to Adjust:**
- Wide gutters: Increase `search_ratio`
- Faint gutter lines: Decrease `peak_thr`
- Broken gutter detection: Increase `line_len_frac`
- False positives: Increase `peak_thr`

### Tag Removal

Removes headers, footers, page numbers, and other marginal annotations.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `tag_removal.enable` | `true` | boolean | Enable/disable tag removal |
| `tag_removal.thresh_dark` | `140` | 100-180 | Dark pixel threshold |
| `tag_removal.row_sum_thresh` | `150` | 100-250 | Row darkness threshold |
| `tag_removal.dark_ratio` | `0.5` | 0.3-0.7 | Minimum dark pixel ratio |
| `tag_removal.min_area` | `1500` | 500-3000 | Minimum tag area |
| `tag_removal.max_area` | `80000` | 50000-150000 | Maximum tag area |
| `tag_removal.min_aspect` | `0.2` | 0.1-0.5 | Minimum aspect ratio |
| `tag_removal.max_aspect` | `2.5` | 2.0-5.0 | Maximum aspect ratio |

**When to Adjust:**
- Missing small tags: Decrease `min_area`
- False positives on text: Increase `min_area` or adjust `dark_ratio`
- Light colored tags: Increase `thresh_dark`
- Wide headers: Increase `max_aspect`

### Deskewing (Stage 1)

Initial rotation correction for entire page images.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `deskewing.enable` | `true` | boolean | Enable/disable deskewing |
| `deskewing.method` | `"radon"` | string | Method: "radon", "histogram_variance", "deskew_library" |
| `deskewing.use_binarization` | `true` | boolean | Binarize before angle detection |
| `deskewing.coarse_range` | `5` | 2-15 | Coarse search angle range (degrees) |
| `deskewing.coarse_step` | `0.5` | 0.2-1.0 | Coarse search step size |
| `deskewing.fine_range` | `1.0` | 0.5-2.0 | Fine search range around coarse result |
| `deskewing.fine_step` | `0.2` | 0.05-0.5 | Fine search step size |
| `deskewing.min_angle_correction` | `0.1` | 0.05-0.5 | Minimum angle to apply correction |

**Method Selection:**
- `"radon"`: Most accurate, requires scikit-image
- `"histogram_variance"`: Fast, good for text-heavy documents
- `"deskew_library"`: Alternative method, requires deskew package

**When to Adjust:**
- Heavily skewed documents: Increase `coarse_range`
- Precision requirements: Decrease `fine_step`
- Over-correction on straight documents: Increase `min_angle_correction`
- Processing speed: Use larger step sizes or "histogram_variance" method

### Table Line Detection (Stage 1)

Detects horizontal and vertical table lines using morphological operations.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `table_line_detection.threshold` | `50` | 20-100 | Binary threshold for line detection |
| `table_line_detection.horizontal_kernel_size` | `40` | 20-100 | Horizontal morphology kernel width |
| `table_line_detection.vertical_kernel_size` | `30` | 20-100 | Vertical morphology kernel height |
| `table_line_detection.alignment_threshold` | `3` | 1-10 | Line alignment clustering threshold |
| `table_line_detection.h_min_length_image_ratio` | `0.15` | 0.1-0.5 | Min horizontal line length (image ratio) |
| `table_line_detection.h_min_length_relative_ratio` | `0.3` | 0.2-0.7 | Min horizontal line length (relative) |
| `table_line_detection.v_min_length_image_ratio` | `0.15` | 0.1-0.5 | Min vertical line length (image ratio) |
| `table_line_detection.v_min_length_relative_ratio` | `0.3` | 0.2-0.7 | Min vertical line length (relative) |
| `table_line_detection.min_aspect_ratio` | `2` | 2-10 | Minimum aspect ratio for lines |
| `table_line_detection.close_line_distance` | `45` | 20-100 | Distance to merge parallel lines |
| `table_line_detection.skew_tolerance` | `3` | 0-10 | Maximum skew angle tolerance |

**When to Adjust:**
- Faint lines: Decrease `threshold`
- Broken lines: Increase kernel sizes
- False positives from text: Increase `min_aspect_ratio`
- Double lines: Adjust `close_line_distance`
- Skewed tables: Increase `skew_tolerance`

### Table Detection

Identifies and crops table regions from the page.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `table_detection.table_detection_eps` | `10` | 5-30 | Clustering tolerance in pixels |
| `table_detection.table_detection_kernel_ratio` | `0.05` | 0.02-0.1 | Morphology kernel size ratio |
| `table_detection.table_crop_padding` | `25` | 10-50 | Padding around detected tables |
| `table_detection.enable_table_cropping` | `true` | boolean | Enable automatic table cropping |

**When to Adjust:**
- Merged adjacent tables: Decrease `table_detection_eps`
- Missing table borders: Increase `table_crop_padding`
- Over-segmentation: Increase `table_detection_eps`

---

## Stage 2 Parameters

Stage 2 refines the cropped table regions for optimal OCR results.

### Deskewing (Stage 2)

Fine-tuned rotation correction for cropped table images.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `deskewing.enable` | `true` | boolean | Enable/disable fine deskewing |
| `deskewing.method` | `"radon"` | string | Same methods as Stage 1 |
| `deskewing.use_binarization` | `true` | boolean | Binarize before angle detection |
| `deskewing.coarse_range` | `5` | 1-10 | Typically smaller than Stage 1 |
| `deskewing.coarse_step` | `0.5` | 0.1-1.0 | Coarse search step |
| `deskewing.fine_range` | `1.0` | 0.2-2.0 | Fine search range |
| `deskewing.fine_step` | `0.2` | 0.05-0.3 | Fine search step |
| `deskewing.min_angle_correction` | `0.1` | 0.05-0.3 | Minimum correction angle |

**Stage 2 Specific Tips:**
- Use smaller ranges since Stage 1 already corrected major skew
- Can use finer steps for precision on cropped regions
- Consider disabling if Stage 1 deskewing was sufficient

### Table Line Detection (Stage 2)

Refined table line detection on cropped tables.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `table_line_detection.threshold` | `50` | 20-100 | Binary threshold |
| `table_line_detection.horizontal_kernel_size` | `20` | 10-50 | Smaller than Stage 1 |
| `table_line_detection.vertical_kernel_size` | `50` | 20-100 | Adjusted for cropped region |
| `table_line_detection.h_min_length_image_ratio` | `0.2` | 0.15-0.6 | Higher ratio for cropped tables |
| `table_line_detection.h_min_length_relative_ratio` | `0.6` | 0.4-0.8 | Stricter relative filtering |
| `table_line_detection.v_min_length_image_ratio` | `0.4` | 0.3-0.7 | Higher for column detection |
| `table_line_detection.v_min_length_relative_ratio` | `0.5` | 0.4-0.8 | Stricter filtering |
| `table_line_detection.min_aspect_ratio` | `4` | 3-10 | Higher than Stage 1 |
| `table_line_detection.close_line_distance` | `65` | 30-100 | Adjusted for scale |
| `table_line_detection.skew_tolerance` | `2` | 0-5 | Lower after deskewing |

**Stage 2 Specific Tips:**
- Parameters are tuned for already-cropped table regions
- Higher minimum length ratios filter out noise better
- Smaller kernel sizes work better on cropped regions

### Table Recovery

Reconstructs complete table structure from detected lines.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `table_recovery.coverage_ratio` | `0.2` | 0.1-0.5 | Minimum coverage for valid recovery |

**When to Adjust:**
- Incomplete table recovery: Decrease `coverage_ratio`
- Too many false recoveries: Increase `coverage_ratio`

### Vertical Strip Cutting

Extracts individual columns from tables for OCR processing.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `vertical_strip_cutting.enable` | `true` | boolean | Enable/disable column extraction |
| `vertical_strip_cutting.padding_left` | `20` | 5-50 | Left padding for columns |
| `vertical_strip_cutting.padding_right` | `20` | 5-50 | Right padding for columns |
| `vertical_strip_cutting.min_width` | `1` | 1-50 | Minimum column width |
| `vertical_strip_cutting.use_longest_lines_only` | `false` | boolean | Use only longest vertical lines |
| `vertical_strip_cutting.min_length_ratio` | `0.9` | 0.7-1.0 | Minimum line length ratio |

**When to Adjust:**
- Overlapping columns: Decrease `padding_left` or `padding_right`
- Text cut off on left: Increase `padding_left`
- Text cut off on right: Increase `padding_right`
- Missing narrow columns: Decrease `min_width`
- Incomplete column extraction: Decrease `min_length_ratio`
- Too many false columns: Enable `use_longest_lines_only`

### Binarization

Converts images to black and white for OCR engines.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `binarization.enable` | `false` | boolean | Enable/disable binarization |
| `binarization.method` | `"otsu"` | string | Method: "otsu", "adaptive", "fixed" |
| `binarization.threshold` | `127` | 0-255 | Fixed threshold value |
| `binarization.adaptive_block_size` | `11` | 3-99 | Adaptive method block size (odd) |
| `binarization.adaptive_c` | `2` | -10-10 | Adaptive method constant |
| `binarization.invert` | `false` | boolean | Invert black/white |
| `binarization.denoise` | `false` | boolean | Apply denoising |

**Method Selection:**
- `"otsu"`: Automatic global threshold (default)
- `"adaptive"`: Local thresholding for uneven lighting
- `"fixed"`: Manual threshold control

**Stroke Enhancement:**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `stroke_enhancement.enable` | `false` | boolean | Enable stroke enhancement |
| `stroke_enhancement.kernel_size` | `2` | 1-5 | Enhancement kernel size |
| `stroke_enhancement.iterations` | `1` | 1-3 | Enhancement iterations |
| `stroke_enhancement.kernel_shape` | `"ellipse"` | string | Kernel shape: "ellipse", "rect", "cross" |

---

## Optimization Parameters

Control processing performance and resource usage.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `optimization.parallel_processing` | `true` | boolean | Enable parallel processing |
| `optimization.max_workers` | `null` | 1-32 | Worker processes (null = CPU count - 1) |
| `optimization.batch_size` | `null` | 1-100 | Images per batch |
| `optimization.memory_mode` | `false` | boolean | Keep intermediates in memory |

**Performance Tuning:**
- **Large datasets**: Enable `parallel_processing`, increase `max_workers`
- **Limited RAM**: Disable `memory_mode`, reduce `batch_size`
- **Fast processing**: Enable `memory_mode`, increase `batch_size`
- **Debug mode**: Automatically disables optimization for clean output

---

## Debug Parameters

Control debug output and visualization.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `save_debug_images` | `false` | boolean | Save debug visualizations |
| `debug_dir` | `"data/debug"` | string | Debug output directory |
| `verbose` | `false` | boolean | Enable verbose console output |

**Debug Mode Notes:**
- Automatically disables parallel processing for sequential debug output
- Creates timestamped subdirectories for each run
- Saves intermediate images at each processing step
- Includes visualization overlays and analysis data

---

## Common Tuning Scenarios

### Historical Documents
```json
{
  "mark_removal": {"enable": true, "dilate_iter": 3},
  "margin_removal": {"blur_ksize": 15, "close_iter": 4},
  "deskewing": {"coarse_range": 10},
  "table_line_detection": {"threshold": 30}
}
```

### Modern Scans
```json
{
  "mark_removal": {"enable": false},
  "margin_removal": {"enable": true, "blur_ksize": 5},
  "deskewing": {"coarse_range": 3},
  "table_line_detection": {"threshold": 50}
}
```

### Book Scans
```json
{
  "page_splitting": {"enable": true, "search_ratio": 0.4},
  "margin_removal": {"use_gradient_detection": true},
  "tag_removal": {"enable": true},
  "deskewing": {"method": "radon", "coarse_range": 7}
}
```

### High-Quality Processing
```json
{
  "deskewing": {"fine_step": 0.1, "min_angle_correction": 0.05},
  "table_line_detection": {"alignment_threshold": 2},
  "optimization": {"memory_mode": true, "parallel_processing": true}
}
```

### Speed Optimization
```json
{
  "deskewing": {"method": "histogram_variance", "coarse_step": 1.0},
  "optimization": {"parallel_processing": true, "max_workers": 16},
  "save_debug_images": false
}
```

---

## Tips for Parameter Tuning

1. **Start with defaults**: The default parameters work well for most documents
2. **Enable debug mode**: Use `save_debug_images` to visualize processing steps
3. **Tune incrementally**: Adjust one parameter at a time
4. **Test on samples**: Use a small subset before processing large batches
5. **Document-specific configs**: Create custom configs for different document types
6. **Monitor performance**: Use verbose mode to identify bottlenecks
7. **Balance quality/speed**: Trade-off between processing time and accuracy

## Troubleshooting

### Common Issues and Solutions

**Over-cropping:**
- Increase `table_crop_padding`
- Decrease `margin_removal.erode_after_close`

**Missing table lines:**
- Decrease `table_line_detection.threshold`
- Increase morphological kernel sizes
- Enable `mark_removal.protect_table_lines`

**False page splits:**
- Increase `page_splitting.peak_thr`
- Adjust `page_splitting.search_ratio`

**Poor deskewing:**
- Try different deskewing methods
- Increase search ranges
- Enable `use_binarization`

**Slow processing:**
- Enable `parallel_processing`
- Increase `max_workers`
- Use `memory_mode` if RAM permits
- Increase step sizes in deskewing