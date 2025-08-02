# OCR Pipeline Parameter Reference

Complete reference guide for all configurable parameters in the simplified OCR table extraction pipeline.

## Table of Contents

1. [Directory Configuration](#directory-configuration)
2. [Page Splitting Parameters](#page-splitting-parameters)
3. [Deskewing Parameters](#deskewing-parameters)
4. [Margin Removal Parameters](#margin-removal-parameters)
5. [Line Detection Parameters](#line-detection-parameters)
6. [Debug and Output Parameters](#debug-and-output-parameters)
7. [Configuration Examples](#configuration-examples)

---

## Directory Configuration

### `input_dir`
- **Type**: `Path`
- **Default**: `"data/input"`
- **Description**: Directory containing input images to process
- **Usage**: Raw scanned document images for processing

### `output_dir`
- **Type**: `Path`
- **Default**: `"data/output"`
- **Description**: Base directory for processed output files
- **Notes**: Automatically creates subdirectories for each processing stage

### `debug_dir`
- **Type**: `Optional[Path]`
- **Default**: `None`
- **Description**: Optional directory for debug-specific output files
- **Usage**: Set when `save_debug_images=True` for organized debug output

---

## Page Splitting Parameters

Page splitting separates double-page scanned documents into individual pages by detecting the gutter (binding area).

### `gutter_search_start`
- **Type**: `float`
- **Default**: `0.4`
- **Range**: `0.0 - 1.0` (fraction of image width)
- **Description**: Start position for searching the page gutter from the left edge
- **Examples**:
  - `0.35`: Start searching at 35% of image width (wider search area)
  - `0.4`: Start searching at 40% of image width (default)
  - `0.45`: Start searching at 45% of image width (narrower search area)
- **Tuning**: Decrease for documents with off-center binding, increase for centered binding

### `gutter_search_end`
- **Type**: `float`
- **Default**: `0.6`
- **Range**: `0.0 - 1.0` (fraction of image width)
- **Description**: End position for searching the page gutter from the left edge
- **Examples**:
  - `0.55`: Stop searching at 55% of image width (narrower search area)
  - `0.6`: Stop searching at 60% of image width (default)
  - `0.65`: Stop searching at 65% of image width (wider search area)
- **Tuning**: Must be greater than `gutter_search_start`

### `min_gutter_width`
- **Type**: `int`
- **Default**: `50`
- **Range**: `20 - 200` pixels
- **Description**: Minimum width required for a valid gutter detection
- **Examples**:
  - `30`: Very narrow binding (magazines, thin books)
  - `50`: Standard book binding (default)
  - `100`: Thick book binding (textbooks, manuals)
- **Tuning**: Increase for thicker books, decrease for thin publications

---

## Deskewing Parameters

Deskewing corrects rotational skew in scanned images to make text horizontal and table lines straight.

### `angle_range`
- **Type**: `int`
- **Default**: `5`
- **Range**: `1 - 45` degrees
- **Description**: Maximum rotation angle to detect and correct (±degrees)
- **Examples**:
  - `3`: Conservative range for mostly straight documents
  - `5`: Standard range for typical scanning variations (default)
  - `10`: Wide range for significantly skewed documents
- **Tuning**: Increase for documents with significant skew, decrease for stability

### `angle_step`
- **Type**: `float`
- **Default**: `0.1`
- **Range**: `0.05 - 2.0` degrees
- **Description**: Precision of angle detection (smaller = more precise, slower)
- **Examples**:
  - `0.05`: High precision detection (slower)
  - `0.1`: Standard precision (default)
  - `0.5`: Fast detection (less precise)
- **Tuning**: Decrease for higher accuracy, increase for faster processing

### `min_angle_correction`
- **Type**: `float`
- **Default**: `0.1`
- **Range**: `0.05 - 5.0` degrees
- **Description**: Minimum detected angle before applying rotation correction
- **Examples**:
  - `0.05`: Very sensitive (corrects tiny rotations)
  - `0.1`: Standard sensitivity (default)
  - `0.5`: Conservative (only corrects obvious skew)
- **Tuning**: Increase to avoid over-correction of straight images

---

## Margin Removal Parameters

Margin removal identifies and removes document margins and background noise, replacing the previous ROI detection system.

### `enable_margin_removal`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Enable/disable margin removal preprocessing
- **Usage**: Set to `False` to skip margin cleaning step

### `blur_kernel_size`
- **Type**: `int`
- **Default**: `7`
- **Range**: `3, 5, 7, 9, 11` (odd numbers)
- **Description**: Gaussian blur kernel size for noise reduction before margin detection
- **Examples**:
  - `3`: Minimal blur (preserves fine details)
  - `7`: Standard blur (default)
  - `11`: Heavy blur (removes more noise)
- **Tuning**: Increase for noisy images, decrease to preserve detail

### `black_threshold`
- **Type**: `int`
- **Default**: `50`
- **Range**: `10 - 100`
- **Description**: Threshold for identifying black pixels (margins/background)
- **Examples**:
  - `30`: More sensitive to dark areas
  - `50`: Standard sensitivity (default)
  - `80`: Less sensitive, only very dark areas
- **Tuning**: Decrease for lighter margins, increase for cleaner backgrounds

### `content_threshold`
- **Type**: `int`
- **Default**: `200`
- **Range**: `100 - 255`
- **Description**: Threshold for identifying content pixels
- **Examples**:
  - `150`: More sensitive to light content
  - `200`: Standard content detection (default)
  - `230`: Only detects strong content
- **Tuning**: Decrease for faint content, increase for high-contrast documents

### `morph_kernel_size`
- **Type**: `int`
- **Default**: `25`
- **Range**: `5 - 50`
- **Description**: Morphological operation kernel size for content area detection
- **Examples**:
  - `15`: Smaller kernel (more precise boundaries)
  - `25`: Standard kernel (default)
  - `35`: Larger kernel (smoother boundaries)
- **Tuning**: Increase for fragmented content, decrease for precise boundaries

### `min_content_area_ratio`
- **Type**: `float`
- **Default**: `0.01`
- **Range**: `0.005 - 0.1`
- **Description**: Minimum content area ratio to preserve (relative to image size)
- **Examples**:
  - `0.005`: Very small content areas preserved
  - `0.01`: Standard threshold (default)
  - `0.05`: Only larger content areas preserved
- **Tuning**: Decrease to preserve small content, increase to filter noise

### `margin_padding`
- **Type**: `int`
- **Default**: `5`
- **Range**: `0 - 20` pixels
- **Description**: Additional padding around detected content areas
- **Examples**:
  - `0`: No padding (tight cropping)
  - `5`: Standard padding (default)
  - `15`: Generous padding
- **Tuning**: Increase to preserve content near margins, decrease for tight cropping

---

## Line Detection Parameters

Line detection uses a connected components method to identify table structure by finding horizontal and vertical lines.

### `threshold`
- **Type**: `int`
- **Default**: `40`
- **Range**: `10 - 100`
- **Description**: Binary threshold for line detection preprocessing
- **Examples**:
  - `25`: More sensitive (detects fainter lines)
  - `40`: Standard sensitivity (default)
  - `60`: Less sensitive (only strong lines)
- **Tuning**: Decrease for faint table lines, increase to reduce noise

### `horizontal_kernel_size`
- **Type**: `int`
- **Default**: `10`
- **Range**: `5 - 30`
- **Description**: Morphological kernel size for horizontal line detection
- **Examples**:
  - `8`: Smaller kernel (shorter line segments)
  - `10`: Standard kernel (default)
  - `20`: Larger kernel (longer line segments)
- **Tuning**: Increase for tables with wide spacing, decrease for dense tables

### `vertical_kernel_size`
- **Type**: `int`
- **Default**: `10`
- **Range**: `5 - 30`
- **Description**: Morphological kernel size for vertical line detection
- **Examples**:
  - `8`: Smaller kernel (shorter line segments)
  - `10`: Standard kernel (default)
  - `20`: Larger kernel (longer line segments)
- **Tuning**: Increase for tables with tall cells, decrease for dense tables

### `alignment_threshold`
- **Type**: `int`
- **Default**: `3`
- **Range**: `1 - 10` pixels
- **Description**: Maximum pixel deviation for line alignment
- **Examples**:
  - `1`: Very strict alignment
  - `3`: Standard tolerance (default)
  - `8`: Permissive alignment
- **Tuning**: Increase for hand-drawn or imperfect tables, decrease for precise tables

### `pre_merge_length_ratio`
- **Type**: `float`
- **Default**: `0.3`
- **Range**: `0.1 - 0.8`
- **Description**: Minimum length ratio for line segments before merging
- **Examples**:
  - `0.2`: More permissive (includes shorter segments)
  - `0.3`: Standard threshold (default)
  - `0.5`: More restrictive (only longer segments)
- **Tuning**: Decrease to include more line fragments, increase for quality control

### `post_merge_length_ratio`
- **Type**: `float`
- **Default**: `0.4`
- **Range**: `0.2 - 0.9`
- **Description**: Minimum length ratio for final lines after merging
- **Examples**:
  - `0.3`: More permissive final filter
  - `0.4`: Standard threshold (default)
  - `0.6`: More restrictive final filter
- **Tuning**: Decrease to keep more lines, increase for cleaner results

### `min_aspect_ratio`
- **Type**: `int`
- **Default**: `5`
- **Range**: `2 - 20`
- **Description**: Minimum aspect ratio for valid line detection (length/width)
- **Examples**:
  - `3`: More permissive (allows shorter lines)
  - `5`: Standard ratio (default)
  - `10`: More restrictive (only long thin lines)
- **Tuning**: Decrease for tables with short borders, increase to filter noise

---

## Debug and Output Parameters

### `save_debug_images`
- **Type**: `bool`
- **Default**: `False`
- **Description**: Save intermediate processing images for debugging and analysis
- **Output**: Creates detailed step-by-step visual outputs in organized directories
- **Warning**: Generates significant disk usage - use selectively

### `verbose`
- **Type**: `bool`
- **Default**: `False`
- **Description**: Enable detailed console output during processing
- **Output**: Progress information, parameter values, and processing statistics
- **Recommendation**: Enable for troubleshooting and analysis

---

## Configuration Examples

### Minimal Configuration
```json
{
  "verbose": true,
  "gutter_search_start": 0.4,
  "gutter_search_end": 0.6,
  "threshold": 40
}
```

### High Sensitivity Configuration
```json
{
  "gutter_search_start": 0.35,
  "gutter_search_end": 0.65,
  "angle_range": 10,
  "min_angle_correction": 0.05,
  "black_threshold": 30,
  "content_threshold": 150,
  "threshold": 25,
  "horizontal_kernel_size": 15,
  "vertical_kernel_size": 15,
  "pre_merge_length_ratio": 0.2,
  "post_merge_length_ratio": 0.3,
  "min_aspect_ratio": 3
}
```

### Conservative Configuration
```json
{
  "gutter_search_start": 0.42,
  "gutter_search_end": 0.58,
  "angle_range": 3,
  "min_angle_correction": 0.5,
  "black_threshold": 60,
  "content_threshold": 220,
  "threshold": 60,
  "horizontal_kernel_size": 8,
  "vertical_kernel_size": 8,
  "pre_merge_length_ratio": 0.4,
  "post_merge_length_ratio": 0.5,
  "min_aspect_ratio": 8
}
```

### Debug Configuration
```json
{
  "verbose": true,
  "save_debug_images": true,
  "debug_dir": "data/debug",
  "gutter_search_start": 0.4,
  "gutter_search_end": 0.6
}
```

## Best Practices

### General Tuning Approach
1. **Start Conservative**: Begin with default or higher threshold values
2. **Single Parameter**: Change one parameter at a time during testing
3. **Visual Verification**: Always inspect results using visualization tools
4. **Representative Testing**: Use diverse test images
5. **Document Changes**: Keep notes on what works for different document types

### Parameter Interdependencies
- **Page Splitting + Deskewing**: Poor page splitting can affect deskewing accuracy
- **Margin Removal + Line Detection**: Aggressive margin removal may remove table lines
- **Kernel Sizes**: Larger kernels create thicker lines but may merge separate elements
- **Thresholds**: Lower thresholds detect more but increase noise

### Performance Considerations
- **Debug Images**: Significant storage overhead - use sparingly
- **Kernel Sizes**: Larger kernels = slower processing but more robust detection
- **Angle Step**: Smaller steps = more accurate but slower deskewing

### Document-Specific Tuning
- **Clean Scans**: Use conservative parameters to avoid over-processing
- **Poor Quality**: Increase noise tolerance (larger kernels, lower thresholds)
- **Varied Content**: Use moderate parameters with good generalization
- **Consistent Layout**: Can use more specialized parameters

This parameter reference should be used in conjunction with the visualization tools in the `tools/` directory to systematically optimize the pipeline for your specific document types.