# OCR Pipeline Parameter Reference

Complete reference guide for all configurable parameters in the OCR Table Extraction Pipeline.

> **📚 Documentation Navigation**: [← Documentation Index](README.md) | [Configuration Guide](../configs/README.md) | [Debug Mode Guide](DEBUG_MODE_GUIDE.md) →

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

Page splitting separates double-page scanned documents into individual pages by detecting the gutter (binding area). The V2 algorithm uses vertical line detection for more robust gutter detection.

### `search_ratio`
- **Type**: `float`
- **Default**: `0.5`
- **Range**: `0.0 - 1.0` (fraction of width to search, centered)
- **Description**: Fraction of image width (centered) to search for the gutter
- **Examples**:
  - `0.3`: Search central 30% of image (narrow search)
  - `0.5`: Search central 50% of image (default)
  - `0.8`: Search central 80% of image (wide search)
- **Tuning**: Increase for off-center bindings, decrease for well-centered documents

### `line_len_frac`
- **Type**: `float`
- **Default**: `0.3`
- **Range**: `0.1 - 0.5` (fraction of image height)
- **Description**: Minimum vertical line length as fraction of image height
- **Examples**:
  - `0.2`: Detect shorter vertical lines (20% of height)
  - `0.3`: Standard detection (30% of height)
  - `0.4`: Detect only longer vertical lines (40% of height)
- **Tuning**: Decrease for documents with partial vertical lines, increase for cleaner documents

### `line_thick`
- **Type**: `int`
- **Default**: `3`
- **Range**: `1 - 10` pixels
- **Description**: Kernel width for vertical line detection
- **Examples**:
  - `2`: Detect thinner lines
  - `3`: Standard line thickness (default)
  - `5`: Detect thicker lines
- **Tuning**: Increase for documents with thick borders, decrease for thin lines

### `peak_thr`
- **Type**: `float`
- **Default**: `0.3`
- **Range**: `0.1 - 0.8` (fraction of max response)
- **Description**: Peak threshold for line detection (fraction of maximum response)
- **Examples**:
  - `0.2`: More sensitive detection (20% of max)
  - `0.3`: Standard sensitivity (30% of max)
  - `0.5`: Less sensitive detection (50% of max)
- **Tuning**: Decrease for faint lines, increase to reduce false detections

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

### Horizontal Line Length Filtering

#### `h_min_length_image_ratio`
- **Type**: `float`
- **Default**: `0.3`
- **Range**: `0.1 - 0.8`
- **Description**: Minimum horizontal line length as ratio of image width
- **Examples**:
  - `0.2`: More permissive (includes lines 20% of image width)
  - `0.3`: Standard threshold (default)
  - `0.5`: More restrictive (only lines 50% of image width or longer)
- **Tuning**: Decrease to include shorter horizontal segments, increase for quality control

#### `h_min_length_relative_ratio`
- **Type**: `float`
- **Default**: `0.4`
- **Range**: `0.2 - 0.9`
- **Description**: Minimum horizontal line length relative to longest detected horizontal line
- **Examples**:
  - `0.3`: More permissive (keeps lines 30% as long as the longest)
  - `0.4`: Standard threshold (default)
  - `0.7`: More restrictive (only lines 70% as long as the longest)
- **Tuning**: Decrease to keep more lines, increase for cleaner results

### Vertical Line Length Filtering

#### `v_min_length_image_ratio`
- **Type**: `float`
- **Default**: `0.3`
- **Range**: `0.1 - 0.8`
- **Description**: Minimum vertical line length as ratio of image height
- **Examples**:
  - `0.2`: More permissive (includes lines 20% of image height)
  - `0.3`: Standard threshold (default)
  - `0.5`: More restrictive (only lines 50% of image height or longer)
- **Tuning**: Decrease to include shorter vertical segments, increase for quality control

#### `v_min_length_relative_ratio`
- **Type**: `float`
- **Default**: `0.4`
- **Range**: `0.2 - 0.9`
- **Description**: Minimum vertical line length relative to longest detected vertical line
- **Examples**:
  - `0.3`: More permissive (keeps lines 30% as long as the longest)
  - `0.4`: Standard threshold (default)
  - `0.7`: More restrictive (only lines 70% as long as the longest)
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
  "search_ratio": 0.5,
  "threshold": 40
}
```

### High Sensitivity Configuration
```json
{
  "search_ratio": 0.8,
  "line_len_frac": 0.2,
  "peak_thr": 0.2,
  "angle_range": 10,
  "min_angle_correction": 0.05,
  "black_threshold": 30,
  "content_threshold": 150,
  "threshold": 25,
  "horizontal_kernel_size": 15,
  "vertical_kernel_size": 15,
  "h_min_length_image_ratio": 0.2,
  "h_min_length_relative_ratio": 0.3,
  "v_min_length_image_ratio": 0.2,
  "v_min_length_relative_ratio": 0.3,
  "min_aspect_ratio": 3
}
```

### Conservative Configuration
```json
{
  "search_ratio": 0.3,
  "line_thick": 5,
  "peak_thr": 0.5,
  "angle_range": 3,
  "min_angle_correction": 0.5,
  "black_threshold": 60,
  "content_threshold": 220,
  "threshold": 60,
  "horizontal_kernel_size": 8,
  "vertical_kernel_size": 8,
  "h_min_length_image_ratio": 0.4,
  "h_min_length_relative_ratio": 0.5,
  "v_min_length_image_ratio": 0.4,
  "v_min_length_relative_ratio": 0.5,
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

---

**Navigation**: [← Documentation Index](README.md) | [Configuration Guide](../configs/README.md) | [Debug Mode Guide](DEBUG_MODE_GUIDE.md) | [Tools Documentation](../tools/README.md) →