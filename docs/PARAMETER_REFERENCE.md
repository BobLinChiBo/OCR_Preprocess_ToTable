# OCR Pipeline Parameter Reference

Complete reference guide for all configurable parameters in the OCR table extraction pipeline.

## Table of Contents

1. [Directory Configuration](#directory-configuration)
2. [Page Splitting Parameters](#page-splitting-parameters)
3. [Deskewing Parameters](#deskewing-parameters)
4. [ROI Detection Parameters](#roi-detection-parameters)
5. [Line Detection Parameters](#line-detection-parameters)
6. [Debug and Output Parameters](#debug-and-output-parameters)
7. [Stage-Specific Configurations](#stage-specific-configurations)
8. [Parameter Validation Rules](#parameter-validation-rules)
9. [Tuning Guidelines](#tuning-guidelines)

---

## Directory Configuration

### `input_dir`
- **Type**: `Path`
- **Default**: `"data/input"`
- **Description**: Directory containing input images to process
- **Stage 1**: Raw scanned document images
- **Stage 2**: Cropped table images from Stage 1 output (`stage1_output/05_cropped_tables/`)

### `output_dir`
- **Type**: `Path`
- **Default**: `"data/output"` (Stage 1: `"data/output/stage1_initial_processing"`, Stage 2: `"data/output/stage2_refinement"`)
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
  - `0.3`: Start searching at 30% of image width (wider search area)
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
  - `0.7`: Stop searching at 70% of image width (wider search area)
- **Tuning**: Must be greater than `gutter_search_start`

### `min_gutter_width`
- **Type**: `int`
- **Default**: `50`
- **Range**: `20 - 200` pixels
- **Description**: Minimum width required for a valid gutter detection
- **Examples**:
  - `20`: Very narrow binding (magazines, thin books)
  - `50`: Standard book binding (default)
  - `100`: Thick book binding (textbooks, manuals)
- **Tuning**: Increase for thicker books, decrease for thin publications

---

## Deskewing Parameters

Deskewing corrects rotational skew in scanned images to make text horizontal and table lines straight.

### `angle_range`
- **Type**: `int`
- **Default**: `5` (Base), `10` (Stage 1), `10` (Stage 2)
- **Range**: `1 - 45` degrees
- **Description**: Maximum rotation angle to detect and correct (±degrees)
- **Examples**:
  - `5`: Conservative range for mostly straight documents
  - `10`: Standard range for typical scanning variations (default)
  - `20`: Wide range for severely skewed documents
- **Tuning**: Increase for documents with significant skew, decrease for stability

### `angle_step`
- **Type**: `float`
- **Default**: `0.1` (Base), `0.2` (Stages 1&2)
- **Range**: `0.05 - 2.0` degrees
- **Description**: Precision of angle detection (smaller = more precise, slower)
- **Examples**:
  - `0.1`: High precision detection (slower)
  - `0.2`: Standard precision (default for stages)
  - `0.5`: Fast detection (less precise)
- **Tuning**: Decrease for higher accuracy, increase for faster processing

### `min_angle_correction`
- **Type**: `float`
- **Default**: `0.1` (Base), `0.2` (Stages 1&2)
- **Range**: `0.05 - 5.0` degrees
- **Description**: Minimum detected angle before applying rotation correction
- **Examples**:
  - `0.1`: Very sensitive (corrects tiny rotations)
  - `0.2`: Standard sensitivity (default for stages)
  - `1.0`: Conservative (only corrects obvious skew)
- **Tuning**: Increase to avoid over-correction of straight images

---

## ROI Detection Parameters

ROI (Region of Interest) detection identifies and crops the content area, removing margins and background noise.

### Core ROI Settings

#### `enable_roi_detection`
- **Type**: `bool`
- **Default**: `True` (Stage 1), `False` (Stage 2)
- **Description**: Enable/disable ROI detection preprocessing
- **Stage 1**: Enabled to focus on content areas
- **Stage 2**: Disabled (images already cropped)

#### `roi_detection_method`
- **Type**: `str`
- **Default**: `"gabor"`
- **Options**: `"gabor"`, `"canny_sobel"`, `"adaptive_threshold"`
- **Description**: Edge detection method for ROI boundary identification
- **Methods**:
  - `gabor`: Original Gabor filter-based edge detection (best for textured content)
  - `canny_sobel`: Combined Canny and Sobel edge detection (good for clear boundaries)
  - `adaptive_threshold`: Adaptive thresholding with edge enhancement (robust for varied lighting)

### Gabor Filter Parameters (roi_detection_method="gabor")

#### `gabor_kernel_size`
- **Type**: `int`
- **Default**: `31`
- **Range**: `15 - 61` (odd numbers only)
- **Description**: Size of the Gabor filter kernel for edge detection
- **Examples**:
  - `21`: Smaller kernel (finer edge detection, more noise sensitive)
  - `31`: Standard kernel (default, balanced)
  - `41`: Larger kernel (broader edge detection, more robust)
- **Tuning**: Increase for noisy images, decrease for fine detail preservation

#### `gabor_sigma`
- **Type**: `float`
- **Default**: `3.0` (Base), `4.0` (Stage 1)
- **Range**: `1.0 - 8.0`
- **Description**: Standard deviation of the Gaussian envelope (edge detection sensitivity)
- **Examples**:
  - `2.0`: Sharp edge detection (high sensitivity)
  - `4.0`: Standard edge detection (default for Stage 1)
  - `6.0`: Smooth edge detection (low sensitivity)
- **Tuning**: Increase for noise reduction, decrease for fine edge detection

#### `gabor_lambda`
- **Type**: `float`
- **Default**: `8.0`
- **Range**: `4.0 - 16.0`
- **Description**: Wavelength of the Gabor filter (spatial frequency)
- **Examples**:
  - `6.0`: Higher frequency (detects finer patterns)
  - `8.0`: Standard frequency (default)
  - `12.0`: Lower frequency (detects broader patterns)
- **Tuning**: Adjust based on typical text/line spacing in documents

#### `gabor_gamma`
- **Type**: `float`
- **Default**: `0.2`
- **Range**: `0.1 - 1.0`
- **Description**: Aspect ratio of the Gabor filter ellipse
- **Examples**:
  - `0.1`: Very elongated filter (directional edge detection)
  - `0.2`: Standard elongation (default)
  - `0.5`: Rounder filter (less directional)
- **Tuning**: Lower values for line-heavy documents, higher for general content

#### `gabor_binary_threshold`
- **Type**: `int`
- **Default**: `127`
- **Range**: `50 - 200`
- **Description**: Threshold for converting Gabor response to binary edge map
- **Examples**:
  - `90`: Sensitive threshold (more edges detected)
  - `127`: Standard threshold (default)
  - `160`: Conservative threshold (fewer edges detected)
- **Tuning**: Decrease for faint content, increase for noise reduction

### Canny + Sobel Parameters (roi_detection_method="canny_sobel")

#### `canny_low_threshold`
- **Type**: `int`
- **Default**: `50`
- **Range**: `20 - 100`
- **Description**: Low threshold for Canny edge detection
- **Tuning**: Lower values detect more edges (including weak ones)

#### `canny_high_threshold`
- **Type**: `int`
- **Default**: `150`
- **Range**: `100 - 300`
- **Description**: High threshold for Canny edge detection
- **Tuning**: Should be 2-3x the low threshold

#### `sobel_kernel_size`
- **Type**: `int`
- **Default**: `3`
- **Range**: `3, 5, 7` (odd numbers)
- **Description**: Size of Sobel operator kernel
- **Tuning**: Larger kernels for smoother gradient estimation

#### `gaussian_blur_size`
- **Type**: `int`
- **Default**: `5`
- **Range**: `3, 5, 7, 9` (odd numbers)
- **Description**: Gaussian blur kernel size for noise reduction
- **Tuning**: Increase for noisy images

#### `edge_binary_threshold`
- **Type**: `int`
- **Default**: `127`
- **Range**: `50 - 200`
- **Description**: Threshold for final edge binary conversion
- **Tuning**: Similar to `gabor_binary_threshold`

#### `morphology_kernel_size`
- **Type**: `int`
- **Default**: `3`
- **Range**: `3, 5, 7` (odd numbers)
- **Description**: Morphological operation kernel size for edge cleanup
- **Tuning**: Larger kernels connect broken edges better

### Adaptive Threshold Parameters (roi_detection_method="adaptive_threshold")

#### `adaptive_method`
- **Type**: `str`
- **Default**: `"gaussian"`
- **Options**: `"mean"`, `"gaussian"`
- **Description**: Adaptive thresholding method
- **Methods**:
  - `mean`: Simple mean of neighborhood area
  - `gaussian`: Gaussian-weighted sum of neighborhood values

#### `adaptive_block_size`
- **Type**: `int`
- **Default**: `11`
- **Range**: `3 - 21` (odd numbers)
- **Description**: Size of neighborhood area for adaptive thresholding
- **Tuning**: Increase for varied lighting conditions

#### `adaptive_C`
- **Type**: `float`
- **Default**: `2.0`
- **Range**: `0.0 - 10.0`
- **Description**: Constant subtracted from the mean for adaptive threshold
- **Tuning**: Increase to reduce noise, decrease for more sensitivity

#### `edge_enhancement`
- **Type**: `bool`
- **Default**: `True`
- **Description**: Apply additional edge enhancement after adaptive thresholding
- **Tuning**: Enable for documents with poor contrast

### ROI Cut Detection Parameters

#### `roi_vertical_mode`
- **Type**: `str`
- **Default**: `"single_best"`
- **Options**: `"both_sides"`, `"single_best"`
- **Description**: Vertical cropping strategy
- **Modes**:
  - `single_best`: Find one optimal vertical cut boundary
  - `both_sides`: Detect both top and bottom boundaries independently

#### `roi_horizontal_mode`
- **Type**: `str`
- **Default**: `"both_sides"`
- **Options**: `"both_sides"`, `"single_best"`
- **Description**: Horizontal cropping strategy
- **Modes**:
  - `both_sides`: Detect left and right boundaries independently
  - `single_best`: Find one optimal horizontal cut boundary

#### `roi_window_size_divisor`
- **Type**: `int`
- **Default**: `20`
- **Range**: `10 - 50`
- **Description**: Divisor for calculating sliding window size (image_size/divisor)
- **Examples**:
  - `10`: Large windows (broader analysis, less precise)
  - `20`: Standard windows (default)
  - `40`: Small windows (fine-grained analysis, more precise)
- **Tuning**: Increase for more precise boundary detection

#### `roi_min_window_size`
- **Type**: `int`
- **Default**: `10`
- **Range**: `5 - 50` pixels
- **Description**: Minimum window size regardless of divisor calculation
- **Tuning**: Prevents windows from becoming too small on small images

#### `roi_min_cut_strength`
- **Type**: `float`
- **Default**: `1000.0` (Base), `20.0` (Stage 1)
- **Range**: `1.0 - 2000.0`
- **Description**: Minimum edge strength required to make a crop cut
- **Examples**:
  - `10.0`: Aggressive cropping (removes more background)
  - `20.0`: Standard cropping (default for Stage 1)
  - `1000.0`: Very conservative cropping (minimal removal)
- **Tuning**: Decrease for more aggressive cropping, increase for conservation

#### `roi_min_confidence_threshold`
- **Type**: `float`
- **Default**: `50.0` (Base), `5.0` (Stage 1)
- **Range**: `1.0 - 100.0`
- **Description**: Minimum confidence score required for applying ROI cropping
- **Examples**:
  - `3.0`: Low confidence required (more cropping applied)
  - `5.0`: Standard confidence (default for Stage 1)
  - `50.0`: High confidence required (very conservative)
- **Tuning**: Increase for more conservative ROI detection

### ROI Margins

Manual margin settings for page-specific cropping adjustments.

#### `roi_margins_page_1`
- **Type**: `dict`
- **Default**: `{"top": 120, "bottom": 120, "left": 0, "right": 100}` (Stage 1)
- **Description**: Manual margin adjustments for left pages (page 1 after splitting)
- **Units**: Pixels
- **Tuning**: Adjust based on typical page layout patterns

#### `roi_margins_page_2`
- **Type**: `dict`
- **Default**: `{"top": 120, "bottom": 120, "left": 60, "right": 5}` (Stage 1)
- **Description**: Manual margin adjustments for right pages (page 2 after splitting)
- **Units**: Pixels
- **Tuning**: Often different from page 1 due to binding asymmetry

#### `roi_margins_default`
- **Type**: `dict`
- **Default**: `{"top": 60, "bottom": 60, "left": 5, "right": 5}` (Stage 1)
- **Description**: Default margin adjustments when page-specific margins not available
- **Units**: Pixels
- **Tuning**: Conservative fallback values

---

## Line Detection Parameters

Line detection identifies table structure by finding horizontal and vertical lines in the processed images.

### `min_line_length`
- **Type**: `int`
- **Default**: `100` (Base), `40` (Stage 1), `30` (Stage 2)
- **Range**: `10 - 200` pixels
- **Description**: Minimum length required for a detected line to be considered valid
- **Examples**:
  - `20`: Very sensitive (detects short line segments)
  - `40`: Standard sensitivity for initial processing (Stage 1)
  - `30`: Refined sensitivity for cropped tables (Stage 2)
  - `80`: Conservative (only long, clear lines)
- **Tuning**: Decrease for tables with short segments, increase to reduce noise

### `max_line_gap`
- **Type**: `int`
- **Default**: `20` (Base), `40` (Stage 1), `30` (Stage 2)
- **Range**: `1 - 50` pixels
- **Description**: Maximum gap allowed within a single line (bridges broken lines)
- **Examples**:
  - `5`: Strict line continuity 
  - `20`: Standard gap tolerance (Base default)
  - `40`: Generous gap tolerance (Stage 1 default)
  - `30`: Moderate gap tolerance (Stage 2 default)
- **Tuning**: Increase for broken/faded lines, decrease for precise detection

### `hough_threshold`
- **Type**: `int`
- **Default**: `80` (Base), `40` (Stages 1&2)
- **Range**: `20 - 200`
- **Description**: Hough transform threshold for line detection sensitivity
- **Examples**:
  - `20`: Very sensitive (detects many weak lines, more noise)
  - `40`: Standard sensitivity (default for stages)
  - `80`: Conservative sensitivity (Base default)
  - `120`: High threshold (only strong, clear lines)
- **Tuning**: Decrease for faint table lines, increase to reduce noise

### Morphological Operation Parameters

#### `horizontal_kernel_ratio`
- **Type**: `int`
- **Default**: `30`
- **Range**: `10 - 50`
- **Description**: Ratio for horizontal morphological kernel size (image_width / ratio)
- **Examples**:
  - `20`: Larger kernel (connects longer horizontal gaps)
  - `30`: Standard kernel (default)
  - `40`: Smaller kernel (more precise horizontal detection)
- **Tuning**: Decrease for tables with wide horizontal spacing, increase for precise detection

#### `vertical_kernel_ratio`
- **Type**: `int`
- **Default**: `20` (Base/Stage1), `20` (Stage2)
- **Range**: `10 - 50`
- **Description**: Ratio for vertical morphological kernel size (image_height / ratio)
- **Examples**:
  - `15`: Larger kernel (connects longer vertical gaps)
  - `20`: Standard kernel (default)
  - `30`: Smaller kernel (more precise vertical detection)
- **Tuning**: Decrease for tables with tall cells, increase for precise detection

#### `h_erode_iterations`
- **Type**: `int`
- **Default**: `1`
- **Range**: `1 - 5`
- **Description**: Number of erosion iterations for horizontal morphological operations
- **Examples**:
  - `1`: Minimal erosion (preserves thin lines)
  - `2`: Standard erosion (removes some noise)
  - `3`: Aggressive erosion (removes more noise, may break thin lines)
- **Tuning**: Increase to remove noise, decrease to preserve thin lines

#### `h_dilate_iterations`
- **Type**: `int`
- **Default**: `1`
- **Range**: `1 - 5`
- **Description**: Number of dilation iterations for horizontal morphological operations
- **Examples**:
  - `1`: Minimal dilation (preserves line precision)
  - `2`: Standard dilation (connects small gaps)
  - `3`: Aggressive dilation (connects larger gaps, may merge separate lines)
- **Tuning**: Increase to connect broken lines, decrease for precise boundaries

#### `v_erode_iterations`
- **Type**: `int`
- **Default**: `1`
- **Range**: `1 - 5`
- **Description**: Number of erosion iterations for vertical morphological operations
- **Examples**:
  - `1`: Minimal erosion (preserves thin lines)
  - `2`: Standard erosion (removes some noise)
  - `3`: Aggressive erosion (removes more noise, may break thin lines)
- **Tuning**: Increase to remove noise, decrease to preserve thin lines

#### `v_dilate_iterations`
- **Type**: `int`
- **Default**: `1`
- **Range**: `1 - 5`
- **Description**: Number of dilation iterations for vertical morphological operations
- **Examples**:
  - `1`: Minimal dilation (preserves line precision)
  - `2`: Standard dilation (connects small gaps)
  - `3`: Aggressive dilation (connects larger gaps, may merge separate lines)
- **Tuning**: Increase to connect broken lines, decrease for precise boundaries

### Coverage and Filtering Parameters

#### `min_table_coverage`
- **Type**: `float`
- **Default**: `0.4` (Base), `0.10` (Stage 1), `0.2` (Stage 2)
- **Range**: `0.05 - 0.8`
- **Description**: Minimum coverage ratio for lines to be considered table borders
- **Examples**:
  - `0.05`: Very permissive (accepts short line segments)
  - `0.10`: Permissive (Stage 1 default)
  - `0.2`: Moderate (Stage 2 default)
  - `0.4`: Conservative (Base default, requires substantial coverage)
- **Tuning**: Decrease for tables with short borders, increase to filter noise

#### `max_parallel_distance`
- **Type**: `int`
- **Default**: `15` (Base), `12` (Stage 1), `10` (Stage 2)
- **Range**: `3 - 30` pixels
- **Description**: Maximum distance between parallel lines before considering them duplicates
- **Examples**:
  - `5`: Strict deduplication (removes very close lines)
  - `12`: Standard deduplication (Stage 1 default)
  - `10`: Moderate deduplication (Stage 2 default)
  - `20`: Loose deduplication (allows closer parallel lines)
- **Tuning**: Decrease for precise line removal, increase to preserve close parallel lines

#### `angle_tolerance`
- **Type**: `float`
- **Default**: `5.0`
- **Range**: `1.0 - 15.0` degrees
- **Description**: Maximum angle deviation from horizontal/vertical for line classification
- **Examples**:
  - `2.0`: Strict orientation (only near-perfect horizontal/vertical lines)
  - `5.0`: Standard tolerance (default)
  - `10.0`: Permissive (accepts moderately skewed lines)
- **Tuning**: Decrease for precise table lines, increase for skewed documents

#### `h_length_filter_ratio`
- **Type**: `float`
- **Default**: `0.6` (Base), `0.5` (Stages 1&2)
- **Range**: `0.1 - 0.9`
- **Description**: Remove horizontal lines shorter than this ratio of the longest horizontal line
- **Examples**:
  - `0.3`: Permissive (keeps shorter horizontal segments)
  - `0.5`: Standard filtering (default for stages)
  - `0.7`: Strict filtering (only keeps longer horizontal lines)
- **Tuning**: Decrease to keep more horizontal segments, increase for quality filtering

#### `v_length_filter_ratio`
- **Type**: `float`
- **Default**: `0.6` (Base), `0.4` (Stage 1), `0.4` (Stage 2)
- **Range**: `0.1 - 0.9`
- **Description**: Remove vertical lines shorter than this ratio of the longest vertical line
- **Examples**:
  - `0.2`: Very permissive (keeps short vertical segments)
  - `0.4`: Standard filtering (default for stages)
  - `0.6`: Strict filtering (Base default)
- **Tuning**: Decrease to keep more vertical segments, increase for quality filtering

### Line Merging Parameters

#### `line_merge_distance_h`
- **Type**: `int`
- **Default**: `15`
- **Range**: `5 - 50` pixels
- **Description**: Maximum horizontal offset to merge horizontal lines that are slightly misaligned
- **Examples**:
  - `5`: Strict horizontal alignment required
  - `15`: Standard tolerance (default)
  - `25`: Permissive horizontal alignment
- **Tuning**: Increase for documents with alignment issues, decrease for precise merging

#### `line_merge_distance_v`
- **Type**: `int`
- **Default**: `15`
- **Range**: `5 - 50` pixels
- **Description**: Maximum vertical offset to merge vertical lines that are slightly misaligned
- **Examples**:
  - `5`: Strict vertical alignment required
  - `15`: Standard tolerance (default)
  - `25`: Permissive vertical alignment
- **Tuning**: Increase for documents with alignment issues, decrease for precise merging

#### `line_extension_tolerance`
- **Type**: `int`
- **Default**: `20`
- **Range**: `5 - 60` pixels
- **Description**: Maximum gap that can be bridged when extending lines for merging
- **Examples**:
  - `10`: Small gaps only
  - `20`: Standard gap bridging (default)
  - `40`: Large gap bridging
- **Tuning**: Increase to connect distant segments, decrease for conservative merging

#### `min_overlap_ratio`
- **Type**: `float`
- **Default**: `0.3`
- **Range**: `0.0 - 0.8`
- **Description**: Minimum overlap ratio required between line segments before merging
- **Examples**:
  - `0.0`: No overlap required (gap bridging only)
  - `0.1`: Minimal overlap (10% of shorter line)
  - `0.3`: Standard overlap (30% of shorter line, default)
  - `0.6`: Strict overlap (60% of shorter line)
- **Tuning**: Decrease for aggressive merging, increase for quality control
- **Note**: Lines can also merge based on `line_extension_tolerance` even without overlap

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
- **Recommendation**: Enable for troubleshooting and parameter tuning

---

## Stage-Specific Configurations

### Stage 1: Initial Processing
- **Purpose**: Aggressive initial detection and cropping
- **Key Characteristics**:
  - Wider angle ranges for deskewing (`angle_range=10`)
  - More permissive line detection (`min_line_length=40`, `max_line_gap=15`)
  - Active ROI detection with moderate cropping (`roi_min_cut_strength=20.0`)
  - Generous margins for content preservation

### Stage 2: Refinement
- **Purpose**: Precise refinement of cropped tables
- **Key Characteristics**:
  - Fine-tuning deskewing on pre-cropped content
  - Precise line detection (`min_line_length=30`, `max_line_gap=5`)
  - ROI detection disabled (content already cropped)
  - Minimal margins (images pre-processed)

---

## Parameter Validation Rules

### Automatic Validation
- `gutter_search_start` and `gutter_search_end` must be between 0.0 and 1.0
- `gutter_search_start` must be less than `gutter_search_end`
- All pixel-based parameters must be positive integers
- All angle parameters must be positive numbers
- Path parameters are automatically converted to `Path` objects

### Range Recommendations
- **Gutter Search**: Keep within center 70% of image (0.15-0.85 range)
- **Angle Range**: Rarely exceed 30° unless documents are severely skewed
- **ROI Thresholds**: Balance sensitivity vs. noise based on document quality
- **Line Parameters**: Adjust based on typical table cell sizes in your documents

---

## Tuning Guidelines

### General Approach
1. **Start Conservative**: Begin with default or higher threshold values
2. **Single Parameter**: Change one parameter at a time during tuning
3. **Visual Verification**: Always inspect results visually
4. **Representative Testing**: Use diverse test images
5. **Document Changes**: Keep notes on what works for different document types

### Line Processing Pipeline Order

The line detection follows a specific processing order for optimal results:

1. **Step 1**: Raw morphological detection (separate horizontal/vertical kernels)
2. **Step 2**: Orientation filtering (removes non-horizontal/vertical lines)
3. **Step 3**: **Line merging** (connects nearby misaligned segments - CRITICAL EARLY STEP)
4. **Step 4**: Coverage filtering (removes lines with insufficient span)
5. **Step 5**: Length filtering (removes lines shorter than ratio of longest)
6. **Step 6**: Deduplication (removes duplicate parallel lines)

**Key Insight**: Line merging happens BEFORE coverage and length filtering. This allows short segments that individually fail quality filters to be merged together first, then evaluated as complete lines.

### Parameter Interdependencies
- **ROI + Line Detection**: Aggressive ROI cropping may remove table lines needed for detection
- **Deskewing + ROI**: Poor deskewing can affect ROI boundary detection
- **Stage 1 ↔ Stage 2**: Stage 1 parameters directly affect Stage 2 input quality
- **Morphological + Merging**: More aggressive dilation creates thicker lines that merge more easily
- **Merging + Coverage**: Early merging allows short segments to meet coverage requirements after combination
- **Kernel Ratios + Line Lengths**: Smaller kernel ratios detect shorter segments that benefit more from merging

### Performance Considerations
- **Gabor Kernel Size**: Larger kernels = slower processing but more robust detection
- **Angle Step**: Smaller steps = more accurate but slower deskewing
- **Debug Images**: Significant storage overhead - use sparingly
- **Window Sizes**: Smaller ROI windows = more computational overhead

### Quality vs. Speed Trade-offs
- **High Quality**: Smaller angle steps, larger Gabor kernels, more sensitive thresholds
- **High Speed**: Larger angle steps, smaller kernels, conservative thresholds
- **Balanced**: Default parameters provide good compromise for most use cases

### Document-Specific Tuning
- **Clean Scans**: Use conservative parameters to avoid over-processing
- **Poor Quality**: Increase noise tolerance (larger kernels, higher thresholds)
- **Varied Content**: Use moderate parameters with good generalization
- **Consistent Layout**: Can use more aggressive/specialized parameters

### Line Merging Tuning Strategy

Line merging is particularly important for fragmented table borders. Follow this systematic approach:

#### Step 1: Assess Fragmentation
Run visualization with `--show-filtering-steps` to see the pipeline:
- Look at `step1_initial.jpg` - how fragmented are the raw lines?
- Compare `step3_merged.jpg` vs `step4_coverage.jpg` - is merging helping?

#### Step 2: Tune Merge Distance Parameters
Start with alignment tolerance:
- **Horizontal tables**: Focus on `line_merge_distance_h` (vertical misalignment of horizontal lines)
- **Vertical tables**: Focus on `line_merge_distance_v` (horizontal misalignment of vertical lines)
- **Poor scan quality**: Increase both merge distances (15→25px)
- **Clean scans**: Decrease for precision (15→8px)

#### Step 3: Adjust Extension Tolerance
Control gap bridging:
- **Broken borders**: Increase `line_extension_tolerance` (20→40px)
- **Dense content**: Decrease to avoid merging separate elements (20→10px)

#### Step 4: Fine-tune Overlap Requirements
Balance quality vs. coverage:
- **Aggressive merging**: Lower `min_overlap_ratio` (0.3→0.1)
- **Quality control**: Higher ratio (0.3→0.6)
- **Gap-only merging**: Set to 0.0 (relies entirely on extension_tolerance)

#### Step 5: Morphological Pre-processing
Optimize line thickness before merging:
- **Thin lines**: Increase `v_dilate_iterations` for better vertical connection
- **Thick lines**: Keep iterations low to preserve boundaries
- **Asymmetric needs**: Different h/v iteration values

#### Common Patterns
- **Chinese/Dense Text**: `line_merge_distance_v=20, line_extension_tolerance=30, min_overlap_ratio=0.1`
- **Technical Diagrams**: `line_merge_distance_h=25, line_merge_distance_v=15, min_overlap_ratio=0.2`
- **Old/Faded Documents**: `line_extension_tolerance=40, min_overlap_ratio=0.0` (gap-bridging only)
- **High-Quality Scans**: `line_merge_distance_h=8, line_merge_distance_v=8, min_overlap_ratio=0.5`

---

## Configuration File Examples

### Minimal Configuration
```json
{
  "verbose": true,
  "gutter_search_start": 0.4,
  "gutter_search_end": 0.6,
  "min_line_length": 40
}
```

### Aggressive Processing
```json
{
  "roi_detection": {
    "roi_min_cut_strength": 10.0,
    "roi_min_confidence_threshold": 3.0
  },
  "line_detection": {
    "min_line_length": 20,
    "max_line_gap": 40,
    "hough_threshold": 20,
    "horizontal_kernel_ratio": 20,
    "vertical_kernel_ratio": 15,
    "h_dilate_iterations": 3,
    "v_dilate_iterations": 3,
    "line_merge_distance_h": 25,
    "line_merge_distance_v": 25,
    "line_extension_tolerance": 40,
    "min_overlap_ratio": 0.1,
    "min_table_coverage": 0.05,
    "h_length_filter_ratio": 0.3,
    "v_length_filter_ratio": 0.2
  }
}
```

### Conservative Processing
```json
{
  "deskewing": {
    "min_angle_correction": 1.0
  },
  "roi_detection": {
    "roi_min_cut_strength": 50.0,
    "roi_min_confidence_threshold": 10.0
  },
  "line_detection": {
    "min_line_length": 80,
    "max_line_gap": 10,
    "hough_threshold": 100,
    "horizontal_kernel_ratio": 40,
    "vertical_kernel_ratio": 30,
    "h_erode_iterations": 1,
    "h_dilate_iterations": 1,
    "v_erode_iterations": 1,
    "v_dilate_iterations": 1,
    "line_merge_distance_h": 8,
    "line_merge_distance_v": 8,
    "line_extension_tolerance": 10,
    "min_overlap_ratio": 0.5,
    "min_table_coverage": 0.3,
    "h_length_filter_ratio": 0.7,
    "v_length_filter_ratio": 0.6
  }
}
```

### Optimized for Fragmented Tables
```json
{
  "line_detection": {
    "min_line_length": 10,
    "max_line_gap": 50,
    "hough_threshold": 30,
    "horizontal_kernel_ratio": 25,
    "vertical_kernel_ratio": 18,
    "h_dilate_iterations": 2,
    "v_dilate_iterations": 3,
    "line_merge_distance_h": 20,
    "line_merge_distance_v": 20,
    "line_extension_tolerance": 30,
    "min_overlap_ratio": 0.2,
    "min_table_coverage": 0.08,
    "h_length_filter_ratio": 0.4,
    "v_length_filter_ratio": 0.3,
    "max_parallel_distance": 8
  }
}
```

This parameter reference should be used in conjunction with the tuning tools in the `tools/` directory to systematically optimize the pipeline for your specific document types.