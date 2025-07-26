# Parameter Reference

Comprehensive reference for all configuration parameters in the OCR Table Extraction Pipeline.

## 📋 Table of Contents

- [Directory Configuration](#directory-configuration)
- [Page Splitting Parameters](#page-splitting-parameters)
- [Deskewing Parameters](#deskewing-parameters)
- [Line Detection Parameters](#line-detection-parameters)
- [Table Reconstruction Parameters](#table-reconstruction-parameters)
- [Table Fitting Parameters](#table-fitting-parameters)
- [Table Cropping Parameters](#table-cropping-parameters)

## 📁 Directory Configuration

### `directories` Section

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `raw_images` | string | Input directory for original scanned images | `"input/raw_images"` |
| `splited_images` | string | Output for page splitting (Stage 1) or input for Stage 2 | `"output/stage1_initial_processing/01_split_pages"` |
| `deskewed_images` | string | Output directory for deskewed images | `"output/stage1_initial_processing/02_deskewed"` |
| `lines_images` | string | Output directory for line detection results | `"output/stage1_initial_processing/03_line_detection"` |
| `table_images` | string | Output directory for table reconstruction | `"output/stage1_initial_processing/04_table_reconstruction"` |
| `table_fit_images` | string | Output directory for fitted tables | `"output/stage1_initial_processing/05_cropped_tables"` |
| `debug_output_dir` | string | Directory for debug images (if enabled) | `"debug/stage1_debug/line_detection"` |

## 📄 Page Splitting Parameters

### `page_splitting` Section (Stage 1 Only)

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `GUTTER_SEARCH_START_PERCENT` | float | 0.0-1.0 | 0.4 | Start searching for gutter at this percentage from left edge |
| `GUTTER_SEARCH_END_PERCENT` | float | 0.0-1.0 | 0.6 | End gutter search at this percentage from left edge |
| `SPLIT_THRESHOLD` | float | 0.0-1.0 | 0.8 | Confidence threshold for detecting double-page layout |
| `LEFT_PAGE_SUFFIX` | string | - | `"_page_2.jpg"` | Suffix for left page (verso) output files |
| `RIGHT_PAGE_SUFFIX` | string | - | `"_page_1.jpg"` | Suffix for right page (recto) output files |

### Parameter Impact:
- **Lower split threshold**: More aggressive splitting (may split single pages)
- **Narrower gutter region**: More precise gutter detection (may miss off-center bindings)
- **Wider gutter region**: More robust but may detect false gutters

## 🔄 Deskewing Parameters

### `deskewing` Section

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `ANGLE_RANGE` | float | 1.0-45.0 | 10.0 | Maximum angle to search (±degrees) |
| `ANGLE_STEP` | float | 0.1-1.0 | 0.2 | Angular precision in degrees |
| `MIN_ANGLE_FOR_CORRECTION` | float | 0.0-5.0 | 0.2 | Minimum detected angle to apply correction |

### Parameter Impact:
- **Larger angle range**: Can correct more severe skew but slower processing
- **Smaller angle step**: Higher precision but longer processing time
- **Lower minimum correction**: More aggressive correction (may over-correct)

## 📏 Line Detection Parameters

### `line_detection` Section

#### General Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `SAVE_DEBUG_IMAGES` | boolean | true | Generate intermediate debug images |
| `OUTPUT_VIZ_LINE_THICKNESS` | integer | 2 | Line thickness in visualization images |
| `OUTPUT_VIZ_V_COLOR_BGR` | array | [0, 0, 255] | Vertical line color (BGR format) |
| `OUTPUT_VIZ_H_COLOR_BGR` | array | [0, 255, 0] | Horizontal line color (BGR format) |
| `OUTPUT_JSON_SUFFIX` | string | `"_lines.json"` | Suffix for line data JSON files |
| `OUTPUT_VIZ_SUFFIX` | string | `"_visualization.jpg"` | Suffix for visualization images |

#### ROI Margins

Margins exclude page headers, footers, and binding areas (in pixels):

| Parameter | Type | Stage 1 Default | Stage 2 Default | Description |
|-----------|------|----------------|-----------------|-------------|
| `ROI_MARGINS_PAGE_1.top` | integer | 120 | 0 | Top margin for recto pages |
| `ROI_MARGINS_PAGE_1.bottom` | integer | 120 | 0 | Bottom margin for recto pages |
| `ROI_MARGINS_PAGE_1.left` | integer | 0 | 0 | Left margin for recto pages |
| `ROI_MARGINS_PAGE_1.right` | integer | 100 | 0 | Right margin for recto pages (binding) |
| `ROI_MARGINS_PAGE_2.top` | integer | 120 | 0 | Top margin for verso pages |
| `ROI_MARGINS_PAGE_2.bottom` | integer | 120 | 0 | Bottom margin for verso pages |
| `ROI_MARGINS_PAGE_2.left` | integer | 60 | 0 | Left margin for verso pages (binding) |
| `ROI_MARGINS_PAGE_2.right` | integer | 5 | 0 | Right margin for verso pages |
| `ROI_MARGINS_DEFAULT.*` | integer | 60/5 | 0 | Margins for unknown page types |

### Vertical Line Parameters (`V_PARAMS`)

#### Morphological Operations
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `morph_open_kernel_ratio` | float | 0.005-0.05 | 0.0166 | Opening kernel height = image_height × ratio |
| `morph_close_kernel_ratio` | float | 0.005-0.05 | 0.0166 | Closing kernel height = image_height × ratio |

#### Hough Transform
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `hough_threshold` | integer | 1-50 | 5 | Minimum intersections for line detection |
| `hough_min_line_length` | integer | 10-200 | 40 | Minimum line length in pixels |
| `hough_max_line_gap_ratio` | float | 0.0001-0.01 | 0.001 | Max gap = image_height × ratio |

#### Clustering and Filtering
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `cluster_distance_threshold` | integer | 5-50 | 15 | Group lines within this pixel distance |
| `qualify_length_ratio` | float | 0.1-1.0 | 0.5 | Keep lines > ratio × max_detected_length |
| `final_selection_ratio` | float | 0.1-1.0 | 0.5 | Final filter: keep lines > ratio × longest |
| `solid_check_std_threshold` | float | 10.0-100.0 | 30.0 | Remove solid regions with std_dev < threshold |

#### Curved Line Detection
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `contour_min_length_ratio` | float | 0.1-1.0 | 0.5 | Min contour length = ratio × image_height |
| `contour_aspect_ratio_threshold` | float | 2.0-20.0 | 5.0 | Height/width ratio for vertical lines |

### Horizontal Line Parameters (`H_PARAMS`)

#### Morphological Operations
| Parameter | Type | Range | Stage 1 | Stage 2 | Description |
|-----------|------|-------|---------|---------|-------------|
| `morph_open_kernel_ratio` | float | 0.005-0.1 | 0.033 | 0.01 | Opening kernel width = image_width × ratio |
| `morph_close_kernel_ratio` | float | 0.005-0.05 | 0.02 | 0.0166 | Closing kernel width = image_width × ratio |

#### Hough Transform
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `hough_threshold` | integer | 5-50 | 10 | Minimum intersections (higher than vertical) |
| `hough_min_line_length` | integer | 10-200 | 30 | Minimum horizontal line length |
| `hough_max_line_gap_ratio` | float | 0.001-0.05 | 0.01 | Max gap = image_width × ratio |

#### Clustering and Filtering
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `cluster_distance_threshold` | integer | 2-20 | 5 | Tighter clustering for horizontal lines |
| `qualify_length_ratio` | float | 0.1-1.0 | 0.5 | Keep lines > ratio × max_detected_length |
| `final_selection_ratio` | float | 0.5-1.0 | 0.7/0.8 | More selective filtering (Stage 1/2) |
| `solid_check_std_threshold` | float | 20.0-100.0 | 50.0 | Higher threshold for horizontal solids |

#### Curved Line Detection
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `contour_min_length_ratio` | float | 0.3-1.0 | 0.8 | Min contour length = ratio × image_width |
| `contour_aspect_ratio_threshold` | float | 2.0-20.0 | 5.0 | Width/height ratio for horizontal lines |

## 🏗️ Table Reconstruction Parameters

### `table_reconstruction` Section

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `OUTPUT_IMAGE_SUFFIX` | string | `"_reconstructed.jpg"` | Suffix for table grid visualizations |
| `OUTPUT_DETAILS_SUFFIX` | string | `"_details.json"` | Suffix for detailed grid metadata |
| `VERTICAL_LINE_COLOR` | array | [0, 0, 255] | Color for vertical grid lines (BGR) |
| `HORIZONTAL_LINE_COLOR` | array | [0, 255, 0] | Color for horizontal grid lines (BGR) |
| `LINE_THICKNESS` | integer | 2 | Grid line thickness in pixels |

## 🎯 Table Fitting Parameters

### `table_fitting` Section

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `INPUT_JSON_SUFFIX` | string | - | `"_details.json"` | Input suffix for table details |
| `OUTPUT_IMAGE_SUFFIX` | string | - | `"_fitted.jpg"` | Output suffix for fitted visualizations |
| `OUTPUT_CELLS_SUFFIX` | string | - | `"_fitted_cells.json"` | Output suffix for cell data |
| `TOLERANCE` | integer | 5-50 | 15 | Pixel tolerance for line alignment |
| `COVERAGE_THRESHOLD` | float | 0.1-1.0 | 0.4 | Minimum line coverage for valid intersections |
| `FIGURE_SIZE` | array | - | [10, 12] | Matplotlib figure size [width, height] |
| `LINE_COLOR` | string | - | `"blue"` | Cell border color for visualization |
| `LINE_WIDTH` | integer | 1-5 | 2 | Cell border thickness |

## ✂️ Table Cropping Parameters

### `table_cropping` Section

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `OUTPUT_DIR` | string | - | varies | Output directory for cropped tables |
| `INPUT_DETAILS_SUFFIX` | string | - | `"_details.json"` | Input suffix for table boundaries |
| `OUTPUT_IMAGE_SUFFIX` | string | - | `"_cropped.jpg"` | Output suffix for cropped images |
| `OUTPUT_LINES_SUFFIX` | string | - | `"_cropped_lines.json"` | Output suffix for adjusted line data |
| `PADDING` | integer | 0-50 | 10 | Extra pixels around table boundaries |

## 🎛️ Parameter Tuning Guidelines

### For Different Document Types

#### Academic Papers
- **ROI Margins**: Use provided defaults (120px top/bottom for headers/footers)
- **Line Detection**: Standard morphological ratios work well
- **Clustering**: Default thresholds handle typical table spacing

#### Historical Documents  
- **Deskewing**: Increase `ANGLE_RANGE` to 15-20 degrees
- **Hough Threshold**: Decrease to 3-5 for faded lines
- **Solid Check**: Lower threshold (20-25) for aged paper texture

#### Technical Manuals
- **Morphological Ratios**: May need adjustment for line thickness
- **Page Splitting**: Different `SPLIT_THRESHOLD` for binding styles
- **ROI Margins**: Adjust for different header/footer sizes

#### Handwritten Tables
- **Curved Line Detection**: Lower `contour_aspect_ratio_threshold` (3.0)
- **Hough Parameters**: More tolerant settings
- **Clustering**: Wider distance thresholds

### Performance vs. Quality Trade-offs

#### Higher Quality (Slower)
- Smaller `ANGLE_STEP` (0.1 degrees)
- Lower Hough thresholds
- Higher morphological kernel ratios
- More debug images enabled

#### Faster Processing (Lower Quality)
- Larger `ANGLE_STEP` (0.5 degrees)  
- Higher Hough thresholds
- Disable debug images
- Higher clustering thresholds

### Common Parameter Relationships

1. **Morphological Kernels**: Ratio to image dimensions maintains scale independence
2. **Hough Gaps**: Ratio-based gaps adapt to image resolution
3. **Quality Filters**: Progressive filtering from qualify → final selection
4. **ROI vs. Margins**: Stage 1 uses margins, Stage 2 uses zero (pre-cropped)

---

For configuration examples and tuning workflows, see [CONFIG_GUIDE.md](CONFIG_GUIDE.md).