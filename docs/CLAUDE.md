# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OCR_Preprocess_ToTable is an OCR table extraction pipeline that processes scanned documents to extract table structures. The project supports both single-stage and two-stage processing workflows:

**Single-Stage Pipeline**: Complete processing in one pass (simple workflow)
**Two-Stage Pipeline**: Professional workflow with initial processing and refinement stages:

1. **Stage 1**: Page splitting, deskewing, ROI detection, line detection, table reconstruction, and cropping
2. **Stage 2**: Re-deskewing, refined line detection, final table reconstruction, and publication fitting

### Architecture

- **src/ocr_pipeline/config.py**: Configuration management with Config, Stage1Config, Stage2Config classes
- **src/ocr_pipeline/pipeline.py**: OCRPipeline (single-stage) and TwoStageOCRPipeline classes
- **src/ocr_pipeline/utils.py**: Core image processing utilities (splitting, deskewing, line detection, cropping, ROI detection)
- **src/ocr_pipeline/processors/**: Empty module for potential future processing extensions
- **tools/**: Debugging and visualization tools for each pipeline stage
- **examples/**: Usage examples and basic integration demos
- **scripts/run_stage1.py**: CLI for Stage 1 initial processing
- **scripts/run_stage2.py**: CLI for Stage 2 refinement processing
- **scripts/run_complete.py**: CLI for complete two-stage pipeline

### Processing Workflows

**Single-Stage Processing**:
1. Split two-page spreads into individual pages
2. Deskew images using line detection and Hough transforms
3. ROI detection (optional): Use Gabor filters to detect content regions and crop
4. Detect horizontal/vertical table lines using morphological operations
5. Crop to table regions based on detected boundaries

**Two-Stage Processing**:
- **Stage 1**: Page splitting → Deskewing → ROI detection → Line detection → Table reconstruction → Table cropping
- **Stage 2**: Re-deskewing → Refined line detection → Final reconstruction → Publication fitting

## Development Commands

### Setup and Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Running the Pipeline

#### Single-Stage Pipeline (Simple)
```bash
# Process images using module
python -m src.ocr_pipeline.pipeline

# Process specific directory
python -m src.ocr_pipeline.pipeline input/ -o output/ --verbose

# Process single file
python -m src.ocr_pipeline.pipeline image.jpg -o output/

# Using installed console script (if available)
ocr-pipeline input/ -o output/ --verbose
```

#### Two-Stage Pipeline (Professional)
```bash
# Complete two-stage pipeline (recommended)
python scripts/run_complete.py input/ --verbose

# Process single image through both stages
python scripts/run_complete.py image.jpg --verbose

# Custom output directory
python scripts/run_complete.py input/ -o custom_output/ --verbose

# Run stages individually
python scripts/run_stage1.py input/ --verbose              # Stage 1: Initial processing
python scripts/run_stage2.py --verbose                     # Stage 2: Refinement

# Stage-specific options
python scripts/run_stage1.py input/ --disable-roi --debug  # Disable ROI detection, save debug images
python scripts/run_stage2.py --s2-min-line-length 25       # Custom line detection parameters

# Run only one stage
python scripts/run_complete.py input/ --stage1-only        # Only initial processing
python scripts/run_complete.py --stage2-only               # Only refinement (requires Stage 1 output)
```

### Testing
```bash
# Run tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_pipeline.py -v

# Run with coverage (install pytest-cov first)
python -m pytest tests/ --cov=src.ocr_pipeline
```

### Development Workflow
```bash
# Run single test
python tests/test_pipeline.py

# Run visualization tools
python tools/run_visualizations.py
python tools/check_results.py
```

### Visualization and Debugging
```bash
# Visualize specific pipeline stages
python tools/visualize_page_split.py
python tools/visualize_deskew.py
python tools/visualize_roi.py
python tools/visualize_table_lines.py
python tools/visualize_table_crop.py
```

## Key Configuration Parameters

### Common Parameters
- `gutter_search_start/end`: Controls two-page splitting region (0.4-0.6 default)
- `save_debug_images`: Enables debug output for troubleshooting
- `verbose`: Enables detailed processing output
- `enable_roi_detection`: Enables ROI preprocessing stage (True default)
- `gabor_kernel_size/sigma/lambda/gamma`: Gabor filter parameters for ROI detection
- `roi_vertical_mode/horizontal_mode`: Cut detection modes ('single_best' or 'both_sides')
- `roi_min_cut_strength/confidence_threshold`: Thresholds for ROI boundary detection

### Stage-Specific Parameters

**Stage 1 (Initial Processing)**:
- `angle_range`: Deskewing angle range (±10° default)
- `angle_step`: Deskewing angle step (0.2° default)
- `min_line_length`: Minimum line length for table detection (40px default)
- `max_line_gap`: Maximum gap in detected lines (15px default)
- `roi_margins_page_1/2`: ROI margins for different page types
- Output directories: `01_split_pages`, `02_deskewed`, `02.5_edge_detection`, `03_line_detection`, `04_table_reconstruction`, `05_cropped_tables`

**Stage 2 (Refinement)**:
- `angle_range`: Refinement deskewing angle range (±10° default)
- `angle_step`: Refinement deskewing angle step (0.2° default)
- `min_line_length`: Minimum line length for refined detection (30px default)
- `max_line_gap`: Maximum gap in refined line detection (5px default)
- `enable_roi_detection`: Disabled by default (images already cropped)
- Output directories: `01_deskewed`, `02_line_detection`, `03_table_reconstruction`, `04_fitted_tables`

## Code Patterns

- Uses pathlib.Path for all file operations
- OpenCV (cv2) for image processing with NumPy arrays
- Configuration via dataclasses with post_init validation
- Error handling with try/catch blocks in pipeline processing
- Type hints throughout for better code clarity

## Testing Approach

Tests use pytest with synthetic test images created via NumPy. Key test patterns:
- Configuration validation tests
- Image processing function unit tests  
- Pipeline integration tests with error conditions
- Uses `pytest.raises()` for exception testing