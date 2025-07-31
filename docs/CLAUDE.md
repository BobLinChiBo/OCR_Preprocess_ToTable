# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a professional two-stage OCR (Optical Character Recognition) table extraction pipeline designed to process scanned document images and extract tables with high accuracy. The system uses computer vision techniques for page splitting, deskewing, ROI detection, and table line detection.

## Project Architecture

### Core Pipeline Structure
- **Two-Stage Processing**: Initial processing (Stage 1) followed by refinement (Stage 2)
- **src/ocr_pipeline/**: Main pipeline modules
  - `pipeline.py`: Core pipeline classes (`OCRPipeline`, `TwoStageOCRPipeline`)
  - `config.py`: Configuration classes (`Config`, `Stage1Config`, `Stage2Config`)
  - `utils.py`: Image processing utilities
- **scripts/**: CLI entry points for pipeline execution
- **tools/**: Comprehensive parameter tuning and visualization tools

### Data Flow
1. **Raw Input**: Scanned document images → `data/input/`
2. **Stage 1**: Page splitting, deskewing, ROI detection, table cropping → `data/output/stage1_initial_processing/`
3. **Stage 2**: Refinement processing on cropped tables → `data/output/stage2_refinement/`
4. **Final Output**: Publication-ready table images → `data/output/stage2_refinement/04_fitted_tables/`

### Configuration System
- JSON-based configuration files in `configs/` directory
- Hierarchical configuration classes with inheritance (`Config` → `Stage1Config`/`Stage2Config`)
- Runtime parameter validation and directory auto-creation
- Support for nested parameter structures in JSON configs

## Common Commands

### Pipeline Execution
```bash
# Complete two-stage pipeline
python scripts/run_complete.py data/input/ --verbose

# Single image processing
python scripts/run_complete.py image.jpg --verbose

# Stage 1 only (initial processing)
python scripts/run_stage1.py data/input/ --verbose

# Stage 2 only (refinement)
python scripts/run_stage2.py --verbose

# With custom output directory
python scripts/run_complete.py data/input/ -o custom_output/ --verbose --debug
```

### Parameter Tuning and Optimization
```bash
# Interactive guided tuning
python tools/quick_start_tuning.py

# Manual step-by-step tuning
python tools/setup_tuning.py                 # One-time setup
python tools/tune_page_splitting.py          # Stage 1: Page separation
python tools/tune_deskewing.py               # Stage 2: Rotation correction
python tools/tune_roi_detection.py           # Stage 3: Content area detection
python tools/tune_line_detection.py          # Stage 4: Table line detection

# Test optimized parameters
python tools/run_tuned_pipeline.py data/input/ --verbose
```

### Visualization and Analysis
```bash
# Complete pipeline analysis
python tools/run_visualizations.py all --pipeline image.jpg --save-intermediates

# Individual step analysis
python tools/visualize_deskew.py image.jpg --angle-range 30
python tools/visualize_page_split.py image.jpg --gutter-start 0.35 --gutter-end 0.65
python tools/visualize_roi.py image.jpg --gabor-threshold 150
python tools/visualize_table_lines.py image.jpg --min-line-length 40

# Results management
python tools/check_results.py list
python tools/check_results.py view latest
python tools/compare_results.py
```

### Development and Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Code quality (if dev dependencies installed)
python -m black src/ scripts/ tools/
python -m flake8 src/ scripts/ tools/
python -m mypy src/
```

## Key Processing Stages

### Stage 1: Initial Processing (Aggressive Parameters)
1. **Page Splitting**: Separate double-page scans using gutter detection
2. **Deskewing**: Correct rotation up to ±10° with 0.2° steps
3. **ROI Detection**: Use Gabor filters or edge detection to identify content areas
4. **Line Detection**: Find table structure with permissive parameters (min_line_length=40)
5. **Table Cropping**: Extract table regions for Stage 2 processing

### Stage 2: Refinement (Precise Parameters)
1. **Re-deskewing**: Fine-tune rotation on cropped tables
2. **Refined Line Detection**: More precise detection (min_line_length=30, max_line_gap=5)
3. **Table Reconstruction**: Advanced table structure analysis
4. **Table Fitting**: Final optimization for publication-ready output

## Configuration Parameters

### Critical Parameters to Tune
- **Page Splitting**: `gutter_search_start` (0.35-0.45), `gutter_search_end` (0.55-0.65)
- **Deskewing**: `angle_range` (5-20°), `min_angle_correction` (0.1-1.0°)
- **ROI Detection**: `gabor_binary_threshold` (90-180), `roi_min_cut_strength` (10-30)
- **Line Detection**: `min_line_length` (20-80), `max_line_gap` (5-25)

### ROI Detection Methods
- `gabor`: Original Gabor filter-based edge detection (default)
- `canny_sobel`: Combined Canny and Sobel edge detection
- `adaptive_threshold`: Adaptive thresholding with edge enhancement

## Data Organization

### Input Structure
```
data/input/
├── raw_images/          # Full dataset of scanned images
└── test_images/         # Subset for parameter tuning (6 representative images)
```

### Output Structure
```
data/output/
├── stage1_initial_processing/
│   ├── 01_split_pages/
│   ├── 02_deskewed/
│   ├── 02.5_edge_detection/
│   ├── 03_line_detection/
│   ├── 04_table_reconstruction/
│   └── 05_cropped_tables/        # Input for Stage 2
├── stage2_refinement/
│   ├── 01_deskewed/
│   ├── 02_line_detection/
│   ├── 03_table_reconstruction/
│   └── 04_fitted_tables/         # Final publication-ready output
└── tuning/                       # Parameter optimization results
```

## Best Practices

### Development Workflow
1. **Parameter Tuning**: Always tune parameters on representative test images before processing large datasets
2. **Stage Validation**: Visually inspect intermediate results at each stage
3. **Configuration Management**: Use JSON config files for reproducible parameter sets
4. **Visualization**: Use built-in visualization tools to understand pipeline behavior

### Performance Optimization
- Use test images (6 representative samples) for parameter tuning
- Process images in batches for better throughput
- Enable debug mode only when needed (generates large intermediate files)
- Consider image resolution - 300+ DPI recommended for best results

### Error Handling
- Pipeline includes comprehensive error handling with graceful degradation
- Use `--verbose` flag for detailed processing information
- Check intermediate outputs if final results are poor
- ROI detection can be disabled if causing issues (`--disable-roi`)

## Troubleshooting

### Common Issues
- **Poor page splitting**: Adjust gutter search parameters based on document layout
- **Excessive rotation**: Increase `min_angle_correction` to be more conservative
- **Over-cropping**: Increase `roi_min_cut_strength` for more conservative ROI detection
- **Missing table lines**: Decrease `min_line_length` or increase `max_line_gap`

### Performance Issues
- Large images may require significant memory - consider resizing for tuning
- Use `--save-intermediates` selectively to manage disk space
- Clean up tuning outputs periodically using provided management tools

The comprehensive tools/ directory provides extensive visualization and tuning capabilities - refer to `tools/README.md` and `tools/TUNING_GUIDE.md` for detailed optimization guidance.