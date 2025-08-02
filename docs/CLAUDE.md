# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a professional two-stage OCR (Optical Character Recognition) table extraction pipeline designed to process scanned document images and extract tables with high accuracy. The system uses computer vision techniques for page splitting, deskewing, margin removal, and table line detection.

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
2. **Stage 1**: Page splitting, deskewing, margin removal, table cropping → `data/output/stage1_initial_processing/`
3. **Stage 2**: Refinement processing on cropped tables → `data/output/stage2_refinement/`
4. **Final Output**: Publication-ready table images → `data/output/stage2_refinement/04_fitted_tables/`

### Configuration System
- JSON-based configuration files in `configs/` directory
- Simplified configuration classes with essential parameters
- Runtime parameter validation and directory auto-creation
- Streamlined parameter structure for easier configuration

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

### Analysis and Debugging
```bash
# Setup utilities
python tools/setup_tuning.py

# Results management
python tools/check_results.py list
python tools/check_results.py cleanup
```

### Visualization and Analysis
```bash
# Complete pipeline analysis
python tools/run_visualizations.py all --pipeline image.jpg --save-intermediates

# Individual step analysis
python tools/visualize_deskew.py image.jpg
python tools/visualize_page_split.py image.jpg
python tools/visualize_roi.py image.jpg
python tools/visualize_table_lines.py image.jpg

# Results management
python tools/check_results.py list
python tools/check_results.py view latest
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

### Stage 1: Initial Processing
1. **Page Splitting**: Separate double-page scans using gutter detection
2. **Deskewing**: Correct rotation with sub-degree precision
3. **Margin Removal**: Remove document margins and background noise
4. **Line Detection**: Find table structure using connected components method
5. **Table Cropping**: Extract table regions for Stage 2 processing

### Stage 2: Refinement
1. **Re-deskewing**: Fine-tune rotation on cropped tables
2. **Refined Line Detection**: Optimized detection with refined parameters
3. **Table Reconstruction**: Advanced table structure analysis
4. **Table Fitting**: Final optimization for publication-ready output

## Configuration Parameters

### Key Parameters
- **Page Splitting**: `gutter_search_start` (0.4), `gutter_search_end` (0.6)
- **Deskewing**: `angle_range` (5°), `min_angle_correction` (0.1°)
- **Margin Removal**: `black_threshold` (50), `content_threshold` (200)
- **Line Detection**: `threshold` (40), `horizontal_kernel_size` (10), `vertical_kernel_size` (10)

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

### Git Operations Safety
**CRITICAL**: Before performing any git operations (stash, checkout, branch switching, etc.), always check for uncommitted changes and commit them first.

Required Process:
1. **Always run `git status` first** to check for uncommitted changes
2. **If there are uncommitted changes, commit them** before proceeding with git operations  
3. **Then perform the intended git operation** (stash, checkout, etc.)

Example workflow:
```bash
# Check status first
git status

# If changes exist, commit them
git add -A
git commit -m "Work in progress: [describe changes]"

# Then proceed with git operation
git stash push -m "temporary backup"
```

This prevents accidental code loss and maintains system functionality during git operations.

### Development Workflow
1. **Testing**: Test on representative samples before processing large datasets
2. **Stage Validation**: Visually inspect intermediate results at each stage
3. **Configuration Management**: Use JSON config files for reproducible parameter sets
4. **Visualization**: Use built-in visualization tools to understand pipeline behavior

### Performance Optimization
- Use test images for representative testing before processing large datasets
- Process images in batches for better throughput
- Enable debug mode only when needed (generates large intermediate files)
- Consider image resolution - 300+ DPI recommended for best results

### Error Handling
- Pipeline includes comprehensive error handling with graceful degradation
- Use `--verbose` flag for detailed processing information
- Check intermediate outputs if final results are poor
- Use visualization tools to debug processing issues

## Troubleshooting

### Common Issues
- **Poor page splitting**: Adjust `gutter_search_start` and `gutter_search_end` based on document layout
- **Excessive rotation**: Increase `min_angle_correction` to be more conservative
- **Over-cropping**: Adjust `content_threshold` and `black_threshold` for margin removal
- **Missing table lines**: Adjust `threshold` and kernel sizes for line detection

### Performance Issues
- Large images may require significant memory - consider resizing for testing
- Use `--save-intermediates` selectively to manage disk space
- Clean up output directories periodically using provided management tools

The tools/ directory provides visualization and analysis capabilities - refer to `tools/README.md` for detailed usage guidance.