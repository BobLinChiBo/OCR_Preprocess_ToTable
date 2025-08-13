# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an OCR preprocessing and table extraction pipeline designed to process scanned document images, particularly those containing tables. The pipeline operates in two distinct stages:

- **Stage 1**: **Table Cropping and Isolation** - Crops out all irrelevant content and focuses on the main table region
- **Stage 2**: **Structure Recovery and Column Extraction** - Recovers the actual table structure and cuts by vertical lines for OCR

## Key Commands

### Running the Pipeline
```bash
# Run complete pipeline with default configs
python scripts/run_pipeline.py

# Run with specific input directory
python scripts/run_pipeline.py data/input/raw_images

# Run with custom output directory
python scripts/run_pipeline.py -o custom_output

# Run parallel processing
python scripts/run_parallel.py

# Run specific stages
python scripts/run_stage1.py
python scripts/run_stage2.py

# Enable debug mode
python scripts/run_pipeline.py --save-debug-images --verbose
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=src/ocr_pipeline tests/
```

### Code Quality
```bash
# Format code with black
python -m black src/ scripts/ tests/

# Lint with flake8
python -m flake8 src/ scripts/ tests/

# Type checking with mypy
python -m mypy src/
```

## Architecture

### Core Components

1. **Pipeline Entry Points** (`scripts/`):
   - `run_pipeline.py`: Main CLI entry point with argument parsing
   - `run_parallel.py`: Parallel processing variant for batch operations
   - `run_stage1.py`: Stage 1 only (table cropping)
   - `run_stage2.py`: Stage 2 only (structure recovery and column cutting)

2. **Stage Processors** (`src/ocr_pipeline/`):
   - `stage1_processor.py`: Unified Stage 1 processing logic
   - `stage2_processor.py`: Unified Stage 2 processing logic  
   - `pipeline.py`: Main pipeline orchestration with memory/parallel modes
   - `config.py`: Configuration management and validation

3. **Image Processors** (`src/ocr_pipeline/processors/`):
   - `base.py`: Base processor class with debug image support
   - `mark_removal.py`: Watermark, stamp, and artifact removal
   - `margin_removal.py`: Paper margin detection and removal  
   - `page_split.py`: Two-page spread splitting
   - `deskew.py`: Image deskewing using Radon transform
   - `table_detection.py`: **Advanced table line detection with preprocessing**
   - `table_recovery.py`: Table structure recovery and grid completion
   - `tag_removal.py`: Header/footer tag removal
   - `vertical_strip_cutter.py`: Column extraction based on grid lines
   - `binarize.py`: **Image binarization with stroke enhancement**
   - `stroke_enhancement.py`: Morphological stroke enhancement utilities

4. **Configuration** (`configs/`):
   - `stage1_default.json`: **Single consolidated Stage 1 configuration**
   - `stage2_default.json`: Stage 2 structure recovery parameters
   - All preprocessing parameters included in main configs

### Processing Modes

The pipeline supports three execution modes:

1. **Regular Mode**: Process images sequentially with disk I/O (default)
2. **Parallel Mode**: Process multiple images concurrently using multiprocessing
3. **Memory Mode**: Keep intermediate results in memory (faster but uses more RAM)

### Data Flow

```
Input Images → Stage 1 (Table Cropping) → Cropped Tables → Stage 2 (Structure Recovery) → Column Strips
```

**Stage 1 - Table Cropping and Isolation:**
- `01_mark_removed/`: Watermarks/stamps removed (optional)
- `02_margin_removed/`: Paper margins removed (optional)
- `03_deskewed/`: Initial rotation correction
- `04_split_pages/`: Split two-page spreads (optional)
- `04b_tags_removed/`: Headers/footers removed (optional) 
- `05_table_lines/`: Table lines detected with preprocessing
- `06_table_structure/`: Grid structure identified
- `07_border_cropped/`: **Final cropped table regions (main output)**

**Stage 2 - Structure Recovery and Column Extraction:**
- `01_refined_deskewed/`: Fine-tuned rotation correction
- `02_table_lines/`: Enhanced line detection with preprocessing
- `03_table_structure/`: Final grid structure analysis
- `04_table_recovered/`: Complete table structure recovery
- `05_vertical_strips/`: **Individual columns for OCR (main output)**
- `06_binarized/`: OCR-ready binarized images (optional)
- `07_final_deskewed/`: Final rotation correction (optional)

## Table Line Detection with Preprocessing

Both stages feature advanced table line detection with built-in preprocessing:

### Key Features
- **Adaptive Binarization**: Smart thresholding for varying image quality
- **Stroke Enhancement**: Morphological operations to enhance thin/faded lines  
- **Denoising**: Removes artifacts that interfere with line detection
- **Multi-angle Detection**: Handles skewed tables up to configurable tolerance
- **Connected Components Analysis**: Robust line extraction and merging
- **Search Regions**: Focus detection on specific image areas

### Configuration
```json
{
  "table_line_detection": {
    "line_detection_use_preprocessing": true,
    "line_detection_binarization_method": "adaptive",
    "line_detection_stroke_enhancement": true,
    "line_detection_binarization_denoise": true,
    "line_detection_adaptive_block_size": 21,
    "line_detection_adaptive_c": 7,
    "skew_tolerance": 2,
    "threshold": 20,
    "horizontal_kernel_size": 80,
    "vertical_kernel_size": 40
  }
}
```

## Important Configuration Parameters

### Stage 1 Key Settings (Table Cropping)
- **Preprocessing Options:**
  - `mark_removal.enable`: Remove watermarks/stamps
  - `margin_removal.enable`: Remove paper edges/black borders
  - `page_splitting.enable`: Split two-page spreads
  - `tag_removal.enable`: Remove headers/footers
  
- **Table Detection:**
  - `line_detection_use_preprocessing`: Enable preprocessing for better detection
  - `line_detection_stroke_enhancement`: Enhance thin/faded table lines
  - `skew_tolerance`: Handle rotated tables (degrees)
  - `search_region_*`: Focus detection on specific areas
  
- **Table Cropping:**
  - `table_detection.enable_table_cropping`: Enable table region extraction
  - `table_detection.table_crop_padding`: Padding around detected tables

### Stage 2 Key Settings (Structure Recovery)
- **Fine Processing:**
  - `deskewing.enable`: Fine rotation correction for cropped tables
  - `line_detection_use_preprocessing`: Enhanced preprocessing for refined detection
  
- **Structure Recovery:**
  - `table_recovery.coverage_ratio`: Minimum coverage for structure recovery
  
- **Column Extraction:**
  - `vertical_strip_cutting.enable`: Extract individual columns
  - `vertical_strip_cutting.use_longest_lines_only`: Use dominant vertical lines
  - `vertical_strip_cutting.min_length_ratio`: Minimum line length threshold

### Performance Options
- `optimization.parallel_processing`: Enable multiprocessing
- `optimization.memory_mode`: Use memory-efficient processing
- `optimization.max_workers`: Number of parallel workers

## Debug Mode

Enable debug output in config files:
```json
{
  "save_debug_images": true,
  "debug_dir": "data/debug",
  "verbose": true
}
```

Debug images show:
- **Preprocessing steps**: binarization, stroke enhancement, denoising
- **Line detection**: morphological operations, connected components
- **Structure analysis**: grid detection, table recovery
- **Before/after comparisons** for each processing step

## Development Guidelines

### Code Organization
- **Processors**: Each processor handles a specific image processing task
- **Unified Logic**: Stage processors contain complete processing workflows  
- **Configuration**: Single config file per stage with all parameters
- **Debug Support**: All processors support debug image generation

### Adding New Processors
1. Inherit from `BaseProcessor` in `processors/base.py`
2. Implement `process()` method with proper validation
3. Add debug image saving using `self.save_debug_image()`
4. Add processor to appropriate stage processor
5. Add configuration parameters to relevant config file

### Performance Optimization

- Use `parallel_processing: true` for batch processing
- Adjust `max_workers` based on CPU cores (default: 6)
- Enable `memory_mode: true` to avoid disk I/O overhead  
- Tune preprocessing parameters for your specific image quality
- Use search regions to limit processing to relevant areas

## Dependencies

Core dependencies (see requirements.txt):
- opencv-python >= 4.5.0
- numpy >= 1.20.0
- Pillow >= 8.0.0
- scikit-image >= 0.19.0

Development tools:
- pytest for testing
- black for formatting (line length: 88)
- flake8 for linting
- mypy for type checking

## Configuration Management

### Single Config Approach
Each stage now uses a single configuration file containing all parameters:
- No need for separate preprocessing configs
- All `line_detection_*` parameters included in main config
- Easy to toggle preprocessing features on/off
- Consistent parameter naming across stages

### Parameter Tuning
Adjust preprocessing parameters based on image quality:
- **Poor quality/faded**: Increase stroke enhancement, adjust binarization
- **High quality/clean**: Disable preprocessing or use minimal settings
- **Skewed tables**: Increase `skew_tolerance` 
- **Noisy images**: Enable denoising, adjust adaptive parameters