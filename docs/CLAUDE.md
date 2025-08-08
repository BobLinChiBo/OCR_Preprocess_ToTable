# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an OCR preprocessing and table extraction pipeline designed to process scanned document images, particularly those containing tables. The pipeline operates in two stages:
- **Stage 1**: Image preprocessing (mark removal, margin removal, page splitting, deskewing, initial table detection)
- **Stage 2**: Table refinement (fine deskewing, table line detection, structure recovery, vertical strip cutting)

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
python scripts/run_complete.py
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
   - `run_pipeline.py`: Main CLI entry point
   - `run_parallel.py`: Parallel processing variant
   - `run_stage1.py`: Stage 1 only
   - `run_stage2.py`: Stage 2 only
   - `run_complete.py`: Complete pipeline execution

2. **Stage Processors** (`src/ocr_pipeline/`):
   - `stage1_processor.py`: Handles all Stage 1 operations
   - `stage2_processor.py`: Handles all Stage 2 operations
   - `pipeline.py`: Main pipeline orchestration with memory/parallel modes

3. **Image Processors** (`src/ocr_pipeline/processors/`):
   - `base.py`: Base processor class
   - `mark_removal.py`: Watermark, stamp, and artifact removal
   - `margin_removal.py`: Paper margin detection and removal
   - `page_split.py`: Two-page spread splitting
   - `deskew.py`: Image deskewing using Radon transform
   - `table_detection.py`: Table line and structure detection
   - `table_recovery.py`: Table structure recovery
   - `tag_removal.py`: Header/footer tag removal
   - `vertical_strip_cutter.py`: Column extraction
   - `binarize.py`: Image binarization

4. **Configuration** (`configs/`):
   - `stage1_default.json`: Default Stage 1 parameters
   - `stage2_default.json`: Default Stage 2 parameters
   - Configuration can be overridden via CLI arguments

### Processing Modes

The pipeline supports three execution modes:

1. **Regular Mode**: Process images sequentially with disk I/O
2. **Parallel Mode**: Process multiple images concurrently using multiprocessing
3. **Memory Mode**: Keep intermediate results in memory (faster but uses more RAM)

### Data Flow

```
Input Images → Stage 1 Processing → Cropped Tables → Stage 2 Processing → Final Output
```

Stage 1 outputs:
- `02_margin_removed/`: Images with margins removed
- `03_deskewed/`: Deskewed images  
- `04_split_pages/`: Split left/right pages
- `04b_tags_removed/`: Page tags removed
- `05_table_lines/`: Detected table lines
- `06_table_structure/`: Table structure analysis
- `07_border_cropped/`: Cropped table regions

Stage 2 outputs:
- `01_refined_deskewed/`: Fine-tuned deskewing
- `02_table_lines/`: Refined table line detection
- `03_table_structure/`: Final table structure
- `04_table_recovered/`: Recovered table data
- `05_vertical_strips/`: Individual column strips

## Important Configuration Parameters

### Stage 1 Key Settings
- `mark_removal.enable`: Enable/disable watermark and stamp removal
- `mark_removal.dilate_iter`: Dilation iterations for text protection (default: 2)
- `mark_removal.protect_table_lines`: Preserve table lines during mark removal
- `margin_removal.enable`: Enable/disable margin removal
- `page_splitting.enable`: Enable/disable two-page splitting
- `deskewing.enable`: Enable/disable initial deskewing
- `tag_removal.enable`: Enable/disable header/footer tag removal
- `table_detection.enable_table_cropping`: Enable/disable table region cropping
- `optimization.parallel_processing`: Enable parallel processing
- `optimization.memory_mode`: Use memory-efficient mode

### Stage 2 Key Settings  
- `deskewing.enable`: Enable/disable fine deskewing
- `vertical_strip_cutting.enable`: Enable/disable column extraction
- `binarization.enable`: Enable/disable image binarization
- `table_recovery.coverage_ratio`: Minimum coverage for table recovery

## Debug Mode

Enable debug output by setting in config files:
```json
{
  "save_debug_images": true,
  "debug_dir": "data/debug"
}
```

Debug images show intermediate processing steps for troubleshooting.

## Performance Optimization

- Use `parallel_processing: true` for batch processing
- Adjust `max_workers` based on CPU cores (default: CPU count - 1)
- Enable `memory_mode: true` to avoid disk I/O overhead
- Set appropriate `batch_size` for memory constraints

## Dependencies

Core dependencies (see requirements.txt):
- opencv-python >= 4.5.0
- numpy >= 1.20.0
- Pillow >= 8.0.0
- deskew >= 1.5.0
- scikit-image >= 0.19.0

Development tools:
- pytest for testing
- black for formatting (line length: 88)
- flake8 for linting
- mypy for type checking