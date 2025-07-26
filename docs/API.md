# API Documentation

## Core Classes

### OCRPipeline

Single-stage pipeline for basic OCR table extraction.

```python
from src.ocr_pipeline.pipeline import OCRPipeline
from src.ocr_pipeline.config import Config

config = Config(
    input_dir=Path("data/input"),
    output_dir=Path("data/output"),
    verbose=True
)

pipeline = OCRPipeline(config)
results = pipeline.process_directory()
```

### TwoStageOCRPipeline

Professional two-stage pipeline with initial processing and refinement.

```python
from src.ocr_pipeline.pipeline import TwoStageOCRPipeline
from src.ocr_pipeline.config import Stage1Config, Stage2Config

# Configure both stages
stage1_config = Stage1Config(
    input_dir=Path("data/input"),
    output_dir=Path("data/output/stage1"),
    verbose=True
)

stage2_config = Stage2Config(
    input_dir=Path("data/output/stage1/05_cropped_tables"),
    output_dir=Path("data/output/stage2"),
    verbose=True
)

# Run complete pipeline
pipeline = TwoStageOCRPipeline(stage1_config, stage2_config)
results = pipeline.run_complete_pipeline()

# Or run stages individually
stage1_results = pipeline.run_stage1()
stage2_results = pipeline.run_stage2()
```

## Configuration Classes

### Config

Base configuration class for single-stage pipeline.

**Key Parameters:**
- `input_dir`: Input directory with raw images
- `output_dir`: Output directory for results
- `angle_range`: Deskewing angle range (±45° default)
- `min_line_length`: Minimum line length for table detection (100px default)
- `enable_roi_detection`: Enable ROI preprocessing (True default)

### Stage1Config

Configuration for initial processing stage.

**Key Parameters:**
- `angle_range`: Wider angle range for initial deskewing (±10° default)
- `min_line_length`: More permissive line detection (40px default)
- `max_line_gap`: Larger gap tolerance (15px default)
- Output directories: `01_split_pages`, `02_deskewed`, `03_line_detection`, etc.

### Stage2Config

Configuration for refinement stage.

**Key Parameters:**
- `angle_range`: Fine-tuning angle range (±10° default)
- `min_line_length`: Precise line detection (30px default)  
- `max_line_gap`: Tight gap tolerance (5px default)
- `enable_roi_detection`: Disabled (False default, images already cropped)
- Output directories: `01_deskewed`, `02_line_detection`, `03_table_reconstruction`, `04_fitted_tables`

## Utility Functions

All utility functions are available in `src.ocr_pipeline.utils`:

- `get_image_files(directory)`: Get list of image files in directory
- `load_image(path)`: Load image from file
- `save_image(image, path)`: Save image to file
- `split_two_page_image(image, start, end)`: Split double-page scans
- `deskew_image(image, angle_range, step)`: Correct image rotation
- `detect_table_lines(image, min_length, max_gap)`: Find table lines
- `crop_table_region(image, h_lines, v_lines)`: Crop to table area
- `detect_roi_for_image(image, config)`: Detect regions of interest
- `crop_to_roi(image, roi_coords)`: Crop to ROI coordinates