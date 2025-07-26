# Configuration Guide

This directory contains default configuration files for the OCR pipeline stages.

## Configuration Files

### stage1_default.json

Default configuration for Stage 1 (Initial Processing). Key sections:

- **page_splitting**: Controls how double-page scans are split
- **deskewing**: Parameters for correcting image rotation
- **line_detection**: Settings for detecting table lines
- **roi_detection**: Region of Interest detection parameters
- **roi_margins**: Page-specific margin settings

### stage2_default.json

Default configuration for Stage 2 (Refinement Processing). Key sections:

- **deskewing**: Fine-tuning rotation correction
- **line_detection**: Precise line detection parameters
- **roi_detection**: Disabled (images already cropped)
- **table_fitting**: Parameters for final table structure fitting

## Usage

### From Python Code

```python
import json
from pathlib import Path
from src.ocr_pipeline.config import Stage1Config, Stage2Config

# Load from JSON
with open("configs/stage1_default.json") as f:
    config_data = json.load(f)

# Create config object with overrides
config = Stage1Config(
    input_dir=Path(config_data["input_dir"]),
    verbose=config_data["verbose"],
    # ... other parameters
)
```

### From Command Line

```bash
# Use default configurations
python scripts/run_stage1.py data/input/ --verbose

# Override specific parameters
python scripts/run_stage1.py data/input/ --angle-range 15 --min-line-length 50

# Custom output directory
python scripts/run_stage1.py data/input/ -o custom_output/stage1/
```

## Configuration Parameters

### Common Parameters

- `input_dir`: Input directory path
- `output_dir`: Output directory path
- `debug_dir`: Debug output directory (optional)
- `verbose`: Enable detailed logging
- `save_debug_images`: Save intermediate images for debugging

### Deskewing Parameters

- `angle_range`: Maximum angle range to search (degrees)
- `angle_step`: Angle increment for search (degrees)
- `min_angle_correction`: Minimum angle to apply correction (degrees)

### Line Detection Parameters

- `min_line_length`: Minimum line length to detect (pixels)
- `max_line_gap`: Maximum gap to bridge in lines (pixels)

### ROI Detection Parameters

- `enable_roi_detection`: Enable region of interest detection
- `gabor_*`: Gabor filter parameters for edge detection
- `roi_*`: ROI boundary detection thresholds

### Stage-Specific Differences

**Stage 1 (Initial Processing)**:
- More permissive line detection (longer gaps allowed)
- ROI detection enabled for content area cropping
- Generous margin settings for different page types

**Stage 2 (Refinement)**:
- Stricter line detection (shorter gaps)
- ROI detection disabled (images already cropped)
- Minimal margins (images already processed)
- Additional table fitting parameters

## Customization

Create custom configuration files by copying and modifying the defaults:

```bash
cp configs/stage1_default.json configs/my_stage1_config.json
# Edit my_stage1_config.json as needed
```

Then load your custom configuration in your processing scripts.