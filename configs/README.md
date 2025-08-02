# Configuration Guide

This directory contains default configuration files for the simplified OCR pipeline.

## Configuration Files

### stage1_default.json

Default configuration for Stage 1 (Initial Processing). Key parameters:

- **page_splitting**: Controls how double-page scans are split
- **deskewing**: Parameters for correcting image rotation
- **margin_removal**: Settings for removing document margins and background
- **line_detection**: Connected components method for table structure detection

### stage2_default.json

Default configuration for Stage 2 (Refinement Processing). Key parameters:

- **deskewing**: Fine-tuning rotation correction on cropped tables
- **line_detection**: Refined parameters for precise table structure detection
- **margin_removal**: Disabled (images already processed in Stage 1)

## Usage

### From Python Code

```python
import json
from pathlib import Path
from src.ocr_pipeline.config import Config

# Load from JSON
with open("configs/stage1_default.json") as f:
    config_data = json.load(f)

# Create config object with overrides
config = Config(
    input_dir=Path(config_data["input_dir"]),
    verbose=config_data["verbose"],
    threshold=config_data["threshold"]
    # ... other parameters
)
```

### From Command Line

```bash
# Use default configurations
python scripts/run_stage1.py data/input/ --verbose

# Custom output directory
python scripts/run_stage1.py data/input/ -o custom_output/stage1/

# With debug mode
python scripts/run_stage1.py data/input/ --debug --verbose
```

## Configuration Parameters

### Common Parameters

- `input_dir`: Input directory path
- `output_dir`: Output directory path
- `debug_dir`: Debug output directory (optional)
- `verbose`: Enable detailed logging
- `save_debug_images`: Save intermediate images for debugging

### Page Splitting Parameters

- `gutter_search_start`: Start position for gutter detection (0.0-1.0)
- `gutter_search_end`: End position for gutter detection (0.0-1.0)
- `min_gutter_width`: Minimum gutter width in pixels

### Deskewing Parameters

- `angle_range`: Maximum angle range to search (degrees)
- `angle_step`: Angle increment for search (degrees)
- `min_angle_correction`: Minimum angle to apply correction (degrees)

### Margin Removal Parameters

- `enable_margin_removal`: Enable margin removal preprocessing
- `black_threshold`: Threshold for identifying background pixels
- `content_threshold`: Threshold for identifying content pixels
- `morph_kernel_size`: Morphological operation kernel size

### Line Detection Parameters

- `threshold`: Binary threshold for line detection
- `horizontal_kernel_size`: Kernel size for horizontal line detection
- `vertical_kernel_size`: Kernel size for vertical line detection
- `alignment_threshold`: Maximum pixel deviation for line alignment

### Stage-Specific Differences

**Stage 1 (Initial Processing)**:
- Full margin removal enabled for content area detection
- Connected components method for robust line detection
- Processing of raw scanned document images

**Stage 2 (Refinement)**:
- Margin removal disabled (images already processed)
- Refined line detection parameters for cropped tables
- Focus on precision over robustness

## Customization

Create custom configuration files by copying and modifying the defaults:

```bash
cp configs/stage1_default.json configs/my_custom_config.json
# Edit my_custom_config.json as needed
```

Then use your custom configuration:

```bash
python scripts/run_stage1.py data/input/ --config configs/my_custom_config.json
```