# V2 Visualization Scripts Migration Guide

## Overview

The V2 visualization scripts represent a major architectural improvement that centralizes parameter handling and reduces maintenance burden. This guide helps you migrate from V1 to V2 scripts.

## Why V2?

### The Problem with V1

In V1, each visualization script directly called utility functions from `utils.py`. This meant:
- When a function signature in `utils.py` changed, ALL visualization scripts needed updates
- Parameter handling was duplicated across scripts
- Configuration loading was inconsistent
- Maintenance was error-prone and time-consuming

### The V2 Solution

V2 introduces a processor wrapper architecture:
- **Processor Wrappers** (`processor_wrappers.py`): Centralized parameter mapping
- **Config Utils** (`config_utils.py`): Unified configuration loading
- **Single Update Point**: Changes to `utils.py` only require updates in processor wrappers

## Architecture Comparison

### V1 Architecture
```
visualize_*.py → utils.py functions (direct calls)
```

### V2 Architecture
```
visualize_*_v2.py → ProcessorWrapper → utils.py functions
                 ↓
           config_utils.py
```

## Migration Examples

### Page Split

**V1:**
```bash
python tools/visualize_page_split.py image.jpg
```

**V2:**
```bash
python tools/visualize_page_split_v2.py image.jpg
# Or via runner:
python tools/run_visualizations.py page-split image.jpg --use-v2
```

### Margin Removal

**V1 (3 separate scripts):**
```bash
# Aggressive method
python tools/visualize_margin_removal.py image.jpg

# Fast/optimized version
python tools/visualize_margin_removal_fast.py image.jpg

# Bounding box method
python tools/visualize_margin_removal_bbox.py image.jpg
```

**V2 (unified script):**
```bash
# Aggressive method (default)
python tools/visualize_margin_removal_v2.py image.jpg

# Optimized version
python tools/visualize_margin_removal_v2.py image.jpg --use-optimized

# Bounding box method
python tools/visualize_margin_removal_v2.py image.jpg --method bounding_box

# Compare all methods
python tools/visualize_margin_removal_v2.py image.jpg --compare
```

### Deskew

**V1:**
```bash
python tools/visualize_deskew.py image.jpg --angle-range 30 --angle-step 1.0
```

**V2:**
```bash
python tools/visualize_deskew_v2.py image.jpg --angle-range "(-30, 30)" --angle-step 1.0
```

### Table Lines

**V1:**
```bash
python tools/visualize_table_lines.py image.jpg --save-debug
```

**V2:**
```bash
python tools/visualize_table_lines_v2.py image.jpg --save-debug
```

### Table Crop

**V1:**
```bash
python tools/visualize_table_crop.py image.jpg
```

**V2:**
```bash
python tools/visualize_table_crop_v2.py image.jpg
```

## Pipeline Mode

**V1:**
```bash
python tools/run_visualizations.py page-split margin-removal deskew --pipeline image.jpg
```

**V2:**
```bash
python tools/run_visualizations.py page-split margin-removal deskew --pipeline --use-v2 image.jpg
```

## Configuration Handling

### V1 Approach
Each script had its own configuration loading logic:
```python
# Inconsistent across scripts
if config_path:
    config = Stage1Config.from_json(config_path)
else:
    config = Stage1Config()
```

### V2 Approach
Unified configuration loading:
```python
from config_utils import load_config
config, config_source = load_config(args, Stage1Config, 'processor_type')
```

## Parameter Mapping

### V1 Approach
Direct parameter passing with potential mismatches:
```python
result = utils.remove_margin_aggressive(
    image, 
    blur_kernel_size=args.blur_kernel_size,
    # ... many parameters, easy to miss or mismatch
)
```

### V2 Approach
Centralized parameter mapping in processor:
```python
processor = MarginRemovalProcessor(config)
result = processor.process(image, method="aggressive", return_analysis=True)
```

## Benefits of V2

1. **Maintainability**: Single point of update for parameter changes
2. **Consistency**: All scripts follow the same pattern
3. **Flexibility**: Easy to add new parameters or methods
4. **Type Safety**: Better parameter validation
5. **Future-Proof**: Easier to extend functionality

## Deprecation Timeline

1. **Current**: V1 scripts show deprecation warnings
2. **Next Release**: V2 becomes default (--use-v1 for old behavior)
3. **Future Release**: V1 scripts removed

## Quick Reference

| Task | V1 Command | V2 Command |
|------|------------|------------|
| Page split | `visualize_page_split.py` | `visualize_page_split_v2.py` |
| Margin removal (aggressive) | `visualize_margin_removal.py` | `visualize_margin_removal_v2.py` |
| Margin removal (fast) | `visualize_margin_removal_fast.py` | `visualize_margin_removal_v2.py --use-optimized` |
| Margin removal (bbox) | `visualize_margin_removal_bbox.py` | `visualize_margin_removal_v2.py --method bounding_box` |
| Deskew | `visualize_deskew.py` | `visualize_deskew_v2.py` |
| Table lines | `visualize_table_lines.py` | `visualize_table_lines_v2.py` |
| Table crop | `visualize_table_crop.py` | `visualize_table_crop_v2.py` |
| All with runner | `run_visualizations.py all` | `run_visualizations.py all --use-v2` |

## Troubleshooting

### Issue: Scripts not found
Make sure you're using the correct script names with `_v2` suffix.

### Issue: Different parameter names
Check `--help` for each v2 script as some parameter names have been standardized.

### Issue: Missing features
All V1 features are available in V2. For margin removal variants, use appropriate flags instead of separate scripts.

## For Developers

### Adding New Processors

1. Create processor class in `processor_wrappers.py`:
```python
class NewProcessor(BaseProcessor):
    def process(self, image, **kwargs):
        # Map parameters and call utils function
        pass
```

2. Add default parameters to `config_utils.py`
3. Create visualization script following v2 pattern

### Updating Existing Processors

When `utils.py` functions change:
1. Update only the processor wrapper
2. No changes needed in visualization scripts
3. Test with existing scripts to ensure compatibility