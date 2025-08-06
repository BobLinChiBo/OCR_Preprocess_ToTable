# Debug Mode Implementation Summary

This document summarizes the debug mode implementation and clarification of visualization tools in the OCR Table Extraction Pipeline.

## Changes Made

### 1. Documentation
- Created comprehensive `DEBUG_MODE_GUIDE.md` explaining:
  - Difference between pipeline debug mode and visualization tools
  - When to use each tool
  - Debug output reference
  - Troubleshooting workflows
  - Directory structure

### 2. Pipeline Debug Mode Implementation
- Added debug functionality to `BaseProcessor` class:
  - `save_debug_image()` - Store debug images during processing
  - `get_debug_images()` - Retrieve stored debug images
  - `save_debug_images_to_dir()` - Save all debug images to directory
  
- Updated `TableDetectionProcessor` to save debug images:
  - `binary_threshold.png` - Binary image after thresholding
  - `horizontal_morph.png` - Horizontal morphological operations
  - `vertical_morph.png` - Vertical morphological operations
  - `connected_components.png` - Color-coded connected components
  - `filtered_lines.png` - Final filtered lines

- Modified `TwoStageOCRPipeline` to:
  - Initialize processor instances for debug mode
  - Use processors when `save_debug_images=True`
  - Save debug images to configured `debug_dir`

### 3. Visualization Scripts Updates
- Removed redundant `--save-debug` flag from visualization scripts
- Debug outputs are now always saved in visualization tools
- Updated help text to clarify that these are analysis tools

### 4. Configuration
- Debug mode controlled by `save_debug_images` flag in config
- Debug outputs saved to directories specified in `debug_dir`:
  - Stage 1: `data/debug/stage1_debug/{timestamp}_{processor_name}/`
  - Stage 2: `data/debug/stage2_debug/{timestamp}_{processor_name}/`

### 5. Documentation Updates
- Updated README.md to:
  - Explain debug mode usage
  - Reference the new debug guide
  - Clarify visualization tools are always saving debug outputs
- Updated tools/README.md to remove references to `--save-debug`

## Testing

Tested with:
```bash
python scripts/run_complete.py --test-images --debug --stage1-only
```

Successfully created debug images in:
```
data/debug/stage1_debug/2025-08-05_01-56-03_table_detection/
├── Shu_Page_671_Image_0001_full/
│   ├── binary_threshold.png
│   ├── connected_components.png
│   ├── filtered_lines.png
│   ├── horizontal_morph.png
│   └── vertical_morph.png
└── ...
```

## Future Work

The following processors still need debug mode implementation:
- DeskewProcessor - Save edge detection, Hough lines, angle histogram
- MarginRemovalProcessor - Save gradient maps, content density, ROI detection
- PageSplitProcessor - Save gutter search region, vertical projection

## Benefits

1. **Clear Separation**: Pipeline debug mode for production issues vs. visualization tools for analysis
2. **Consistent Interface**: All debug outputs follow same pattern
3. **Better Debugging**: Intermediate steps visible for troubleshooting
4. **Reduced Confusion**: No more overlapping debug flags and modes
5. **Comprehensive Documentation**: Users know exactly what to expect