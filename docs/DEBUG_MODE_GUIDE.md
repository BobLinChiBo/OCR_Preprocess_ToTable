# Debug Mode and Visualization Tools Guide

This guide explains the debugging and analysis capabilities of the OCR Table Extraction Pipeline, including the differences between pipeline debug mode and visualization tools.

> **📚 Documentation Navigation**: [← Documentation Index](README.md) | [Parameter Reference](PARAMETER_REFERENCE.md) | [Tools Documentation](../tools/README.md) →

## Table of Contents
- [Overview](#overview)
- [Pipeline Debug Mode](#pipeline-debug-mode)
- [Visualization Tools](#visualization-tools)
- [When to Use Each Tool](#when-to-use-each-tool)
- [Debug Output Reference](#debug-output-reference)
- [Troubleshooting Workflows](#troubleshooting-workflows)
- [Directory Structure](#directory-structure)

## Overview

The OCR pipeline provides two distinct systems for debugging and analysis:

1. **Pipeline Debug Mode**: Saves intermediate processing steps during batch processing
2. **Visualization Tools**: Interactive scripts for analyzing individual images and tuning parameters

### Key Differences

| Feature | Pipeline Debug Mode | Visualization Tools |
|---------|-------------------|-------------------|
| Purpose | Diagnose batch processing issues | Analyze individual images |
| Usage | `--debug` flag during pipeline run | Standalone scripts in `tools/` |
| Output Location | `data/debug/` | `data/output/visualization/` |
| Processing Mode | Batch (multiple images) | Single image focus |
| Interactivity | Automated | Interactive parameter testing |

## Pipeline Debug Mode

### Enabling Debug Mode

```bash
# For complete pipeline
python scripts/run_complete.py --test-images --debug

# For stage-specific runs
python scripts/run_stage1.py --debug
python scripts/run_stage2.py --debug
```

### What Debug Mode Saves

When enabled, debug mode saves intermediate processing images for each major step:

#### Stage 1 Debug Outputs

1. **Page Splitting** (`data/debug/stage1_debug/{timestamp}_page_split/`)
   - `gutter_search_region.png`: Highlighted area being searched for page split
   - `vertical_projection.png`: Intensity projection used for gutter detection
   - `split_line_detection.png`: Final detected split position

2. **Margin Removal** (`data/debug/stage1_debug/{timestamp}_margin_removal/`)
   - `gradient_map.png`: Edge gradient visualization
   - `content_density.png`: Pixel density heatmap
   - `roi_detection.png`: Detected regions of interest
   - `margin_mask.png`: Areas marked for removal

3. **Deskewing** (`data/debug/stage1_debug/{timestamp}_deskew/`)
   - `edge_detection.png`: Canny edge detection result
   - `hough_lines.png`: Lines detected for angle calculation
   - `angle_histogram.png`: Distribution of detected angles
   - `rotation_comparison.png`: Before/after rotation comparison

4. **Table Detection** (`data/debug/stage1_debug/{timestamp}_table_detection/`)
   - `binary_threshold.png`: Binary image after thresholding
   - `horizontal_morph.png`: Result of horizontal morphological operations
   - `vertical_morph.png`: Result of vertical morphological operations
   - `connected_components.png`: Labeled connected components
   - `filtered_lines.png`: Lines after length and aspect ratio filtering

#### Stage 2 Debug Outputs

Similar debug outputs are saved for Stage 2 refinement steps in `data/debug/stage2_debug/`.

### Debug Configuration Options

In your config files (`configs/stage1_default.json`, etc.):

```json
{
  "debug_dir": "data/debug/stage1_debug",
  "save_debug_images": false,
  "debug_image_format": "png",
  "debug_compression_quality": 95,
  "debug_steps": ["all"]  // or specific steps: ["deskew", "table_detection"]
}
```

### Run Information File

Each debug run creates a `run_info.json` file containing:

```json
{
  "timestamp": "2024-01-15_10-30-45",
  "stage": "stage1",
  "input_path": "data/input/test_images",
  "num_images": 26,
  "status": "completed",
  "config": {
    "save_debug_images": true,
    "debug_dir": "data/debug/stage1_debug",
    "verbose": true
  }
}
```

This helps track which debug outputs belong to which pipeline run.

## Visualization Tools

### Purpose

Visualization tools are standalone scripts for:
- Analyzing why specific images fail
- Testing different parameters
- Understanding algorithm behavior
- Comparing processing methods

### Available Tools

#### 1. Page Split Visualization
```bash
python tools/visualize_page_split_v2.py image.jpg
```
Shows gutter detection and page separation analysis.

#### 2. Margin Removal Visualization
```bash
# Compare all methods
python tools/visualize_margin_removal_v2.py image.jpg --compare

# Test specific method
python tools/visualize_margin_removal_v2.py image.jpg --method aggressive
```
Compares different margin removal approaches.

#### 3. Deskew Visualization
```bash
python tools/visualize_deskew_v2.py image.jpg --angle-range 15
```
Shows rotation detection and correction process.

#### 4. Table Line Detection
```bash
python tools/visualize_table_lines_v2.py image.jpg --min-line-length 30
```
Analyzes table structure detection with customizable parameters.

#### 5. Table Crop Visualization
```bash
python tools/visualize_table_crop_v2.py image.jpg
```
Shows final table extraction boundaries.

### Visualization Output Structure

```
data/output/visualization/
├── page_split_v2/
│   ├── {image_name}_page_split_overlay.jpg
│   ├── parameters.json
│   └── debug/
│       └── {image_name}/
│           ├── projection_profile.png
│           └── gutter_detection.png
├── margin_removal_v2/
├── deskew_v2/
├── table_lines_v2/
└── table_crop_v2/
```

## When to Use Each Tool

### Use Pipeline Debug Mode When:
- Processing fails for multiple images in a batch
- You need to understand why the pipeline produces unexpected results
- Investigating systematic issues across your dataset
- Debugging configuration problems
- Tracking down performance bottlenecks

### Use Visualization Tools When:
- A specific image produces poor results
- You need to find optimal parameters for your image type
- Comparing different processing methods
- Creating documentation or reports about processing steps
- Training users on how the algorithms work

## Debug Output Reference

### Reading Debug Images

#### Binary Threshold Images
- **White pixels**: Detected content
- **Black pixels**: Background
- Look for: Incomplete text, merged characters, noise

#### Morphological Operation Results
- **Horizontal morphology**: Should show only horizontal lines
- **Vertical morphology**: Should show only vertical lines
- Look for: Broken lines, false positives from text

#### Edge Detection Images
- **Bright lines**: Strong edges
- **Dark areas**: No edges detected
- Look for: Missing edges, excessive noise

#### Angle Histograms
- **Peak**: Most common angle
- **Spread**: Angle variation
- Look for: Multiple peaks (mixed rotations), wide spread (unclear rotation)

## Troubleshooting Workflows

### Workflow 1: Table Lines Not Detected

1. Run visualization tool on problem image:
   ```bash
   python tools/visualize_table_lines_v2.py problem_image.jpg
   ```

2. Check the output for line counts and detection status

3. If lines are missing, test with different parameters:
   ```bash
   python tools/visualize_table_lines_v2.py problem_image.jpg --threshold 30 --min-line-length 20
   ```

4. Once optimal parameters are found, update your config file

### Workflow 2: Incorrect Page Rotation

1. Run deskew visualization:
   ```bash
   python tools/visualize_deskew_v2.py problem_image.jpg
   ```

2. Check if angle detection is correct in the output

3. Try wider angle range if needed:
   ```bash
   python tools/visualize_deskew_v2.py problem_image.jpg --angle-range 20 --angle-step 0.1
   ```

4. Enable debug mode and check edge detection quality:
   ```bash
   python scripts/run_complete.py problem_image.jpg --debug
   ```

### Workflow 3: Batch Processing Issues

1. Enable debug mode for the full pipeline:
   ```bash
   python scripts/run_complete.py --input-dir problematic_batch/ --debug
   ```

2. Check debug outputs in `data/debug/` for patterns

3. Identify which step is failing across multiple images

4. Use visualization tools to test solutions on representative images

5. Update configuration and re-run pipeline

## Directory Structure

### Production Outputs
```
data/output/
├── stage1/
│   ├── 01_pages/
│   ├── 02_no_margin/
│   ├── 03_deskewed/
│   ├── 04_table_lines/
│   ├── 05_table_structure/
│   └── 06_cropped/
└── stage2/
    └── final_tables/
```

### Debug Outputs
```
data/debug/
├── stage1_debug/
│   └── 2024-01-15_10-30-45_run/  # Single timestamp per pipeline run
│       ├── run_info.json         # Metadata about the run
│       ├── page_split/           # Debug images organized by processor
│       │   ├── image1_full/
│       │   └── image2_full/
│       ├── margin_removal/
│       │   ├── image1_full/
│       │   └── image2_full/
│       ├── deskew/
│       │   ├── image1_full/
│       │   └── image2_full/
│       └── table_detection/
│           ├── image1_full/
│           │   ├── binary_threshold.png
│           │   ├── horizontal_morph.png
│           │   ├── vertical_morph.png
│           │   ├── connected_components.png
│           │   └── filtered_lines.png
│           └── image2_full/
└── stage2_debug/
    └── 2024-01-15_10-30-45_run/
        └── ...

### Visualization Outputs
```
data/output/visualization/
├── page_split_v2/
├── margin_removal_v2/
├── deskew_v2/
├── table_lines_v2/
└── table_crop_v2/
```

## Best Practices

1. **Start with visualization tools** to understand your specific images
2. **Find optimal parameters** using single-image testing
3. **Update configuration** with discovered parameters
4. **Enable debug mode** only when investigating issues
5. **Clean debug directories** periodically (they can grow large)
6. **Document parameter choices** for different image types

## Performance Considerations

- Debug mode increases processing time by ~20-30%
- Debug images can consume significant disk space
- PNG format preserves quality but uses more space than JPG
- Consider using `debug_steps` to limit debug output to specific processors

## FAQ

**Q: Why isn't debug mode saving any images?**
A: Check that `save_debug_images` is `true` in your config file and that the `debug_dir` exists and is writable.

**Q: Can I use visualization tools in a pipeline?**
A: Visualization tools are designed for interactive use. For batch processing, use the main pipeline with debug mode.

**Q: How do I compare parameters across multiple images?**
A: Use the batch mode of visualization tools:
```bash
python tools/run_visualizations.py table-lines --test-images --use-v2
```

**Q: Debug images are too large. How can I reduce their size?**
A: Set `debug_image_format` to `"jpg"` and adjust `debug_compression_quality` (0-100) in your config.

---

**Navigation**: [← Documentation Index](README.md) | [Parameter Reference](PARAMETER_REFERENCE.md) | [Tools Documentation](../tools/README.md) | [Troubleshooting Guide](TROUBLESHOOTING.md) →