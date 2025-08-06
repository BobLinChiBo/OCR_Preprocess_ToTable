# Enabling/Disabling Pipeline Steps

The OCR pipeline supports enabling or disabling certain processing steps through configuration files. This allows you to customize the pipeline to your specific needs and skip unnecessary processing steps.

> **📚 Documentation Navigation**: [← Documentation Index](README.md) | [Configuration Guide](../configs/README.md) | [Parameter Reference](PARAMETER_REFERENCE.md) →

## Important Note

Some steps in the pipeline **cannot be disabled** because they generate JSON files that are required by downstream processes:
- Table Line Detection
- Table Structure Detection  
- Table Recovery (Stage 2 only)

These steps will always run to ensure the pipeline functions correctly.

## Stage 1 Optional Processing Steps

Stage 1 supports the following toggleable steps:

1. **Mark Removal** (`mark_removal.enable`)
   - Removes watermarks, stamps, and other artifacts from the image
   - Default: `true`
   - When disabled: Uses original image

2. **Margin Removal** (`margin_removal.enable`)
   - Removes black margins and borders from the image
   - Default: `true`
   - When disabled: Uses previous step's output (mark-removed or original)

3. **Page Splitting** (`page_splitting.enable`)
   - Splits double-page scans into individual pages
   - Default: `true`
   - When disabled: Processes the entire image as a single page

4. **Deskewing** (`deskewing.enable`)
   - Corrects image rotation/skew
   - Default: `true`
   - When disabled: Uses previous step's output

5. **Table Cropping** (`table_detection.enable_table_cropping`)
   - Crops the image to table boundaries
   - Default: `true`
   - When disabled: Uses deskewed image
   - Note: Requires table structure to be detected

## Stage 2 Optional Processing Steps

Stage 2 supports the following toggleable steps:

1. **Deskewing** (`deskewing.enable`)
   - Fine-tunes rotation correction
   - Default: `true`
   - When disabled: Uses input image from Stage 1

2. **Vertical Strip Cutting** (`vertical_strip_cutting.enable`)
   - Cuts tables into vertical strips (columns)
   - Default: `true`
   - When disabled: Skips column extraction
   - Note: Requires table structure to be detected

## Example Configurations

### Minimal Processing (Stage 1)
```json
{
  "mark_removal": {
    "enable": false
  },
  "margin_removal": {
    "enable": false
  },
  "page_splitting": {
    "enable": false
  },
  "deskewing": {
    "enable": true
  },
  "table_detection": {
    "enable_table_cropping": false
  }
}
```

### Single Page Processing (No Splitting)
```json
{
  "page_splitting": {
    "enable": false
  }
}
```

### Minimal Stage 2 Processing
```json
{
  "deskewing": {
    "enable": false
  },
  "vertical_strip_cutting": {
    "enable": false
  }
}
```

## Image Flow

When steps are disabled, the pipeline uses intelligent fallback to ensure continuity:

### Stage 1 Image Flow:
1. Original image →
2. Mark removal (or original if disabled) →
3. Margin removal (or previous if disabled) →
4. Page splitting (or full image if disabled) →
5. Deskewing (or previous if disabled) →
6. Table detection (always runs) →
7. Table cropping (or deskewed if disabled)

### Stage 2 Image Flow:
1. Input from Stage 1 →
2. Deskewing (or input if disabled) →
3. Table detection & recovery (always runs) →
4. Vertical strip cutting (optional)

## Usage

1. Create or modify a configuration JSON file
2. Set the `enable` flag to `false` for any steps you want to skip
3. Run the pipeline with your custom configuration:

```bash
python -m ocr_pipeline.pipeline --config configs/my_custom_config.json
```

The pipeline will skip disabled steps and log messages when in verbose mode to indicate which steps were skipped.

---

**Navigation**: [← Documentation Index](README.md) | [Configuration Guide](../configs/README.md) | [Parameter Reference](PARAMETER_REFERENCE.md) | [Quick Start Guide](QUICK_START.md) →