# Examples

This directory contains usage examples for the OCR table extraction pipeline.

## Basic Usage

### basic_usage.py

Demonstrates the fundamental usage of both single-stage and two-stage pipelines.

```bash
python examples/basic_usage.py
```

### Running Examples

```bash
# From project root
cd OCR_Preprocess_ToTable

# Run basic example
python examples/basic_usage.py

# Run with custom data
python examples/basic_usage.py --input data/input/custom_images/
```

## Example Scripts

Each example script demonstrates different aspects of the pipeline:

- **basic_usage.py**: Simple pipeline usage with default settings
- **advanced_config.py**: Custom configuration examples (planned)
- **batch_processing.py**: Processing multiple documents (planned)
- **custom_visualization.py**: Using visualization tools (planned)

## Input Data

Place your test images in `data/input/` or use the provided test images in `data/input/test_images/`.

## Output

Examples will create output in the `data/output/examples/` directory to avoid conflicts with production data.