# API Reference

Complete API documentation for developers using the OCR Table Extraction Pipeline programmatically.

## Table of Contents

- [Core Pipeline Classes](#core-pipeline-classes)
- [Configuration Classes](#configuration-classes)
- [Processor Classes](#processor-classes)
- [Utility Functions](#utility-functions)
- [Usage Examples](#usage-examples)
- [Error Handling](#error-handling)

## Core Pipeline Classes

### `OCRPipeline`

Main single-stage pipeline class for basic OCR preprocessing.

```python
from src.ocr_pipeline import OCRPipeline
from src.ocr_pipeline.config import Stage1Config

# Initialize pipeline
config = Stage1Config.from_json("configs/stage1_default.json")
pipeline = OCRPipeline(config)

# Process single image
result = pipeline.process_image("input.jpg")

# Process directory
results = pipeline.process_directory("input_dir/", "output_dir/")
```

#### Methods

##### `__init__(config: Stage1Config, save_debug_images: bool = False)`

Initialize the pipeline with configuration.

**Parameters:**
- `config` (Stage1Config): Configuration object with processing parameters
- `save_debug_images` (bool): Enable debug image saving

##### `process_image(image_path: str, output_dir: str = "data/output") -> Dict[str, Any]`

Process a single image through the pipeline.

**Parameters:**
- `image_path` (str): Path to input image
- `output_dir` (str): Output directory for results

**Returns:**
- `Dict[str, Any]`: Processing results with paths and metadata

**Example:**
```python
result = pipeline.process_image(
    "document.jpg",
    output_dir="custom_output/"
)
print(f"Processed image: {result['final_image_path']}")
```

##### `process_directory(input_dir: str, output_dir: str = "data/output") -> List[Dict[str, Any]]`

Process all images in a directory.

**Parameters:**
- `input_dir` (str): Directory containing input images
- `output_dir` (str): Output directory for results

**Returns:**
- `List[Dict[str, Any]]`: List of processing results for each image

### `TwoStageOCRPipeline`

Advanced two-stage pipeline for high-quality table extraction.

```python
from src.ocr_pipeline import TwoStageOCRPipeline
from src.ocr_pipeline.config import Stage1Config, Stage2Config

# Initialize with both stage configurations
stage1_config = Stage1Config.from_json("configs/stage1_default.json")
stage2_config = Stage2Config.from_json("configs/stage2_default.json")

pipeline = TwoStageOCRPipeline(stage1_config, stage2_config)

# Process with both stages
result = pipeline.process_complete("input.jpg")
```

#### Methods

##### `__init__(stage1_config: Stage1Config, stage2_config: Stage2Config, save_debug_images: bool = False)`

Initialize two-stage pipeline.

##### `process_complete(image_path: str, output_dir: str = "data/output") -> Dict[str, Any]`

Process image through both stages for optimal results.

##### `process_stage1_only(image_path: str, output_dir: str = "data/output") -> Dict[str, Any]`

Run only Stage 1 processing.

##### `process_stage2_only(input_dir: str = None, output_dir: str = "data/output") -> Dict[str, Any]`

Run only Stage 2 refinement (requires Stage 1 output).

## Configuration Classes

### `Stage1Config`

Configuration for initial processing stage.

```python
from src.ocr_pipeline.config import Stage1Config

# Load from JSON
config = Stage1Config.from_json("configs/stage1_default.json")

# Create programmatically
config = Stage1Config(
    verbose=True,
    angle_range=10,
    threshold=25,
    search_ratio=0.7
)

# Access parameters
print(f"Angle range: {config.angle_range}")
print(f"Table detection threshold: {config.threshold}")
```

#### Key Attributes

##### Page Splitting
- `search_ratio` (float): Fraction of image width to search for gutter (default: 0.5)
- `line_len_frac` (float): Minimum line length as fraction of height (default: 0.3)
- `line_thick` (int): Line detection kernel size (default: 3)
- `peak_thr` (float): Peak threshold for line detection (default: 0.3)

##### Deskewing  
- `angle_range` (int): Maximum rotation angle to detect (default: 5)
- `angle_step` (float): Angle detection precision (default: 0.1)
- `min_angle_correction` (float): Minimum angle to apply correction (default: 0.1)

##### Margin Removal
- `blur_kernel_size` (int): Gaussian blur kernel size (default: 7)
- `black_threshold` (int): Threshold for black pixel detection (default: 50)
- `content_threshold` (int): Threshold for content detection (default: 200)

##### Table Detection
- `threshold` (int): Binary threshold for line detection (default: 40)
- `horizontal_kernel_size` (int): Horizontal line kernel (default: 10)
- `vertical_kernel_size` (int): Vertical line kernel (default: 10)

#### Methods

##### `from_json(json_path: str) -> Stage1Config`

Load configuration from JSON file.

##### `to_json(json_path: str)`

Save configuration to JSON file.

##### `update(**kwargs)`

Update configuration parameters.

```python
config.update(angle_range=15, threshold=25)
```

### `Stage2Config`

Configuration for refinement processing stage.

```python
from src.ocr_pipeline.config import Stage2Config

config = Stage2Config.from_json("configs/stage2_default.json")

# Similar structure to Stage1Config with refinement-specific parameters
```

## Processor Classes

Individual processing components that can be used independently.

### `DeskewProcessor`

Handles image rotation correction.

```python
from src.ocr_pipeline.processor_wrappers import DeskewProcessor

processor = DeskewProcessor(config)
result = processor.process(image, return_analysis=True)

# Access results
corrected_image = result['image']
detected_angle = result['analysis']['detected_angle']
```

### `MarginRemovalProcessor`

Handles margin and border removal.

```python
from src.ocr_pipeline.processor_wrappers import MarginRemovalProcessor

processor = MarginRemovalProcessor(config)
result = processor.process(image, method="inscribed", return_analysis=True)

# Available methods: "inscribed", "gradient", "bounding_box", "aggressive"
```

### `TableLineProcessor`

Handles table structure detection.

```python
from src.ocr_pipeline.processor_wrappers import TableLineProcessor

processor = TableLineProcessor(config)
result = processor.process(image, return_analysis=True)

# Access detected lines
horizontal_lines = result['analysis']['horizontal_lines']
vertical_lines = result['analysis']['vertical_lines']
```

### `PageSplitProcessor`

Handles page splitting for double-page documents.

```python
from src.ocr_pipeline.processor_wrappers import PageSplitProcessor

processor = PageSplitProcessor(config)
result = processor.process(image, return_analysis=True)

# Check if split was detected
if result['analysis']['split_detected']:
    left_page = result['analysis']['left_page']
    right_page = result['analysis']['right_page']
```

## Utility Functions

Core image processing utilities available for direct use.

### Image Processing

```python
from src.ocr_pipeline.utils_optimized import *

# Load and preprocess image
image = load_image("input.jpg")
gray = convert_to_grayscale(image)
binary = apply_threshold(gray, threshold=40)

# Geometric transformations
rotated = rotate_image(image, angle=-2.5)
cropped = crop_image(image, x=100, y=50, width=500, height=400)

# Morphological operations
dilated = dilate_image(binary, kernel_size=3, iterations=2)
eroded = erode_image(binary, kernel_size=3, iterations=1)
```

### Line Detection

```python
from src.ocr_pipeline.utils_optimized import detect_table_lines

# Detect table structure
lines_data = detect_table_lines(
    image,
    threshold=40,
    horizontal_kernel_size=10,
    vertical_kernel_size=10
)

horizontal_lines = lines_data['horizontal_lines']
vertical_lines = lines_data['vertical_lines']
```

### File Operations

```python
from src.ocr_pipeline.utils_optimized import save_image, ensure_directory

# Save processed images
save_image(processed_image, "output/result.jpg")

# Ensure directory exists
ensure_directory("output/subdirectory/")
```

## Usage Examples

### Basic Single Image Processing

```python
from src.ocr_pipeline import OCRPipeline
from src.ocr_pipeline.config import Stage1Config

def process_document(image_path: str, output_dir: str):
    """Process a single document through the pipeline."""
    
    # Load configuration
    config = Stage1Config.from_json("configs/stage1_default.json")
    
    # Customize for this document type
    config.update(
        angle_range=15,  # Wide angle range
        threshold=25     # Lower threshold for faint lines
    )
    
    # Initialize pipeline
    pipeline = OCRPipeline(config, save_debug_images=False)
    
    # Process image
    result = pipeline.process_image(image_path, output_dir)
    
    return result

# Usage
result = process_document("document.jpg", "output/")
print(f"Final image: {result['final_image_path']}")
```

### Batch Processing with Custom Parameters

```python
import os
from src.ocr_pipeline import TwoStageOCRPipeline
from src.ocr_pipeline.config import Stage1Config, Stage2Config

def batch_process_tables(input_dir: str, output_dir: str, document_type: str = "academic"):
    """Process multiple documents with type-specific parameters."""
    
    # Load base configurations
    stage1_config = Stage1Config.from_json("configs/stage1_default.json")
    stage2_config = Stage2Config.from_json("configs/stage2_default.json")
    
    # Customize based on document type
    if document_type == "academic":
        stage1_config.update(
            angle_range=3,      # Conservative for academic papers
            threshold=35,       # Higher threshold for clean papers
            search_ratio=0.3    # Narrow search for centered binding
        )
    elif document_type == "historical":
        stage1_config.update(
            angle_range=20,     # Wide range for old documents
            threshold=20,       # Lower for faded text
            search_ratio=0.8    # Wide search for varied layouts
        )
    
    # Initialize pipeline
    pipeline = TwoStageOCRPipeline(stage1_config, stage2_config)
    
    # Process all images
    results = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.png', '.tiff')):
            image_path = os.path.join(input_dir, filename)
            result = pipeline.process_complete(image_path, output_dir)
            results.append(result)
    
    return results

# Usage
results = batch_process_tables("documents/", "output/", "academic")
print(f"Processed {len(results)} documents")
```

### Using Individual Processors

```python
from src.ocr_pipeline.processor_wrappers import *
from src.ocr_pipeline.config import Stage1Config
from src.ocr_pipeline.utils_optimized import load_image

def custom_processing_workflow(image_path: str):
    """Custom workflow using individual processors."""
    
    config = Stage1Config.from_json("configs/stage1_default.json")
    image = load_image(image_path)
    
    # Step 1: Remove margins
    margin_processor = MarginRemovalProcessor(config)
    margin_result = margin_processor.process(image, method="inscribed")
    cleaned_image = margin_result['image']
    
    # Step 2: Correct rotation
    deskew_processor = DeskewProcessor(config)
    deskew_result = deskew_processor.process(cleaned_image, return_analysis=True)
    straight_image = deskew_result['image']
    detected_angle = deskew_result['analysis']['detected_angle']
    
    # Step 3: Detect table structure
    table_processor = TableLineProcessor(config)
    table_result = table_processor.process(straight_image, return_analysis=True)
    
    return {
        'final_image': table_result['image'],
        'detected_angle': detected_angle,
        'table_lines': table_result['analysis'],
        'processing_steps': ['margin_removal', 'deskew', 'table_detection']
    }

# Usage
result = custom_processing_workflow("document.jpg")
print(f"Detected rotation: {result['detected_angle']} degrees")
```

### Configuration Management

```python
from src.ocr_pipeline.config import Stage1Config

def create_custom_configurations():
    """Create and manage custom configurations for different document types."""
    
    # Base configuration
    base_config = Stage1Config()
    
    # Academic papers configuration
    academic_config = Stage1Config(
        verbose=True,
        angle_range=5,
        min_angle_correction=0.2,
        threshold=40,
        search_ratio=0.3,
        black_threshold=60
    )
    academic_config.to_json("configs/academic_papers.json")
    
    # Historical documents configuration
    historical_config = Stage1Config(
        verbose=True,
        angle_range=20,
        min_angle_correction=0.05,
        threshold=20,
        search_ratio=0.8,
        black_threshold=30
    )
    historical_config.to_json("configs/historical_docs.json")
    
    # Quick processing configuration
    quick_config = Stage1Config(
        verbose=False,
        angle_range=3,
        angle_step=0.5,  # Faster but less precise
        threshold=50,    # Higher threshold for speed
        horizontal_kernel_size=15,
        vertical_kernel_size=15
    )
    quick_config.to_json("configs/quick_processing.json")

# Create configurations
create_custom_configurations()

# Load and use custom configuration
config = Stage1Config.from_json("configs/academic_papers.json")
```

## Error Handling

### Exception Types

The pipeline defines custom exceptions for better error handling:

```python
from src.ocr_pipeline.exceptions import (
    OCRPipelineError,
    ImageProcessingError,
    ConfigurationError,
    FileNotFoundError
)

def robust_processing(image_path: str):
    """Example of robust error handling."""
    
    try:
        config = Stage1Config.from_json("configs/stage1_default.json")
        pipeline = OCRPipeline(config)
        result = pipeline.process_image(image_path)
        return result
        
    except FileNotFoundError as e:
        print(f"Input file not found: {e}")
        return None
        
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        # Fall back to default configuration
        config = Stage1Config()
        pipeline = OCRPipeline(config)
        return pipeline.process_image(image_path)
        
    except ImageProcessingError as e:
        print(f"Image processing failed: {e}")
        # Could implement retry logic or alternative processing
        return None
        
    except OCRPipelineError as e:
        print(f"Pipeline error: {e}")
        return None
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

### Validation

```python
def validate_processing_result(result: Dict[str, Any]) -> bool:
    """Validate that processing completed successfully."""
    
    required_keys = ['final_image_path', 'processing_steps', 'success']
    
    if not all(key in result for key in required_keys):
        return False
        
    if not result.get('success', False):
        return False
        
    if not os.path.exists(result.get('final_image_path', '')):
        return False
        
    return True
```

## Advanced Usage

### Custom Processor Development

```python
from src.ocr_pipeline.processors.base import BaseProcessor

class CustomProcessor(BaseProcessor):
    """Example custom processor."""
    
    def __init__(self, config):
        super().__init__(config)
        self.processor_name = "custom_processing"
    
    def process_image(self, image, **kwargs):
        """Implement custom processing logic."""
        
        # Add your custom processing here
        processed_image = self.apply_custom_filter(image)
        
        # Save debug images if enabled
        if self.config.save_debug_images:
            self.save_debug_image("input", image)
            self.save_debug_image("output", processed_image)
        
        return processed_image
    
    def apply_custom_filter(self, image):
        """Custom image processing logic."""
        # Implement your algorithm here
        return image

# Use custom processor
config = Stage1Config()
processor = CustomProcessor(config)
result = processor.process_image(image)
```

### Pipeline Extension

```python
class ExtendedOCRPipeline(OCRPipeline):
    """Extended pipeline with additional processing steps."""
    
    def __init__(self, config, custom_processors=None):
        super().__init__(config)
        self.custom_processors = custom_processors or []
    
    def process_image(self, image_path: str, output_dir: str = "data/output"):
        """Extended processing with custom steps."""
        
        # Run standard pipeline
        result = super().process_image(image_path, output_dir)
        
        # Apply custom processors
        image = load_image(result['final_image_path'])
        for processor in self.custom_processors:
            image = processor.process_image(image)
        
        # Save final result
        final_path = os.path.join(output_dir, "final_custom.jpg")
        save_image(image, final_path)
        result['final_custom_path'] = final_path
        
        return result
```

## Performance Optimization

### Memory Management

```python
import gc
from src.ocr_pipeline.utils_optimized import optimize_memory

def memory_efficient_batch_processing(image_paths: List[str]):
    """Process large batches with memory optimization."""
    
    config = Stage1Config()
    pipeline = OCRPipeline(config)
    
    results = []
    for i, image_path in enumerate(image_paths):
        
        # Process image
        result = pipeline.process_image(image_path)
        results.append(result)
        
        # Periodic memory cleanup
        if i % 10 == 0:
            gc.collect()
            optimize_memory()
    
    return results
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

def parallel_batch_processing(image_paths: List[str], max_workers: int = 4):
    """Process images in parallel."""
    
    def process_single(image_path: str, config: Stage1Config):
        pipeline = OCRPipeline(config)
        return pipeline.process_image(image_path)
    
    config = Stage1Config()
    process_func = partial(process_single, config=config)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_func, image_paths))
    
    return results
```

---

**Navigation**: [← Quick Start](QUICK_START.md) | [Documentation Index](README.md) | [Implementation Summary →](IMPLEMENTATION_SUMMARY.md)