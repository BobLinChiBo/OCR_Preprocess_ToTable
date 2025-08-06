# Troubleshooting Guide

Complete guide to diagnosing and solving common issues with the OCR Table Extraction Pipeline.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Processing Issues](#processing-issues)
- [Configuration Issues](#configuration-issues)
- [Performance Issues](#performance-issues)  
- [Output Quality Issues](#output-quality-issues)
- [Debug Mode Issues](#debug-mode-issues)
- [Platform-Specific Issues](#platform-specific-issues)
- [Advanced Debugging](#advanced-debugging)

## Installation Issues

### Python/Package Installation Problems

#### Issue: `ImportError: No module named 'cv2'`

**Cause**: OpenCV not properly installed

**Solutions**:
```bash
# Solution 1: Reinstall OpenCV
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python>=4.5.0

# Solution 2: Try headless version (for servers)
pip install opencv-python-headless>=4.5.0

# Solution 3: Use conda (alternative)
conda install -c conda-forge opencv
```

#### Issue: `ImportError: No module named 'numpy'` or NumPy version conflicts

**Cause**: NumPy installation or version compatibility issues

**Solutions**:
```bash
# Solution 1: Reinstall NumPy with compatible version
pip uninstall numpy
pip install numpy>=1.20.0,<1.25.0

# Solution 2: Install specific compatible versions
pip install numpy==1.21.0 scipy==1.7.0 scikit-image==0.18.3

# Solution 3: Use conda for better dependency management
conda install numpy scipy scikit-image
```

#### Issue: `Microsoft Visual C++ 14.0 is required` (Windows)

**Cause**: Missing C++ build tools for package compilation

**Solutions**:
1. Install Microsoft C++ Build Tools from https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Or install Visual Studio Community with C++ development tools
3. Or use pre-compiled packages: `pip install --only-binary=all opencv-python`

#### Issue: Permission denied during installation

**Cause**: Insufficient permissions or system-wide installation conflicts

**Solutions**:
```bash
# Solution 1: Use virtual environment (recommended)
python -m venv ocr_env
source ocr_env/bin/activate  # Windows: ocr_env\Scripts\activate
pip install -r requirements.txt

# Solution 2: User installation
pip install --user -r requirements.txt

# Solution 3: Fix permissions (Linux/macOS)
sudo chown -R $(whoami) /usr/local/lib/python3.x/site-packages/
```

### Path and Directory Issues

#### Issue: `FileNotFoundError: configs/stage1_default.json`

**Cause**: Working directory not set correctly

**Solutions**:
```bash
# Ensure you're in the project root directory
cd /path/to/OCR_Preprocess_ToTable
ls configs/  # Should show stage1_default.json

# Or use absolute paths
python scripts/run_complete.py --stage1-config /full/path/to/configs/stage1_default.json
```

#### Issue: Data directories not found

**Cause**: Data directories not created or incorrect permissions

**Solutions**:
```bash
# Create required directories
mkdir -p data/input/raw_images
mkdir -p data/input/test_images  
mkdir -p data/output
mkdir -p data/debug

# Fix permissions (Linux/macOS)
chmod -R 755 data/
```

## Processing Issues

### Stage 1 Processing Problems

#### Issue: `RuntimeError: No images found in input directory`

**Cause**: No supported image files in input directory

**Solutions**:
1. **Check supported formats**: `.jpg`, `.jpeg`, `.png`, `.tiff`, `.bmp`
2. **Verify directory contents**:
   ```bash
   ls -la data/input/test_images/
   # Should show image files with supported extensions
   ```
3. **Add test images**: Place sample images in `data/input/test_images/`

#### Issue: Pipeline crashes with `cv2.error: (-215:Assertion failed)`

**Cause**: Usually invalid image data or corrupted image files

**Solutions**:
```bash
# Test individual images
python -c "
import cv2
img = cv2.imread('problematic_image.jpg')
if img is None:
    print('Image could not be loaded')
else:
    print(f'Image shape: {img.shape}')
"

# Convert problematic images
convert problematic_image.jpg -quality 95 fixed_image.jpg  # ImageMagick
# or
python -c "
from PIL import Image
img = Image.open('problematic_image.jpg')
img.save('fixed_image.jpg', quality=95)
"
```

#### Issue: Memory errors with large images

**Cause**: Insufficient RAM for large image processing

**Solutions**:
```bash
# Resize images before processing
python -c "
from PIL import Image
img = Image.open('large_image.jpg')
# Resize to max 3000px width while maintaining aspect ratio
if img.width > 3000:
    ratio = 3000 / img.width
    new_size = (3000, int(img.height * ratio))
    img = img.resize(new_size, Image.LANCZOS)
img.save('resized_image.jpg', quality=95)
"

# Process one image at a time
python scripts/run_complete.py single_image.jpg --verbose

# Use swap space (Linux)
sudo swapon --show  # Check current swap
# Add swap if needed
```

### Stage 2 Processing Problems

#### Issue: Stage 2 fails with "No input files found"

**Cause**: Stage 1 didn't complete successfully or output directory issue

**Solutions**:
```bash
# Check Stage 1 outputs
ls data/output/stage1/07_border_cropped/
# Should contain cropped table images

# Run Stage 1 first
python scripts/run_stage1.py --test-images --verbose

# Then run Stage 2
python scripts/run_stage2.py --verbose
```

#### Issue: Stage 2 produces poor quality results

**Cause**: Usually Stage 1 output quality issues or configuration problems

**Solutions**:
1. **Verify Stage 1 results**:
   ```bash
   # Visual inspection of Stage 1 outputs
   python tools/check_results.py view latest
   ```

2. **Adjust Stage 2 parameters**:
   ```json
   // Edit configs/stage2_default.json
   {
     "threshold": 25,  // Lower for faint lines
     "min_angle_correction": 0.05  // More sensitive rotation
   }
   ```

## Configuration Issues

### JSON Configuration Problems

#### Issue: `JSONDecodeError: Invalid JSON format`

**Cause**: Malformed JSON in configuration files

**Solutions**:
```bash
# Validate JSON syntax
python -m json.tool configs/stage1_default.json

# Common JSON fixes:
# - Add missing commas between items
# - Remove trailing commas
# - Use double quotes, not single quotes
# - Escape backslashes in paths: "C:\\path" or "C:/path"
```

#### Issue: Configuration parameters not taking effect

**Cause**: Configuration not loaded properly or parameter name mismatch

**Solutions**:
```bash
# Test configuration loading
python -c "
from src.ocr_pipeline.config import Stage1Config
config = Stage1Config.from_json('configs/stage1_default.json')
print(f'Threshold: {config.threshold}')
print(f'Angle range: {config.angle_range}')
"

# Use verbose mode to see active parameters
python scripts/run_complete.py --test-images --verbose
```

#### Issue: Custom configuration ignored

**Cause**: Wrong command line arguments or file path issues

**Solutions**:
```bash
# Verify configuration file path
ls -la configs/my_config.json

# Use absolute path if needed
python scripts/run_complete.py --stage1-config /full/path/to/my_config.json

# Check configuration is loaded
python scripts/run_complete.py --stage1-config configs/my_config.json --verbose | grep -i config
```

## Performance Issues

### Slow Processing

#### Issue: Processing takes extremely long

**Cause**: Large images, inefficient parameters, or system resource constraints

**Solutions**:
1. **Resize large images**:
   ```bash
   # Check image dimensions
   identify large_image.jpg  # ImageMagick
   
   # Resize if over 3000px width
   convert large_image.jpg -resize "3000x3000>" resized_image.jpg
   ```

2. **Optimize parameters for speed**:
   ```json
   // Fast processing config
   {
     "angle_step": 0.5,        // Larger step = faster
     "horizontal_kernel_size": 15,  // Larger kernel = faster
     "save_debug_images": false     // Disable debug output
   }
   ```

3. **Monitor system resources**:
   ```bash
   # Check memory/CPU usage during processing
   htop  # or top on older systems
   
   # Process smaller batches
   python scripts/run_complete.py single_image.jpg
   ```

### Memory Issues

#### Issue: Out of memory errors

**Cause**: Large images or insufficient RAM

**Solutions**:
```bash
# Check available memory
free -h  # Linux
# or Activity Monitor on macOS
# or Task Manager on Windows

# Reduce image size before processing
python -c "
from PIL import Image
import os
for filename in os.listdir('data/input/'):
    if filename.endswith(('.jpg', '.png')):
        img = Image.open(f'data/input/{filename}')
        if img.width > 2000 or img.height > 2000:
            # Resize to max 2000px
            img.thumbnail((2000, 2000), Image.LANCZOS)
            img.save(f'data/input/resized_{filename}', quality=90)
"

# Process one image at a time
for img in data/input/*.jpg; do
    python scripts/run_complete.py "$img" --verbose
done
```

## Output Quality Issues

### Poor Table Detection

#### Issue: Table lines not detected or incomplete

**Cause**: Lines too faint, wrong threshold, or inappropriate parameters

**Solutions**:
1. **Visualize detection process**:
   ```bash
   python tools/visualize_table_lines_v2.py problem_image.jpg
   ```

2. **Adjust detection parameters**:
   ```json
   // Lower threshold for faint lines
   {
     "threshold": 20,  // Default: 40
     "horizontal_kernel_size": 15,  // Larger kernel
     "vertical_kernel_size": 15
   }
   ```

3. **Try different preprocessing**:
   ```bash
   # Enable debug mode to see intermediate steps
   python scripts/run_complete.py problem_image.jpg --debug --verbose
   
   # Check data/debug/ for processing steps
   ls data/debug/stage1_debug/latest_run/table_detection/
   ```

#### Issue: Excessive rotation or skew correction

**Cause**: Over-sensitive deskewing parameters

**Solutions**:
```json
// More conservative deskewing
{
  "angle_range": 3,           // Smaller range
  "min_angle_correction": 0.5,  // Higher threshold
  "angle_step": 0.2           // More precise
}
```

```bash
# Visualize rotation detection
python tools/visualize_deskew_v2.py problem_image.jpg
```

#### Issue: Pages not splitting correctly

**Cause**: Gutter detection parameters not suited for document layout

**Solutions**:
```bash
# Visualize page splitting
python tools/visualize_page_split_v2.py double_page_image.jpg

# Adjust parameters for your document type
```

```json
// Wide search for off-center binding
{
  "search_ratio": 0.8,    // Search wider area
  "line_len_frac": 0.2    // Shorter minimum line length
}

// Narrow search for well-centered documents  
{
  "search_ratio": 0.3,    // Search narrow center area
  "peak_thr": 0.4         // Higher threshold for clear gutters
}
```

### Poor Margin Removal

#### Issue: Content cropped too aggressively

**Cause**: Margin removal parameters too sensitive

**Solutions**:
```bash
# Test different margin removal methods
python tools/visualize_margin_removal_v2.py image.jpg --compare
```

```json
// More conservative margin removal
{
  "black_threshold": 70,      // Less sensitive to dark areas
  "content_threshold": 180,   // Lower content threshold
  "margin_padding": 20        // More padding around content
}
```

#### Issue: Margins not removed effectively

**Cause**: Margins too light or parameters not sensitive enough

**Solutions**:
```json
// More aggressive margin removal
{
  "black_threshold": 30,      // More sensitive to dark areas
  "blur_kernel_size": 11,     // Stronger blur
  "morph_kernel_size": 35     // Larger morphology kernel
}
```

## Debug Mode Issues

### Debug Images Not Generated

#### Issue: No debug images in `data/debug/` directory

**Cause**: Debug mode not enabled or configuration issue

**Solutions**:
```bash
# Ensure debug mode is enabled
python scripts/run_complete.py image.jpg --debug --verbose

# Check debug directory was created
ls -la data/debug/

# Verify debug configuration
python -c "
from src.ocr_pipeline.config import Stage1Config
config = Stage1Config.from_json('configs/stage1_default.json')
print(f'Save debug images: {config.save_debug_images}')
"
```

#### Issue: Debug images corrupted or not viewable

**Cause**: Image format issues or processing errors

**Solutions**:
```bash
# Check debug image format and size
file data/debug/stage1_debug/latest_run/deskew/image_name/binary_threshold.png

# Convert if needed
convert problematic_debug.png fixed_debug.jpg

# Check debug image generation
python tools/visualize_deskew_v2.py test_image.jpg --save-debug
```

## Platform-Specific Issues

### Windows Issues

#### Issue: Path separator problems

**Cause**: Windows path separators in JSON or scripts

**Solutions**:
```json
// Use forward slashes or escaped backslashes in JSON
{
  "input_dir": "C:/data/input",
  "output_dir": "C:/data/output"
}
// or
{
  "input_dir": "C:\\data\\input",  
  "output_dir": "C:\\data\\output"
}
```

#### Issue: PowerShell execution policy

**Cause**: Script execution disabled in PowerShell

**Solutions**:
```powershell
# Enable script execution (run as Administrator)
Set-ExecutionPolicy RemoteSigned

# Or run directly with Python
python scripts/run_complete.py
```

### macOS Issues

#### Issue: Permission denied for system directories

**Cause**: macOS security restrictions

**Solutions**:
```bash
# Use user directories instead of system directories
export PYTHONPATH="/Users/$(whoami)/Library/Python/3.x/lib/python/site-packages:$PYTHONPATH"

# Install in user directory
pip install --user -r requirements.txt
```

#### Issue: OpenCV GUI issues on macOS

**Cause**: macOS GUI restrictions with OpenCV

**Solutions**:
```bash
# Use headless version
pip uninstall opencv-python
pip install opencv-python-headless

# Or ensure XQuartz is installed for GUI features
brew install --cask xquartz
```

### Linux Issues

#### Issue: Missing system libraries

**Cause**: System dependencies not installed

**Solutions**:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# CentOS/RHEL  
sudo yum install python3-devel mesa-libGL-devel glib2-devel libSM-devel libXext-devel libXrender-devel

# Arch Linux
sudo pacman -S python python-pip mesa glib2 libsm libxext libxrender
```

## Advanced Debugging

### Custom Debug Workflows

#### Creating Debug Scripts

```python
# debug_pipeline.py
import sys
from src.ocr_pipeline import OCRPipeline
from src.ocr_pipeline.config import Stage1Config

def debug_single_image(image_path):
    """Debug processing for a single problematic image."""
    
    print(f"Debugging: {image_path}")
    
    # Load configuration with debug enabled
    config = Stage1Config.from_json("configs/stage1_default.json")
    config.save_debug_images = True
    config.verbose = True
    
    # Process with debug mode
    pipeline = OCRPipeline(config, save_debug_images=True)
    
    try:
        result = pipeline.process_image(image_path, "debug_output/")
        print(f"Success: {result}")
        
        # Check each processing step
        debug_dir = "data/debug/stage1_debug/"
        # Add custom analysis here
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_pipeline.py image_path")
        sys.exit(1)
    
    debug_single_image(sys.argv[1])
```

#### Parameter Sensitivity Analysis

```python
# test_parameters.py
from src.ocr_pipeline import OCRPipeline
from src.ocr_pipeline.config import Stage1Config

def test_parameter_sensitivity(image_path, param_name, values):
    """Test different parameter values on the same image."""
    
    base_config = Stage1Config.from_json("configs/stage1_default.json")
    results = {}
    
    for value in values:
        print(f"Testing {param_name} = {value}")
        
        # Create modified configuration
        config = Stage1Config.from_json("configs/stage1_default.json")
        setattr(config, param_name, value)
        
        # Process image
        pipeline = OCRPipeline(config)
        result = pipeline.process_image(image_path, f"test_output/{param_name}_{value}/")
        results[value] = result
    
    return results

# Example usage
threshold_results = test_parameter_sensitivity(
    "problem_image.jpg",
    "threshold", 
    [20, 30, 40, 50, 60]
)
```

### Logging and Monitoring

```python
# Enhanced logging setup
import logging
import sys

def setup_enhanced_logging():
    """Set up detailed logging for debugging."""
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ocr_pipeline_debug.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific module log levels
    logging.getLogger('src.ocr_pipeline.pipeline').setLevel(logging.DEBUG)
    logging.getLogger('src.ocr_pipeline.utils_optimized').setLevel(logging.INFO)

# Use in scripts
setup_enhanced_logging()
```

## Getting Additional Help

### Information to Include When Reporting Issues

1. **System Information**:
   ```bash
   python --version
   pip list | grep -E "(opencv|numpy|pillow|scikit-image)"
   uname -a  # Linux/macOS
   # or systeminfo on Windows
   ```

2. **Error Messages**: Full stack trace
3. **Sample Images**: If possible, provide sample images that reproduce the issue
4. **Configuration**: Your configuration files
5. **Debug Output**: Relevant debug images or logs

### Community Resources

- **GitHub Issues**: https://github.com/yourusername/OCR_Preprocess_ToTable/issues
- **Documentation**: All documentation in `docs/` directory
- **Examples**: Check `tools/` directory for usage examples

---

**Navigation**: [← API Reference](API_REFERENCE.md) | [Documentation Index](README.md) | [Parameter Reference →](PARAMETER_REFERENCE.md)

For additional help, please check the [Parameter Reference](PARAMETER_REFERENCE.md) for detailed configuration guidance or create a GitHub issue with the information outlined above.