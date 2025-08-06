# Installation Guide

Complete installation and setup guide for the OCR Table Extraction Pipeline.

## Table of Contents
- [System Requirements](#system-requirements)
- [Quick Installation](#quick-installation)
- [Detailed Installation](#detailed-installation)
- [Development Installation](#development-installation)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Docker Installation](#docker-installation)

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space (more for debug outputs)

### Recommended Requirements
- **Python**: 3.9 or 3.10 (best compatibility)
- **RAM**: 16GB for large image processing
- **CPU**: Multi-core processor for better performance
- **Storage**: SSD for faster I/O operations

### Dependencies
- **OpenCV**: 4.5.0+ (computer vision operations)
- **NumPy**: 1.20.0+ (numerical computations)  
- **Pillow**: 8.0.0+ (image processing)
- **matplotlib**: 3.3.0+ (visualization)
- **scikit-image**: 0.18.0+ (advanced image processing)

## Quick Installation

### Using pip (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/OCR_Preprocess_ToTable.git
cd OCR_Preprocess_ToTable

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install package in development mode
pip install -e .

# 5. Verify installation
python -c "from src.ocr_pipeline import OCRPipeline; print('Installation successful!')"
```

### Using conda

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/OCR_Preprocess_ToTable.git
cd OCR_Preprocess_ToTable

# 2. Create conda environment
conda create -n ocr-pipeline python=3.9
conda activate ocr-pipeline

# 3. Install dependencies
conda install opencv numpy pillow matplotlib scikit-image
pip install -r requirements.txt

# 4. Install package
pip install -e .
```

## Detailed Installation

### Step 1: Python Environment Setup

#### Option A: Virtual Environment (Recommended)
```bash
# Create isolated environment
python -m venv ocr-pipeline-env

# Activate environment
# Windows:
ocr-pipeline-env\Scripts\activate
# macOS/Linux:
source ocr-pipeline-env/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

#### Option B: Conda Environment
```bash
# Create environment with specific Python version
conda create -n ocr-pipeline python=3.9

# Activate environment
conda activate ocr-pipeline

# Install conda-forge packages for better compatibility
conda config --add channels conda-forge
```

### Step 2: Core Dependencies

```bash
# Install core computer vision libraries
pip install opencv-python>=4.5.0
pip install numpy>=1.20.0
pip install Pillow>=8.0.0

# Install scientific computing libraries
pip install scipy>=1.7.0
pip install scikit-image>=0.18.0
pip install matplotlib>=3.3.0

# Install additional utilities
pip install tqdm  # Progress bars
pip install argparse  # Command line interface
```

### Step 3: Project Installation

```bash
# Clone repository
git clone https://github.com/yourusername/OCR_Preprocess_ToTable.git
cd OCR_Preprocess_ToTable

# Install project in editable mode
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Step 4: Configuration Setup

```bash
# Create data directories
mkdir -p data/input/raw_images
mkdir -p data/input/test_images
mkdir -p data/output
mkdir -p data/debug

# Copy test images (if available)
# Place your test images in data/input/test_images/

# Verify configuration
python scripts/run_complete.py --help
```

## Development Installation

### Full Development Setup

```bash
# 1. Clone with all development tools
git clone https://github.com/yourusername/OCR_Preprocess_ToTable.git
cd OCR_Preprocess_ToTable

# 2. Create development environment
python -m venv dev-env
source dev-env/bin/activate  # Windows: dev-env\Scripts\activate

# 3. Install with development dependencies
pip install -e ".[dev]"

# 4. Install pre-commit hooks (if available)
# pre-commit install

# 5. Install testing and code quality tools
pip install pytest>=6.0
pip install black>=21.0  # Code formatting
pip install flake8>=3.9  # Linting
pip install mypy>=0.910  # Type checking

# 6. Run tests to verify setup
python -m pytest tests/ -v
```

### IDE Setup

#### Visual Studio Code
```json
// .vscode/settings.json
{
    "python.pythonPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
```

#### PyCharm
1. Open project in PyCharm
2. Go to File → Settings → Project → Python Interpreter
3. Add new interpreter from existing virtual environment
4. Select the `venv/bin/python` (or `venv\Scripts\python.exe` on Windows)

## Verification

### Basic Functionality Test

```bash
# Test basic imports
python -c "
from src.ocr_pipeline import OCRPipeline, TwoStageOCRPipeline
from src.ocr_pipeline.config import Stage1Config, Stage2Config
print('✅ Core imports successful')
"

# Test configuration loading
python -c "
from src.ocr_pipeline.config import Stage1Config
config = Stage1Config.from_json('configs/stage1_default.json')
print('✅ Configuration loading successful')
"

# Test utility functions
python -c "
from src.ocr_pipeline.utils_optimized import *
print('✅ Utility functions loaded')
"
```

### Processing Test

```bash
# Test with sample image (if available)
# Place a test image in data/input/test_images/
python scripts/run_complete.py --test-images --verbose

# Test individual stages
python scripts/run_stage1.py --test-images --verbose
```

### Visualization Tools Test

```bash
# Test V2 visualization tools
python tools/run_visualizations.py --help

# Test individual visualization tool (with sample image)
# python tools/visualize_deskew_v2.py sample_image.jpg
```

## Troubleshooting

### Common Installation Issues

#### 1. OpenCV Installation Problems

**Issue**: `ImportError: No module named cv2`
```bash
# Solution: Reinstall OpenCV
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python>=4.5.0
```

**Issue**: OpenCV conflicts with system packages
```bash
# Solution: Use headless version
pip install opencv-python-headless>=4.5.0
```

#### 2. NumPy/SciPy Compatibility Issues

**Issue**: Version conflicts between NumPy and other packages
```bash
# Solution: Install specific compatible versions
pip install numpy==1.21.0
pip install scipy==1.7.0
pip install scikit-image==0.18.3
```

#### 3. Windows-Specific Issues

**Issue**: Microsoft Visual C++ build tools required
- Install Microsoft C++ Build Tools from https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Or install Visual Studio with C++ development tools

**Issue**: Path issues with scripts
```bash
# Use python -m instead of direct script calls
python -m scripts.run_complete --help
```

#### 4. Memory Issues

**Issue**: Out of memory errors with large images
```bash
# Solution: Process smaller batches or resize images
python scripts/run_complete.py --batch-size 1 input/
```

### Verification Commands

```bash
# Check Python version
python --version

# Check pip packages
pip list | grep -E "(opencv|numpy|pillow|matplotlib|scikit-image)"

# Check package installation
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# Test data directories
ls -la data/
ls -la configs/
```

## Docker Installation

### Using Docker (Alternative Installation)

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package
RUN pip install -e .

# Create data directories
RUN mkdir -p data/input data/output data/debug

# Set default command
CMD ["python", "scripts/run_complete.py", "--help"]
```

```bash
# Build Docker image
docker build -t ocr-pipeline .

# Run container
docker run -v $(pwd)/data:/app/data ocr-pipeline python scripts/run_complete.py --test-images
```

### Docker Compose (Advanced)

```yaml
# docker-compose.yml
version: '3.8'

services:
  ocr-pipeline:
    build: .
    volumes:
      - ./data:/app/data
      - ./configs:/app/configs
    environment:
      - PYTHONPATH=/app/src
    command: python scripts/run_complete.py --help
```

```bash
# Using Docker Compose
docker-compose build
docker-compose run ocr-pipeline python scripts/run_complete.py --test-images
```

## Next Steps

After successful installation:

1. **Read the Quick Start Guide**: [docs/QUICK_START.md](QUICK_START.md)
2. **Run your first processing**: Follow examples in the [main README](../README.md#quick-start)
3. **Configure parameters**: See [configs/README.md](../configs/README.md)
4. **Explore visualization tools**: Check [tools/README.md](../tools/README.md)

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
2. Search existing [GitHub issues](https://github.com/yourusername/OCR_Preprocess_ToTable/issues)
3. Create a new issue with installation details and error messages

---

**Navigation**: [← Documentation Index](README.md) | [Quick Start Guide →](QUICK_START.md)