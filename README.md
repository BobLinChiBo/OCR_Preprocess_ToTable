# OCR Table Extraction Pipeline

A **Windows-native** OCR preprocessing pipeline for extracting table structures from scanned documents. This project has been completely rewritten to provide first-class Windows support while maintaining cross-platform compatibility.

## 🚀 Quick Start (Windows)

### Setup
```cmd
# Clone the repository
git clone https://github.com/your-username/OCR_Preprocess_ToTable.git
cd OCR_Preprocess_ToTable

# Run Windows setup (creates virtual environment and installs dependencies)
scripts\setup.bat
```

### Basic Usage
```cmd
# Format and lint your code
scripts\dev.bat quick

# Run tests
scripts\test.bat fast

# Execute the OCR pipeline
scripts\run-pipeline.bat stage1
scripts\run-pipeline.bat stage2

# Or run the complete pipeline
scripts\run-pipeline.bat pipeline
```

## 📋 System Requirements

- **Windows 10/11** (primary platform)
- **Python 3.8+** installed and in PATH
- **Virtual environment support** (venv)
- **Optional**: Poetry for advanced dependency management

## 🛠️ Windows Development Tools

This project provides **three levels of Windows integration**:

### 1. Native Windows Scripts (Recommended)
- `scripts\setup.bat` - Complete environment setup
- `scripts\dev.bat` - Development commands (format, lint, type-check)
- `scripts\test.bat` - Testing with multiple modes
- `scripts\run-pipeline.bat` - Pipeline execution and monitoring

### 2. Advanced PowerShell Tools
- `scripts\setup.ps1` - Advanced setup with validation
- `scripts\invoke-dev.ps1` - Enhanced development workflows
- `scripts\manage-pipeline.ps1` - Pipeline monitoring and reporting

### 3. Cross-Platform Python Tools
- `python make.py` - Works on Windows, macOS, and Linux
- Automatic environment detection and package manager selection

## 📁 Project Structure

```
OCR_Preprocess_ToTable/
├── scripts/                    # Windows-native development scripts
│   ├── setup.bat              # Environment setup
│   ├── dev.bat                # Development commands
│   ├── test.bat               # Testing commands
│   └── run-pipeline.bat       # Pipeline execution
├── src/ocr_pipeline/          # Main Python package
│   ├── config/                # Configuration management
│   ├── processors/            # Image processing modules
│   └── utils/                 # Utilities (including Windows-specific)
├── tests/                     # Test suite with Windows compatibility
├── input/raw_images/          # Input scanned documents
├── output/                    # Processing results
├── debug/                     # Debug output and visualizations
├── make.py                    # Cross-platform make replacement
└── OCR-Pipeline.code-workspace # VSCode workspace configuration
```

## 🔧 Development Commands

### Setup and Installation
```cmd
scripts\setup.bat              # Quick setup
scripts\setup.ps1              # Advanced PowerShell setup
python make.py setup-dev       # Cross-platform setup
```

### Code Quality
```cmd
scripts\dev.bat format         # Format with black + isort
scripts\dev.bat lint           # Run all linters
scripts\dev.bat type-check     # MyPy type checking
scripts\dev.bat quick          # Format + lint + test-fast
```

### Testing
```cmd
scripts\test.bat               # All tests
scripts\test.bat fast          # Fast tests (skip slow integration)
scripts\test.bat coverage      # Tests with coverage report
scripts\test.bat unit          # Unit tests only
```

### Pipeline Execution
```cmd
scripts\run-pipeline.bat stage1     # OCR Stage 1 (raw → cropped tables)
scripts\run-pipeline.bat stage2     # OCR Stage 2 (tables → structured data)
scripts\run-pipeline.bat pipeline   # Complete pipeline
scripts\run-pipeline.bat status     # Show pipeline status
```

## 🧪 Architecture Overview

This is a **two-stage OCR preprocessing pipeline**:

### Stage 1: Initial Processing
Raw scanned images → Cropped table regions
1. **Page splitting** - Detect two-page spreads and split
2. **Deskewing** - Correct rotation
3. **Edge detection** - Gabor filters + windowing  
4. **Line detection** - Hough transforms + morphological operations
5. **Table reconstruction** - Combine detected lines
6. **Table fitting** - Optimize line placement
7. **Table cropping** - Extract final table regions

### Stage 2: Advanced Processing  
Cropped tables → Structured data *(implementation pending)*

## 🖥️ Windows-Specific Features

- **Path handling** - Automatic Windows path normalization and validation
- **Reserved names** - Handles Windows reserved filenames (CON, PRN, etc.)
- **Path length limits** - Validates against Windows MAX_PATH constraints
- **Package manager detection** - Automatically finds Poetry, pip, or conda
- **Terminal integration** - Enhanced Command Prompt and PowerShell support
- **VSCode integration** - Complete Windows development environment

## 💻 IDE Setup (VSCode)

Open the workspace file for the best Windows development experience:
```cmd
code OCR-Pipeline.code-workspace
```

This provides:
- **Windows terminal profiles** with automatic venv activation
- **Task definitions** for all development commands
- **Debug configurations** for pipeline components
- **Extension recommendations** for Python development
- **Settings optimized** for Windows development

## 🔍 Configuration

The pipeline uses **Pydantic-based configuration** with comprehensive validation:

- **Default config**: `src\ocr_pipeline\config\default_stage1.yaml`
- **Windows-safe paths** with automatic normalization
- **Type-safe validation** with helpful error messages
- **Legacy format support** for backward compatibility

## 🧪 Testing

The test suite includes **Windows-specific compatibility**:

```cmd
# Run Windows compatibility tests
pytest -m windows

# Skip Windows-only tests on other platforms  
pytest -m "not windows_only"

# Test with specific markers
pytest -m "unit and not slow"
```

## 📊 Project Status

- ✅ **Windows compatibility** - Complete rewrite for Windows-first development
- ✅ **Cross-platform support** - Works on Windows, macOS, and Linux
- ✅ **Modern Python** - Type hints, Pydantic validation, comprehensive testing
- ✅ **Stage 1 pipeline** - Raw images to cropped table regions
- 🚧 **Stage 2 pipeline** - Structured data extraction (in development)

## 🤝 Contributing

This project is designed for **Windows developers**. To contribute:

1. **Setup**: Run `scripts\setup.bat` for quick setup
2. **Development**: Use `scripts\dev.bat quick` for code quality checks  
3. **Testing**: Run `scripts\test.bat coverage` before committing
4. **Pipeline**: Test with `scripts\run-pipeline.bat validate`

## 📝 License

MIT License - see LICENSE file for details.

## 🏗️ Technical Details

### Key Technologies
- **OpenCV** - Computer vision operations
- **NumPy/SciPy** - Numerical processing
- **Pydantic** - Configuration management with validation
- **Click** - Command-line interfaces
- **Rich** - Enhanced console output
- **Pytest** - Testing framework with Windows compatibility

### Python Version Support
- Python 3.8+ (tested on Windows 10/11)
- Full type hint support with mypy
- Modern async/await patterns where applicable

---

**Note**: This project has been completely rewritten for Windows compatibility. All functionality is available through native Windows scripts without requiring WSL, Git Bash, or Unix tools.