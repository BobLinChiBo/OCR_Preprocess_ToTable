# OCR Table Extraction Pipeline - Documentation

Complete documentation for the professional OCR Table Extraction Pipeline system.

## 📖 Documentation Index

### Getting Started
- **[Main README](../README.md)** - Project overview, quick start, and basic usage
- **[Installation Guide](INSTALLATION.md)** - Detailed installation and setup instructions  
- **[Quick Start Guide](QUICK_START.md)** - Fast track to processing your first images
- **[Configuration Guide](../configs/README.md)** - Setting up and customizing pipeline parameters

### User Guides
- **[Parameter Reference](PARAMETER_REFERENCE.md)** - Complete guide to all configurable parameters
- **[Debug Mode Guide](DEBUG_MODE_GUIDE.md)** - Comprehensive debugging and analysis workflows
- **[Enable/Disable Steps](enable_disable_steps.md)** - Customizing pipeline processing steps
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Common issues and solutions

### Tools and Visualization
- **[Tools Documentation](../tools/README.md)** - Visualization and analysis tools guide
- **[V2 Migration Guide](V2_MIGRATION_GUIDE.md)** - Upgrading to V2 visualization architecture

### Developer Resources
- **[API Reference](API_REFERENCE.md)** - Code API documentation for developers
- **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - Technical implementation details
- **[Debug Mode Implementation](DEBUG_MODE_IMPLEMENTATION_SUMMARY.md)** - Debug system architecture
- **[CLAUDE.md](CLAUDE.md)** - AI assistant integration guide

## 🚀 Quick Navigation

### I want to...
- **Get started quickly** → [Quick Start Guide](QUICK_START.md)
- **Install the system** → [Installation Guide](INSTALLATION.md)
- **Process my first images** → [Main README](../README.md#quick-start)
- **Understand parameters** → [Parameter Reference](PARAMETER_REFERENCE.md)
- **Debug processing issues** → [Debug Mode Guide](DEBUG_MODE_GUIDE.md)
- **Use visualization tools** → [Tools Documentation](../tools/README.md)
- **Customize processing** → [Configuration Guide](../configs/README.md)
- **Fix problems** → [Troubleshooting Guide](TROUBLESHOOTING.md)

### By User Type

#### **New Users**
1. [Installation Guide](INSTALLATION.md)
2. [Quick Start Guide](QUICK_START.md)  
3. [Main README](../README.md)
4. [Configuration Guide](../configs/README.md)

#### **Advanced Users**
1. [Parameter Reference](PARAMETER_REFERENCE.md)
2. [Tools Documentation](../tools/README.md)
3. [Debug Mode Guide](DEBUG_MODE_GUIDE.md)
4. [Enable/Disable Steps](enable_disable_steps.md)

#### **Developers**
1. [API Reference](API_REFERENCE.md)
2. [Implementation Summary](IMPLEMENTATION_SUMMARY.md)
3. [CLAUDE.md](CLAUDE.md)
4. [V2 Migration Guide](V2_MIGRATION_GUIDE.md)

## 📊 Pipeline Overview

The OCR Table Extraction Pipeline is a two-stage system:

```
Stage 1: Initial Processing
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Page Split  │ -> │   Deskew    │ -> │   Margin    │ -> │ Table Lines │
│             │    │             │    │  Removal    │    │ Detection   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘

Stage 2: Refinement Processing  
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Deskew    │ -> │ Table Line  │ -> │   Table     │ -> │  Vertical   │
│ (Fine-tune) │    │ Detection   │    │  Recovery   │    │ Strip Cut   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## 🔧 Key Features

- **Advanced Computer Vision**: Page splitting, deskewing, margin removal
- **Robust Table Detection**: Connected components method for accurate structure detection
- **Two-Stage Architecture**: Initial processing + precision refinement  
- **Comprehensive Debugging**: Visual analysis tools and debug mode
- **Flexible Configuration**: JSON-based parameter management
- **V2 Architecture**: Enhanced visualization tools with processor wrappers

## 📁 Documentation Structure

```
docs/
├── README.md                              # This file - documentation index
├── INSTALLATION.md                        # Installation and setup guide
├── QUICK_START.md                         # Fast-track getting started
├── API_REFERENCE.md                       # Developer API documentation
├── TROUBLESHOOTING.md                     # Common issues and solutions
├── PARAMETER_REFERENCE.md                 # Complete parameter guide
├── DEBUG_MODE_GUIDE.md                    # Debugging and analysis guide
├── V2_MIGRATION_GUIDE.md                  # V2 architecture migration
├── enable_disable_steps.md                # Customizing pipeline steps
├── IMPLEMENTATION_SUMMARY.md              # Technical implementation details
├── DEBUG_MODE_IMPLEMENTATION_SUMMARY.md   # Debug system architecture
└── CLAUDE.md                              # AI assistant integration
```

## 🤝 Contributing

See the [main README](../README.md#contributing) for contribution guidelines and development workflow.

## 📧 Support

For questions, issues, or contributions:
- Use the GitHub issue tracker for bugs and feature requests
- Submit pull requests for improvements
- Check the [Troubleshooting Guide](TROUBLESHOOTING.md) for common solutions

---

**Quick Links**: [Main README](../README.md) | [Installation](INSTALLATION.md) | [Quick Start](QUICK_START.md) | [Parameter Reference](PARAMETER_REFERENCE.md) | [Tools Guide](../tools/README.md)