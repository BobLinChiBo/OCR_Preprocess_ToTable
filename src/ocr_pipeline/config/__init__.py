"""
Modern configuration system with Pydantic models and validation.

Provides comprehensive configuration management with type safety,
validation, and support for multiple configuration formats.
"""

from .models import (
    Config,
    DirectoryConfig,
    PageSplittingConfig,
    DeskewingConfig,
    EdgeDetectionConfig,
    LineDetectionConfig,
    TableReconstructionConfig,
    TableFittingConfig,
    TableCroppingConfig,
    LoggingConfig,
)
from .loader import (
    load_config,
    load_config_from_dict,
    save_config,
    get_default_config,
    validate_config_file,
)

__all__ = [
    # Configuration models
    "Config",
    "DirectoryConfig",
    "PageSplittingConfig", 
    "DeskewingConfig",
    "EdgeDetectionConfig",
    "LineDetectionConfig",
    "TableReconstructionConfig",
    "TableFittingConfig",
    "TableCroppingConfig",
    "LoggingConfig",
    # Configuration loading
    "load_config",
    "load_config_from_dict",
    "save_config",
    "get_default_config",
    "validate_config_file",
]