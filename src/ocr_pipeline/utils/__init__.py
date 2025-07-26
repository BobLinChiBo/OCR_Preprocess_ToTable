"""
Utility modules for the OCR Table Extraction Pipeline.

Provides common functionality used across all processors including
image operations, file handling, and logging utilities.
"""

from .file_utils import (
    ensure_directory_exists,
    get_base_filename,
    get_file_extension,
    create_output_path,
    safe_file_operation,
)
from .image_utils import (
    load_image,
    save_image,
    get_image_files,
    convert_to_grayscale,
    create_binary_mask,
    apply_morphological_operation,
    create_morphological_kernel,
    apply_roi_mask,
    normalize_image,
    resize_image,
    validate_image,
)
from .logging_utils import (
    setup_logging,
    get_logger,
    log_processing_stats,
)

__all__ = [
    # File utilities
    "ensure_directory_exists",
    "get_base_filename", 
    "get_file_extension",
    "create_output_path",
    "safe_file_operation",
    # Image utilities
    "load_image",
    "save_image",
    "get_image_files",
    "convert_to_grayscale",
    "create_binary_mask",
    "apply_morphological_operation",
    "create_morphological_kernel",
    "apply_roi_mask",
    "normalize_image",
    "resize_image",
    "validate_image",
    # Logging utilities
    "setup_logging",
    "get_logger",
    "log_processing_stats",
]