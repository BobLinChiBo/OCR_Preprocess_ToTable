"""
OCR Table Extraction Pipeline

A professional OCR preprocessing pipeline for table extraction from scanned documents.
Implements a two-stage workflow that processes raw scanned academic papers and
technical documents to extract clean, publication-ready table structures.
"""

__version__ = "0.2.0"
__author__ = "OCR Pipeline Team"
__email__ = "team@ocrpipeline.com"

from .config import Config, load_config
from .exceptions import OCRPipelineError, ProcessingError, ConfigurationError

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "Config",
    "load_config",
    "OCRPipelineError",
    "ProcessingError", 
    "ConfigurationError",
]