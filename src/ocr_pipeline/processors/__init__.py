"""
Modern image processors for the OCR Table Extraction Pipeline.

Provides a comprehensive set of processors for transforming scanned document
images into clean, publication-ready table structures using a two-stage approach.
"""

from .base import BaseProcessor, ProcessorResult
from .page_splitter import PageSplitter
from .deskewer import Deskewer
from .edge_detector import EdgeDetector
from .line_detector import LineDetector
from .table_reconstructor import TableReconstructor
from .table_fitter import TableFitter
from .table_cropper import TableCropper

__all__ = [
    # Base classes
    "BaseProcessor",
    "ProcessorResult",
    # Processors
    "PageSplitter",
    "Deskewer", 
    "EdgeDetector",
    "LineDetector",
    "TableReconstructor",
    "TableFitter",
    "TableCropper",
]