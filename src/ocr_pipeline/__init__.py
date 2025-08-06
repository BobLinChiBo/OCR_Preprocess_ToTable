"""Simple OCR Table Extraction Pipeline."""

__version__ = "2.0.0"
__author__ = "OCR Pipeline Team"

from .pipeline import OCRPipeline, TwoStageOCRPipeline

__all__ = ["OCRPipeline", "TwoStageOCRPipeline"]
