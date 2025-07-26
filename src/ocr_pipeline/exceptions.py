"""
Custom exceptions for the OCR Table Extraction Pipeline.

Provides a hierarchy of exceptions for different types of errors that can occur
during image processing and table extraction operations.
"""

from typing import Optional, Any


class OCRPipelineError(Exception):
    """Base exception for all OCR pipeline errors."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} (Details: {details_str})"
        return self.message


class ConfigurationError(OCRPipelineError):
    """Raised when there are configuration-related errors."""
    pass


class ProcessingError(OCRPipelineError):
    """Raised when image processing operations fail."""
    
    def __init__(self, message: str, processor: Optional[str] = None, 
                 image_path: Optional[str] = None, **kwargs: Any) -> None:
        details = kwargs
        if processor:
            details["processor"] = processor
        if image_path:
            details["image_path"] = image_path
        super().__init__(message, details)


class ImageLoadError(ProcessingError):
    """Raised when an image cannot be loaded or is invalid."""
    pass


class ImageSaveError(ProcessingError):
    """Raised when an image cannot be saved."""
    pass


class LineDetectionError(ProcessingError):
    """Raised when line detection fails."""
    pass


class TableReconstructionError(ProcessingError):
    """Raised when table reconstruction fails."""
    pass


class ValidationError(OCRPipelineError):
    """Raised when input validation fails."""
    pass


class DirectoryError(OCRPipelineError):
    """Raised when directory operations fail."""
    pass


class PipelineStageError(OCRPipelineError):
    """Raised when a pipeline stage fails."""
    
    def __init__(self, message: str, stage: str, step: Optional[str] = None, 
                 **kwargs: Any) -> None:
        details = kwargs
        details["stage"] = stage
        if step:
            details["step"] = step
        super().__init__(message, details)