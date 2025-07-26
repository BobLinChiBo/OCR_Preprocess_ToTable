"""
Base classes and interfaces for OCR pipeline processors.

Defines the common interface and functionality shared by all image processors
in the OCR table extraction pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Protocol, TypeVar, Generic
import logging
import time
from contextlib import contextmanager

import numpy as np

from ..config.models import Config
from ..exceptions import ProcessingError, ValidationError
from ..utils.logging_utils import log_processing_stats

logger = logging.getLogger(__name__)

# Type definitions
ImageArray = np.ndarray
PathLike = Union[str, Path]
T = TypeVar('T')


@dataclass
class ProcessorResult:
    """
    Result object containing processor output and metadata.
    
    Standardizes the return format for all processors to include
    both the processed data and useful metadata about the operation.
    """
    
    success: bool
    data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}
    
    @classmethod
    def success_result(
        cls, 
        data: Any, 
        processing_time: Optional[float] = None,
        **metadata: Any
    ) -> 'ProcessorResult':
        """Create a successful result."""
        return cls(
            success=True,
            data=data,
            processing_time=processing_time,
            metadata=metadata
        )
    
    @classmethod
    def error_result(
        cls,
        error_message: str,
        processing_time: Optional[float] = None,
        **metadata: Any
    ) -> 'ProcessorResult':
        """Create an error result."""
        return cls(
            success=False,
            error_message=error_message,
            processing_time=processing_time,
            metadata=metadata
        )


class ProcessorProtocol(Protocol[T]):
    """Protocol defining the interface for image processors."""
    
    def process(self, input_data: T, **kwargs: Any) -> ProcessorResult:
        """Process input data and return result."""
        ...
    
    def validate_input(self, input_data: T) -> bool:
        """Validate input data format and requirements."""
        ...


class BaseProcessor(ABC, Generic[T]):
    """
    Abstract base class for all OCR pipeline processors.
    
    Provides common functionality including logging, validation, error handling,
    and performance monitoring. All processors should inherit from this class.
    """
    
    def __init__(self, config: Config, processor_name: Optional[str] = None):
        """
        Initialize processor with configuration.
        
        Args:
            config: Pipeline configuration object
            processor_name: Optional custom name for logging
        """
        self.config = config
        self.processor_name = processor_name or self.__class__.__name__
        self.logger = logging.getLogger(f"{__name__}.{self.processor_name}")
        
        # Performance tracking
        self._processing_stats = {
            "total_processed": 0,
            "total_failed": 0,
            "total_time": 0.0,
            "average_time": 0.0
        }
        
        self.logger.debug(f"Initialized {self.processor_name}")
    
    @abstractmethod
    def process(self, input_data: T, **kwargs: Any) -> ProcessorResult:
        """
        Process input data and return result.
        
        Args:
            input_data: Data to process
            **kwargs: Additional processing parameters
            
        Returns:
            ProcessorResult with success status and data/error information
        """
        pass
    
    @abstractmethod
    def validate_input(self, input_data: T) -> bool:
        """
        Validate input data format and requirements.
        
        Args:
            input_data: Data to validate
            
        Returns:
            True if input is valid
            
        Raises:
            ValidationError: If input is invalid
        """
        pass
    
    def process_safe(self, input_data: T, **kwargs: Any) -> ProcessorResult:
        """
        Safely process input with comprehensive error handling.
        
        Args:
            input_data: Data to process
            **kwargs: Additional processing parameters
            
        Returns:
            ProcessorResult with success status and data/error information
        """
        start_time = time.time()
        
        try:
            # Validate input
            self.validate_input(input_data)
            
            # Process data
            result = self.process(input_data, **kwargs)
            
            # Update processing time if not set
            if result.processing_time is None:
                result.processing_time = time.time() - start_time
            
            # Update statistics
            self._update_stats(success=result.success, processing_time=result.processing_time)
            
            if result.success:
                self.logger.debug(f"Processing completed successfully in {result.processing_time:.3f}s")
            else:
                self.logger.warning(f"Processing failed: {result.error_message}")
            
            return result
            
        except ValidationError as e:
            processing_time = time.time() - start_time
            self._update_stats(success=False, processing_time=processing_time)
            error_msg = f"Validation failed: {e}"
            self.logger.error(error_msg)
            return ProcessorResult.error_result(error_msg, processing_time)
        
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(success=False, processing_time=processing_time)
            error_msg = f"Processing failed with exception: {e}"
            self.logger.error(error_msg, exc_info=True)
            return ProcessorResult.error_result(error_msg, processing_time)
    
    def process_batch(
        self, 
        input_items: List[T], 
        **kwargs: Any
    ) -> List[ProcessorResult]:
        """
        Process a batch of items.
        
        Args:
            input_items: List of items to process
            **kwargs: Additional processing parameters
            
        Returns:
            List of ProcessorResult objects
        """
        results = []
        
        with log_processing_stats(f"{self.processor_name} batch processing", self.logger):
            for i, item in enumerate(input_items):
                self.logger.debug(f"Processing item {i+1}/{len(input_items)}")
                result = self.process_safe(item, **kwargs)
                results.append(result)
        
        # Log batch statistics
        successful = sum(1 for r in results if r.success)
        self.logger.info(f"Batch processing complete: {successful}/{len(input_items)} successful")
        
        return results
    
    @contextmanager
    def _processing_context(self, operation_name: str):
        """Context manager for processing operations."""
        self.logger.debug(f"Starting {operation_name}")
        start_time = time.time()
        
        try:
            yield
            duration = time.time() - start_time
            self.logger.debug(f"Completed {operation_name} in {duration:.3f}s")
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Failed {operation_name} after {duration:.3f}s: {e}")
            raise
    
    def _update_stats(self, success: bool, processing_time: float) -> None:
        """Update internal processing statistics."""
        if success:
            self._processing_stats["total_processed"] += 1
        else:
            self._processing_stats["total_failed"] += 1
        
        self._processing_stats["total_time"] += processing_time
        
        total_operations = (
            self._processing_stats["total_processed"] + 
            self._processing_stats["total_failed"]
        )
        
        if total_operations > 0:
            self._processing_stats["average_time"] = (
                self._processing_stats["total_time"] / total_operations
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "processor_name": self.processor_name,
            **self._processing_stats,
            "success_rate": (
                self._processing_stats["total_processed"] / 
                max(1, self._processing_stats["total_processed"] + self._processing_stats["total_failed"])
            )
        }
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self._processing_stats = {
            "total_processed": 0,
            "total_failed": 0,
            "total_time": 0.0,
            "average_time": 0.0
        }
        self.logger.debug("Processing statistics reset")
    
    def __repr__(self) -> str:
        """String representation of processor."""
        return f"{self.__class__.__name__}(name='{self.processor_name}')"


class ImageProcessor(BaseProcessor[ImageArray]):
    """
    Base class for processors that work with image arrays.
    
    Provides common image validation and utility methods.
    """
    
    def validate_input(self, input_data: ImageArray) -> bool:
        """
        Validate image input.
        
        Args:
            input_data: Image array to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If image is invalid
        """
        if not isinstance(input_data, np.ndarray):
            raise ValidationError("Input must be a numpy array")
        
        if len(input_data.shape) < 2:
            raise ValidationError("Input must be at least 2-dimensional")
        
        if len(input_data.shape) > 3:
            raise ValidationError("Input must be at most 3-dimensional")
        
        if input_data.size == 0:
            raise ValidationError("Input image is empty")
        
        # Check for reasonable image dimensions
        height, width = input_data.shape[:2]
        if height < 10 or width < 10:
            raise ValidationError(f"Image too small: {width}x{height}")
        
        if height > 50000 or width > 50000:
            raise ValidationError(f"Image too large: {width}x{height}")
        
        return True
    
    def _get_image_info(self, image: ImageArray) -> Dict[str, Any]:
        """Get information about an image."""
        return {
            "shape": image.shape,
            "dtype": str(image.dtype),
            "size": image.size,
            "channels": image.shape[2] if len(image.shape) == 3 else 1,
            "memory_mb": image.nbytes / (1024 * 1024)
        }


class FileProcessor(BaseProcessor[PathLike]):
    """
    Base class for processors that work with file paths.
    
    Provides common file validation and handling methods.
    """
    
    def validate_input(self, input_data: PathLike) -> bool:
        """
        Validate file path input.
        
        Args:
            input_data: File path to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If path is invalid
        """
        if not input_data:
            raise ValidationError("File path cannot be empty")
        
        path = Path(input_data)
        
        if not path.exists():
            raise ValidationError(f"File does not exist: {path}")
        
        if not path.is_file():
            raise ValidationError(f"Path is not a file: {path}")
        
        return True
    
    def _get_file_info(self, file_path: PathLike) -> Dict[str, Any]:
        """Get information about a file."""
        path = Path(file_path)
        stat = path.stat()
        
        return {
            "path": str(path),
            "name": path.name,
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "extension": path.suffix,
            "modified": stat.st_mtime
        }


class BatchProcessor(Generic[T]):
    """
    Utility class for batch processing with progress tracking and error handling.
    
    Can be mixed with processor classes to add batch processing capabilities.
    """
    
    def __init__(self, processor: BaseProcessor[T]):
        self.processor = processor
        self.logger = logging.getLogger(f"{__name__}.BatchProcessor")
    
    def process_directory(
        self,
        input_dir: PathLike,
        output_dir: PathLike,
        pattern: str = "*",
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Process all files in a directory matching a pattern.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            pattern: File pattern to match
            **kwargs: Additional processing parameters
            
        Returns:
            Summary of batch processing results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.is_dir():
            raise ValidationError(f"Input directory does not exist: {input_path}")
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find matching files
        files = list(input_path.glob(pattern))
        if not files:
            self.logger.warning(f"No files found matching pattern '{pattern}' in {input_path}")
            return {"total": 0, "successful": 0, "failed": 0, "results": []}
        
        self.logger.info(f"Processing {len(files)} files from {input_path}")
        
        # Process files
        results = []
        successful = 0
        failed = 0
        
        for file_path in files:
            try:
                result = self.processor.process_safe(file_path, **kwargs)
                results.append({
                    "file": str(file_path),
                    "success": result.success,
                    "processing_time": result.processing_time,
                    "error": result.error_message
                })
                
                if result.success:
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                results.append({
                    "file": str(file_path),
                    "success": False,
                    "processing_time": None,
                    "error": str(e)
                })
                failed += 1
        
        summary = {
            "total": len(files),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(files) if files else 0,
            "results": results
        }
        
        self.logger.info(f"Batch processing complete: {successful}/{len(files)} successful")
        return summary