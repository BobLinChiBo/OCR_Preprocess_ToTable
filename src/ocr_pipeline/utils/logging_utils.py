"""
Modern logging utilities with structured logging and performance tracking.

Provides comprehensive logging setup and utilities for the OCR pipeline
with support for different output formats and performance monitoring.
"""

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Union, Generator
from datetime import datetime

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()

# Custom log levels
PERFORMANCE_LEVEL = 25
logging.addLevelName(PERFORMANCE_LEVEL, "PERFORMANCE")


class PerformanceFilter(logging.Filter):
    """Filter for performance-related log messages."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == PERFORMANCE_LEVEL


class OCRFormatter(logging.Formatter):
    """Custom formatter with enhanced information for OCR pipeline."""
    
    def __init__(self, include_module: bool = True, include_function: bool = False):
        self.include_module = include_module
        self.include_function = include_function
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        # Build the format string dynamically
        format_parts = ["%(asctime)s"]
        
        if self.include_module:
            format_parts.append("%(name)s")
        
        format_parts.append("%(levelname)s")
        
        if self.include_function and hasattr(record, 'funcName'):
            format_parts.append("%(funcName)s")
        
        format_parts.append("%(message)s")
        
        format_string = " - ".join(format_parts)
        formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    use_rich: bool = True,
    include_performance: bool = True,
    format_style: str = "detailed"
) -> logging.Logger:
    """
    Set up comprehensive logging for the OCR pipeline.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        use_rich: Whether to use rich console output
        include_performance: Whether to include performance logging
        format_style: Format style ('simple', 'detailed', 'minimal')
        
    Returns:
        Configured root logger
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set logging level
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    root_logger.setLevel(level)
    
    # Configure console handler
    if use_rich:
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=format_style == "detailed",
            markup=True,
            rich_tracebacks=True
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        
        if format_style == "minimal":
            formatter = OCRFormatter(include_module=False, include_function=False)
        elif format_style == "simple":
            formatter = OCRFormatter(include_module=True, include_function=False)
        else:  # detailed
            formatter = OCRFormatter(include_module=True, include_function=True)
        
        console_handler.setFormatter(formatter)
    
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Configure file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_formatter = OCRFormatter(include_module=True, include_function=True)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)  # More verbose in file
        root_logger.addHandler(file_handler)
    
    # Add performance logging if requested
    if include_performance:
        # Create separate logger for performance metrics
        perf_logger = logging.getLogger("performance")
        perf_logger.setLevel(PERFORMANCE_LEVEL)
        
        # Add method to log performance metrics
        def log_performance(message: str, **kwargs: Any) -> None:
            if kwargs:
                extra_info = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                message = f"{message} ({extra_info})"
            perf_logger.log(PERFORMANCE_LEVEL, message)
        
        # Monkey-patch the performance logging method
        logging.Logger.performance = log_performance
    
    # Reduce noise from external libraries
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


@contextmanager
def log_processing_stats(
    operation: str,
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager for logging processing statistics.
    
    Args:
        operation: Description of the operation
        logger: Logger instance (uses root if None)
        level: Logging level for the stats
        
    Yields:
        Dictionary to store additional statistics
    """
    if logger is None:
        logger = logging.getLogger()
    
    stats = {
        "operation": operation,
        "start_time": time.time(),
        "files_processed": 0,
        "files_failed": 0,
        "total_size": 0,
    }
    
    logger.log(level, f"Starting {operation}")
    
    try:
        yield stats
        
        # Calculate final statistics
        duration = time.time() - stats["start_time"]
        stats["duration"] = duration
        stats["success_rate"] = (
            stats["files_processed"] / (stats["files_processed"] + stats["files_failed"])
            if (stats["files_processed"] + stats["files_failed"]) > 0 else 0
        )
        
        # Log completion
        logger.log(level, 
                  f"Completed {operation}: "
                  f"processed={stats['files_processed']}, "
                  f"failed={stats['files_failed']}, "
                  f"duration={duration:.2f}s, "
                  f"success_rate={stats['success_rate']*100:.1f}%")
        
        # Log performance metrics if available
        if hasattr(logger, 'performance'):
            logger.performance(f"{operation} completed", 
                             duration=duration,
                             success_rate=stats['success_rate'],
                             throughput=stats['files_processed']/duration if duration > 0 else 0)
    
    except Exception as e:
        duration = time.time() - stats["start_time"]
        logger.error(f"Failed {operation} after {duration:.2f}s: {e}")
        raise


class ProcessingProgress:
    """Enhanced progress tracking for batch operations."""
    
    def __init__(self, description: str, total: int, logger: Optional[logging.Logger] = None):
        self.description = description
        self.total = total
        self.logger = logger or logging.getLogger()
        self.start_time = time.time()
        self.completed = 0
        self.failed = 0
        
        # Rich progress bar
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        )
        self.task_id = None
    
    def __enter__(self) -> 'ProcessingProgress':
        self.progress.__enter__()
        self.task_id = self.progress.add_task(self.description, total=self.total)
        self.logger.info(f"Starting {self.description}: {self.total} items")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.__exit__(exc_type, exc_val, exc_tb)
        
        duration = time.time() - self.start_time
        success_rate = self.completed / self.total if self.total > 0 else 0
        
        self.logger.info(
            f"Completed {self.description}: "
            f"{self.completed}/{self.total} successful "
            f"({self.failed} failed) in {duration:.2f}s "
            f"(success rate: {success_rate*100:.1f}%)"
        )
    
    def update(self, advance: int = 1, success: bool = True) -> None:
        """Update progress and statistics."""
        if self.task_id is not None:
            self.progress.update(self.task_id, advance=advance)
        
        if success:
            self.completed += advance
        else:
            self.failed += advance
    
    def set_description(self, description: str) -> None:
        """Update the progress description."""
        if self.task_id is not None:
            self.progress.update(self.task_id, description=description)


def log_system_info(logger: Optional[logging.Logger] = None) -> None:
    """Log system information for debugging purposes."""
    if logger is None:
        logger = logging.getLogger()
    
    try:
        import platform
        import psutil
        import cv2
        import numpy as np
        
        logger.info("System Information:")
        logger.info(f"  Platform: {platform.platform()}")
        logger.info(f"  Python: {platform.python_version()}")
        logger.info(f"  OpenCV: {cv2.__version__}")
        logger.info(f"  NumPy: {np.__version__}")
        logger.info(f"  CPU cores: {psutil.cpu_count()}")
        logger.info(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
    except ImportError as e:
        logger.debug(f"Could not log full system info: {e}")
        logger.info(f"Python: {sys.version}")


def configure_opencv_logging(level: int = logging.WARNING) -> None:
    """Configure OpenCV logging to reduce noise."""
    try:
        import cv2
        # Set OpenCV log level (0=SILENT, 1=FATAL, 2=ERROR, 3=WARN, 4=INFO, 5=DEBUG)
        if level >= logging.DEBUG:
            cv2.setLogLevel(5)
        elif level >= logging.INFO:
            cv2.setLogLevel(4)
        elif level >= logging.WARNING:
            cv2.setLogLevel(3)
        else:
            cv2.setLogLevel(0)
    except ImportError:
        pass


def create_debug_logger(name: str, debug_dir: Optional[Union[str, Path]] = None) -> logging.Logger:
    """
    Create a dedicated debug logger for detailed analysis.
    
    Args:
        name: Logger name
        debug_dir: Directory for debug log files
        
    Returns:
        Debug logger instance
    """
    debug_logger = logging.getLogger(f"debug.{name}")
    debug_logger.setLevel(logging.DEBUG)
    
    if debug_dir:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped debug file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_file = debug_path / f"{name}_debug_{timestamp}.log"
        
        handler = logging.FileHandler(debug_file, encoding='utf-8')
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S.%f"
        )
        handler.setFormatter(formatter)
        debug_logger.addHandler(handler)
        
        debug_logger.info(f"Debug logging started: {debug_file}")
    
    return debug_logger