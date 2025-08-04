"""Base processor class and common utilities for OCR pipeline processors."""

from typing import Any, Dict, Optional
import numpy as np
from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    """Base class for all image processors."""
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize processor with optional configuration."""
        self.config = config
        
    def get_config_value(self, key: str, default: Any) -> Any:
        """Safely get a config value with a default."""
        if self.config is None:
            return default
        return getattr(self.config, key, default)
    
    @abstractmethod
    def process(self, image: np.ndarray, **kwargs) -> Any:
        """Process an image. Must be implemented by subclasses."""
        pass
    
    def validate_image(self, image: np.ndarray) -> None:
        """Validate that the input is a valid image."""
        if image is None:
            raise ValueError("Image cannot be None")
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy array")
        if image.size == 0:
            raise ValueError("Image cannot be empty")