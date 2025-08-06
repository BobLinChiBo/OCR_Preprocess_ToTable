"""Base processor class and common utilities for OCR pipeline processors."""

from typing import Any, Dict, Optional, Union
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
import cv2
from datetime import datetime


class BaseProcessor(ABC):
    """Base class for all image processors."""
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize processor with optional configuration."""
        self.config = config
        self.debug_images = {}  # Store debug images during processing
        
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
    
    def save_debug_image(self, name: str, image: np.ndarray) -> None:
        """Store a debug image for later saving."""
        if self.get_config_value('save_debug_images', False):
            self.debug_images[name] = image
    
    def get_debug_images(self) -> Dict[str, np.ndarray]:
        """Get all stored debug images."""
        return self.debug_images
    
    def clear_debug_images(self) -> None:
        """Clear stored debug images."""
        self.debug_images = {}
    
    def save_debug_images_to_dir(self, debug_dir: Path, prefix: str = "") -> None:
        """Save all debug images to the specified directory."""
        if not self.debug_images:
            return
            
        # Create debug directory if it doesn't exist
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Get image format and quality from config
        img_format = self.get_config_value('debug_image_format', 'png')
        quality = self.get_config_value('debug_compression_quality', 95)
        
        for name, image in self.debug_images.items():
            filename = f"{prefix}_{name}.{img_format}" if prefix else f"{name}.{img_format}"
            filepath = debug_dir / filename
            
            if img_format == 'jpg' or img_format == 'jpeg':
                cv2.imwrite(str(filepath), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(str(filepath), image)