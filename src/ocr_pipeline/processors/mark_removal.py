"""Mark and noise removal operations for OCR pipeline."""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
from .base import BaseProcessor


class MarkRemovalProcessor(BaseProcessor):
    """Processor for removing marks, watermarks, and noise from images."""
    
    def process(
        self,
        image: np.ndarray,
        dilate_iter: int = 2,
        **kwargs
    ) -> np.ndarray:
        """Remove marks and noise from an image.
        
        Args:
            image: Input image
            dilate_iter: Number of dilation iterations for the protect mask
            **kwargs: Additional parameters
            
        Returns:
            Cleaned image
        """
        self.validate_image(image)
        return remove_marks(image, dilate_iter=dilate_iter)


def build_protect_mask(gray: np.ndarray, dilate_iter: int) -> np.ndarray:
    """Return a solid white mask for every text / line pixel.
    
    Args:
        gray: Grayscale image
        dilate_iter: Number of dilation iterations to expand the mask
        
    Returns:
        Binary mask where text/line pixels are white (255)
    """
    # Otsu → dark foreground becomes white (255)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Safety dilation so no thin strokes fall through
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    protect = cv2.dilate(mask, kernel, iterations=dilate_iter)
    return protect


def remove_marks(image: np.ndarray, dilate_iter: int = 2) -> np.ndarray:
    """Remove watermarks, stamps, and artifacts while preserving text and table lines.
    
    This function removes everything except text/table lines by:
    1. Otsu threshold → binary foreground (text + lines).
    2. Mild dilation to make sure every stroke is covered.
    3. Copy original pixels where the mask is white; paint the rest white.
    
    Args:
        image: Input image (BGR or grayscale)
        dilate_iter: Number of dilation iterations for the protect mask (default: 2)
        
    Returns:
        Cleaned image with marks removed (grayscale)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Build protection mask for text/lines
    protect = build_protect_mask(gray, dilate_iter)
    
    # Produce final: keep gray under protect, paint rest white
    clean = np.full_like(gray, 255, dtype=np.uint8)
    clean[protect > 0] = gray[protect > 0]
    
    return clean