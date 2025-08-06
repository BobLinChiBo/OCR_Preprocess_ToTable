"""Stroke enhancement operations for OCR pipeline."""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
from .base import BaseProcessor


class StrokeEnhancementProcessor(BaseProcessor):
    """Processor for enhancing text strokes using morphological dilation."""
    
    def process(
        self,
        image: np.ndarray,
        kernel_size: int = 2,
        iterations: int = 1,
        kernel_shape: str = "ellipse",
        **kwargs
    ) -> np.ndarray:
        """Enhance text strokes using morphological dilation.
        
        Args:
            image: Input image to enhance
            kernel_size: Size of the morphological kernel
            iterations: Number of dilation iterations
            kernel_shape: Shape of kernel - "ellipse", "rect", or "cross"
            
        Returns:
            np.ndarray: Image with enhanced strokes
        """
        self.validate_image(image)
        
        # Clear previous debug images
        self.clear_debug_images()
        
        # Pass processor instance for debug saving
        kwargs['_processor'] = self
        
        return enhance_strokes(
            image,
            kernel_size=kernel_size,
            iterations=iterations,
            kernel_shape=kernel_shape,
            **kwargs
        )
    
    def save_debug_images_to_dir(self, debug_dir: 'Path') -> None:
        """Save debug images to a directory."""
        if not self.debug_images:
            return
            
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        for name, img in self.debug_images.items():
            if img is not None:
                path = debug_dir / f"{name}.jpg"
                cv2.imwrite(str(path), img)


def enhance_strokes(
    image: np.ndarray,
    kernel_size: int = 2,
    iterations: int = 1,
    kernel_shape: str = "ellipse",
    **kwargs
) -> np.ndarray:
    """Enhance text strokes using morphological dilation.
    
    This function applies morphological dilation to thicken text strokes, making them
    more robust for subsequent binarization. This is particularly useful for degraded
    or thin text that might be lost during thresholding.
    
    Args:
        image: Input image (grayscale or color)
        kernel_size: Size of the morphological kernel (pixels)
        iterations: Number of dilation iterations to apply
        kernel_shape: Shape of the morphological kernel:
            - "ellipse": Elliptical kernel (good for general text)
            - "rect": Rectangular kernel (preserves edges)
            - "cross": Cross-shaped kernel (minimal expansion)
        
    Returns:
        np.ndarray: Image with enhanced strokes
    """
    # Get processor instance if available for debug saving
    processor = kwargs.get('_processor', None)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Save original for debug
    if processor:
        processor.save_debug_image('01_original_input', gray)
    
    # Create morphological kernel based on shape
    if kernel_shape.lower() == "ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    elif kernel_shape.lower() == "rect":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    elif kernel_shape.lower() == "cross":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    else:
        raise ValueError(f"Unknown kernel shape: {kernel_shape}. Use 'ellipse', 'rect', or 'cross'")
    
    # Apply morphological dilation to enhance strokes
    # Note: For dark text on light background, we invert, dilate, then invert back
    # This ensures we dilate the text strokes, not the background
    
    # Invert image so text becomes white (dilation expands white regions)
    inverted = cv2.bitwise_not(gray)
    
    if processor:
        processor.save_debug_image('02_inverted_for_dilation', inverted)
    
    # Apply dilation to expand text strokes
    dilated = cv2.dilate(inverted, kernel, iterations=iterations)
    
    if processor:
        processor.save_debug_image('03_dilated_strokes', dilated)
    
    # Invert back to original polarity (dark text on light background)
    enhanced = cv2.bitwise_not(dilated)
    
    if processor:
        processor.save_debug_image('04_final_enhanced', enhanced)
    
    # Convert back to original format if needed
    if len(image.shape) == 3:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return enhanced