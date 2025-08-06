"""Image binarization operations for OCR pipeline."""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
from .base import BaseProcessor


class BinarizeProcessor(BaseProcessor):
    """Processor for binarizing images."""
    
    def process(
        self,
        image: np.ndarray,
        method: str = "otsu",
        threshold: int = 127,
        adaptive_block_size: int = 11,
        adaptive_c: int = 2,
        invert: bool = False,
        **kwargs
    ) -> np.ndarray:
        """Binarize an image using various thresholding methods.
        
        Args:
            image: Input image to binarize
            method: Binarization method - "otsu", "adaptive", or "fixed"
            threshold: Threshold value for fixed method (0-255)
            adaptive_block_size: Block size for adaptive method (must be odd)
            adaptive_c: Constant subtracted from mean for adaptive method
            invert: If True, invert the binary image (white text on black background)
            
        Returns:
            np.ndarray: Binarized image
        """
        self.validate_image(image)
        
        # Clear previous debug images
        self.clear_debug_images()
        
        # Pass processor instance for debug saving
        kwargs['_processor'] = self
        
        return binarize_image(
            image,
            method=method,
            threshold=threshold,
            adaptive_block_size=adaptive_block_size,
            adaptive_c=adaptive_c,
            invert=invert,
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


def binarize_image(
    image: np.ndarray,
    method: str = "otsu",
    threshold: int = 127,
    adaptive_block_size: int = 11,
    adaptive_c: int = 2,
    invert: bool = False,
    **kwargs
) -> np.ndarray:
    """Binarize image using various thresholding methods.
    
    This function converts a grayscale or color image to a binary (black and white) image
    using different thresholding techniques. This is often the final preprocessing step
    before OCR to ensure clean, high-contrast text.
    
    Args:
        image: Input image to binarize (can be color or grayscale)
        method: Binarization method:
            - "otsu": Automatic threshold selection using Otsu's method
            - "adaptive": Local adaptive thresholding based on local mean
            - "fixed": Simple fixed threshold value
        threshold: Threshold value for fixed method (0-255)
        adaptive_block_size: Size of pixel neighborhood for adaptive method (must be odd)
        adaptive_c: Constant subtracted from weighted mean for adaptive method
        invert: If True, invert the binary image (white text on black background)
        
    Returns:
        np.ndarray: Binary image (0 or 255 values only)
    """
    # Get processor instance if available for debug saving
    processor = kwargs.get('_processor', None)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Save original grayscale for debug
    if processor:
        processor.save_debug_image('01_grayscale_input', gray)
    
    # Apply the selected binarization method
    if method.lower() == "otsu":
        # Otsu's method automatically finds optimal threshold
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, binary = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)
        
        if processor:
            # Calculate and save histogram for debug
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            processor.save_debug_image('02_otsu_threshold', binary)
            
    elif method.lower() == "adaptive":
        # Adaptive thresholding adjusts threshold based on local neighborhood
        # Ensure block size is odd
        if adaptive_block_size % 2 == 0:
            adaptive_block_size += 1
            
        thresh_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        binary_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            thresh_type,
            binary_type,
            adaptive_block_size,
            adaptive_c
        )
        
        if processor:
            processor.save_debug_image('02_adaptive_threshold', binary)
            
    elif method.lower() == "fixed":
        # Simple fixed threshold
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, binary = cv2.threshold(gray, threshold, 255, thresh_type)
        
        if processor:
            processor.save_debug_image('02_fixed_threshold', binary)
            
    else:
        raise ValueError(f"Unknown binarization method: {method}. Use 'otsu', 'adaptive', or 'fixed'")
    
    # Optional: Apply morphological operations to clean up noise
    # This can be controlled via configuration if needed
    if kwargs.get('denoise', False):
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        if processor:
            processor.save_debug_image('03_denoised', binary)
    
    # Save final result for debug
    if processor:
        processor.save_debug_image('04_final_binary', binary)
    
    return binary