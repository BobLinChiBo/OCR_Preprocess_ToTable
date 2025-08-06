"""Conservative margin removal that preserves thin black borders."""

import cv2
import numpy as np
from typing import Tuple, Optional


def paper_mask_conservative(
    img: np.ndarray, 
    blur_ksize: int = 5, 
    fill_kernel_size: int = 5,
    fill_iterations: int = 2,
    preserve_border_pixels: int = 5,
    processor=None
) -> np.ndarray:
    """Create paper mask while preserving thin black borders.
    
    Args:
        img: Input image
        blur_ksize: Gaussian blur kernel size
        fill_kernel_size: Small kernel for filling text holes only
        fill_iterations: Number of morphology iterations
        preserve_border_pixels: Number of pixels to preserve from image borders
        processor: Optional processor for debug images
        
    Returns:
        Binary mask where 255=paper, 0=background/borders
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    h, w = gray.shape
    
    # Create a mask to protect borders
    border_mask = np.ones((h, w), dtype=np.uint8) * 255
    if preserve_border_pixels > 0:
        border_mask[preserve_border_pixels:-preserve_border_pixels, 
                   preserve_border_pixels:-preserve_border_pixels] = 0
    
    if processor:
        processor.save_debug_image('border_protection_mask', border_mask)
    
    # Apply Otsu threshold
    gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0) if blur_ksize else gray
    _, th = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if np.mean(th) < 127:  # ensure paper is white
        th = cv2.bitwise_not(th)
    
    # Use smaller kernel for filling only text holes
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fill_kernel_size, fill_kernel_size))
    
    # First: Small closing to fill text holes
    filled = cv2.morphologyEx(th, cv2.MORPH_CLOSE, small_kernel, iterations=fill_iterations)
    
    # Second: Small opening to clean up noise without expanding
    cleaned = cv2.morphologyEx(filled, cv2.MORPH_OPEN, small_kernel, iterations=1)
    
    # Apply border protection - force borders to be black
    cleaned = cv2.bitwise_and(cleaned, cv2.bitwise_not(border_mask))
    
    if processor:
        processor.save_debug_image('after_border_protection', cleaned)
    
    # Find largest component
    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("No white component detected")
    
    biggest = max(cnts, key=cv2.contourArea)
    final_mask = np.zeros_like(cleaned)
    cv2.drawContours(final_mask, [biggest], -1, 255, cv2.FILLED)
    
    return final_mask


def remove_margin_preserve_borders(
    image: np.ndarray,
    blur_ksize: int = 5,
    fill_kernel_size: int = 7,
    fill_iterations: int = 2,
    preserve_border_pixels: int = 5,
    safety_margin: int = 3,
    **kwargs
) -> np.ndarray:
    """Remove margins while preserving thin black borders.
    
    Args:
        image: Input image
        blur_ksize: Gaussian blur kernel size
        fill_kernel_size: Kernel for filling text (keep small)
        fill_iterations: Morphology iterations
        preserve_border_pixels: Pixels to preserve from edges
        safety_margin: Additional margin to ensure borders aren't cut
        
    Returns:
        Cropped image with borders preserved
    """
    processor = kwargs.get('_processor', None)
    
    # Get conservative mask
    mask = paper_mask_conservative(
        image, 
        blur_ksize=blur_ksize,
        fill_kernel_size=fill_kernel_size,
        fill_iterations=fill_iterations,
        preserve_border_pixels=preserve_border_pixels,
        processor=processor
    )
    
    # Find bounding box of the mask
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return image
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Add safety margin to ensure we don't cut into borders
    y_min = max(0, y_min - safety_margin)
    x_min = max(0, x_min - safety_margin)
    y_max = min(image.shape[0] - 1, y_max + safety_margin)
    x_max = min(image.shape[1] - 1, x_max + safety_margin)
    
    # Crop
    cropped = image[y_min:y_max+1, x_min:x_max+1]
    
    if processor:
        # Visualize crop region
        vis = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(vis, f'Safety margin: {safety_margin}px', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        processor.save_debug_image('conservative_crop_region', vis)
        processor.save_debug_image('conservative_crop_result', cropped)
    
    return cropped