"""
Common image processing utilities.

Shared image processing functions used across the OCR table extraction pipeline.
"""

import cv2
import numpy as np
import os
import logging
from typing import List, Tuple, Optional, Union

logger = logging.getLogger(__name__)


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load image from file with error handling.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Loaded image as numpy array, or None if loading failed
    """
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
        
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        logger.debug(f"Loaded image: {image_path} ({image.shape})")
        return image
        
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save image to file with error handling.
    
    Args:
        image: Image array to save
        output_path: Path where to save the image
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        success = cv2.imwrite(output_path, image)
        if success:
            logger.debug(f"Saved image: {output_path}")
            return True
        else:
            logger.error(f"Failed to save image: {output_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error saving image {output_path}: {e}")
        return False


def get_image_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Get list of image files in directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to include (default: common image formats)
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    if not os.path.isdir(directory):
        logger.warning(f"Directory not found: {directory}")
        return []
    
    image_files = []
    try:
        for filename in os.listdir(directory):
            if any(filename.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(directory, filename))
        
        image_files.sort()  # Ensure consistent ordering
        logger.debug(f"Found {len(image_files)} image files in {directory}")
        return image_files
        
    except Exception as e:
        logger.error(f"Error reading directory {directory}: {e}")
        return []


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale if needed.
    
    Args:
        image: Input image (color or grayscale)
        
    Returns:
        Grayscale image
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def create_binary_mask(gray_image: np.ndarray, threshold: int = 127, 
                      adaptive: bool = True) -> np.ndarray:
    """
    Create binary mask from grayscale image.
    
    Args:
        gray_image: Grayscale input image
        threshold: Threshold value for simple thresholding
        adaptive: Whether to use adaptive thresholding
        
    Returns:
        Binary mask (0 or 255 values)
    """
    if adaptive:
        return cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY_INV, 15, 4
        )
    else:
        _, binary = cv2.threshold(
            gray_image, threshold, 255, cv2.THRESH_BINARY_INV
        )
        return binary


def apply_morphological_operation(image: np.ndarray, operation: str, 
                                kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Apply morphological operation to image.
    
    Args:
        image: Input binary image
        operation: Type of operation ('open', 'close', 'erode', 'dilate')
        kernel: Morphological kernel
        iterations: Number of iterations
        
    Returns:
        Processed image
    """
    operations = {
        'open': cv2.MORPH_OPEN,
        'close': cv2.MORPH_CLOSE,
        'erode': cv2.MORPH_ERODE,
        'dilate': cv2.MORPH_DILATE
    }
    
    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}")
    
    return cv2.morphologyEx(image, operations[operation], kernel, iterations=iterations)


def create_morphological_kernel(shape: str, size: Tuple[int, int]) -> np.ndarray:
    """
    Create morphological kernel.
    
    Args:
        shape: Kernel shape ('rect', 'ellipse', 'cross')
        size: Kernel size (width, height)
        
    Returns:
        Morphological kernel
    """
    shapes = {
        'rect': cv2.MORPH_RECT,
        'ellipse': cv2.MORPH_ELLIPSE,
        'cross': cv2.MORPH_CROSS
    }
    
    if shape not in shapes:
        raise ValueError(f"Unknown kernel shape: {shape}")
    
    return cv2.getStructuringElement(shapes[shape], size)


def apply_roi_mask(image: np.ndarray, margins: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply region of interest mask to image.
    
    Args:
        image: Input image
        margins: Dictionary with 'top', 'bottom', 'left', 'right' margins
        
    Returns:
        Tuple of (masked_image, roi_mask)
    """
    height, width = image.shape[:2]
    
    # Create ROI mask
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    roi_mask[
        margins['top']:height - margins['bottom'],
        margins['left']:width - margins['right']
    ] = 255
    
    # Apply mask to image
    if len(image.shape) == 3:
        masked_image = cv2.bitwise_and(image, image, mask=roi_mask)
    else:
        masked_image = cv2.bitwise_and(image, roi_mask)
    
    return masked_image, roi_mask


def normalize_image(image: np.ndarray, target_type: int = cv2.CV_8U) -> np.ndarray:
    """
    Normalize image to target type.
    
    Args:
        image: Input image
        target_type: Target OpenCV data type
        
    Returns:
        Normalized image
    """
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=target_type)


def resize_image(image: np.ndarray, scale_factor: float = None, 
                target_size: Tuple[int, int] = None) -> np.ndarray:
    """
    Resize image by scale factor or to target size.
    
    Args:
        image: Input image
        scale_factor: Scale factor for resizing
        target_size: Target size (width, height)
        
    Returns:
        Resized image
    """
    if scale_factor is not None:
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    elif target_size is not None:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    else:
        raise ValueError("Either scale_factor or target_size must be provided")


def validate_image(image: np.ndarray, min_size: Tuple[int, int] = (100, 100)) -> bool:
    """
    Validate that image meets minimum requirements.
    
    Args:
        image: Image to validate
        min_size: Minimum size (width, height)
        
    Returns:
        True if image is valid
    """
    if image is None:
        return False
    
    if len(image.shape) < 2:
        return False
    
    height, width = image.shape[:2]
    if width < min_size[0] or height < min_size[1]:
        logger.warning(f"Image too small: {width}x{height}, minimum: {min_size}")
        return False
    
    return True