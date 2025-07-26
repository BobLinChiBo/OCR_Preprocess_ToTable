"""
Modern image processing utilities with enhanced error handling and type safety.

Provides comprehensive image processing functions used across the OCR pipeline
with proper type hints, validation, and robust error handling.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
import logging

from ..exceptions import ImageLoadError, ImageSaveError, ValidationError
from .file_utils import ensure_directory_exists, get_files_with_extensions

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
ImageArray = np.ndarray


def load_image(image_path: PathLike, color_mode: str = "color") -> ImageArray:
    """
    Load image from file with comprehensive error handling.
    
    Args:
        image_path: Path to image file
        color_mode: Loading mode ('color', 'grayscale', 'unchanged')
        
    Returns:
        Loaded image as numpy array
        
    Raises:
        ImageLoadError: If image cannot be loaded
        ValidationError: If parameters are invalid
    """
    path = Path(image_path)
    
    if not path.exists():
        raise ImageLoadError(f"Image file not found: {path}")
    
    # Map color modes to OpenCV flags
    mode_map = {
        "color": cv2.IMREAD_COLOR,
        "grayscale": cv2.IMREAD_GRAYSCALE,
        "unchanged": cv2.IMREAD_UNCHANGED,
    }
    
    if color_mode not in mode_map:
        raise ValidationError(f"Invalid color mode: {color_mode}. "
                            f"Must be one of: {list(mode_map.keys())}")
    
    try:
        image = cv2.imread(str(path), mode_map[color_mode])
        if image is None:
            raise ImageLoadError(f"Failed to load image (OpenCV returned None): {path}")
        
        logger.debug(f"Loaded image: {path} ({image.shape}, dtype={image.dtype})")
        return image
        
    except Exception as e:
        raise ImageLoadError(f"Error loading image {path}: {e}")


def save_image(
    image: ImageArray,
    output_path: PathLike,
    quality: int = 95,
    create_dirs: bool = True
) -> None:
    """
    Save image to file with error handling.
    
    Args:
        image: Image array to save
        output_path: Path where to save the image
        quality: JPEG quality (0-100, ignored for other formats)
        create_dirs: Whether to create parent directories
        
    Raises:
        ImageSaveError: If image cannot be saved
        ValidationError: If parameters are invalid
    """
    if not isinstance(image, np.ndarray):
        raise ValidationError("Image must be a numpy array")
    
    if not 0 <= quality <= 100:
        raise ValidationError(f"Quality must be between 0 and 100, got {quality}")
    
    path = Path(output_path)
    
    try:
        if create_dirs:
            ensure_directory_exists(path.parent)
        
        # Set compression parameters based on file extension
        params = []
        if path.suffix.lower() in ['.jpg', '.jpeg']:
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif path.suffix.lower() == '.png':
            # PNG compression level (0-9)
            compression = min(9, max(0, int((100 - quality) / 10)))
            params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
        
        success = cv2.imwrite(str(path), image, params)
        if not success:
            raise ImageSaveError(f"OpenCV failed to save image: {path}")
        
        logger.debug(f"Saved image: {path} (quality={quality})")
        
    except Exception as e:
        raise ImageSaveError(f"Error saving image {path}: {e}")


def get_image_files(
    directory: PathLike,
    extensions: Optional[List[str]] = None,
    recursive: bool = False
) -> List[Path]:
    """
    Get list of image files in directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to include
        recursive: Whether to search recursively
        
    Returns:
        List of image file paths sorted by name
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    
    return get_files_with_extensions(directory, extensions, recursive)


def convert_to_grayscale(image: ImageArray) -> ImageArray:
    """
    Convert image to grayscale if needed.
    
    Args:
        image: Input image (color or grayscale)
        
    Returns:
        Grayscale image
        
    Raises:
        ValidationError: If image format is invalid
    """
    if not isinstance(image, np.ndarray):
        raise ValidationError("Image must be a numpy array")
    
    if len(image.shape) == 2:
        # Already grayscale
        return image
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            # BGR to grayscale
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            # BGRA to grayscale
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValidationError(f"Unsupported number of channels: {image.shape[2]}")
    else:
        raise ValidationError(f"Invalid image shape: {image.shape}")


def create_binary_mask(
    gray_image: ImageArray,
    threshold: int = 127,
    adaptive: bool = True,
    invert: bool = True
) -> ImageArray:
    """
    Create binary mask from grayscale image.
    
    Args:
        gray_image: Grayscale input image
        threshold: Threshold value for simple thresholding
        adaptive: Whether to use adaptive thresholding
        invert: Whether to invert the binary result
        
    Returns:
        Binary mask (0 or 255 values)
        
    Raises:
        ValidationError: If image is not grayscale
    """
    if len(gray_image.shape) != 2:
        raise ValidationError("Input must be a grayscale image")
    
    if not 0 <= threshold <= 255:
        raise ValidationError(f"Threshold must be between 0 and 255, got {threshold}")
    
    try:
        if adaptive:
            thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
            binary = cv2.adaptiveThreshold(
                gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                thresh_type, 15, 4
            )
        else:
            thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
            _, binary = cv2.threshold(gray_image, threshold, 255, thresh_type)
        
        return binary
        
    except Exception as e:
        raise ValidationError(f"Error creating binary mask: {e}")


def apply_morphological_operation(
    image: ImageArray,
    operation: str,
    kernel: ImageArray,
    iterations: int = 1
) -> ImageArray:
    """
    Apply morphological operation to image.
    
    Args:
        image: Input binary image
        operation: Type of operation ('open', 'close', 'erode', 'dilate', 'gradient', 'tophat', 'blackhat')
        kernel: Morphological kernel
        iterations: Number of iterations
        
    Returns:
        Processed image
        
    Raises:
        ValidationError: If parameters are invalid
    """
    operations = {
        'open': cv2.MORPH_OPEN,
        'close': cv2.MORPH_CLOSE,
        'erode': cv2.MORPH_ERODE,
        'dilate': cv2.MORPH_DILATE,
        'gradient': cv2.MORPH_GRADIENT,
        'tophat': cv2.MORPH_TOPHAT,
        'blackhat': cv2.MORPH_BLACKHAT,
    }
    
    if operation not in operations:
        raise ValidationError(f"Unknown operation: {operation}. "
                            f"Must be one of: {list(operations.keys())}")
    
    if iterations < 1:
        raise ValidationError(f"Iterations must be >= 1, got {iterations}")
    
    try:
        return cv2.morphologyEx(image, operations[operation], kernel, iterations=iterations)
    except Exception as e:
        raise ValidationError(f"Error applying morphological operation '{operation}': {e}")


def create_morphological_kernel(
    shape: str,
    size: Tuple[int, int],
    anchor: Optional[Tuple[int, int]] = None
) -> ImageArray:
    """
    Create morphological kernel.
    
    Args:
        shape: Kernel shape ('rect', 'ellipse', 'cross')
        size: Kernel size (width, height)
        anchor: Anchor point (default is center)
        
    Returns:
        Morphological kernel
        
    Raises:
        ValidationError: If parameters are invalid
    """
    shapes = {
        'rect': cv2.MORPH_RECT,
        'ellipse': cv2.MORPH_ELLIPSE,
        'cross': cv2.MORPH_CROSS,
    }
    
    if shape not in shapes:
        raise ValidationError(f"Unknown kernel shape: {shape}. "
                            f"Must be one of: {list(shapes.keys())}")
    
    if len(size) != 2 or size[0] < 1 or size[1] < 1:
        raise ValidationError(f"Size must be a tuple of two positive integers, got {size}")
    
    try:
        kernel = cv2.getStructuringElement(shapes[shape], size, anchor)
        return kernel
    except Exception as e:
        raise ValidationError(f"Error creating kernel: {e}")


def apply_roi_mask(
    image: ImageArray,
    margins: Dict[str, int]
) -> Tuple[ImageArray, ImageArray]:
    """
    Apply region of interest mask to image.
    
    Args:
        image: Input image
        margins: Dictionary with 'top', 'bottom', 'left', 'right' margins
        
    Returns:
        Tuple of (masked_image, roi_mask)
        
    Raises:
        ValidationError: If margins are invalid
    """
    required_keys = {'top', 'bottom', 'left', 'right'}
    if not all(key in margins for key in required_keys):
        raise ValidationError(f"Margins must contain keys: {required_keys}")
    
    for key, value in margins.items():
        if not isinstance(value, int) or value < 0:
            raise ValidationError(f"Margin '{key}' must be a non-negative integer, got {value}")
    
    height, width = image.shape[:2]
    
    # Validate margins don't exceed image dimensions
    if margins['top'] + margins['bottom'] >= height:
        raise ValidationError("Top + bottom margins exceed image height")
    if margins['left'] + margins['right'] >= width:
        raise ValidationError("Left + right margins exceed image width")
    
    try:
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
        
    except Exception as e:
        raise ValidationError(f"Error applying ROI mask: {e}")


def normalize_image(
    image: ImageArray,
    target_type: int = cv2.CV_8U,
    alpha: float = 0.0,
    beta: float = 255.0
) -> ImageArray:
    """
    Normalize image to target type and range.
    
    Args:
        image: Input image
        target_type: Target OpenCV data type
        alpha: Lower bound of the range
        beta: Upper bound of the range
        
    Returns:
        Normalized image
    """
    try:
        return cv2.normalize(image, None, alpha, beta, cv2.NORM_MINMAX, dtype=target_type)
    except Exception as e:
        raise ValidationError(f"Error normalizing image: {e}")


def resize_image(
    image: ImageArray,
    scale_factor: Optional[float] = None,
    target_size: Optional[Tuple[int, int]] = None,
    interpolation: int = cv2.INTER_CUBIC
) -> ImageArray:
    """
    Resize image by scale factor or to target size.
    
    Args:
        image: Input image
        scale_factor: Scale factor for resizing
        target_size: Target size (width, height)
        interpolation: Interpolation method
        
    Returns:
        Resized image
        
    Raises:
        ValidationError: If parameters are invalid
    """
    if scale_factor is None and target_size is None:
        raise ValidationError("Either scale_factor or target_size must be provided")
    
    if scale_factor is not None and target_size is not None:
        raise ValidationError("Cannot specify both scale_factor and target_size")
    
    try:
        if scale_factor is not None:
            if scale_factor <= 0:
                raise ValidationError(f"Scale factor must be positive, got {scale_factor}")
            
            height, width = image.shape[:2]
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            return cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        
        else:  # target_size is not None
            if len(target_size) != 2 or target_size[0] <= 0 or target_size[1] <= 0:
                raise ValidationError(f"Target size must be (width, height) with positive values, got {target_size}")
            
            return cv2.resize(image, target_size, interpolation=interpolation)
    
    except Exception as e:
        raise ValidationError(f"Error resizing image: {e}")


def validate_image(
    image: ImageArray,
    min_size: Tuple[int, int] = (100, 100),
    allowed_dtypes: Optional[List[np.dtype]] = None
) -> bool:
    """
    Validate that image meets requirements.
    
    Args:
        image: Image to validate
        min_size: Minimum size (width, height)
        allowed_dtypes: List of allowed data types
        
    Returns:
        True if image is valid
        
    Raises:
        ValidationError: If image is invalid
    """
    if image is None:
        raise ValidationError("Image is None")
    
    if not isinstance(image, np.ndarray):
        raise ValidationError("Image must be a numpy array")
    
    if len(image.shape) < 2:
        raise ValidationError(f"Image must have at least 2 dimensions, got {len(image.shape)}")
    
    height, width = image.shape[:2]
    if width < min_size[0] or height < min_size[1]:
        raise ValidationError(f"Image too small: {width}x{height}, minimum: {min_size}")
    
    if allowed_dtypes is not None and image.dtype not in allowed_dtypes:
        raise ValidationError(f"Invalid image dtype: {image.dtype}, allowed: {allowed_dtypes}")
    
    return True


def get_image_info(image: ImageArray) -> Dict[str, Any]:
    """
    Get comprehensive information about an image.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with image information
    """
    if not isinstance(image, np.ndarray):
        raise ValidationError("Image must be a numpy array")
    
    info = {
        "shape": image.shape,
        "dtype": str(image.dtype),
        "size": image.size,
        "memory_usage": image.nbytes,
        "min_value": float(np.min(image)),
        "max_value": float(np.max(image)),
        "mean_value": float(np.mean(image)),
        "std_value": float(np.std(image)),
    }
    
    if len(image.shape) == 2:
        info["type"] = "grayscale"
        info["channels"] = 1
    elif len(image.shape) == 3:
        info["type"] = "color"
        info["channels"] = image.shape[2]
    else:
        info["type"] = "unknown"
        info["channels"] = None
    
    return info


def create_visualization_overlay(
    base_image: ImageArray,
    lines: List[Tuple[int, int, int, int]],
    line_color: Tuple[int, int, int] = (0, 255, 0),
    line_thickness: int = 2
) -> ImageArray:
    """
    Create visualization overlay with detected lines.
    
    Args:
        base_image: Base image for overlay
        lines: List of line tuples (x1, y1, x2, y2)
        line_color: Color for lines in BGR format
        line_thickness: Thickness of lines
        
    Returns:
        Image with line overlay
    """
    overlay = base_image.copy()
    
    for line in lines:
        if len(line) != 4:
            logger.warning(f"Invalid line format: {line}, expected (x1, y1, x2, y2)")
            continue
        
        try:
            cv2.line(overlay, (int(line[0]), int(line[1])), 
                    (int(line[2]), int(line[3])), line_color, line_thickness)
        except Exception as e:
            logger.warning(f"Error drawing line {line}: {e}")
    
    return overlay


def calculate_image_histogram(
    image: ImageArray,
    mask: Optional[ImageArray] = None,
    bins: int = 256
) -> Dict[str, np.ndarray]:
    """
    Calculate histogram for image channels.
    
    Args:
        image: Input image
        mask: Optional mask for histogram calculation
        bins: Number of histogram bins
        
    Returns:
        Dictionary with histogram data for each channel
    """
    if len(image.shape) == 2:
        # Grayscale
        hist = cv2.calcHist([image], [0], mask, [bins], [0, 256])
        return {"gray": hist.flatten()}
    
    elif len(image.shape) == 3:
        # Color image
        histograms = {}
        for i, channel in enumerate(['blue', 'green', 'red']):
            hist = cv2.calcHist([image], [i], mask, [bins], [0, 256])
            histograms[channel] = hist.flatten()
        return histograms
    
    else:
        raise ValidationError(f"Unsupported image shape for histogram: {image.shape}")