"""Optimized utility functions for OCR pipeline."""

import cv2
import numpy as np
from typing import Tuple, Optional

# Try to import numba, but make it optional
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def find_largest_rectangle_optimized(binary_mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Optimized version of finding largest inscribed rectangle.
    
    Args:
        binary_mask: Binary mask where 1 indicates valid content area
        
    Returns:
        Tuple of (x, y, width, height) for the largest inscribed rectangle
    """
    height, width = binary_mask.shape
    max_area = 0
    best_x = 0
    best_y = 0
    best_w = 0
    best_h = 0
    
    # Create histogram for each row
    histogram = np.zeros(width, dtype=np.int32)
    
    for row in range(height):
        # Update histogram
        for col in range(width):
            if binary_mask[row, col] == 1:
                histogram[col] += 1
            else:
                histogram[col] = 0
        
        # Find largest rectangle in current histogram using stack-based approach
        stack = []
        
        for i in range(width):
            h = histogram[i]
            
            while len(stack) > 0 and histogram[stack[-1]] > h:
                height_idx = stack.pop()
                rect_height = histogram[height_idx]
                
                if len(stack) == 0:
                    rect_width = i
                    left = 0
                else:
                    rect_width = i - stack[-1] - 1
                    left = stack[-1] + 1
                
                area = rect_height * rect_width
                
                if area > max_area:
                    max_area = area
                    best_x = left
                    best_y = row - rect_height + 1
                    best_w = rect_width
                    best_h = rect_height
            
            stack.append(i)
        
        # Process remaining bars
        while len(stack) > 0:
            height_idx = stack.pop()
            rect_height = histogram[height_idx]
            
            if len(stack) == 0:
                rect_width = width
                left = 0
            else:
                rect_width = width - stack[-1] - 1
                left = stack[-1] + 1
            
            area = rect_height * rect_width
            
            if area > max_area:
                max_area = area
                best_x = left
                best_y = row - rect_height + 1
                best_w = rect_width
                best_h = rect_height
    
    return best_x, best_y, best_w, best_h


# Apply numba JIT compilation if available
if HAS_NUMBA:
    find_largest_rectangle_optimized = numba.jit(nopython=True)(find_largest_rectangle_optimized)


def detect_curved_margins_optimized(
    image: np.ndarray,
    blur_kernel_size: int = 7,
    black_threshold: int = 50,
    content_threshold: int = 200,
    morph_kernel_size: int = 25,
    min_content_area_ratio: float = 0.1,
    downsample_factor: float = 0.25,  # Process at 1/4 resolution
) -> np.ndarray:
    """Optimized version of curved margin detection using downsampling.
    
    Args:
        image: Input image (BGR or grayscale)
        blur_kernel_size: Gaussian blur kernel size
        black_threshold: Threshold for detecting very dark regions
        content_threshold: Threshold for detecting content regions
        morph_kernel_size: Morphological operation kernel size
        min_content_area_ratio: Minimum content area ratio
        downsample_factor: Factor to downsample image for processing
        
    Returns:
        Content mask at original resolution
    """
    # Convert to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    original_shape = gray.shape
    
    # Downsample for faster processing
    if downsample_factor < 1.0:
        downsampled = cv2.resize(
            gray, 
            None, 
            fx=downsample_factor, 
            fy=downsample_factor, 
            interpolation=cv2.INTER_AREA
        )
        # Adjust kernel sizes for downsampled image
        blur_kernel_size = max(3, int(blur_kernel_size * downsample_factor))
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        morph_kernel_size = max(3, int(morph_kernel_size * downsample_factor))
        if morph_kernel_size % 2 == 0:
            morph_kernel_size += 1
    else:
        downsampled = gray
    
    height, width = downsampled.shape
    total_area = height * width
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(downsampled, (blur_kernel_size, blur_kernel_size), 0)
    
    # Create mask of content regions (not very dark)
    _, content_mask = cv2.threshold(
        blurred, content_threshold, 255, cv2.THRESH_BINARY_INV
    )
    
    # Create mask of potential margins (very dark areas)
    _, margin_mask = cv2.threshold(blurred, black_threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Remove very dark margins from content mask
    content_mask = cv2.bitwise_and(content_mask, cv2.bitwise_not(margin_mask))
    
    # Apply opening to remove small noise without losing main content
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_OPEN, opening_kernel, iterations=1)
    
    # Apply morphological closing with a smaller kernel to avoid reconnecting to edges
    # Reduce kernel size to prevent connecting to margins
    reduced_morph_kernel_size = max(5, morph_kernel_size // 3)
    if reduced_morph_kernel_size % 2 == 0:
        reduced_morph_kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (reduced_morph_kernel_size, reduced_morph_kernel_size)
    )
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(
        content_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filter contours by area
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > total_area * min_content_area_ratio:
            valid_contours.append(contour)
    
    # Create final mask
    if valid_contours:
        final_mask = np.zeros_like(downsampled)
        cv2.drawContours(final_mask, valid_contours, -1, 255, -1)
    else:
        final_mask = np.ones_like(downsampled) * 255
    
    # Upsample mask back to original size
    if downsample_factor < 1.0:
        final_mask = cv2.resize(
            final_mask, 
            (original_shape[1], original_shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )
    
    return final_mask


def remove_margin_aggressive_optimized(
    image: np.ndarray,
    blur_kernel_size: int = 7,
    black_threshold: int = 50,
    content_threshold: int = 200,
    morph_kernel_size: int = 25,
    min_content_area_ratio: float = 0.01,
    padding: int = 5,
    return_analysis: bool = False,
    use_numba: bool = True,
    downsample_factor: float = 0.25,
) -> tuple:
    """Optimized version of aggressive margin removal.
    
    Uses downsampling for content detection and optional Numba acceleration
    for rectangle finding.
    
    Args:
        image: Input image
        blur_kernel_size: Gaussian blur kernel size
        black_threshold: Threshold for detecting very dark regions
        content_threshold: Threshold for detecting content regions
        morph_kernel_size: Morphological operation kernel size
        min_content_area_ratio: Minimum content area ratio
        padding: Padding to subtract from detected boundary
        return_analysis: If True, returns analysis information
        use_numba: If True, use Numba-optimized rectangle finding
        downsample_factor: Factor to downsample for content detection
        
    Returns:
        Cropped image or (cropped_image, analysis) if return_analysis=True
    """
    # Detect content mask using optimized function
    content_mask = detect_curved_margins_optimized(
        image,
        blur_kernel_size,
        black_threshold,
        content_threshold,
        morph_kernel_size,
        min_content_area_ratio,
        downsample_factor,
    )
    
    # Convert to binary format for rectangle finding
    binary_mask = (content_mask > 0).astype(np.uint8)
    
    # Find largest inscribed rectangle
    if use_numba and HAS_NUMBA:
        try:
            x, y, w, h = find_largest_rectangle_optimized(binary_mask)
        except:
            # Fallback to standard version if optimized fails
            from . import utils
            x, y, w, h = utils.find_largest_inscribed_rectangle(content_mask)
    else:
        from . import utils
        x, y, w, h = utils.find_largest_inscribed_rectangle(content_mask)
    
    # Apply padding
    if padding > 0:
        x = x + padding
        y = y + padding
        w = max(0, w - 2 * padding)
        h = max(0, h - 2 * padding)
    
    # Crop the image
    cropped = image[y : y + h, x : x + w]
    
    if not return_analysis:
        return cropped
    
    # Calculate analysis
    original_area = image.shape[0] * image.shape[1]
    cropped_area = cropped.shape[0] * cropped.shape[1]
    
    analysis = {
        "original_shape": image.shape,
        "cropped_shape": cropped.shape,
        "crop_bounds": (x, y, w, h),
        "area_retention": cropped_area / original_area if original_area > 0 else 0,
        "content_mask": content_mask,
        "method": "largest_inscribed_rectangle_optimized",
        "downsample_factor": downsample_factor,
    }
    
    return cropped, analysis


def remove_margin_bounding_box_optimized(
    image: np.ndarray,
    blur_kernel_size: int = 7,
    black_threshold: int = 50,
    content_threshold: int = 200,
    morph_kernel_size: int = 25,
    min_content_area_ratio: float = 0.01,
    padding: int = 5,
    expansion_factor: float = 0.0,
    use_min_area_rect: bool = False,
    downsample_factor: float = 0.25,
    return_analysis: bool = False,
) -> tuple:
    """Ultra-fast margin removal using bounding box with downsampling.

    This combines the speed of downsampling with the simplicity of bounding box
    calculation, making it the fastest margin removal method.

    Args:
        image: Input image (BGR or grayscale)
        blur_kernel_size: Gaussian blur kernel size
        black_threshold: Threshold for detecting very dark regions
        content_threshold: Threshold for detecting content regions
        morph_kernel_size: Morphological operation kernel size
        min_content_area_ratio: Minimum content area ratio
        padding: Padding to add/subtract from boundary
        expansion_factor: Factor to expand bounding box (0.1 = 10%)
        use_min_area_rect: If True, use minimum area rectangle
        downsample_factor: Factor to downsample for processing
        return_analysis: If True, returns analysis information

    Returns:
        Cropped image or (cropped_image, analysis) if return_analysis=True
    """
    # Get original dimensions
    original_height, original_width = image.shape[:2]
    
    # Use the optimized margin detection function
    content_mask = detect_curved_margins_optimized(
        image,
        blur_kernel_size,
        black_threshold,
        content_threshold,
        morph_kernel_size,
        min_content_area_ratio,
        downsample_factor,
    )
    
    # If downsampled, we need to work with downsampled mask for contour finding
    if downsample_factor < 1.0:
        # Downsample the mask for contour finding
        downsampled_mask = cv2.resize(
            content_mask,
            None,
            fx=downsample_factor,
            fy=downsample_factor,
            interpolation=cv2.INTER_NEAREST
        )
    else:
        downsampled_mask = content_mask
    
    # Find contours
    contours, _ = cv2.findContours(downsampled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # No content found
        if return_analysis:
            return image, {
                "original_shape": image.shape,
                "cropped_shape": image.shape,
                "crop_bounds": (0, 0, original_width, original_height),
                "area_retention": 1.0,
                "method": "bbox_optimized_no_content",
                "downsample_factor": downsample_factor,
            }
        return image
    
    # No need to filter by area since detect_curved_margins_optimized already did that
    valid_contours = contours
    
    # Find bounding box
    all_points = np.vstack(valid_contours)
    
    if use_min_area_rect:
        rect = cv2.minAreaRect(all_points)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        x = np.min(box[:, 0])
        y = np.min(box[:, 1])
        w = np.max(box[:, 0]) - x
        h = np.max(box[:, 1]) - y
    else:
        x, y, w, h = cv2.boundingRect(all_points)
    
    # Scale back to original resolution
    if downsample_factor < 1.0:
        scale = 1.0 / downsample_factor
        x = int(x * scale)
        y = int(y * scale)
        w = int(w * scale)
        h = int(h * scale)
    
    # Apply expansion factor
    if expansion_factor > 0:
        center_x = x + w / 2
        center_y = y + h / 2
        new_w = w * (1 + expansion_factor)
        new_h = h * (1 + expansion_factor)
        x = int(center_x - new_w / 2)
        y = int(center_y - new_h / 2)
        w = int(new_w)
        h = int(new_h)
    
    # Apply padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(original_width - x, w + 2 * padding)
    h = min(original_height - y, h + 2 * padding)
    
    # Ensure valid bounds
    x = max(0, min(x, original_width - 1))
    y = max(0, min(y, original_height - 1))
    w = max(1, min(w, original_width - x))
    h = max(1, min(h, original_height - y))
    
    # Crop
    cropped = image[y : y + h, x : x + w]
    
    if not return_analysis:
        return cropped
    
    # Analysis
    original_area = original_height * original_width
    cropped_area = cropped.shape[0] * cropped.shape[1]
    
    analysis = {
        "original_shape": image.shape,
        "cropped_shape": cropped.shape,
        "crop_bounds": (x, y, w, h),
        "area_retention": cropped_area / original_area if original_area > 0 else 0,
        "method": "min_area_rect_bbox_optimized" if use_min_area_rect else "axis_aligned_bbox_optimized",
        "expansion_factor": expansion_factor,
        "downsample_factor": downsample_factor,
        "num_contours": len(valid_contours),
    }
    
    return cropped, analysis


def paper_mask_optimized(
    img: np.ndarray,
    blur_ksize: int = 5,
    close_ksize: int = 25,
    close_iter: int = 2
) -> np.ndarray:
    """Optimized version of paper mask detection.
    
    Args:
        img: Input image (BGR or grayscale)
        blur_ksize: Gaussian blur kernel size for noise reduction
        close_ksize: Morphological closing kernel size for hole filling
        close_iter: Number of closing iterations
        
    Returns:
        Binary mask where 255=paper, 0=background
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Downsample for faster processing on large images
    height, width = gray.shape
    if height > 2000 or width > 2000:
        scale = min(2000 / height, 2000 / width, 1.0)
        new_height = int(height * scale)
        new_width = int(width * scale)
        gray_small = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
        blur_ksize_small = max(3, int(blur_ksize * scale))
        close_ksize_small = max(3, int(close_ksize * scale))
    else:
        gray_small = gray
        blur_ksize_small = blur_ksize
        close_ksize_small = close_ksize
        scale = 1.0

    # Otsu threshold on a blurred version gives robust split
    gray_blur = cv2.GaussianBlur(gray_small, (blur_ksize_small, blur_ksize_small), 0) \
        if blur_ksize_small else gray_small
    _, th = cv2.threshold(
        gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if np.mean(th) < 127:        # ensure paper is white
        th = cv2.bitwise_not(th)

    # Morphological closing fills text holes, unites regions
    kernel = np.ones((close_ksize_small, close_ksize_small), np.uint8)
    mask_small = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    # Keep largest connected component (the page)
    cnts, _ = cv2.findContours(mask_small, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("No white component detected – tune parameters.")

    biggest = max(cnts, key=cv2.contourArea)
    clean_small = np.zeros_like(mask_small)
    cv2.drawContours(clean_small, [biggest], -1, 255, cv2.FILLED)
    
    # Scale back up if we downsampled
    if scale < 1.0:
        clean = cv2.resize(clean_small, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        clean = clean_small
        
    return clean


def largest_inside_rect_optimized(mask: np.ndarray) -> tuple:
    """Optimized version of finding largest inscribed rectangle.

    Args:
        mask: Binary image with 0/255 values where 255=valid area
        
    Returns:
        Tuple of (x1, y1, x2, y2) coordinates of largest inscribed rectangle
    """
    # Convert mask to binary (0/1) for processing
    binary_mask = (mask > 0).astype(np.uint8)
    h, w = binary_mask.shape
    
    # Use the existing optimized function that already exists in this file
    x, y, width, height = find_largest_rectangle_optimized(binary_mask)
    
    # Convert to (x1, y1, x2, y2) format
    x1, y1 = x, y
    x2, y2 = x + width - 1, y + height - 1
    
    return (x1, y1, x2, y2)


def remove_margin_inscribed_optimized(
    image: np.ndarray,
    blur_ksize: int = 5,
    close_ksize: int = 25,
    close_iter: int = 2,
    return_analysis: bool = False,
) -> tuple:
    """Optimized version of inscribed rectangle margin removal.
    
    This method:
    1. Detects paper mask based on brightness (white paper) with optimization
    2. Fills holes so the mask is one solid blob
    3. Finds the largest axis-aligned rectangle fully contained in the mask
    4. Crops the original image to that rectangle
    
    Args:
        image: Input image (BGR or grayscale)
        blur_ksize: Gaussian blur kernel size (default 5)
        close_ksize: Closing kernel size (default 25)
        close_iter: Closing iterations (default 2)
        return_analysis: If True, returns additional analysis information
        
    Returns:
        Cropped image or tuple with analysis if return_analysis=True
    """
    try:
        # Step 1: Create paper mask with optimization
        mask = paper_mask_optimized(image, blur_ksize, close_ksize, close_iter)
        
        # Step 2: Find largest inscribed rectangle
        x1, y1, x2, y2 = largest_inside_rect_optimized(mask)
        
        # Step 3: Crop the original image
        crop = image[y1:y2 + 1, x1:x2 + 1]
        
        if not return_analysis:
            return crop
            
        # Step 4: Calculate analysis information
        original_area = image.shape[0] * image.shape[1]
        cropped_area = crop.shape[0] * crop.shape[1]
        
        analysis = {
            "method": "inscribed_rectangle_optimized",
            "success": True,
            "original_shape": image.shape,
            "cropped_shape": crop.shape,
            "crop_bounds": (x1, y1, x2 - x1 + 1, y2 - y1 + 1),
            "area_retention": cropped_area / original_area if original_area > 0 else 0,
            "inscribed_rectangle": (x1, y1, x2, y2),
            "parameters": {
                "blur_ksize": blur_ksize,
                "close_ksize": close_ksize,
                "close_iter": close_iter,
            },
            "paper_mask": mask,
        }
        
        return crop, analysis
        
    except RuntimeError as e:
        # Fallback - return original image if inscribed method fails
        if return_analysis:
            analysis = {
                "method": "inscribed_rectangle_optimized",
                "success": False,
                "error": str(e),
                "original_shape": image.shape,
                "cropped_shape": image.shape,
                "crop_bounds": (0, 0, image.shape[1], image.shape[0]),
                "area_retention": 1.0,
                "fallback_reason": str(e),
            }
            return image, analysis
        else:
            # Return original image if method fails
            return image
    except Exception as e:
        # Handle any other errors gracefully
        if return_analysis:
            analysis = {
                "method": "inscribed_rectangle_optimized",
                "success": False,
                "error": str(e),
                "original_shape": image.shape,
                "cropped_shape": image.shape,
                "crop_bounds": (0, 0, image.shape[1], image.shape[0]),
                "area_retention": 1.0,
            }
            return image, analysis
        else:
            return image