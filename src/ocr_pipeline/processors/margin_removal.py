"""Margin removal operations for OCR pipeline."""

from typing import Tuple, Dict, Any, Optional, List, Union
import cv2
import numpy as np
from .base import BaseProcessor


class MarginRemovalProcessor(BaseProcessor):
    """Processor for removing margins from images using various methods."""
    
    def process(
        self,
        image: np.ndarray,
        method: str = "inscribed",
        **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """Remove margins from an image using the specified method.
        
        Args:
            image: Input image
            method: Margin removal method ("inscribed", "aggressive", "bounding_box", 
                   "smart", "gradient", "edge_transition", "hybrid", "curved_black_background")
            **kwargs: Method-specific parameters
            
        Returns:
            Cropped image or (cropped_image, analysis) if return_analysis=True
        """
        self.validate_image(image)
        
        if method == "inscribed":
            return remove_margin_inscribed(image, **kwargs)
        elif method == "aggressive":
            return remove_margin_aggressive(image, **kwargs)
        elif method == "bounding_box":
            return remove_margin_bounding_box(image, **kwargs)
        elif method == "smart":
            return remove_margin_smart(image, **kwargs)
        elif method == "gradient":
            return remove_margins_gradient(image, **kwargs)
        elif method == "edge_transition":
            return remove_margins_edge_transition(image, **kwargs)
        elif method == "hybrid":
            return remove_margins_hybrid(image, **kwargs)
        elif method == "curved_black_background":
            return remove_curved_black_background(image, **kwargs)
        else:
            raise ValueError(f"Unknown margin removal method: {method}")


def paper_mask(
    img: np.ndarray, blur_ksize: int = 5, close_ksize: int = 25, close_iter: int = 2
) -> np.ndarray:
    """Return binary mask (255=paper) from BGR/grayscale image.

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

    # Otsu threshold on a blurred version gives robust split
    gray_blur = (
        cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0) if blur_ksize else gray
    )
    _, th = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(th) < 127:  # ensure paper is white
        th = cv2.bitwise_not(th)

    # Morphological closing fills text holes, unites regions
    kernel = np.ones((close_ksize, close_ksize), np.uint8)
    mask = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    # Keep largest connected component (the page)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("No white component detected – tune parameters.")

    biggest = max(cnts, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [biggest], -1, 255, cv2.FILLED)
    return clean


def largest_inside_rect(mask: np.ndarray) -> tuple:
    """Return (x1, y1, x2, y2) of the largest axis-aligned rectangle
    that fits entirely inside the white area of `mask`.

    Args:
        mask: Binary image with 0/255 values where 255=valid area

    Returns:
        Tuple of (x1, y1, x2, y2) coordinates of largest inscribed rectangle
    """
    h, w = mask.shape
    hist = [0] * w
    best_area = 0
    best_coords = None

    for y in range(h):
        # build histogram of consecutive white pixels
        row = mask[y]
        for x in range(w):
            hist[x] = hist[x] + 1 if row[x] else 0

        # largest rectangle in this histogram
        stack = []
        x = 0
        while x <= w:
            curr_h = hist[x] if x < w else 0
            if not stack or curr_h >= hist[stack[-1]]:
                stack.append(x)
                x += 1
            else:
                top = stack.pop()
                width = x if not stack else x - stack[-1] - 1
                area = hist[top] * width
                if area > best_area:
                    best_area = area
                    height = hist[top]
                    x2 = x - 1
                    x1 = 0 if not stack else stack[-1] + 1
                    y2 = y
                    y1 = y - height + 1
                    best_coords = (x1, y1, x2, y2)
        # end while
    if best_coords is None:
        raise RuntimeError("Failed to find inscribed rectangle.")
    return best_coords


def remove_margin_inscribed(
    image: np.ndarray,
    blur_ksize: int = 5,
    close_ksize: int = 25,
    close_iter: int = 2,
    return_analysis: bool = False,
) -> tuple:
    """Remove margins using inscribed rectangle method.

    This method:
    1. Detects paper mask based on brightness (white paper)
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
        # Step 1: Create paper mask
        mask = paper_mask(image, blur_ksize, close_ksize, close_iter)

        # Step 2: Find largest inscribed rectangle
        x1, y1, x2, y2 = largest_inside_rect(mask)

        # Step 3: Crop the original image
        crop = image[y1 : y2 + 1, x1 : x2 + 1]

        if not return_analysis:
            return crop

        # Step 4: Calculate analysis information
        original_area = image.shape[0] * image.shape[1]
        cropped_area = crop.shape[0] * crop.shape[1]

        analysis = {
            "method": "inscribed_rectangle",
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
        # Fallback to aggressive method if inscribed fails
        if return_analysis:
            fallback_result = remove_margin_aggressive(image, return_analysis=True)
            fallback_crop, fallback_analysis = fallback_result
            fallback_analysis["method"] = "inscribed_fallback_to_aggressive"
            fallback_analysis["fallback_reason"] = str(e)
            return fallback_crop, fallback_analysis
        else:
            return remove_margin_aggressive(image)
    except Exception as e:
        # Handle any other errors gracefully
        if return_analysis:
            analysis = {
                "method": "inscribed_rectangle",
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


def find_largest_inscribed_rectangle(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Find the largest inscribed rectangle within a binary mask.
    
    Args:
        mask: Binary mask where True/255 indicates valid content area
        
    Returns:
        Tuple of (x, y, width, height) of the largest inscribed rectangle
    """
    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, mask.shape[1], mask.shape[0]
    
    # Get bounding box of largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Use the mask to find actual inscribed rectangle
    content_region = mask[y:y+h, x:x+w]
    
    # Find largest inscribed rectangle in the content region
    try:
        x1, y1, x2, y2 = largest_inside_rect(content_region.astype(np.uint8) * 255)
        return x + x1, y + y1, x2 - x1, y2 - y1
    except:
        # Fallback to bounding box
        return x, y, w, h


def _largest_rectangle_in_histogram(
    histogram: List[int], height: int
) -> Tuple[int, int, int]:
    """Find the largest rectangle that can be drawn in a histogram.
    
    Args:
        histogram: List of bar heights
        height: Height of the current row (for area calculation)
        
    Returns:
        Tuple of (x_start, width, area) of the largest rectangle
    """
    stack = []
    max_area = 0
    best_x_start = 0
    best_width = 0
    i = 0
    
    while i < len(histogram):
        if not stack or histogram[i] >= histogram[stack[-1]]:
            stack.append(i)
            i += 1
        else:
            top = stack.pop()
            width = i if not stack else i - stack[-1] - 1
            area = histogram[top] * width
            if area > max_area:
                max_area = area
                best_width = width
                best_x_start = stack[-1] + 1 if stack else 0
    
    while stack:
        top = stack.pop()
        width = i if not stack else i - stack[-1] - 1
        area = histogram[top] * width
        if area > max_area:
            max_area = area
            best_width = width
            best_x_start = stack[-1] + 1 if stack else 0
    
    return best_x_start, best_width, max_area


# Placeholder implementations for other margin removal methods
# These would need to be imported from utils.py or implemented here

def remove_margin_aggressive(
    image: np.ndarray,
    blur_kernel_size: int = 7,
    black_threshold: int = 50,
    content_threshold: int = 200,
    morph_kernel_size: int = 25,
    min_content_area_ratio: float = 0.01,
    padding: int = 5,
    return_analysis: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """Remove margins aggressively using largest inscribed rectangle approach."""
    # Simplified implementation - would need full implementation
    # For now, fallback to inscribed method
    return remove_margin_inscribed(image, return_analysis=return_analysis)


def remove_margin_bounding_box(
    image: np.ndarray,
    **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """Remove margins using bounding box method."""
    # Placeholder - needs implementation
    return remove_margin_inscribed(image, **kwargs)


def remove_margin_smart(
    image: np.ndarray,
    **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """Remove margins using smart asymmetric projection-based method."""
    # Placeholder - needs implementation
    return remove_margin_inscribed(image, **kwargs)


def remove_margins_gradient(
    image: np.ndarray,
    **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """Remove margins using gradient-based sustained transition detection."""
    # Placeholder - needs implementation
    return remove_margin_inscribed(image, **kwargs)


def remove_margins_edge_transition(
    image: np.ndarray,
    **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """Remove margins using simple edge intensity jump detection."""
    # Placeholder - needs implementation
    return remove_margin_inscribed(image, **kwargs)


def remove_margins_hybrid(
    image: np.ndarray,
    **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """Remove margins using position + texture analysis."""
    # Placeholder - needs implementation
    return remove_margin_inscribed(image, **kwargs)


def remove_curved_black_background(
    image: np.ndarray,
    black_threshold: int = 30,
    min_contour_area: int = 1000,
    padding: int = 2,
    fill_method: str = "color_fill",
    return_analysis: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """Remove curved black background from book page images."""
    # Placeholder - needs implementation
    return remove_margin_inscribed(image, return_analysis=return_analysis)