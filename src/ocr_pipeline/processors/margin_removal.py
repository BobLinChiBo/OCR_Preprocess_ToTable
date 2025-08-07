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
            method: Margin removal method (currently only "inscribed" is supported)
            **kwargs: Method-specific parameters
            
        Returns:
            Cropped image or (cropped_image, analysis) if return_analysis=True
        """
        self.validate_image(image)
        
        # Clear previous debug images
        self.clear_debug_images()
        
        # Pass processor instance for debug saving
        kwargs['_processor'] = self
        
        if method == "inscribed":
            return remove_margin_inscribed(image, **kwargs)
        else:
            raise ValueError(f"Unknown margin removal method: {method}. Only 'inscribed' method is supported.")


def paper_mask(
    img: np.ndarray, blur_ksize: int = 5, close_ksize: int = 25, close_iter: int = 2,
    processor=None, **kwargs
) -> np.ndarray:
    """Return binary mask (255=paper) from BGR/grayscale image.

    Args:
        img: Input image (BGR or grayscale)
        blur_ksize: Gaussian blur kernel size for noise reduction
        close_ksize: Morphological closing kernel size for hole filling
        close_iter: Number of closing iterations
        processor: Optional processor instance for debug saving
        **kwargs: Additional parameters:
            - erode_after_close: Erosion size after closing

    Returns:
        Binary mask where 255=paper, 0=background
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Save grayscale version
    if processor:
        processor.save_debug_image('01_grayscale', gray)

    # Apply blur if specified
    gray_blur = (
        cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0) if blur_ksize else gray
    )
    
    # Save blurred version
    if processor and blur_ksize:
        processor.save_debug_image('02_gaussian_blur', gray_blur)
    
    # Use Otsu threshold
    _, th = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Save initial threshold
    if processor:
        processor.save_debug_image('03_otsu_threshold', th)
    
    # Ensure paper is white (only invert if needed)
    if np.mean(th) < 127:
        th = cv2.bitwise_not(th)
        if processor:
            processor.save_debug_image('04_inverted_threshold', th)

    # Morphological closing fills text holes, unites regions
    kernel = np.ones((close_ksize, close_ksize), np.uint8)
    mask = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    
    # Save after morphological closing
    if processor:
        processor.save_debug_image('05_after_morphology', mask)
    
    # Optional: Erode to compensate for closing expansion
    erode_size = kwargs.get('erode_after_close', 0)
    if erode_size > 0:
        erode_kernel = np.ones((erode_size, erode_size), np.uint8)
        mask = cv2.erode(mask, erode_kernel, iterations=1)
        if processor:
            processor.save_debug_image('05b_after_erosion', mask)

    # Keep largest connected component (the page)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("No white component detected – tune parameters.")

    # Visualize all contours before selection
    if processor:
        contours_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for i, cnt in enumerate(cnts):
            color = (0, 255, 0) if i == np.argmax([cv2.contourArea(c) for c in cnts]) else (0, 0, 255)
            cv2.drawContours(contours_vis, [cnt], -1, color, 2)
        processor.save_debug_image('06_all_contours', contours_vis)

    biggest = max(cnts, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [biggest], -1, 255, cv2.FILLED)
    
    # Save final clean mask
    if processor:
        processor.save_debug_image('07_final_paper_mask', clean)
    
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
    erode_after_close: int = 0,
    use_gradient_detection: bool = False,
    gradient_threshold: int = 30,
    return_analysis: bool = False,
    **kwargs
) -> tuple:
    """Remove margins using inscribed rectangle method.

    This method:
    1. Detects paper mask based on brightness (white paper) or gradients
    2. Fills holes so the mask is one solid blob (for original method)
    3. Finds the largest axis-aligned rectangle fully contained in the mask
    4. Crops the original image to that rectangle

    Args:
        image: Input image (BGR or grayscale)
        blur_ksize: Gaussian blur kernel size (default 5)
        close_ksize: Closing kernel size (default 25)
        close_iter: Closing iterations (default 2)
        erode_after_close: Erosion size after closing to shrink mask (default 0)
        use_gradient_detection: Use gradient-based margin detection (default False)
        gradient_threshold: Threshold for gradient detection (default 30)
        return_analysis: If True, returns additional analysis information

    Returns:
        Cropped image or tuple with analysis if return_analysis=True
    """
    # Get processor instance if available for debug saving
    processor = kwargs.get('_processor', None)
    
    try:
        # Save original image first
        if processor:
            processor.save_debug_image('00_input_image', image)
        
        # Optional: Use gradient detection for finding margins
        if use_gradient_detection:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Calculate gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            if processor:
                grad_vis = (grad_mag / grad_mag.max() * 255).astype(np.uint8)
                processor.save_debug_image('00b_gradient_magnitude', grad_vis)
            
            # Find strong gradients near borders
            h, w = gray.shape
            border_size = 50  # Look within 50 pixels of edges
            
            # Create border mask
            border_mask = np.zeros_like(gray)
            border_mask[:border_size, :] = 1  # Top
            border_mask[-border_size:, :] = 1  # Bottom
            border_mask[:, :border_size] = 1  # Left
            border_mask[:, -border_size:] = 1  # Right
            
            # Find strong gradients in border regions
            border_grads = grad_mag * border_mask
            strong_grads = border_grads > gradient_threshold
            
            if processor:
                processor.save_debug_image('00c_border_gradients', (strong_grads * 255).astype(np.uint8))
            
            # Find margin boundaries from gradients
            # Top margin
            for y in range(border_size):
                if np.sum(strong_grads[y, :]) > w * 0.3:  # If >30% of row has strong gradient
                    top_margin = y + 5  # Add small offset
                    break
            else:
                top_margin = 0
                
            # Bottom margin
            for y in range(h-1, h-border_size-1, -1):
                if np.sum(strong_grads[y, :]) > w * 0.3:
                    bottom_margin = y - 5
                    break
            else:
                bottom_margin = h - 1
                
            # Left margin
            for x in range(border_size):
                if np.sum(strong_grads[:, x]) > h * 0.3:
                    left_margin = x + 5
                    break
            else:
                left_margin = 0
                
            # Right margin
            for x in range(w-1, w-border_size-1, -1):
                if np.sum(strong_grads[:, x]) > h * 0.3:
                    right_margin = x - 5
                    break
            else:
                right_margin = w - 1
            
            # Create mask from gradient boundaries
            mask = np.zeros_like(gray)
            mask[top_margin:bottom_margin+1, left_margin:right_margin+1] = 255
            
            if processor:
                processor.save_debug_image('00d_gradient_based_mask', mask)
        else:
            # Step 1: Create paper mask (with debug saving)
            mask = paper_mask(image, blur_ksize, close_ksize, close_iter, processor, 
                             erode_after_close=erode_after_close)
        
        # Save additional visualizations
        if processor:
            # Create mask overlay on original
            overlay = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_colored[:, :, 1] = 0  # Remove green channel to make it red
            mask_colored[:, :, 2] = 0  # Remove blue channel to make it red
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
            processor.save_debug_image('08_mask_overlay', overlay)

        # Step 2: Find largest inscribed rectangle
        x1, y1, x2, y2 = largest_inside_rect(mask)
        
        # Save inscribed rectangle visualization
        if processor:
            # Show the mask with the inscribed rectangle
            mask_rect_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(mask_rect_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add corner markers
            marker_size = 10
            cv2.line(mask_rect_vis, (x1, y1), (x1 + marker_size, y1), (0, 255, 0), 3)
            cv2.line(mask_rect_vis, (x1, y1), (x1, y1 + marker_size), (0, 255, 0), 3)
            cv2.line(mask_rect_vis, (x2, y1), (x2 - marker_size, y1), (0, 255, 0), 3)
            cv2.line(mask_rect_vis, (x2, y1), (x2, y1 + marker_size), (0, 255, 0), 3)
            cv2.line(mask_rect_vis, (x1, y2), (x1 + marker_size, y2), (0, 255, 0), 3)
            cv2.line(mask_rect_vis, (x1, y2), (x1, y2 - marker_size), (0, 255, 0), 3)
            cv2.line(mask_rect_vis, (x2, y2), (x2 - marker_size, y2), (0, 255, 0), 3)
            cv2.line(mask_rect_vis, (x2, y2), (x2, y2 - marker_size), (0, 255, 0), 3)
            processor.save_debug_image('09_inscribed_rect_on_mask', mask_rect_vis)
            
            # Show on original image with margins highlighted
            rect_vis = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(rect_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # Draw margin areas in semi-transparent red
            h, w = rect_vis.shape[:2]
            overlay = rect_vis.copy()
            # Top margin
            if y1 > 0:
                cv2.rectangle(overlay, (0, 0), (w, y1), (0, 0, 255), -1)
            # Bottom margin
            if y2 < h - 1:
                cv2.rectangle(overlay, (0, y2+1), (w, h), (0, 0, 255), -1)
            # Left margin
            if x1 > 0:
                cv2.rectangle(overlay, (0, y1), (x1, y2+1), (0, 0, 255), -1)
            # Right margin
            if x2 < w - 1:
                cv2.rectangle(overlay, (x2+1, y1), (w, y2+1), (0, 0, 255), -1)
            rect_vis = cv2.addWeighted(rect_vis, 0.7, overlay, 0.3, 0)
            
            # Add text labels
            cv2.putText(rect_vis, 'KEEP', (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(rect_vis, 'REMOVE', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            processor.save_debug_image('10_final_crop_visualization', rect_vis)

        # Step 3: Crop the original image
        crop = image[y1 : y2 + 1, x1 : x2 + 1]
        
        # Debug: Print the result only in debug mode
        if processor and processor.config and getattr(processor.config, 'save_debug_images', False):
            h, w = image.shape[:2]
            margins_removed = {
                'top': y1,
                'bottom': h - y2 - 1,
                'left': x1,
                'right': w - x2 - 1
            }
            total_removed = margins_removed['top'] + margins_removed['bottom'] + margins_removed['left'] + margins_removed['right']
            print(f"    [DEBUG] Margin removal: Removed {total_removed}px total (T:{margins_removed['top']} B:{margins_removed['bottom']} L:{margins_removed['left']} R:{margins_removed['right']})")
        
        # Save final cropped result
        if processor:
            processor.save_debug_image('11_final_cropped_result', crop)

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
                "erode_after_close": erode_after_close,
            },
            "paper_mask": mask,
        }

        return crop, analysis

    except RuntimeError as e:
        # Fallback - return original image if inscribed method fails
        if return_analysis:
            analysis = {
                "method": "inscribed_rectangle",
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


