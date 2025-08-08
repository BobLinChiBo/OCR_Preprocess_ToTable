"""Mark and noise removal operations for OCR pipeline."""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base import BaseProcessor


class MarkRemovalProcessor(BaseProcessor):
    """Processor for removing marks, watermarks, and noise from images."""

    def process(
        self, image: np.ndarray, dilate_iter: int = 2, kernel_size: int = 3, **kwargs
    ) -> np.ndarray:
        """Remove marks and noise from an image.

        Args:
            image: Input image
            dilate_iter: Number of dilation iterations for the protect mask
            kernel_size: Size of the morphological kernel (default: 3)
            **kwargs: Additional parameters

        Returns:
            Cleaned image
        """
        self.validate_image(image)

        # Clear previous debug images
        self.clear_debug_images()

        # Pass processor instance for debug saving
        kwargs["_processor"] = self

        return remove_marks(
            image, dilate_iter=dilate_iter, kernel_size=kernel_size, **kwargs
        )


def build_protect_mask(
    gray: np.ndarray,
    dilate_iter: int,
    kernel_size: int = 3,
    table_lines_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a solid white mask for every text / line pixel.

    Args:
        gray: Grayscale image
        dilate_iter: Number of dilation iterations to expand the mask
        kernel_size: Size of the morphological kernel (default: 3)
        table_lines_mask: Optional mask of detected table lines to protect

    Returns:
        Binary mask where text/line pixels are white (255)
    """
    # Otsu -> dark foreground becomes white (255)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Parameter
    # _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # If table lines mask is provided, combine it with the text mask
    if table_lines_mask is not None:
        mask = cv2.bitwise_or(mask, table_lines_mask)

    # Safety dilation so no thin strokes fall through
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    protect = cv2.dilate(mask, kernel, iterations=dilate_iter)
    return protect


def remove_marks(
    image: np.ndarray,
    dilate_iter: int = 2,
    kernel_size: int = 3,
    table_lines_mask: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """Remove watermarks, stamps, and artifacts while preserving text and table lines.

    This function removes everything except text/table lines by:
    1. Otsu threshold -> binary foreground (text + lines).
    2. Mild dilation to make sure every stroke is covered.
    3. Copy original pixels where the mask is white; paint the rest white.

    Args:
        image: Input image (BGR or grayscale)
        dilate_iter: Number of dilation iterations for the protect mask (default: 2)
        kernel_size: Size of the morphological kernel (default: 3)
        table_lines_mask: Optional mask of detected table lines to protect

    Returns:
        Cleaned image with marks removed (grayscale)
    """
    # Get processor instance if available for debug saving
    processor = kwargs.get("_processor", None)

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Save input image
    if processor:
        processor.save_debug_image("input_grayscale", gray)

    # Build protection mask for text/lines
    protect = build_protect_mask(gray, dilate_iter, kernel_size, table_lines_mask)

    # Save debug images
    if processor:
        # Save the basic Otsu threshold (before dilation)
        _, otsu_mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        processor.save_debug_image("otsu_threshold", otsu_mask)

        # Save protection mask (after dilation)
        processor.save_debug_image("protection_mask", protect)

        # Create visualization showing what will be removed
        removal_vis = gray.copy()
        removal_vis[protect == 0] = 255  # Show what will be painted white
        processor.save_debug_image("areas_to_remove", removal_vis)

        # If table lines mask was provided, save it
        if table_lines_mask is not None:
            processor.save_debug_image("table_lines_mask", table_lines_mask)

    # Produce final: keep gray under protect, paint rest white
    clean = np.full_like(gray, 255, dtype=np.uint8)
    clean[protect > 0] = gray[protect > 0]
    
    # Debug: Print the result only in debug mode
    if processor and processor.config and getattr(processor.config, 'save_debug_images', False):
        # Calculate statistics about what was removed
        total_pixels = gray.shape[0] * gray.shape[1]
        protected_pixels = np.sum(protect > 0)
        removed_pixels = total_pixels - protected_pixels
        removal_percent = (removed_pixels / total_pixels) * 100
        
        # Count connected components that were removed (marks/artifacts)
        removed_mask = (protect == 0) & (gray < 250)  # Dark pixels that weren't protected
        num_labels, labels = cv2.connectedComponents(removed_mask.astype(np.uint8))
        num_marks = num_labels - 1  # Subtract background
        
        print(f"    [DEBUG] Mark removal: Removed {num_marks} marks/artifacts ({removal_percent:.1f}% of image cleaned)")

    # Save final result
    if processor:
        processor.save_debug_image("cleaned_result", clean)

    return clean


def create_table_lines_mask(
    image_shape: Tuple[int, int],
    h_lines: List[Tuple[int, int, int, int]],
    v_lines: List[Tuple[int, int, int, int]],
    line_thickness: int = 3,
) -> np.ndarray:
    """Create a binary mask from detected table lines.

    Args:
        image_shape: Shape of the image (height, width)
        h_lines: List of horizontal lines (x1, y1, x2, y2)
        v_lines: List of vertical lines (x1, y1, x2, y2)
        line_thickness: Thickness of lines in the mask

    Returns:
        Binary mask with table lines marked as white (255)
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    # Draw horizontal lines
    for x1, y1, x2, y2 in h_lines:
        cv2.line(mask, (x1, y1), (x2, y2), 255, line_thickness)

    # Draw vertical lines
    for x1, y1, x2, y2 in v_lines:
        cv2.line(mask, (x1, y1), (x2, y2), 255, line_thickness)

    return mask
