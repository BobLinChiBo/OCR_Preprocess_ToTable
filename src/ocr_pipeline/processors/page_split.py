"""Page splitting functionality for two-page scanned images."""

from typing import Dict, Tuple, Any, Optional
import cv2
import numpy as np
from .base import BaseProcessor


class PageSplitProcessor(BaseProcessor):
    """Processor for splitting two-page scanned images."""
    
    def process(
        self,
        image: np.ndarray,
        search_ratio: float = 0.3,
        blur_k: int = 21,
        open_k: int = 9,
        width_min: int = 20,
        return_analysis: bool = False,
        **kwargs
    ) -> tuple:
        """Split a two-page scanned image into separate pages.
        
        Args:
            image: Input two-page image
            search_ratio: Fraction of width, centered, to search for gutter (0.0-1.0)
            blur_k: Odd kernel size for Gaussian blur (higher = more noise removal)
            open_k: Horizontal kernel width for morphological opening (removes thin lines)
            width_min: Minimum gutter width in pixels
            return_analysis: If True, returns detailed analysis information
            
        Returns:
            tuple: (left_page, right_page) or (left_page, right_page, analysis) if return_analysis=True
        """
        self.validate_image(image)
        return split_two_page_image(
            image,
            search_ratio=search_ratio,
            blur_k=blur_k,
            open_k=open_k,
            width_min=width_min,
            return_analysis=return_analysis
        )


def split_two_page_image(
    image: np.ndarray,
    search_ratio: float = 0.3,
    blur_k: int = 21,
    open_k: int = 9,
    width_min: int = 20,
    return_analysis: bool = False,
) -> tuple:
    """Split a two-page scanned image into separate pages using robust algorithm.

    Args:
        image: Input two-page image
        search_ratio: Fraction of width, centered, to search for gutter (0.0-1.0)
        blur_k: Odd kernel size for Gaussian blur (higher = more noise removal)
        open_k: Horizontal kernel width for morphological opening (removes thin lines)
        width_min: Minimum gutter width in pixels
        return_analysis: If True, returns detailed analysis information

    Returns:
        tuple: (left_page, right_page) or (left_page, right_page, analysis) if return_analysis=True
    """

    # Ensure blur_k is odd
    if blur_k % 2 == 0:
        blur_k += 1

    height, width = image.shape[:2]

    # 1. Pre-processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blur = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    # 2. Remove thin vertical lines (table borders)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, 1))
    cleaned = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)

    # 3. Column-wise darkness profile
    col_dark = cleaned.mean(axis=0)  # lower values = darker
    w = col_dark.size
    margin = int((1 - search_ratio) / 2 * w)
    window = col_dark[margin : w - margin]

    # 4. Boolean mask of "dark" columns (20th percentile)
    thresh = np.percentile(window, 20)
    darkmask = window < thresh

    # 5. Find contiguous dark segments inside the search window
    segments, start = [], None
    for i, is_dark in enumerate(darkmask):
        if is_dark and start is None:
            start = i
        elif not is_dark and start is not None:
            segments.append((start + margin, i - 1 + margin))
            start = None
    if start is not None:  # mask ended while still dark
        segments.append((start + margin, len(window) - 1 + margin))

    if not segments:
        # Fallback: use center if no segments found
        gutter_x = width // 2
        left_page = image[:, :gutter_x]
        right_page = image[:, gutter_x:]

        if not return_analysis:
            return left_page, right_page

        analysis = {
            "gutter_x": gutter_x,
            "gutter_strength": 0.0,
            "gutter_width": 0,
            "search_start": margin,
            "search_end": w - margin,
            "segments": [],
            "selected_segment": None,
            "fallback_used": True,
            "meets_min_width": False,
            "has_two_pages": False,
            "col_dark": col_dark,
            "darkmask": darkmask,
            "thresh": thresh,
        }
        return left_page, right_page, analysis

    # 6. Keep only segments wide enough to be the gutter
    valid_segments = [(s, e) for s, e in segments if (e - s + 1) >= width_min]
    if not valid_segments:
        # Fallback to center if no valid segments
        gutter_x = width // 2
        left_page = image[:, :gutter_x]
        right_page = image[:, gutter_x:]

        if not return_analysis:
            return left_page, right_page

        analysis = {
            "gutter_x": gutter_x,
            "gutter_strength": 0.0,
            "gutter_width": 0,
            "search_start": margin,
            "search_end": w - margin,
            "segments": segments,
            "valid_segments": [],
            "selected_segment": None,
            "fallback_used": True,
            "meets_min_width": False,
            "has_two_pages": False,
            "col_dark": col_dark,
            "darkmask": darkmask,
            "thresh": thresh,
        }
        return left_page, right_page, analysis

    # 7. Choose the best candidate (widest, then nearest center)
    widest = max(e - s + 1 for s, e in valid_segments)
    centre = w // 2
    selected_segment = min(
        (
            (abs(((s + e) // 2) - centre), (s + e) // 2, s, e)
            for s, e in valid_segments
            if (e - s + 1) == widest
        ),
        key=lambda x: x[0],
    )
    gutter_x = selected_segment[1]
    selected_s, selected_e = selected_segment[2], selected_segment[3]

    # 8. Crop pages
    left_page = image[:, :gutter_x]
    right_page = image[:, gutter_x:]

    if not return_analysis:
        return left_page, right_page

    # Calculate enhanced analysis information
    gutter_width = selected_e - selected_s + 1

    # Calculate gutter strength based on contrast within the selected segment
    segment_values = col_dark[selected_s : selected_e + 1]
    avg_segment = np.mean(segment_values)
    avg_all = np.mean(col_dark)
    gutter_strength = (avg_all - avg_segment) / avg_all if avg_all > 0 else 0

    # Determine if image has two pages based on segment analysis
    has_two_pages = (
        len(valid_segments) > 0 and gutter_strength >= 0.15 and gutter_width >= 1
    )

    analysis = {
        "gutter_x": gutter_x,
        "gutter_strength": gutter_strength,
        "gutter_width": gutter_width,
        "search_start": margin,
        "search_end": w - margin,
        "segments": segments,
        "valid_segments": valid_segments,
        "selected_segment": (selected_s, selected_e),
        "fallback_used": False,
        "meets_min_width": gutter_width >= width_min,
        "has_two_pages": has_two_pages,
        "col_dark": col_dark,
        "darkmask": darkmask,
        "thresh": thresh,
        "widest_segment_width": widest,
        # Legacy compatibility fields
        "vertical_sums": None,  # Not used in new algorithm
        "min_sum": avg_segment * height,  # Approximate for compatibility
        "avg_sum": avg_all * height,  # Approximate for compatibility
    }

    return left_page, right_page, analysis