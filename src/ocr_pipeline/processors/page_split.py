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
        search_ratio: float = 0.5,
        line_len_frac: float = 0.3,
        line_thick: int = 3,
        peak_thr: float = 0.3,
        return_analysis: bool = False,
        **kwargs
    ) -> tuple:
        """Split a two-page scanned image into separate pages.
        
        Args:
            image: Input two-page image
            search_ratio: Fraction of width, centered, to search for gutter (0.0-1.0)
            line_len_frac: Vertical kernel = fraction of image height
            line_thick: Kernel width in pixels
            peak_thr: Peak threshold (fraction of max response)
            return_analysis: If True, returns detailed analysis information
            
        Returns:
            tuple: (right_page, left_page) or (right_page, left_page, analysis) if return_analysis=True
        """
        self.validate_image(image)
        return split_two_page_image(
            image,
            search_ratio=search_ratio,
            line_len_frac=line_len_frac,
            line_thick=line_thick,
            peak_thr=peak_thr,
            return_analysis=return_analysis
        )


def split_two_page_image(
    image: np.ndarray,
    search_ratio: float = 0.5,
    line_len_frac: float = 0.3,
    line_thick: int = 3,
    peak_thr: float = 0.3,
    return_analysis: bool = False,
) -> tuple:
    """Split a two-page scanned image into separate pages using V2 algorithm.

    Args:
        image: Input two-page image
        search_ratio: Fraction of width, centered, to search for gutter (0.0-1.0)
        line_len_frac: Vertical kernel = fraction of image height
        line_thick: Kernel width in pixels
        peak_thr: Peak threshold (fraction of max response)
        return_analysis: If True, returns detailed analysis information

    Returns:
        tuple: (right_page, left_page) or (right_page, left_page, analysis) if return_analysis=True
    """
    h, w = image.shape[:2]
    
    # 1) extract long vertical lines
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, bw  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    vert   = 255 - bw
    k_len  = max(15, int(line_len_frac * h))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_thick, k_len))
    lines  = cv2.morphologyEx(vert, cv2.MORPH_OPEN, kernel)

    # 2) column response & search window
    col_sum = lines.sum(axis=0)
    centre  = w // 2
    margin  = int((1 - search_ratio) / 2 * w)
    window  = col_sum[margin:w - margin]

    peaks = np.where(window >= peak_thr * window.max())[0] + margin
    if len(peaks) < 2:                     # fallback if only one border found
        split_x = centre
        left_page = image[:, :split_x]
        right_page = image[:, split_x:]
        
        if not return_analysis:
            return right_page, left_page
            
        # Build analysis for fallback case
        analysis = {
            "gutter_x": split_x,
            "gutter_strength": 0.0,
            "gutter_width": 0,
            "search_start": margin,
            "search_end": w - margin,
            "segments": [],
            "selected_segment": None,
            "fallback_used": True,
            "meets_min_width": False,
            "has_two_pages": False,
            "col_dark": None,
            "darkmask": None,
            "thresh": None,
            "widest_segment_width": 0,
            # Legacy compatibility fields
            "vertical_sums": None,
            "min_sum": 0,
            "avg_sum": 0,
        }
        return right_page, left_page, analysis

    # 3) contiguous-peak grouping
    segments, s = [], peaks[0]
    for p, q in zip(peaks, peaks[1:]):
        if q != p + 1:
            segments.append((s, p)); s = q
    segments.append((s, peaks[-1]))

    # 4) choose the two segments nearest the centre
    segs = sorted(segments, key=lambda ab: abs(((ab[0]+ab[1])//2) - centre))[:2]
    if len(segs) < 2:  # If only one segment found, use center fallback
        split_x = centre
        left_page = image[:, :split_x]
        right_page = image[:, split_x:]
        
        if not return_analysis:
            return right_page, left_page
            
        # Build analysis for single segment case
        analysis = {
            "gutter_x": split_x,
            "gutter_strength": 0.5,
            "gutter_width": 1,
            "search_start": margin,
            "search_end": w - margin,
            "segments": segments,
            "selected_segment": segments[0] if segments else None,
            "fallback_used": True,
            "meets_min_width": False,
            "has_two_pages": False,
            "col_dark": None,
            "darkmask": None,
            "thresh": None,
            "widest_segment_width": 1,
            # Legacy compatibility fields
            "vertical_sums": None,
            "min_sum": 0,
            "avg_sum": 0,
        }
        return right_page, left_page, analysis
    
    (a1, b1), (a2, b2) = sorted(segs, key=lambda ab: ab[0])
    split_x = (b1 + a2) // 2
    
    left_page = image[:, :split_x]
    right_page = image[:, split_x:]
    
    if not return_analysis:
        return right_page, left_page
    
    # Build full analysis dict for compatibility
    analysis = {
        "gutter_x": split_x,
        "gutter_strength": 1.0,  # V2 doesn't calculate this exactly
        "gutter_width": a2 - b1,
        "search_start": margin,
        "search_end": w - margin,
        "segments": segments,
        "selected_segment": (b1, a2),
        "fallback_used": False,
        "meets_min_width": True,
        "has_two_pages": True,
        "col_dark": None,
        "darkmask": None,
        "thresh": None,
        "widest_segment_width": max(b - a + 1 for a, b in segments) if segments else 0,
        # Legacy compatibility fields
        "vertical_sums": None,
        "min_sum": 0,
        "avg_sum": 0,
    }
    
    return right_page, left_page, analysis