"""Tag removal operations for OCR pipeline."""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base import BaseProcessor


class TagRemovalProcessor(BaseProcessor):
    """Processor for removing generation number tags from genealogical documents."""

    def process(
        self,
        image: np.ndarray,
        thresh_dark: int = 110,
        row_sum_thresh: int = 200,
        dark_ratio: float = 0.7,
        min_area: int = 2000,
        max_area: int = 60000,
        min_aspect: float = 0.3,
        max_aspect: float = 1.8,
        **kwargs
    ) -> np.ndarray:
        """Remove generation number tags from an image.

        Args:
            image: Input image (BGR or grayscale)
            thresh_dark: Threshold for dark pixel detection (default: 110)
            row_sum_thresh: Threshold for row sum in band detection (default: 200)
            dark_ratio: Minimum ratio of dark pixels in tag (default: 0.7)
            min_area: Minimum tag area in pixels (default: 2000)
            max_area: Maximum tag area in pixels (default: 60000)
            min_aspect: Minimum aspect ratio (default: 0.3)
            max_aspect: Maximum aspect ratio (default: 1.8)
            **kwargs: Additional parameters

        Returns:
            Cleaned image with tags removed
        """
        self.validate_image(image)

        # Clear previous debug images
        self.clear_debug_images()

        # Pass processor instance for debug saving
        kwargs["_processor"] = self

        return remove_tags(
            image,
            thresh_dark=thresh_dark,
            row_sum_thresh=row_sum_thresh,
            dark_ratio=dark_ratio,
            min_area=min_area,
            max_area=max_area,
            min_aspect=min_aspect,
            max_aspect=max_aspect,
            **kwargs
        )


def find_tag_band(gray: np.ndarray, thresh_dark: int = 110, row_sum_thresh: int = 200) -> Optional[Tuple[int, int]]:
    """Locate the horizontal band that contains the generation tags.
    
    Args:
        gray: Grayscale image
        thresh_dark: Threshold for dark pixel detection
        row_sum_thresh: Threshold for row sum detection
        
    Returns:
        Tuple of (top, bottom) row indices, or None if no band found
    """
    dark = (gray < thresh_dark).astype(np.uint8)
    proj = dark.sum(axis=1)

    h = gray.shape[0]
    # consider only upper 40% of the page, skip very top margin
    rows = [
        i for i, v in enumerate(proj) if h * 0.02 < i < h * 0.4 and v > row_sum_thresh
    ]
    if not rows:
        return None

    return max(min(rows) - 10, 0), min(max(rows) + 10, h - 1)


def detect_tags_in_band(
    band: np.ndarray,
    thresh_dark: int = 110,
    dark_ratio: float = 0.7,
    min_area: int = 2000,
    max_area: int = 60000,
    min_aspect: float = 0.3,
    max_aspect: float = 1.8
) -> List[Tuple[int, int, int, int]]:
    """Return bounding boxes of tag squares inside the cropped band.
    
    Args:
        band: Cropped band image
        thresh_dark: Threshold for dark pixel detection
        dark_ratio: Minimum ratio of dark pixels in tag
        min_area: Minimum tag area in pixels
        max_area: Maximum tag area in pixels
        min_aspect: Minimum aspect ratio
        max_aspect: Maximum aspect ratio
        
    Returns:
        List of bounding boxes (x, y, w, h)
    """
    # binary: dark stuff → 1
    _, mask = cv2.threshold(band, thresh_dark, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=2,
    )

    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    boxes = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        aspect = w / h
        if not (min_aspect < aspect < max_aspect):
            continue
        if not (min_area < area < max_area):
            continue
        if np.mean(band[y : y + h, x : x + w] < thresh_dark) < dark_ratio:
            continue
        boxes.append((x, y, w, h))
    return boxes


def remove_tags(
    image: np.ndarray,
    thresh_dark: int = 110,
    row_sum_thresh: int = 200,
    dark_ratio: float = 0.7,
    min_area: int = 2000,
    max_area: int = 60000,
    min_aspect: float = 0.3,
    max_aspect: float = 1.8,
    **kwargs
) -> np.ndarray:
    """Remove generation number tags from genealogical documents.

    This function removes black generation-number squares from genealogical page scans
    by detecting them in the upper portion of the image and painting them white.

    Args:
        image: Input image (BGR or grayscale)
        thresh_dark: Threshold for dark pixel detection (default: 110)
        row_sum_thresh: Threshold for row sum in band detection (default: 200)
        dark_ratio: Minimum ratio of dark pixels in tag (default: 0.7)
        min_area: Minimum tag area in pixels (default: 2000)
        max_area: Maximum tag area in pixels (default: 60000)
        min_aspect: Minimum aspect ratio (default: 0.3)
        max_aspect: Maximum aspect ratio (default: 1.8)
        **kwargs: Additional parameters

    Returns:
        Cleaned image with tags removed
    """
    # Get processor instance if available for debug saving
    processor = kwargs.get("_processor", None)

    # Convert to BGR for processing if needed
    if len(image.shape) == 3:
        img_bgr = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Convert grayscale to BGR for consistent processing
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = image.copy()

    # Save input image
    if processor:
        processor.save_debug_image("input_image", img_bgr)
        processor.save_debug_image("input_grayscale", gray)

    # Find the tag band
    band_rows = find_tag_band(gray, thresh_dark, row_sum_thresh)
    if band_rows is None:
        if processor:
            processor.save_debug_image("no_band_found", img_bgr)
        # Return original image converted back to original format
        if len(image.shape) == 3:
            return img_bgr
        else:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    top, bottom = band_rows
    band = gray[top:bottom, :]

    # Save debug images for band detection
    if processor:
        # Visualize the detected band
        band_vis = img_bgr.copy()
        cv2.rectangle(band_vis, (0, top), (img_bgr.shape[1], bottom), (0, 255, 0), 3)
        processor.save_debug_image("detected_band", band_vis)
        processor.save_debug_image("band_crop", band)

    # Detect tags in the band
    boxes = detect_tags_in_band(
        band, thresh_dark, dark_ratio, min_area, max_area, min_aspect, max_aspect
    )

    # Save debug image showing detected tags before removal
    if processor and boxes:
        tags_vis = img_bgr.copy()
        for x, y, w, h in boxes:
            y_global = top + y
            cv2.rectangle(tags_vis, (x, y_global), (x + w, y_global + h), (0, 0, 255), 2)
        processor.save_debug_image("detected_tags", tags_vis)

    # Remove the tags
    cleaned = img_bgr.copy()
    removed_rects = []

    for x, y, w, h in boxes:
        y_global = top + y
        cv2.rectangle(
            cleaned,
            (x - 2, y_global - 2),
            (x + w + 2, y_global + h + 2),
            (255, 255, 255),
            -1,
        )
        removed_rects.append((x, y_global, w, h))
    
    # Debug: Print the result only in debug mode
    if processor and processor.config and getattr(processor.config, 'save_debug_images', False):
        if boxes:
            band_height = bottom - top
            print(f"    [DEBUG] Tag removal: Removed {len(boxes)} generation tags from band [{top}:{bottom}] (height: {band_height}px)")
        else:
            if band_rows is not None:
                print(f"    [DEBUG] Tag removal: No tags found in detected band [{top}:{bottom}]")
            else:
                print(f"    [DEBUG] Tag removal: No tag band detected (image may not contain generation tags)")

    # Save final result
    if processor:
        processor.save_debug_image("cleaned_result", cleaned)
        if boxes:
            processor.save_debug_image(
                "removal_overlay",
                cv2.addWeighted(img_bgr, 0.7, cleaned, 0.3, 0)
            )

    # Convert back to original format
    if len(image.shape) == 3:
        return cleaned
    else:
        return cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)