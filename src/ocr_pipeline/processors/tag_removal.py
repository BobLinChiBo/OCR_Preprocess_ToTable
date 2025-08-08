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
        method: str = "classic",
        # Classic method parameters
        thresh_dark: int = 110,
        row_sum_thresh: int = 200,
        dark_ratio: float = 0.7,
        min_area: int = 2000,
        max_area: int = 60000,
        min_aspect: float = 0.3,
        max_aspect: float = 1.8,
        morph_kernel_size: int = 5,
        morph_iterations: int = 2,
        # Auto whitefill broadmask method parameters
        band_top: float = 0.08,
        band_bottom: float = 0.42,
        rows_mode: str = "median",
        min_dark: float = 0.52,
        min_score: float = 0.55,
        reject_red: bool = True,
        nms_iou: float = 0.30,
        pad_px: int = 8,
        # Relative size parameters
        min_width_ratio: float = 0.015,
        max_width_ratio: float = 0.10,
        min_height_ratio: float = 0.015,
        max_height_ratio: float = 0.10,
        min_aspect_ratio: float = 0.3,
        max_aspect_ratio: float = 2.5,
        # Kernel size parameters
        glyph_kernel_size: int = 3,
        mask_close_kernel_size: int = 11,
        mask_open_kernel_size: int = 3,
        mask_dilate_kernel_size: int = 3,
        **kwargs
    ) -> np.ndarray:
        """Remove generation number tags from an image.

        Args:
            image: Input image (BGR or grayscale)
            method: Detection method - "classic" or "auto_whitefill_broadmask" (default: "classic")
            
            Classic method parameters:
            thresh_dark: Threshold for dark pixel detection (default: 110)
            row_sum_thresh: Threshold for row sum in band detection (default: 200)
            dark_ratio: Minimum ratio of dark pixels in tag (default: 0.7)
            min_area: Minimum tag area in pixels (default: 2000)
            max_area: Maximum tag area in pixels (default: 60000)
            min_aspect: Minimum aspect ratio (default: 0.3)
            max_aspect: Maximum aspect ratio (default: 1.8)
            morph_kernel_size: Size of morphological kernel (default: 5)
            morph_iterations: Number of morphological closing iterations (default: 2)
            
            Auto whitefill broadmask method parameters:
            band_top: Top fraction of page height (default: 0.08)
            band_bottom: Bottom fraction of page height (default: 0.42)
            rows_mode: Row geometry filter - "median", "theilsen", or "none" (default: "median")
            min_score: Minimum candidate confidence (default: 0.55)
            reject_red: Disable HSV red-stamp rejection (default: True)
            nms_iou: NMS IoU threshold (default: 0.30)
            pad_px: Padding pixels for mask (default: 8)
            **kwargs: Additional parameters

        Returns:
            Cleaned image with tags removed
        """
        self.validate_image(image)

        # Clear previous debug images
        self.clear_debug_images()

        # Pass processor instance for debug saving
        kwargs["_processor"] = self

        if method == "auto_whitefill_broadmask":
            return remove_tags_auto_whitefill(
                image,
                band_top=band_top,
                band_bottom=band_bottom,
                rows_mode=rows_mode,
                min_dark=min_dark,
                min_score=min_score,
                reject_red=reject_red,
                nms_iou=nms_iou,
                pad_px=pad_px,
                min_width_ratio=min_width_ratio,
                max_width_ratio=max_width_ratio,
                min_height_ratio=min_height_ratio,
                max_height_ratio=max_height_ratio,
                min_aspect_ratio=min_aspect_ratio,
                max_aspect_ratio=max_aspect_ratio,
                glyph_kernel_size=glyph_kernel_size,
                mask_close_kernel_size=mask_close_kernel_size,
                mask_open_kernel_size=mask_open_kernel_size,
                mask_dilate_kernel_size=mask_dilate_kernel_size,
                **kwargs
            )
        else:  # classic method
            return remove_tags(
                image,
                thresh_dark=thresh_dark,
                row_sum_thresh=row_sum_thresh,
                dark_ratio=dark_ratio,
                min_area=min_area,
                max_area=max_area,
                min_aspect=min_aspect,
                max_aspect=max_aspect,
                morph_kernel_size=morph_kernel_size,
                morph_iterations=morph_iterations,
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
    max_aspect: float = 1.8,
    morph_kernel_size: int = 5,
    morph_iterations: int = 2
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
        morph_kernel_size: Size of morphological kernel (default: 5)
        morph_iterations: Number of morphological closing iterations (default: 2)
        
    Returns:
        List of bounding boxes (x, y, w, h)
    """
    # binary: dark stuff -> 1
    _, mask = cv2.threshold(band, thresh_dark, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)),
        iterations=morph_iterations,
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
    morph_kernel_size: int = 5,
    morph_iterations: int = 2,
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
        morph_kernel_size: Size of morphological kernel (default: 5)
        morph_iterations: Number of morphological closing iterations (default: 2)
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
        # Visualize the detected band with annotations
        band_vis = img_bgr.copy()
        W = img_bgr.shape[1]
        
        # Draw band with transparency
        overlay = band_vis.copy()
        cv2.rectangle(overlay, (0, top), (W, bottom), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.1, band_vis, 0.9, 0, band_vis)
        
        # Draw band borders
        cv2.line(band_vis, (0, top), (W, top), (0, 255, 0), 3)
        cv2.line(band_vis, (0, bottom), (W, bottom), (0, 255, 0), 3)
        
        # Add annotations
        cv2.putText(band_vis, f"Classic Method Tag Band", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(band_vis, f"Height: {bottom-top}px (rows {top}-{bottom})", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(band_vis, f"Dark threshold: {thresh_dark}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        processor.save_debug_image("detected_band", band_vis)
        processor.save_debug_image("band_crop", band)

    # Detect tags in the band
    boxes = detect_tags_in_band(
        band, thresh_dark, dark_ratio, min_area, max_area, min_aspect, max_aspect,
        morph_kernel_size, morph_iterations
    )

    # Save debug image showing detected tags before removal
    if processor and boxes:
        tags_vis = img_bgr.copy()
        for i, (x, y, w, h) in enumerate(boxes, start=1):
            y_global = top + y
            cv2.rectangle(tags_vis, (x, y_global), (x + w, y_global + h), (0, 0, 255), 2)
            # Add tag number
            cv2.putText(tags_vis, f"#{i}", (x + 2, y_global - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # Add dimensions and aspect ratio
            aspect = w / h
            cv2.putText(tags_vis, f"{w}x{h}", (x + 2, y_global + h - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(tags_vis, f"AR:{aspect:.2f}", (x + 2, y_global + h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Add summary text
        cv2.putText(tags_vis, f"Classic Method: {len(boxes)} tags detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(tags_vis, f"Band: rows {top}-{bottom}", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(tags_vis, f"Filters: area[{min_area}-{max_area}] AR[{min_aspect:.1f}-{max_aspect:.1f}]", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
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
            # Create overlay showing removed areas
            processor.save_debug_image(
                "removal_overlay",
                cv2.addWeighted(img_bgr, 0.7, cleaned, 0.3, 0)
            )
            
            # Create side-by-side comparison
            h, w = img_bgr.shape[:2]
            comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
            comparison[:, :w] = img_bgr
            comparison[:, w:] = cleaned
            
            # Add dividing line
            cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 2)
            
            # Add labels
            cv2.putText(comparison, "BEFORE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(comparison, "AFTER", (w + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(comparison, f"Classic Method: Removed {len(boxes)} tags", (w//2 - 150, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            processor.save_debug_image("before_after_comparison", comparison)

    # Convert back to original format
    if len(image.shape) == 3:
        return cleaned
    else:
        return cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)


# =========================================================
# Auto Whitefill Broadmask Method Functions
# =========================================================

def nms(boxes: List[Tuple[int, int, int, int]], iou_thr: float = 0.30) -> List[Tuple[int, int, int, int]]:
    """Non-maximum suppression on (x,y,w,h) boxes (IoU).
    
    Args:
        boxes: List of bounding boxes (x, y, w, h)
        iou_thr: IoU threshold for suppression
        
    Returns:
        List of filtered boxes after NMS
    """
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    keep = []
    
    def iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0
    
    for b in boxes:
        if all(iou(b, k) <= iou_thr for k in keep):
            keep.append(b)
    return keep


def theil_sen_line(points: np.ndarray) -> Tuple[float, float]:
    """
    Robust line fit y = a*x + b using Theil–Sen estimator.
    
    Args:
        points: array of shape (N,2)
        
    Returns:
        (a, b): slope and intercept. If degenerate, returns (0.0, median(y))
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 2:
        return 0.0, (float(pts[0, 1]) if pts.size else 0.0)
    
    x = pts[:, 0]
    y = pts[:, 1]
    slopes = []
    
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = x[j] - x[i]
            if abs(dx) < 1e-6:
                continue
            slopes.append((y[j] - y[i]) / dx)
    
    a = float(np.median(slopes)) if slopes else 0.0
    b = float(np.median(y - a * x))
    return a, b


def detect_generation_tags_auto(
    img_bgr: np.ndarray,
    band: Tuple[float, float] = (0.08, 0.42),
    pad: int = 24,
    min_dark: float = 0.52,
    reject_red: bool = True,
    rows_mode: str = "median",
    min_score: float = 0.55,
    nms_iou: float = 0.30,
    min_width_ratio: float = 0.015,
    max_width_ratio: float = 0.10,
    min_height_ratio: float = 0.015,
    max_height_ratio: float = 0.10,
    min_aspect_ratio: float = 0.3,
    max_aspect_ratio: float = 2.5,
    glyph_kernel_size: int = 3,
    _debug: bool = False,
    _processor: Optional[Any] = None,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect generation tags using auto method.
    
    Returns a list of (x,y,w,h) boxes in page coordinates.
    Single-pass, no fallbacks, no row-by-count.
    """
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr

    # 1) band + padding
    y1, y2 = int(band[0] * H), int(band[1] * H)
    y1 = max(0, min(H - 1, y1))
    y2 = max(y1 + 1, min(H, y2))
    band_g = gray[y1:y2].copy()
    
    if _debug:
        print(f"      Band: y1={y1}, y2={y2}, height={y2-y1}")
        print(f"      Min dark ratio threshold: {min_dark}")
    
    if len(img_bgr.shape) == 3:
        band_hsv = cv2.cvtColor(img_bgr[y1:y2], cv2.COLOR_BGR2HSV)
    else:
        band_hsv = None

    pad_band = cv2.copyMakeBorder(band_g, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

    # 2) MSER on the inverted band (treat dark as bright)
    mser = cv2.MSER_create()
    # Try to set params if available (OpenCV version can differ)
    for setter, val in (("setDelta", 5), ("setMinArea", 1500), ("setMaxArea", 60000)):
        if hasattr(mser, setter):
            getattr(mser, setter)(val)
    regions, _ = mser.detectRegions(255 - pad_band)
    
    # Debug: Visualize all MSER regions before filtering
    if _processor:
        mser_vis = cv2.cvtColor(band_g, cv2.COLOR_GRAY2BGR)
        for i, pts in enumerate(regions[:100]):  # Limit to first 100 for performance
            x, y, w, h = cv2.boundingRect(pts)
            x -= pad
            y -= pad
            if x >= 0 and y >= 0 and x + w <= band_g.shape[1] and y + h <= band_g.shape[0]:
                color = (0, 255, 0) if i < 50 else (0, 165, 255)  # Green for first 50, orange for rest
                cv2.rectangle(mser_vis, (x, y), (x + w, y + h), color, 1)
        # Add text overlay with MSER stats
        cv2.putText(mser_vis, f"MSER Regions: {len(regions)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(mser_vis, f"Band: {y1}-{y2} ({y2-y1}px)", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        _processor.save_debug_image("mser_all_regions", mser_vis)

    # Calculate absolute pixel sizes from ratios
    img_width = W
    img_height = H
    min_width = int(img_width * min_width_ratio)
    max_width = int(img_width * max_width_ratio)
    min_height = int(img_height * min_height_ratio)
    max_height = int(img_height * max_height_ratio)
    
    if _debug:
        print(f"      Size constraints: width [{min_width}-{max_width}]px, height [{min_height}-{max_height}]px")
        print(f"      Aspect ratio: [{min_aspect_ratio:.2f}-{max_aspect_ratio:.2f}]")
    
    def is_shape(w, h):
        a = w / max(h, 1)
        # Check size constraints based on relative ratios
        size_ok = (min_width <= w <= max_width and min_height <= h <= max_height)
        # Check aspect ratio
        aspect_ok = (min_aspect_ratio <= a <= max_aspect_ratio)
        return size_ok and aspect_ok

    # 3) filter candidates
    cand = []
    rejected_regions = []  # Store rejected regions with reasons for visualization
    total_regions = len(regions)
    if _debug:
        print(f"      MSER found {total_regions} regions")
        rejected_bounds = 0
        rejected_shape = 0
        rejected_dark = 0
        rejected_color = 0
    
    for pts in regions:
        x, y, w, h = cv2.boundingRect(pts)
        x -= pad
        y -= pad
        rejection_reason = None
        
        if x < 0 or y < 0 or x + w > band_g.shape[1] or y + h > band_g.shape[0]:
            if _debug: rejected_bounds += 1
            rejection_reason = "bounds"
        elif not is_shape(w, h):
            if _debug: rejected_shape += 1
            rejection_reason = "shape"
        else:
            roi = band_g[y:y + h, x:x + w]
            dark_ratio = float((roi < 160).mean())
            if dark_ratio < min_dark:
                if _debug: rejected_dark += 1
                rejection_reason = f"light:{dark_ratio:.2f}"
            # optional color reject (auto-bypass when region is very dark overall)
            elif reject_red and band_hsv is not None:
                patch = band_hsv[y:y + h, x:x + w]
                sat = float(np.mean(patch[:, :, 1]))
                hue = float(np.mean(patch[:, :, 0]))
                is_redish = (sat > 70 and (hue < 15 or hue > 165))
                if is_redish and dark_ratio < 0.70:  # if truly dark capsule, keep; else drop
                    if _debug: rejected_color += 1
                    rejection_reason = "red"
        
        if rejection_reason:
            if x >= 0 and y >= 0 and x + w <= band_g.shape[1] and y + h <= band_g.shape[0]:
                rejected_regions.append((x, y, w, h, rejection_reason))
        else:
            cand.append((x, y, w, h))
    
    if _debug:
        print(f"      Filtering: {rejected_bounds} out-of-bounds, {rejected_shape} wrong shape, {rejected_dark} too light, {rejected_color} red stamps")
        print(f"      Candidates after filtering: {len(cand)}")
    
    # Debug: Visualize rejected regions with color coding
    if _processor and rejected_regions:
        reject_vis = cv2.cvtColor(band_g, cv2.COLOR_GRAY2BGR)
        
        # Color mapping for rejection reasons
        colors = {
            "bounds": (128, 128, 128),  # Gray for out of bounds
            "shape": (0, 0, 255),        # Red for wrong shape
            "red": (0, 128, 255),        # Orange for red stamps
        }
        
        for x, y, w, h, reason in rejected_regions[:50]:  # Limit to 50 for clarity
            # Get color based on reason (light reasons start with "light:")
            if reason.startswith("light:"):
                color = (255, 0, 255)  # Magenta for too light
                label = reason
            else:
                color = colors.get(reason, (255, 255, 0))  # Default cyan
                label = reason
            
            cv2.rectangle(reject_vis, (x, y), (x + w, y + h), color, 2)
            # Add small label
            cv2.putText(reject_vis, label, (x, max(y-5, 15)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw accepted candidates in green
        for x, y, w, h in cand:
            cv2.rectangle(reject_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(reject_vis, "OK", (x, max(y-5, 15)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Add legend
        legend_y = 30
        cv2.putText(reject_vis, f"Total: {total_regions} | Accepted: {len(cand)} | Rejected: {len(rejected_regions)}", 
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        legend_y += 25
        cv2.putText(reject_vis, "Green=Accepted", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        legend_y += 20
        cv2.putText(reject_vis, "Red=Wrong Shape", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        legend_y += 20
        cv2.putText(reject_vis, "Magenta=Too Light", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        legend_y += 20
        cv2.putText(reject_vis, "Orange=Red Stamp", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)
        
        _processor.save_debug_image("filtering_visualization", reject_vis)

    # 4) NMS (band coords)
    cand = nms(cand, iou_thr=nms_iou)
    if not cand:
        return []

    # 5) confidence score (no quotas)
    scored = []
    score_details = []  # Store scoring details for visualization
    if _debug:
        print(f"      Scoring {len(cand)} candidates (min_score={min_score}):")
    
    for i, (x, y, w, h) in enumerate(cand):
        roi_g = band_g[y:y + h, x:x + w]
        dark_score = float((roi_g < 160).mean())
        # count bright blobs (approx. white glyphs)
        _, bright = cv2.threshold(roi_g, 185, 255, cv2.THRESH_BINARY)
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (glyph_kernel_size, glyph_kernel_size)), 1)
        n2, _, st2, _ = cv2.connectedComponentsWithStats(bright, 8)
        glyphs = sum(1 for j in range(1, n2) if st2[j, cv2.CC_STAT_AREA] > 60)
        glyphs_eff = min(glyphs, 6)  # clamp to resist watermarks/noise
        glyph_score = 1.0 / (1 + abs(glyphs_eff - 3))
        a = w / max(h, 1)
        shape_score = 1.0 if (0.60 <= a <= 1.60 or 0.28 <= a <= 0.62) else 0.5
        score = 0.60 * dark_score + 0.25 * glyph_score + 0.15 * shape_score
        
        score_details.append({
            'bbox': (x, y, w, h),
            'score': score,
            'dark_score': dark_score,
            'glyph_score': glyph_score,
            'shape_score': shape_score,
            'glyphs': glyphs,
            'aspect': a
        })
        
        if _debug and i < 5:  # Show first 5 for brevity
            print(f"        [{i}] pos=({x},{y}) size=({w}x{h}) dark={dark_score:.2f} glyph={glyph_score:.2f} shape={shape_score:.2f} -> score={score:.2f}")
        
        if score >= min_score:
            scored.append((score, x, y, w, h))
    
    if not scored:
        return []
    
    # Debug: Visualize scoring results
    if _processor and score_details:
        score_vis = cv2.cvtColor(band_g, cv2.COLOR_GRAY2BGR)
        
        for detail in score_details:
            x, y, w, h = detail['bbox']
            score = detail['score']
            
            # Color based on acceptance
            if score >= min_score:
                color = (0, 255, 0)  # Green for accepted
                thickness = 2
            else:
                color = (0, 165, 255)  # Orange for below threshold
                thickness = 1
            
            cv2.rectangle(score_vis, (x, y), (x + w, y + h), color, thickness)
            
            # Add score label
            label = f"{score:.2f}"
            cv2.putText(score_vis, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Add detailed scores inside box (if large enough)
            if w > 60 and h > 40:
                cv2.putText(score_vis, f"D:{detail['dark_score']:.1f}", (x + 2, y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                cv2.putText(score_vis, f"G:{detail['glyph_score']:.1f}", (x + 2, y + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                cv2.putText(score_vis, f"S:{detail['shape_score']:.1f}", (x + 2, y + 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Add header info
        cv2.putText(score_vis, f"Scoring Results (threshold={min_score:.2f})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(score_vis, f"Accepted: {len(scored)} / {len(score_details)}", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(score_vis, "D=Dark, G=Glyph, S=Shape", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        _processor.save_debug_image("scoring_visualization", score_vis)

    boxes = [(x, y, w, h) for (s, x, y, w, h) in scored]

    # 6) geometric row consistency (no counting)
    if rows_mode and rows_mode.lower() != "none" and len(boxes) >= 2:
        centers = np.array([[x + w / 2.0, y + h / 2.0] for (x, y, w, h) in boxes], dtype=float)
        h_med = float(np.median([h for (_, _, _, h) in boxes]))
        tau = max(0.25 * h_med, 28.0)
        
        if rows_mode.lower() == "theilsen":
            a, b = theil_sen_line(centers)
            def on_row(bx):
                x, y, w, h = bx
                yc = y + h / 2.0
                y_fit = a * (x + w / 2.0) + b
                return abs(yc - y_fit) <= tau
            boxes = [b for b in boxes if on_row(b)]
        else:
            y_centers = np.array([y + h / 2.0 for (_, y, _, h) in boxes])
            y0 = float(np.median(y_centers))
            boxes = [b for b in boxes if abs((b[1] + b[3] / 2.0) - y0) <= tau]

    # Map band coords to page coords
    boxes_page = [(x, y1 + y, w, h) for (x, y, w, h) in boxes]
    return boxes_page


def broad_capsule_mask(
    patch_gray: np.ndarray, 
    pad_px: int = 8,
    close_kernel_size: int = 11,
    open_kernel_size: int = 3,
    dilate_kernel_size: int = 3,
    _processor: Optional[Any] = None,
    _mask_index: int = 0
) -> np.ndarray:
    """
    Broad but safe mask:
      - pad patch to keep neighbors protected
      - union = Otsu INV ∪ INV@190 ∪ adaptive INV
      - CLOSE 11x11 to seal rounded rectangle
      - OPEN 3x3 to drop specks
      - keep component closest to center
      - remove padding; +1 px dilation
    """
    PAD = int(pad_px)
    g = cv2.copyMakeBorder(patch_gray, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=255)
    gb = cv2.GaussianBlur(g, (5, 5), 0)
    _, m1 = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, m2 = cv2.threshold(gb, 190, 255, cv2.THRESH_BINARY_INV)
    m3 = cv2.adaptiveThreshold(gb, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 5)
    mu = cv2.bitwise_or(m1, cv2.bitwise_or(m2, m3))
    
    # Debug: Save mask creation steps
    if _processor and _mask_index <= 3:  # Only for first 3 masks
        # Create a composite visualization of mask steps
        h, w = g.shape
        composite = np.zeros((h * 2, w * 3), dtype=np.uint8)
        
        # Top row: original, Otsu, fixed threshold
        composite[:h, :w] = g  # Padded grayscale
        composite[:h, w:w*2] = m1  # Otsu threshold
        composite[:h, w*2:] = m2  # Fixed threshold
        
        # Bottom row: adaptive, union, final after morphology
        composite[h:, :w] = m3  # Adaptive threshold
        composite[h:, w:w*2] = mu  # Union of all masks
        # Final will be added after morphology
        
        _processor.save_debug_image(f"mask_steps_{_mask_index}_pre", composite)
    # kill artificial border
    mu[:PAD, :] = 0
    mu[-PAD:, :] = 0
    mu[:, :PAD] = 0
    mu[:, -PAD:] = 0
    mu = cv2.morphologyEx(mu, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size)), 1)
    mu = cv2.morphologyEx(mu, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size)), 1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mu, 8)
    mk = np.zeros_like(mu)
    if num > 1:
        cy, cx = (mu.shape[0] / 2.0, mu.shape[1] / 2.0)
        best, lab = 1e18, 0
        for k in range(1, num):
            x, y, w, h, area = stats[k]
            yc, xc = y + h / 2.0, x + w / 2.0
            d = (yc - cy) ** 2 + (xc - cx) ** 2
            if d < best:
                best, lab = d, k
        mk[labels == lab] = 255
    else:
        mk = mu

    mk = mk[PAD:-PAD, PAD:-PAD]
    mk = cv2.dilate(mk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size)), 1)
    return mk


def remove_tags_auto_whitefill(
    image: np.ndarray,
    band_top: float = 0.08,
    band_bottom: float = 0.42,
    rows_mode: str = "median",
    min_dark: float = 0.52,
    min_score: float = 0.55,
    reject_red: bool = True,
    nms_iou: float = 0.30,
    pad_px: int = 8,
    min_width_ratio: float = 0.015,
    max_width_ratio: float = 0.10,
    min_height_ratio: float = 0.015,
    max_height_ratio: float = 0.10,
    min_aspect_ratio: float = 0.3,
    max_aspect_ratio: float = 2.5,
    glyph_kernel_size: int = 3,
    mask_close_kernel_size: int = 11,
    mask_open_kernel_size: int = 3,
    mask_dilate_kernel_size: int = 3,
    **kwargs
) -> np.ndarray:
    """
    Remove generation tags using auto whitefill broadmask method.
    
    Returns: cleaned image with tags removed.
    White-fill removal (no inpainting) using the broad capsule mask.
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
    
    # Detect tags
    _debug = processor and processor.config and getattr(processor.config, 'verbose', False)
    boxes = detect_generation_tags_auto(
        img_bgr,
        band=(band_top, band_bottom),
        min_dark=min_dark,
        reject_red=reject_red,
        rows_mode=rows_mode,
        min_score=min_score,
        nms_iou=nms_iou,
        min_width_ratio=min_width_ratio,
        max_width_ratio=max_width_ratio,
        min_height_ratio=min_height_ratio,
        max_height_ratio=max_height_ratio,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
        glyph_kernel_size=glyph_kernel_size,
        _debug=_debug,
        _processor=processor,
    )
    
    # Debug output
    if _debug:
        print(f"    [DEBUG] Auto detection: Found {len(boxes)} tag(s)")
    
    # Save debug image showing detected tags before removal
    if processor and boxes:
        tags_vis = img_bgr.copy()
        for i, (x, y, w, h) in enumerate(boxes, start=1):
            cv2.rectangle(tags_vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # Add tag number
            cv2.putText(tags_vis, f"#{i}", (x + 2, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # Add dimensions
            cv2.putText(tags_vis, f"{w}x{h}", (x + 2, y + h - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add summary text
        cv2.putText(tags_vis, f"Detected {len(boxes)} generation tags", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(tags_vis, f"Method: auto_whitefill_broadmask", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        processor.save_debug_image("detected_tags", tags_vis)
        
        # Also visualize the detection band with annotations
        H, W = img_bgr.shape[:2]
        band_vis = img_bgr.copy()
        y1, y2 = int(band_top * H), int(band_bottom * H)
        
        # Draw band with transparency
        overlay = band_vis.copy()
        cv2.rectangle(overlay, (0, y1), (W, y2), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.1, band_vis, 0.9, 0, band_vis)
        
        # Draw band borders
        cv2.line(band_vis, (0, y1), (W, y1), (0, 255, 0), 3)
        cv2.line(band_vis, (0, y2), (W, y2), (0, 255, 0), 3)
        
        # Add text annotations
        cv2.putText(band_vis, f"Detection Band: {band_top:.0%} - {band_bottom:.0%}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(band_vis, f"Band Height: {y2-y1}px ({(y2-y1)/H:.1%} of image)", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(band_vis, f"Y-range: [{y1}, {y2}]", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        processor.save_debug_image("detected_band", band_vis)
    
    # Apply removal with broad capsule mask
    clean = img_bgr.copy()
    gray_clean = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
    
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        patch = gray_clean[y:y + h, x:x + w]
        mk = broad_capsule_mask(
            patch, 
            pad_px=pad_px,
            close_kernel_size=mask_close_kernel_size,
            open_kernel_size=mask_open_kernel_size,
            dilate_kernel_size=mask_dilate_kernel_size,
            _processor=processor if i <= 3 else None,  # Pass processor for first 3 masks
            _mask_index=i
        )
        
        # Save individual mask for debug with annotations
        if processor and i <= 5:  # Save first 5 masks only
            # Create annotated mask visualization
            mask_vis = cv2.cvtColor(mk, cv2.COLOR_GRAY2BGR)
            cv2.putText(mask_vis, f"Tag {i}", (5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(mask_vis, f"Size: {w}x{h}", (5, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            processor.save_debug_image(f"mask_{i}_final", mask_vis)
        
        # WHITE fill
        region = clean[y:y + h, x:x + w]
        if region.ndim == 3:
            region[mk > 0] = (255, 255, 255)
        else:
            region[mk > 0] = 255
        clean[y:y + h, x:x + w] = region
    
    # Debug: Print the result only in debug mode
    if processor and processor.config and getattr(processor.config, 'save_debug_images', False):
        if boxes:
            print(f"    [DEBUG] Tag removal (auto_whitefill): Removed {len(boxes)} generation tags")
        else:
            print(f"    [DEBUG] Tag removal (auto_whitefill): No tags detected")
    
    # Save final result
    if processor:
        processor.save_debug_image("cleaned_result", clean)
        if boxes:
            # Create overlay showing removed areas
            processor.save_debug_image(
                "removal_overlay",
                cv2.addWeighted(img_bgr, 0.7, clean, 0.3, 0)
            )
            
            # Create side-by-side comparison
            h, w = img_bgr.shape[:2]
            comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
            comparison[:, :w] = img_bgr
            comparison[:, w:] = clean
            
            # Add dividing line
            cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 2)
            
            # Add labels
            cv2.putText(comparison, "BEFORE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(comparison, "AFTER", (w + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(comparison, f"Removed {len(boxes)} tags", (w//2 - 100, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            processor.save_debug_image("before_after_comparison", comparison)
    
    # Convert back to original format
    if len(image.shape) == 3:
        return clean
    else:
        return cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)