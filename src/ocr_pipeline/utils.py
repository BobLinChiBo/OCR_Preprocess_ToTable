"""Simple utilities for OCR pipeline."""

from pathlib import Path
from typing import Dict, List, Tuple, Union
from functools import lru_cache

import cv2
import numpy as np


def load_image(image_path: Path) -> np.ndarray:
    """Load image from file."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image


def save_image(image: np.ndarray, output_path: Path) -> None:
    """Save image to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from directory."""
    extensions = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
    image_files = []

    for ext in extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))

    return sorted(image_files)


def split_two_page_image(
    image: np.ndarray,
    gutter_start: float = 0.4,
    gutter_end: float = 0.6,
    min_gutter_width: int = 50,
    return_analysis: bool = False,
) -> tuple:
    """Split a two-page scanned image into separate pages.

    Args:
        image: Input two-page image
        gutter_start: Start of gutter search region (0.0-1.0)
        gutter_end: End of gutter search region (0.0-1.0)
        min_gutter_width: Minimum gutter width for validation
        return_analysis: If True, returns detailed analysis information

    Returns:
        tuple: (left_page, right_page) or (left_page, right_page, analysis) if return_analysis=True
    """
    height, width = image.shape[:2]

    # Search for gutter in the middle region
    search_start = int(width * gutter_start)
    search_end = int(width * gutter_end)
    search_region = image[:, search_start:search_end]

    # Find the darkest vertical line (gutter)
    gray = (
        cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        if len(search_region.shape) == 3
        else search_region
    )
    vertical_sums = np.sum(gray, axis=0)
    gutter_offset = np.argmin(vertical_sums)
    gutter_x = search_start + gutter_offset

    # Split the image
    left_page = image[:, :gutter_x]
    right_page = image[:, gutter_x:]

    if not return_analysis:
        return left_page, right_page

    # Calculate analysis information
    min_sum = vertical_sums[gutter_offset]
    avg_sum = np.mean(vertical_sums)
    gutter_strength = (avg_sum - min_sum) / avg_sum if avg_sum > 0 else 0

    # Find gutter width (continuous dark region)
    threshold = min_sum * 1.2  # 20% tolerance
    left_edge = gutter_offset
    right_edge = gutter_offset

    # Expand left
    while left_edge > 0 and vertical_sums[left_edge - 1] <= threshold:
        left_edge -= 1

    # Expand right
    while (
        right_edge < len(vertical_sums) - 1
        and vertical_sums[right_edge + 1] <= threshold
    ):
        right_edge += 1

    gutter_width = right_edge - left_edge + 1

    analysis = {
        "gutter_x": gutter_x,
        "gutter_strength": gutter_strength,
        "gutter_width": gutter_width,
        "search_start": search_start,
        "search_end": search_end,
        "vertical_sums": vertical_sums,
        "min_sum": min_sum,
        "avg_sum": avg_sum,
        "meets_min_width": gutter_width >= min_gutter_width,
    }

    return left_page, right_page, analysis


def deskew_image(
    image: np.ndarray,
    angle_range: int = 15,
    angle_step: float = 0.5,
    min_angle_correction: float = 0.5,
    return_analysis_data: bool = False,
) -> tuple:
    """Deskew image using optimized coarse-to-fine histogram variance optimization.

    Performance optimizations:
    - Coarse-to-fine search: reduces iterations from ~100 to ~40
    - Multi-resolution: coarse search on downsampled image (16x faster)
    - Fast interpolation for search phase, high quality for final rotation
    - Early termination when improvement is minimal

    Philosophy: Well-aligned text creates sharp peaks in horizontal projection.
    This approach maximizes the variance of adjacent histogram differences,
    which occurs when text lines are perfectly horizontal.

    Args:
        image: Input image to deskew
        angle_range: Maximum rotation angle in degrees (±)
        angle_step: Step size for fine search in degrees
        min_angle_correction: Minimum angle threshold to apply correction
        return_analysis_data: If True, return additional analysis data for visualization
    Returns:
        tuple: (deskewed_image, detected_angle) or (deskewed_image, detected_angle, analysis_data)
    """
    # Convert to binary for optimal histogram analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    def histogram_variance_score(
        binary_img: np.ndarray, angle: float, use_fast_interpolation: bool = True
    ) -> float:
        """Calculate sharpness of horizontal projection after rotation.
        When text is well-aligned, the horizontal projection shows sharp peaks
        (text rows) and valleys (white space). This maximizes the variance of
        adjacent differences in the projection histogram.
        """
        h, w = binary_img.shape
        center = (w // 2, h // 2)
        # Rotate binary image - use fast interpolation for search phase
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        interpolation = cv2.INTER_LINEAR if use_fast_interpolation else cv2.INTER_CUBIC
        rotated = cv2.warpAffine(
            binary_img,
            rotation_matrix,
            (w, h),
            flags=interpolation,
            borderMode=cv2.BORDER_REPLICATE,
        )
        # Calculate horizontal projection (sum of pixels in each row)
        horizontal_projection = np.sum(rotated, axis=1)
        # Calculate variance of adjacent differences (sharpness measure)
        # Sharp transitions between text and whitespace maximize this value
        differences = horizontal_projection[1:] - horizontal_projection[:-1]
        return np.sum(differences**2)

    # Phase 1: Coarse search on downsampled image for speed
    # Downsample by 4x for ~16x speedup (4x4 pixels -> 1 pixel)
    h, w = binary.shape
    small_binary = cv2.resize(binary, (w // 4, h // 4), interpolation=cv2.INTER_AREA)

    # Coarse search with 1-degree steps
    coarse_step = 1.0
    coarse_angles = np.arange(-angle_range, angle_range + coarse_step, coarse_step)
    coarse_scores = []

    for angle in coarse_angles:
        score = histogram_variance_score(
            small_binary, angle, use_fast_interpolation=True
        )
        coarse_scores.append(score)

        # Early termination: if we have enough samples and last few scores are decreasing
        if len(coarse_scores) >= 5:
            recent_scores = coarse_scores[-5:]
            if all(
                recent_scores[i] >= recent_scores[i + 1]
                for i in range(len(recent_scores) - 1)
            ):
                # Scores are consistently decreasing, likely past the optimum
                break

    # Find best coarse angle
    best_coarse_idx = np.argmax(coarse_scores)
    best_coarse_angle = coarse_angles[best_coarse_idx]

    # Phase 2: Fine search around best coarse candidate on full resolution
    # Search ±2 degrees around best coarse angle with original step size
    fine_range = 2.0
    fine_start = max(-angle_range, best_coarse_angle - fine_range)
    fine_end = min(angle_range, best_coarse_angle + fine_range)
    fine_angles = np.arange(fine_start, fine_end + angle_step, angle_step)

    fine_scores = []
    for angle in fine_angles:
        score = histogram_variance_score(binary, angle, use_fast_interpolation=True)
        fine_scores.append(score)

    # Find best fine angle
    best_fine_idx = np.argmax(fine_scores)
    best_angle = fine_angles[best_fine_idx]

    # Apply rotation only if significant
    if abs(best_angle) < min_angle_correction:
        if return_analysis_data:
            # Create analysis data even when no rotation is applied
            all_angles = list(coarse_angles[:len(coarse_scores)]) + list(fine_angles)
            all_scores = coarse_scores + fine_scores
            analysis_data = {
                "has_lines": True,
                "rotation_angle": 0.0,
                "angles": all_angles,
                "scores": all_scores,
                "coarse_angles": list(coarse_angles[:len(coarse_scores)]),
                "coarse_scores": coarse_scores,
                "fine_angles": list(fine_angles),
                "fine_scores": fine_scores,
                "best_score": max(all_scores) if all_scores else 0,
                "confidence": 1.0,
                "will_rotate": False,
                "method": "histogram_variance",
                "gray": gray,
                "binary": binary,
                "line_count": len(all_angles),
                "angle_std": 0.0,
            }
            return image, 0.0, analysis_data
        return image, 0.0

    # Apply optimal rotation to original color image with high quality interpolation
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    deskewed = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,  # High quality for final result
        borderMode=cv2.BORDER_REPLICATE,
    )
    
    if return_analysis_data:
        # Create comprehensive analysis data for visualization
        all_angles = list(coarse_angles[:len(coarse_scores)]) + list(fine_angles)
        all_scores = coarse_scores + fine_scores
        
        # Calculate confidence based on score distribution
        score_std = np.std(all_scores)
        best_score = max(all_scores) if all_scores else 0
        confidence = min(1.0, best_score / (np.mean(all_scores) + score_std + 1e-6))
        
        analysis_data = {
            "has_lines": True,
            "rotation_angle": best_angle,
            "angles": all_angles,
            "scores": all_scores,
            "coarse_angles": list(coarse_angles[:len(coarse_scores)]),
            "coarse_scores": coarse_scores,
            "fine_angles": list(fine_angles),
            "fine_scores": fine_scores,
            "best_score": best_score,
            "confidence": confidence,
            "angle_std": score_std,
            "will_rotate": True,
            "method": "histogram_variance",
            "gray": gray,
            "binary": binary,
            "line_count": len(all_angles),
        }
        return deskewed, best_angle, analysis_data
    
    return deskewed, best_angle


def visualize_detected_lines(
    image, h_lines, v_lines, line_color=(0, 0, 255), line_thickness=2
) -> np.ndarray:
    """Create visualization of detected lines on the image."""
    vis_image = image.copy()

    # Draw horizontal lines in red
    for x1, y1, x2, y2 in h_lines:
        cv2.line(vis_image, (x1, y1), (x2, y2), line_color, line_thickness)

    # Draw vertical lines in blue
    for x1, y1, x2, y2 in v_lines:
        cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)

    return vis_image


def detect_table_lines(
    image: np.ndarray,
    min_line_length: int = 100,
    max_line_gap: int = 10,
    kernel_h_size: int = 40,
    kernel_v_size: int = 40,
    hough_threshold: int = 50,
    return_analysis: bool = False,
) -> tuple:
    """Detect horizontal and vertical lines in image.

    Args:
        image: Input image
        min_line_length: Minimum line length for detection
        max_line_gap: Maximum gap in line segments
        return_analysis: If True, returns additional statistics

    Returns:
        tuple: (h_lines, v_lines) or (h_lines, v_lines, analysis) if return_analysis=True
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Apply morphological operations to enhance lines
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_h_size, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_v_size))

    # Detect horizontal lines
    horizontal = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_h)
    h_lines = cv2.HoughLinesP(
        horizontal,
        1,
        np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    # Detect vertical lines
    vertical = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_v)
    v_lines = cv2.HoughLinesP(
        vertical,
        1,
        np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    # Convert to list of tuples
    h_lines = [tuple(line[0]) for line in h_lines] if h_lines is not None else []
    v_lines = [tuple(line[0]) for line in v_lines] if v_lines is not None else []

    if not return_analysis:
        return h_lines, v_lines

    # Calculate line statistics
    h_line_lengths = []
    v_line_lengths = []

    for x1, y1, x2, y2 in h_lines:
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        h_line_lengths.append(length)

    for x1, y1, x2, y2 in v_lines:
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        v_line_lengths.append(length)

    # Find potential table boundaries
    h_bounds = None
    v_bounds = None

    if h_lines:
        h_y_coords = []
        for x1, y1, x2, y2 in h_lines:
            h_y_coords.extend([y1, y2])
        h_bounds = (min(h_y_coords), max(h_y_coords))

    if v_lines:
        v_x_coords = []
        for x1, y1, x2, y2 in v_lines:
            v_x_coords.extend([x1, x2])
        v_bounds = (min(v_x_coords), max(v_x_coords))

    analysis = {
        "horizontal_morph": horizontal,
        "vertical_morph": vertical,
        "h_lines": h_lines,
        "v_lines": v_lines,
        "h_line_count": len(h_lines),
        "v_line_count": len(v_lines),
        "total_lines": len(h_lines) + len(v_lines),
        "h_line_lengths": h_line_lengths,
        "v_line_lengths": v_line_lengths,
        "h_avg_length": np.mean(h_line_lengths) if h_line_lengths else 0,
        "v_avg_length": np.mean(v_line_lengths) if v_line_lengths else 0,
        "h_max_length": max(h_line_lengths) if h_line_lengths else 0,
        "v_max_length": max(v_line_lengths) if v_line_lengths else 0,
        "h_bounds": h_bounds,
        "v_bounds": v_bounds,
        "has_table_structure": len(h_lines) > 0 and len(v_lines) > 0,
        "config_used": {
            "min_line_length": min_line_length,
            "max_line_gap": max_line_gap,
            "kernel_h_size": kernel_h_size,
            "kernel_v_size": kernel_v_size,
            "hough_threshold": hough_threshold,
        },
    }

    return h_lines, v_lines, analysis


def crop_table_region(
    image: np.ndarray,
    h_lines: List[Tuple],
    v_lines: List[Tuple],
    crop_padding: int = 10,
    return_analysis: bool = False,
) -> tuple:
    """Crop image to table region based on detected lines.

    Args:
        image: Input image
        h_lines: List of horizontal line tuples (x1, y1, x2, y2)
        v_lines: List of vertical line tuples (x1, y1, x2, y2)
        crop_padding: Padding around detected table boundaries
        return_analysis: If True, returns detailed analysis information

    Returns:
        np.ndarray or tuple: Cropped image, or (cropped_image, analysis) if return_analysis=True
    """
    if not h_lines or not v_lines:
        if return_analysis:
            return image, {"error": "No lines detected", "boundaries": None}
        return image

    height, width = image.shape[:2]

    # Find boundaries
    min_x = min([min(x1, x2) for x1, y1, x2, y2 in v_lines])
    max_x = max([max(x1, x2) for x1, y1, x2, y2 in v_lines])
    min_y = min([min(y1, y2) for x1, y1, x2, y2 in h_lines])
    max_y = max([max(y1, y2) for x1, y1, x2, y2 in h_lines])

    # Add padding
    padded_min_x = max(0, min_x - crop_padding)
    padded_max_x = min(width, max_x + crop_padding)
    padded_min_y = max(0, min_y - crop_padding)
    padded_max_y = min(height, max_y + crop_padding)

    # Crop the image
    cropped = image[padded_min_y:padded_max_y, padded_min_x:padded_max_x]

    if not return_analysis:
        return cropped

    # Calculate analysis information
    original_area = width * height
    table_area = (max_x - min_x) * (max_y - min_y)
    cropped_area = (padded_max_x - padded_min_x) * (padded_max_y - padded_min_y)

    analysis = {
        "original_dimensions": (width, height),
        "table_boundaries": {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
        },
        "padded_boundaries": {
            "min_x": padded_min_x,
            "max_x": padded_max_x,
            "min_y": padded_min_y,
            "max_y": padded_max_y,
        },
        "cropped_dimensions": (
            padded_max_x - padded_min_x,
            padded_max_y - padded_min_y,
        ),
        "table_area": table_area,
        "cropped_area": cropped_area,
        "original_area": original_area,
        "table_coverage": table_area / original_area if original_area > 0 else 0,
        "crop_efficiency": table_area / cropped_area if cropped_area > 0 else 0,
        "padding_used": crop_padding,
        "h_line_count": len(h_lines),
        "v_line_count": len(v_lines),
    }

    return cropped, analysis


def find_vertical_cuts(
    binary_mask: np.ndarray,
    mode: str = "single_best",
    window_size_divisor: int = 20,
    min_window_size: int = 50,
    min_cut_strength: float = 10.0,
    min_confidence_threshold: float = 5.0,
) -> Tuple[int, int, Dict]:
    """Find vertical cuts in the binary mask using optimized vectorized sliding window approach."""
    height, width = binary_mask.shape
    vertical_projection = np.sum(binary_mask // 255, axis=0)

    window_size = max(width // window_size_divisor, min_window_size)

    # Find left cut (search only in leftmost 30%) - OPTIMIZED VECTORIZED VERSION
    left_search_end = int(width * 0.3)
    max_drop_left = 0
    cut_x_left = 0
    
    if left_search_end > window_size:
        # Create search range exactly matching original: range(window_size, left_search_end)
        search_indices = np.arange(window_size, left_search_end)
        # Apply original bounds check: if i - window_size > 0 (always true since i >= window_size)
        left_values = vertical_projection[search_indices].astype(float)
        left_prev_values = vertical_projection[search_indices - window_size].astype(float)
        left_drops = np.abs(left_values - left_prev_values)
        
        # Find maximum drop position exactly as original
        if len(left_drops) > 0:
            max_drop_idx = np.argmax(left_drops)
            max_drop_left = float(left_drops[max_drop_idx])
            cut_x_left = int(search_indices[max_drop_idx])

    # Find right cut (search only in rightmost 30%) - OPTIMIZED VECTORIZED VERSION  
    right_search_end = int(width * 0.7)
    max_drop_right = 0
    cut_x_right = width
    
    # Create search range exactly matching original: range(width - window_size, right_search_end, -1)
    right_search_start = width - window_size
    if right_search_start >= right_search_end:
        search_indices = np.arange(right_search_start, right_search_end - 1, -1)  # -1 to match range behavior
        
        # Apply original bounds check: if i + window_size < width
        valid_mask = search_indices + window_size < width
        search_indices = search_indices[valid_mask]
        
        if len(search_indices) > 0:
            # Vectorized calculation matching original logic
            right_values = vertical_projection[search_indices].astype(float)
            right_next_values = vertical_projection[search_indices + window_size].astype(float)
            right_drops = np.abs(right_values - right_next_values)
            
            # Find maximum drop position exactly as original
            max_drop_idx = np.argmax(right_drops)
            max_drop_right = float(right_drops[max_drop_idx])
            cut_x_right = int(search_indices[max_drop_idx])

    # Evaluate whether cuts should be applied
    apply_left_cut = (
        max_drop_left >= min_cut_strength and max_drop_left >= min_confidence_threshold
    )
    apply_right_cut = (
        max_drop_right >= min_cut_strength
        and max_drop_right >= min_confidence_threshold
    )

    # Decide what to return based on mode
    if mode == "both_sides":
        final_left = cut_x_left if apply_left_cut else 0
        final_right = cut_x_right if apply_right_cut else width
        return (
            final_left,
            final_right,
            {
                "projection": vertical_projection,
                "left_cut_strength": max_drop_left,
                "right_cut_strength": max_drop_right,
                "left_cut_applied": apply_left_cut,
                "right_cut_applied": apply_right_cut,
            },
        )

    # For single_best mode, choose the stronger cut
    left_density = np.sum(vertical_projection[:left_search_end]) // (
        left_search_end - 0
    )
    right_density = np.sum(vertical_projection[right_search_end:]) // (
        width - right_search_end
    )

    if right_density < left_density:
        # Prefer right cut
        final_right = cut_x_right if apply_right_cut else width
        return (
            0,
            final_right,
            {
                "projection": vertical_projection,
                "right_cut_strength": max_drop_right,
                "right_cut_applied": apply_right_cut,
                "cut_side": "right",
            },
        )
    else:
        # Prefer left cut
        final_left = cut_x_left if apply_left_cut else 0
        return (
            final_left,
            width,
            {
                "projection": vertical_projection,
                "left_cut_strength": max_drop_left,
                "left_cut_applied": apply_left_cut,
                "cut_side": "left",
            },
        )


def find_horizontal_cuts(
    binary_mask: np.ndarray,
    mode: str = "single_best",
    window_size_divisor: int = 20,
    min_window_size: int = 50,
    min_cut_strength: float = 10.0,
    min_confidence_threshold: float = 5.0,
) -> Tuple[int, int, Dict]:
    """Find horizontal cuts in the binary mask using optimized vectorized sliding window approach."""
    height, width = binary_mask.shape
    horizontal_projection = np.sum(binary_mask // 255, axis=1)

    window_size = max(height // window_size_divisor, min_window_size)

    # Find top cut (search only in topmost 30%) - OPTIMIZED VECTORIZED VERSION
    top_search_end = int(height * 0.3)
    max_drop_top = 0
    cut_y_top = 0
    
    if top_search_end > window_size:
        # Create search range exactly matching original: range(window_size, top_search_end)
        search_indices = np.arange(window_size, top_search_end)
        # Apply original bounds check: if i - window_size > 0 (always true since i >= window_size)
        top_values = horizontal_projection[search_indices].astype(float)
        top_prev_values = horizontal_projection[search_indices - window_size].astype(float)
        top_drops = np.abs(top_values - top_prev_values)
        
        # Find maximum drop position exactly as original
        if len(top_drops) > 0:
            max_drop_idx = np.argmax(top_drops)
            max_drop_top = float(top_drops[max_drop_idx])
            cut_y_top = int(search_indices[max_drop_idx])

    # Find bottom cut (search only in bottommost 30%) - OPTIMIZED VECTORIZED VERSION
    bottom_search_end = int(height * 0.7)
    max_drop_bottom = 0
    cut_y_bottom = height
    
    # Create search range exactly matching original: range(height - window_size, bottom_search_end, -1)
    bottom_search_start = height - window_size
    if bottom_search_start >= bottom_search_end:
        search_indices = np.arange(bottom_search_start, bottom_search_end - 1, -1)  # -1 to match range behavior
        
        # Apply original bounds check: if i + window_size < height
        valid_mask = search_indices + window_size < height
        search_indices = search_indices[valid_mask]
        
        if len(search_indices) > 0:
            # Vectorized calculation matching original logic
            bottom_values = horizontal_projection[search_indices].astype(float)
            bottom_next_values = horizontal_projection[search_indices + window_size].astype(float)
            bottom_drops = np.abs(bottom_values - bottom_next_values)
            
            # Find maximum drop position exactly as original
            max_drop_idx = np.argmax(bottom_drops)
            max_drop_bottom = float(bottom_drops[max_drop_idx])
            cut_y_bottom = int(search_indices[max_drop_idx])

    # Evaluate whether cuts should be applied
    apply_top_cut = (
        max_drop_top >= min_cut_strength and max_drop_top >= min_confidence_threshold
    )
    apply_bottom_cut = (
        max_drop_bottom >= min_cut_strength
        and max_drop_bottom >= min_confidence_threshold
    )

    if mode == "both_sides":
        final_top = cut_y_top if apply_top_cut else 0
        final_bottom = cut_y_bottom if apply_bottom_cut else height
        return (
            final_top,
            final_bottom,
            {
                "projection": horizontal_projection,
                "top_cut_strength": max_drop_top,
                "bottom_cut_strength": max_drop_bottom,
                "top_cut_applied": apply_top_cut,
                "bottom_cut_applied": apply_bottom_cut,
            },
        )

    # For single_best mode, choose the stronger cut
    top_density = np.sum(horizontal_projection[:top_search_end]) // (top_search_end - 0)
    bottom_density = np.sum(horizontal_projection[bottom_search_end:]) // (
        height - bottom_search_end
    )

    if bottom_density < top_density:
        # Prefer bottom cut
        final_bottom = cut_y_bottom if apply_bottom_cut else height
        return (
            0,
            final_bottom,
            {
                "projection": horizontal_projection,
                "bottom_cut_strength": max_drop_bottom,
                "bottom_cut_applied": apply_bottom_cut,
                "cut_side": "bottom",
            },
        )
    else:
        # Prefer top cut
        final_top = cut_y_top if apply_top_cut else 0
        return (
            final_top,
            height,
            {
                "projection": horizontal_projection,
                "top_cut_strength": max_drop_top,
                "top_cut_applied": apply_top_cut,
                "cut_side": "top",
            },
        )


@lru_cache(maxsize=16)
def _get_cached_gabor_kernels(
    kernel_size: int, sigma: float, lambda_val: float, gamma: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Create and cache Gabor kernels for vertical and horizontal orientations.
    
    Caching prevents expensive kernel recreation for repeated operations.
    Cache size of 16 handles multiple parameter combinations efficiently.
    """
    # Create Gabor kernels for vertical and horizontal orientations only
    vertical_kernel = cv2.getGaborKernel(
        (kernel_size, kernel_size),
        sigma,
        0.0,  # 0° (vertical)
        lambda_val,
        gamma,
        0,
        ktype=cv2.CV_32F,
    )
    horizontal_kernel = cv2.getGaborKernel(
        (kernel_size, kernel_size),
        sigma,
        float(np.pi / 2),  # 90° (horizontal)
        lambda_val,
        gamma,
        0,
        ktype=cv2.CV_32F,
    )
    return vertical_kernel, horizontal_kernel


def detect_roi_gabor(
    image: np.ndarray,
    kernel_size: int = 31,
    sigma: float = 4.0,
    lambda_val: float = 10.0,
    gamma: float = 0.5,
    binary_threshold: int = 127,
) -> np.ndarray:
    """Apply Gabor filters to detect edge structures with optimized kernel caching."""
    gray_img = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    )

    # Get cached Gabor kernels (avoids expensive recreation)
    vertical_kernel, horizontal_kernel = _get_cached_gabor_kernels(
        kernel_size, sigma, lambda_val, gamma
    )

    # Apply kernels and combine responses (optimized memory allocation)
    vertical_response = cv2.filter2D(gray_img, cv2.CV_32F, vertical_kernel)
    horizontal_response = cv2.filter2D(gray_img, cv2.CV_32F, horizontal_kernel)
    
    # Combine responses efficiently
    combined_response = vertical_response + horizontal_response

    # Normalize and threshold
    gabor_response_map = cv2.normalize(
        combined_response, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    _, binary_mask = cv2.threshold(
        gabor_response_map, binary_threshold, 255, cv2.THRESH_BINARY
    )

    return binary_mask


def detect_roi_canny_sobel(
    image: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    sobel_kernel_size: int = 3,
    gaussian_blur_size: int = 5,
    binary_threshold: int = 127,
    morphology_kernel_size: int = 3,
) -> np.ndarray:
    """Apply Canny and Sobel edge detection to detect ROI structures.

    This alternative method uses:
    - Canny edge detection for precise edge localization
    - Sobel operators for gradient-based edge detection
    - Morphological operations for noise cleanup
    - Combined response for robust edge detection

    Args:
        image: Input image
        canny_low: Lower threshold for Canny edge detection
        canny_high: Upper threshold for Canny edge detection
        sobel_kernel_size: Kernel size for Sobel operators (3, 5, or 7)
        gaussian_blur_size: Gaussian blur kernel size for preprocessing
        binary_threshold: Final binary threshold for mask creation
        morphology_kernel_size: Kernel size for morphological operations

    Returns:
        Binary mask with detected edges
    """
    gray_img = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    )

    # Preprocessing: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_img, (gaussian_blur_size, gaussian_blur_size), 0)

    # Method 1: Canny edge detection
    canny_edges = cv2.Canny(blurred, canny_low, canny_high)

    # Method 2: Sobel edge detection (horizontal and vertical)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)

    # Combine Sobel responses using magnitude
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_magnitude = cv2.normalize(
        sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # Method 3: Laplacian edge detection for fine details
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = cv2.normalize(
        np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # Combine all edge detection methods
    # Weight Canny higher for precise edges, Sobel for gradients, Laplacian for details
    combined_edges = (
        canny_edges.astype(np.float32) * 0.5  # 50% weight for Canny
        + sobel_magnitude.astype(np.float32) * 0.35  # 35% weight for Sobel
        + laplacian.astype(np.float32) * 0.15  # 15% weight for Laplacian
    )

    # Normalize combined response
    combined_edges = cv2.normalize(
        combined_edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # Apply morphological operations to clean up noise and connect edges
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morphology_kernel_size, morphology_kernel_size)
    )
    combined_edges = cv2.morphologyEx(
        combined_edges, cv2.MORPH_CLOSE, kernel
    )  # Close gaps
    combined_edges = cv2.morphologyEx(
        combined_edges, cv2.MORPH_OPEN, kernel
    )  # Remove noise

    # Final binary threshold
    _, binary_mask = cv2.threshold(
        combined_edges, binary_threshold, 255, cv2.THRESH_BINARY
    )

    return binary_mask


def detect_roi_adaptive_threshold(
    image: np.ndarray,
    adaptive_method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    threshold_type: int = cv2.THRESH_BINARY,
    block_size: int = 11,
    C: float = 2.0,
    morphology_kernel_size: int = 3,
    edge_enhancement: bool = True,
) -> np.ndarray:
    """Apply adaptive thresholding with edge enhancement for ROI detection.

    This method uses:
    - Adaptive thresholding to handle varying lighting conditions
    - Optional edge enhancement using gradient information
    - Morphological operations for structure cleanup

    Args:
        image: Input image
        adaptive_method: Adaptive method (ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C)
        threshold_type: Threshold type (THRESH_BINARY or THRESH_BINARY_INV)
        block_size: Size of neighborhood area for threshold calculation (odd number)
        C: Constant subtracted from the mean
        morphology_kernel_size: Kernel size for morphological operations
        edge_enhancement: Whether to enhance edges using gradient information

    Returns:
        Binary mask with detected structures
    """
    gray_img = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    )

    # Apply adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        gray_img, 255, adaptive_method, threshold_type, block_size, C
    )

    if edge_enhancement:
        # Enhance edges using gradient information
        grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_magnitude = cv2.normalize(
            gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        # Threshold gradient magnitude
        _, gradient_mask = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)

        # Combine adaptive threshold with gradient edges
        combined_mask = cv2.bitwise_or(adaptive_thresh, gradient_mask)
    else:
        combined_mask = adaptive_thresh

    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morphology_kernel_size, morphology_kernel_size)
    )
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    return combined_mask


def detect_roi_for_image(
    image: np.ndarray, config, return_analysis: bool = False
) -> Union[Dict, Tuple[Dict, Dict]]:
    """Detect ROI for an image using configurable edge detection methods and sliding window analysis.

    Args:
        image: Input image
        config: Configuration object with ROI parameters
        return_analysis: If True, returns additional analysis information

    Returns:
        dict: ROI coordinates and optionally analysis information
    """
    # Apply edge detection based on configured method
    method = getattr(config, "roi_detection_method", "gabor")

    if method == "canny_sobel":
        binary_mask = detect_roi_canny_sobel(
            image,
            canny_low=config.canny_low_threshold,
            canny_high=config.canny_high_threshold,
            sobel_kernel_size=config.sobel_kernel_size,
            gaussian_blur_size=config.gaussian_blur_size,
            binary_threshold=config.edge_binary_threshold,
            morphology_kernel_size=config.morphology_kernel_size,
        )
    elif method == "adaptive_threshold":
        # Convert adaptive method string to OpenCV constant
        adaptive_method = (
            cv2.ADAPTIVE_THRESH_MEAN_C
            if config.adaptive_method == "mean"
            else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        )
        binary_mask = detect_roi_adaptive_threshold(
            image,
            adaptive_method=adaptive_method,
            threshold_type=cv2.THRESH_BINARY,
            block_size=config.adaptive_block_size,
            C=config.adaptive_C,
            morphology_kernel_size=config.morphology_kernel_size,
            edge_enhancement=config.edge_enhancement,
        )
    else:  # Default to 'gabor'
        binary_mask = detect_roi_gabor(
            image,
            config.gabor_kernel_size,
            config.gabor_sigma,
            config.gabor_lambda,
            config.gabor_gamma,
            config.gabor_binary_threshold,
        )

    # Find cuts
    cut_x_left, cut_x_right, vertical_info = find_vertical_cuts(
        binary_mask,
        mode=config.roi_vertical_mode,
        window_size_divisor=config.roi_window_size_divisor,
        min_window_size=config.roi_min_window_size,
        min_cut_strength=config.roi_min_cut_strength,
        min_confidence_threshold=config.roi_min_confidence_threshold,
    )

    cut_y_top, cut_y_bottom, horizontal_info = find_horizontal_cuts(
        binary_mask,
        mode=config.roi_horizontal_mode,
        window_size_divisor=config.roi_window_size_divisor,
        min_window_size=config.roi_min_window_size,
        min_cut_strength=config.roi_min_cut_strength,
        min_confidence_threshold=config.roi_min_confidence_threshold,
    )

    roi_coords = {
        "roi_left": int(cut_x_left),
        "roi_right": int(cut_x_right),
        "roi_top": int(cut_y_top),
        "roi_bottom": int(cut_y_bottom),
        "image_width": image.shape[1],
        "image_height": image.shape[0],
    }

    if not return_analysis:
        return roi_coords

    # Calculate additional analysis
    roi_width = cut_x_right - cut_x_left
    roi_height = cut_y_bottom - cut_y_top
    roi_area = roi_width * roi_height
    total_area = image.shape[1] * image.shape[0]

    analysis = {
        "roi_width": roi_width,
        "roi_height": roi_height,
        "roi_area": roi_area,
        "roi_area_percentage": (roi_area / total_area) * 100,
        "vertical_info": vertical_info,
        "horizontal_info": horizontal_info,
        "binary_mask": binary_mask,
    }

    return roi_coords, analysis


def crop_to_roi(image: np.ndarray, roi_coords: Dict) -> np.ndarray:
    """Crop image to ROI coordinates."""
    left = roi_coords["roi_left"]
    right = roi_coords["roi_right"]
    top = roi_coords["roi_top"]
    bottom = roi_coords["roi_bottom"]

    return image[top:bottom, left:right]
