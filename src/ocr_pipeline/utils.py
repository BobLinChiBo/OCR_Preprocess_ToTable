"""Simple utilities for OCR pipeline."""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Union

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
            all_angles = list(coarse_angles[: len(coarse_scores)]) + list(fine_angles)
            all_scores = coarse_scores + fine_scores
            analysis_data = {
                "has_lines": True,
                "rotation_angle": 0.0,
                "angles": all_angles,
                "scores": all_scores,
                "coarse_angles": list(coarse_angles[: len(coarse_scores)]),
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
        all_angles = list(coarse_angles[: len(coarse_scores)]) + list(fine_angles)
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
            "coarse_angles": list(coarse_angles[: len(coarse_scores)]),
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


def filter_long_lines(
    h_lines: List[Tuple],
    v_lines: List[Tuple],
    img_width: int,
    img_height: int,
    max_h_ratio: float,
    max_v_ratio: float,
) -> Tuple[List[Tuple], List[Tuple]]:
    """Filter out lines that are too long (likely image borders or noise).

    Args:
        h_lines: List of horizontal lines (x1, y1, x2, y2)
        v_lines: List of vertical lines (x1, y1, x2, y2)
        img_width: Image width
        img_height: Image height
        max_h_ratio: Maximum allowed horizontal line length ratio (1.0 = disable filtering)
        max_v_ratio: Maximum allowed vertical line length ratio (1.0 = disable filtering)

    Returns:
        Tuple of (filtered_h_lines, filtered_v_lines)
    """
    # Filter horizontal lines
    if max_h_ratio >= 1.0:
        filtered_h = h_lines
    else:
        max_h_length = max_h_ratio * img_width
        filtered_h = [
            (x1, y1, x2, y2) for x1, y1, x2, y2 in h_lines if (x2 - x1) <= max_h_length
        ]

    # Filter vertical lines
    if max_v_ratio >= 1.0:
        filtered_v = v_lines
    else:
        max_v_length = max_v_ratio * img_height
        filtered_v = [
            (x1, y1, x2, y2) for x1, y1, x2, y2 in v_lines if (y2 - y1) <= max_v_length
        ]

    return filtered_h, filtered_v


def merge_close_parallel_lines(
    h_lines: List[Tuple], v_lines: List[Tuple], distance_threshold: int
) -> Tuple[List[Tuple], List[Tuple]]:
    """Merge lines that are very close to each other and parallel.

    Args:
        h_lines: List of horizontal lines (x1, y1, x2, y2)
        v_lines: List of vertical lines (x1, y1, x2, y2)
        distance_threshold: Maximum distance in pixels for merging (0 = disable)

    Returns:
        Tuple of (merged_h_lines, merged_v_lines)
    """
    if distance_threshold <= 0:
        return h_lines, v_lines

    def merge_lines_by_direction(lines, is_horizontal=True):
        """Merge lines by grouping close parallel lines."""
        if not lines:
            return lines

        # Sort lines by position (y for horizontal, x for vertical)
        if is_horizontal:
            lines_sorted = sorted(lines, key=lambda line: line[1])  # Sort by y
        else:
            lines_sorted = sorted(lines, key=lambda line: line[0])  # Sort by x

        merged = []
        current_group = [lines_sorted[0]]

        for i in range(1, len(lines_sorted)):
            current_line = lines_sorted[i]
            prev_line = lines_sorted[i - 1]

            if is_horizontal:
                # For horizontal lines, check y distance
                distance = abs(current_line[1] - prev_line[1])
            else:
                # For vertical lines, check x distance
                distance = abs(current_line[0] - prev_line[0])

            if distance <= distance_threshold:
                # Close enough, add to current group
                current_group.append(current_line)
            else:
                # Too far, finalize current group and start new one
                merged.append(merge_group(current_group, is_horizontal))
                current_group = [current_line]

        # Handle last group
        if current_group:
            merged.append(merge_group(current_group, is_horizontal))

        return merged

    def merge_group(group, is_horizontal=True):
        """Merge a group of close lines into a single line."""
        if len(group) == 1:
            return group[0]

        if is_horizontal:
            # For horizontal lines: average y, extend x bounds
            y_avg = int(sum(line[1] for line in group) / len(group))
            x_min = min(line[0] for line in group)
            x_max = max(line[2] for line in group)
            return (x_min, y_avg, x_max, y_avg)
        else:
            # For vertical lines: average x, extend y bounds
            x_avg = int(sum(line[0] for line in group) / len(group))
            y_min = min(line[1] for line in group)
            y_max = max(line[3] for line in group)
            return (x_avg, y_min, x_avg, y_max)

    merged_h = merge_lines_by_direction(h_lines, is_horizontal=True)
    merged_v = merge_lines_by_direction(v_lines, is_horizontal=False)

    return merged_h, merged_v


def detect_table_lines(
    image: np.ndarray,
    threshold: int = 40,  # Binary threshold
    horizontal_kernel_size: int = 8,  # Morphological kernel width
    vertical_kernel_size: int = 8,  # Morphological kernel height
    alignment_threshold: int = 3,  # Clustering threshold for line alignment
    # New separated parameters for horizontal lines
    h_min_length_image_ratio: float = 0.3,  # Min horizontal length as ratio of image width
    h_min_length_relative_ratio: float = 0.5,  # Min horizontal length relative to longest h-line
    # New separated parameters for vertical lines
    v_min_length_image_ratio: float = 0.3,  # Min vertical length as ratio of image height
    v_min_length_relative_ratio: float = 0.5,  # Min vertical length relative to longest v-line
    min_aspect_ratio: int = 5,  # Min aspect ratio for line-like components
    # New post-processing parameters
    max_h_length_ratio: float = 1.0,  # Max horizontal line length ratio (1.0 = disable filtering)
    max_v_length_ratio: float = 1.0,  # Max vertical line length ratio (1.0 = disable filtering)
    close_line_distance: int = 45,  # Distance for merging close lines (0 = disable)
    return_analysis: bool = False,
    # Keep old parameters for backwards compatibility
    hough_threshold: int = None,
    min_line_length: int = None,
    max_line_gap: int = None,
    merge_distance_h: int = None,
    merge_distance_v: int = None,
    merge_iterations: int = None,
    length_filter_ratio_h: float = None,
    length_filter_ratio_v: float = None,
    **kwargs,
) -> tuple:
    """Detect table lines using binary threshold and morphological operations.

    New approach based on connected components analysis rather than edge detection.

    Args:
        image: Input image (grayscale or BGR)
        threshold: Binary threshold value
        horizontal_kernel_size: Width of morphological kernel for horizontal lines
        vertical_kernel_size: Height of morphological kernel for vertical lines
        alignment_threshold: Maximum distance for clustering aligned line segments
        h_min_length_image_ratio: Min horizontal length as ratio of image width
        h_min_length_relative_ratio: Min horizontal length relative to longest h-line
        v_min_length_image_ratio: Min vertical length as ratio of image height
        v_min_length_relative_ratio: Min vertical length relative to longest v-line
        min_aspect_ratio: Minimum aspect ratio to consider component as line-like
        max_h_length_ratio: Maximum horizontal line length ratio (1.0 = disable filtering)
        max_v_length_ratio: Maximum vertical line length ratio (1.0 = disable filtering)
        close_line_distance: Distance for merging close parallel lines (0 = disable)
        return_analysis: If True, returns additional statistics

    Returns:
        tuple: (h_lines, v_lines) or (h_lines, v_lines, analysis) if return_analysis=True
    """
    # Convert to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    img_h, img_w = gray.shape

    # Invert image (make dark lines white)
    gray_inv = cv2.bitwise_not(gray)

    # Binary thresholding
    _, binary = cv2.threshold(gray_inv, threshold, 255, cv2.THRESH_BINARY)

    # Morphological opening to extract lines
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_size))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_size, 1))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v, iterations=1)
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h, iterations=1)

    def extract_and_merge_lines(
        line_img,
        direction="vertical",
        alignment_thresh=3,
        min_length_ratio=0.3,
        min_aspect=5,
    ):
        """Extract line segments using connected components and merge aligned segments."""
        img_h, img_w = line_img.shape[:2]
        num, labels, stats, _ = cv2.connectedComponentsWithStats(line_img, 8)

        segments = []  # (x1, y1, x2, y2, true_center, label, score)

        for lab in range(1, num):
            x, y, w, h, _ = stats[lab]
            x1, y1, x2, y2 = x, y, x + w, y + h

            # Length and aspect ratio filtering
            if direction == "vertical":
                if h < min_length_ratio * img_h:
                    continue
                aspect = h / max(w, 1)
                if aspect < min_aspect:
                    continue
            else:
                if w < min_length_ratio * img_w:
                    continue
                aspect = w / max(h, 1)
                if aspect < min_aspect:
                    continue

            # Find true line center within component
            comp_mask = (labels == lab).astype(np.uint8)[y1:y2, x1:x2]

            if direction == "vertical" and w > 3:
                # For wide vertical components, find the densest column
                col_sum = comp_mask.sum(axis=0)
                best_cols = np.where(col_sum == col_sum.max())[0]
                center = x1 + int(best_cols.mean())
            elif direction == "horizontal" and h > 3:
                # For tall horizontal components, find the densest row
                row_sum = comp_mask.sum(axis=1)
                best_rows = np.where(row_sum == row_sum.max())[0]
                center = y1 + int(best_rows.mean())
            else:
                # Already thin, use bbox center
                if direction == "vertical":
                    center = (x1 + x2) // 2
                else:
                    center = (y1 + y2) // 2

            line_score = aspect  # Higher aspect ratio = better line
            segments.append((x1, y1, x2, y2, center, lab, line_score))

        # Cluster by proximity
        segments.sort(key=lambda s: s[4])  # Sort by center coordinate
        clusters = []

        for seg in segments:
            # Try to assign to existing cluster
            assigned = False
            for cluster in clusters:
                if abs(seg[4] - cluster["center"]) <= alignment_thresh:
                    cluster["members"].append(seg)
                    # Update cluster center to weighted average
                    total_score = sum(s[6] for s in cluster["members"])
                    cluster["center"] = (
                        sum(s[4] * s[6] for s in cluster["members"]) / total_score
                    )
                    assigned = True
                    break

            if not assigned:
                clusters.append({"center": seg[4], "members": [seg]})

        # Create one merged line per cluster
        merged = []
        for cluster in clusters:
            members = cluster["members"]
            if direction == "vertical":
                # Merge vertical segments
                y_min = min(s[1] for s in members)
                y_max = max(s[3] for s in members)
                x_center = int(cluster["center"])
                merged.append((x_center, y_min, x_center, y_max))
            else:
                # Merge horizontal segments
                x_min = min(s[0] for s in members)
                x_max = max(s[2] for s in members)
                y_center = int(cluster["center"])
                merged.append((x_min, y_center, x_max, y_center))

        return merged, segments

    # Extract and merge lines
    merged_vertical, vertical_segments = extract_and_merge_lines(
        vertical_lines,
        "vertical",
        alignment_threshold,
        v_min_length_image_ratio,
        min_aspect_ratio,
    )

    merged_horizontal, horizontal_segments = extract_and_merge_lines(
        horizontal_lines,
        "horizontal",
        alignment_threshold,
        h_min_length_image_ratio,
        min_aspect_ratio,
    )

    # Post-merge length filtering
    v_lengths = [y2 - y1 for x1, y1, x2, y2 in merged_vertical]
    h_lengths = [x2 - x1 for x1, y1, x2, y2 in merged_horizontal]

    v_thresh = v_min_length_relative_ratio * max(v_lengths) if v_lengths else 0
    h_thresh = h_min_length_relative_ratio * max(h_lengths) if h_lengths else 0

    filtered_vertical = [
        (x1, y1, x2, y2)
        for (x1, y1, x2, y2) in merged_vertical
        if (y2 - y1) >= v_thresh
    ]

    filtered_horizontal = [
        (x1, y1, x2, y2)
        for (x1, y1, x2, y2) in merged_horizontal
        if (x2 - x1) >= h_thresh
    ]

    # New post-processing steps
    # Step 1: Filter out overly long lines (if enabled)
    h_lines, v_lines = filter_long_lines(
        filtered_horizontal,
        filtered_vertical,
        img_w,
        img_h,
        max_h_length_ratio,
        max_v_length_ratio,
    )

    # Step 2: Merge close parallel lines (if enabled)
    h_lines, v_lines = merge_close_parallel_lines(h_lines, v_lines, close_line_distance)

    if not return_analysis:
        return h_lines, v_lines

    # Calculate analysis if requested
    h_line_lengths = [
        np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) for x1, y1, x2, y2 in h_lines
    ]
    v_line_lengths = [
        np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) for x1, y1, x2, y2 in v_lines
    ]

    analysis = {
        "h_lines_count": len(h_lines),
        "v_lines_count": len(v_lines),
        "h_line_lengths": h_line_lengths,
        "v_line_lengths": v_line_lengths,
        "avg_h_length": np.mean(h_line_lengths) if h_line_lengths else 0,
        "avg_v_length": np.mean(v_line_lengths) if v_line_lengths else 0,
        "threshold_used": threshold,
        "total_segments_before_merge": len(vertical_segments)
        + len(horizontal_segments),
        "vertical_clusters": (
            len(set(s[4] for s in vertical_segments)) if vertical_segments else 0
        ),
        "horizontal_clusters": (
            len(set(s[4] for s in horizontal_segments)) if horizontal_segments else 0
        ),
        # New post-processing analysis
        "lines_after_length_filter": len(filtered_horizontal) + len(filtered_vertical),
        "h_lines_after_long_filter": (
            len([line for line in h_lines]) if max_h_length_ratio < 1.0 else "disabled"
        ),
        "v_lines_after_long_filter": (
            len([line for line in v_lines]) if max_v_length_ratio < 1.0 else "disabled"
        ),
        "close_line_merging": "enabled" if close_line_distance > 0 else "disabled",
        "max_h_length_ratio_used": max_h_length_ratio,
        "max_v_length_ratio_used": max_v_length_ratio,
        "close_line_distance_used": close_line_distance,
    }

    return h_lines, v_lines, analysis


def find_largest_inscribed_rectangle(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Find the largest rectangle that fits entirely within the content mask.

    Args:
        mask: Binary mask where 255 indicates valid content area

    Returns:
        Tuple of (x, y, width, height) for the largest inscribed rectangle
    """
    height, width = mask.shape

    # Convert mask to binary (0 or 1)
    binary_mask = (mask > 0).astype(np.uint8)

    # Use dynamic programming approach for largest rectangle in histogram
    max_area = 0
    best_rect = (0, 0, 0, 0)

    # Create histogram for each row
    histogram = np.zeros(width, dtype=int)

    for row in range(height):
        # Update histogram
        for col in range(width):
            if binary_mask[row, col] == 1:
                histogram[col] += 1
            else:
                histogram[col] = 0

        # Find largest rectangle in current histogram
        rect = _largest_rectangle_in_histogram(histogram, row)
        if rect[2] * rect[3] > max_area:  # width * height
            max_area = rect[2] * rect[3]
            best_rect = rect

    return best_rect


def _largest_rectangle_in_histogram(
    heights: np.ndarray, current_row: int
) -> Tuple[int, int, int, int]:
    """Find the largest rectangle in a histogram using stack-based approach.

    Args:
        heights: Array of histogram heights
        current_row: Current row index (for calculating y coordinate)

    Returns:
        Tuple of (x, y, width, height) for the largest rectangle
    """
    stack = []
    max_area = 0
    best_rect = (0, 0, 0, 0)

    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            area = height * width

            if area > max_area:
                max_area = area
                left = 0 if not stack else stack[-1] + 1
                y = current_row - height + 1
                best_rect = (left, y, width, height)

        stack.append(i)

    # Process remaining bars in stack
    while stack:
        height = heights[stack.pop()]
        width = len(heights) if not stack else len(heights) - stack[-1] - 1
        area = height * width

        if area > max_area:
            max_area = area
            left = 0 if not stack else stack[-1] + 1
            y = current_row - height + 1
            best_rect = (left, y, width, height)

    return best_rect


def remove_margin_aggressive(
    image: np.ndarray,
    blur_kernel_size: int = 7,
    black_threshold: int = 50,
    content_threshold: int = 200,
    morph_kernel_size: int = 25,
    min_content_area_ratio: float = 0.01,
    padding: int = 5,
    return_analysis: bool = False,
) -> tuple:
    """Remove margins aggressively using largest inscribed rectangle approach.

    This method finds the largest box that fits inside the image content,
    effectively cutting all margin parts even if it cuts some actual images.

    Args:
        image: Input image (BGR or grayscale)
        blur_kernel_size: Gaussian blur kernel size
        black_threshold: Threshold for detecting very dark regions (margins)
        content_threshold: Threshold for detecting content regions
        morph_kernel_size: Morphological operation kernel size
        min_content_area_ratio: Minimum content area ratio to consider valid
        padding: Padding to subtract from the detected boundary (negative padding)
        return_analysis: If True, returns additional analysis information

    Returns:
        np.ndarray or tuple: Cropped image, or (cropped_image, analysis) if return_analysis=True
    """
    # Detect content mask
    content_mask = detect_curved_margins(
        image,
        blur_kernel_size,
        black_threshold,
        content_threshold,
        morph_kernel_size,
        min_content_area_ratio,
    )

    # Find the largest inscribed rectangle
    x, y, w, h = find_largest_inscribed_rectangle(content_mask)

    # Apply negative padding (shrink the rectangle to be more conservative)
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
        "method": "largest_inscribed_rectangle",
    }

    return cropped, analysis


def remove_margin_bounding_box(
    image: np.ndarray,
    blur_kernel_size: int = 7,
    black_threshold: int = 50,
    content_threshold: int = 200,
    morph_kernel_size: int = 25,
    min_content_area_ratio: float = 0.01,
    padding: int = 5,
    expansion_factor: float = 0.0,
    use_min_area_rect: bool = False,
    return_analysis: bool = False,
) -> tuple:
    """Remove margins using simple bounding box approach.

    This method finds the bounding box around all content regions,
    which is much faster than finding the largest inscribed rectangle.

    Args:
        image: Input image (BGR or grayscale)
        blur_kernel_size: Gaussian blur kernel size
        black_threshold: Threshold for detecting very dark regions (margins)
        content_threshold: Threshold for detecting content regions
        morph_kernel_size: Morphological operation kernel size
        min_content_area_ratio: Minimum content area ratio to consider valid
        padding: Padding to add/subtract from the detected boundary
        expansion_factor: Factor to expand the bounding box (0.1 = 10% expansion)
        use_min_area_rect: If True, use minimum area rectangle (can be rotated)
        return_analysis: If True, returns additional analysis information

    Returns:
        np.ndarray or tuple: Cropped image, or (cropped_image, analysis) if return_analysis=True
    """
    # Detect content mask
    content_mask = detect_curved_margins(
        image,
        blur_kernel_size,
        black_threshold,
        content_threshold,
        morph_kernel_size,
        min_content_area_ratio,
    )

    # Find contours in the content mask
    contours, _ = cv2.findContours(
        content_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        # No content found, return original image
        if return_analysis:
            return image, {
                "original_shape": image.shape,
                "cropped_shape": image.shape,
                "crop_bounds": (0, 0, image.shape[1], image.shape[0]),
                "area_retention": 1.0,
                "content_mask": content_mask,
                "method": "bounding_box_no_content",
            }
        return image

    # Combine all contours to find overall bounding box
    all_points = np.vstack(contours)

    if use_min_area_rect:
        # Find minimum area rectangle (can be rotated)
        rect = cv2.minAreaRect(all_points)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Get axis-aligned bounding box of the rotated rectangle
        x = np.min(box[:, 0])
        y = np.min(box[:, 1])
        w = np.max(box[:, 0]) - x
        h = np.max(box[:, 1]) - y
    else:
        # Find axis-aligned bounding rectangle
        x, y, w, h = cv2.boundingRect(all_points)

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
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)

    # Ensure we don't go out of bounds
    x = max(0, min(x, image.shape[1] - 1))
    y = max(0, min(y, image.shape[0] - 1))
    w = max(1, min(w, image.shape[1] - x))
    h = max(1, min(h, image.shape[0] - y))

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
        "method": "min_area_rect_bbox" if use_min_area_rect else "axis_aligned_bbox",
        "expansion_factor": expansion_factor,
        "num_contours": len(contours),
    }

    return cropped, analysis


def remove_margin_smart(
    image: np.ndarray,
    blur_kernel_size: int = 7,
    black_threshold: int = 50,
    content_threshold: int = 200,
    morph_kernel_size: int = 25,
    min_content_area_ratio: float = 0.01,
    padding_top: int = 10,
    padding_bottom: int = 10,
    padding_left: int = 10,
    padding_right: int = 10,
    histogram_threshold: float = 0.05,
    projection_smoothing: int = 3,
    return_analysis: bool = False,
) -> tuple:
    """Remove margins using smart asymmetric detection with projection histograms.

    This method analyzes each margin side independently using projection histograms
    to find true content boundaries, preventing over-cropping while effectively
    removing margins.

    Args:
        image: Input image (BGR or grayscale)
        blur_kernel_size: Gaussian blur kernel size
        black_threshold: Threshold for detecting very dark regions (margins)
        content_threshold: Threshold for detecting content regions
        morph_kernel_size: Morphological operation kernel size
        min_content_area_ratio: Minimum content area ratio to consider valid
        padding_top: Additional padding to subtract from top boundary
        padding_bottom: Additional padding to subtract from bottom boundary
        padding_left: Additional padding to subtract from left boundary
        padding_right: Additional padding to subtract from right boundary
        histogram_threshold: Threshold for projection histogram content detection (0-1)
        projection_smoothing: Smoothing window size for projection histograms
        return_analysis: If True, returns additional analysis information

    Returns:
        np.ndarray or tuple: Cropped image, or (cropped_image, analysis) if return_analysis=True
    """
    # Convert to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    height, width = gray.shape

    # Step 1: Adaptive thresholding based on image characteristics
    # Analyze image histogram to adjust thresholds
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    total_pixels = height * width

    # Find predominant background/margin intensity
    dark_ratio = np.sum(hist[:black_threshold]) / total_pixels
    if dark_ratio > 0.3:  # Image has significant dark margins
        # Increase content threshold for better separation
        adaptive_content_threshold = min(content_threshold + 30, 240)
        adaptive_black_threshold = min(black_threshold + 10, 80)
    else:
        # Use original thresholds
        adaptive_content_threshold = content_threshold
        adaptive_black_threshold = black_threshold

    # Step 2: Create content mask using adaptive thresholds
    content_mask = detect_curved_margins(
        image,
        blur_kernel_size,
        adaptive_black_threshold,
        adaptive_content_threshold,
        morph_kernel_size,
        min_content_area_ratio,
    )

    # Step 3: Analyze projection histograms for smart boundary detection

    # Horizontal projection (for top/bottom margins)
    horizontal_projection = np.sum(content_mask, axis=1)
    max_horizontal = (
        np.max(horizontal_projection) if len(horizontal_projection) > 0 else 1
    )
    normalized_horizontal = horizontal_projection / max_horizontal

    # Smooth the projection to reduce noise
    if projection_smoothing > 1:
        kernel = np.ones(projection_smoothing) / projection_smoothing
        normalized_horizontal = np.convolve(normalized_horizontal, kernel, mode="same")

    # Vertical projection (for left/right margins)
    vertical_projection = np.sum(content_mask, axis=0)
    max_vertical = np.max(vertical_projection) if len(vertical_projection) > 0 else 1
    normalized_vertical = vertical_projection / max_vertical

    # Smooth the projection
    if projection_smoothing > 1:
        normalized_vertical = np.convolve(normalized_vertical, kernel, mode="same")

    # Adaptive histogram threshold based on content distribution
    # Use a higher threshold if content is very sparse or very dense
    content_ratio = np.sum(content_mask > 0) / total_pixels
    if content_ratio < 0.1:  # Very sparse content
        adaptive_histogram_threshold = histogram_threshold * 2
    elif content_ratio > 0.8:  # Very dense content
        adaptive_histogram_threshold = histogram_threshold * 3
    else:
        adaptive_histogram_threshold = histogram_threshold

    # Step 4: Find content boundaries using projection analysis

    # Find top boundary (first significant content row)
    top_boundary = 0
    for i in range(height):
        if normalized_horizontal[i] > adaptive_histogram_threshold:
            top_boundary = max(0, i - padding_top)
            break

    # Find bottom boundary (last significant content row)
    bottom_boundary = height - 1
    for i in range(height - 1, -1, -1):
        if normalized_horizontal[i] > adaptive_histogram_threshold:
            bottom_boundary = min(height - 1, i + padding_bottom)
            break

    # Find left boundary (first significant content column)
    left_boundary = 0
    for i in range(width):
        if normalized_vertical[i] > adaptive_histogram_threshold:
            left_boundary = max(0, i - padding_left)
            break

    # Find right boundary (last significant content column)
    right_boundary = width - 1
    for i in range(width - 1, -1, -1):
        if normalized_vertical[i] > adaptive_histogram_threshold:
            right_boundary = min(width - 1, i + padding_right)
            break

    # Step 5: Validate boundaries and apply fallback
    # Ensure we have a valid rectangle
    if (
        bottom_boundary <= top_boundary
        or right_boundary <= left_boundary
        or (bottom_boundary - top_boundary) * (right_boundary - left_boundary)
        < height * width * min_content_area_ratio
    ):

        # Fallback to bounding box method if smart detection fails
        fallback_result = remove_margin_bounding_box(
            image,
            blur_kernel_size,
            adaptive_black_threshold,
            adaptive_content_threshold,
            morph_kernel_size,
            min_content_area_ratio,
            max(padding_top, padding_bottom, padding_left, padding_right),
            return_analysis=return_analysis,
        )

        if return_analysis:
            fallback_cropped, fallback_analysis = fallback_result
            fallback_analysis["method"] = "smart_fallback_to_bounding_box"
            fallback_analysis["fallback_reason"] = "invalid_smart_boundaries"
            fallback_analysis["adaptive_thresholds"] = {
                "black_threshold": adaptive_black_threshold,
                "content_threshold": adaptive_content_threshold,
                "histogram_threshold": adaptive_histogram_threshold,
            }
            return fallback_cropped, fallback_analysis
        else:
            return fallback_result

    # Step 6: Crop the image
    x, y = left_boundary, top_boundary
    w = right_boundary - left_boundary + 1
    h = bottom_boundary - top_boundary + 1

    cropped = image[y : y + h, x : x + w]

    if not return_analysis:
        return cropped

    # Step 7: Create analysis information
    original_area = image.shape[0] * image.shape[1]
    cropped_area = cropped.shape[0] * cropped.shape[1]

    analysis = {
        "original_shape": image.shape,
        "cropped_shape": cropped.shape,
        "crop_bounds": (x, y, w, h),
        "area_retention": cropped_area / original_area if original_area > 0 else 0,
        "content_mask": content_mask,
        "method": "smart_asymmetric",
        "boundaries": {
            "top": top_boundary,
            "bottom": bottom_boundary,
            "left": left_boundary,
            "right": right_boundary,
        },
        "margins_removed": {
            "top": top_boundary,
            "bottom": height - 1 - bottom_boundary,
            "left": left_boundary,
            "right": width - 1 - right_boundary,
        },
        "projections": {
            "horizontal": horizontal_projection,
            "vertical": vertical_projection,
            "normalized_horizontal": normalized_horizontal,
            "normalized_vertical": normalized_vertical,
        },
        "parameters": {
            "histogram_threshold": histogram_threshold,
            "projection_smoothing": projection_smoothing,
            "padding": {
                "top": padding_top,
                "bottom": padding_bottom,
                "left": padding_left,
                "right": padding_right,
            },
        },
        "adaptive_thresholds": {
            "black_threshold": adaptive_black_threshold,
            "content_threshold": adaptive_content_threshold,
            "histogram_threshold": adaptive_histogram_threshold,
            "content_ratio": content_ratio,
            "dark_ratio": dark_ratio,
        },
    }

    return cropped, analysis


def _extend_margins_by_texture(
    gray, initial_margin_mask, detected_margins, texture_threshold
):
    """Extend detected margins inward based on texture consistency."""
    height, width = gray.shape
    extended_mask = initial_margin_mask.copy()

    # Define extension directions for each edge
    extensions = {
        "left": (0, 1, width // 4),  # Extend right
        "right": (0, -1, width // 4),  # Extend left
        "top": (1, 0, height // 4),  # Extend down
        "bottom": (-1, 0, height // 4),  # Extend up
    }

    for margin_edge in detected_margins:
        if margin_edge not in extensions:
            continue

        dy, dx, max_extend = extensions[margin_edge]

        # Find current margin boundary
        if margin_edge == "left":
            boundary_x = (
                np.max(np.where(initial_margin_mask)[1])
                if np.any(initial_margin_mask)
                else 0
            )
            for x in range(boundary_x + 1, min(boundary_x + max_extend, width)):
                column = gray[:, x]
                if np.var(column) > texture_threshold:
                    break
                extended_mask[:, x] = True

        elif margin_edge == "right":
            boundary_x = (
                np.min(np.where(initial_margin_mask)[1])
                if np.any(initial_margin_mask)
                else width - 1
            )
            for x in range(boundary_x - 1, max(boundary_x - max_extend, 0), -1):
                column = gray[:, x]
                if np.var(column) > texture_threshold:
                    break
                extended_mask[:, x] = True

        elif margin_edge == "top":
            boundary_y = (
                np.max(np.where(initial_margin_mask)[0])
                if np.any(initial_margin_mask)
                else 0
            )
            for y in range(boundary_y + 1, min(boundary_y + max_extend, height)):
                row = gray[y, :]
                if np.var(row) > texture_threshold:
                    break
                extended_mask[y, :] = True

        elif margin_edge == "bottom":
            boundary_y = (
                np.min(np.where(initial_margin_mask)[0])
                if np.any(initial_margin_mask)
                else height - 1
            )
            for y in range(boundary_y - 1, max(boundary_y - max_extend, 0), -1):
                row = gray[y, :]
                if np.var(row) > texture_threshold:
                    break
                extended_mask[y, :] = True

    return extended_mask

    """Find the first significant intensity jump in a 1D profile.
    
    Args:
        profile: 1D array of intensity values
        threshold: Minimum intensity increase to consider a jump
        scan_direction: 'forward' (start to end) or 'backward' (end to start)
        
    Returns:
        Index of the jump, or None if no significant jump found
    """
    if len(profile) < 2:
        return None

    # Calculate intensity differences between adjacent points
    if scan_direction == "forward":
        diffs = np.diff(profile)  # profile[i+1] - profile[i]
        indices = range(len(diffs))
    else:  # backward
        diffs = -np.diff(profile[::-1])  # profile[i] - profile[i+1] (reversed)
        indices = range(len(diffs) - 1, -1, -1)

    # Find first difference that exceeds threshold
    for i, diff in zip(indices, diffs):
        if diff >= threshold:
            return i + 1 if scan_direction == "forward" else len(profile) - i - 1

    return None


def _find_gradient_boundary(
    profile: np.ndarray,
    window_size: int,
    shift_threshold: float,
    confidence_threshold: float,
    scan_direction: str = "forward",
) -> dict:
    """Find boundary using gradient analysis and statistical methods.

    Args:
        profile: 1D intensity profile
        window_size: Size of smoothing window
        shift_threshold: Minimum sustained intensity shift
        confidence_threshold: Required confidence for boundary detection
        scan_direction: 'forward' or 'backward'

    Returns:
        Dictionary with boundary analysis data or None if not found
    """
    if len(profile) < window_size * 2:
        return None

    # Step 1: Smooth the profile to reduce text noise using numpy-only implementation
    def uniform_filter_numpy(data, size):
        """Simple uniform filter using numpy convolution."""
        kernel = np.ones(size) / size
        # Use 'same' mode to maintain array length
        return np.convolve(data, kernel, mode="same")

    smoothed = uniform_filter_numpy(profile.astype(float), window_size)

    # Step 2: Calculate gradient (rate of change)
    gradient = np.gradient(smoothed)

    # Step 3: Find sustained positive gradients (dark to bright transitions)
    if scan_direction == "backward":
        gradient = -gradient[::-1]  # Reverse and flip sign

    # Step 4: Look for regions with sustained positive gradient
    # Use a sliding window to find regions where gradient is consistently positive
    sustained_regions = []

    for i in range(len(gradient) - window_size):
        window_gradient = gradient[i : i + window_size]

        # Check if this window shows sustained increase
        positive_ratio = np.sum(window_gradient > 0) / len(window_gradient)
        avg_gradient = (
            np.mean(window_gradient[window_gradient > 0])
            if np.any(window_gradient > 0)
            else 0
        )

        if (
            positive_ratio >= confidence_threshold
            and avg_gradient * window_size >= shift_threshold
        ):
            sustained_regions.append((i, positive_ratio, avg_gradient))

    if not sustained_regions:
        return None

    # Step 5: Choose the best sustained region (earliest with highest confidence)
    best_region = min(
        sustained_regions, key=lambda x: (x[0], -x[1])
    )  # Earliest, then highest confidence
    boundary_pos, positive_ratio, avg_gradient = best_region

    # Convert back for backward scan
    if scan_direction == "backward":
        boundary_pos = len(profile) - boundary_pos - 1

    # Step 6: Calculate additional gradient characteristics for adaptive cutting
    # Gradient steepness: how quickly intensity changes in the transition region
    transition_window = gradient[
        max(0, boundary_pos - window_size // 2) : boundary_pos + window_size // 2
    ]
    gradient_steepness = np.std(transition_window) if len(transition_window) > 0 else 0

    # Transition sharpness: ratio of max gradient to average gradient in transition
    if len(transition_window) > 0 and avg_gradient > 0:
        max_gradient = np.max(transition_window)
        transition_sharpness = min(max_gradient / avg_gradient, 3.0)  # Cap at 3x
    else:
        transition_sharpness = 1.0

    return {
        "boundary_pos": boundary_pos,
        "avg_gradient": avg_gradient,
        "positive_ratio": positive_ratio,
        "gradient_steepness": gradient_steepness,
        "transition_sharpness": transition_sharpness,
    }


def _has_black_margin_strict(
    gradient_analysis: dict,
    profile: np.ndarray,
    min_margin_intensity: int = 80,
    contrast_threshold: float = 40.0,
    is_binary: bool = False,
    side_position: str = "unknown",
    image_shape: tuple = (0, 0),
) -> bool:
    """Detect if a side actually has a black margin using strict criteria.

    Uses much stricter thresholds to avoid false positives on clean sides
    after page-split operations, while being more sensitive to actual margins.

    Args:
        gradient_analysis: Gradient analysis data from _find_gradient_boundary
        profile: 1D intensity profile of the edge region
        is_binary: True if working with binary images (simplified logic)

    Returns:
        True if side has black margin requiring removal, False otherwise
    """
    if gradient_analysis is None or len(profile) == 0:
        return False

    # Detect edge type first
    boundary_pos = gradient_analysis.get("boundary_pos", 0)
    edge_type = detect_edge_type(profile, boundary_pos, side_position, image_shape)

    # For clean split edges, be very conservative
    if edge_type == "clean":
        # Clean edges should have minimal margin removal
        # Only remove margins if there's compelling evidence of actual margin
        has_very_strong_confidence = gradient_analysis["positive_ratio"] > 0.9
        has_strong_gradient = (
            gradient_analysis["avg_gradient"] > 3.0
        )  # Reduced from 5.0
        has_sufficient_evidence = has_very_strong_confidence and has_strong_gradient

        # Additional check: boundary must be reasonably far from edge for real margin
        boundary_pos = gradient_analysis.get("boundary_pos", 0)
        has_substantial_margin = boundary_pos > 20  # At least 20px of margin space

        return has_sufficient_evidence and has_substantial_margin

    # Simplified logic for binary images (original edges only)
    if is_binary:
        # For binary images, just check if we have strong gradient characteristics
        has_strong_confidence = gradient_analysis["positive_ratio"] > 0.7
        has_strong_gradient = (
            gradient_analysis["avg_gradient"] > 2.0
        )  # Binary gradients are similar to grayscale
        has_sufficient_width = len(profile) > 10  # Basic width check

        return has_strong_confidence and has_strong_gradient and has_sufficient_width

    # Use larger edge sampling for more accurate analysis (15% instead of 10%)
    edge_sample_size = max(10, int(len(profile) * 0.15))
    edge_intensity = np.mean(profile[:edge_sample_size])

    # Also check intensity variation to distinguish margins from clean content edges
    edge_std = np.std(profile[:edge_sample_size])

    # Check content intensity further in for comparison
    content_sample_start = min(len(profile) // 2, 50)  # Start sampling content area
    content_sample_size = max(10, int(len(profile) * 0.2))
    content_end = min(len(profile), content_sample_start + content_sample_size)
    content_intensity = np.mean(profile[content_sample_start:content_end])

    # Calculate intensity contrast between edge and content
    intensity_contrast = content_intensity - edge_intensity

    # Additional strictness criteria for clean side detection
    # 1. Minimum margin width - check if dark region is sustained enough to be a real margin
    dark_region_width = 0
    for i in range(min(edge_sample_size * 2, len(profile))):
        if profile[i] < min_margin_intensity:
            dark_region_width += 1
        else:
            break
    min_margin_width = max(
        5, int(len(profile) * 0.05)
    )  # At least 5 pixels or 5% of edge region
    has_sufficient_margin_width = dark_region_width >= min_margin_width

    # 2. Sustained darkness - check if the dark region is consistently dark (not just noise)
    dark_region_consistency = 0
    if dark_region_width > 0:
        dark_region_std = np.std(profile[:dark_region_width])
        dark_region_consistency = dark_region_std < 20  # Low variation in dark region

    # Core gradient criteria (proven reliable)
    has_very_dark_edge = edge_intensity < min_margin_intensity  # Dark margin threshold
    has_very_bright_edge = edge_intensity > (
        255 - min_margin_intensity
    )  # Bright margin threshold
    has_significant_contrast = (
        abs(intensity_contrast) > contrast_threshold
    )  # Edge much different from content
    has_consistent_edge = edge_std < 25  # Low variation in edge (not text)
    has_strong_confidence = (
        gradient_analysis["positive_ratio"] > 0.7
    )  # High confidence threshold
    has_strong_gradient = (
        gradient_analysis["avg_gradient"] > 2.3
    )  # Adjusted to match real data (2.388 observed)
    has_steep_gradient = (
        gradient_analysis["gradient_steepness"] > 0.1
    )  # Adjusted to match real data (0.114 observed)

    # Enhanced validation criteria (additional confidence, but not required)
    # Make these more lenient to avoid false negatives
    min_margin_width_relaxed = max(3, int(len(profile) * 0.03))  # Reduced from 5% to 3%
    has_sufficient_margin_width_relaxed = dark_region_width >= min_margin_width_relaxed
    dark_region_consistency_relaxed = 0
    if dark_region_width > 0:
        dark_region_std = np.std(profile[:dark_region_width])
        dark_region_consistency_relaxed = dark_region_std < 30  # Relaxed from 20 to 30

    # GRADUATED DETECTION APPROACH:

    # Level 1: Original proven criteria (primary detection) - for both dark and bright margins
    original_primary = (
        (has_very_dark_edge or has_very_bright_edge)
        and has_consistent_edge
        and has_strong_confidence
        and has_strong_gradient
        and has_steep_gradient
    )

    # Level 2: Contrast-based detection with relaxed enhanced validation
    original_secondary = (
        has_significant_contrast
        and has_consistent_edge
        and has_strong_confidence
        and has_strong_gradient
    )

    # Level 3: Enhanced validation (optional boost for extra confidence)
    enhanced_validation = (
        has_sufficient_margin_width_relaxed and dark_region_consistency_relaxed
    )

    # Level 4: Relaxed detection for potential missed margins (especially bottom/right edges)
    # Use more permissive criteria to catch bottom margins that might be missed
    relaxed_detection = (
        (
            has_very_dark_edge or has_very_bright_edge or has_significant_contrast
        )  # Dark, bright, or good contrast
        and (
            has_strong_confidence or has_strong_gradient
        )  # Either high confidence OR strong gradient
        and has_sufficient_margin_width_relaxed  # Must have some sustained margin width
    )

    # Accept margin if:
    # 1. Original criteria pass (proven reliable), OR
    # 2. Contrast-based detection, OR
    # 3. Enhanced validation for edge cases, OR
    # 4. Relaxed detection for potentially missed margins
    margin_detected = (
        original_primary  # Proven reliable detection
        or original_secondary  # Contrast-based detection
        or (has_significant_contrast and enhanced_validation)  # Enhanced edge cases
        or relaxed_detection  # Catch potentially missed margins
    )

    return margin_detected


def _calculate_simple_cutting_point(
    gradient_analysis: dict,
    profile: np.ndarray,
    original_boundary: int,
    image_dimension: int,
    side_position: str = "top",  # "top", "bottom", "left", "right"
    binary_intensity_threshold: int = 200,  # Tunable threshold for binary cutting decision
    binary_profile: np.ndarray = None,  # Binary version of profile for analysis
) -> int:
    """Calculate cutting point using simplified boundary position logic with binary intensity only.

    Simple rules (binary only):
    1. Binarize profile first for cleaner edge detection
    2. If boundary_pos < 5, reset to 0 (no margin)
    3. Cut when binary edge intensity > threshold value
    4. Use boundary position directly for cutting

    Args:
        gradient_analysis: Gradient analysis data from _find_gradient_boundary
        profile: 1D intensity profile of the edge region (grayscale - not used for decision)
        original_boundary: Original detected boundary position
        image_dimension: Width or height of image (for calculating max cut)
        side_position: "top", "bottom", "left", "right"
        binary_intensity_threshold: Threshold for binary cutting decision (tunable)
        binary_profile: Binary version of profile for analysis

    Returns:
        New cutting point position
    """
    # Step 1: Handle case where no boundary was detected
    if gradient_analysis is None:
        # No boundary detected - return no cutting
        if side_position in ["top", "left"]:
            return 0  # No cutting from top/left edges
        else:
            return image_dimension  # No cutting from bottom/right edges

    boundary_pos = gradient_analysis.get("boundary_pos", 0)

    # Step 2: Clean edge detection based on boundary position
    # If boundary is very close to edge (< 5 pixels), it's a clean split edge - don't cut
    if boundary_pos < 5:
        # Clean edge detected - no cutting
        if side_position in ["top", "left"]:
            return 0
        else:
            return image_dimension

    # Step 5: Use boundary position directly for cutting
    if side_position == "top":
        return max(0, boundary_pos)
    elif side_position == "bottom":
        # For bottom, we need to convert from edge region coordinates to full image coordinates
        # The bottom profile scans the last edge_percentage of the image
        # So boundary_pos is relative to the start of the bottom region
        edge_region_size = int(image_dimension * 0.20)  # 20% edge region
        bottom_region_start = image_dimension - edge_region_size
        full_image_boundary = bottom_region_start + boundary_pos
        return min(image_dimension, full_image_boundary)
    elif side_position == "left":
        return max(0, boundary_pos)
    elif side_position == "right":
        # For right, we need to convert from edge region coordinates to full image coordinates
        # The right profile scans the last edge_percentage of the image
        # So boundary_pos is relative to the start of the right region
        edge_region_size = int(image_dimension * 0.20)  # 20% edge region
        right_region_start = image_dimension - edge_region_size
        full_image_boundary = right_region_start + boundary_pos
        return min(image_dimension, full_image_boundary)

    return original_boundary  # Fallback


def remove_margins_gradient(
    image: np.ndarray,
    edge_percentage: float = 0.20,
    gradient_window_size: int = 21,
    intensity_shift_threshold: float = 50.0,
    margin_confidence_threshold: float = 0.7,
    fill_method: str = "crop",
    direct_gradient_cutting: bool = True,
    max_content_cut_ratio: float = 0.25,
    strict_margin_detection: bool = True,
    gradient_aggressive_cutting_threshold: float = 80.0,
    gradient_min_margin_intensity: int = 80,
    gradient_contrast_threshold: float = 40.0,
    use_binarization: bool = True,
    return_analysis: bool = False,
) -> tuple:
    """Remove margins using gradient-based boundary detection.

    This method handles curved margins and black content by analyzing sustained
    intensity transitions rather than single intensity jumps.

    Args:
        image: Input image (BGR or grayscale)
        edge_percentage: Percentage of image edge to scan
        gradient_window_size: Size of smoothing window for gradient analysis
        intensity_shift_threshold: Minimum sustained intensity change for boundary
        margin_confidence_threshold: Statistical confidence required for detection
        fill_method: Method for removing margins ("color_fill" or "crop")
        direct_gradient_cutting: If True, use direct gradient-based cutting point calculation
        max_content_cut_ratio: Maximum ratio of image dimension to cut into content (0.25 = 25%)
        strict_margin_detection: If True, use strict criteria to avoid false positives
        gradient_aggressive_cutting_threshold: Edge intensity threshold for aggressive cutting
        gradient_min_margin_intensity: Minimum intensity to consider a margin (lower = darker required)
        gradient_contrast_threshold: Minimum contrast between edge and content for margin detection
        return_analysis: Whether to return analysis information

    Returns:
        Processed image or tuple with analysis
    """
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    height, width = gray.shape
    original_area = height * width

    # Step 1: Detect margin boundaries using gradient analysis
    boundaries, debug_info = detect_gradient_boundaries(
        image,
        edge_percentage,
        gradient_window_size,
        intensity_shift_threshold,
        margin_confidence_threshold,
        use_binarization,
        return_debug=True,
    )

    # Extract gradient analysis for adaptive cutting
    gradient_analyses = {
        "top": debug_info.get("top_analysis"),
        "bottom": debug_info.get("bottom_analysis"),
        "left": debug_info.get("left_analysis"),
        "right": debug_info.get("right_analysis"),
    }

    # Step 2: Process the image based on detected boundaries
    processed_image = image.copy()

    # Initialize crop boundaries (used for analysis even if not cropping)
    top = boundaries["top"]
    bottom = boundaries["bottom"]
    left = boundaries["left"]
    right = boundaries["right"]

    if fill_method == "crop":
        if direct_gradient_cutting:
            # Direct gradient cutting: calculate optimal cutting points based on gradient analysis
            # Extract profile data for margin detection (both grayscale and binary)
            edge_height = int(height * edge_percentage)
            edge_width = int(width * edge_percentage)

            # Create binary version for enhanced edge analysis
            _, binary_gray = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Grayscale profiles
            top_profile = np.mean(gray[:edge_height, :], axis=1)
            bottom_profile = np.mean(gray[height - edge_height :, :], axis=1)
            left_profile = np.mean(gray[:, :edge_width], axis=0)
            right_profile = np.mean(gray[:, width - edge_width :], axis=0)

            # Binary profiles for enhanced analysis
            # For consistent inward scanning: index 0 = actual edge, increasing = toward center
            top_binary_profile = np.mean(
                binary_gray[:edge_height, :], axis=1
            )  # Top edge, index 0 = top
            bottom_binary_profile = np.mean(
                binary_gray[height - edge_height :, :], axis=1
            )[
                ::-1
            ]  # Reverse: index 0 = bottom edge
            left_binary_profile = np.mean(
                binary_gray[:, :edge_width], axis=0
            )  # Left edge, index 0 = left
            right_binary_profile = np.mean(
                binary_gray[:, width - edge_width :], axis=0
            )[
                ::-1
            ]  # Reverse: index 0 = right edge

            # Calculate new cutting points using simplified logic with binary intensity only
            # Start with tunable binary intensity threshold - can be adjusted based on results
            binary_intensity_threshold = 200  # Aggressive threshold - cuts more margins (edges < 200 intensity will be cut, clean edges 218+ protected)

            top = _calculate_simple_cutting_point(
                gradient_analyses["top"],
                top_profile,
                top,
                height,
                "top",
                binary_intensity_threshold,
                top_binary_profile,
            )
            bottom = _calculate_simple_cutting_point(
                gradient_analyses["bottom"],
                bottom_profile,
                bottom,
                height,
                "bottom",
                binary_intensity_threshold,
                bottom_binary_profile,
            )
            left = _calculate_simple_cutting_point(
                gradient_analyses["left"],
                left_profile,
                left,
                width,
                "left",
                binary_intensity_threshold,
                left_binary_profile,
            )
            right = _calculate_simple_cutting_point(
                gradient_analyses["right"],
                right_profile,
                right,
                width,
                "right",
                binary_intensity_threshold,
                right_binary_profile,
            )

            # Simplified cutting logic handles coordinate systems correctly for all sides
            # No complex adjustments needed - just use boundary positions directly when intensity > threshold

            # Ensure we still have a valid crop region
            if top >= bottom or left >= right:
                # Fallback to conservative cropping if adaptive would eliminate image
                top = max(0, boundaries["top"] - 5)
                bottom = min(height, boundaries["bottom"] + 5)
                left = max(0, boundaries["left"] - 5)
                right = min(width, boundaries["right"] + 5)
        else:
            # Conservative cropping: add small padding away from content
            padding = 10
            top = max(0, top - padding)
            bottom = min(height, bottom + padding)
            left = max(0, left - padding)
            right = min(width, right + padding)

        processed_image = processed_image[top:bottom, left:right]

    elif fill_method == "color_fill":
        # Create margin mask and fill with page color
        margin_mask = np.zeros_like(gray, dtype=bool)

        # Mark margin areas
        margin_mask[: boundaries["top"], :] = True  # Top margin
        margin_mask[boundaries["bottom"] :, :] = True  # Bottom margin
        margin_mask[:, : boundaries["left"]] = True  # Left margin
        margin_mask[:, boundaries["right"] :] = True  # Right margin

        if np.any(margin_mask):
            # Get page background color from non-margin areas
            page_mask = ~margin_mask
            page_color = detect_page_background_color(image, page_mask)

            # Fill margin areas with page color
            if len(image.shape) == 3:
                processed_image[margin_mask] = page_color
            else:
                processed_image[margin_mask] = np.mean(page_color)

    # Calculate area retention
    if fill_method == "crop":
        new_area = processed_image.shape[0] * processed_image.shape[1]
        area_retention = new_area / original_area
    else:
        area_retention = 1.0  # No cropping, just filling

    if return_analysis:
        analysis = {
            "method": "gradient_boundary",
            "success": True,
            "boundaries": boundaries,
            "original_shape": image.shape,
            "processed_shape": processed_image.shape,
            "area_retention": area_retention,
            "debug_info": debug_info,
        }

        # Add crop_bounds for visualization compatibility
        if fill_method == "crop":
            # Calculate actual crop bounds used
            crop_width = right - left
            crop_height = bottom - top
            analysis["crop_bounds"] = (left, top, crop_width, crop_height)
        else:
            # For color_fill method, no actual cropping occurred
            analysis["crop_bounds"] = (0, 0, width, height)

        return processed_image, analysis

    return processed_image


def remove_margins_edge_transition(
    image: np.ndarray,
    edge_percentage: float = 0.15,
    intensity_jump_threshold: int = 30,
    fill_method: str = "color_fill",
    return_analysis: bool = False,
) -> tuple:
    """Remove margins using edge transition detection.

    This method finds drastic intensity changes at image edges to identify
    the boundary between dark margins and bright page content.

    Args:
        image: Input image (BGR or grayscale)
        edge_percentage: Percentage of image edge to scan for transitions
        intensity_jump_threshold: Minimum intensity jump to consider a boundary
        fill_method: Method for removing margins ("color_fill" or "crop")
        return_analysis: Whether to return analysis information

    Returns:
        Processed image or tuple with analysis
    """
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    height, width = gray.shape
    original_area = height * width

    # Step 1: Detect margin boundaries
    boundaries, debug_info = detect_edge_transitions(
        image, edge_percentage, intensity_jump_threshold, return_debug=True
    )

    # Step 2: Process the image based on detected boundaries
    processed_image = image.copy()

    if fill_method == "crop":
        # Crop to the detected boundaries
        top = boundaries["top"]
        bottom = boundaries["bottom"]
        left = boundaries["left"]
        right = boundaries["right"]

        # Add small padding to avoid cutting into content
        padding = 5
        top = max(0, top - padding)
        bottom = min(height, bottom + padding)
        left = max(0, left - padding)
        right = min(width, right + padding)

        processed_image = processed_image[top:bottom, left:right]

    elif fill_method == "color_fill":
        # Create margin mask and fill with page color
        margin_mask = np.zeros_like(gray, dtype=bool)

        # Mark margin areas
        margin_mask[: boundaries["top"], :] = True  # Top margin
        margin_mask[boundaries["bottom"] :, :] = True  # Bottom margin
        margin_mask[:, : boundaries["left"]] = True  # Left margin
        margin_mask[:, boundaries["right"] :] = True  # Right margin

        if np.any(margin_mask):
            # Get page background color from non-margin areas
            page_mask = ~margin_mask
            page_color = detect_page_background_color(image, page_mask)

            # Fill margin areas with page color
            if len(image.shape) == 3:
                processed_image[margin_mask] = page_color
            else:
                processed_image[margin_mask] = np.mean(page_color)

    # Calculate area retention
    if fill_method == "crop":
        new_area = processed_image.shape[0] * processed_image.shape[1]
        area_retention = new_area / original_area
    else:
        area_retention = 1.0  # No cropping, just filling

    if return_analysis:
        analysis = {
            "method": "edge_transition",
            "success": True,
            "boundaries": boundaries,
            "original_shape": image.shape,
            "processed_shape": processed_image.shape,
            "area_retention": area_retention,
            "debug_info": debug_info,
        }
        return processed_image, analysis

    return processed_image


def remove_margins_hybrid(
    image: np.ndarray,
    edge_margin_width: int = 50,
    texture_threshold: float = 10.0,
    black_intensity_max: int = 75,
    fill_method: str = "color_fill",
    return_analysis: bool = False,
) -> tuple:
    """Remove margins using hybrid position + texture analysis.

    This method avoids the problem of misclassifying black text as margins
    by using position and texture rather than intensity alone.

    Args:
        image: Input image (BGR or grayscale)
        edge_margin_width: Width of edge regions to analyze
        texture_threshold: Variance threshold for uniform areas
        black_intensity_max: Maximum intensity for dark areas
        fill_method: Method for filling margins
        return_analysis: Whether to return analysis info

    Returns:
        Processed image or tuple with analysis
    """
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    height, width = gray.shape
    original_area = height * width

    # Step 1: Detect margin areas using hybrid approach
    page_mask, margin_mask, debug_info = detect_margin_areas_hybrid(
        image,
        edge_margin_width,
        texture_threshold,
        black_intensity_max,
        return_debug=True,
    )

    # Step 2: Process the image based on detected margins
    processed_image = image.copy()

    if np.any(margin_mask):
        if fill_method == "color_fill":
            # Get page background color from non-margin areas
            page_color = detect_page_background_color(image, page_mask)

            # Fill margin areas with page color
            if len(image.shape) == 3:
                processed_image[margin_mask] = page_color
            else:
                processed_image[margin_mask] = np.mean(page_color)

        elif fill_method == "crop":
            # Find bounding box of page content and crop
            page_coords = np.where(page_mask)
            if len(page_coords[0]) > 0:
                min_y, max_y = np.min(page_coords[0]), np.max(page_coords[0])
                min_x, max_x = np.min(page_coords[1]), np.max(page_coords[1])

                # Add small padding
                padding = 5
                min_y = max(0, min_y - padding)
                max_y = min(height, max_y + padding)
                min_x = max(0, min_x - padding)
                max_x = min(width, max_x + padding)

                processed_image = processed_image[min_y:max_y, min_x:max_x]

    # Calculate area retention
    if fill_method == "crop":
        new_area = processed_image.shape[0] * processed_image.shape[1]
        area_retention = new_area / original_area
    else:
        area_retention = 1.0  # No cropping, just filling

    if return_analysis:
        analysis = {
            "method": "hybrid_margin_removal",
            "success": True,
            "detected_margins": debug_info.get("detected_margins", []),
            "margin_pixels": debug_info.get("margin_pixels", 0),
            "page_pixels": debug_info.get("page_pixels", original_area),
            "original_shape": image.shape,
            "processed_shape": processed_image.shape,
            "area_retention": area_retention,
            "edge_analysis": {
                k: v
                for k, v in debug_info.items()
                if "texture" in k or "intensity" in k
            },
        }
        return processed_image, analysis

    return processed_image


def remove_curved_black_background(
    image: np.ndarray,
    black_threshold: int = 30,
    min_contour_area: int = 1000,
    padding: int = 2,
    fill_method: str = "color_fill",
    return_analysis: bool = False,
) -> tuple:
    """Remove curved black background from book page images.

    This method detects the page contour, crops to bounding rectangle,
    and fills remaining black areas with page-like colors.

    Args:
        image: Input image (BGR or grayscale)
        black_threshold: Pixel intensity below which is considered black
        min_contour_area: Minimum area for valid page contour
        padding: Safety padding around detected contour
        fill_method: Method for filling black areas ("color_fill" or "inpaint")
        return_analysis: If True, returns additional analysis information

    Returns:
        np.ndarray or tuple: Processed image, or (processed_image, analysis) if return_analysis=True
    """
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    height, width = gray.shape
    original_area = height * width

    # Step 1: Create binary mask with much higher threshold to exclude gray margins
    # Use higher threshold since scanner "black" background can be gray (up to ~70)
    effective_black_threshold = max(
        black_threshold, 75
    )  # Increase to 75 for better margin detection
    page_mask = gray >= effective_black_threshold

    # Clean up the mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    page_mask_uint8 = page_mask.astype(np.uint8) * 255
    page_mask_uint8 = cv2.morphologyEx(page_mask_uint8, cv2.MORPH_CLOSE, kernel)
    page_mask_uint8 = cv2.morphologyEx(page_mask_uint8, cv2.MORPH_OPEN, kernel)
    page_mask = page_mask_uint8 > 0

    # Step 2: Find contours and filter out image-boundary contours
    contours, _ = cv2.findContours(
        page_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        # No contours found - return original image
        if return_analysis:
            analysis = {
                "method": "curved_black_background",
                "success": False,
                "error": "no_contours_found",
                "original_shape": image.shape,
                "cropped_shape": image.shape,
                "area_retention": 1.0,
            }
            return image, analysis
        return image

    # Filter contours to exclude those that touch image boundaries (likely full image)
    margin = 10  # pixels from edge
    valid_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Skip contours that start very close to image edges or are too large
        if (
            x > margin
            and y > margin
            and x + w < width - margin
            and y + h < height - margin
            and cv2.contourArea(contour) > min_contour_area
        ):
            valid_contours.append(contour)

    # Choose the best contour
    if valid_contours:
        # Use the largest valid contour (should be the page content)
        page_contour = max(valid_contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(page_contour)
    else:
        # Fallback: use largest overall contour but with stricter validation
        page_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(page_contour)

        # Check if this contour is likely the full image (too large)
        image_area = height * width
        if contour_area > 0.95 * image_area:
            # Contour is too large, likely includes margins - try tighter threshold
            tighter_mask = gray >= (effective_black_threshold + 10)
            tighter_mask_uint8 = tighter_mask.astype(np.uint8) * 255
            tighter_contours, _ = cv2.findContours(
                tighter_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if tighter_contours:
                page_contour = max(tighter_contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(page_contour)
                page_mask = tighter_mask  # Update mask to match

    # Validate contour area
    if contour_area < min_contour_area:
        # Contour too small - return original image
        if return_analysis:
            analysis = {
                "method": "curved_black_background",
                "success": False,
                "error": "contour_too_small",
                "contour_area": contour_area,
                "min_contour_area": min_contour_area,
                "original_shape": image.shape,
                "cropped_shape": image.shape,
                "area_retention": 1.0,
            }
            return image, analysis
        return image

    # Step 3: Get bounding rectangle with padding
    x, y, w, h = cv2.boundingRect(page_contour)

    # Apply padding while staying within image bounds
    x_padded = max(0, x - padding)
    y_padded = max(0, y - padding)
    x2_padded = min(width, x + w + padding)
    y2_padded = min(height, y + h + padding)

    w_padded = x2_padded - x_padded
    h_padded = y2_padded - y_padded

    # Step 4: Crop to bounding rectangle
    cropped_image = image[y_padded:y2_padded, x_padded:x2_padded]
    cropped_mask = page_mask[y_padded:y2_padded, x_padded:x2_padded]

    # Step 5: Fill ONLY margin areas (outside page contour)
    # Adjust page contour coordinates to match cropped image
    crop_bounds = (x_padded, y_padded, w_padded, h_padded)
    adjusted_contour = adjust_contour_to_crop(page_contour, crop_bounds)

    # Create mask for page interior (content area to preserve)
    page_interior_mask = create_contour_mask(adjusted_contour, cropped_image.shape)

    # Identify black pixels that are ONLY in margins (outside page contour)
    black_pixels = ~cropped_mask  # All black pixels in cropped area
    margin_black_pixels = black_pixels & (
        ~page_interior_mask
    )  # Only margins, not content

    if fill_method == "inpaint":
        # Use OpenCV inpainting for natural fill - ONLY on margin areas
        margin_areas_uint8 = margin_black_pixels.astype(np.uint8) * 255
        if len(cropped_image.shape) == 3:
            result = cv2.inpaint(
                cropped_image, margin_areas_uint8, 3, cv2.INPAINT_TELEA
            )
        else:
            result = cv2.inpaint(
                cropped_image, margin_areas_uint8, 3, cv2.INPAINT_TELEA
            )
    else:
        # Color fill method (default) - ONLY on margin areas
        page_color = detect_page_background_color(cropped_image, cropped_mask)
        result = cropped_image.copy()

        # Fill ONLY margin black areas, preserve all page content
        if len(result.shape) == 3:
            result[margin_black_pixels] = page_color
        else:
            result[margin_black_pixels] = np.mean(page_color)  # Convert to grayscale

    if not return_analysis:
        return result

    # Step 6: Create analysis information
    cropped_area = result.shape[0] * result.shape[1]

    analysis = {
        "method": "curved_black_background",
        "success": True,
        "original_shape": image.shape,
        "cropped_shape": result.shape,
        "crop_bounds": (x_padded, y_padded, w_padded, h_padded),
        "area_retention": cropped_area / original_area if original_area > 0 else 0,
        "contour_area": contour_area,
        "page_contour": page_contour.tolist(),  # Convert for JSON serialization
        "black_threshold": black_threshold,
        "fill_method": fill_method,
        "padding": padding,
        "margins_removed": {
            "left": x_padded,
            "top": y_padded,
            "right": width - x2_padded,
            "bottom": height - y2_padded,
        },
        "page_background_color": detect_page_background_color(
            cropped_image, cropped_mask
        ).tolist(),
        "debug_info": {
            "effective_black_threshold": effective_black_threshold,
            "original_dimensions": (width, height),
            "cropped_dimensions": (w_padded, h_padded),
            "contour_bounding_rect": (x, y, w, h),
            "contour_area": contour_area,
            "total_contours_found": len(contours),
            "valid_contours_found": (
                len(valid_contours) if "valid_contours" in locals() else 0
            ),
        },
    }

    return result, analysis


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


# ----------------------------------------------------------------------------- #
#                     Table Structure Detection Functions                       #
# ----------------------------------------------------------------------------- #


def cluster_line_positions(pos_list: List[int], eps: int = 10) -> List[int]:
    """
    Merge nearly identical x or y coordinates (≤ eps apart).

    Args:
        pos_list: List of positions to cluster
        eps: Maximum distance for clustering

    Returns:
        List of clustered positions (averages)
    """
    if not pos_list:
        return []
    pos_list = sorted(pos_list)
    groups, current = [], [pos_list[0]]
    for p in pos_list[1:]:
        if abs(p - current[-1]) <= eps:
            current.append(p)
        else:
            groups.append(current)
            current = [p]
    groups.append(current)
    return [int(np.mean(g)) for g in groups]


def enumerate_table_cells(
    xs: List[int], ys: List[int]
) -> List[Tuple[int, int, int, int]]:
    """
    Create cell rectangles from grid line coordinates.

    Args:
        xs: Sorted list of vertical line x-coordinates
        ys: Sorted list of horizontal line y-coordinates

    Returns:
        List of cell rectangles as (x1, y1, x2, y2) tuples
    """
    return [
        (xi, yi, xj, yj)
        for xi, xj in zip(xs[:-1], xs[1:])
        for yi, yj in zip(ys[:-1], ys[1:])
    ]


def detect_table_structure(
    h_lines: List[Tuple[int, int, int, int]],
    v_lines: List[Tuple[int, int, int, int]],
    eps: int = 10,
    return_analysis: bool = True,
) -> Union[Dict, Tuple]:
    """
    Detect table structure from pre-detected lines.

    Args:
        h_lines: List of horizontal lines as (x1, y1, x2, y2) tuples
        v_lines: List of vertical lines as (x1, y1, x2, y2) tuples
        eps: Clustering tolerance in pixels
        return_analysis: If True, return detailed analysis

    Returns:
        Dict with 'xs', 'ys', 'cells' keys

    Raises:
        ValueError: If no line information is provided
    """
    # Check if line information is provided
    if h_lines is None or v_lines is None:
        raise ValueError(
            "Line information is required. Both h_lines and v_lines must be provided."
        )

    if not isinstance(h_lines, list) or not isinstance(v_lines, list):
        raise ValueError("h_lines and v_lines must be lists of line tuples")

    # Extract y positions from horizontal lines
    y_positions = []
    for x1, y1, x2, y2 in h_lines:
        # Use average y position for each line
        y_pos = (y1 + y2) // 2
        y_positions.append(y_pos)

    # Extract x positions from vertical lines
    x_positions = []
    for x1, y1, x2, y2 in v_lines:
        # Use average x position for each line
        x_pos = (x1 + x2) // 2
        x_positions.append(x_pos)

    # Cluster positions to handle near-duplicate lines
    ys = cluster_line_positions(y_positions, eps)
    xs = cluster_line_positions(x_positions, eps)

    # Generate table cells
    cells = enumerate_table_cells(xs, ys)

    result = {"xs": xs, "ys": ys, "cells": cells}

    if return_analysis:
        # Calculate bounds if we have lines
        if xs and ys:
            bounds = {
                "min_x": min(xs),
                "max_x": max(xs),
                "min_y": min(ys),
                "max_y": max(ys),
            }
        else:
            bounds = None

        analysis = {
            "table_detected": len(xs) > 1 and len(ys) > 1,
            "num_vertical_lines": len(xs),
            "num_horizontal_lines": len(ys),
            "num_cells": len(cells),
            "table_bounds": bounds,
            "clustering_eps": eps,
            "lines_input": {
                "horizontal_count": len(h_lines),
                "vertical_count": len(v_lines),
            },
        }
        result["analysis"] = analysis

    return result


def crop_to_table_borders(
    deskewed_image: np.ndarray,
    table_structure: Dict,
    padding: int = 20,
    return_analysis: bool = False,
) -> Union[np.ndarray, Tuple]:
    """
    Crop deskewed image to table borders with padding.

    Args:
        deskewed_image: The deskewed image to crop
        table_structure: Result from detect_table_structure()
        padding: Padding around detected table borders
        return_analysis: If True, return analysis info

    Returns:
        Cropped image, or (cropped_image, analysis) if return_analysis=True
    """
    cells = table_structure.get("cells", [])
    xs = table_structure.get("xs", [])
    ys = table_structure.get("ys", [])

    # Prefer using cell boundaries if available
    if cells:
        # Calculate boundaries from cells
        min_x = min(x1 for x1, y1, x2, y2 in cells)
        max_x = max(x2 for x1, y1, x2, y2 in cells)
        min_y = min(y1 for x1, y1, x2, y2 in cells)
        max_y = max(y2 for x1, y1, x2, y2 in cells)
        crop_method = "cells"
    elif xs and ys:
        # Fall back to grid lines if no cells detected
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        crop_method = "grid_lines"
    else:
        # No table detected, return original image
        if return_analysis:
            analysis = {
                "cropped": False,
                "reason": "No table structure detected",
                "original_shape": deskewed_image.shape,
                "cropped_shape": deskewed_image.shape,
            }
            return deskewed_image, analysis
        return deskewed_image

    # Apply padding and ensure bounds are within image
    min_x = max(0, min_x - padding)
    max_x = min(deskewed_image.shape[1], max_x + padding)
    min_y = max(0, min_y - padding)
    max_y = min(deskewed_image.shape[0], max_y + padding)

    # Crop the image
    cropped = deskewed_image[min_y:max_y, min_x:max_x]

    if return_analysis:
        # Calculate actual table bounds based on method used
        if crop_method == "cells":
            table_min_x = min(x1 for x1, y1, x2, y2 in cells)
            table_max_x = max(x2 for x1, y1, x2, y2 in cells)
            table_min_y = min(y1 for x1, y1, x2, y2 in cells)
            table_max_y = max(y2 for x1, y1, x2, y2 in cells)
        else:
            table_min_x = min(xs)
            table_max_x = max(xs)
            table_min_y = min(ys)
            table_max_y = max(ys)

        analysis = {
            "cropped": True,
            "crop_method": crop_method,
            "original_shape": deskewed_image.shape,
            "cropped_shape": cropped.shape,
            "crop_bounds": {
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
            },
            "padding_applied": padding,
            "table_bounds": {
                "min_x": table_min_x,
                "max_x": table_max_x,
                "min_y": table_min_y,
                "max_y": table_max_y,
            },
            "size_reduction": {
                "width": deskewed_image.shape[1] - cropped.shape[1],
                "height": deskewed_image.shape[0] - cropped.shape[0],
            },
            "num_cells_used": len(cells) if crop_method == "cells" else 0,
        }
        return cropped, analysis

    return cropped


def visualize_table_structure(
    image: np.ndarray,
    table_structure: Dict,
    cell_color: Tuple[int, int, int] = (0, 0, 255),
    line_color: Tuple[int, int, int] = (0, 255, 0),
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Create visualization of detected table structure.

    Args:
        image: Base image for visualization
        table_structure: Result from detect_table_structure()
        cell_color: Color for cell rectangles (B, G, R)
        line_color: Color for grid lines (B, G, R)
        line_thickness: Thickness of drawn lines

    Returns:
        Image with table structure overlay
    """
    vis = image.copy()

    xs = table_structure.get("xs", [])
    ys = table_structure.get("ys", [])
    cells = table_structure.get("cells", [])

    # Draw grid lines
    height, width = image.shape[:2]
    for x in xs:
        cv2.line(vis, (x, 0), (x, height), line_color, line_thickness)
    for y in ys:
        cv2.line(vis, (0, y), (width, y), line_color, line_thickness)

    # Draw cell rectangles
    for x1, y1, x2, y2 in cells:
        cv2.rectangle(vis, (x1, y1), (x2, y2), cell_color, line_thickness)

    return vis
