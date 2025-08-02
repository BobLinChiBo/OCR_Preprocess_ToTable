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


def detect_table_lines(
    image: np.ndarray,
    threshold: int = 40,  # Binary threshold
    horizontal_kernel_size: int = 10,  # Morphological kernel width
    vertical_kernel_size: int = 10,  # Morphological kernel height
    alignment_threshold: int = 3,  # Clustering threshold for line alignment
    pre_merge_length_ratio: float = 0.3,  # Min length ratio before merging (0 = no filter)
    post_merge_length_ratio: float = 0.4,  # Min length ratio after merging
    min_aspect_ratio: int = 5,  # Min aspect ratio for line-like components
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
        pre_merge_length_ratio: Minimum length ratio before merging (0 = no filtering)
        post_merge_length_ratio: Minimum length ratio after merging
        min_aspect_ratio: Minimum aspect ratio to consider component as line-like
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
        pre_merge_length_ratio,
        min_aspect_ratio,
    )

    merged_horizontal, horizontal_segments = extract_and_merge_lines(
        horizontal_lines,
        "horizontal",
        alignment_threshold,
        pre_merge_length_ratio,
        min_aspect_ratio,
    )

    # Post-merge length filtering
    v_lengths = [y2 - y1 for x1, y1, x2, y2 in merged_vertical]
    h_lengths = [x2 - x1 for x1, y1, x2, y2 in merged_horizontal]

    v_thresh = post_merge_length_ratio * max(v_lengths) if v_lengths else 0
    h_thresh = post_merge_length_ratio * max(h_lengths) if h_lengths else 0

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

    # Convert to the expected format
    h_lines = filtered_horizontal
    v_lines = filtered_vertical

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
            return image, {"cropped": False, "reason": "No lines detected"}
        return image

    # Get table boundaries from lines
    h_ys = [y for _, y, _, _ in h_lines] + [y for _, _, _, y in h_lines]
    v_xs = [x for x, _, _, _ in v_lines] + [x for _, _, x, _ in v_lines]

    min_x = max(0, min(v_xs) - crop_padding)
    max_x = min(image.shape[1], max(v_xs) + crop_padding)
    min_y = max(0, min(h_ys) - crop_padding)
    max_y = min(image.shape[0], max(h_ys) + crop_padding)

    # Crop the image
    cropped = image[min_y:max_y, min_x:max_x]

    if not return_analysis:
        return cropped

    analysis = {
        "cropped": True,
        "original_shape": image.shape,
        "cropped_shape": cropped.shape,
        "boundaries": {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
        },
        "table_width": max_x - min_x,
        "table_height": max_y - min_y,
    }

    return cropped, analysis


def compute_roi_histogram_projection(
    binary_image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute horizontal and vertical histogram projections for ROI detection.

    Args:
        binary_image: Binary image (white text on black background)

    Returns:
        tuple: (horizontal_projection, vertical_projection)
    """
    # Ensure binary image
    if len(binary_image.shape) == 3:
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)

    # Compute projections
    horizontal_projection = np.sum(binary_image, axis=1)
    vertical_projection = np.sum(binary_image, axis=0)

    return horizontal_projection, vertical_projection


def find_content_boundaries(
    projection: np.ndarray,
    min_content_threshold: float = 0.01,
    gap_threshold: int = 50,
    edge_margin: float = 0.05,
) -> Tuple[int, int]:
    """Find content boundaries from histogram projection.

    Args:
        projection: 1D histogram projection
        min_content_threshold: Minimum relative content to consider as valid
        gap_threshold: Minimum gap size to consider as boundary
        edge_margin: Margin from edges as fraction of total size

    Returns:
        tuple: (start_idx, end_idx)
    """
    max_val = np.max(projection)
    if max_val == 0:
        return 0, len(projection) - 1

    # Threshold for content detection
    threshold = max_val * min_content_threshold

    # Find content regions
    content_mask = projection > threshold
    content_indices = np.where(content_mask)[0]

    if len(content_indices) == 0:
        return 0, len(projection) - 1

    # Simple approach: find first and last content
    start_idx = content_indices[0]
    end_idx = content_indices[-1]

    # Apply edge margin
    margin = int(len(projection) * edge_margin)
    start_idx = max(margin, start_idx)
    end_idx = min(len(projection) - margin - 1, end_idx)

    return start_idx, end_idx


def detect_roi_simple(
    image: np.ndarray,
    threshold_method: str = "otsu",
    min_content_threshold: float = 0.01,
    return_analysis: bool = False,
) -> tuple:
    """Simple ROI detection using histogram projections.

    Args:
        image: Input image
        threshold_method: Thresholding method ('otsu' or 'adaptive')
        min_content_threshold: Minimum content threshold for boundary detection
        return_analysis: If True, returns detailed analysis

    Returns:
        tuple: (x, y, w, h) or ((x, y, w, h), analysis) if return_analysis=True
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Apply thresholding
    if threshold_method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
        )

    # Compute projections
    h_proj, v_proj = compute_roi_histogram_projection(binary)

    # Find boundaries
    y_start, y_end = find_content_boundaries(h_proj, min_content_threshold)
    x_start, x_end = find_content_boundaries(v_proj, min_content_threshold)

    # Calculate ROI
    x, y = x_start, y_start
    w = x_end - x_start + 1
    h = y_end - y_start + 1

    # Validate ROI
    if w <= 0 or h <= 0:
        x, y, w, h = 0, 0, image.shape[1], image.shape[0]

    if not return_analysis:
        return (x, y, w, h)

    analysis = {
        "threshold_method": threshold_method,
        "horizontal_projection": h_proj,
        "vertical_projection": v_proj,
        "boundaries": {
            "x_start": x_start,
            "x_end": x_end,
            "y_start": y_start,
            "y_end": y_end,
        },
        "roi": (x, y, w, h),
        "binary_image": binary,
    }

    return (x, y, w, h), analysis


def detect_curved_margins(
    image: np.ndarray,
    blur_kernel_size: int = 7,
    black_threshold: int = 50,
    content_threshold: int = 200,
    morph_kernel_size: int = 25,
    min_content_area_ratio: float = 0.1,
    return_debug_info: bool = False,
) -> tuple:
    """Detect curved margins and create content mask.

    Args:
        image: Input image (BGR or grayscale)
        blur_kernel_size: Gaussian blur kernel size
        black_threshold: Threshold for detecting very dark regions (margins)
        content_threshold: Threshold for detecting content regions
        morph_kernel_size: Morphological operation kernel size
        min_content_area_ratio: Minimum content area ratio to consider valid
        return_debug_info: If True, return debug information

    Returns:
        tuple: (content_mask, debug_info) if return_debug_info else content_mask
    """
    # Convert to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    height, width = gray.shape
    total_area = height * width

    # Step 1: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)

    # Step 2: Create mask of content regions (not very dark)
    _, content_mask = cv2.threshold(
        blurred, content_threshold, 255, cv2.THRESH_BINARY_INV
    )

    # Step 3: Create mask of potential margins (very dark areas)
    _, margin_mask = cv2.threshold(blurred, black_threshold, 255, cv2.THRESH_BINARY_INV)

    # Step 4: Apply morphological operations to connect content regions
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 5: Find all content contours
    contours, _ = cv2.findContours(
        content_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Step 6: Filter contours by area and find the main content region
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > total_area * min_content_area_ratio:
            valid_contours.append(contour)

    # Step 7: Create final content mask
    if valid_contours:
        # If multiple valid contours, merge them
        final_mask = np.zeros_like(gray)
        cv2.drawContours(final_mask, valid_contours, -1, 255, -1)

        # Optional: Fill any holes within the content region
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_small)

        debug_info = {
            "gray": gray,
            "blurred": blurred,
            "initial_content_mask": content_mask,
            "margin_mask": margin_mask,
            "valid_contours": valid_contours,
            "all_contours": contours,
        }
    else:
        # No valid contours found, use the whole image
        final_mask = np.ones_like(gray) * 255
        debug_info = {
            "gray": gray,
            "blurred": blurred,
            "initial_content_mask": content_mask,
            "margin_mask": margin_mask,
            "valid_contours": [],
            "all_contours": contours,
        }

    if return_debug_info:
        return final_mask, debug_info
    return final_mask


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
