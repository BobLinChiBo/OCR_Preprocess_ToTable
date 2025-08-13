"""Table line detection and structure analysis for OCR pipeline."""

from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .base import BaseProcessor


class TableDetectionProcessor(BaseProcessor):
    """Processor for detecting table lines and structure."""

    def process(
        self, image: np.ndarray, **kwargs
    ) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
        """Detect table lines in an image.

        Args:
            image: Input image
            **kwargs: Parameters for table line detection

        Returns:
            Tuple of (horizontal_lines, vertical_lines)
        """
        self.validate_image(image)

        # Clear previous debug images
        self.clear_debug_images()

        # Pass processor instance to detect_table_lines for debug saving
        kwargs["_processor"] = self

        return detect_table_lines(image, **kwargs)


def detect_table_lines(
    image: np.ndarray,
    threshold: int = 40,  # Binary threshold
    horizontal_kernel_size: int = 8,  # Morphological kernel width
    vertical_kernel_size: int = 8,  # Morphological kernel height
    alignment_threshold: int = 3,  # Clustering threshold for line alignment
    # New separated parameters for horizontal lines
    h_min_length_image_ratio: float = 0.0,  # Min horizontal length as ratio of image width
    h_min_length_relative_ratio: float = 0.0,  # Min horizontal length relative to longest h-line
    # New separated parameters for vertical lines
    v_min_length_image_ratio: float = 0.0,  # Min vertical length as ratio of image height
    v_min_length_relative_ratio: float = 0.0,  # Min vertical length relative to longest v-line
    min_aspect_ratio: int = 2,  # Min aspect ratio for line-like components
    # New post-processing parameters
    max_h_length_ratio: float = 1.0,  # Max horizontal line length ratio (1.0 = disable filtering)
    max_v_length_ratio: float = 1.0,  # Max vertical line length ratio (1.0 = disable filtering)
    close_line_distance: int = 45,  # Distance for merging close lines (0 = disable)
    # Search region parameters
    search_region_top: int = 0,  # Pixels to ignore from top
    search_region_bottom: int = 0,  # Pixels to ignore from bottom
    search_region_left: int = 0,  # Pixels to ignore from left
    search_region_right: int = 0,  # Pixels to ignore from right
    # Skew tolerance parameters
    skew_tolerance: float = 0,  # Maximum angle in degrees to tolerate for skewed lines
    skew_angle_step: float = 0.2,  # Step size for angle search
    # Line detection preprocessing parameters
    line_detection_use_preprocessing: bool = False,  # Enable preprocessing for better detection
    line_detection_binarization_method: str = "adaptive",  # Binarization method
    line_detection_binarization_threshold: int = 127,  # For fixed method
    line_detection_adaptive_block_size: int = 17,  # For adaptive method
    line_detection_adaptive_c: int = 5,  # For adaptive method
    line_detection_binarization_invert: bool = False,  # Invert black/white
    line_detection_binarization_denoise: bool = True,  # Apply denoising
    line_detection_stroke_enhancement: bool = False,  # Enable stroke enhancement
    line_detection_stroke_kernel_size: int = 1,  # Stroke kernel size
    line_detection_stroke_iterations: int = 3,  # Stroke iterations
    line_detection_stroke_kernel_shape: str = "cross",  # Stroke kernel shape
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
        skew_tolerance: Maximum angle in degrees to tolerate for skewed lines (0 = disable)
        skew_angle_step: Step size for angle search when skew_tolerance > 0
        return_analysis: If True, returns additional statistics

    Returns:
        tuple: (h_lines, v_lines) or (h_lines, v_lines, analysis) if return_analysis=True
    """
    # Get processor instance if available for debug saving
    processor = kwargs.get("_processor", None)

    # Convert to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    img_h, img_w = gray.shape
    
    # Store original image for later use
    original_gray = gray.copy()
    
    # Apply preprocessing if enabled (for better line detection)
    if line_detection_use_preprocessing:
        # Import binarization processor
        from .binarize import BinarizeProcessor
        
        # Create temporary binarize processor for preprocessing
        binarizer = BinarizeProcessor()
        
        # Apply binarization and stroke enhancement for better line detection
        preprocessed = binarizer.process(
            gray,
            method=line_detection_binarization_method,
            threshold=line_detection_binarization_threshold,
            adaptive_block_size=line_detection_adaptive_block_size,
            adaptive_c=line_detection_adaptive_c,
            invert=line_detection_binarization_invert,
            denoise=line_detection_binarization_denoise,
            enhance_strokes=line_detection_stroke_enhancement,
            stroke_kernel_size=line_detection_stroke_kernel_size,
            stroke_iterations=line_detection_stroke_iterations,
            stroke_kernel_shape=line_detection_stroke_kernel_shape,
        )
        
        # Use preprocessed image for line detection
        gray = preprocessed
        
        # Save debug image if processor available
        if processor:
            processor.save_debug_image("preprocessed_for_lines", gray)

    # Apply search region by cropping if specified
    if (
        search_region_top > 0
        or search_region_bottom > 0
        or search_region_left > 0
        or search_region_right > 0
    ):
        # Calculate crop boundaries
        y1 = search_region_top
        y2 = img_h - search_region_bottom
        x1 = search_region_left
        x2 = img_w - search_region_right

        # Ensure valid bounds
        y1 = max(0, min(y1, img_h))
        y2 = max(0, min(y2, img_h))
        x1 = max(0, min(x1, img_w))
        x2 = max(0, min(x2, img_w))

        if y2 > y1 and x2 > x1:
            # Save debug image showing search region before cropping
            if processor:
                search_vis = (
                    cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    if len(gray.shape) == 2
                    else gray.copy()
                )
                cv2.rectangle(search_vis, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
                processor.save_debug_image("search_region", search_vis)

            # Crop the image to the search region
            gray = gray[y1:y2, x1:x2]

            # Store offsets for later coordinate adjustment
            y1_offset = y1
            x1_offset = x1
        else:
            # Invalid search region, use full image
            y1_offset = 0
            x1_offset = 0
    else:
        # No offset needed if no search region
        y1_offset = 0
        x1_offset = 0

    # Invert image (make dark lines white)
    gray_inv = cv2.bitwise_not(gray)

    # Binary thresholding
    _, binary = cv2.threshold(gray_inv, threshold, 255, cv2.THRESH_BINARY)

    # Save debug image
    if processor:
        processor.save_debug_image("binary_threshold", binary)

    # Morphological opening to extract lines
    if skew_tolerance > 0:
        # Multi-angle detection for skewed lines
        vertical_lines = np.zeros_like(binary)
        horizontal_lines = np.zeros_like(binary)
        
        # Generate angles to test
        angles = np.arange(-skew_tolerance, skew_tolerance + skew_angle_step, skew_angle_step)
        
        for angle in angles:
            # Get rotation matrix
            rows, cols = binary.shape
            center = (cols // 2, rows // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Rotate the binary image
            rotated = cv2.warpAffine(binary, M, (cols, rows), flags=cv2.INTER_NEAREST)
            
            # Apply morphological operations on rotated image
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_size))
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_size, 1))
            v_lines_rot = cv2.morphologyEx(rotated, cv2.MORPH_OPEN, kernel_v, iterations=1)
            h_lines_rot = cv2.morphologyEx(rotated, cv2.MORPH_OPEN, kernel_h, iterations=1)
            
            # Rotate back to original orientation
            M_inv = cv2.getRotationMatrix2D(center, -angle, 1.0)
            v_lines_back = cv2.warpAffine(v_lines_rot, M_inv, (cols, rows), flags=cv2.INTER_NEAREST)
            h_lines_back = cv2.warpAffine(h_lines_rot, M_inv, (cols, rows), flags=cv2.INTER_NEAREST)
            
            # Accumulate results using bitwise OR
            vertical_lines = cv2.bitwise_or(vertical_lines, v_lines_back)
            horizontal_lines = cv2.bitwise_or(horizontal_lines, h_lines_back)
        
        # Clean up accumulated results with a small morphological closing
        kernel_cleanup = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        vertical_lines = cv2.morphologyEx(vertical_lines, cv2.MORPH_CLOSE, kernel_cleanup, iterations=1)
        horizontal_lines = cv2.morphologyEx(horizontal_lines, cv2.MORPH_CLOSE, kernel_cleanup, iterations=1)
    else:
        # Standard single-angle detection
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_size))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_size, 1))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v, iterations=1)
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h, iterations=1)

    # Save morphological operation results
    if processor:
        processor.save_debug_image("vertical_morph", vertical_lines)
        processor.save_debug_image("horizontal_morph", horizontal_lines)

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

    # Save debug image showing filtered lines
    if processor:
        # Create visualization of filtered lines
        filtered_vis = np.zeros_like(gray)
        for x1, y1, x2, y2 in h_lines:
            cv2.line(filtered_vis, (x1, y1), (x2, y2), 255, 2)
        for x1, y1, x2, y2 in v_lines:
            cv2.line(filtered_vis, (x1, y1), (x2, y2), 255, 2)
        processor.save_debug_image("filtered_lines", filtered_vis)

        # Create connected components visualization
        components_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Color code connected components
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
        for i, (x1, y1, x2, y2) in enumerate(h_lines[:20]):  # Show first 20 lines
            color = colors[i % len(colors)]
            cv2.line(components_vis, (x1, y1), (x2, y2), color, 3)
        for i, (x1, y1, x2, y2) in enumerate(v_lines[:20]):  # Show first 20 lines
            color = colors[i % len(colors)]
            cv2.line(components_vis, (x1, y1), (x2, y2), color, 3)
        processor.save_debug_image("connected_components", components_vis)

    # Adjust line coordinates if search region was used
    if y1_offset > 0 or x1_offset > 0:
        h_lines = [
            (x1 + x1_offset, y1 + y1_offset, x2 + x1_offset, y2 + y1_offset)
            for x1, y1, x2, y2 in h_lines
        ]
        v_lines = [
            (x1 + x1_offset, y1 + y1_offset, x2 + x1_offset, y2 + y1_offset)
            for x1, y1, x2, y2 in v_lines
        ]

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
    }

    return h_lines, v_lines, analysis


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

    def merge_lines_by_proximity(lines, is_horizontal):
        """Merge a list of lines based on proximity."""
        if not lines:
            return []

        # Sort lines by their position (y for horizontal, x for vertical)
        if is_horizontal:
            sorted_lines = sorted(lines, key=lambda l: l[1])  # Sort by y1
        else:
            sorted_lines = sorted(lines, key=lambda l: l[0])  # Sort by x1

        merged = []
        current_group = [sorted_lines[0]]

        for line in sorted_lines[1:]:
            if is_horizontal:
                # Check if horizontal line is close to current group
                group_y = np.mean([l[1] for l in current_group])
                if abs(line[1] - group_y) <= distance_threshold:
                    current_group.append(line)
                else:
                    # Merge current group and start new one
                    merged.append(merge_line_group(current_group, is_horizontal))
                    current_group = [line]
            else:
                # Check if vertical line is close to current group
                group_x = np.mean([l[0] for l in current_group])
                if abs(line[0] - group_x) <= distance_threshold:
                    current_group.append(line)
                else:
                    # Merge current group and start new one
                    merged.append(merge_line_group(current_group, is_horizontal))
                    current_group = [line]

        # Don't forget the last group
        if current_group:
            merged.append(merge_line_group(current_group, is_horizontal))

        return merged

    def merge_line_group(group, is_horizontal):
        """Merge a group of lines into a single line."""
        if len(group) == 1:
            return group[0]

        if is_horizontal:
            # For horizontal lines, take min x1, max x2, and average y
            x1 = min(l[0] for l in group)
            x2 = max(l[2] for l in group)
            y = int(np.mean([l[1] for l in group]))
            return (x1, y, x2, y)
        else:
            # For vertical lines, take average x and min y1, max y2
            x = int(np.mean([l[0] for l in group]))
            y1 = min(l[1] for l in group)
            y2 = max(l[3] for l in group)
            return (x, y1, x, y2)

    # Merge horizontal and vertical lines separately
    merged_h = merge_lines_by_proximity(h_lines, is_horizontal=True)
    merged_v = merge_lines_by_proximity(v_lines, is_horizontal=False)

    return merged_h, merged_v


def cluster_line_positions(pos_list: List[int], eps: int = 10) -> List[int]:
    """
    Merge nearly identical x or y coordinates (<= eps apart).

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
        return_analysis: If True, returns analysis data

    Returns:
        Dictionary with detected structure or tuple with analysis
    """
    # Extract unique positions
    x_positions = []
    y_positions = []

    for x1, y1, x2, y2 in v_lines:
        x_positions.append(x1)

    for x1, y1, x2, y2 in h_lines:
        y_positions.append(y1)

    # Cluster positions
    xs = cluster_line_positions(x_positions, eps)
    ys = cluster_line_positions(y_positions, eps)

    # Sort positions
    xs = sorted(xs)
    ys = sorted(ys)

    result = {
        "xs": xs,
        "ys": ys,
        "cells": enumerate_table_cells(xs, ys) if len(xs) > 1 and len(ys) > 1 else [],
        "num_rows": len(ys) - 1 if len(ys) > 1 else 0,
        "num_cols": len(xs) - 1 if len(xs) > 1 else 0,
    }

    if not return_analysis:
        return result

    analysis = {
        "h_lines_count": len(h_lines),
        "v_lines_count": len(v_lines),
        "unique_x_count": len(xs),
        "unique_y_count": len(ys),
        "clustering_eps": eps,
        "detected_cells": len(result["cells"]),
    }

    return result, analysis


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
