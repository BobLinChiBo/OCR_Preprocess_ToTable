#!/usr/bin/env python3
"""
ROI Visualization Script

This script creates visual overlays showing detected ROI regions on your images,
helping you assess the quality of ROI detection and adjust parameters.

By default, this script displays analysis graphs and cutting point indicators
similar to the page split visualization, providing immediate visual feedback
on ROI detection quality and parameter effectiveness.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import sys
from typing import Dict, Any

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.ocr_pipeline.config import Stage1Config, Stage2Config
import src.ocr_pipeline.utils as ocr_utils
from output_manager import get_test_images, convert_numpy_types, save_step_parameters


def load_config_from_file(config_path: Path = None, stage: int = 1):
    """Load configuration from JSON file or use defaults."""
    if config_path is None:
        # Use default config based on stage
        if stage == 2:
            config_path = Path("configs/stage2_default.json")
        else:
            config_path = Path("configs/stage1_default.json")

    if config_path.exists():
        if stage == 2:
            return Stage2Config.from_json(config_path)
        else:
            return Stage1Config.from_json(config_path)
    else:
        print(f"Warning: Config file {config_path} not found, using hardcoded defaults")
        if stage == 2:
            return Stage2Config()
        else:
            return Stage1Config()


def draw_roi_overlay(
    image: np.ndarray,
    roi_coords: Dict[str, Any],
    show_gabor: bool = False,
    config=None,
    analysis: Dict = None,
) -> np.ndarray:
    """Draw ROI detection overlay on image with search regions and cut lines."""
    overlay = image.copy()
    height, width = image.shape[:2]

    # Create semi-transparent overlay
    roi_overlay = np.zeros_like(overlay)

    # Extract ROI coordinates
    left = roi_coords["roi_left"]
    right = roi_coords["roi_right"]
    top = roi_coords["roi_top"]
    bottom = roi_coords["roi_bottom"]

    # Draw search regions first (very transparent)
    if analysis:
        # Vertical search regions (70% left, 30% right)
        left_search_end = int(width * 0.7)
        right_search_start = int(width * 0.3)

        # Left search region (light blue)
        cv2.rectangle(roi_overlay, (0, 0), (left_search_end, height), (255, 255, 0), -1)
        # Right search region (light yellow)
        cv2.rectangle(
            roi_overlay, (right_search_start, 0), (width, height), (0, 255, 255), -1
        )

        # Horizontal search regions (60% top, 40% bottom)
        top_search_end = int(height * 0.6)
        bottom_search_start = int(height * 0.4)

        # Top search region (light green) - blend with existing
        cv2.rectangle(roi_overlay, (0, 0), (width, top_search_end), (0, 255, 0), -1)
        # Bottom search region (light pink) - blend with existing
        cv2.rectangle(
            roi_overlay, (0, bottom_search_start), (width, height), (255, 0, 255), -1
        )

    # Draw ROI rectangle (green for included region)
    cv2.rectangle(roi_overlay, (left, top), (right, bottom), (0, 255, 0), -1)

    # Draw excluded regions (red overlay with higher alpha)
    excluded_overlay = np.zeros_like(overlay)
    if left > 0:  # Left exclusion
        cv2.rectangle(excluded_overlay, (0, 0), (left, height), (0, 0, 255), -1)
    if right < width:  # Right exclusion
        cv2.rectangle(excluded_overlay, (right, 0), (width, height), (0, 0, 255), -1)
    if top > 0:  # Top exclusion
        cv2.rectangle(excluded_overlay, (0, 0), (width, top), (0, 0, 255), -1)
    if bottom < height:  # Bottom exclusion
        cv2.rectangle(excluded_overlay, (0, bottom), (width, height), (0, 0, 255), -1)

    # Blend overlays with original image
    alpha_search = 0.1  # Very light for search regions
    alpha_excluded = 0.3  # More visible for excluded regions
    overlay = cv2.addWeighted(overlay, 1 - alpha_search, roi_overlay, alpha_search, 0)
    overlay = cv2.addWeighted(
        overlay, 1 - alpha_excluded, excluded_overlay, alpha_excluded, 0
    )

    # Draw cut lines with different colors based on whether they were applied
    font = cv2.FONT_HERSHEY_SIMPLEX

    if analysis and "vertical_info" in analysis:
        vertical_info = analysis["vertical_info"]

        # Left cut line
        left_applied = vertical_info.get("left_cut_applied", False)
        left_color = (
            (0, 255, 0) if left_applied else (0, 100, 255)
        )  # Green if applied, orange if not
        if left > 0:
            cv2.line(overlay, (left, 0), (left, height), left_color, 3)
            # Add cut strength annotation
            if "left_cut_strength" in vertical_info:
                strength = vertical_info["left_cut_strength"]
                cv2.putText(
                    overlay,
                    f"L:{strength:.1f}",
                    (left + 5, 25),
                    font,
                    0.5,
                    left_color,
                    2,
                )

        # Right cut line
        right_applied = vertical_info.get("right_cut_applied", False)
        right_color = (0, 255, 0) if right_applied else (0, 100, 255)
        if right < width:
            cv2.line(overlay, (right, 0), (right, height), right_color, 3)
            # Add cut strength annotation
            if "right_cut_strength" in vertical_info:
                strength = vertical_info["right_cut_strength"]
                cv2.putText(
                    overlay,
                    f"R:{strength:.1f}",
                    (right - 60, 25),
                    font,
                    0.5,
                    right_color,
                    2,
                )

    if analysis and "horizontal_info" in analysis:
        horizontal_info = analysis["horizontal_info"]

        # Top cut line
        top_applied = horizontal_info.get("top_cut_applied", False)
        top_color = (0, 255, 0) if top_applied else (0, 100, 255)
        if top > 0:
            cv2.line(overlay, (0, top), (width, top), top_color, 3)
            # Add cut strength annotation
            if "top_cut_strength" in horizontal_info:
                strength = horizontal_info["top_cut_strength"]
                cv2.putText(
                    overlay, f"T:{strength:.1f}", (5, top - 5), font, 0.5, top_color, 2
                )

        # Bottom cut line
        bottom_applied = horizontal_info.get("bottom_cut_applied", False)
        bottom_color = (0, 255, 0) if bottom_applied else (0, 100, 255)
        if bottom < height:
            cv2.line(overlay, (0, bottom), (width, bottom), bottom_color, 3)
            # Add cut strength annotation
            if "bottom_cut_strength" in horizontal_info:
                strength = horizontal_info["bottom_cut_strength"]
                cv2.putText(
                    overlay,
                    f"B:{strength:.1f}",
                    (5, bottom + 20),
                    font,
                    0.5,
                    bottom_color,
                    2,
                )

    # Draw ROI boundary rectangle
    cv2.rectangle(overlay, (left, top), (right, bottom), (0, 255, 0), 2)

    # Add coordinate and statistics text
    font_scale = 0.6
    thickness = 2

    # ROI dimensions text
    roi_width = right - left
    roi_height = bottom - top
    coverage = (roi_width * roi_height) / (width * height) * 100

    text_lines = [
        f"ROI: ({left}, {top}) to ({right}, {bottom})",
        f"Size: {roi_width} x {roi_height}",
        f"Coverage: {coverage:.1f}%",
    ]

    # Add method info if available
    if config:
        method = getattr(config, "roi_detection_method", "gabor")
        text_lines.insert(0, f"Method: {method}")

    # Draw text background
    text_y_start = 30
    for i, line in enumerate(text_lines):
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        cv2.rectangle(
            overlay,
            (10, text_y_start + i * 30 - 20),
            (20 + text_size[0], text_y_start + i * 30 + 8),
            (0, 0, 0),
            -1,
        )

    # Draw text
    for i, line in enumerate(text_lines):
        cv2.putText(
            overlay,
            line,
            (15, text_y_start + i * 30),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    # Show edge detection response if requested
    if show_gabor and config and analysis and "binary_mask" in analysis:
        binary_mask = analysis["binary_mask"]

        # Create a small overlay for edge detection response
        mask_small = cv2.resize(binary_mask, (200, 150))
        mask_colored = cv2.applyColorMap(mask_small, cv2.COLORMAP_JET)

        # Overlay response in top-right corner
        y_offset = 10
        x_offset = width - 210
        overlay[y_offset : y_offset + 150, x_offset : x_offset + 200] = mask_colored

        # Label for edge detection method
        method = getattr(config, "roi_detection_method", "gabor")
        method_name = method.replace("_", " ").title()
        cv2.putText(
            overlay,
            f"{method_name} Response",
            (x_offset, y_offset - 5),
            font,
            0.5,
            (255, 255, 255),
            2,
        )

    return overlay


def create_comparison_grid(
    original: np.ndarray,
    roi_overlay: np.ndarray,
    cropped_roi: np.ndarray = None,
    analysis: Dict = None,
    roi_coords: Dict = None,
    show_projections: bool = True,
) -> np.ndarray:
    """Create a comprehensive comparison grid with projection graphs (enabled by default)."""
    # Resize images to same height for comparison
    target_height = 400  # Reduced to make room for projections

    # Resize original and overlay
    scale = target_height / original.shape[0]
    new_width = int(original.shape[1] * scale)

    orig_resized = cv2.resize(original, (new_width, target_height))
    overlay_resized = cv2.resize(roi_overlay, (new_width, target_height))

    # Create labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_height = 30

    def create_label(
        text: str, width: int, color: tuple = (255, 255, 255)
    ) -> np.ndarray:
        label_img = np.zeros((label_height, width, 3), dtype=np.uint8)
        cv2.putText(label_img, text, (10, 20), font, 0.6, color, 2)
        return label_img

    # Create main image panels
    orig_label = create_label("Original", new_width)
    overlay_label = create_label("ROI Detection", new_width, (0, 255, 0))

    # Build the layout - always show projections if analysis data is available
    if show_projections and analysis and roi_coords:
        # Create projection plots
        vertical_info = analysis.get("vertical_info", {})
        horizontal_info = analysis.get("horizontal_info", {})

        vertical_plot = create_vertical_projection_plot(
            vertical_info, original.shape[1], roi_coords
        )
        horizontal_plot = create_horizontal_projection_plot(
            horizontal_info, original.shape[0], roi_coords
        )

        # Resize vertical projection plot to match image width
        v_plot_resized = cv2.resize(vertical_plot, (new_width, vertical_plot.shape[0]))

        # Create projection labels
        v_proj_label = create_label("Vertical Projection", new_width, (255, 255, 0))

        # Left column: Original + Overlay + Vertical Projection (similar to page split layout)
        left_panel = np.vstack(
            [
                orig_label,
                orig_resized,
                overlay_label,
                overlay_resized,
                v_proj_label,
                v_plot_resized,
            ]
        )

        # Right column: Cropped ROI + Horizontal Projection
        if cropped_roi is not None and cropped_roi.size > 0:
            # Resize cropped ROI to fit nicely
            crop_scale = target_height / cropped_roi.shape[0]
            crop_width = int(cropped_roi.shape[1] * crop_scale)
            cropped_resized = cv2.resize(cropped_roi, (crop_width, target_height))

            # Resize horizontal plot to match crop width
            h_plot_height = min(250, horizontal_plot.shape[0])  # Reasonable height
            h_plot_resized = cv2.resize(horizontal_plot, (crop_width, h_plot_height))

            crop_label = create_label("Cropped ROI", crop_width, (0, 255, 255))
            h_proj_label = create_label(
                "Horizontal Projection", crop_width, (255, 0, 255)
            )

            # Combine cropped ROI with horizontal projection
            right_panel = np.vstack(
                [crop_label, cropped_resized, h_proj_label, h_plot_resized]
            )

            # Pad to match left panel height if needed
            height_diff = left_panel.shape[0] - right_panel.shape[0]
            if height_diff > 0:
                padding = np.zeros(
                    (height_diff, right_panel.shape[1], 3), dtype=np.uint8
                )
                right_panel = np.vstack([right_panel, padding])
            elif height_diff < 0:
                padding = np.zeros(
                    (-height_diff, left_panel.shape[1], 3), dtype=np.uint8
                )
                left_panel = np.vstack([left_panel, padding])
        else:
            # No cropped ROI - just show horizontal projection with reasonable width
            h_plot_width = min(400, new_width // 2)
            h_plot_height = min(250, horizontal_plot.shape[0])
            h_plot_resized = cv2.resize(horizontal_plot, (h_plot_width, h_plot_height))
            h_proj_label = create_label(
                "Horizontal Projection", h_plot_width, (255, 0, 255)
            )
            right_panel = np.vstack([h_proj_label, h_plot_resized])

            # Pad to match left panel height
            height_diff = left_panel.shape[0] - right_panel.shape[0]
            if height_diff > 0:
                padding = np.zeros(
                    (height_diff, right_panel.shape[1], 3), dtype=np.uint8
                )
                right_panel = np.vstack([right_panel, padding])

        # Combine horizontally
        comparison = np.hstack([left_panel, right_panel])

    elif analysis and roi_coords:
        # Fallback: show projections even if show_projections is False, but analysis is available
        vertical_info = analysis.get("vertical_info", {})
        horizontal_info = analysis.get("horizontal_info", {})

        if vertical_info and horizontal_info:
            # Recursive call with projections enabled
            return create_comparison_grid(
                original, roi_overlay, cropped_roi, analysis, roi_coords, True
            )

    # Fallback to original layout without projections
    if not (analysis and roi_coords):
        # Original layout without projections
        left_panel = np.vstack(
            [orig_label, orig_resized, overlay_label, overlay_resized]
        )

        # Add cropped ROI if provided
        if cropped_roi is not None and cropped_roi.size > 0:
            # Resize cropped ROI to fit
            crop_scale = target_height / cropped_roi.shape[0]
            crop_width = int(cropped_roi.shape[1] * crop_scale)
            cropped_resized = cv2.resize(cropped_roi, (crop_width, target_height))

            # Create label for cropped
            crop_label = create_label("Cropped ROI", crop_width, (0, 255, 255))

            # Combine with cropped
            right_panel = np.vstack([crop_label, cropped_resized])

            # Pad to match height if needed
            height_diff = left_panel.shape[0] - right_panel.shape[0]
            if height_diff > 0:
                padding = np.zeros((height_diff, crop_width, 3), dtype=np.uint8)
                right_panel = np.vstack([right_panel, padding])

            # Combine horizontally
            comparison = np.hstack([left_panel, right_panel])
        else:
            comparison = left_panel

    return comparison


def save_debug_images(image: np.ndarray, config, output_dir: Path, base_name: str):
    """Save debug images for edge detection steps based on configured method."""
    # Convert to grayscale for processing
    gray_img = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    )

    # Step 1: Save grayscale image
    cv2.imwrite(str(output_dir / f"{base_name}_01_grayscale.jpg"), gray_img)

    # Get the current method
    method = getattr(config, "roi_detection_method", "gabor")

    if method == "gabor":
        return save_gabor_debug_images(gray_img, config, output_dir, base_name)
    elif method == "canny_sobel":
        return save_canny_sobel_debug_images(gray_img, config, output_dir, base_name)
    elif method == "adaptive_threshold":
        return save_adaptive_debug_images(gray_img, config, output_dir, base_name)
    else:
        return save_gabor_debug_images(gray_img, config, output_dir, base_name)


def save_gabor_debug_images(
    gray_img: np.ndarray, config, output_dir: Path, base_name: str
):
    """Save debug images for Gabor filter method."""
    # Step 2: Create and save Gabor kernels visualization
    kernels = []
    for theta in [0, np.pi / 2]:  # 0° (vertical), 90° (horizontal)
        kernel = cv2.getGaborKernel(
            (config.gabor_kernel_size, config.gabor_kernel_size),
            config.gabor_sigma,
            float(theta),
            config.gabor_lambda,
            config.gabor_gamma,
            0,
            ktype=cv2.CV_32F,
        )
        kernels.append(kernel)

    # Visualize kernels
    kernel_vis = np.hstack(
        [
            cv2.normalize(k, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            for k in kernels
        ]
    )
    cv2.imwrite(str(output_dir / f"{base_name}_02_gabor_kernels.jpg"), kernel_vis)

    # Step 3: Apply individual Gabor filters and save responses
    combined_response = np.zeros_like(gray_img, dtype=np.float32)
    for i, kernel in enumerate(kernels):
        filtered_img = cv2.filter2D(gray_img, cv2.CV_8UC3, kernel)
        combined_response += filtered_img.astype(np.float32)

        # Save individual filter response
        filter_normalized = cv2.normalize(
            filtered_img.astype(np.float32),
            None,
            0,
            255,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        orientation = "vertical" if i == 0 else "horizontal"
        cv2.imwrite(
            str(output_dir / f"{base_name}_03_gabor_{orientation}.jpg"),
            filter_normalized,
        )

    # Step 4: Save combined Gabor response
    gabor_response_map = cv2.normalize(
        combined_response, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    cv2.imwrite(
        str(output_dir / f"{base_name}_04_gabor_combined.jpg"), gabor_response_map
    )

    # Step 5: Save binarized result
    _, binary_mask = cv2.threshold(
        gabor_response_map, config.gabor_binary_threshold, 255, cv2.THRESH_BINARY
    )
    cv2.imwrite(str(output_dir / f"{base_name}_05_gabor_binary.jpg"), binary_mask)

    # Step 6: Save threshold comparison
    threshold_comparison = np.hstack([gabor_response_map, binary_mask])
    cv2.imwrite(
        str(output_dir / f"{base_name}_06_threshold_comparison.jpg"),
        threshold_comparison,
    )

    return {
        "grayscale": str(output_dir / f"{base_name}_01_grayscale.jpg"),
        "gabor_kernels": str(output_dir / f"{base_name}_02_gabor_kernels.jpg"),
        "gabor_vertical": str(output_dir / f"{base_name}_03_gabor_vertical.jpg"),
        "gabor_horizontal": str(output_dir / f"{base_name}_03_gabor_horizontal.jpg"),
        "gabor_combined": str(output_dir / f"{base_name}_04_gabor_combined.jpg"),
        "gabor_binary": str(output_dir / f"{base_name}_05_gabor_binary.jpg"),
        "threshold_comparison": str(
            output_dir / f"{base_name}_06_threshold_comparison.jpg"
        ),
    }


def save_canny_sobel_debug_images(
    gray_img: np.ndarray, config, output_dir: Path, base_name: str
):
    """Save debug images for Canny + Sobel method."""
    # Step 2: Apply Gaussian blur
    blurred = cv2.GaussianBlur(
        gray_img, (config.gaussian_blur_size, config.gaussian_blur_size), 0
    )
    cv2.imwrite(str(output_dir / f"{base_name}_02_blurred.jpg"), blurred)

    # Step 3: Canny edge detection
    canny_edges = cv2.Canny(
        blurred, config.canny_low_threshold, config.canny_high_threshold
    )
    cv2.imwrite(str(output_dir / f"{base_name}_03_canny_edges.jpg"), canny_edges)

    # Step 4: Sobel edge detection
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=config.sobel_kernel_size)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=config.sobel_kernel_size)

    # Save individual Sobel responses
    sobel_x_norm = cv2.normalize(
        np.abs(sobel_x), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    sobel_y_norm = cv2.normalize(
        np.abs(sobel_y), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    cv2.imwrite(str(output_dir / f"{base_name}_04_sobel_x.jpg"), sobel_x_norm)
    cv2.imwrite(str(output_dir / f"{base_name}_04_sobel_y.jpg"), sobel_y_norm)

    # Step 5: Combined Sobel magnitude
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_magnitude = cv2.normalize(
        sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    cv2.imwrite(
        str(output_dir / f"{base_name}_05_sobel_magnitude.jpg"), sobel_magnitude
    )

    # Step 6: Laplacian edge detection
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian_norm = cv2.normalize(
        np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    cv2.imwrite(str(output_dir / f"{base_name}_06_laplacian.jpg"), laplacian_norm)

    # Step 7: Combined edge response
    combined_edges = (
        canny_edges.astype(np.float32) * 0.5
        + sobel_magnitude.astype(np.float32) * 0.35
        + laplacian_norm.astype(np.float32) * 0.15
    )
    combined_edges = cv2.normalize(
        combined_edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    cv2.imwrite(str(output_dir / f"{base_name}_07_combined_edges.jpg"), combined_edges)

    # Step 8: Apply morphological operations
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (config.morphology_kernel_size, config.morphology_kernel_size)
    )
    morph_close = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
    morph_final = cv2.morphologyEx(morph_close, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(str(output_dir / f"{base_name}_08_morphology.jpg"), morph_final)

    # Step 9: Final binary threshold
    _, binary_mask = cv2.threshold(
        morph_final, config.edge_binary_threshold, 255, cv2.THRESH_BINARY
    )
    cv2.imwrite(str(output_dir / f"{base_name}_09_final_binary.jpg"), binary_mask)

    return {
        "grayscale": str(output_dir / f"{base_name}_01_grayscale.jpg"),
        "blurred": str(output_dir / f"{base_name}_02_blurred.jpg"),
        "canny_edges": str(output_dir / f"{base_name}_03_canny_edges.jpg"),
        "sobel_x": str(output_dir / f"{base_name}_04_sobel_x.jpg"),
        "sobel_y": str(output_dir / f"{base_name}_04_sobel_y.jpg"),
        "sobel_magnitude": str(output_dir / f"{base_name}_05_sobel_magnitude.jpg"),
        "laplacian": str(output_dir / f"{base_name}_06_laplacian.jpg"),
        "combined_edges": str(output_dir / f"{base_name}_07_combined_edges.jpg"),
        "morphology": str(output_dir / f"{base_name}_08_morphology.jpg"),
        "final_binary": str(output_dir / f"{base_name}_09_final_binary.jpg"),
    }


def save_adaptive_debug_images(
    gray_img: np.ndarray, config, output_dir: Path, base_name: str
):
    """Save debug images for Adaptive threshold method."""
    # Step 2: Apply adaptive thresholding
    adaptive_method = (
        cv2.ADAPTIVE_THRESH_MEAN_C
        if config.adaptive_method == "mean"
        else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    )
    adaptive_thresh = cv2.adaptiveThreshold(
        gray_img,
        255,
        adaptive_method,
        cv2.THRESH_BINARY,
        config.adaptive_block_size,
        config.adaptive_C,
    )
    cv2.imwrite(
        str(output_dir / f"{base_name}_02_adaptive_thresh.jpg"), adaptive_thresh
    )

    if config.edge_enhancement:
        # Step 3: Gradient calculation for edge enhancement
        grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)

        # Save individual gradients
        grad_x_norm = cv2.normalize(
            np.abs(grad_x), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        grad_y_norm = cv2.normalize(
            np.abs(grad_y), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        cv2.imwrite(str(output_dir / f"{base_name}_03_gradient_x.jpg"), grad_x_norm)
        cv2.imwrite(str(output_dir / f"{base_name}_03_gradient_y.jpg"), grad_y_norm)

        # Step 4: Gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_magnitude = cv2.normalize(
            gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        cv2.imwrite(
            str(output_dir / f"{base_name}_04_gradient_magnitude.jpg"),
            gradient_magnitude,
        )

        # Step 5: Threshold gradient
        _, gradient_mask = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)
        cv2.imwrite(
            str(output_dir / f"{base_name}_05_gradient_mask.jpg"), gradient_mask
        )

        # Step 6: Combine adaptive threshold with gradient edges
        combined_mask = cv2.bitwise_or(adaptive_thresh, gradient_mask)
        cv2.imwrite(
            str(output_dir / f"{base_name}_06_combined_mask.jpg"), combined_mask
        )
    else:
        combined_mask = adaptive_thresh

    # Step 7: Apply morphological operations
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (config.morphology_kernel_size, config.morphology_kernel_size)
    )
    morph_close = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    morph_final = cv2.morphologyEx(morph_close, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(str(output_dir / f"{base_name}_07_morphology.jpg"), morph_final)

    debug_files = {
        "grayscale": str(output_dir / f"{base_name}_01_grayscale.jpg"),
        "adaptive_thresh": str(output_dir / f"{base_name}_02_adaptive_thresh.jpg"),
        "morphology": str(output_dir / f"{base_name}_07_morphology.jpg"),
    }

    if config.edge_enhancement:
        debug_files.update(
            {
                "gradient_x": str(output_dir / f"{base_name}_03_gradient_x.jpg"),
                "gradient_y": str(output_dir / f"{base_name}_03_gradient_y.jpg"),
                "gradient_magnitude": str(
                    output_dir / f"{base_name}_04_gradient_magnitude.jpg"
                ),
                "gradient_mask": str(output_dir / f"{base_name}_05_gradient_mask.jpg"),
                "combined_mask": str(output_dir / f"{base_name}_06_combined_mask.jpg"),
            }
        )

    return debug_files


def create_vertical_projection_plot(
    vertical_info: Dict, width: int, roi_coords: Dict
) -> np.ndarray:
    """Create a plot showing the vertical projection analysis for ROI detection."""
    projection = vertical_info["projection"]
    plot_height = 250
    plot_width = len(projection)

    # Normalize values for plotting
    max_val = np.max(projection) if len(projection) > 0 else 1
    min_val = np.min(projection) if len(projection) > 0 else 0
    if max_val > min_val:
        normalized = (
            (projection - min_val) / (max_val - min_val) * (plot_height - 60)
        ).astype(int)
    else:
        normalized = np.full_like(projection, plot_height // 2, dtype=int)

    # Create plot image
    plot_img = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255

    # Draw search regions background
    left_search_end = int(width * 0.7)
    right_search_start = int(width * 0.3)

    # Left search region (light blue)
    cv2.rectangle(plot_img, (0, 0), (left_search_end, plot_height), (255, 255, 200), -1)
    # Right search region (light green)
    cv2.rectangle(
        plot_img,
        (right_search_start, 0),
        (plot_width, plot_height),
        (200, 255, 200),
        -1,
    )

    # Draw the projection curve
    for i in range(len(normalized) - 1):
        y1 = plot_height - 30 - normalized[i]
        y2 = plot_height - 30 - normalized[i + 1]
        cv2.line(plot_img, (i, y1), (i + 1, y2), (0, 0, 255), 2)

    # Mark ROI boundaries
    roi_left = roi_coords["roi_left"]
    roi_right = roi_coords["roi_right"]

    # Left cut line (green if applied, red if not)
    left_applied = vertical_info.get("left_cut_applied", False)
    left_color = (0, 255, 0) if left_applied else (0, 0, 255)
    if roi_left > 0:
        cv2.line(plot_img, (roi_left, 0), (roi_left, plot_height), left_color, 3)
        cv2.putText(
            plot_img,
            f"L:{roi_left}",
            (roi_left + 5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            left_color,
            1,
        )

    # Right cut line (green if applied, red if not)
    right_applied = vertical_info.get("right_cut_applied", False)
    right_color = (0, 255, 0) if right_applied else (0, 0, 255)
    if roi_right < width:
        cv2.line(plot_img, (roi_right, 0), (roi_right, plot_height), right_color, 3)
        cv2.putText(
            plot_img,
            f"R:{roi_right}",
            (roi_right - 40, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            right_color,
            1,
        )

    # Add labels and statistics
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(plot_img, "Vertical Projection", (10, 15), font, 0.6, (0, 0, 0), 1)
    cv2.putText(
        plot_img, f"Min: {min_val:.0f}", (10, plot_height - 5), font, 0.4, (0, 0, 0), 1
    )
    cv2.putText(
        plot_img,
        f"Max: {max_val:.0f}",
        (plot_width - 80, plot_height - 5),
        font,
        0.4,
        (0, 0, 0),
        1,
    )

    # Add cut strength info if available
    if "left_cut_strength" in vertical_info:
        left_strength = vertical_info["left_cut_strength"]
        cv2.putText(
            plot_img,
            f"L_str: {left_strength:.1f}",
            (10, plot_height - 20),
            font,
            0.4,
            (0, 0, 0),
            1,
        )
    if "right_cut_strength" in vertical_info:
        right_strength = vertical_info["right_cut_strength"]
        cv2.putText(
            plot_img,
            f"R_str: {right_strength:.1f}",
            (plot_width - 120, plot_height - 20),
            font,
            0.4,
            (0, 0, 0),
            1,
        )

    return plot_img


def create_horizontal_projection_plot(
    horizontal_info: Dict, height: int, roi_coords: Dict
) -> np.ndarray:
    """Create a plot showing the horizontal projection analysis for ROI detection."""
    projection = horizontal_info["projection"]
    plot_width = 300  # Increased width to accommodate labels
    plot_height = len(projection)

    # Normalize values for plotting
    max_val = np.max(projection) if len(projection) > 0 else 1
    min_val = np.min(projection) if len(projection) > 0 else 0
    if max_val > min_val:
        normalized = (
            (projection - min_val) / (max_val - min_val) * (plot_width - 60)
        ).astype(int)
    else:
        normalized = np.full_like(projection, plot_width // 2, dtype=int)

    # Create plot image
    plot_img = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255

    # Draw search regions background
    top_search_end = int(height * 0.6)
    bottom_search_start = int(height * 0.4)

    # Top search region (light blue)
    cv2.rectangle(plot_img, (0, 0), (plot_width, top_search_end), (255, 255, 200), -1)
    # Bottom search region (light green)
    cv2.rectangle(
        plot_img,
        (0, bottom_search_start),
        (plot_width, plot_height),
        (200, 255, 200),
        -1,
    )

    # Draw the projection curve (horizontal, so we plot from left to right)
    for i in range(len(normalized) - 1):
        x1 = 30 + normalized[i]
        x2 = 30 + normalized[i + 1]
        cv2.line(plot_img, (x1, i), (x2, i + 1), (0, 0, 255), 2)

    # Mark ROI boundaries
    roi_top = roi_coords["roi_top"]
    roi_bottom = roi_coords["roi_bottom"]

    # Top cut line (green if applied, red if not)
    top_applied = horizontal_info.get("top_cut_applied", False)
    top_color = (0, 255, 0) if top_applied else (0, 0, 255)
    if roi_top > 0:
        cv2.line(plot_img, (0, roi_top), (plot_width, roi_top), top_color, 3)
        cv2.putText(
            plot_img,
            f"T:{roi_top}",
            (5, roi_top - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            top_color,
            1,
        )

    # Bottom cut line (green if applied, red if not)
    bottom_applied = horizontal_info.get("bottom_cut_applied", False)
    bottom_color = (0, 255, 0) if bottom_applied else (0, 0, 255)
    if roi_bottom < height:
        cv2.line(plot_img, (0, roi_bottom), (plot_width, roi_bottom), bottom_color, 3)
        cv2.putText(
            plot_img,
            f"B:{roi_bottom}",
            (5, roi_bottom + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            bottom_color,
            1,
        )

    # Add labels and statistics
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Rotate text for vertical plot
    cv2.putText(plot_img, "Horizontal", (5, 20), font, 0.5, (0, 0, 0), 1)
    cv2.putText(plot_img, "Projection", (5, 35), font, 0.5, (0, 0, 0), 1)
    cv2.putText(
        plot_img, f"Min: {min_val:.0f}", (5, plot_height - 25), font, 0.4, (0, 0, 0), 1
    )
    cv2.putText(
        plot_img, f"Max: {max_val:.0f}", (5, plot_height - 10), font, 0.4, (0, 0, 0), 1
    )

    # Add cut strength info if available
    if "top_cut_strength" in horizontal_info:
        top_strength = horizontal_info["top_cut_strength"]
        cv2.putText(
            plot_img, f"T_str: {top_strength:.1f}", (120, 20), font, 0.4, (0, 0, 0), 1
        )
    if "bottom_cut_strength" in horizontal_info:
        bottom_strength = horizontal_info["bottom_cut_strength"]
        cv2.putText(
            plot_img,
            f"B_str: {bottom_strength:.1f}",
            (120, plot_height - 10),
            font,
            0.4,
            (0, 0, 0),
            1,
        )

    return plot_img


def process_image_visualization(
    image_path: Path,
    config,
    output_dir: Path,
    show_gabor: bool = False,
    save_debug: bool = False,
    show_projections: bool = True,
    command_args: Dict[str, Any] = None,
    config_source: str = "default",
) -> Dict[str, Any]:
    """Process a single image and create visualization."""
    print(f"Processing: {image_path.name}")

    try:
        # Load image
        image = ocr_utils.load_image(image_path)

        # Detect ROI with analysis data
        roi_coords, analysis = ocr_utils.detect_roi_for_image(
            image, config, return_analysis=True
        )

        # Create ROI overlay with analysis data
        roi_overlay = draw_roi_overlay(image, roi_coords, show_gabor, config, analysis)

        # Crop to ROI
        left = roi_coords["roi_left"]
        right = roi_coords["roi_right"]
        top = roi_coords["roi_top"]
        bottom = roi_coords["roi_bottom"]
        cropped_roi = image[top:bottom, left:right]

        # Create comparison grid with projection graphs if requested
        comparison = create_comparison_grid(
            image, roi_overlay, cropped_roi, analysis, roi_coords, show_projections
        )

        # Always save individual projection plots if analysis data is available
        projection_files = {}
        vertical_info = analysis.get("vertical_info", {})
        horizontal_info = analysis.get("horizontal_info", {})

        if vertical_info:
            vertical_plot = create_vertical_projection_plot(
                vertical_info, image.shape[1], roi_coords
            )
            v_plot_path = output_dir / f"{image_path.stem}_vertical_projection.jpg"
            cv2.imwrite(str(v_plot_path), vertical_plot)
            projection_files["vertical_projection"] = str(v_plot_path)

        if horizontal_info:
            horizontal_plot = create_horizontal_projection_plot(
                horizontal_info, image.shape[0], roi_coords
            )
            h_plot_path = output_dir / f"{image_path.stem}_horizontal_projection.jpg"
            cv2.imwrite(str(h_plot_path), horizontal_plot)
            projection_files["horizontal_projection"] = str(h_plot_path)

        # Save outputs
        base_name = image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save individual images
        cv2.imwrite(str(output_dir / f"{base_name}_original.jpg"), image)
        cv2.imwrite(str(output_dir / f"{base_name}_roi_overlay.jpg"), roi_overlay)
        cv2.imwrite(str(output_dir / f"{base_name}_roi_cropped.jpg"), cropped_roi)
        cv2.imwrite(str(output_dir / f"{base_name}_comparison.jpg"), comparison)

        # Save debug images if requested
        debug_files = {}
        if save_debug:
            debug_files = save_debug_images(image, config, output_dir, base_name)

        # Save ROI coordinates
        roi_file = output_dir / f"{base_name}_roi_coords.json"
        with open(roi_file, "w") as f:
            json.dump(convert_numpy_types(roi_coords), f, indent=2)

        # Calculate metrics
        original_area = roi_coords["image_width"] * roi_coords["image_height"]
        roi_area = (right - left) * (bottom - top)
        coverage = roi_area / original_area if original_area > 0 else 0

        output_files = {
            "original": str(output_dir / f"{base_name}_original.jpg"),
            "overlay": str(output_dir / f"{base_name}_roi_overlay.jpg"),
            "cropped": str(output_dir / f"{base_name}_roi_cropped.jpg"),
            "comparison": str(output_dir / f"{base_name}_comparison.jpg"),
            "coords": str(roi_file),
        }

        # Add debug files if they were saved
        if debug_files:
            output_files.update(debug_files)

        # Add projection files if they were saved
        if projection_files:
            output_files.update(projection_files)

        # Prepare processing results for parameter documentation
        processing_results = {
            "image_name": image_path.name,
            "success": True,
            "roi_info": roi_coords,
            "coverage": coverage,
            "output_files": output_files,
            "analysis": analysis,
        }

        # Save parameter documentation
        if command_args is None:
            command_args = {}

        param_file = save_step_parameters(
            step_name="roi_detection",
            config_obj=config,
            command_args=command_args,
            processing_results=processing_results,
            output_dir=output_dir,
            config_source=config_source,
        )

        # Include parameter file in output files
        output_files["parameters"] = str(param_file)

        result = {
            "image_name": image_path.name,
            "success": True,
            "roi_coords": roi_coords,
            "coverage": coverage,
            "output_files": output_files,
            "parameter_file": str(param_file) if param_file else None,
        }

        print(f"  SUCCESS: ROI Coverage: {coverage*100:.1f}%")
        print(f"  SUCCESS: ROI Size: {right-left} x {bottom-top}")
        print(f"  SUCCESS: Saved to: {output_dir}")

        return result

    except Exception as e:
        print(f"  ERROR: Error: {e}")
        return {"image_name": image_path.name, "success": False, "error": str(e)}


def main():
    """Main function for ROI visualization."""
    parser = argparse.ArgumentParser(description="Visualize ROI detection results")
    parser.add_argument(
        "images",
        nargs="*",
        default=["input/raw_images/Wang2017_Page_001.jpg"],
        help="Images to visualize",
    )
    parser.add_argument(
        "--test-images",
        action="store_true",
        help="Process all images in input/test_images directory",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Custom input directory (overrides --test-images)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/output/visualization/roi",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--show-gabor", action="store_true", help="Show edge detection response overlay"
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save debug images showing binarization and edge detection steps",
    )
    parser.add_argument(
        "--show-projections",
        action="store_true",
        default=True,
        help="Show projection graphs and cutting lines analysis (enabled by default)",
    )
    parser.add_argument(
        "--hide-projections",
        action="store_true",
        help="Hide projection graphs and cutting lines analysis",
    )
    parser.add_argument(
        "--config-params", type=str, help="JSON string of config parameters to test"
    )

    # Config file options
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="JSON config file to use (default: configs/stage1_default.json)",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=1,
        help="Use stage1 or stage2 default config (default: 1)",
    )

    # ROI detection method selection
    parser.add_argument(
        "--roi-method",
        type=str,
        choices=["gabor", "canny_sobel", "adaptive_threshold"],
        default=None,
        help="ROI detection method to use",
    )

    # Parameter overrides (take precedence over config file)
    parser.add_argument(
        "--gabor-threshold", type=int, default=None, help="Gabor binary threshold"
    )
    parser.add_argument(
        "--cut-strength", type=float, default=None, help="Minimum cut strength"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Minimum confidence threshold",
    )

    # New method parameters
    parser.add_argument(
        "--canny-low", type=int, default=None, help="Canny low threshold"
    )
    parser.add_argument(
        "--canny-high", type=int, default=None, help="Canny high threshold"
    )
    parser.add_argument(
        "--adaptive-block-size",
        type=int,
        default=None,
        help="Adaptive threshold block size",
    )

    args = parser.parse_args()

    # Handle projection display logic
    if args.hide_projections:
        args.show_projections = False

    # Determine which images to process
    if args.input_dir:
        print(f"Using custom input directory: {args.input_dir}")
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Input directory {args.input_dir} does not exist!")
            return
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            image_paths.extend(input_dir.glob(ext))
        if not image_paths:
            print(f"No images found in {args.input_dir}!")
            return
    elif args.test_images:
        print("Using batch mode: processing all images in input/test_images/")
        image_paths = get_test_images()
        if not image_paths:
            print("No images found in test_images directory!")
            return
    else:
        # Resolve image paths from command line arguments
        image_paths = []
        for img_path in args.images:
            path = Path(img_path)
            if path.exists():
                image_paths.append(path)
            else:
                print(f"Warning: {img_path} not found, skipping")

        if not image_paths:
            print("No valid images found!")
            return

    # Load configuration from JSON file
    config = load_config_from_file(args.config_file, args.stage)

    # Handle config-params JSON override if provided
    config_params = {}
    if args.config_params:
        try:
            config_params = json.loads(args.config_params)
            for key, value in config_params.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        except json.JSONDecodeError:
            print("Warning: Invalid JSON in config-params, using command line args")

    # Apply ROI method override
    if args.roi_method is not None:
        config.roi_detection_method = args.roi_method

    # Collect command line arguments for parameter documentation
    command_args = {
        "roi_method": args.roi_method,
        "gabor_threshold": args.gabor_threshold,
        "cut_strength": args.cut_strength,
        "confidence_threshold": args.confidence_threshold,
        "canny_low": args.canny_low,
        "canny_high": args.canny_high,
        "adaptive_block_size": args.adaptive_block_size,
        "config_params": args.config_params,
        "config_file": str(args.config_file) if args.config_file else None,
        "stage": args.stage,
        "show_gabor": args.show_gabor,
        "save_debug": args.save_debug,
        "show_projections": args.show_projections,
    }

    # Determine config source
    config_source = "default"
    if args.config_file and args.config_file.exists():
        config_source = "file"
    if args.config_params or any(
        v is not None
        for v in [
            args.roi_method,
            args.gabor_threshold,
            args.cut_strength,
            args.confidence_threshold,
            args.canny_low,
            args.canny_high,
            args.adaptive_block_size,
        ]
    ):
        config_source += "_with_overrides"

    # Apply command line parameter overrides
    if args.gabor_threshold is not None:
        config.gabor_binary_threshold = args.gabor_threshold
    if args.cut_strength is not None:
        config.roi_min_cut_strength = args.cut_strength
    if args.confidence_threshold is not None:
        config.roi_min_confidence_threshold = args.confidence_threshold

    # New method parameter overrides
    if args.canny_low is not None:
        config.canny_low_threshold = args.canny_low
    if args.canny_high is not None:
        config.canny_high_threshold = args.canny_high
    if args.adaptive_block_size is not None:
        config.adaptive_block_size = args.adaptive_block_size

    config.verbose = False  # Keep visualization quiet
    config.enable_roi_detection = True  # Ensure ROI detection is enabled
    output_dir = Path(args.output_dir)

    print(f"Visualizing ROI detection on {len(image_paths)} images")
    if args.test_images:
        print("Batch mode: Processing all images from test_images directory")

    # Get the current ROI detection method
    current_method = getattr(config, "roi_detection_method", "gabor")
    print(f"ROI Detection Method: {current_method}")
    print("Parameters:")

    if current_method == "gabor":
        print(f"  - Gabor threshold: {config.gabor_binary_threshold}")
        print(f"  - Gabor kernel size: {config.gabor_kernel_size}")
        print(f"  - Gabor sigma: {config.gabor_sigma}")
    elif current_method == "canny_sobel":
        print(f"  - Canny low: {config.canny_low_threshold}")
        print(f"  - Canny high: {config.canny_high_threshold}")
        print(f"  - Sobel kernel: {config.sobel_kernel_size}")
        print(f"  - Edge threshold: {config.edge_binary_threshold}")
    elif current_method == "adaptive_threshold":
        print(f"  - Adaptive method: {config.adaptive_method}")
        print(f"  - Block size: {config.adaptive_block_size}")
        print(f"  - C value: {config.adaptive_C}")
        print(f"  - Edge enhancement: {config.edge_enhancement}")

    print(f"  - Cut strength: {config.roi_min_cut_strength}")
    print(f"  - Confidence threshold: {config.roi_min_confidence_threshold}")
    print(f"  - Show edge response: {args.show_gabor}")
    print(f"  - Save debug images: {args.save_debug}")
    print(f"  - Show projections: {args.show_projections}")
    print(f"Output directory: {output_dir}")
    print()

    # Process all images
    results = []
    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Processing: {image_path.name}")
        result = process_image_visualization(
            image_path,
            config,
            output_dir,
            args.show_gabor,
            args.save_debug,
            args.show_projections,
            command_args,
            config_source,
        )
        results.append(result)

    # Summary
    successful_results = [r for r in results if r["success"]]
    print(f"\n{'='*60}")
    print("VISUALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {len(successful_results)}/{len(image_paths)} images")

    if successful_results:
        avg_coverage = sum(r["coverage"] for r in successful_results) / len(
            successful_results
        )
        print(f"Average ROI coverage: {avg_coverage*100:.1f}%")

        coverage_range = [r["coverage"] for r in successful_results]
        print(
            f"Coverage range: {min(coverage_range)*100:.1f}% - {max(coverage_range)*100:.1f}%"
        )

    print(f"\nOutput files saved to: {output_dir}")
    print("Review the '_comparison.jpg' files to assess ROI quality")

    # Save summary
    summary_file = output_dir / "visualization_summary.json"
    summary_data = {
        "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "config_parameters": config_params,
        "results": results,
    }

    with open(summary_file, "w") as f:
        json.dump(convert_numpy_types(summary_data), f, indent=2)

    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
