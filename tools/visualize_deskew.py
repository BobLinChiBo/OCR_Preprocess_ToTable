#!/usr/bin/env python3
"""
Deskew Visualization Script

This script visualizes the deskewing process, helping you assess angle detection
quality and adjust deskewing parameters.
"""

import warnings
warnings.warn(
    "This script is deprecated and will be removed in a future version. "
    "Please use visualize_deskew_v2.py or run_visualizations.py with --use-v2 flag.",
    DeprecationWarning,
    stacklevel=2
)

import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import sys
import time
from typing import Dict, Any, List

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.ocr_pipeline.config import Config, Stage1Config, Stage2Config  # noqa: E402
import src.ocr_pipeline.utils as ocr_utils  # noqa: E402
from tools.output_manager import (  # noqa: E402
    get_default_output_manager,
    organize_visualization_output,
    get_test_images,
    convert_numpy_types,
    save_step_parameters,
)


def create_lightweight_summary(skew_info):
    """Create a lightweight summary of skew info, excluding heavy image data."""
    return {
        "rotation_angle": skew_info["rotation_angle"],
        "confidence": skew_info["confidence"],
        "will_rotate": skew_info["will_rotate"],
        "method": skew_info["method"],
        "line_count": skew_info.get("line_count", 0),
        "best_score": skew_info.get("best_score", 0),
        "angle_std": skew_info.get("angle_std", 0),
        "has_lines": skew_info.get("has_lines", True),
        # Exclude heavy data: gray, binary, deskewed_image, angles, scores, angle_histogram
    }


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
        print(f"Warning: Config file {config_path} not found, using defaults")
        if stage == 2:
            return Stage2Config()
        else:
            return Stage1Config()


def analyze_skew_detailed(
    image,
    angle_range=10,
    angle_step=0.2,
    min_angle_correction=0.5,
    save_intermediates=False,
    output_base=None,
):
    """Enhanced skew analysis using the optimized utils.deskew_image function.

    This function uses the optimized deskew_image() function from utils.py to get both
    the result and analysis data in a single pass, eliminating redundant computation.
    """
    # Use the optimized deskewing function with analysis data
    # For visualization purposes, use very aggressive optimization parameters
    viz_angle_range = min(angle_range, 3)  # Limit to ±3° for visualization
    viz_angle_step = max(angle_step, 1.0)  # Use at least 1.0° steps for speed
    
    print(f"    Calling deskew_image with visualization params: range=±{viz_angle_range}°, step={viz_angle_step}°")
    try:
        deskewed_image, detected_angle, analysis_data = ocr_utils.deskew_image(
            image, viz_angle_range, viz_angle_step, min_angle_correction, return_analysis_data=True
        )
        print(f"    deskew_image returned angle: {detected_angle}")
    except Exception as e:
        print(f"    ERROR in deskew_image: {e}")
        raise

    # Step 1: Save grayscale and binary conversion if requested
    if save_intermediates and output_base:
        grayscale_file = f"{output_base}_01_grayscale.jpg"
        binary_file = f"{output_base}_02_binary.jpg"
        print(f"  DEBUG: Saving grayscale to {grayscale_file}")
        print(f"  DEBUG: Saving binary to {binary_file}")
        cv2.imwrite(grayscale_file, analysis_data["gray"])
        cv2.imwrite(binary_file, analysis_data["binary"])

        # Step 2: Save histogram analysis visualizations
        # Use optimized projection calculation - avoid full rotation for performance
        # For visualization purposes, we can use a smaller preview or skip this step
        if abs(detected_angle) > 0.1:  # Only create projection for meaningful rotations
            # Use a smaller version for performance
            h, w = analysis_data["binary"].shape
            small_binary = cv2.resize(analysis_data["binary"], (w//2, h//2), interpolation=cv2.INTER_AREA)
            best_rotated_small = cv2.warpAffine(
                small_binary,
                cv2.getRotationMatrix2D((w//4, h//4), detected_angle, 1.0),
                (w//2, h//2),
                flags=cv2.INTER_LINEAR,  # Use faster interpolation
                borderMode=cv2.BORDER_REPLICATE,
            )
            best_projection = np.sum(best_rotated_small, axis=1)
            
            # Create projection visualization with smaller image
            proj_vis = create_projection_visualization(
                best_rotated_small, best_projection, detected_angle
            )
            cv2.imwrite(f"{output_base}_03_best_projection.jpg", proj_vis)
        else:
            # Skip projection visualization for minimal rotations
            print(f"  Skipping projection visualization for minimal rotation ({detected_angle:.2f}°)")

        # Create angle score plot with data sampling for performance
        print(f"  DEBUG: Creating angle score plot...")
        # Limit data points for performance - sample every N points if too many
        angles_to_plot = analysis_data["angles"]
        scores_to_plot = analysis_data["scores"]
        if len(angles_to_plot) > 50:  # Sample data if too many points
            step = len(angles_to_plot) // 50
            angles_to_plot = angles_to_plot[::step]
            scores_to_plot = scores_to_plot[::step]
        
        score_plot = create_angle_score_plot(angles_to_plot, scores_to_plot, detected_angle)
        print(f"  DEBUG: Angle score plot created, saving...")
        cv2.imwrite(f"{output_base}_04_angle_scores.jpg", score_plot)
        print(f"  DEBUG: Angle score plot saved")

    # Create angle_histogram for compatibility with existing visualization code
    angle_histogram = {}
    for i, angle in enumerate(analysis_data["angles"]):
        if i < len(analysis_data["scores"]):
            angle_histogram[round(angle, 1)] = analysis_data["scores"][i]

    # Return enhanced analysis data from the optimized function including deskewed image
    return {
        "has_lines": analysis_data["has_lines"],
        "rotation_angle": analysis_data["rotation_angle"],
        "line_count": analysis_data["line_count"],
        "angles": analysis_data["angles"],
        "scores": analysis_data["scores"],
        "confidence": analysis_data["confidence"],
        "angle_std": analysis_data.get("angle_std", 0),
        "angle_histogram": angle_histogram,
        "best_score": analysis_data["best_score"],
        "gray": analysis_data["gray"],
        "binary": analysis_data["binary"],
        "will_rotate": analysis_data["will_rotate"],
        "method": analysis_data["method"],
        "deskewed_image": deskewed_image,  # Include the deskewed image to avoid recomputation
    }


def create_projection_visualization(
    binary_rotated: np.ndarray, projection: np.ndarray, angle: float
) -> np.ndarray:
    """Create visualization showing horizontal projection histogram."""
    height, width = binary_rotated.shape

    # Create side-by-side visualization: rotated image + projection plot
    vis_width = width + 300  # Extra space for projection plot
    vis_img = np.zeros((height, vis_width, 3), dtype=np.uint8)

    # Left side: show rotated binary image as RGB
    binary_rgb = cv2.cvtColor(binary_rotated, cv2.COLOR_GRAY2BGR)
    vis_img[:, :width] = binary_rgb

    # Right side: draw projection histogram
    plot_start_x = width + 20
    plot_width = 250

    if len(projection) > 0:
        max_proj = max(projection)
        if max_proj > 0:
            # Normalize projection to plot width
            normalized_proj = (projection / max_proj) * plot_width

            # Draw horizontal bars for each row
            for i, val in enumerate(normalized_proj):
                if i < height:
                    bar_length = int(val)
                    if bar_length > 0:
                        cv2.line(
                            vis_img,
                            (plot_start_x, i),
                            (plot_start_x + bar_length, i),
                            (0, 255, 0),
                            1,
                        )

    # Add labels and info
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        vis_img,
        f"Angle: {angle:.2f}°",
        (plot_start_x, 30),
        font,
        0.6,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        vis_img,
        "Horizontal Projection",
        (plot_start_x, 50),
        font,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        vis_img, "(Row pixel sums)", (plot_start_x, 65), font, 0.4, (128, 128, 128), 1
    )

    return vis_img


def create_angle_score_plot(
    angles: np.ndarray, scores: List[float], best_angle: float
) -> np.ndarray:
    """Create plot showing histogram variance scores vs angle."""
    plot_width = 600
    plot_height = 400
    plot_img = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255

    if not scores or len(scores) == 0:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(plot_img, "No scores to plot", (200, 200), font, 1.0, (0, 0, 0), 2)
        return plot_img

    # Normalize scores for plotting
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score

    if score_range == 0:
        return plot_img

    # Plot parameters
    margin = 50
    plot_area_width = plot_width - 2 * margin
    plot_area_height = plot_height - 2 * margin

    # Draw axes
    cv2.line(
        plot_img,
        (margin, plot_height - margin),
        (plot_width - margin, plot_height - margin),
        (0, 0, 0),
        2,
    )  # x-axis
    cv2.line(
        plot_img, (margin, margin), (margin, plot_height - margin), (0, 0, 0), 2
    )  # y-axis

    # Plot score curve
    points = []
    for i, (angle, score) in enumerate(zip(angles, scores)):
        # Avoid division by zero when there's only one angle
        if len(angles) > 1:
            x = margin + int((i / (len(angles) - 1)) * plot_area_width)
        else:
            x = margin + plot_area_width // 2  # Center the single point
        y = (
            plot_height
            - margin
            - int(((score - min_score) / score_range) * plot_area_height)
        )
        points.append((x, y))

        # Highlight best angle
        if abs(angle - best_angle) < 0.01:
            cv2.circle(plot_img, (x, y), 6, (0, 255, 0), -1)
        else:
            cv2.circle(plot_img, (x, y), 2, (255, 0, 0), -1)

    # Draw connecting lines
    for i in range(len(points) - 1):
        cv2.line(plot_img, points[i], points[i + 1], (0, 0, 255), 1)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        plot_img, "Histogram Variance Scores", (margin, 30), font, 0.7, (0, 0, 0), 2
    )
    cv2.putText(
        plot_img,
        f"Best: {best_angle:.2f}° (score: {max_score:.0f})",
        (margin, plot_height - 10),
        font,
        0.5,
        (0, 128, 0),
        1,
    )

    # Angle labels on x-axis
    for i in range(0, len(angles), max(1, len(angles) // 5)):
        x = margin + int((i / (len(angles) - 1)) * plot_area_width)
        cv2.putText(
            plot_img,
            f"{angles[i]:.1f}°",
            (x - 15, plot_height - 25),
            font,
            0.4,
            (0, 0, 0),
            1,
        )

    return plot_img


def draw_line_detection_overlay(
    image: np.ndarray, skew_info: Dict[str, Any]
) -> np.ndarray:
    """Draw overlay showing histogram-based skew analysis."""
    overlay = image.copy()

    # For histogram-based analysis, we show a different kind of visualization
    # Draw a rotation indicator line showing the detected angle
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2

    # Draw reference lines showing the detected rotation
    rotation_angle = skew_info["rotation_angle"]

    # Calculate line endpoints for the detected angle
    line_length = min(width, height) // 3
    angle_rad = np.radians(rotation_angle)

    # Horizontal reference line (what text should be aligned to)
    ref_x1 = center_x - line_length
    ref_x2 = center_x + line_length
    ref_y = center_y

    # Detected skew line
    skew_x1 = center_x - int(line_length * np.cos(angle_rad))
    skew_y1 = center_y - int(line_length * np.sin(angle_rad))
    skew_x2 = center_x + int(line_length * np.cos(angle_rad))
    skew_y2 = center_y + int(line_length * np.sin(angle_rad))

    # Draw reference horizontal line (green)
    cv2.line(overlay, (ref_x1, ref_y), (ref_x2, ref_y), (0, 255, 0), 2)
    cv2.putText(
        overlay,
        "Target alignment",
        (ref_x1, ref_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )

    # Draw detected skew line (red if will rotate, blue if won't)
    line_color = (0, 0, 255) if skew_info["will_rotate"] else (255, 0, 0)
    cv2.line(overlay, (skew_x1, skew_y1), (skew_x2, skew_y2), line_color, 3)

    # Draw angle arc
    if abs(rotation_angle) > 0.1:
        cv2.ellipse(
            overlay,
            (center_x, center_y),
            (60, 60),
            0,
            0,
            -rotation_angle,
            (255, 255, 0),
            2,
        )

    # Add analysis text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    method_name = skew_info.get("method", "histogram_variance")
    text_lines = [
        f"Method: {method_name}",
        f"Rotation angle: {skew_info['rotation_angle']:.2f}°",
        f"Confidence: {skew_info['confidence']:.3f}",
        f"Best score: {skew_info.get('best_score', 0):.0f}",
        f"Will rotate: {skew_info['will_rotate']}",
    ]

    # Draw text background
    text_y_start = 30
    for i, line in enumerate(text_lines):
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        cv2.rectangle(
            overlay,
            (10, text_y_start + i * 35 - 25),
            (20 + text_size[0], text_y_start + i * 35 + 10),
            (0, 0, 0),
            -1,
        )

    # Draw text
    for i, line in enumerate(text_lines):
        if i == 4:  # "Will rotate" line
            color = (0, 255, 0) if skew_info["will_rotate"] else (0, 0, 255)
        else:
            color = (255, 255, 255)
        cv2.putText(
            overlay,
            line,
            (15, text_y_start + i * 35),
            font,
            font_scale,
            color,
            thickness,
        )

    return overlay


def create_angle_histogram_plot(skew_info: Dict[str, Any]) -> np.ndarray:
    """Create a plot showing the histogram variance scores for different angles."""
    if not skew_info.get("scores"):
        plot_img = np.ones((300, 600, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            plot_img, "No scores available", (200, 150), font, 1.0, (0, 0, 0), 2
        )
        return plot_img

    angles = skew_info["angles"]
    scores = skew_info["scores"]

    plot_width = 600
    plot_height = 300
    plot_img = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255

    if not scores or len(scores) == 0:
        return plot_img

    # Normalize scores for plotting
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score

    if score_range == 0:
        return plot_img

    # Draw bars for each angle's score
    bar_width = max(1, plot_width // len(scores))
    for i, (angle, score) in enumerate(zip(angles, scores)):
        bar_height = (
            int(((score - min_score) / score_range) * (plot_height - 80))
            if score_range > 0
            else 0
        )
        x = i * bar_width
        y = plot_height - 40 - bar_height

        # Color based on score value (higher scores are greener)
        score_intensity = (
            int(((score - min_score) / score_range) * 255) if score_range > 0 else 0
        )
        color = (0, score_intensity, 255 - score_intensity)

        cv2.rectangle(
            plot_img, (x, y), (x + bar_width - 1, plot_height - 40), color, -1
        )
        cv2.rectangle(
            plot_img, (x, y), (x + bar_width - 1, plot_height - 40), (0, 0, 0), 1
        )

    # Mark the selected angle (best score)
    rotation_angle = skew_info["rotation_angle"]
    # Find closest angle index instead of exact match to avoid floating-point precision issues
    best_angle_idx = -1
    if angles:
        angle_diffs = [abs(angle - rotation_angle) for angle in angles]
        min_diff_idx = angle_diffs.index(min(angle_diffs))
        # Only use it if the difference is small (within 0.1 degrees)
        if angle_diffs[min_diff_idx] < 0.1:
            best_angle_idx = min_diff_idx
    if best_angle_idx >= 0:
        x = best_angle_idx * bar_width + bar_width // 2
        cv2.line(plot_img, (x, 0), (x, plot_height - 40), (0, 255, 0), 3)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        plot_img, "Histogram Variance Scores", (10, 25), font, 0.7, (0, 0, 0), 2
    )
    cv2.putText(
        plot_img,
        f"Best: {rotation_angle:.2f}° (score: {max_score:.0f})",
        (10, plot_height - 10),
        font,
        0.6,
        (0, 255, 0),
        1,
    )
    cv2.putText(
        plot_img,
        f"Range: {min(angles):.1f}° to {max(angles):.1f}°",
        (300, plot_height - 10),
        font,
        0.6,
        (0, 0, 0),
        1,
    )

    return plot_img


def create_binary_visualization(binary: np.ndarray) -> np.ndarray:
    """Create a colored visualization of the binary threshold image."""
    # Convert binary to 3-channel for visualization
    binary_colored = cv2.applyColorMap(binary, cv2.COLORMAP_JET)
    return binary_colored


def create_deskew_comparison(
    original: np.ndarray,
    deskewed: np.ndarray,
    overlay: np.ndarray,
    angle_plot: np.ndarray,
    binary_vis: np.ndarray,
    skew_info: Dict[str, Any],
) -> np.ndarray:
    """Create a comprehensive comparison showing all deskew results."""
    # Use smaller target height to reduce memory usage and speed up processing
    target_height = 300  # Further reduced from 400 to 300 for better performance

    # Resize main images
    scale = target_height / original.shape[0]
    new_width = int(original.shape[1] * scale)

    orig_resized = cv2.resize(original, (new_width, target_height))
    deskewed_resized = cv2.resize(deskewed, (new_width, target_height))
    overlay_resized = cv2.resize(overlay, (new_width, target_height))
    binary_resized = cv2.resize(binary_vis, (new_width, target_height))

    # Resize plots to match width
    plot_scale = new_width / angle_plot.shape[1]
    plot_height = int(angle_plot.shape[0] * plot_scale)
    plot_resized = cv2.resize(angle_plot, (new_width, plot_height))

    # Create labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_height = 40

    def create_label(
        text: str, width: int, color: tuple = (255, 255, 255)
    ) -> np.ndarray:
        label_img = np.zeros((label_height, width, 3), dtype=np.uint8)
        cv2.putText(label_img, text, (10, 25), font, 0.8, color, 2)
        return label_img

    # Create top row (analysis)
    orig_label = create_label("Original", new_width)
    overlay_label = create_label("Skew Analysis", new_width)
    binary_label = create_label("Binary Threshold", new_width)

    top_row = np.vstack(
        [
            orig_label,
            orig_resized,
            overlay_label,
            overlay_resized,
            binary_label,
            binary_resized,
        ]
    )

    # Create middle row (result and analysis)
    result_color = (0, 255, 0) if skew_info["will_rotate"] else (255, 255, 255)
    deskewed_label = create_label(
        f"Deskewed ({skew_info['rotation_angle']:.2f}°)", new_width, result_color
    )
    plot_label = create_label("Angle Analysis", new_width)

    middle_row = np.vstack([deskewed_label, deskewed_resized, plot_label, plot_resized])

    # Combine rows
    max_width = max(top_row.shape[1], middle_row.shape[1])

    # Pad to same width if needed
    if top_row.shape[1] < max_width:
        padding = np.zeros(
            (top_row.shape[0], max_width - top_row.shape[1], 3), dtype=np.uint8
        )
        top_row = np.hstack([top_row, padding])
    if middle_row.shape[1] < max_width:
        padding = np.zeros(
            (middle_row.shape[0], max_width - middle_row.shape[1], 3), dtype=np.uint8
        )
        middle_row = np.hstack([middle_row, padding])

    comparison = np.vstack([top_row, middle_row])

    # Clear intermediate arrays to free memory
    del top_row, middle_row
    del orig_resized, deskewed_resized, overlay_resized, binary_resized, plot_resized

    return comparison


def process_image_deskew_visualization(
    image_path: Path,
    config,
    output_dir: Path,
    use_organized_output: bool = True,
    command_args: Dict[str, Any] = None,
    config_source: str = "default",
) -> Dict[str, Any]:
    """Process a single image and create deskew visualization."""
    print(f"  Processing: {image_path.name} (analyzing skew...)")

    try:
        # Load image
        print(f"  Loading image: {image_path.name}")
        image = ocr_utils.load_image(image_path)
        print(f"  Image loaded, shape: {image.shape}")
        
        # For visualization, downsample large images to improve performance
        max_dimension = 1200  # Maximum width or height for processing
        if max(image.shape[:2]) > max_dimension:
            scale = max_dimension / max(image.shape[:2])
            new_height = int(image.shape[0] * scale)
            new_width = int(image.shape[1] * scale)
            print(f"  Downsampling large image from {image.shape[:2]} to ({new_height}, {new_width})")
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"  Image downsampled for performance")
        base_name = image_path.stem

        # Analyze skew with optional intermediate step saving
        output_base = (
            str(output_dir / base_name)
            if use_organized_output
            else str(output_dir / base_name)
        )
        save_intermediates = (
            command_args.get("save_intermediates", False) if command_args else False
        )
        print(f"  Starting skew analysis with range: {config.angle_range}, step: {config.angle_step}")
        
        # Add performance monitoring
        start_time = time.time()
        skew_info = analyze_skew_detailed(
            image,
            config.angle_range,
            config.angle_step,
            config.min_angle_correction,
            save_intermediates=save_intermediates,
            output_base=output_base,
        )
        analysis_time = time.time() - start_time
        print(f"  Skew analysis complete in {analysis_time:.1f}s, angle: {skew_info['rotation_angle']:.2f}°")

        # Use the deskewed image from analysis (no duplicate computation needed!)
        print(f"  Using already computed deskewed image...")
        deskewed = skew_info["deskewed_image"]
        print(f"  Deskewed image ready")

        # Create visualizations (optimized for performance)
        print(f"  Creating visualizations...")
        overlay = draw_line_detection_overlay(image, skew_info)
        print(f"  Line detection overlay created")
        angle_plot = create_angle_histogram_plot(skew_info)
        print(f"  Angle histogram plot created")
        
        # Use smaller binary image for visualization to improve performance
        binary_small = cv2.resize(skew_info["binary"], 
                                 (skew_info["binary"].shape[1]//2, skew_info["binary"].shape[0]//2), 
                                 interpolation=cv2.INTER_AREA)
        binary_vis = create_binary_visualization(binary_small)
        print(f"  Binary visualization created")

        # Create comparison
        print(f"  Creating comprehensive comparison visualization...")
        comparison = create_deskew_comparison(
            image, deskewed, overlay, angle_plot, binary_vis, skew_info
        )
        print(f"  Comparison visualization created")

        if use_organized_output:
            # Create temporary files first
            temp_dir = Path(output_dir) / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)

            temp_files = {
                "original": str(temp_dir / f"{base_name}_original.jpg"),
                "grayscale": str(temp_dir / f"{base_name}_01_grayscale.jpg"),
                "binary": str(temp_dir / f"{base_name}_02_binary.jpg"),
                "projection": str(temp_dir / f"{base_name}_03_best_projection.jpg"),
                "angle_scores": str(temp_dir / f"{base_name}_04_angle_scores.jpg"),
                "overlay": str(temp_dir / f"{base_name}_05_skew_analysis.jpg"),
                "deskewed": str(temp_dir / f"{base_name}_06_deskewed.jpg"),
                "angle_plot": str(temp_dir / f"{base_name}_07_histogram_scores.jpg"),
                "comparison": str(temp_dir / f"{base_name}_08_deskew_comparison.jpg"),
            }

            print(f"  Saving visualization files...")
            cv2.imwrite(temp_files["original"], image)
            # Intermediate files already saved by analyze_skew_detailed
            cv2.imwrite(temp_files["overlay"], overlay)
            cv2.imwrite(temp_files["deskewed"], deskewed)
            cv2.imwrite(temp_files["angle_plot"], angle_plot)
            cv2.imwrite(temp_files["comparison"], comparison)
            print(f"  5 visualization files saved to temp directory")

            # Prepare analysis data
            analysis_data = {
                "image_name": image_path.name,
                "skew_info": {
                    k: (
                        float(v)
                        if isinstance(v, np.floating)
                        else (
                            int(v)
                            if isinstance(v, np.integer)
                            else (
                                v.tolist()
                                if isinstance(v, np.ndarray) and k != "edges"
                                else v
                            )
                        )
                    )
                    for k, v in skew_info.items()
                    if k != "edges"
                },
                "config_used": {
                    "angle_range": config.angle_range,
                    "angle_step": config.angle_step,
                    "min_angle_correction": config.min_angle_correction,
                },
            }

            # Organize into structured output
            output_files = organize_visualization_output(
                "deskew", temp_files, analysis_data, output_dir
            )

            # Clean up temp directory
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

        else:
            # Use old flat structure
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectory for processed images (for pipeline use)
            processed_dir = output_dir / "processed_images"
            processed_dir.mkdir(exist_ok=True)

            output_files = {
                "original": str(output_dir / f"{base_name}_original.jpg"),
                "grayscale": str(output_dir / f"{base_name}_01_grayscale.jpg"),
                "binary": str(output_dir / f"{base_name}_02_binary.jpg"),
                "projection": str(output_dir / f"{base_name}_03_best_projection.jpg"),
                "angle_scores": str(output_dir / f"{base_name}_04_angle_scores.jpg"),
                "overlay": str(output_dir / f"{base_name}_05_skew_analysis.jpg"),
                "deskewed": str(processed_dir / f"{base_name}_06_deskewed.jpg"),
                "angle_plot": str(output_dir / f"{base_name}_07_histogram_scores.jpg"),
                "comparison": str(output_dir / f"{base_name}_08_deskew_comparison.jpg"),
            }

            print(f"  Saving visualization files to flat structure...")
            cv2.imwrite(output_files["original"], image)
            # Intermediate files already saved by analyze_skew_detailed
            cv2.imwrite(output_files["overlay"], overlay)
            cv2.imwrite(output_files["deskewed"], deskewed)
            cv2.imwrite(output_files["angle_plot"], angle_plot)
            cv2.imwrite(output_files["comparison"], comparison)
            print(f"  5 visualization files saved")

            # Save analysis data (skip if --no-params flag is set to avoid huge files)
            if not command_args.get("no_params", False):
                analysis_file = output_dir / f"{base_name}_deskew_analysis.json"
                # Use lightweight summary instead of full data to prevent huge JSON files
                analysis_data = {
                    "skew_info": create_lightweight_summary(skew_info),
                    "config_used": {
                        "angle_range": config.angle_range,
                        "angle_step": config.angle_step,
                        "min_angle_correction": config.min_angle_correction,
                    },
                }

                with open(analysis_file, "w") as f:
                    json.dump(convert_numpy_types(analysis_data), f, indent=2)

                output_files["analysis"] = str(analysis_file)
            else:
                print(f"  Skipping detailed analysis file (--no-params flag set)")

        # Prepare processing results for parameter documentation
        processing_results = {
            "image_name": image_path.name,
            "success": True,
            "skew_info": skew_info,
            "output_files": output_files,
        }

        # Save parameter documentation (skip if --no-params flag is set)
        if command_args is None:
            command_args = {}

        param_file = None
        if not command_args.get("no_params", False):
            try:
                param_file = save_step_parameters(
                    step_name="deskew",
                    config_obj=config,
                    command_args=command_args,
                    processing_results=processing_results,
                    output_dir=(
                        output_dir if not use_organized_output else output_dir.parent
                    ),
                    config_source=config_source,
                )
            except Exception as param_error:
                print(f"  WARNING: Could not save parameter file: {param_error}")
        else:
            print(f"  Skipping parameter documentation (--no-params flag set)")

        # Include parameter file in organized output if using organized structure
        if use_organized_output and param_file:
            output_files["parameters"] = str(param_file)

        result = {
            "image_name": image_path.name,
            "success": True,
            "skew_info": skew_info,
            "output_files": output_files,
            "parameter_file": str(param_file) if param_file else None,
        }

        print(
            f"  SUCCESS: Rotation angle: {skew_info['rotation_angle']:.2f}°, confidence: {skew_info['confidence']:.3f}"
        )
        print(
            f"  SUCCESS: Lines detected: {skew_info['line_count']}, will rotate: {skew_info['will_rotate']}"
        )

        return result

    except Exception as e:
        print(f"  ERROR: {e}")
        return {"image_name": image_path.name, "success": False, "error": str(e)}


def main():
    """Main function for deskew visualization."""
    parser = argparse.ArgumentParser(description="Visualize deskewing results")
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
        default=None,
        help="Output directory for visualizations (default: organized structure)",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Use flat output structure instead of organized folders",
    )
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Save all intermediate processing steps (grayscale, edges, raw lines, filtered lines)",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Fast processing mode - minimal visualizations for pipeline use",
    )
    parser.add_argument(
        "--no-params",
        action="store_true",
        help="Skip parameter documentation saving (prevents hanging in pipeline mode)",
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

    # Parameter overrides (take precedence over config file)
    parser.add_argument(
        "--angle-range",
        type=int,
        default=None,
        help="Maximum angle range for detection (degrees)",
    )
    parser.add_argument(
        "--angle-step",
        type=float,
        default=None,
        help="Angle step for detection (degrees)",
    )
    parser.add_argument(
        "--min-angle-correction",
        type=float,
        default=None,
        help="Minimum angle to trigger rotation (degrees)",
    )

    args = parser.parse_args()

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

    # Collect command line arguments for parameter documentation
    command_args = {
        "angle_range": args.angle_range,
        "angle_step": args.angle_step,
        "min_angle_correction": args.min_angle_correction,
        "config_file": str(args.config_file) if args.config_file else None,
        "stage": args.stage,
        "save_intermediates": args.save_intermediates,
        "no_params": args.no_params,
    }

    # Determine config source
    config_source = "default"
    if args.config_file and args.config_file.exists():
        config_source = "file"
    if any(
        v is not None
        for v in [args.angle_range, args.angle_step, args.min_angle_correction]
    ):
        config_source += "_with_overrides"

    # Apply command line parameter overrides
    if args.angle_range is not None:
        config.angle_range = args.angle_range
    if args.angle_step is not None:
        config.angle_step = args.angle_step
    if args.min_angle_correction is not None:
        config.min_angle_correction = args.min_angle_correction

    config.verbose = False  # Keep visualization quiet

    # Handle output directory
    if args.flat_output or args.output_dir:
        # Use specified directory or flat structure
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else Path("data/output/visualization/deskew")
        )
        use_organized = False
    else:
        # Use organized structure
        manager = get_default_output_manager()
        output_dir = manager.create_run_directory("deskew")
        use_organized = True

    print(f"Visualizing deskewing on {len(image_paths)} images")
    if args.test_images:
        print("Batch mode: Processing all images from test_images directory")
    print("Parameters:")
    print(f"  - Angle range: ±{config.angle_range}°")
    print(f"  - Angle step: {config.angle_step}°")
    print(f"  - Min correction: {config.min_angle_correction}°")
    print(f"  - Save intermediates: {args.save_intermediates}")
    print(f"Output directory: {output_dir}")
    print()

    # Process all images
    results = []
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Starting: {image_path.name}")
        result = process_image_deskew_visualization(
            image_path, config, output_dir, use_organized, command_args, config_source
        )
        results.append(result)
        if result["success"]:
            print(f"[{i}/{len(image_paths)}] SUCCESS: {image_path.name}")
        else:
            print(f"[{i}/{len(image_paths)}] FAILED: {image_path.name}")

    # Summary
    successful_results = [r for r in results if r["success"]]
    print(f"\n{'='*60}")
    print("DESKEW VISUALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {len(successful_results)}/{len(image_paths)} images")

    if successful_results:
        rotated_count = sum(
            1 for r in successful_results if r["skew_info"]["will_rotate"]
        )
        print(f"Images that will be rotated: {rotated_count}/{len(successful_results)}")

        if rotated_count > 0:
            avg_angle = (
                sum(
                    abs(r["skew_info"]["rotation_angle"])
                    for r in successful_results
                    if r["skew_info"]["will_rotate"]
                )
                / rotated_count
            )
            print(f"Average rotation angle: {avg_angle:.2f}°")

        avg_confidence = sum(
            r["skew_info"]["confidence"] for r in successful_results
        ) / len(successful_results)
        print(f"Average detection confidence: {avg_confidence:.3f}")

    print(f"\nOutput files saved to: {output_dir}")

    if use_organized:
        print(
            "Use 'python tools/check_results.py latest deskew --view' to view results"
        )
        print("Use 'python tools/check_results.py list' to see all runs")
    else:
        print(
            "Review the '_08_deskew_comparison.jpg' files to assess deskewing quality"
        )
        if args.save_intermediates:
            print("Intermediate steps saved:")
            print("  01_grayscale.jpg - Converted to grayscale")
            print("  02_edges.jpg - Canny edge detection")
            print("  03_raw_hough_lines.jpg - All detected Hough lines")
            print("  04_filtered_lines.jpg - Lines after angle filtering")
            print("  05_line_detection.jpg - Final analysis overlay")

    # Save summary (only for flat structure, organized structure handles this automatically)
    if not use_organized:
        summary_file = output_dir / "deskew_visualization_summary.json"
        
        # Create lightweight summary to prevent huge JSON files
        lightweight_results = []
        for result in results:
            if result["success"] and "skew_info" in result:
                lightweight_result = {
                    "image_name": result["image_name"],
                    "success": result["success"],
                    "skew_info": create_lightweight_summary(result["skew_info"]),
                    # Exclude output_files and other heavy data
                }
            else:
                lightweight_result = {
                    "image_name": result["image_name"],
                    "success": result["success"],
                    "error": result.get("error", "Unknown error"),
                }
            lightweight_results.append(lightweight_result)
        
        summary_data = {
            "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
            "config_parameters": {
                "angle_range": config.angle_range,
                "angle_step": config.angle_step,
                "min_angle_correction": config.min_angle_correction,
            },
            "results": lightweight_results,
        }

        with open(summary_file, "w") as f:
            json.dump(convert_numpy_types(summary_data), f, indent=2)

        print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
