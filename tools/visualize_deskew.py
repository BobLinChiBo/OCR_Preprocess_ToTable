#!/usr/bin/env python3
"""
Deskew Visualization Script

This script visualizes the deskewing process, helping you assess angle detection
quality and adjust deskewing parameters.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import sys
from typing import Dict, Any, List, Tuple

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.ocr_pipeline.config import Stage1Config, Stage2Config
import src.ocr_pipeline.utils as ocr_utils
from output_manager import (
    get_default_output_manager,
    organize_visualization_output,
    get_test_images,
    convert_numpy_types,
    save_step_parameters,
)


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


def analyze_skew_detailed(
    image,
    angle_range=10,
    angle_step=0.2,
    min_angle_correction=0.5,
    save_intermediates=False,
    output_base=None,
):
    """Enhanced skew analysis using histogram variance optimization - matches main pipeline logic.

    This function uses the exact same algorithm as utils.deskew_image() but provides
    detailed analysis and intermediate step visualization for debugging.
    """
    # Convert to binary for optimal histogram analysis (same as main pipeline)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 1: Save grayscale and binary conversion
    if save_intermediates and output_base:
        grayscale_file = f"{output_base}_01_grayscale.jpg"
        binary_file = f"{output_base}_02_binary.jpg"
        print(f"  DEBUG: Saving grayscale to {grayscale_file}")
        print(f"  DEBUG: Saving binary to {binary_file}")
        cv2.imwrite(grayscale_file, gray)
        cv2.imwrite(binary_file, binary)

    # Use the optimized deskewing function from utils.py instead of duplicating logic
    _, detected_angle = ocr_utils.deskew_image(
        image, angle_range, angle_step, min_angle_correction
    )

    # For visualization purposes, we still need to generate some score data
    # Use a simplified approach that matches the optimized algorithm's coarse search
    def histogram_variance_score(binary_img: np.ndarray, angle: float) -> float:
        """Calculate sharpness of horizontal projection after rotation."""
        h, w = binary_img.shape
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            binary_img,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,  # Fast interpolation for visualization
            borderMode=cv2.BORDER_REPLICATE,
        )
        horizontal_projection = np.sum(rotated, axis=1)
        differences = horizontal_projection[1:] - horizontal_projection[:-1]
        return np.sum(differences**2)

    # Generate visualization data using coarse search (much faster than exhaustive)
    coarse_step = max(1.0, angle_step * 5)  # Use larger steps for visualization
    angles = np.arange(-angle_range, angle_range + coarse_step, coarse_step)

    # Use downsampled image for speed (same as optimized algorithm)
    h, w = binary.shape
    small_binary = cv2.resize(binary, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    scores = [histogram_variance_score(small_binary, angle) for angle in angles]

    best_angle = detected_angle  # Use the accurate result from optimized algorithm
    best_score = max(scores) if scores else 0

    # Calculate confidence based on score distribution
    score_std = np.std(scores)
    confidence = min(1.0, best_score / (np.mean(scores) + score_std + 1e-6))

    # Step 2: Save histogram analysis visualizations
    if save_intermediates and output_base:
        # Save best angle projection
        best_rotated = cv2.warpAffine(
            binary,
            cv2.getRotationMatrix2D(
                (binary.shape[1] // 2, binary.shape[0] // 2), best_angle, 1.0
            ),
            (binary.shape[1], binary.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        best_projection = np.sum(best_rotated, axis=1)

        # Create projection visualization
        proj_vis = create_projection_visualization(
            best_rotated, best_projection, best_angle
        )
        cv2.imwrite(f"{output_base}_03_best_projection.jpg", proj_vis)

        # Create angle score plot
        score_plot = create_angle_score_plot(angles, scores, best_angle)
        cv2.imwrite(f"{output_base}_04_angle_scores.jpg", score_plot)

    # Create angle histogram for compatibility with existing visualization code
    angle_histogram = {}
    for i, angle in enumerate(angles):
        angle_histogram[round(angle, 1)] = scores[i]

    # Determine if rotation will be applied (same logic as main pipeline)
    will_rotate = abs(best_angle) >= min_angle_correction

    return {
        "has_lines": True,  # Always true for histogram-based analysis
        "rotation_angle": best_angle,
        "line_count": len(angles),  # Number of tested angles
        "angles": angles.tolist(),
        "scores": scores,
        "confidence": confidence,
        "angle_std": score_std,
        "angle_histogram": angle_histogram,
        "best_score": best_score,
        "gray": gray,
        "binary": binary,
        "will_rotate": will_rotate,
        "method": "histogram_variance",  # Identifier for visualization
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
        x = margin + int((i / (len(angles) - 1)) * plot_area_width)
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
    best_angle_idx = angles.index(rotation_angle) if rotation_angle in angles else -1
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
    target_height = 500

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
    print(f"Processing: {image_path.name}")

    try:
        # Load image
        image = ocr_utils.load_image(image_path)
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
        skew_info = analyze_skew_detailed(
            image,
            config.angle_range,
            config.angle_step,
            config.min_angle_correction,
            save_intermediates=save_intermediates,
            output_base=output_base,
        )

        # Get the deskewed image using the detected angle from analysis
        deskewed, _ = ocr_utils.deskew_image(
            image, config.angle_range, config.angle_step, config.min_angle_correction
        )
        # Note: We use the detected_angle from skew_info which was calculated in analyze_skew_detailed

        # Create visualizations
        overlay = draw_line_detection_overlay(image, skew_info)
        angle_plot = create_angle_histogram_plot(skew_info)
        binary_vis = create_binary_visualization(skew_info["binary"])

        # Create comparison
        comparison = create_deskew_comparison(
            image, deskewed, overlay, angle_plot, binary_vis, skew_info
        )

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

            cv2.imwrite(temp_files["original"], image)
            # Intermediate files already saved by analyze_skew_detailed
            cv2.imwrite(temp_files["overlay"], overlay)
            cv2.imwrite(temp_files["deskewed"], deskewed)
            cv2.imwrite(temp_files["angle_plot"], angle_plot)
            cv2.imwrite(temp_files["comparison"], comparison)

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

            cv2.imwrite(output_files["original"], image)
            # Intermediate files already saved by analyze_skew_detailed
            cv2.imwrite(output_files["overlay"], overlay)
            cv2.imwrite(output_files["deskewed"], deskewed)
            cv2.imwrite(output_files["angle_plot"], angle_plot)
            cv2.imwrite(output_files["comparison"], comparison)

            # Save analysis data
            analysis_file = output_dir / f"{base_name}_deskew_analysis.json"
            analysis_data = {
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

            with open(analysis_file, "w") as f:
                json.dump(convert_numpy_types(analysis_data), f, indent=2)

            output_files["analysis"] = str(analysis_file)

        # Prepare processing results for parameter documentation
        processing_results = {
            "image_name": image_path.name,
            "success": True,
            "skew_info": skew_info,
            "output_files": output_files,
        }

        # Save parameter documentation
        if command_args is None:
            command_args = {}

        param_file = None
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
        print(f"Batch mode: Processing all images from test_images directory")
    print(f"Parameters:")
    print(f"  - Angle range: ±{config.angle_range}°")
    print(f"  - Angle step: {config.angle_step}°")
    print(f"  - Min correction: {config.min_angle_correction}°")
    print(f"  - Save intermediates: {args.save_intermediates}")
    print(f"Output directory: {output_dir}")
    print()

    # Process all images
    results = []
    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Processing: {image_path.name}")
        result = process_image_deskew_visualization(
            image_path, config, output_dir, use_organized, command_args, config_source
        )
        results.append(result)

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
            f"Use 'python tools/check_results.py latest deskew --view' to view results"
        )
        print(f"Use 'python tools/check_results.py list' to see all runs")
    else:
        print(
            f"Review the '_08_deskew_comparison.jpg' files to assess deskewing quality"
        )
        if args.save_intermediates:
            print(f"Intermediate steps saved:")
            print(f"  01_grayscale.jpg - Converted to grayscale")
            print(f"  02_edges.jpg - Canny edge detection")
            print(f"  03_raw_hough_lines.jpg - All detected Hough lines")
            print(f"  04_filtered_lines.jpg - Lines after angle filtering")
            print(f"  05_line_detection.jpg - Final analysis overlay")

    # Save summary (only for flat structure, organized structure handles this automatically)
    if not use_organized:
        summary_file = output_dir / "deskew_visualization_summary.json"
        summary_data = {
            "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
            "config_parameters": {
                "angle_range": config.angle_range,
                "angle_step": config.angle_step,
                "min_angle_correction": config.min_angle_correction,
            },
            "results": results,
        }

        with open(summary_file, "w") as f:
            json.dump(convert_numpy_types(summary_data), f, indent=2)

        print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
