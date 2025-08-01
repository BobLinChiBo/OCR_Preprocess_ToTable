#!/usr/bin/env python3
"""
Table Crop Visualization Script

This script visualizes the table cropping process, helping you assess
the final cropping boundaries and region selection parameters.
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

from src.ocr_pipeline.config import Config  # noqa: E402
from src.ocr_pipeline.utils import (
    load_image,
    detect_table_lines,
    crop_table_region,
)  # noqa: E402
from output_manager import (
    get_test_images,
    convert_numpy_types,
    save_step_parameters,
)  # noqa: E402


def analyze_table_crop_detailed(
    image: np.ndarray,
    min_line_length: int = 400,
    max_line_gap: int = 30,
    hough_threshold: int = 300,
    crop_padding: int = 10,
) -> Dict[str, Any]:
    """Enhanced table cropping analysis with detailed boundary detection."""
    # First detect lines using shared function
    h_lines, v_lines = detect_table_lines(
        image, 
        min_line_length=min_line_length, 
        max_line_gap=max_line_gap,
        hough_threshold=hough_threshold
    )

    # Then crop using shared function
    crop_region, crop_analysis = crop_table_region(
        image, h_lines, v_lines, crop_padding=crop_padding, return_analysis=True
    )

    height, width = image.shape[:2]

    # Additional analysis for visualization
    h_line_density = len(h_lines) / height if height > 0 else 0
    v_line_density = len(v_lines) / width if width > 0 else 0

    # Line distribution analysis
    h_y_positions = []
    v_x_positions = []

    for x1, y1, x2, y2 in h_lines:
        h_y_positions.extend([y1, y2])

    for x1, y1, x2, y2 in v_lines:
        v_x_positions.extend([x1, x2])

    # Calculate line spacing
    h_spacing = []
    v_spacing = []

    if len(h_y_positions) > 1:
        sorted_h = sorted(set(h_y_positions))
        h_spacing = [sorted_h[i + 1] - sorted_h[i] for i in range(len(sorted_h) - 1)]

    if len(v_x_positions) > 1:
        sorted_v = sorted(set(v_x_positions))
        v_spacing = [sorted_v[i + 1] - sorted_v[i] for i in range(len(sorted_v) - 1)]

    # Convert crop analysis to match expected format
    if "error" in crop_analysis:
        crop_bounds = None
        has_crop_region = False
        coverage_ratio = 0
        crop_size = (0, 0)
    else:
        bounds = crop_analysis["padded_boundaries"]
        crop_bounds = (
            bounds["min_x"],
            bounds["min_y"],
            bounds["max_x"],
            bounds["max_y"],
        )
        has_crop_region = True
        coverage_ratio = crop_analysis["crop_efficiency"]
        crop_size = crop_analysis["cropped_dimensions"]

    return {
        "h_lines": h_lines,
        "v_lines": v_lines,
        "crop_bounds": crop_bounds,
        "crop_region": crop_region,
        "has_crop_region": has_crop_region,
        "coverage_ratio": coverage_ratio,
        "original_size": (width, height),
        "crop_size": crop_size,
        "h_line_density": h_line_density,
        "v_line_density": v_line_density,
        "h_spacing": h_spacing,
        "v_spacing": v_spacing,
        "avg_h_spacing": np.mean(h_spacing) if h_spacing else 0,
        "avg_v_spacing": np.mean(v_spacing) if v_spacing else 0,
        "config_used": {
            "min_line_length": min_line_length,
            "max_line_gap": max_line_gap,
            "crop_padding": crop_padding,
        },
    }


def draw_crop_overlay(image: np.ndarray, crop_info: Dict[str, Any]) -> np.ndarray:
    """Draw overlay showing crop boundaries and detected lines."""
    overlay = image.copy()

    # Draw horizontal lines in red
    for x1, y1, x2, y2 in crop_info["h_lines"]:
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Draw vertical lines in blue
    for x1, y1, x2, y2 in crop_info["v_lines"]:
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw crop boundary
    if crop_info["crop_bounds"]:
        min_x, min_y, max_x, max_y = crop_info["crop_bounds"]
        cv2.rectangle(overlay, (min_x, min_y), (max_x, max_y), (0, 255, 0), 4)

        # Add semi-transparent overlay for excluded regions
        excluded_overlay = np.zeros_like(overlay)

        # Top exclusion
        if min_y > 0:
            cv2.rectangle(
                excluded_overlay, (0, 0), (overlay.shape[1], min_y), (0, 0, 255), -1
            )

        # Bottom exclusion
        if max_y < overlay.shape[0]:
            cv2.rectangle(
                excluded_overlay,
                (0, max_y),
                (overlay.shape[1], overlay.shape[0]),
                (0, 0, 255),
                -1,
            )

        # Left exclusion
        if min_x > 0:
            cv2.rectangle(
                excluded_overlay, (0, 0), (min_x, overlay.shape[0]), (0, 0, 255), -1
            )

        # Right exclusion
        if max_x < overlay.shape[1]:
            cv2.rectangle(
                excluded_overlay,
                (max_x, 0),
                (overlay.shape[1], overlay.shape[0]),
                (0, 0, 255),
                -1,
            )

        # Blend excluded regions
        alpha = 0.2
        overlay = cv2.addWeighted(overlay, 1 - alpha, excluded_overlay, alpha, 0)

    # Add info text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    crop_width, crop_height = crop_info["crop_size"]
    orig_width, orig_height = crop_info["original_size"]

    text_lines = [
        f"H lines: {len(crop_info['h_lines'])}",
        f"V lines: {len(crop_info['v_lines'])}",
        f"Original: {orig_width}x{orig_height}",
        f"Cropped: {crop_width}x{crop_height}",
        f"Coverage: {crop_info['coverage_ratio']:.1%}",
        f"Has crop: {crop_info['has_crop_region']}",
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
        color = (
            (255, 255, 255)
            if i < 5
            else ((0, 255, 0) if crop_info["has_crop_region"] else (0, 0, 255))
        )
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


def create_spacing_analysis_plot(crop_info: Dict[str, Any]) -> np.ndarray:
    """Create a plot showing line spacing analysis."""
    plot_width = 600
    plot_height = 400
    plot_img = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255

    h_spacing = crop_info["h_spacing"]
    v_spacing = crop_info["v_spacing"]

    if not h_spacing and not v_spacing:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            plot_img, "No spacing data available", (150, 200), font, 1.0, (0, 0, 0), 2
        )
        return plot_img

    # Split plot into two halves
    half_width = plot_width // 2

    # Plot horizontal spacing
    if h_spacing:
        max_h_spacing = max(h_spacing)
        for i, spacing in enumerate(h_spacing):
            bar_height = (
                int((spacing / max_h_spacing) * (plot_height - 100))
                if max_h_spacing > 0
                else 0
            )
            x = i * (half_width // len(h_spacing))
            y = plot_height - 50 - bar_height
            cv2.rectangle(plot_img, (x, y), (x + 15, plot_height - 50), (0, 0, 255), -1)

    # Plot vertical spacing
    if v_spacing:
        max_v_spacing = max(v_spacing)
        for i, spacing in enumerate(v_spacing):
            bar_height = (
                int((spacing / max_v_spacing) * (plot_height - 100))
                if max_v_spacing > 0
                else 0
            )
            x = half_width + i * (half_width // len(v_spacing))
            y = plot_height - 50 - bar_height
            cv2.rectangle(plot_img, (x, y), (x + 15, plot_height - 50), (255, 0, 0), -1)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(plot_img, "Line Spacing Analysis", (200, 30), font, 0.8, (0, 0, 0), 2)
    cv2.putText(plot_img, "Horizontal", (10, 60), font, 0.6, (0, 0, 255), 1)
    cv2.putText(plot_img, "Vertical", (half_width + 10, 60), font, 0.6, (255, 0, 0), 1)

    if h_spacing:
        cv2.putText(
            plot_img,
            f"Avg: {crop_info['avg_h_spacing']:.1f}px",
            (10, plot_height - 20),
            font,
            0.5,
            (0, 0, 255),
            1,
        )

    if v_spacing:
        cv2.putText(
            plot_img,
            f"Avg: {crop_info['avg_v_spacing']:.1f}px",
            (half_width + 10, plot_height - 20),
            font,
            0.5,
            (255, 0, 0),
            1,
        )

    return plot_img


def create_crop_comparison(
    original: np.ndarray,
    overlay: np.ndarray,
    cropped: np.ndarray,
    spacing_plot: np.ndarray,
    crop_info: Dict[str, Any],
) -> np.ndarray:
    """Create a comprehensive comparison showing all crop results."""
    target_height = 500

    # Resize main images
    scale = target_height / original.shape[0]
    new_width = int(original.shape[1] * scale)

    orig_resized = cv2.resize(original, (new_width, target_height))
    overlay_resized = cv2.resize(overlay, (new_width, target_height))

    # Resize cropped image if it exists
    if cropped is not None and cropped.size > 0:
        crop_scale = target_height / cropped.shape[0]
        crop_width = int(cropped.shape[1] * crop_scale)
        cropped_resized = cv2.resize(cropped, (crop_width, target_height))
    else:
        # Create placeholder for no crop
        cropped_resized = np.zeros((target_height, new_width // 2, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            cropped_resized,
            "No crop region",
            (50, target_height // 2),
            font,
            1.0,
            (255, 255, 255),
            2,
        )
        crop_width = new_width // 2

    # Resize spacing plot
    plot_scale = new_width / spacing_plot.shape[1]
    plot_height = int(spacing_plot.shape[0] * plot_scale)
    plot_resized = cv2.resize(spacing_plot, (new_width, plot_height))

    # Create labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_height = 40

    def create_label(
        text: str, width: int, color: tuple = (255, 255, 255)
    ) -> np.ndarray:
        label_img = np.zeros((label_height, width, 3), dtype=np.uint8)
        cv2.putText(label_img, text, (10, 25), font, 0.8, color, 2)
        return label_img

    # Create top row (original and overlay)
    orig_label = create_label("Original", new_width)
    overlay_label = create_label("Crop Detection", new_width)

    top_row = np.vstack([orig_label, orig_resized, overlay_label, overlay_resized])

    # Create bottom row (cropped result and analysis)
    crop_color = (0, 255, 0) if crop_info["has_crop_region"] else (255, 255, 255)
    cropped_label = create_label(
        f"Cropped Result ({crop_info['coverage_ratio']:.1%} coverage)",
        crop_width,
        crop_color,
    )
    plot_label = create_label("Spacing Analysis", new_width)

    cropped_panel = np.vstack([cropped_label, cropped_resized])
    plot_panel = np.vstack([plot_label, plot_resized])

    # Pad cropped panel to match plot width if needed
    if cropped_panel.shape[1] < new_width:
        padding_width = new_width - cropped_panel.shape[1]
        padding = np.zeros((cropped_panel.shape[0], padding_width, 3), dtype=np.uint8)
        cropped_panel = np.hstack([cropped_panel, padding])

    bottom_row = np.vstack([cropped_panel, plot_panel])

    # Ensure same width
    max_width = max(top_row.shape[1], bottom_row.shape[1])

    if top_row.shape[1] < max_width:
        padding = np.zeros(
            (top_row.shape[0], max_width - top_row.shape[1], 3), dtype=np.uint8
        )
        top_row = np.hstack([top_row, padding])

    if bottom_row.shape[1] < max_width:
        padding = np.zeros(
            (bottom_row.shape[0], max_width - bottom_row.shape[1], 3), dtype=np.uint8
        )
        bottom_row = np.hstack([bottom_row, padding])

    comparison = np.vstack([top_row, bottom_row])

    return comparison


def process_image_table_crop_visualization(
    image_path: Path,
    config: Config,
    output_dir: Path,
    crop_padding: int = 10,
    command_args: Dict[str, Any] = None,
    config_source: str = "default",
) -> Dict[str, Any]:
    """Process a single image and create table crop visualization."""
    print(f"Processing: {image_path.name}")

    try:
        # Load image
        image = load_image(image_path)

        # Analyze table cropping
        crop_info = analyze_table_crop_detailed(
            image, config.min_line_length, config.max_line_gap, crop_padding
        )

        # Create visualizations
        overlay = draw_crop_overlay(image, crop_info)
        spacing_plot = create_spacing_analysis_plot(crop_info)

        # Get cropped region
        cropped = crop_info["crop_region"] if crop_info["has_crop_region"] else None

        # Create comparison
        comparison = create_crop_comparison(
            image, overlay, cropped, spacing_plot, crop_info
        )

        # Save outputs
        base_name = image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        output_files = {
            "original": str(output_dir / f"{base_name}_original.jpg"),
            "overlay": str(output_dir / f"{base_name}_crop_overlay.jpg"),
            "spacing_plot": str(output_dir / f"{base_name}_spacing_analysis.jpg"),
            "comparison": str(output_dir / f"{base_name}_crop_comparison.jpg"),
        }

        cv2.imwrite(output_files["original"], image)
        cv2.imwrite(output_files["overlay"], overlay)
        cv2.imwrite(output_files["spacing_plot"], spacing_plot)
        cv2.imwrite(output_files["comparison"], comparison)

        # Save cropped region if it exists
        if cropped is not None:
            output_files["cropped"] = str(output_dir / f"{base_name}_cropped.jpg")
            cv2.imwrite(output_files["cropped"], cropped)

        # Save analysis data
        analysis_file = output_dir / f"{base_name}_crop_analysis.json"
        analysis_data = {
            "crop_info": {k: v for k, v in crop_info.items() if k != "crop_region"},
            "config_used": {
                "min_line_length": config.min_line_length,
                "max_line_gap": config.max_line_gap,
                "crop_padding": crop_padding,
            },
        }

        with open(analysis_file, "w") as f:
            json.dump(convert_numpy_types(analysis_data), f, indent=2)

        output_files["analysis"] = str(analysis_file)

        # Prepare processing results for parameter documentation
        processing_results = {
            "image_name": image_path.name,
            "success": True,
            "crop_info": crop_info,
            "output_files": output_files,
            "analysis_data": analysis_data,
        }

        # Save parameter documentation
        if command_args is None:
            command_args = {}

        param_file = save_step_parameters(
            step_name="table_crop",
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
            "crop_info": crop_info,
            "output_files": output_files,
            "parameter_file": str(param_file) if param_file else None,
        }

        print(f"  SUCCESS: Has crop region: {crop_info['has_crop_region']}")
        if crop_info["has_crop_region"]:
            print(
                f"  SUCCESS: Coverage: {crop_info['coverage_ratio']:.1%}, Size: {crop_info['crop_size']}"
            )

        return result

    except Exception as e:
        print(f"  ERROR: {e}")
        return {"image_name": image_path.name, "success": False, "error": str(e)}


def main():
    """Main function for table crop visualization."""
    parser = argparse.ArgumentParser(description="Visualize table cropping results")
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
        "--output-dir",
        default="data/output/visualization/table_crop",
        help="Output directory for visualizations",
    )

    # Parameter options
    parser.add_argument(
        "--min-line-length",
        type=int,
        default=100,
        help="Minimum line length for detection",
    )
    parser.add_argument(
        "--max-line-gap", type=int, default=10, help="Maximum gap in line detection"
    )
    parser.add_argument(
        "--crop-padding", type=int, default=10, help="Padding around crop boundaries"
    )

    args = parser.parse_args()

    # Determine which images to process
    if args.test_images:
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

    # Collect command line arguments for parameter documentation
    command_args = {
        "min_line_length": args.min_line_length,
        "max_line_gap": args.max_line_gap,
        "crop_padding": args.crop_padding,
        "save_debug": args.save_debug,
    }

    # Config source is always command line for this script since Config is created from args
    config_source = "command_line"

    # Create configuration
    config = Config(
        min_line_length=args.min_line_length,
        max_line_gap=args.max_line_gap,
        verbose=False,
    )
    output_dir = Path(args.output_dir)

    print(f"Visualizing table cropping on {len(image_paths)} images")
    if args.test_images:
        print("Batch mode: Processing all images from test_images directory")
    print("Parameters:")
    print(f"  - Min line length: {config.min_line_length}px")
    print(f"  - Max line gap: {config.max_line_gap}px")
    print(f"  - Crop padding: {args.crop_padding}px")
    print(f"Output directory: {output_dir}")
    print()

    # Process all images
    results = []
    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Processing: {image_path.name}")
        result = process_image_table_crop_visualization(
            image_path,
            config,
            output_dir,
            args.crop_padding,
            command_args,
            config_source,
        )
        results.append(result)

    # Summary
    successful_results = [r for r in results if r["success"]]
    print(f"\n{'='*60}")
    print("TABLE CROP VISUALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {len(successful_results)}/{len(image_paths)} images")

    if successful_results:
        crop_success_count = sum(
            1 for r in successful_results if r["crop_info"]["has_crop_region"]
        )
        print(
            f"Images with successful crops: {crop_success_count}/{len(successful_results)}"
        )

        if crop_success_count > 0:
            avg_coverage = (
                sum(
                    r["crop_info"]["coverage_ratio"]
                    for r in successful_results
                    if r["crop_info"]["has_crop_region"]
                )
                / crop_success_count
            )
            print(f"Average coverage ratio: {avg_coverage:.1%}")

            coverage_range = [
                r["crop_info"]["coverage_ratio"]
                for r in successful_results
                if r["crop_info"]["has_crop_region"]
            ]
            print(
                f"Coverage range: {min(coverage_range):.1%} - {max(coverage_range):.1%}"
            )

    print(f"\nOutput files saved to: {output_dir}")
    print("Review the '_crop_comparison.jpg' files to assess cropping quality")

    # Save summary
    summary_file = output_dir / "crop_visualization_summary.json"
    summary_data = {
        "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "config_parameters": {
            "min_line_length": config.min_line_length,
            "max_line_gap": config.max_line_gap,
            "crop_padding": args.crop_padding,
        },
        "results": results,
    }

    with open(summary_file, "w") as f:
        json.dump(convert_numpy_types(summary_data), f, indent=2)

    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
