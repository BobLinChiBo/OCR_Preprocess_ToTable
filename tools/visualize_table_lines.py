#!/usr/bin/env python3
"""
Table Lines Visualization Script

This script visualizes the table line detection process, helping you assess
morphological operations and line detection parameters.
"""

import warnings
warnings.warn(
    "This script is deprecated and will be removed in a future version. "
    "Please use visualize_table_lines_v2.py or run_visualizations.py with --use-v2 flag.",
    DeprecationWarning,
    stacklevel=2
)

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

from src.ocr_pipeline.config import Config, Stage1Config, Stage2Config  # noqa: E402
import src.ocr_pipeline.utils as ocr_utils  # noqa: E402
from output_manager import (
    get_test_images,
    convert_numpy_types,
    save_step_parameters,
)  # noqa: E402


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


def detect_table_lines_detailed(
    image: np.ndarray,
    config,
    hough_threshold: int = 60,
) -> Dict[str, Any]:
    """Enhanced table line detection with detailed analysis."""
    h_lines, v_lines, analysis = ocr_utils.detect_table_lines(
        image,
        min_line_length=None,  # Let function calculate dynamically
        max_line_gap=None,     # Let function calculate dynamically
        hough_threshold=hough_threshold,
        horizontal_kernel_ratio=getattr(config, 'horizontal_kernel_ratio', 30),
        vertical_kernel_ratio=getattr(config, 'vertical_kernel_ratio', 30),
        h_erode_iterations=getattr(config, 'h_erode_iterations', 1),
        h_dilate_iterations=getattr(config, 'h_dilate_iterations', 1),
        v_erode_iterations=getattr(config, 'v_erode_iterations', 1),
        v_dilate_iterations=getattr(config, 'v_dilate_iterations', 1),
        min_table_coverage=getattr(config, 'min_table_coverage', 0.15),
        max_parallel_distance=getattr(config, 'max_parallel_distance', 12),
        angle_tolerance=getattr(config, 'angle_tolerance', 5.0),
        h_length_filter_ratio=getattr(config, 'h_length_filter_ratio', 0.6),
        v_length_filter_ratio=getattr(config, 'v_length_filter_ratio', 0.6),
        line_merge_distance_h=getattr(config, 'line_merge_distance_h', 15),
        line_merge_distance_v=getattr(config, 'line_merge_distance_v', 15),
        line_extension_tolerance=getattr(config, 'line_extension_tolerance', 20),
        max_merge_iterations=getattr(config, 'max_merge_iterations', 3),
        return_analysis=True,
    )
    # Add the lines to the analysis dict for easier access
    analysis["h_lines"] = h_lines
    analysis["v_lines"] = v_lines
    # Add line counts with the expected names
    analysis["h_line_count"] = analysis.get("h_lines_count", len(h_lines))
    analysis["v_line_count"] = analysis.get("v_lines_count", len(v_lines))
    # Add average lengths with the expected names
    analysis["h_avg_length"] = analysis.get("avg_h_length", 0)
    analysis["v_avg_length"] = analysis.get("avg_v_length", 0)
    # Determine if table structure is detected
    analysis["has_table_structure"] = len(h_lines) > 0 and len(v_lines) > 0
    return analysis


def draw_table_lines_overlay(
    image: np.ndarray, line_info: Dict[str, Any]
) -> np.ndarray:
    """Draw overlay showing detected table lines."""
    overlay = image.copy()

    # Draw horizontal lines in red
    for x1, y1, x2, y2 in line_info["h_lines"]:
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Draw vertical lines in blue
    for x1, y1, x2, y2 in line_info["v_lines"]:
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw potential table boundary (if available)
    if line_info.get("h_bounds") and line_info.get("v_bounds"):
        h_min, h_max = line_info["h_bounds"]
        v_min, v_max = line_info["v_bounds"]
        cv2.rectangle(overlay, (v_min, h_min), (v_max, h_max), (0, 255, 0), 3)

    # Add info text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    text_lines = [
        f"Horizontal lines: {line_info['h_line_count']}",
        f"Vertical lines: {line_info['v_line_count']}",
        f"H avg length: {line_info['h_avg_length']:.1f}px",
        f"V avg length: {line_info['v_avg_length']:.1f}px",
        f"Table structure: {line_info['has_table_structure']}",
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
            if i < 4
            else ((0, 255, 0) if line_info["has_table_structure"] else (0, 0, 255))
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


def draw_step_lines_overlay(
    image: np.ndarray, step_info: Dict[str, Any], step_name: str
) -> np.ndarray:
    """Draw overlay showing detected lines for a specific filtering step."""
    overlay = image.copy()
    
    # Draw horizontal lines in red
    for x1, y1, x2, y2 in step_info["h_lines"]:
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Draw vertical lines in blue
    for x1, y1, x2, y2 in step_info["v_lines"]:
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Add step info text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    text_lines = [
        f"Step: {step_name}",
        step_info["description"],
        f"H lines: {step_info['h_count']}, V lines: {step_info['v_count']}",
        f"Total: {step_info['total_count']} lines"
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


def create_filtering_steps_visualization(
    image: np.ndarray, line_info: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """Create visualizations for each filtering step."""
    step_images = {}
    
    if "intermediate_steps" not in line_info:
        return step_images
    
    for step_key, step_data in line_info["intermediate_steps"].items():
        step_name = step_key.replace("step", "Step ").replace("_", " ").title()
        step_overlay = draw_step_lines_overlay(image, step_data, step_name)
        step_images[step_key] = step_overlay
    
    return step_images


def create_morphological_visualization(line_info: Dict[str, Any]) -> np.ndarray:
    """Create visualization of preprocessing operations."""
    # Show edge detection and filtering statistics
    if "preprocessing" in line_info:
        # Create visualization showing method info
        vis_height, vis_width = 300, 600
        combined = np.ones((vis_height, vis_width, 3), dtype=np.uint8) * 50
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_lines = [
            "Table Line Detection",
            "Method: Edge Detection + Filtering",
            f"Edge pixels: {line_info['preprocessing'].get('edge_pixels', 'N/A')}",
            f"Edge density: {line_info['preprocessing'].get('edge_density', 0):.3f}",
            f"Blur applied: {line_info['preprocessing'].get('blur_applied', False)}",
            f"Initial lines: {line_info['filtering_stats'].get('initial_lines', 0)}",
            f"After filtering: {line_info['filtering_stats'].get('after_dedup', 0)}"
        ]
        
        for i, line in enumerate(text_lines):
            y_pos = 50 + i * 35
            color = (255, 255, 255) if i == 0 else (200, 200, 200)
            thickness = 2 if i == 0 else 1
            cv2.putText(combined, line, (20, y_pos), font, 0.7, color, thickness)
        
        return combined
    else:
        # Fallback: create simple info display
        vis_height, vis_width = 300, 600
        combined = np.ones((vis_height, vis_width, 3), dtype=np.uint8) * 50
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Table Line Detection", (150, 150), font, 1.0, (255, 255, 255), 2)
        return combined


def create_line_statistics_plot(line_info: Dict[str, Any]) -> np.ndarray:
    """Create a plot showing line length distributions."""
    plot_width = 600
    plot_height = 400
    plot_img = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255

    h_lengths = line_info["h_line_lengths"]
    v_lengths = line_info["v_line_lengths"]

    if not h_lengths and not v_lengths:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(plot_img, "No lines detected", (200, 200), font, 1.0, (0, 0, 0), 2)
        return plot_img

    # Create histograms
    all_lengths = h_lengths + v_lengths
    if all_lengths:
        min_length = min(all_lengths)
        max_length = max(all_lengths)

        # Create bins
        num_bins = 20
        bin_width = (
            (max_length - min_length) / num_bins if max_length > min_length else 1
        )

        h_hist = np.zeros(num_bins)
        v_hist = np.zeros(num_bins)

        # Fill histograms
        for length in h_lengths:
            bin_idx = min(int((length - min_length) / bin_width), num_bins - 1)
            h_hist[bin_idx] += 1

        for length in v_lengths:
            bin_idx = min(int((length - min_length) / bin_width), num_bins - 1)
            v_hist[bin_idx] += 1

        # Draw histograms
        max_count = (
            max(max(h_hist), max(v_hist)) if max(h_hist) > 0 or max(v_hist) > 0 else 1
        )
        bar_width = plot_width // (num_bins * 2)

        for i in range(num_bins):
            # Horizontal lines (red)
            h_height = (
                int((h_hist[i] / max_count) * (plot_height - 100))
                if max_count > 0
                else 0
            )
            x1 = i * bar_width * 2
            y1 = plot_height - 50 - h_height
            cv2.rectangle(
                plot_img,
                (x1, y1),
                (x1 + bar_width - 1, plot_height - 50),
                (0, 0, 255),
                -1,
            )

            # Vertical lines (blue)
            v_height = (
                int((v_hist[i] / max_count) * (plot_height - 100))
                if max_count > 0
                else 0
            )
            x2 = x1 + bar_width
            y2 = plot_height - 50 - v_height
            cv2.rectangle(
                plot_img,
                (x2, y2),
                (x2 + bar_width - 1, plot_height - 50),
                (255, 0, 0),
                -1,
            )

    # Add labels and legend
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(plot_img, "Line Length Distribution", (10, 30), font, 0.8, (0, 0, 0), 2)
    cv2.putText(
        plot_img,
        f"H lines: {len(h_lengths)} (avg: {line_info['h_avg_length']:.1f})",
        (10, plot_height - 20),
        font,
        0.6,
        (0, 0, 255),
        1,
    )
    cv2.putText(
        plot_img,
        f"V lines: {len(v_lengths)} (avg: {line_info['v_avg_length']:.1f})",
        (10, plot_height - 5),
        font,
        0.6,
        (255, 0, 0),
        1,
    )

    return plot_img


def create_table_lines_comparison(
    original: np.ndarray,
    overlay: np.ndarray,
    morph_vis: np.ndarray,
    stats_plot: np.ndarray,
    line_info: Dict[str, Any],
) -> np.ndarray:
    """Create a comprehensive comparison showing all table line detection results."""
    target_height = 500

    # Resize main images
    scale = target_height / original.shape[0]
    new_width = int(original.shape[1] * scale)

    orig_resized = cv2.resize(original, (new_width, target_height))
    overlay_resized = cv2.resize(overlay, (new_width, target_height))

    # Resize morphological visualization to match width
    morph_scale = new_width / morph_vis.shape[1]
    morph_height = int(morph_vis.shape[0] * morph_scale)
    morph_resized = cv2.resize(morph_vis, (new_width, morph_height))

    # Resize statistics plot
    stats_scale = new_width / stats_plot.shape[1]
    stats_height = int(stats_plot.shape[0] * stats_scale)
    stats_resized = cv2.resize(stats_plot, (new_width, stats_height))

    # Create labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_height = 40

    def create_label(
        text: str, width: int, color: tuple = (255, 255, 255)
    ) -> np.ndarray:
        label_img = np.zeros((label_height, width, 3), dtype=np.uint8)
        cv2.putText(label_img, text, (10, 25), font, 0.8, color, 2)
        return label_img

    # Create labels
    orig_label = create_label("Original", new_width)
    overlay_label = create_label("Detected Lines", new_width)
    morph_label = create_label("Morphological Operations", new_width)
    stats_label = create_label("Line Statistics", new_width)

    # Combine all components
    comparison = np.vstack(
        [
            orig_label,
            orig_resized,
            overlay_label,
            overlay_resized,
            morph_label,
            morph_resized,
            stats_label,
            stats_resized,
        ]
    )

    return comparison


def process_image_table_lines_visualization(
    image_path: Path,
    config,
    output_dir: Path,
    hough_threshold: int = 300,
    command_args: Dict[str, Any] = None,
    config_source: str = "default",
    show_filtering_steps: bool = False,
) -> Dict[str, Any]:
    """Process a single image and create table lines visualization."""
    print(f"Processing: {image_path.name}")

    try:
        # Load image
        image = ocr_utils.load_image(image_path)

        # Detect table lines with detailed analysis
        line_info = detect_table_lines_detailed(
            image,
            config,
            hough_threshold,
        )

        # Create visualizations
        overlay = draw_table_lines_overlay(image, line_info)
        morph_vis = create_morphological_visualization(line_info)
        stats_plot = create_line_statistics_plot(line_info)

        # Create step-by-step visualizations if requested
        step_images = {}
        if show_filtering_steps:
            step_images = create_filtering_steps_visualization(image, line_info)

        # Create comparison
        comparison = create_table_lines_comparison(
            image, overlay, morph_vis, stats_plot, line_info
        )

        # Save outputs
        base_name = image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        output_files = {
            "original": str(output_dir / f"{base_name}_original.jpg"),
            "overlay": str(output_dir / f"{base_name}_table_lines.jpg"),
            "morphology": str(output_dir / f"{base_name}_morphology.jpg"),
            "statistics": str(output_dir / f"{base_name}_line_stats.jpg"),
            "comparison": str(output_dir / f"{base_name}_table_lines_comparison.jpg"),
        }

        cv2.imwrite(output_files["original"], image)
        cv2.imwrite(output_files["overlay"], overlay)
        cv2.imwrite(output_files["morphology"], morph_vis)
        cv2.imwrite(output_files["statistics"], stats_plot)
        cv2.imwrite(output_files["comparison"], comparison)
        
        # Save step-by-step images if generated
        if step_images:
            for step_key, step_image in step_images.items():
                step_filename = f"{base_name}_{step_key}.jpg"
                step_path = output_dir / step_filename
                cv2.imwrite(str(step_path), step_image)
                output_files[step_key] = str(step_path)
        
        # Create processed_images subdirectory for pipeline use
        processed_dir = output_dir / "processed_images"
        processed_dir.mkdir(exist_ok=True)
        
        # Copy main output images to processed_images (for pipeline stages)
        # Only copy the essential images that subsequent stages should process
        cv2.imwrite(str(processed_dir / f"{base_name}_table_lines.jpg"), overlay)

        # Save analysis data
        analysis_file = output_dir / f"{base_name}_table_lines_analysis.json"
        analysis_data = {
            "line_detection": {
                k: v
                for k, v in line_info.items()
                if k not in ["horizontal_morph", "vertical_morph"]
            },
            "config_used": {
                "min_line_length": None,  # Calculated dynamically
                "max_line_gap": None,      # Calculated dynamically
                "hough_threshold": hough_threshold,
                "horizontal_kernel_ratio": getattr(config, 'horizontal_kernel_ratio', 30),
                "vertical_kernel_ratio": getattr(config, 'vertical_kernel_ratio', 30),
                "min_table_coverage": getattr(config, 'min_table_coverage', 0.15),
                "max_parallel_distance": getattr(config, 'max_parallel_distance', 12),
                "angle_tolerance": getattr(config, 'angle_tolerance', 5.0),
                "h_length_filter_ratio": getattr(config, 'h_length_filter_ratio', 0.6),
                "v_length_filter_ratio": getattr(config, 'v_length_filter_ratio', 0.6),
                "h_erode_iterations": getattr(config, 'h_erode_iterations', 1),
                "h_dilate_iterations": getattr(config, 'h_dilate_iterations', 1),
                "v_erode_iterations": getattr(config, 'v_erode_iterations', 1),
                "v_dilate_iterations": getattr(config, 'v_dilate_iterations', 1),
                "line_merge_distance_h": getattr(config, 'line_merge_distance_h', 15),
                "line_merge_distance_v": getattr(config, 'line_merge_distance_v', 15),
                "line_extension_tolerance": getattr(config, 'line_extension_tolerance', 20),
                "max_merge_iterations": getattr(config, 'max_merge_iterations', 3),
            },
        }

        with open(analysis_file, "w") as f:
            json.dump(convert_numpy_types(analysis_data), f, indent=2)

        output_files["analysis"] = str(analysis_file)

        # Prepare processing results for parameter documentation
        processing_results = {
            "image_name": image_path.name,
            "success": True,
            "line_info": line_info,
            "output_files": output_files,
            "analysis_data": analysis_data,
        }

        # Save parameter documentation
        if command_args is None:
            command_args = {}

        param_file = save_step_parameters(
            step_name="table_lines",
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
            "line_info": line_info,
            "output_files": output_files,
            "parameter_file": str(param_file) if param_file else None,
        }

        print(
            f"  SUCCESS: H lines: {line_info['h_line_count']}, V lines: {line_info['v_line_count']}"
        )
        print(
            f"  SUCCESS: Table structure detected: {line_info['has_table_structure']}"
        )

        return result

    except Exception as e:
        print(f"  ERROR: {e}")
        return {"image_name": image_path.name, "success": False, "error": str(e)}


def main():
    """Main function for table lines visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize table line detection results"
    )
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
        default="data/output/visualization/table_lines",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save debug images showing line detection steps",
    )
    parser.add_argument(
        "--show-filtering-steps",
        action="store_true",
        help="Generate step-by-step filtering visualization images",
    )
    parser.add_argument(
        "--no-filtering-steps",
        action="store_true",
        help="Disable step-by-step filtering visualization (default: enabled)",
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
        "--min-line-length",
        type=int,
        default=None,
        help="Minimum line length for detection",
    )
    parser.add_argument(
        "--max-line-gap", type=int, default=None, help="Maximum gap in line detection"
    )
    parser.add_argument(
        "--hough-threshold", type=int, default=None, help="Hough transform threshold"
    )
    parser.add_argument(
        "--horizontal-kernel-ratio", type=int, default=None, help="Horizontal kernel ratio (width = image_width // ratio)"
    )
    parser.add_argument(
        "--vertical-kernel-ratio", type=int, default=None, help="Vertical kernel ratio (height = image_height // ratio)"
    )
    parser.add_argument(
        "--h-length-filter-ratio", type=float, default=None, help="Remove horizontal lines shorter than this ratio of the longest horizontal line"
    )
    parser.add_argument(
        "--v-length-filter-ratio", type=float, default=None, help="Remove vertical lines shorter than this ratio of the longest vertical line"
    )
    parser.add_argument(
        "--h-erode-iterations", type=int, default=None, help="Horizontal erosion iterations for morphological operations"
    )
    parser.add_argument(
        "--h-dilate-iterations", type=int, default=None, help="Horizontal dilation iterations for morphological operations"
    )
    parser.add_argument(
        "--v-erode-iterations", type=int, default=None, help="Vertical erosion iterations for morphological operations"
    )
    parser.add_argument(
        "--v-dilate-iterations", type=int, default=None, help="Vertical dilation iterations for morphological operations"
    )
    parser.add_argument(
        "--line-merge-distance-h", type=int, default=None, help="Maximum horizontal offset to merge horizontal lines (pixels)"
    )
    parser.add_argument(
        "--line-merge-distance-v", type=int, default=None, help="Maximum vertical offset to merge vertical lines (pixels)"
    )
    parser.add_argument(
        "--line-extension-tolerance", type=int, default=None, help="Maximum gap to extend lines for connection (pixels)"
    )
    parser.add_argument(
        "--max-merge-iterations", type=int, default=None, help="Maximum number of iterative line merging passes"
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
    
    # Determine if filtering steps should be shown (default: True, unless disabled)
    show_filtering_steps = args.show_filtering_steps or not args.no_filtering_steps
    if not args.show_filtering_steps and not args.no_filtering_steps:
        # Default behavior: show filtering steps
        show_filtering_steps = True

    # Standardized parameter precedence: CLI args > Config file > Hardcoded defaults
    # Handle hough_threshold
    if args.hough_threshold is not None:
        hough_threshold = args.hough_threshold
    else:
        hough_threshold = getattr(config, 'hough_threshold', 60)
    
    # Apply all command line parameter overrides to config
    cli_overrides = [
        ('min_line_length', args.min_line_length),
        ('max_line_gap', args.max_line_gap),
        ('horizontal_kernel_ratio', args.horizontal_kernel_ratio),
        ('vertical_kernel_ratio', args.vertical_kernel_ratio),
        ('h_length_filter_ratio', args.h_length_filter_ratio),
        ('v_length_filter_ratio', args.v_length_filter_ratio),
        ('h_erode_iterations', args.h_erode_iterations),
        ('h_dilate_iterations', args.h_dilate_iterations),
        ('v_erode_iterations', args.v_erode_iterations),
        ('v_dilate_iterations', args.v_dilate_iterations),
        ('line_merge_distance_h', args.line_merge_distance_h),
        ('line_merge_distance_v', args.line_merge_distance_v),
        ('line_extension_tolerance', args.line_extension_tolerance),
        ('max_merge_iterations', args.max_merge_iterations),
    ]
    
    for param_name, param_value in cli_overrides:
        if param_value is not None:
            setattr(config, param_name, param_value)
    
    # Collect command line arguments for parameter documentation
    command_args = {
        "min_line_length": args.min_line_length,
        "max_line_gap": args.max_line_gap,
        "hough_threshold": args.hough_threshold,
        "horizontal_kernel_ratio": args.horizontal_kernel_ratio,
        "vertical_kernel_ratio": args.vertical_kernel_ratio,
        "h_length_filter_ratio": args.h_length_filter_ratio,
        "v_length_filter_ratio": args.v_length_filter_ratio,
        "h_erode_iterations": args.h_erode_iterations,
        "h_dilate_iterations": args.h_dilate_iterations,
        "v_erode_iterations": args.v_erode_iterations,
        "v_dilate_iterations": args.v_dilate_iterations,
        "line_merge_distance_h": args.line_merge_distance_h,
        "line_merge_distance_v": args.line_merge_distance_v,
        "line_extension_tolerance": args.line_extension_tolerance,
        "max_merge_iterations": args.max_merge_iterations,
        "config_file": str(args.config_file) if args.config_file else None,
        "stage": args.stage,
        "save_debug": args.save_debug,
        "show_filtering_steps": show_filtering_steps,
    }

    # Determine config source
    config_source = "default"
    if args.config_file and args.config_file.exists():
        config_source = "file"
    cli_override_values = [args.min_line_length, args.max_line_gap, args.hough_threshold, 
                          args.horizontal_kernel_ratio, args.vertical_kernel_ratio, 
                          args.h_length_filter_ratio, args.v_length_filter_ratio,
                          args.h_erode_iterations, args.h_dilate_iterations,
                          args.v_erode_iterations, args.v_dilate_iterations,
                          args.line_merge_distance_h, args.line_merge_distance_v,
                          args.line_extension_tolerance, args.max_merge_iterations]
    if any(v is not None for v in cli_override_values):
        config_source += "_with_overrides"

    config.verbose = False  # Keep visualization quiet
    output_dir = Path(args.output_dir)

    print(f"Visualizing table line detection on {len(image_paths)} images")
    if args.test_images:
        print("Batch mode: Processing all images from test_images directory")
    print("Parameters:")
    # Note: min_line_length and max_line_gap are calculated dynamically in detect_table_lines
    print(f"  - Hough threshold: {hough_threshold}")
    print(f"  - Horizontal kernel ratio: {getattr(config, 'horizontal_kernel_ratio', 30)}")
    print(f"  - Vertical kernel ratio: {getattr(config, 'vertical_kernel_ratio', 30)}")
    print(f"  - Min table coverage: {getattr(config, 'min_table_coverage', 0.15)}")
    print(f"  - H length filter ratio: {getattr(config, 'h_length_filter_ratio', 0.6)}")
    print(f"  - V length filter ratio: {getattr(config, 'v_length_filter_ratio', 0.6)}")
    print(f"  - H erode iterations: {getattr(config, 'h_erode_iterations', 1)}")
    print(f"  - H dilate iterations: {getattr(config, 'h_dilate_iterations', 1)}")
    print(f"  - V erode iterations: {getattr(config, 'v_erode_iterations', 1)}")
    print(f"  - V dilate iterations: {getattr(config, 'v_dilate_iterations', 1)}")
    print(f"  - H line merge distance: {getattr(config, 'line_merge_distance_h', 15)}px")
    print(f"  - V line merge distance: {getattr(config, 'line_merge_distance_v', 15)}px")
    print(f"  - Line extension tolerance: {getattr(config, 'line_extension_tolerance', 20)}px")
    print(f"  - Max merge iterations: {getattr(config, 'max_merge_iterations', 3)}")
    print(f"Output directory: {output_dir}")
    print()

    # Process all images
    results = []
    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Processing: {image_path.name}")
        result = process_image_table_lines_visualization(
            image_path,
            config,
            output_dir,
            hough_threshold,
            command_args,
            config_source,
            show_filtering_steps,
        )
        results.append(result)

    # Summary
    successful_results = [r for r in results if r["success"]]
    print(f"\n{'='*60}")
    print("TABLE LINES VISUALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {len(successful_results)}/{len(image_paths)} images")

    if successful_results:
        table_structure_count = sum(
            1 for r in successful_results if r["line_info"]["has_table_structure"]
        )
        print(
            f"Images with table structure: {table_structure_count}/{len(successful_results)}"
        )

        total_h_lines = sum(r["line_info"]["h_line_count"] for r in successful_results)
        total_v_lines = sum(r["line_info"]["v_line_count"] for r in successful_results)
        print(f"Total horizontal lines: {total_h_lines}")
        print(f"Total vertical lines: {total_v_lines}")

        if table_structure_count > 0:
            avg_h_length = sum(
                r["line_info"]["h_avg_length"]
                for r in successful_results
                if r["line_info"]["h_avg_length"] > 0
            ) / max(
                1,
                sum(
                    1 for r in successful_results if r["line_info"]["h_avg_length"] > 0
                ),
            )
            avg_v_length = sum(
                r["line_info"]["v_avg_length"]
                for r in successful_results
                if r["line_info"]["v_avg_length"] > 0
            ) / max(
                1,
                sum(
                    1 for r in successful_results if r["line_info"]["v_avg_length"] > 0
                ),
            )
            print(f"Average H line length: {avg_h_length:.1f}px")
            print(f"Average V line length: {avg_v_length:.1f}px")

    print(f"\nOutput files saved to: {output_dir}")
    print(
        "Review the '_table_lines_comparison.jpg' files to assess line detection quality"
    )

    # Save summary
    summary_file = output_dir / "table_lines_visualization_summary.json"
    summary_data = {
        "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "config_parameters": {
            "min_line_length": getattr(config, 'min_line_length', 50),
            "max_line_gap": getattr(config, 'max_line_gap', 10),
            "hough_threshold": hough_threshold,
            "horizontal_kernel_ratio": getattr(config, 'horizontal_kernel_ratio', 30),
            "vertical_kernel_ratio": getattr(config, 'vertical_kernel_ratio', 30),
            "min_table_coverage": getattr(config, 'min_table_coverage', 0.15),
            "max_parallel_distance": getattr(config, 'max_parallel_distance', 12),
            "angle_tolerance": getattr(config, 'angle_tolerance', 5.0),
            "h_length_filter_ratio": getattr(config, 'h_length_filter_ratio', 0.6),
            "v_length_filter_ratio": getattr(config, 'v_length_filter_ratio', 0.6),
            "h_erode_iterations": getattr(config, 'h_erode_iterations', 1),
            "h_dilate_iterations": getattr(config, 'h_dilate_iterations', 1),
            "v_erode_iterations": getattr(config, 'v_erode_iterations', 1),
            "v_dilate_iterations": getattr(config, 'v_dilate_iterations', 1),
            "line_merge_distance_h": getattr(config, 'line_merge_distance_h', 15),
            "line_merge_distance_v": getattr(config, 'line_merge_distance_v', 15),
            "line_extension_tolerance": getattr(config, 'line_extension_tolerance', 20),
            "max_merge_iterations": getattr(config, 'max_merge_iterations', 3),
        },
        "results": results,
    }

    with open(summary_file, "w") as f:
        json.dump(convert_numpy_types(summary_data), f, indent=2)

    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
