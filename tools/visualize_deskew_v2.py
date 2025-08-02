#!/usr/bin/env python3
"""
Deskew Visualization Script (Version 2)

This version uses the new processor architecture for better maintainability.
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

from src.ocr_pipeline.processor_wrappers import DeskewProcessor
from src.ocr_pipeline.config import Stage1Config
from config_utils import (
    load_config, 
    add_config_arguments, 
    add_processor_specific_arguments,
    get_command_args_dict
)
from output_manager import (
    get_test_images,
    convert_numpy_types,
    save_step_parameters,
)


def analyze_skew_detailed(
    image: np.ndarray, 
    processor: DeskewProcessor,
    angle_range: Tuple[float, float] = (-5, 5),
    angle_step: float = 0.1,
    method: str = "projection",
) -> Dict[str, Any]:
    """Enhanced skew analysis with detailed angle detection."""
    # Use processor to get deskew with angle
    deskewed, angle = processor.process(
        image,
        angle_range=angle_range,
        angle_step=angle_step,
        method=method,
        return_angle=True
    )
    
    # Create analysis dict
    analysis = {
        "detected_angle": angle,
        "angle_range": angle_range,
        "angle_step": angle_step,
        "method": method,
    }
    
    return analysis


def draw_deskew_overlay(image: np.ndarray, analysis: Dict[str, Any]) -> np.ndarray:
    """Draw deskew analysis overlay on the image."""
    overlay = image.copy()
    height, width = image.shape[:2]
    
    angle = analysis.get("detected_angle", 0.0)
    
    # Draw grid lines to show skew
    grid_spacing = 100
    line_color = (0, 255, 0) if abs(angle) < 0.5 else (0, 255, 255) if abs(angle) < 2 else (0, 0, 255)
    
    # Vertical lines
    for x in range(0, width, grid_spacing):
        cv2.line(overlay, (x, 0), (x, height), line_color, 1)
    
    # Horizontal lines - these will show the skew most clearly
    for y in range(0, height, grid_spacing):
        cv2.line(overlay, (0, y), (width, y), line_color, 1)
    
    # Draw rotation center
    center_x, center_y = width // 2, height // 2
    cv2.circle(overlay, (center_x, center_y), 5, (255, 0, 0), -1)
    cv2.circle(overlay, (center_x, center_y), 10, (255, 0, 0), 2)
    
    # Draw angle arc
    if abs(angle) > 0.1:
        arc_radius = 100
        start_angle = 0
        end_angle = int(angle * 10)  # Scale for visibility
        cv2.ellipse(overlay, (center_x, center_y), (arc_radius, arc_radius),
                   0, start_angle, end_angle, (255, 0, 255), 2)
    
    # Add text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # Detected angle
    angle_color = (0, 255, 0) if abs(angle) < 0.5 else (0, 255, 255) if abs(angle) < 2 else (0, 0, 255)
    cv2.putText(
        overlay,
        f"Skew angle: {angle:.2f}°",
        (10, 30),
        font,
        font_scale,
        angle_color,
        thickness,
    )
    
    # Method used
    method = analysis.get("method", "unknown")
    cv2.putText(
        overlay,
        f"Method: {method}",
        (10, 60),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
    )
    
    # Confidence if available
    if "confidence" in analysis:
        cv2.putText(
            overlay,
            f"Confidence: {analysis['confidence']:.1%}",
            (10, 90),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )
    
    # Search range
    angle_range = analysis.get("angle_range", (-5, 5))
    cv2.putText(
        overlay,
        f"Search range: {angle_range[0]}° to {angle_range[1]}°",
        (10, 120),
        font,
        0.6,
        (255, 255, 255),
        1,
    )
    
    return overlay


def create_angle_plot(analysis: Dict[str, Any], output_path: Path):
    """Create a plot showing angle vs score if available."""
    if "angle_scores" not in analysis:
        return False
    
    try:
        import matplotlib.pyplot as plt
        
        angles = analysis["angle_scores"]["angles"]
        scores = analysis["angle_scores"]["scores"]
        detected_angle = analysis.get("detected_angle", 0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(angles, scores, 'b-', linewidth=2)
        plt.axvline(x=detected_angle, color='r', linestyle='--', 
                   label=f'Detected: {detected_angle:.2f}°')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Score')
        plt.title('Deskew Angle Detection')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"Warning: Failed to create angle plot: {e}")
        return False


def process_image(
    image_path: Path,
    processor: DeskewProcessor,
    output_dir: Path,
    save_deskewed: bool = True,
    save_debug: bool = False,
    no_params: bool = False,
    command_args: Dict[str, Any] = None,
    config_source: str = "default",
) -> Dict[str, Any]:
    """Process a single image with deskew visualization."""
    print(f"Processing: {image_path.name}")
    
    try:
        # Load image
        import src.ocr_pipeline.utils as ocr_utils
        image = ocr_utils.load_image(image_path)
        
        # Get processing parameters
        angle_range = command_args.get('angle_range', (-5, 5))
        angle_step = command_args.get('angle_step', 0.1)
        method = command_args.get('method', 'projection')
        
        # Analyze skew
        analysis = analyze_skew_detailed(
            image, processor, angle_range, angle_step, method
        )
        
        # Process with deskew
        deskewed_image, angle = processor.process(
            image, 
            angle_range=angle_range,
            angle_step=angle_step,
            method=method,
            return_angle=True
        )
        
        # Create visualizations
        base_name = image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Draw overlay
        overlay_image = draw_deskew_overlay(image, analysis)
        overlay_path = output_dir / f"{base_name}_deskew_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay_image)
        
        output_files = {
            "overlay": str(overlay_path),
        }
        
        # Save deskewed image if requested
        if save_deskewed:
            deskewed_path = output_dir / f"{base_name}_deskewed.jpg"
            cv2.imwrite(str(deskewed_path), deskewed_image)
            output_files["deskewed"] = str(deskewed_path)
            
            # Save to processed_images directory for pipeline
            processed_dir = output_dir / "processed_images"
            processed_dir.mkdir(exist_ok=True)
            processed_path = processed_dir / f"{base_name}.jpg"
            cv2.imwrite(str(processed_path), deskewed_image)
        
        # Create angle plot if debug enabled
        if save_debug:
            debug_dir = output_dir / "debug" / base_name
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Angle plot
            plot_path = debug_dir / "angle_analysis.png"
            if create_angle_plot(analysis, plot_path):
                output_files["angle_plot"] = str(plot_path)
            
            # Before/after comparison
            comparison = np.hstack([image, deskewed_image])
            comparison_path = debug_dir / "before_after.jpg"
            cv2.imwrite(str(comparison_path), comparison)
            output_files["comparison"] = str(comparison_path)
        
        # Save parameter documentation (unless --no-params is set)
        param_file = None
        if not no_params:
            param_file = save_step_parameters(
                step_name="deskew_v2",
                config_obj=processor.config,
                command_args=command_args,
                processing_results={
                    "image_name": image_path.name,
                    "success": True,
                    "output_files": output_files,
                    "detected_angle": angle,
                    "method": method,
                    "angle_range": angle_range,
                    "angle_step": angle_step,
                },
                output_dir=output_dir,
                config_source=config_source,
            )
            
            if param_file:
                output_files["parameters"] = str(param_file)
        
        result = {
            "image_name": image_path.name,
            "success": True,
            "output_files": output_files,
            "parameter_file": str(param_file) if param_file else None,
            "detected_angle": angle,
            "needs_deskew": abs(angle) > 0.5,
        }
        
        print(f"  SUCCESS: Angle={angle:.2f}°, Needs deskew={abs(angle) > 0.5}")
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "image_name": image_path.name,
            "success": False,
            "error": str(e),
        }


def main():
    """Main function for deskew visualization."""
    parser = argparse.ArgumentParser(
        description="Deskew visualization (Version 2)"
    )
    
    # Image input arguments
    parser.add_argument(
        "images",
        nargs="*",
        help="Images to process",
    )
    parser.add_argument(
        "--test-images",
        action="store_true",
        help="Process all images in test_images directory",
    )
    parser.add_argument(
        "--input-dir",
        help="Input directory containing images (for pipeline mode)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/output/visualization/deskew_v2",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--save-deskewed",
        action="store_true",
        default=True,
        help="Save deskewed images",
    )
    parser.add_argument(
        "--no-deskewed",
        dest="save_deskewed",
        action="store_false",
        help="Don't save deskewed images",
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save debug visualizations (angle plots, comparisons)",
    )
    parser.add_argument(
        "--no-params",
        action="store_true",
        help="Don't save parameter files (for pipeline mode)",
    )
    
    # Add configuration arguments
    add_config_arguments(parser, 'deskew')
    add_processor_specific_arguments(parser, 'deskew')
    
    # Deskew-specific arguments
    parser.add_argument(
        "--method",
        choices=["projection", "hough", "fourier"],
        default="projection",
        help="Deskew detection method",
    )
    
    args = parser.parse_args()
    
    # Determine which images to process
    if args.input_dir:
        print(f"Using pipeline input directory: {args.input_dir}")
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
        print("Using batch mode: processing all images in test_images/")
        image_paths = get_test_images()
        if not image_paths:
            print("No images found in test_images directory!")
            return
    else:
        # Use individual image arguments
        image_paths = []
        for img_path in args.images:
            path = Path(img_path)
            if path.exists():
                image_paths.append(path)
            else:
                print(f"Warning: {img_path} not found, skipping")
        
        if not image_paths:
            print("No valid images found!")
            parser.print_help()
            return
    
    # Load configuration
    config, config_source = load_config(args, Stage1Config, 'deskew')
    
    # Create processor
    processor = DeskewProcessor(config)
    
    # Get command arguments
    command_args = get_command_args_dict(args, 'deskew')
    command_args['method'] = args.method
    command_args['angle_range'] = args.angle_range
    command_args['angle_step'] = args.angle_step
    
    output_dir = Path(args.output_dir)
    
    print(f"Deskew Visualization (Version 2)")
    print(f"Processing {len(image_paths)} images")
    print(f"Method: {args.method}")
    print(f"Configuration source: {config_source}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Process all images
    results = []
    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] {image_path.name}")
        result = process_image(
            image_path,
            processor,
            output_dir,
            args.save_deskewed,
            args.save_debug,
            args.no_params,
            command_args,
            config_source,
        )
        results.append(result)
    
    # Summary
    successful_results = [r for r in results if r["success"]]
    print(f"\n{'='*60}")
    print("DESKEW SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {len(successful_results)}/{len(image_paths)} images")
    
    if successful_results:
        needs_deskew = [r for r in successful_results if r.get("needs_deskew", False)]
        print(f"Images needing deskew (>0.5°): {len(needs_deskew)}")
        
        angles = [abs(r.get("detected_angle", 0)) for r in successful_results]
        if angles:
            avg_angle = sum(angles) / len(angles)
            max_angle = max(angles)
            print(f"Average skew: {avg_angle:.2f}°")
            print(f"Maximum skew: {max_angle:.2f}°")
            
            # Show images with significant skew
            significant_skew = [
                r for r in successful_results 
                if abs(r.get("detected_angle", 0)) > 1.0
            ]
            if significant_skew:
                print(f"\nImages with significant skew (>1°):")
                for r in significant_skew[:5]:  # Show top 5
                    print(f"  {r['image_name']}: {r['detected_angle']:.2f}°")
                if len(significant_skew) > 5:
                    print(f"  ... and {len(significant_skew) - 5} more")
    
    print(f"\nOutput files saved to: {output_dir}")
    
    # Save summary
    summary_file = output_dir / "deskew_visualization_summary.json"
    summary_data = {
        "processor_version": "v2",
        "architecture": "processor_based",
        "config_source": config_source,
        "command_args": command_args,
        "results": results,
    }
    
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(convert_numpy_types(summary_data), f, indent=2)
    
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()