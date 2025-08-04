#!/usr/bin/env python3
"""
Margin Removal Visualization Script (Version 2)

This version uses the new processor architecture for better maintainability.
Supports multiple margin removal methods: aggressive, bounding_box, and comparison mode.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import sys
from typing import Dict, Any, List, Tuple, Optional

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.ocr_pipeline.processor_wrappers import MarginRemovalProcessor
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


def calculate_margins(analysis: Dict[str, Any]) -> Dict[str, int]:
    """Calculate margins removed from analysis data."""
    if "crop_bounds" in analysis and "original_shape" in analysis:
        x, y, w, h = analysis["crop_bounds"]
        orig_h, orig_w = analysis["original_shape"][:2]
        return {
            'left': x,
            'top': y,
            'right': orig_w - (x + w),
            'bottom': orig_h - (y + h)
        }
    return {'left': 0, 'top': 0, 'right': 0, 'bottom': 0}


def draw_margins_overlay(
    image: np.ndarray, analysis: Dict[str, Any], method: str = "aggressive"
) -> np.ndarray:
    """Draw margin detection overlay on the image."""
    overlay = image.copy()
    height, width = image.shape[:2]

    # Draw content region
    if method == "bounding_box":
        # For bounding box method
        x, y, w, h = analysis.get("bbox", (0, 0, width, height))
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Draw margin areas
        if x > 0:  # Left margin
            cv2.rectangle(overlay, (0, 0), (x, height), (0, 0, 255), -1)
            overlay[:, :x] = cv2.addWeighted(image[:, :x], 0.7, overlay[:, :x], 0.3, 0)
        if y > 0:  # Top margin
            cv2.rectangle(overlay, (0, 0), (width, y), (0, 0, 255), -1)
            overlay[:y, :] = cv2.addWeighted(image[:y, :], 0.7, overlay[:y, :], 0.3, 0)
        if x + w < width:  # Right margin
            cv2.rectangle(overlay, (x + w, 0), (width, height), (0, 0, 255), -1)
            overlay[:, x + w:] = cv2.addWeighted(image[:, x + w:], 0.7, overlay[:, x + w:], 0.3, 0)
        if y + h < height:  # Bottom margin
            cv2.rectangle(overlay, (0, y + h), (width, height), (0, 0, 255), -1)
            overlay[y + h:, :] = cv2.addWeighted(image[y + h:, :], 0.7, overlay[y + h:, :], 0.3, 0)
            
    elif method == "smart":
        # For smart asymmetric method
        if "boundaries" in analysis:
            boundaries = analysis["boundaries"]
            top, bottom = boundaries["top"], boundaries["bottom"]
            left, right = boundaries["left"], boundaries["right"]
            
            # Draw the detected content rectangle
            cv2.rectangle(overlay, (left, top), (right, bottom), (0, 255, 0), 3)
            
            # Draw margin areas in red with transparency
            margin_overlay = np.zeros_like(overlay)
            margin_overlay[:, :] = (0, 0, 255)
            
            # Top margin
            if top > 0:
                overlay[:top, :] = cv2.addWeighted(image[:top, :], 0.7, margin_overlay[:top, :], 0.3, 0)
            # Bottom margin  
            if bottom < height - 1:
                overlay[bottom:, :] = cv2.addWeighted(image[bottom:, :], 0.7, margin_overlay[bottom:, :], 0.3, 0)
            # Left margin
            if left > 0:
                overlay[:, :left] = cv2.addWeighted(image[:, :left], 0.7, margin_overlay[:, :left], 0.3, 0)
            # Right margin
            if right < width - 1:
                overlay[:, right:] = cv2.addWeighted(image[:, right:], 0.7, margin_overlay[:, right:], 0.3, 0)
                
            # Draw projection histograms if available
            if "projections" in analysis:
                proj = analysis["projections"]
                # Draw horizontal projection (scaled)
                h_proj = proj["normalized_horizontal"]
                for i, val in enumerate(h_proj):
                    if i < height:
                        bar_length = int(val * 50)  # Scale for visibility
                        cv2.line(overlay, (width - 60, i), (width - 60 + bar_length, i), (255, 255, 0), 1)
                
                # Draw vertical projection (scaled)
                v_proj = proj["normalized_vertical"]
                for i, val in enumerate(v_proj):
                    if i < width:
                        bar_length = int(val * 50)  # Scale for visibility
                        cv2.line(overlay, (i, height - 60), (i, height - 60 + bar_length), (255, 255, 0), 1)
                        
    elif method == "curved_black_background":
        # For curved black background method
        if "page_contour" in analysis and analysis["page_contour"]:
            # Convert back from list to numpy array
            contour = np.array(analysis["page_contour"], dtype=np.int32)
            
            # Draw the detected page contour
            cv2.drawContours(overlay, [contour], 0, (0, 255, 0), 3)
            
            # Draw bounding rectangle
            if "crop_bounds" in analysis:
                x, y, w, h = analysis["crop_bounds"]
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 255), 2)
            
            # Highlight black areas that will be removed/filled
            black_threshold = analysis.get("black_threshold", 30)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            black_mask = gray < black_threshold
            
            # Show black areas in red with transparency
            black_overlay = np.zeros_like(overlay)
            black_overlay[:, :] = (0, 0, 255)
            overlay = np.where(black_mask[..., None], 
                             cv2.addWeighted(image, 0.6, black_overlay, 0.4, 0),
                             overlay)
            
            # Show detected page background color
            if "page_background_color" in analysis:
                bg_color = analysis["page_background_color"]
                # Draw a color sample in the corner
                cv2.rectangle(overlay, (width - 100, 10), (width - 10, 60), bg_color, -1)
                cv2.rectangle(overlay, (width - 100, 10), (width - 10, 60), (255, 255, 255), 2)
                cv2.putText(overlay, "Page", (width - 95, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(overlay, "Color", (width - 95, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
    else:
        # For aggressive method (inscribed rectangle)
        if "padded_rect" in analysis:
            rect = analysis["padded_rect"]
            if rect is not None:
                # Draw the inscribed rectangle
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(overlay, [box], 0, (0, 255, 0), 3)
                
                # Draw a semi-transparent overlay for margins
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [box], 0, 255, -1)
                margin_mask = cv2.bitwise_not(mask)
                margin_overlay = np.zeros_like(overlay)
                margin_overlay[:, :] = (0, 0, 255)
                overlay = np.where(margin_mask[..., None], 
                                 cv2.addWeighted(image, 0.7, margin_overlay, 0.3, 0),
                                 overlay)

    # Add text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    # Method name
    cv2.putText(
        overlay,
        f"Method: {method}",
        (10, 30),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
    )

    # Area retention
    area_retention = analysis.get("area_retention", 0) * 100  # Convert to percentage
    color = (0, 255, 0) if area_retention > 80 else (0, 255, 255) if area_retention > 50 else (0, 0, 255)
    cv2.putText(
        overlay,
        f"Area retained: {area_retention:.1f}%",
        (10, 60),
        font,
        font_scale,
        color,
        thickness,
    )

    # Content detection info
    if "content_area_ratio" in analysis:
        cv2.putText(
            overlay,
            f"Content ratio: {analysis['content_area_ratio']:.1%}",
            (10, 90),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    # Calculate margins removed
    if "crop_bounds" in analysis and "original_shape" in analysis:
        x, y, w, h = analysis["crop_bounds"]
        orig_h, orig_w = analysis["original_shape"][:2]
        margins = {
            'left': x,
            'top': y,
            'right': orig_w - (x + w),
            'bottom': orig_h - (y + h)
        }
    else:
        margins = {'left': 0, 'top': 0, 'right': 0, 'bottom': 0}
    margin_text = f"Margins: L:{margins['left']} T:{margins['top']} R:{margins['right']} B:{margins['bottom']}"
    cv2.putText(
        overlay,
        margin_text,
        (10, 120),
        font,
        0.6,
        (255, 255, 255),
        1,
    )

    return overlay


def process_image_single_method(
    image_path: Path,
    processor: MarginRemovalProcessor,
    output_dir: Path,
    method: str = "aggressive",
    save_cropped: bool = True,
    save_debug: bool = False,
    command_args: Dict[str, Any] = None,
    config_source: str = "default",
) -> Dict[str, Any]:
    """Process a single image with one margin removal method."""
    print(f"  Method: {method}")
    
    try:
        # Load image
        from src.ocr_pipeline.processors import load_image
        image = load_image(image_path)
        
        # Process with margin removal
        processing_params = {
            'return_analysis': True,
        }
        
        # Add method-specific parameters
        if method == "bounding_box":
            processing_params.update({
                'expansion_factor': command_args.get('expansion_factor', 0.0),
                'use_min_area_rect': command_args.get('use_min_area_rect', False),
            })
        
        # Check if we should use optimized version
        use_optimized = command_args.get('use_optimized', False) or method == "fast"
        if use_optimized:
            # Import optimized utils
            from src.ocr_pipeline import utils_optimized
            if method == "bounding_box":
                result = utils_optimized.remove_margin_bounding_box_optimized(image, **processing_params)
            else:
                result = utils_optimized.remove_margin_aggressive_optimized(image, **processing_params)
        else:
            # Use standard processor
            result = processor.process(image, method=method, **processing_params)
        
        if isinstance(result, tuple):
            cropped_image, analysis = result
        else:
            cropped_image = result
            analysis = {}
        
        # Create visualizations
        base_name = image_path.stem
        method_suffix = "_fast" if use_optimized else ""
        
        # Draw overlay
        overlay_image = draw_margins_overlay(image, analysis, method)
        overlay_path = output_dir / f"{base_name}_margin_{method}{method_suffix}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay_image)
        
        output_files = {
            "overlay": str(overlay_path),
        }
        
        # Save cropped image if requested
        if save_cropped:
            cropped_path = output_dir / f"{base_name}_margin_{method}{method_suffix}_cropped.jpg"
            cv2.imwrite(str(cropped_path), cropped_image)
            output_files["cropped"] = str(cropped_path)
            
            # Save to processed_images directory for pipeline
            processed_dir = output_dir / "processed_images"
            processed_dir.mkdir(exist_ok=True)
            processed_path = processed_dir / f"{base_name}.jpg"
            cv2.imwrite(str(processed_path), cropped_image)
        
        # Save debug images if requested
        if save_debug and "debug_images" in analysis:
            debug_dir = output_dir / "debug" / base_name / method
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            for debug_name, debug_image in analysis["debug_images"].items():
                if isinstance(debug_image, np.ndarray):
                    debug_path = debug_dir / f"{debug_name}.png"
                    cv2.imwrite(str(debug_path), debug_image)
                    output_files[f"debug_{debug_name}"] = str(debug_path)
        
        return {
            "method": method,
            "success": True,
            "output_files": output_files,
            "area_retention": analysis.get("area_retention", 0) * 100,  # Convert to percentage
            "margins_removed": calculate_margins(analysis),
            "content_ratio": analysis.get("content_area_ratio", 0),
        }
        
    except Exception as e:
        print(f"    ERROR: {e}")
        return {
            "method": method,
            "success": False,
            "error": str(e),
        }


def process_image(
    image_path: Path,
    processor: MarginRemovalProcessor,
    output_dir: Path,
    methods: List[str],
    save_cropped: bool = True,
    save_debug: bool = False,
    command_args: Dict[str, Any] = None,
    config_source: str = "default",
) -> Dict[str, Any]:
    """Process a single image with specified margin removal methods."""
    print(f"Processing: {image_path.name}")
    
    results_by_method = {}
    
    # Process each method
    for method in methods:
        result = process_image_single_method(
            image_path,
            processor,
            output_dir,
            method,
            save_cropped,
            save_debug,
            command_args,
            config_source,
        )
        results_by_method[method] = result
    
    # Create comparison visualization if multiple methods
    if len(methods) > 1:
        try:
            from src.ocr_pipeline.processors import load_image
            image = load_image(image_path)
            
            # Create side-by-side comparison
            overlays = []
            for method in methods:
                if results_by_method[method]["success"]:
                    overlay_path = results_by_method[method]["output_files"]["overlay"]
                    overlay = cv2.imread(overlay_path)
                    if overlay is not None:
                        overlays.append(overlay)
            
            if len(overlays) > 1:
                # Stack horizontally
                comparison = np.hstack(overlays)
                comparison_path = output_dir / f"{image_path.stem}_margin_comparison.jpg"
                cv2.imwrite(str(comparison_path), comparison)
                
        except Exception as e:
            print(f"  Warning: Failed to create comparison: {e}")
    
    # Aggregate results
    all_success = all(r["success"] for r in results_by_method.values())
    best_method = None
    best_retention = 0
    
    for method, result in results_by_method.items():
        if result["success"] and result.get("area_retention", 0) > best_retention:
            best_retention = result["area_retention"]
            best_method = method
    
    # Save parameter documentation
    param_file = save_step_parameters(
        step_name="margin_removal_v2",
        config_obj=processor.config,
        command_args=command_args,
        processing_results={
            "image_name": image_path.name,
            "success": all_success,
            "methods": methods,
            "results_by_method": results_by_method,
            "best_method": best_method,
            "best_retention": best_retention,
        },
        output_dir=output_dir,
        config_source=config_source,
    )
    
    result = {
        "image_name": image_path.name,
        "success": all_success,
        "methods": methods,
        "results_by_method": results_by_method,
        "best_method": best_method,
        "best_retention": best_retention,
        "parameter_file": str(param_file) if param_file else None,
    }
    
    if all_success:
        print(f"  SUCCESS: Best method={best_method} ({best_retention:.1f}% retention)")
    
    return result


def main():
    """Main function for margin removal visualization."""
    parser = argparse.ArgumentParser(
        description="Margin removal visualization (Version 2)"
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
        default="data/output/visualization/margin_removal_v2",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--save-cropped",
        action="store_true",
        default=True,
        help="Save cropped images",
    )
    parser.add_argument(
        "--no-cropped",
        dest="save_cropped",
        action="store_false",
        help="Don't save cropped images",
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save debug visualizations",
    )
    
    # Add configuration arguments
    add_config_arguments(parser, 'margin_removal')
    add_processor_specific_arguments(parser, 'margin_removal')
    
    # Additional method selection arguments
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all methods",
    )
    parser.add_argument(
        "--use-optimized",
        action="store_true",
        help="Use optimized implementations",
    )
    
    args = parser.parse_args()
    
    # Determine which methods to use
    if args.compare:
        methods = ["inscribed", "aggressive", "bounding_box"]
    else:
        methods = [args.method]
    
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
    config, config_source = load_config(args, Stage1Config, 'margin_removal')
    
    # Create processor
    processor = MarginRemovalProcessor(config)
    
    # Get command arguments
    command_args = get_command_args_dict(args, 'margin_removal')
    command_args['use_optimized'] = args.use_optimized
    
    output_dir = Path(args.output_dir)
    
    print(f"Margin Removal Visualization (Version 2)")
    print(f"Processing {len(image_paths)} images")
    print(f"Methods: {', '.join(methods)}")
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
            methods,
            args.save_cropped,
            args.save_debug,
            command_args,
            config_source,
        )
        results.append(result)
    
    # Summary
    successful_results = [r for r in results if r["success"]]
    print(f"\n{'='*60}")
    print("MARGIN REMOVAL SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {len(successful_results)}/{len(image_paths)} images")
    
    if successful_results:
        # Method performance summary
        method_stats = {}
        for method in methods:
            method_results = [
                r["results_by_method"][method]["area_retention"]
                for r in successful_results
                if method in r["results_by_method"] and r["results_by_method"][method]["success"]
            ]
            if method_results:
                method_stats[method] = {
                    "avg_retention": sum(method_results) / len(method_results),
                    "min_retention": min(method_results),
                    "max_retention": max(method_results),
                }
        
        print("\nMethod Performance:")
        for method, stats in method_stats.items():
            print(f"  {method}:")
            print(f"    Average retention: {stats['avg_retention']:.1f}%")
            print(f"    Range: {stats['min_retention']:.1f}% - {stats['max_retention']:.1f}%")
    
    print(f"\nOutput files saved to: {output_dir}")
    
    # Save summary
    summary_file = output_dir / "margin_removal_visualization_summary.json"
    summary_data = {
        "processor_version": "v2",
        "architecture": "processor_based",
        "config_source": config_source,
        "command_args": command_args,
        "methods": methods,
        "results": results,
    }
    
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(convert_numpy_types(summary_data), f, indent=2)
    
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()