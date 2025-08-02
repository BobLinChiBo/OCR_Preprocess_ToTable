#!/usr/bin/env python3
"""
Margin Removal Bounding Box Visualization Script

This script visualizes margin removal using the fast bounding box approach,
which is much faster than the inscribed rectangle method.
"""

import warnings
warnings.warn(
    "This script is deprecated and will be removed in a future version. "
    "Please use visualize_margin_removal_v2.py with --method bounding_box or run_visualizations.py with --use-v2 flag.",
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

from src.ocr_pipeline.config import Config, Stage1Config
import src.ocr_pipeline.utils as ocr_utils
from src.ocr_pipeline import utils_optimized
from output_manager import (
    get_test_images,
    convert_numpy_types,
    save_step_parameters,
)


def process_image_margin_removal_bbox(
    image_path: Path,
    config: Config,
    output_dir: Path,
    save_debug: bool = False,
    command_args: Dict[str, Any] = None,
    config_source: str = "default",
    use_optimized: bool = True,
    downsample_factor: float = 0.25,
    expansion_factor: float = 0.0,
    use_min_area_rect: bool = False,
    compare_methods: bool = False,
) -> Dict[str, Any]:
    """Process a single image with bounding box margin removal."""
    print(f"Processing: {image_path.name}")
    
    try:
        # Load image
        image = ocr_utils.load_image(image_path)
        
        start_time = time.time()
        
        # Apply margin removal
        if use_optimized:
            margin_removed, analysis = utils_optimized.remove_margin_bounding_box_optimized(
                image,
                blur_kernel_size=config.blur_kernel_size,
                black_threshold=config.black_threshold,
                content_threshold=config.content_threshold,
                morph_kernel_size=config.morph_kernel_size,
                min_content_area_ratio=config.min_content_area_ratio,
                padding=config.margin_padding,
                expansion_factor=expansion_factor,
                use_min_area_rect=use_min_area_rect,
                downsample_factor=downsample_factor,
                return_analysis=True,
            )
        else:
            margin_removed, analysis = ocr_utils.remove_margin_bounding_box(
                image,
                blur_kernel_size=config.blur_kernel_size,
                black_threshold=config.black_threshold,
                content_threshold=config.content_threshold,
                morph_kernel_size=config.morph_kernel_size,
                min_content_area_ratio=config.min_content_area_ratio,
                padding=config.margin_padding,
                expansion_factor=expansion_factor,
                use_min_area_rect=use_min_area_rect,
                return_analysis=True,
            )
        
        bbox_time = time.time() - start_time
        
        # Save outputs
        base_name = image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the margin-removed image
        margin_removed_path = output_dir / f"{base_name}_margin_removed_bbox.jpg"
        cv2.imwrite(str(margin_removed_path), margin_removed)
        
        output_files = {
            "margin_removed_bbox": str(margin_removed_path),
        }
        
        # Compare with inscribed rectangle method if requested
        comparison_data = {}
        if compare_methods:
            print("  Comparing with inscribed rectangle method...")
            start_time = time.time()
            
            margin_removed_inscribed, analysis_inscribed = ocr_utils.remove_margin_aggressive(
                image,
                blur_kernel_size=config.blur_kernel_size,
                black_threshold=config.black_threshold,
                content_threshold=config.content_threshold,
                morph_kernel_size=config.morph_kernel_size,
                min_content_area_ratio=config.min_content_area_ratio,
                padding=config.margin_padding,
                return_analysis=True,
            )
            
            inscribed_time = time.time() - start_time
            
            # Save inscribed rectangle result
            inscribed_path = output_dir / f"{base_name}_margin_removed_inscribed.jpg"
            cv2.imwrite(str(inscribed_path), margin_removed_inscribed)
            output_files["margin_removed_inscribed"] = str(inscribed_path)
            
            # Create comparison visualization
            comparison_img = create_comparison_visualization(
                image, margin_removed, margin_removed_inscribed,
                analysis, analysis_inscribed
            )
            comparison_path = output_dir / f"{base_name}_comparison.jpg"
            cv2.imwrite(str(comparison_path), comparison_img)
            output_files["comparison"] = str(comparison_path)
            
            comparison_data = {
                "bbox_time": bbox_time,
                "inscribed_time": inscribed_time,
                "speedup": inscribed_time / bbox_time if bbox_time > 0 else 0,
                "bbox_area_retention": analysis.get("area_retention", 0),
                "inscribed_area_retention": analysis_inscribed.get("area_retention", 0),
            }
        
        # Create processed_images subdirectory for pipeline use
        processed_dir = output_dir / "processed_images"
        processed_dir.mkdir(exist_ok=True)
        
        # Clean up filename
        clean_base_name = base_name
        for suffix in ['_left_page', '_right_page', '_split', '_deskewed']:
            if clean_base_name.endswith(suffix):
                clean_base_name = clean_base_name[:-len(suffix)]
                break
        
        processed_filename = f"{clean_base_name}_margin_removed.jpg"
        processed_path = processed_dir / processed_filename
        cv2.imwrite(str(processed_path), margin_removed)
        output_files["processed"] = str(processed_path)
        
        # Save parameter documentation
        if command_args is None:
            command_args = {}
            
        param_file = save_step_parameters(
            step_name="margin_removal_bbox",
            config_obj=config,
            command_args=command_args,
            processing_results={
                "image_name": image_path.name,
                "success": True,
                "output_files": output_files,
                "processing_time": bbox_time,
                "method": analysis.get("method", "unknown"),
                "comparison": comparison_data if compare_methods else None,
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
            "area_retention": analysis.get("area_retention", 0),
            "processing_time": bbox_time,
            "method": analysis.get("method", "unknown"),
            "comparison": comparison_data,
        }
        
        area_retention = analysis.get("area_retention", 0)
        print(f"  SUCCESS: Area retained: {area_retention:.1%} (Time: {bbox_time:.3f}s)")
        if compare_methods:
            print(f"  Speedup vs inscribed: {comparison_data['speedup']:.1f}x")
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "image_name": image_path.name,
            "success": False,
            "error": str(e),
        }


def create_comparison_visualization(
    original: np.ndarray,
    bbox_result: np.ndarray,
    inscribed_result: np.ndarray,
    bbox_analysis: Dict[str, Any],
    inscribed_analysis: Dict[str, Any],
) -> np.ndarray:
    """Create a side-by-side comparison of the two methods."""
    # Calculate dimensions
    h, w = original.shape[:2]
    
    # Create visualization showing crop boundaries on original
    vis_bbox = original.copy()
    vis_inscribed = original.copy()
    
    # Draw bounding box crop area
    bbox_bounds = bbox_analysis.get("crop_bounds", (0, 0, w, h))
    x1, y1, w1, h1 = bbox_bounds
    cv2.rectangle(vis_bbox, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 3)
    cv2.putText(vis_bbox, "Bounding Box", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(vis_bbox, f"Area: {bbox_analysis['area_retention']:.1%}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Draw inscribed rectangle crop area
    inscribed_bounds = inscribed_analysis.get("crop_bounds", (0, 0, w, h))
    x2, y2, w2, h2 = inscribed_bounds
    cv2.rectangle(vis_inscribed, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 3)
    cv2.putText(vis_inscribed, "Inscribed Rect", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(vis_inscribed, f"Area: {inscribed_analysis['area_retention']:.1%}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # Create side-by-side comparison
    comparison = np.hstack([vis_bbox, vis_inscribed])
    
    return comparison


def main():
    """Main function for bounding box margin removal visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize margin removal using fast bounding box method"
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Images to visualize",
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
        default="data/output/visualization/margin_removal_bbox",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--config",
        help="Path to config file",
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save debug images",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable optimizations (use standard algorithm)",
    )
    parser.add_argument(
        "--downsample-factor",
        type=float,
        default=0.25,
        help="Downsample factor for content detection (0.25 = 1/4 resolution)",
    )
    parser.add_argument(
        "--expansion-factor",
        type=float,
        default=0.0,
        help="Factor to expand the bounding box (0.1 = 10%% expansion)",
    )
    parser.add_argument(
        "--use-min-area-rect",
        action="store_true",
        help="Use minimum area rectangle (can be rotated) instead of axis-aligned box",
    )
    parser.add_argument(
        "--compare-methods",
        action="store_true",
        help="Compare with inscribed rectangle method",
    )
    
    # Parameter options
    parser.add_argument("--blur-kernel-size", type=int, default=9)
    parser.add_argument("--black-threshold", type=int, default=45)
    parser.add_argument("--content-threshold", type=int, default=180)
    parser.add_argument("--morph-kernel-size", type=int, default=30)
    parser.add_argument("--min-content-area-ratio", type=float, default=0.005)
    parser.add_argument("--margin-padding", type=int, default=10)
    
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
            return
    
    # Load configuration
    if args.config:
        config = Stage1Config.from_json(Path(args.config))
        config_source = f"file:{args.config}"
    else:
        # Create config from command line arguments
        config = Stage1Config(
            blur_kernel_size=args.blur_kernel_size,
            black_threshold=args.black_threshold,
            content_threshold=args.content_threshold,
            morph_kernel_size=args.morph_kernel_size,
            min_content_area_ratio=args.min_content_area_ratio,
            margin_padding=args.margin_padding,
            enable_margin_removal=True,
            verbose=False,
        )
        config_source = "command_line"
    
    # Collect command line arguments
    command_args = {
        "blur_kernel_size": args.blur_kernel_size,
        "black_threshold": args.black_threshold,
        "content_threshold": args.content_threshold,
        "morph_kernel_size": args.morph_kernel_size,
        "min_content_area_ratio": args.min_content_area_ratio,
        "margin_padding": args.margin_padding,
        "save_debug": args.save_debug,
        "optimized": not args.no_optimize,
        "downsample_factor": args.downsample_factor,
        "expansion_factor": args.expansion_factor,
        "use_min_area_rect": args.use_min_area_rect,
        "compare_methods": args.compare_methods,
    }
    
    output_dir = Path(args.output_dir)
    
    print(f"Bounding Box Margin Removal Visualization")
    print(f"Processing {len(image_paths)} images")
    print(f"Method: {'Minimum Area Rectangle' if args.use_min_area_rect else 'Axis-Aligned Box'}")
    print(f"Optimizations: {'ENABLED' if not args.no_optimize else 'DISABLED'}")
    if not args.no_optimize:
        print(f"Downsample factor: {args.downsample_factor}")
    if args.expansion_factor > 0:
        print(f"Expansion factor: {args.expansion_factor:.1%}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Process all images
    total_start = time.time()
    results = []
    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Processing: {image_path.name}")
        result = process_image_margin_removal_bbox(
            image_path,
            config,
            output_dir,
            args.save_debug,
            command_args,
            config_source,
            use_optimized=not args.no_optimize,
            downsample_factor=args.downsample_factor,
            expansion_factor=args.expansion_factor,
            use_min_area_rect=args.use_min_area_rect,
            compare_methods=args.compare_methods,
        )
        results.append(result)
    
    total_time = time.time() - total_start
    
    # Summary
    successful_results = [r for r in results if r["success"]]
    print(f"\n{'='*60}")
    print("BOUNDING BOX MARGIN REMOVAL SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {len(successful_results)}/{len(image_paths)} images")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per image: {total_time/len(image_paths):.3f}s")
    
    if successful_results:
        avg_retention = sum(
            r.get("area_retention", 0)
            for r in successful_results
        ) / len(successful_results)
        avg_time = sum(
            r.get("processing_time", 0)
            for r in successful_results
        ) / len(successful_results)
        
        print(f"Average area retention: {avg_retention:.1%}")
        print(f"Average processing time: {avg_time:.3f}s per image")
        
        if args.compare_methods:
            avg_speedup = sum(
                r.get("comparison", {}).get("speedup", 1)
                for r in successful_results
            ) / len(successful_results)
            print(f"Average speedup vs inscribed: {avg_speedup:.1f}x")
    
    print(f"\nOutput files saved to: {output_dir}")
    
    # Save summary
    summary_file = output_dir / "margin_removal_bbox_summary.json"
    summary_data = {
        "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "config_parameters": {
            "blur_kernel_size": config.blur_kernel_size,
            "black_threshold": config.black_threshold,
            "content_threshold": config.content_threshold,
            "morph_kernel_size": config.morph_kernel_size,
            "min_content_area_ratio": config.min_content_area_ratio,
            "margin_padding": config.margin_padding,
        },
        "method_settings": {
            "use_min_area_rect": args.use_min_area_rect,
            "expansion_factor": args.expansion_factor,
            "optimized": not args.no_optimize,
            "downsample_factor": args.downsample_factor,
        },
        "performance": {
            "total_time": total_time,
            "average_time_per_image": total_time/len(image_paths) if image_paths else 0,
        },
        "results": results,
    }
    
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(convert_numpy_types(summary_data), f, indent=2)
    
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()