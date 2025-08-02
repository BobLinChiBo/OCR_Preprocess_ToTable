#!/usr/bin/env python3
"""
Margin Removal Visualization Script

This script visualizes the margin removal process, helping you assess 
parameter tuning for aggressive margin removal functionality.
"""

import warnings
warnings.warn(
    "This script is deprecated and will be removed in a future version. "
    "Please use visualize_margin_removal_v2.py or run_visualizations.py with --use-v2 flag.",
    DeprecationWarning,
    stacklevel=2
)

import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import sys
from typing import Dict, Any, List

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.ocr_pipeline.config import Config, Stage1Config, Stage2Config  # noqa: E402
import src.ocr_pipeline.utils as ocr_utils  # noqa: E402
from output_manager import (  # noqa: E402
    get_test_images,
    convert_numpy_types,
    save_step_parameters,
)



def process_image_margin_removal(
    image_path: Path,
    config: Config,
    output_dir: Path,
    save_debug: bool = False,
    command_args: Dict[str, Any] = None,
    config_source: str = "default",
) -> Dict[str, Any]:
    """Process a single image and apply margin removal."""
    print(f"Processing: {image_path.name}")
    
    try:
        # Load image
        image = ocr_utils.load_image(image_path)
        
        # Apply margin removal
        margin_removed, analysis = ocr_utils.remove_margin_aggressive(
            image,
            blur_kernel_size=config.blur_kernel_size,
            black_threshold=config.black_threshold,
            content_threshold=config.content_threshold,
            morph_kernel_size=config.morph_kernel_size,
            min_content_area_ratio=config.min_content_area_ratio,
            padding=config.margin_padding,
            return_analysis=True,
        )
        
        # Save outputs
        base_name = image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the margin-removed image
        margin_removed_path = output_dir / f"{base_name}_margin_removed.jpg"
        cv2.imwrite(str(margin_removed_path), margin_removed)
        
        output_files = {
            "margin_removed": str(margin_removed_path),
        }
        
        # Create processed_images subdirectory for pipeline use
        processed_dir = output_dir / "processed_images"
        processed_dir.mkdir(exist_ok=True)
        
        # Save margin-removed image for next pipeline stage
        # Clean up filename to avoid suffix accumulation
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
            step_name="margin_removal",
            config_obj=config,
            command_args=command_args,
            processing_results={
                "image_name": image_path.name,
                "success": True,
                "output_files": output_files,
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
            "area_retention": analysis.get("area_retention", 0),  # Keep for summary stats
        }
        
        area_retention = analysis.get("area_retention", 0)
        print(f"  SUCCESS: Area retained: {area_retention:.1%}")
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "image_name": image_path.name,
            "success": False,
            "error": str(e),
        }


def main():
    """Main function for margin removal visualization."""
    parser = argparse.ArgumentParser(description="Visualize margin removal results")
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
        default="data/output/visualization/margin_removal",
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
    
    # Parameter options
    parser.add_argument(
        "--blur-kernel-size",
        type=int,
        default=9,
        help="Blur kernel size",
    )
    parser.add_argument(
        "--black-threshold",
        type=int,
        default=45,
        help="Black threshold",
    )
    parser.add_argument(
        "--content-threshold",
        type=int,
        default=180,
        help="Content threshold",
    )
    parser.add_argument(
        "--morph-kernel-size",
        type=int,
        default=30,
        help="Morphological kernel size",
    )
    parser.add_argument(
        "--min-content-area-ratio",
        type=float,
        default=0.005,
        help="Minimum content area ratio",
    )
    parser.add_argument(
        "--margin-padding",
        type=int,
        default=10,
        help="Margin padding",
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
    
    # Collect command line arguments for parameter documentation
    command_args = {
        "blur_kernel_size": args.blur_kernel_size,
        "black_threshold": args.black_threshold,
        "content_threshold": args.content_threshold,
        "morph_kernel_size": args.morph_kernel_size,
        "min_content_area_ratio": args.min_content_area_ratio,
        "margin_padding": args.margin_padding,
        "save_debug": args.save_debug,
    }
    
    output_dir = Path(args.output_dir)
    
    print(f"Visualizing margin removal on {len(image_paths)} images")
    if args.input_dir:
        print("Pipeline mode: Processing images from input directory")
    elif args.test_images:
        print("Batch mode: Processing all images from test_images directory")
    print("Margin Removal Parameters:")
    print(f"  - Blur kernel size: {config.blur_kernel_size}")
    print(f"  - Black threshold: {config.black_threshold}")
    print(f"  - Content threshold: {config.content_threshold}")
    print(f"  - Morph kernel size: {config.morph_kernel_size}")
    print(f"  - Min content area ratio: {config.min_content_area_ratio}")
    print(f"  - Margin padding: {config.margin_padding}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Process all images
    results = []
    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Processing: {image_path.name}")
        result = process_image_margin_removal(
            image_path,
            config,
            output_dir,
            args.save_debug,
            command_args,
            config_source,
        )
        results.append(result)
    
    # Summary
    successful_results = [r for r in results if r["success"]]
    print(f"\n{'='*60}")
    print("MARGIN REMOVAL VISUALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {len(successful_results)}/{len(image_paths)} images")
    
    if successful_results:
        avg_retention = sum(
            r.get("area_retention", 0)
            for r in successful_results
        ) / len(successful_results)
        print(f"Average area retention: {avg_retention:.1%}")
        
        retention_values = [
            r.get("area_retention", 0)
            for r in successful_results
        ]
        print(f"Area retention range: {min(retention_values):.1%} - {max(retention_values):.1%}")
    
    print(f"\nOutput files saved to: {output_dir}")
    
    # Save summary
    summary_file = output_dir / "margin_removal_summary.json"
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
        "results": results,
    }
    
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(convert_numpy_types(summary_data), f, indent=2)
    
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()