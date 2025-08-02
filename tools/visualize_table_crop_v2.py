#!/usr/bin/env python3
"""
Table Crop Visualization Script (Version 2)

This version uses the new processor architecture for better maintainability.
Note: This script uses Stage2Config as it's a stage 2 operation.
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

from src.ocr_pipeline.processor_wrappers import TableCropProcessor, TableLineProcessor
from src.ocr_pipeline.config import Stage2Config, Stage1Config
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


def draw_table_crop_overlay(
    image: np.ndarray, 
    h_lines: List, 
    v_lines: List,
    crop_info: Dict[str, Any]
) -> np.ndarray:
    """Draw table cropping overlay showing detected boundaries."""
    overlay = image.copy()
    height, width = image.shape[:2]
    
    # Draw all detected lines (faint)
    for line in h_lines:
        x1, y1, x2, y2 = line
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 200, 0), 1)
    
    for line in v_lines:
        x1, y1, x2, y2 = line
        cv2.line(overlay, (x1, y1), (x2, y2), (200, 0, 0), 1)
    
    # Draw crop boundaries (thick)
    if "padded_boundaries" in crop_info:
        bounds = crop_info["padded_boundaries"]
        min_x = bounds["min_x"]
        min_y = bounds["min_y"]
        max_x = bounds["max_x"]
        max_y = bounds["max_y"]
        
        # Draw crop rectangle
        cv2.rectangle(overlay, (min_x, min_y), (max_x, max_y), (0, 255, 255), 3)
        
        # Draw margin area
        if "margin" in crop_info:
            margin = crop_info["margin"]
            # Original bounds without padding
            orig_bounds = crop_info.get("original_boundaries", bounds)
            orig_min_x = orig_bounds.get("min_x", min_x + margin)
            orig_min_y = orig_bounds.get("min_y", min_y + margin)
            orig_max_x = orig_bounds.get("max_x", max_x - margin)
            orig_max_y = orig_bounds.get("max_y", max_y - margin)
            
            # Draw original bounds
            cv2.rectangle(overlay, (orig_min_x, orig_min_y), 
                         (orig_max_x, orig_max_y), (255, 255, 0), 1)
    
    # Add text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # Line counts
    cv2.putText(
        overlay,
        f"H lines: {len(h_lines)}",
        (10, 30),
        font,
        font_scale,
        (0, 255, 0),
        thickness,
    )
    cv2.putText(
        overlay,
        f"V lines: {len(v_lines)}",
        (10, 60),
        font,
        font_scale,
        (255, 0, 0),
        thickness,
    )
    
    # Crop status
    if crop_info.get("has_table"):
        crop_dims = crop_info.get("cropped_dimensions", (0, 0))
        cv2.putText(
            overlay,
            f"Table found: {crop_dims[0]}x{crop_dims[1]}px",
            (10, 90),
            font,
            font_scale,
            (0, 255, 255),
            thickness,
        )
        
        # Efficiency
        efficiency = crop_info.get("crop_efficiency", 0)
        cv2.putText(
            overlay,
            f"Efficiency: {efficiency:.1%}",
            (10, 120),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )
    else:
        cv2.putText(
            overlay,
            "No table found",
            (10, 90),
            font,
            font_scale,
            (0, 0, 255),
            thickness,
        )
    
    return overlay


def process_image(
    image_path: Path,
    line_processor: TableLineProcessor,
    crop_processor: TableCropProcessor,
    output_dir: Path,
    save_cropped: bool = True,
    save_debug: bool = False,
    command_args: Dict[str, Any] = None,
    config_source: str = "default",
) -> Dict[str, Any]:
    """Process a single image with table crop visualization."""
    print(f"Processing: {image_path.name}")
    
    try:
        # Load image
        import src.ocr_pipeline.utils as ocr_utils
        image = ocr_utils.load_image(image_path)
        
        # First detect table lines
        line_params = {}
        if command_args:
            for key in ['min_line_length', 'max_line_gap', 'hough_threshold']:
                if key in command_args and command_args[key] is not None:
                    line_params[key] = command_args[key]
        
        h_lines, v_lines, line_analysis = line_processor.process(image, **line_params)
        
        print(f"  Detected lines: H={len(h_lines)}, V={len(v_lines)}")
        
        # Then crop table region
        crop_params = {
            'return_crop_info': True,
        }
        if command_args:
            for key in ['margin', 'min_width', 'min_height']:
                if key in command_args:
                    crop_params[key] = command_args[key]
        
        result = crop_processor.process(image, h_lines, v_lines, **crop_params)
        
        if isinstance(result, tuple):
            cropped_image, crop_info = result
        else:
            cropped_image = result
            crop_info = {}
        
        # Check if table was found
        has_table = crop_info.get("has_table", cropped_image is not None)
        
        # Create visualizations
        base_name = image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Draw overlay
        overlay_image = draw_table_crop_overlay(image, h_lines, v_lines, crop_info)
        overlay_path = output_dir / f"{base_name}_table_crop_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay_image)
        
        output_files = {
            "overlay": str(overlay_path),
        }
        
        # Save cropped image if table found and requested
        if save_cropped and has_table and cropped_image is not None:
            cropped_path = output_dir / f"{base_name}_table_cropped.jpg"
            cv2.imwrite(str(cropped_path), cropped_image)
            output_files["cropped"] = str(cropped_path)
            
            # Save to processed_images directory for pipeline
            processed_dir = output_dir / "processed_images"
            processed_dir.mkdir(exist_ok=True)
            processed_path = processed_dir / f"{base_name}.jpg"
            cv2.imwrite(str(processed_path), cropped_image)
        
        # Save debug visualizations if requested
        if save_debug:
            debug_dir = output_dir / "debug" / base_name
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Line detection debug
            if "horizontal_morph" in line_analysis:
                h_morph_path = debug_dir / "horizontal_morph.png"
                cv2.imwrite(str(h_morph_path), line_analysis["horizontal_morph"])
                output_files["debug_h_morph"] = str(h_morph_path)
            
            if "vertical_morph" in line_analysis:
                v_morph_path = debug_dir / "vertical_morph.png"
                cv2.imwrite(str(v_morph_path), line_analysis["vertical_morph"])
                output_files["debug_v_morph"] = str(v_morph_path)
        
        # Save parameter documentation
        param_file = save_step_parameters(
            step_name="table_crop_v2",
            config_obj=crop_processor.config,
            command_args=command_args,
            processing_results={
                "image_name": image_path.name,
                "success": True,
                "output_files": output_files,
                "has_table": has_table,
                "h_line_count": len(h_lines),
                "v_line_count": len(v_lines),
                "crop_info": {
                    k: v for k, v in crop_info.items()
                    if k not in ["cropped_image"]
                },
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
            "has_table": has_table,
            "h_line_count": len(h_lines),
            "v_line_count": len(v_lines),
            "crop_efficiency": crop_info.get("crop_efficiency", 0),
        }
        
        if has_table:
            crop_dims = crop_info.get("cropped_dimensions", (0, 0))
            print(f"  SUCCESS: Table found ({crop_dims[0]}x{crop_dims[1]}px, {crop_info.get('crop_efficiency', 0):.1%} efficiency)")
        else:
            print(f"  SUCCESS: No table found")
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "image_name": image_path.name,
            "success": False,
            "error": str(e),
        }


def main():
    """Main function for table crop visualization."""
    parser = argparse.ArgumentParser(
        description="Table crop visualization (Version 2)"
    )
    
    # Image input arguments - note table-crop takes PNG files directly
    parser.add_argument(
        "images",
        nargs="*",
        help="Images to process (PNG files with table lines)",
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
        default="data/output/visualization/table_crop_v2",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--save-cropped",
        action="store_true",
        default=True,
        help="Save cropped table images",
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
    
    # Add configuration arguments for both processors
    add_config_arguments(parser, 'table_crop')
    
    # Table line detection parameters (for initial detection)
    parser.add_argument(
        "--min-line-length",
        type=int,
        help="Minimum line length for table detection",
    )
    parser.add_argument(
        "--max-line-gap",
        type=int,
        help="Maximum line gap for table detection",
    )
    parser.add_argument(
        "--hough-threshold",
        type=int,
        default=60,
        help="Hough transform threshold",
    )
    
    # Table crop specific parameters
    parser.add_argument(
        "--margin",
        type=int,
        default=10,
        help="Margin around table in pixels",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=100,
        help="Minimum table width",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=100,
        help="Minimum table height",
    )
    
    args = parser.parse_args()
    
    # Determine which images to process
    if args.input_dir:
        print(f"Using pipeline input directory: {args.input_dir}")
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Input directory {args.input_dir} does not exist!")
            return
        # For table-crop, we look for PNG files (output from table-lines)
        image_paths = list(input_dir.glob("*.png"))
        if not image_paths:
            # Fallback to regular image formats
            for ext in ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.PNG"]:
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
    
    # Load configurations
    # Use Stage1Config for line detection
    line_config, _ = load_config(args, Stage1Config, 'table_lines')
    # Use Stage2Config for table cropping
    crop_config, config_source = load_config(args, Stage2Config, 'table_crop')
    
    # Create processors
    line_processor = TableLineProcessor(line_config)
    crop_processor = TableCropProcessor(crop_config)
    
    # Get command arguments
    command_args = get_command_args_dict(args, 'table_crop')
    # Add line detection parameters
    command_args.update({
        'min_line_length': args.min_line_length,
        'max_line_gap': args.max_line_gap,
        'hough_threshold': args.hough_threshold,
        'margin': args.margin,
        'min_width': args.min_width,
        'min_height': args.min_height,
    })
    
    output_dir = Path(args.output_dir)
    
    print(f"Table Crop Visualization (Version 2)")
    print(f"Processing {len(image_paths)} images")
    print(f"Configuration source: {config_source}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Process all images
    results = []
    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] {image_path.name}")
        result = process_image(
            image_path,
            line_processor,
            crop_processor,
            output_dir,
            args.save_cropped,
            args.save_debug,
            command_args,
            config_source,
        )
        results.append(result)
    
    # Summary
    successful_results = [r for r in results if r["success"]]
    print(f"\n{'='*60}")
    print("TABLE CROP SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {len(successful_results)}/{len(image_paths)} images")
    
    if successful_results:
        with_tables = [r for r in successful_results if r.get("has_table", False)]
        print(f"Images with tables: {len(with_tables)}")
        
        if with_tables:
            efficiencies = [r.get("crop_efficiency", 0) for r in with_tables]
            avg_efficiency = sum(efficiencies) / len(efficiencies) if efficiencies else 0
            print(f"Average crop efficiency: {avg_efficiency:.1%}")
            
            # Show least efficient crops
            sorted_by_efficiency = sorted(with_tables, key=lambda x: x.get("crop_efficiency", 0))
            if len(sorted_by_efficiency) > 0:
                print(f"\nLeast efficient crops:")
                for r in sorted_by_efficiency[:3]:
                    print(f"  {r['image_name']}: {r.get('crop_efficiency', 0):.1%}")
    
    print(f"\nOutput files saved to: {output_dir}")
    
    # Save summary
    summary_file = output_dir / "table_crop_visualization_summary.json"
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