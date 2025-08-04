#!/usr/bin/env python3
"""
Table Crop Visualization Script (Version 2)

This script performs step 6 of the pipeline: crop to table borders using table structure.
Uses the new table structure detection and border cropping architecture.
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

from src.ocr_pipeline.processor_wrappers import TableDetectionProcessor, TableCropProcessor
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


def draw_table_crop_overlay(
    image: np.ndarray, 
    table_structure: Dict,
    crop_analysis: Dict[str, Any]
) -> np.ndarray:
    """Draw table cropping overlay showing table structure and crop boundaries."""
    overlay = image.copy()
    height, width = image.shape[:2]
    
    # Draw detected table structure (faint grid)
    xs = table_structure.get("xs", [])
    ys = table_structure.get("ys", [])
    
    # Draw vertical lines (faint green)
    for x in xs:
        cv2.line(overlay, (x, 0), (x, height), (0, 200, 0), 1)
    
    # Draw horizontal lines (faint blue)
    for y in ys:
        cv2.line(overlay, (0, y), (width, y), (200, 0, 0), 1)
    
    # Draw table cells (very faint red rectangles)
    cells = table_structure.get("cells", [])
    for (x1, y1, x2, y2) in cells:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 150), 1)
    
    # Draw crop boundaries if cropping was successful
    if crop_analysis.get("cropped", False):
        # Draw actual table boundaries (red)
        if "table_bounds" in crop_analysis:
            table_bounds = crop_analysis["table_bounds"]
            table_min_x = table_bounds["min_x"]
            table_max_x = table_bounds["max_x"]
            table_min_y = table_bounds["min_y"]
            table_max_y = table_bounds["max_y"]
            
            cv2.rectangle(overlay, (table_min_x, table_min_y), (table_max_x, table_max_y), (0, 0, 255), 2)
        
        # Draw crop boundaries with padding (thick yellow)
        if "crop_bounds" in crop_analysis:
            bounds = crop_analysis["crop_bounds"]
            min_x, max_x = bounds["min_x"], bounds["max_x"]
            min_y, max_y = bounds["min_y"], bounds["max_y"]
            
            cv2.rectangle(overlay, (min_x, min_y), (max_x, max_y), (0, 255, 255), 3)
    
    # Add text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # Table structure info
    structure_analysis = table_structure.get("analysis", {})
    cv2.putText(
        overlay,
        f"Table detected: {structure_analysis.get('table_detected', False)}",
        (10, 30),
        font,
        font_scale,
        (0, 255, 0) if structure_analysis.get('table_detected', False) else (0, 0, 255),
        thickness,
    )
    cv2.putText(
        overlay,
        f"Grid: {structure_analysis.get('num_vertical_lines', 0)}x{structure_analysis.get('num_horizontal_lines', 0)}",
        (10, 60),
        font,
        font_scale,
        (255, 0, 0),
        thickness,
    )
    cv2.putText(
        overlay,
        f"Cells: {structure_analysis.get('num_cells', 0)}",
        (10, 90),
        font,
        font_scale,
        (255, 0, 255),
        thickness,
    )
    
    # Crop status
    if crop_analysis.get("cropped", False):
        cropped_shape = crop_analysis.get("cropped_shape", (0, 0))
        padding = crop_analysis.get("padding_applied", 0)
        crop_method = crop_analysis.get("crop_method", "unknown")
        cv2.putText(
            overlay,
            f"Cropped: {cropped_shape[1]}x{cropped_shape[0]}px (pad={padding})",
            (10, 120),
            font,
            font_scale,
            (0, 255, 255),
            thickness,
        )
        cv2.putText(
            overlay,
            f"Method: {crop_method}",
            (10, 150),
            font,
            font_scale,
            (0, 255, 255),
            thickness,
        )
        
        # Size reduction
        if "size_reduction" in crop_analysis:
            reduction = crop_analysis["size_reduction"]
            cv2.putText(
                overlay,
                f"Reduction: {reduction['width']}x{reduction['height']}px",
                (10, 180),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )
    else:
        reason = crop_analysis.get("reason", "Unknown")
        cv2.putText(
            overlay,
            f"Not cropped: {reason}",
            (10, 120),
            font,
            font_scale,
            (0, 0, 255),
            thickness,
        )
    
    return overlay


def process_image_table_crop(
    image_path: Path,
    config: Stage1Config,
    processor_args: Dict[str, Any],
    pipeline_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Process a single image for table border cropping (step 6)."""
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    target_image = image  # This is what we'll crop
    table_structure = None  # Initialize to None
    
    # In pipeline mode, we MUST have the table structure data
    if pipeline_dir:
        # Extract base name - handle both deskewed and regular image names
        base_name = image_path.stem
        if "_deskewed" in base_name:
            base_name = base_name.replace("_deskewed", "")
        elif "_table" in base_name:
            # Handle other suffixes if present
            base_name = base_name.split("_table")[0]
        
        # Look for table structure JSON in stage 5 output
        structure_json_path = pipeline_dir / "05_table-structure" / "table_structure_data" / f"{base_name}_table_lines_structure.json"
        
        if not structure_json_path.exists():
            raise FileNotFoundError(
                f"Table structure data not found: {structure_json_path}\n"
                f"Make sure table-structure stage completed successfully and saved data for {base_name}"
            )
        
        # Load pre-computed table structure
        print(f"  Loading table structure from: {structure_json_path.name}")
        try:
            with open(structure_json_path, 'r') as f:
                table_structure = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {structure_json_path}: {e}")
        
        # Validate structure
        if not isinstance(table_structure, dict):
            raise ValueError(f"Table structure should be a dict, got: {type(table_structure)}")
        if "cells" not in table_structure:
            raise ValueError(f"Table structure missing 'cells' key. Keys found: {list(table_structure.keys())}")
        
        print(f"  Loaded structure with {len(table_structure.get('cells', []))} cells")
    else:
        # Standalone mode - we cannot proceed without table structure
        raise ValueError(
            "Table crop requires table structure data from previous pipeline stages.\n"
            "Please run the full pipeline or provide pipeline directory with --pipeline-dir"
        )
    
    # Create crop processor
    crop_processor = TableCropProcessor(config)
    
    # Step 6: Crop the target image (deskewed) using detected structure
    crop_args = {k: v for k, v in processor_args.items() 
                if k in ['padding']}
    result = crop_processor.process(target_image, table_structure, return_analysis=True, **crop_args)
    
    if isinstance(result, tuple):
        cropped_image, crop_analysis = result
    else:
        cropped_image = result
        crop_analysis = {"cropped": False, "reason": "No analysis returned"}
    
    # Create visualization overlay using the target image
    overlay = draw_table_crop_overlay(target_image, table_structure, crop_analysis)
    
    # Prepare combined analysis data
    structure_analysis = table_structure.get("analysis", {})
    analysis_data = {
        "image_path": str(image_path),
        "image_shape": target_image.shape,
        "table_detected": structure_analysis.get("table_detected", False),
        "num_vertical_lines": structure_analysis.get("num_vertical_lines", 0),
        "num_horizontal_lines": structure_analysis.get("num_horizontal_lines", 0),
        "num_cells": structure_analysis.get("num_cells", 0),
        "cropped": crop_analysis.get("cropped", False),
        "crop_analysis": convert_numpy_types(crop_analysis),
        "original_size": {"width": target_image.shape[1], "height": target_image.shape[0]},
        "cropped_size": {
            "width": cropped_image.shape[1], 
            "height": cropped_image.shape[0]
        } if cropped_image is not None else None,
    }
    
    return {
        "original": target_image,
        "cropped": cropped_image,
        "overlay": overlay,
        "analysis": analysis_data,
        "table_structure": table_structure,
    }


# Table crop arguments are now handled by the config system in DEFAULT_PARAMS


def main():
    parser = argparse.ArgumentParser(
        description="Visualize table border cropping using table structure (Step 6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_table_crop_v2.py                          # Process test images
  python visualize_table_crop_v2.py --test-images *.jpg      # Process specific files
  python visualize_table_crop_v2.py --pipeline --padding 30  # Custom padding
  python visualize_table_crop_v2.py --eps 15 --kernel-ratio 0.08  # Custom detection params

Pipeline Context:
  This script performs Step 6: Crop to table borders with padding
  - Input: Deskewed image (from step 3)
  - Process: Detect table structure → Crop using borders
  - Output: Border-cropped table with visualization
        """
    )
    
    # Standard visualization arguments
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
        default="output/visualization/table_crop_v2",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show visualization results",
    )
    
    # Add common arguments
    add_config_arguments(parser, 'table_crop')
    add_processor_specific_arguments(parser, 'table_crop')
    
    args = parser.parse_args()
    
    # Load configuration
    config, config_source = load_config(args, Stage1Config, 'table_crop')
    
    # Get processor arguments
    processor_args = get_command_args_dict(args, 'table_crop')
    
    # Determine which images to process
    if args.input_dir:
        print(f"Using pipeline input directory: {args.input_dir}")
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Input directory {args.input_dir} does not exist!")
            return
        test_images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            test_images.extend(input_dir.glob(ext))
        if not test_images:
            print(f"No images found in {args.input_dir}!")
            return
    elif args.test_images:
        print("Using batch mode: processing all images in test_images/")
        test_images = get_test_images()
        if not test_images:
            print("No images found in test_images directory!")
            return
    else:
        # Use individual image arguments
        test_images = []
        for img_path in args.images:
            path = Path(img_path)
            if path.exists():
                test_images.append(path)
            else:
                print(f"Warning: {img_path} not found, skipping")
        
        if not test_images:
            print("No valid images found!")
            parser.print_help()
            return
    
    print(f"Processing {len(test_images)} images for table border cropping...")
    
    # Create output directory
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Create processed_images subdirectory for pipeline
        processed_dir = output_dir / "processed_images"
        processed_dir.mkdir(exist_ok=True)
    
    # Detect if we're in pipeline mode by checking output directory structure
    pipeline_dir = None
    if output_dir and output_dir.parent.name.startswith("pipeline_"):
        pipeline_dir = output_dir.parent
        print(f"Detected pipeline mode, pipeline directory: {pipeline_dir}")
    
    # Process images
    results = []
    for image_path in test_images:
        try:
            print(f"Processing: {image_path.name}")
            result = process_image_table_crop(image_path, config, processor_args, pipeline_dir)
            results.append({
                "image_name": image_path.name,
                **result
            })
            
            # Save visualization overlay if output directory specified
            if output_dir:
                base_name = image_path.stem
                overlay_path = output_dir / f"{base_name}_table_crop.png"
                cv2.imwrite(str(overlay_path), result["overlay"])
                
                # Save cropped image to processed_images for pipeline
                if result["cropped"] is not None:
                    processed_path = processed_dir / f"{base_name}.jpg"
                    cv2.imwrite(str(processed_path), result["cropped"])
                else:
                    # If no cropping, save original
                    processed_path = processed_dir / f"{base_name}.jpg"
                    cv2.imwrite(str(processed_path), result["original"])
                
                result["output_files"] = {
                    "overlay": str(overlay_path),
                    "processed": str(processed_path)
                }
            
            analysis = result["analysis"]
            print(f"  Table detected: {analysis['table_detected']}")
            print(f"  Grid: {analysis['num_vertical_lines']}x{analysis['num_horizontal_lines']}")
            print(f"  Cells: {analysis['num_cells']}")
            print(f"  Cropped: {analysis['cropped']}")
            if analysis['cropped']:
                crop_method = analysis.get('crop_analysis', {}).get('crop_method', 'unknown')
                print(f"  Crop method: {crop_method}")
                if analysis['cropped_size']:
                    print(f"  Final size: {analysis['cropped_size']['width']}x{analysis['cropped_size']['height']}px")
            if output_dir:
                print(f"  Saved: {base_name}_table_crop.png")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    if not results:
        print("No images processed successfully.")
        return
    
    # Save parameters used
    if output_dir:
        # Prepare summary of all results
        summary_data = {
            "total_images": len(results),
            "tables_detected": sum(1 for r in results if r["analysis"]["table_detected"]),
            "successfully_cropped": sum(1 for r in results if r["analysis"]["cropped"]),
            "processor_args": processor_args,
            "config_summary": {
                "table_detection_eps": getattr(config, 'table_detection_eps', 10),
                "table_detection_kernel_ratio": getattr(config, 'table_detection_kernel_ratio', 0.05),
                "table_crop_padding": getattr(config, 'table_crop_padding', 20),
            }
        }
        
        param_file = save_step_parameters(
            step_name="table_border_crop_v2",
            config_obj=config,
            command_args=processor_args,
            processing_results=summary_data,
            output_dir=output_dir,
            config_source=config_source,
        )
    
    # Show results if requested
    if args.show:
        print("\nShowing results (press any key to continue, ESC to quit)...")
        for result in results:
            image_name = result["image_name"]
            overlay = result["overlay"]
            cropped = result["cropped"]
            
            # Show overlay with crop boundaries
            window_name = f"Table Border Crop: {image_name}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, overlay)
            
            # Show cropped result if available
            if cropped is not None:
                crop_window_name = f"Cropped Result: {image_name}"
                cv2.namedWindow(crop_window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(crop_window_name, cropped)
            
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            
            if key == 27:  # ESC key
                break
    
    print(f"\nTable border crop visualization complete! Processed {len(results)} images.")
    
    # Print summary statistics
    cropped_count = sum(1 for r in results if r["analysis"]["cropped"])
    detected_count = sum(1 for r in results if r["analysis"]["table_detected"])
    
    print(f"Summary:")
    print(f"  Tables detected: {detected_count}/{len(results)}")
    print(f"  Successfully cropped: {cropped_count}/{len(results)}")
    
    if cropped_count > 0:
        # Calculate average size reduction
        size_reductions = []
        for r in results:
            if r["analysis"]["cropped"] and r["analysis"]["cropped_size"]:
                orig = r["analysis"]["original_size"]
                crop = r["analysis"]["cropped_size"]
                reduction_ratio = 1 - ((crop["width"] * crop["height"]) / (orig["width"] * orig["height"]))
                size_reductions.append(reduction_ratio)
        
        if size_reductions:
            avg_reduction = sum(size_reductions) / len(size_reductions)
            print(f"  Average size reduction: {avg_reduction:.1%}")


if __name__ == "__main__":
    main()