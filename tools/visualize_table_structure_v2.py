#!/usr/bin/env python3
"""
Table Structure Detection Visualization Script (Version 2)

This version uses the new table detection architecture for better table structure analysis.
"""

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

from src.ocr_pipeline.processor_wrappers import TableDetectionProcessor
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


def draw_table_structure_overlay(
    image: np.ndarray, table_structure: Dict[str, Any]
) -> np.ndarray:
    """Draw table structure on the image."""
    overlay = image.copy()
    
    xs = table_structure.get("xs", [])
    ys = table_structure.get("ys", [])
    cells = table_structure.get("cells", [])
    
    # Draw grid lines
    height, width = image.shape[:2]
    
    # Draw vertical lines in green
    for x in xs:
        cv2.line(overlay, (x, 0), (x, height), (0, 255, 0), 2)
    
    # Draw horizontal lines in blue
    for y in ys:
        cv2.line(overlay, (0, y), (width, y), (255, 0, 0), 2)
    
    # Draw cell rectangles in red
    for (x1, y1, x2, y2) in cells:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Add text annotations
    analysis = table_structure.get("analysis", {})
    if analysis:
        # Add table info text
        info_text = [
            f"Table Detected: {analysis.get('table_detected', False)}",
            f"Vertical Lines: {analysis.get('num_vertical_lines', 0)}",
            f"Horizontal Lines: {analysis.get('num_horizontal_lines', 0)}",
            f"Cells: {analysis.get('num_cells', 0)}",
        ]
        
        y_offset = 30
        for i, text in enumerate(info_text):
            y_pos = y_offset + i * 25
            cv2.putText(overlay, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 0), 1, cv2.LINE_AA)
    
    return overlay


def process_image_table_structure(
    image_path: Path, 
    config: Stage1Config, 
    processor_args: Dict[str, Any]
) -> Dict[str, Any]:
    """Process a single image for table structure detection."""
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Create processor
    processor = TableDetectionProcessor(config)
    
    # Process image
    result = processor.process(image, **processor_args)
    
    # Create visualization
    overlay = draw_table_structure_overlay(image, result)
    
    # Prepare analysis data
    analysis = result.get("analysis", {})
    analysis_data = {
        "image_path": str(image_path),
        "image_shape": image.shape,
        "table_detected": analysis.get("table_detected", False),
        "num_vertical_lines": analysis.get("num_vertical_lines", 0),
        "num_horizontal_lines": analysis.get("num_horizontal_lines", 0),
        "num_cells": analysis.get("num_cells", 0),
        "table_bounds": analysis.get("table_bounds"),
        "kernel_sizes": analysis.get("kernel_sizes"),
        "clustering_eps": analysis.get("clustering_eps"),
        "vertical_lines": result.get("xs", []),
        "horizontal_lines": result.get("ys", []),
        "cells": result.get("cells", []),
    }
    
    return {
        "original": image,
        "overlay": overlay,
        "analysis": convert_numpy_types(analysis_data),
        "table_structure": result,
    }


def add_table_structure_arguments(parser: argparse.ArgumentParser):
    """Add table structure specific arguments."""
    group = parser.add_argument_group('Table Structure Detection Display Options')
    group.add_argument('--show-cells', action='store_true',
                      help='Show individual cell rectangles')
    group.add_argument('--show-grid-only', action='store_true',
                      help='Show only grid lines, not cell rectangles')


def main():
    parser = argparse.ArgumentParser(
        description="Visualize table structure detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_table_structure_v2.py                    # Process test images
  python visualize_table_structure_v2.py --test-images *.jpg    # Process specific files
  python visualize_table_structure_v2.py --pipeline --eps 15    # Run pipeline with custom eps
  python visualize_table_structure_v2.py --show-cells           # Show cell rectangles
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
        default="output/visualization/table_structure_v2",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show visualization results",
    )
    
    # Add common arguments
    add_config_arguments(parser, 'table_structure')
    add_processor_specific_arguments(parser, 'table_structure')
    add_table_structure_arguments(parser)
    
    args = parser.parse_args()
    
    # Load configuration
    config, config_source = load_config(args, Stage1Config, 'table_structure')
    
    # Get processor arguments
    processor_args = get_command_args_dict(args, 'table_structure')
    
    # Determine which images to process
    if args.input_dir:
        print(f"Using pipeline input directory: {args.input_dir}")
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Input directory {args.input_dir} does not exist!")
            return
        
        # Check if this is a table-lines output directory
        table_lines_dir = input_dir / "table_lines_images"
        if table_lines_dir.exists():
            print(f"Found table_lines_images subdirectory, using table lines as input")
            test_images = list(table_lines_dir.glob("*_table_lines.png"))
        else:
            # Look for table lines images in the main directory
            test_images = list(input_dir.glob("*_table_lines.png"))
            if not test_images:
                # Fallback to all PNG files
                test_images = []
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                    test_images.extend(input_dir.glob(ext))
        
        if not test_images:
            print(f"No images found in {args.input_dir}!")
            return
        
        print(f"Found {len(test_images)} images to process")
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
    
    print(f"Processing {len(test_images)} images for table structure detection...")
    
    # Create output directory
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Process images
    results = []
    for image_path in test_images:
        try:
            print(f"Processing: {image_path.name}")
            result = process_image_table_structure(image_path, config, processor_args)
            
            # Save visualization overlay if output directory specified
            if output_dir:
                base_name = image_path.stem
                overlay_path = output_dir / f"{base_name}_table_structure.png"
                cv2.imwrite(str(overlay_path), result["overlay"])
                
                # Save table structure data for pipeline use
                structure_data_dir = output_dir / "table_structure_data"
                structure_data_dir.mkdir(exist_ok=True)
                
                # Save the table structure as JSON
                structure_json_path = structure_data_dir / f"{base_name}_structure.json"
                with open(structure_json_path, 'w') as f:
                    json.dump(result["table_structure"], f, indent=2, default=str)
                
                result["output_files"] = {
                    "overlay": str(overlay_path),
                    "structure_data": str(structure_json_path)
                }
            
            results.append({
                "image_name": image_path.name,
                **result
            })
            
            analysis = result["analysis"]
            print(f"  Table detected: {analysis['table_detected']}")
            print(f"  Grid: {analysis['num_vertical_lines']}x{analysis['num_horizontal_lines']}")
            print(f"  Cells: {analysis['num_cells']}")
            if output_dir:
                print(f"  Saved: {base_name}_table_structure.png")
            
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
            "total_cells": sum(r["analysis"]["num_cells"] for r in results),
            "processor_args": processor_args,
            "config_summary": {
                "table_detection_eps": getattr(config, 'table_detection_eps', 10),
                "table_detection_kernel_ratio": getattr(config, 'table_detection_kernel_ratio', 0.05),
            }
        }
        
        param_file = save_step_parameters(
            step_name="table_structure_detection_v2",
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
            
            # Create window
            window_name = f"Table Structure: {image_name}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, overlay)
            
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow(window_name)
            
            if key == 27:  # ESC key
                break
    
    print(f"\nTable structure detection complete! Processed {len(results)} images.")
    
    # Print summary statistics
    detected_count = sum(1 for r in results if r["analysis"]["table_detected"])
    total_cells = sum(r["analysis"]["num_cells"] for r in results)
    
    print(f"Summary:")
    print(f"  Tables detected: {detected_count}/{len(results)}")
    print(f"  Total cells found: {total_cells}")
    if results:
        avg_cells = total_cells / len(results)
        print(f"  Average cells per image: {avg_cells:.1f}")


if __name__ == "__main__":
    main()