#!/usr/bin/env python3
"""
Table Line Detection Visualization Script (Version 2)

This version uses the new processor architecture for better maintainability.
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

from src.ocr_pipeline.processor_wrappers import TableLineProcessor
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


def draw_table_lines_overlay(
    image: np.ndarray, line_info: Dict[str, Any]
) -> np.ndarray:
    """Draw table lines on the image."""
    overlay = image.copy()
    
    # Draw horizontal lines
    for line in line_info["h_lines"]:
        x1, y1, x2, y2 = line
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw vertical lines
    for line in line_info["v_lines"]:
        x1, y1, x2, y2 = line
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Add statistics
    cv2.putText(
        overlay,
        f"Horizontal lines: {line_info['h_line_count']}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        overlay,
        f"Vertical lines: {line_info['v_line_count']}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )
    
    if line_info.get("h_avg_length", 0) > 0:
        cv2.putText(
            overlay,
            f"H avg length: {line_info['h_avg_length']:.1f}px",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
        )
    
    if line_info.get("v_avg_length", 0) > 0:
        cv2.putText(
            overlay,
            f"V avg length: {line_info['v_avg_length']:.1f}px",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            1,
        )
        
    table_status = "Table structure: Yes" if line_info.get("has_table_structure") else "Table structure: No"
    cv2.putText(
        overlay,
        table_status,
        (10, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    
    return overlay


def draw_lines_only(
    image_shape: tuple, line_info: Dict[str, Any], background_color: int = 255
) -> np.ndarray:
    """Draw only the detected table lines on a clean background."""
    height, width = image_shape[:2]
    
    # Create clean background (white by default)
    lines_image = np.full((height, width, 3), background_color, dtype=np.uint8)
    
    # Draw horizontal lines in green
    for line in line_info["h_lines"]:
        x1, y1, x2, y2 = line
        cv2.line(lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw vertical lines in blue  
    for line in line_info["v_lines"]:
        x1, y1, x2, y2 = line
        cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    return lines_image


def process_image(
    image_path: Path,
    processor: TableLineProcessor,
    output_dir: Path,
    save_debug: bool = False,
    command_args: Dict[str, Any] = None,
    config_source: str = "default",
) -> Dict[str, Any]:
    """Process a single image with table line detection."""
    print(f"Processing: {image_path.name}")
    
    try:
        # Load image
        import src.ocr_pipeline.utils as ocr_utils
        image = ocr_utils.load_image(image_path)
        
        # Get processing parameters from command args
        processing_params = {}
        if command_args:
            for key in ['min_line_length', 'max_line_gap', 'hough_threshold']:
                if key in command_args and command_args[key] is not None:
                    processing_params[key] = command_args[key]
        
        # Process with enhanced analysis
        line_info = processor.process_with_enhanced_analysis(image, **processing_params)
        
        # Create visualizations
        base_name = image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Draw overlay
        overlay_image = draw_table_lines_overlay(image, line_info)
        overlay_path = output_dir / f"{base_name}_table_lines_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay_image)
        
        output_files = {
            "overlay": str(overlay_path),
        }
        
        # Save debug visualizations if requested
        if save_debug and 'filtering_steps' in line_info:
            debug_dir = output_dir / "debug" / base_name
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            for step_name, step_image in line_info['filtering_steps'].items():
                if isinstance(step_image, np.ndarray):
                    debug_path = debug_dir / f"{step_name}.png"
                    cv2.imwrite(str(debug_path), step_image)
                    output_files[f"debug_{step_name}"] = str(debug_path)
        
        # Save parameter documentation
        param_file = save_step_parameters(
            step_name="table_lines_detection_v2",
            config_obj=processor.config,
            command_args=command_args,
            processing_results={
                "image_name": image_path.name,
                "success": True,
                "output_files": output_files,
                "line_info": {
                    k: v
                    for k, v in line_info.items()
                    if k not in ["h_lines", "v_lines", "horizontal_morph", "vertical_morph", "filtering_steps"]
                },
            },
            output_dir=output_dir,
            config_source=config_source,
        )
        
        if param_file:
            output_files["parameters"] = str(param_file)
        
        # Save clean lines-only visualization if table structure detected
        if line_info["has_table_structure"]:
            lines_only_image = draw_lines_only(image.shape, line_info)
            table_data_path = output_dir / f"{base_name}_table_lines.png"
            cv2.imwrite(str(table_data_path), lines_only_image)
            output_files["table_lines_only"] = str(table_data_path)
        
        # Save to processed_images directory for pipeline
        processed_dir = output_dir / "processed_images"
        processed_dir.mkdir(exist_ok=True)
        processed_path = processed_dir / f"{base_name}.jpg"
        cv2.imwrite(str(processed_path), image)  # Save original image for pipeline
        
        # Also save table lines images to a dedicated directory for table structure detection
        if line_info["has_table_structure"]:
            table_lines_dir = output_dir / "table_lines_images"
            table_lines_dir.mkdir(exist_ok=True)
            table_lines_copy_path = table_lines_dir / f"{base_name}_table_lines.png"
            cv2.imwrite(str(table_lines_copy_path), lines_only_image)
        
        result = {
            "image_name": image_path.name,
            "success": True,
            "output_files": output_files,
            "parameter_file": str(param_file) if param_file else None,
            "h_line_count": line_info.get("h_line_count", 0),
            "v_line_count": line_info.get("v_line_count", 0),
            "has_table_structure": line_info.get("has_table_structure", False),
        }
        
        print(f"  SUCCESS: H={line_info['h_line_count']}, V={line_info['v_line_count']}")
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "image_name": image_path.name,
            "success": False,
            "error": str(e),
        }


def main():
    """Main function for table line detection visualization."""
    parser = argparse.ArgumentParser(
        description="Table line detection visualization (Version 2)"
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
        default="data/output/visualization/table_lines_v2",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save debug visualizations",
    )
    parser.add_argument(
        "--show-filtering-steps",
        action="store_true",
        help="Show intermediate filtering steps (implies --save-debug)",
    )
    
    # Add configuration arguments
    add_config_arguments(parser, 'table_lines')
    add_processor_specific_arguments(parser, 'table_lines')
    
    args = parser.parse_args()
    
    # Handle show-filtering-steps
    if args.show_filtering_steps:
        args.save_debug = True
    
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
    config, config_source = load_config(args, Stage1Config, 'table_lines')
    
    # Create processor
    processor = TableLineProcessor(config)
    
    # Get command arguments
    command_args = get_command_args_dict(args, 'table_lines')
    
    output_dir = Path(args.output_dir)
    
    print(f"Table Line Detection Visualization (Version 2)")
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
            processor,
            output_dir,
            args.save_debug,
            command_args,
            config_source,
        )
        results.append(result)
    
    # Summary
    successful_results = [r for r in results if r["success"]]
    print(f"\n{'='*60}")
    print("TABLE LINE DETECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {len(successful_results)}/{len(image_paths)} images")
    
    if successful_results:
        with_tables = [
            r for r in successful_results if r.get("has_table_structure", False)
        ]
        print(f"Images with table structure: {len(with_tables)}")
        
        total_h_lines = sum(r.get("h_line_count", 0) for r in successful_results)
        total_v_lines = sum(r.get("v_line_count", 0) for r in successful_results)
        print(f"Total horizontal lines: {total_h_lines}")
        print(f"Total vertical lines: {total_v_lines}")
    
    print(f"\nOutput files saved to: {output_dir}")
    
    # Save summary
    summary_file = output_dir / "table_lines_visualization_summary.json"
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