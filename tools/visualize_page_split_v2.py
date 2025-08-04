#!/usr/bin/env python3
"""
Page Split Visualization Script (Version 2)

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

from src.ocr_pipeline.processor_wrappers import PageSplitProcessor
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


def find_gutter_detailed(
    image: np.ndarray,
    processor: PageSplitProcessor,
    **kwargs
) -> Dict[str, Any]:
    """Enhanced gutter detection with detailed analysis using new robust algorithm."""
    import src.ocr_pipeline.utils as utils
    
    # Get gutter detection parameters
    params = {
        'search_ratio': kwargs.get('search_ratio', 0.3),
        'blur_k': kwargs.get('blur_k', 21),
        'open_k': kwargs.get('open_k', 9),
        'width_min': kwargs.get('width_min', 20),
        'return_analysis': True,
    }
    
    _, _, analysis = utils.split_two_page_image(image, **params)
    
    # Determine if image has two pages based on enhanced gutter analysis
    gutter_strength_threshold = 0.15  # Minimum contrast needed for valid gutter
    min_effective_width = 1  # Minimum width for any recognizable gutter
    
    meets_width_requirement = analysis.get('meets_min_width', False)
    gutter_strength = analysis.get('gutter_strength', 0)
    gutter_width = analysis.get('gutter_width', 0)
    fallback_used = analysis.get('fallback_used', False)
    
    # Enhanced two-page detection logic for new algorithm
    if fallback_used:
        # If fallback was used, probably single page or very poor quality
        has_two_pages = False
    else:
        # Use improved criteria based on segment analysis
        valid_segments = analysis.get('valid_segments', [])
        segments = analysis.get('segments', [])
        
        # Accept if we found valid segments and good strength
        has_two_pages = (
            len(valid_segments) > 0 and 
            gutter_strength >= gutter_strength_threshold and
            gutter_width >= min_effective_width
        ) or (
            # Alternative: good strength with minimal width and some segments found
            len(segments) > 0 and 
            gutter_width >= min_effective_width and 
            gutter_strength >= 0.3
        )
    
    analysis['has_two_pages'] = has_two_pages
    
    return analysis


def draw_split_overlay(image: np.ndarray, gutter_info: Dict[str, Any]) -> np.ndarray:
    """Draw page splitting overlay showing gutter detection with enhanced visualization."""
    overlay = image.copy()
    height, width = image.shape[:2]

    gutter_x = gutter_info["gutter_x"]
    search_start = gutter_info["search_start"]
    search_end = gutter_info["search_end"]

    # Draw search region (light blue overlay)
    search_overlay = np.zeros_like(overlay)
    cv2.rectangle(
        search_overlay, (search_start, 0), (search_end, height), (255, 255, 0), -1
    )
    overlay = cv2.addWeighted(overlay, 0.9, search_overlay, 0.1, 0)

    # Draw all detected segments (light green)
    segments = gutter_info.get("segments", [])
    for start_x, end_x in segments:
        cv2.rectangle(overlay, (start_x, 0), (end_x, height), (0, 255, 0), 1)

    # Draw valid segments (bright green)
    valid_segments = gutter_info.get("valid_segments", [])
    for start_x, end_x in valid_segments:
        cv2.rectangle(overlay, (start_x, 0), (end_x, height), (0, 255, 0), 2)

    # Draw selected segment (magenta)
    selected_segment = gutter_info.get("selected_segment")
    if selected_segment:
        start_x, end_x = selected_segment
        cv2.rectangle(overlay, (start_x, 0), (end_x, height), (255, 0, 255), 3)

    # Draw gutter line (red)
    cv2.line(overlay, (gutter_x, 0), (gutter_x, height), (0, 0, 255), 3)

    # Draw page boundaries (green)
    cv2.line(overlay, (0, 0), (0, height), (0, 255, 0), 2)
    cv2.line(overlay, (width - 1, 0), (width - 1, height), (0, 255, 0), 2)

    # Draw search region boundaries (yellow)
    cv2.line(overlay, (search_start, 0), (search_start, height), (0, 255, 255), 1)
    cv2.line(overlay, (search_end, 0), (search_end, height), (0, 255, 255), 1)

    # Add info text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    cv2.putText(
        overlay,
        f"Gutter: {gutter_x}px",
        (10, 30),
        font,
        font_scale,
        (0, 0, 255),
        thickness,
    )
    cv2.putText(
        overlay,
        f"Search: {search_start}-{search_end}px",
        (10, 60),
        font,
        font_scale,
        (0, 255, 255),
        thickness,
    )
    cv2.putText(
        overlay,
        f"Gutter width: {gutter_info['gutter_width']}px",
        (10, 90),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
    )

    # Show segment information
    segments_count = len(gutter_info.get("segments", []))
    valid_count = len(gutter_info.get("valid_segments", []))
    cv2.putText(
        overlay,
        f"Segments: {segments_count} found, {valid_count} valid",
        (10, 120),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
    )

    # Show if fallback was used
    fallback_used = gutter_info.get("fallback_used", False)
    if fallback_used:
        cv2.putText(
            overlay,
            "FALLBACK: Center split used",
            (10, 150),
            font,
            font_scale,
            (0, 165, 255),  # Orange
            thickness,
        )

    has_two_pages = gutter_info.get("has_two_pages", False)
    cv2.putText(
        overlay,
        f"Two pages detected: {'Yes' if has_two_pages else 'No'}",
        (10, 180),
        font,
        font_scale,
        (0, 255, 0) if has_two_pages else (0, 0, 255),
        thickness,
    )

    return overlay


def process_image(
    image_path: Path,
    processor: PageSplitProcessor,
    output_dir: Path,
    save_individual_pages: bool = True,
    command_args: Dict[str, Any] = None,
    config_source: str = "default",
) -> Dict[str, Any]:
    """Process a single image with page split visualization."""
    print(f"Processing: {image_path.name}")
    
    try:
        # Load image
        import src.ocr_pipeline.utils as ocr_utils
        image = ocr_utils.load_image(image_path)
        
        # Get gutter analysis
        gutter_info = find_gutter_detailed(image, processor)
        
        # Process with page split
        left_page, right_page = processor.process(image)
        
        # Create visualizations
        base_name = image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Draw gutter overlay
        overlay_image = draw_split_overlay(image, gutter_info)
        overlay_path = output_dir / f"{base_name}_page_split_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay_image)
        
        output_files = {
            "overlay": str(overlay_path),
        }
        
        # Save individual pages if requested
        if save_individual_pages and gutter_info.get("has_two_pages", False):
            left_path = output_dir / f"{base_name}_left_page.jpg"
            right_path = output_dir / f"{base_name}_right_page.jpg"
            cv2.imwrite(str(left_path), left_page)
            cv2.imwrite(str(right_path), right_page)
            output_files["left_page"] = str(left_path)
            output_files["right_page"] = str(right_path)
            
            # Save to processed_images directory for pipeline
            processed_dir = output_dir / "processed_images"
            processed_dir.mkdir(exist_ok=True)
            
            processed_left = processed_dir / f"{base_name}_left_page.jpg"
            processed_right = processed_dir / f"{base_name}_right_page.jpg"
            cv2.imwrite(str(processed_left), left_page)
            cv2.imwrite(str(processed_right), right_page)
        else:
            # For single page images or when individual pages not requested,
            # still save to processed_images for pipeline support
            processed_dir = output_dir / "processed_images"
            processed_dir.mkdir(exist_ok=True)
            
            # For single page, save the original or left page (which is the processed result)
            processed_image = processed_dir / f"{base_name}_processed.jpg"
            cv2.imwrite(str(processed_image), left_page)  # left_page contains the processed result
        
        # Save parameter documentation
        param_file = save_step_parameters(
            step_name="page_split_v2",
            config_obj=processor.config,
            command_args=command_args,
            processing_results={
                "image_name": image_path.name,
                "success": True,
                "output_files": output_files,
                "gutter_info": {
                    k: v
                    for k, v in gutter_info.items()
                    if k not in ["projection_profile", "smoothed_profile"]
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
            "has_two_pages": gutter_info.get("has_two_pages", False),
            "gutter_x": gutter_info.get("gutter_x"),
            "gutter_width": gutter_info.get("gutter_width"),
        }
        
        print(f"  SUCCESS: Two pages={gutter_info.get('has_two_pages', False)}, Gutter at {gutter_info.get('gutter_x')}px")
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "image_name": image_path.name,
            "success": False,
            "error": str(e),
        }


def main():
    """Main function for page split visualization."""
    parser = argparse.ArgumentParser(
        description="Page split visualization (Version 2)"
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
        default="data/output/visualization/page_split_v2",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--save-individual-pages",
        action="store_true",
        default=True,
        help="Save left and right pages separately",
    )
    parser.add_argument(
        "--no-individual-pages",
        dest="save_individual_pages",
        action="store_false",
        help="Don't save individual pages",
    )
    
    # Add configuration arguments
    add_config_arguments(parser, 'page_split')
    
    # Page split specific arguments
    parser.add_argument(
        "--search-ratio",
        type=float,
        default=0.3,
        help="Fraction of width (centered) to search for gutter (0-1)",
    )
    parser.add_argument(
        "--blur-k",
        type=int,
        default=21,
        help="Gaussian blur kernel size (odd number, higher = more noise removal)",
    )
    parser.add_argument(
        "--open-k",
        type=int,
        default=9,
        help="Morphological opening kernel width (removes thin vertical lines)",
    )
    parser.add_argument(
        "--width-min",
        type=int,
        default=20,
        help="Minimum gutter width in pixels",
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
    config, config_source = load_config(args, Stage1Config, 'page_split')
    
    # Create processor
    processor = PageSplitProcessor(config)
    
    # Get command arguments
    command_args = get_command_args_dict(args, 'page_split')
    # Add page split specific args
    command_args.update({
        'search_ratio': args.search_ratio,
        'blur_k': args.blur_k,
        'open_k': args.open_k,
        'width_min': args.width_min,
    })
    
    output_dir = Path(args.output_dir)
    
    print(f"Page Split Visualization (Version 2)")
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
            args.save_individual_pages,
            command_args,
            config_source,
        )
        results.append(result)
    
    # Summary
    successful_results = [r for r in results if r["success"]]
    print(f"\n{'='*60}")
    print("PAGE SPLIT SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {len(successful_results)}/{len(image_paths)} images")
    
    if successful_results:
        with_two_pages = [
            r for r in successful_results if r.get("has_two_pages", False)
        ]
        print(f"Images with two pages: {len(with_two_pages)}")
        
        if with_two_pages:
            avg_gutter = sum(r.get("gutter_x", 0) for r in with_two_pages) / len(with_two_pages)
            print(f"Average gutter position: {avg_gutter:.1f}px")
    
    print(f"\nOutput files saved to: {output_dir}")
    
    # Save summary
    summary_file = output_dir / "page_split_visualization_summary.json"
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