#!/usr/bin/env python3
"""
Page Split Visualization Script

This script visualizes the two-page splitting process, helping you assess 
gutter detection quality and adjust splitting parameters.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import sys
from typing import Dict, Any, Tuple

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from ocr.config import Config
from src.ocr_pipeline.utils import load_image, split_two_page_image
from visualization.output_manager import get_test_images, convert_numpy_types


def find_gutter_detailed(image: np.ndarray, gutter_start: float = 0.4, 
                        gutter_end: float = 0.6, min_gutter_width: int = 50) -> Dict[str, Any]:
    """Enhanced gutter detection with detailed analysis."""
    _, _, analysis = split_two_page_image(
        image, 
        gutter_start=gutter_start, 
        gutter_end=gutter_end, 
        min_gutter_width=min_gutter_width,
        return_analysis=True
    )
    return analysis


def draw_split_overlay(image: np.ndarray, gutter_info: Dict[str, Any]) -> np.ndarray:
    """Draw page splitting overlay showing gutter detection."""
    overlay = image.copy()
    height, width = image.shape[:2]
    
    gutter_x = gutter_info['gutter_x']
    search_start = gutter_info['search_start']
    search_end = gutter_info['search_end']
    
    # Draw search region (light blue overlay)
    search_overlay = np.zeros_like(overlay)
    cv2.rectangle(search_overlay, (search_start, 0), (search_end, height), (255, 255, 0), -1)
    overlay = cv2.addWeighted(overlay, 0.9, search_overlay, 0.1, 0)
    
    # Draw gutter line (red)
    cv2.line(overlay, (gutter_x, 0), (gutter_x, height), (0, 0, 255), 3)
    
    # Draw page boundaries (green)
    cv2.line(overlay, (0, 0), (0, height), (0, 255, 0), 2)
    cv2.line(overlay, (width-1, 0), (width-1, height), (0, 255, 0), 2)
    
    # Draw search region boundaries (yellow)
    cv2.line(overlay, (search_start, 0), (search_start, height), (0, 255, 255), 1)
    cv2.line(overlay, (search_end, 0), (search_end, height), (0, 255, 255), 1)
    
    # Add info text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    text_lines = [
        f"Gutter X: {gutter_x}",
        f"Strength: {gutter_info['gutter_strength']:.3f}",
        f"Width: {gutter_info['gutter_width']}px",
        f"Search: {search_start}-{search_end}",
        f"Min Width OK: {gutter_info['meets_min_width']}"
    ]
    
    # Draw text background
    text_y_start = 30
    for i, line in enumerate(text_lines):
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        cv2.rectangle(overlay, (10, text_y_start + i*35 - 25), 
                     (20 + text_size[0], text_y_start + i*35 + 10), 
                     (0, 0, 0), -1)
    
    # Draw text
    for i, line in enumerate(text_lines):
        cv2.putText(overlay, line, (15, text_y_start + i*35), 
                   font, font_scale, (255, 255, 255), thickness)
    
    return overlay


def create_gutter_analysis_plot(gutter_info: Dict[str, Any], width: int) -> np.ndarray:
    """Create a plot showing the vertical projection analysis."""
    vertical_sums = gutter_info['vertical_sums']
    plot_height = 300
    plot_width = len(vertical_sums)
    
    # Normalize values for plotting
    max_val = np.max(vertical_sums)
    min_val = np.min(vertical_sums)
    if max_val > min_val:
        normalized = ((vertical_sums - min_val) / (max_val - min_val) * (plot_height - 40)).astype(int)
    else:
        normalized = np.full_like(vertical_sums, plot_height // 2, dtype=int)
    
    # Create plot image
    plot_img = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255
    
    # Draw the curve
    for i in range(len(normalized) - 1):
        y1 = plot_height - 20 - normalized[i]
        y2 = plot_height - 20 - normalized[i + 1]
        cv2.line(plot_img, (i, y1), (i + 1, y2), (0, 0, 255), 2)
    
    # Mark the gutter position
    gutter_offset = gutter_info['gutter_x'] - gutter_info['search_start']
    if 0 <= gutter_offset < len(normalized):
        y_pos = plot_height - 20 - normalized[gutter_offset]
        cv2.circle(plot_img, (gutter_offset, y_pos), 5, (0, 255, 0), -1)
    
    # Add axis labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(plot_img, "Vertical Projection", (10, 20), font, 0.6, (0, 0, 0), 1)
    cv2.putText(plot_img, f"Min: {min_val:.0f}", (10, plot_height - 5), font, 0.5, (0, 0, 0), 1)
    cv2.putText(plot_img, f"Max: {max_val:.0f}", (plot_width - 80, plot_height - 5), font, 0.5, (0, 0, 0), 1)
    
    return plot_img


def create_split_comparison(original: np.ndarray, left_page: np.ndarray, 
                           right_page: np.ndarray, overlay: np.ndarray, 
                           gutter_plot: np.ndarray) -> np.ndarray:
    """Create a comprehensive comparison showing all split results."""
    target_height = 600
    
    # Resize original and overlay
    scale = target_height / original.shape[0]
    new_width = int(original.shape[1] * scale)
    
    orig_resized = cv2.resize(original, (new_width, target_height))
    overlay_resized = cv2.resize(overlay, (new_width, target_height))
    
    # Resize pages
    left_scale = target_height / left_page.shape[0]
    left_width = int(left_page.shape[1] * left_scale)
    left_resized = cv2.resize(left_page, (left_width, target_height))
    
    right_scale = target_height / right_page.shape[0]
    right_width = int(right_page.shape[1] * right_scale)
    right_resized = cv2.resize(right_page, (right_width, target_height))
    
    # Resize gutter plot
    plot_scale = new_width / gutter_plot.shape[1]
    plot_height = int(gutter_plot.shape[0] * plot_scale)
    plot_resized = cv2.resize(gutter_plot, (new_width, plot_height))
    
    # Create labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_height = 40
    
    def create_label(text: str, width: int) -> np.ndarray:
        label_img = np.zeros((label_height, width, 3), dtype=np.uint8)
        cv2.putText(label_img, text, (10, 25), font, 0.8, (255, 255, 255), 2)
        return label_img
    
    # Create top row (original with analysis)
    orig_label = create_label("Original", new_width)
    overlay_label = create_label("Gutter Detection", new_width)
    plot_label = create_label("Vertical Projection", new_width)
    
    top_row = np.vstack([
        orig_label, orig_resized,
        overlay_label, overlay_resized,
        plot_label, plot_resized
    ])
    
    # Create bottom row (split pages)
    left_label = create_label("Left Page", left_width)
    right_label = create_label("Right Page", right_width)
    
    left_panel = np.vstack([left_label, left_resized])
    right_panel = np.vstack([right_label, right_resized])
    
    # Pad to match heights
    max_height = max(left_panel.shape[0], right_panel.shape[0])
    if left_panel.shape[0] < max_height:
        padding = np.zeros((max_height - left_panel.shape[0], left_width, 3), dtype=np.uint8)
        left_panel = np.vstack([left_panel, padding])
    if right_panel.shape[0] < max_height:
        padding = np.zeros((max_height - right_panel.shape[0], right_width, 3), dtype=np.uint8)
        right_panel = np.vstack([right_panel, padding])
    
    bottom_row = np.hstack([left_panel, right_panel])
    
    # Pad bottom row to match top row width if needed
    if bottom_row.shape[1] < top_row.shape[1]:
        padding_width = top_row.shape[1] - bottom_row.shape[1]
        padding = np.zeros((bottom_row.shape[0], padding_width, 3), dtype=np.uint8)
        bottom_row = np.hstack([bottom_row, padding])
    elif bottom_row.shape[1] > top_row.shape[1]:
        bottom_row = bottom_row[:, :top_row.shape[1]]
    
    # Combine all
    comparison = np.vstack([top_row, bottom_row])
    
    return comparison


def process_image_split_visualization(image_path: Path, config: Config, 
                                    output_dir: Path) -> Dict[str, Any]:
    """Process a single image and create split visualization."""
    print(f"Processing: {image_path.name}")
    
    try:
        # Load image
        image = load_image(image_path)
        
        # Get detailed gutter analysis
        gutter_info = find_gutter_detailed(
            image, 
            config.gutter_search_start, 
            config.gutter_search_end,
            config.min_gutter_width
        )
        
        # Split the image
        left_page, right_page = split_two_page_image(
            image, 
            config.gutter_search_start, 
            config.gutter_search_end
        )
        
        # Create overlay
        overlay = draw_split_overlay(image, gutter_info)
        
        # Create gutter analysis plot
        gutter_plot = create_gutter_analysis_plot(gutter_info, image.shape[1])
        
        # Create comparison
        comparison = create_split_comparison(image, left_page, right_page, overlay, gutter_plot)
        
        # Save outputs
        base_name = image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = {
            'original': str(output_dir / f"{base_name}_original.jpg"),
            'overlay': str(output_dir / f"{base_name}_split_overlay.jpg"),
            'left_page': str(output_dir / f"{base_name}_left_page.jpg"),
            'right_page': str(output_dir / f"{base_name}_right_page.jpg"),
            'gutter_plot': str(output_dir / f"{base_name}_gutter_analysis.jpg"),
            'comparison': str(output_dir / f"{base_name}_split_comparison.jpg")
        }
        
        cv2.imwrite(output_files['original'], image)
        cv2.imwrite(output_files['overlay'], overlay)
        cv2.imwrite(output_files['left_page'], left_page)
        cv2.imwrite(output_files['right_page'], right_page)
        cv2.imwrite(output_files['gutter_plot'], gutter_plot)
        cv2.imwrite(output_files['comparison'], comparison)
        
        # Save analysis data
        analysis_file = output_dir / f"{base_name}_split_analysis.json"
        analysis_data = {
            'gutter_info': {k: float(v) if isinstance(v, np.floating) else 
                           int(v) if isinstance(v, np.integer) else
                           v.tolist() if isinstance(v, np.ndarray) else v 
                           for k, v in gutter_info.items()},
            'image_dimensions': {'width': image.shape[1], 'height': image.shape[0]},
            'page_dimensions': {
                'left': {'width': left_page.shape[1], 'height': left_page.shape[0]},
                'right': {'width': right_page.shape[1], 'height': right_page.shape[0]}
            }
        }
        
        with open(analysis_file, 'w') as f:
            json.dump(convert_numpy_types(analysis_data), f, indent=2)
        
        output_files['analysis'] = str(analysis_file)
        
        result = {
            'image_name': image_path.name,
            'success': True,
            'gutter_info': gutter_info,
            'output_files': output_files
        }
        
        print(f"  SUCCESS: Gutter at X={gutter_info['gutter_x']}, strength={gutter_info['gutter_strength']:.3f}")
        print(f"  SUCCESS: Width check: {gutter_info['meets_min_width']} ({gutter_info['gutter_width']}px)")
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            'image_name': image_path.name,
            'success': False,
            'error': str(e)
        }


def main():
    """Main function for page split visualization."""
    parser = argparse.ArgumentParser(description="Visualize page splitting results")
    parser.add_argument("images", nargs="*", 
                       default=["input/raw_images/Wang2017_Page_001.jpg"],
                       help="Images to visualize")
    parser.add_argument("--test-images", action="store_true",
                       help="Process all images in input/test_images directory")
    parser.add_argument("--output-dir", default="page_split_visualization",
                       help="Output directory for visualizations")
    
    # Parameter options
    parser.add_argument("--gutter-start", type=float, default=0.4,
                       help="Gutter search start position (0.0-1.0)")
    parser.add_argument("--gutter-end", type=float, default=0.6,
                       help="Gutter search end position (0.0-1.0)")
    parser.add_argument("--min-gutter-width", type=int, default=50,
                       help="Minimum gutter width in pixels")
    
    args = parser.parse_args()
    
    # Determine which images to process
    if args.test_images:
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
    
    # Create configuration
    config = Config(
        gutter_search_start=args.gutter_start,
        gutter_search_end=args.gutter_end,
        min_gutter_width=args.min_gutter_width,
        verbose=False
    )
    output_dir = Path(args.output_dir)
    
    print(f"Visualizing page splitting on {len(image_paths)} images")
    if args.test_images:
        print(f"Batch mode: Processing all images from test_images directory")
    print(f"Parameters:")
    print(f"  - Gutter search range: {config.gutter_search_start:.2f} - {config.gutter_search_end:.2f}")
    print(f"  - Minimum gutter width: {config.min_gutter_width}px")
    print(f"Output directory: {output_dir}")
    print()
    
    # Process all images
    results = []
    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Processing: {image_path.name}")
        result = process_image_split_visualization(image_path, config, output_dir)
        results.append(result)
    
    # Summary
    successful_results = [r for r in results if r['success']]
    print(f"\n{'='*60}")
    print("PAGE SPLIT VISUALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {len(successful_results)}/{len(image_paths)} images")
    
    if successful_results:
        avg_strength = sum(r['gutter_info']['gutter_strength'] for r in successful_results) / len(successful_results)
        print(f"Average gutter strength: {avg_strength:.3f}")
        
        width_ok_count = sum(1 for r in successful_results if r['gutter_info']['meets_min_width'])
        print(f"Images meeting min width requirement: {width_ok_count}/{len(successful_results)}")
    
    print(f"\nOutput files saved to: {output_dir}")
    print(f"Review the '_split_comparison.jpg' files to assess splitting quality")
    
    # Save summary
    summary_file = output_dir / "split_visualization_summary.json"
    summary_data = {
        'timestamp': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
        'config_parameters': {
            'gutter_search_start': config.gutter_search_start,
            'gutter_search_end': config.gutter_search_end,
            'min_gutter_width': config.min_gutter_width
        },
        'results': results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(convert_numpy_types(summary_data), f, indent=2)
    
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()