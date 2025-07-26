#!/usr/bin/env python3
"""
Table Lines Visualization Script

This script visualizes the table line detection process, helping you assess 
morphological operations and line detection parameters.
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

from ocr.config import Config
from ocr.utils import load_image, detect_table_lines


def detect_table_lines_detailed(image: np.ndarray, min_line_length: int = 100, 
                               max_line_gap: int = 10, kernel_h_size: int = 40,
                               kernel_v_size: int = 40, hough_threshold: int = 50) -> Dict[str, Any]:
    """Enhanced table line detection with detailed analysis."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Apply morphological operations to enhance lines
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_h_size, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_v_size))
    
    # Detect horizontal lines
    horizontal = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_h)
    h_lines = cv2.HoughLinesP(horizontal, 1, np.pi/180, threshold=hough_threshold,
                             minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    # Detect vertical lines
    vertical = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_v)
    v_lines = cv2.HoughLinesP(vertical, 1, np.pi/180, threshold=hough_threshold,
                             minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    # Convert to list of tuples
    h_lines_list = [tuple(line[0]) for line in h_lines] if h_lines is not None else []
    v_lines_list = [tuple(line[0]) for line in v_lines] if v_lines is not None else []
    
    # Calculate line statistics
    h_line_lengths = []
    v_line_lengths = []
    
    for x1, y1, x2, y2 in h_lines_list:
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        h_line_lengths.append(length)
    
    for x1, y1, x2, y2 in v_lines_list:
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        v_line_lengths.append(length)
    
    # Find potential table boundaries
    h_bounds = None
    v_bounds = None
    
    if h_lines_list:
        h_y_coords = []
        for x1, y1, x2, y2 in h_lines_list:
            h_y_coords.extend([y1, y2])
        h_bounds = (min(h_y_coords), max(h_y_coords))
    
    if v_lines_list:
        v_x_coords = []
        for x1, y1, x2, y2 in v_lines_list:
            v_x_coords.extend([x1, x2])
        v_bounds = (min(v_x_coords), max(v_x_coords))
    
    return {
        'horizontal_morph': horizontal,
        'vertical_morph': vertical,
        'h_lines': h_lines_list,
        'v_lines': v_lines_list,
        'h_line_count': len(h_lines_list),
        'v_line_count': len(v_lines_list),
        'h_line_lengths': h_line_lengths,
        'v_line_lengths': v_line_lengths,
        'h_avg_length': np.mean(h_line_lengths) if h_line_lengths else 0,
        'v_avg_length': np.mean(v_line_lengths) if v_line_lengths else 0,
        'h_bounds': h_bounds,
        'v_bounds': v_bounds,
        'has_table_structure': len(h_lines_list) > 0 and len(v_lines_list) > 0,
        'config_used': {
            'min_line_length': min_line_length,
            'max_line_gap': max_line_gap,
            'kernel_h_size': kernel_h_size,
            'kernel_v_size': kernel_v_size,
            'hough_threshold': hough_threshold
        }
    }


def draw_table_lines_overlay(image: np.ndarray, line_info: Dict[str, Any]) -> np.ndarray:
    """Draw overlay showing detected table lines."""
    overlay = image.copy()
    
    # Draw horizontal lines in red
    for x1, y1, x2, y2 in line_info['h_lines']:
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Draw vertical lines in blue
    for x1, y1, x2, y2 in line_info['v_lines']:
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Draw potential table boundary
    if line_info['h_bounds'] and line_info['v_bounds']:
        h_min, h_max = line_info['h_bounds']
        v_min, v_max = line_info['v_bounds']
        cv2.rectangle(overlay, (v_min, h_min), (v_max, h_max), (0, 255, 0), 3)
    
    # Add info text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    text_lines = [
        f"Horizontal lines: {line_info['h_line_count']}",
        f"Vertical lines: {line_info['v_line_count']}",
        f"H avg length: {line_info['h_avg_length']:.1f}px",
        f"V avg length: {line_info['v_avg_length']:.1f}px",
        f"Table structure: {line_info['has_table_structure']}"
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
        color = (255, 255, 255) if i < 4 else ((0, 255, 0) if line_info['has_table_structure'] else (0, 0, 255))
        cv2.putText(overlay, line, (15, text_y_start + i*35), 
                   font, font_scale, color, thickness)
    
    return overlay


def create_morphological_visualization(line_info: Dict[str, Any]) -> np.ndarray:
    """Create visualization of morphological operations."""
    h_morph = line_info['horizontal_morph']
    v_morph = line_info['vertical_morph']
    
    # Normalize for better visualization
    h_morph_norm = cv2.normalize(h_morph, None, 0, 255, cv2.NORM_MINMAX)
    v_morph_norm = cv2.normalize(v_morph, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert to color for combination
    h_colored = cv2.applyColorMap(h_morph_norm, cv2.COLORMAP_HOT)
    v_colored = cv2.applyColorMap(v_morph_norm, cv2.COLORMAP_WINTER)
    
    # Combine horizontally
    combined = np.hstack([h_colored, v_colored])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Horizontal Lines", (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, "Vertical Lines", (h_colored.shape[1] + 10, 30), font, 0.8, (255, 255, 255), 2)
    
    return combined


def create_line_statistics_plot(line_info: Dict[str, Any]) -> np.ndarray:
    """Create a plot showing line length distributions."""
    plot_width = 600
    plot_height = 400
    plot_img = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255
    
    h_lengths = line_info['h_line_lengths']
    v_lengths = line_info['v_line_lengths']
    
    if not h_lengths and not v_lengths:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(plot_img, "No lines detected", (200, 200), font, 1.0, (0, 0, 0), 2)
        return plot_img
    
    # Create histograms
    all_lengths = h_lengths + v_lengths
    if all_lengths:
        min_length = min(all_lengths)
        max_length = max(all_lengths)
        
        # Create bins
        num_bins = 20
        bin_width = (max_length - min_length) / num_bins if max_length > min_length else 1
        
        h_hist = np.zeros(num_bins)
        v_hist = np.zeros(num_bins)
        
        # Fill histograms
        for length in h_lengths:
            bin_idx = min(int((length - min_length) / bin_width), num_bins - 1)
            h_hist[bin_idx] += 1
        
        for length in v_lengths:
            bin_idx = min(int((length - min_length) / bin_width), num_bins - 1)
            v_hist[bin_idx] += 1
        
        # Draw histograms
        max_count = max(max(h_hist), max(v_hist)) if max(h_hist) > 0 or max(v_hist) > 0 else 1
        bar_width = plot_width // (num_bins * 2)
        
        for i in range(num_bins):
            # Horizontal lines (red)
            h_height = int((h_hist[i] / max_count) * (plot_height - 100)) if max_count > 0 else 0
            x1 = i * bar_width * 2
            y1 = plot_height - 50 - h_height
            cv2.rectangle(plot_img, (x1, y1), (x1 + bar_width - 1, plot_height - 50), (0, 0, 255), -1)
            
            # Vertical lines (blue)
            v_height = int((v_hist[i] / max_count) * (plot_height - 100)) if max_count > 0 else 0
            x2 = x1 + bar_width
            y2 = plot_height - 50 - v_height
            cv2.rectangle(plot_img, (x2, y2), (x2 + bar_width - 1, plot_height - 50), (255, 0, 0), -1)
    
    # Add labels and legend
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(plot_img, "Line Length Distribution", (10, 30), font, 0.8, (0, 0, 0), 2)
    cv2.putText(plot_img, f"H lines: {len(h_lengths)} (avg: {line_info['h_avg_length']:.1f})", 
               (10, plot_height - 20), font, 0.6, (0, 0, 255), 1)
    cv2.putText(plot_img, f"V lines: {len(v_lengths)} (avg: {line_info['v_avg_length']:.1f})", 
               (10, plot_height - 5), font, 0.6, (255, 0, 0), 1)
    
    return plot_img


def create_table_lines_comparison(original: np.ndarray, overlay: np.ndarray,
                                 morph_vis: np.ndarray, stats_plot: np.ndarray,
                                 line_info: Dict[str, Any]) -> np.ndarray:
    """Create a comprehensive comparison showing all table line detection results."""
    target_height = 500
    
    # Resize main images
    scale = target_height / original.shape[0]
    new_width = int(original.shape[1] * scale)
    
    orig_resized = cv2.resize(original, (new_width, target_height))
    overlay_resized = cv2.resize(overlay, (new_width, target_height))
    
    # Resize morphological visualization to match width
    morph_scale = new_width / morph_vis.shape[1]
    morph_height = int(morph_vis.shape[0] * morph_scale)
    morph_resized = cv2.resize(morph_vis, (new_width, morph_height))
    
    # Resize statistics plot
    stats_scale = new_width / stats_plot.shape[1]
    stats_height = int(stats_plot.shape[0] * stats_scale)
    stats_resized = cv2.resize(stats_plot, (new_width, stats_height))
    
    # Create labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_height = 40
    
    def create_label(text: str, width: int, color: tuple = (255, 255, 255)) -> np.ndarray:
        label_img = np.zeros((label_height, width, 3), dtype=np.uint8)
        cv2.putText(label_img, text, (10, 25), font, 0.8, color, 2)
        return label_img
    
    # Create labels
    orig_label = create_label("Original", new_width)
    overlay_label = create_label("Detected Lines", new_width)
    morph_label = create_label("Morphological Operations", new_width)
    stats_label = create_label("Line Statistics", new_width)
    
    # Combine all components
    comparison = np.vstack([
        orig_label, orig_resized,
        overlay_label, overlay_resized,
        morph_label, morph_resized,
        stats_label, stats_resized
    ])
    
    return comparison


def process_image_table_lines_visualization(image_path: Path, config: Config, 
                                          output_dir: Path, kernel_h_size: int = 40,
                                          kernel_v_size: int = 40, hough_threshold: int = 50) -> Dict[str, Any]:
    """Process a single image and create table lines visualization."""
    print(f"Processing: {image_path.name}")
    
    try:
        # Load image
        image = load_image(image_path)
        
        # Detect table lines with detailed analysis
        line_info = detect_table_lines_detailed(
            image, 
            config.min_line_length, 
            config.max_line_gap,
            kernel_h_size,
            kernel_v_size,
            hough_threshold
        )
        
        # Create visualizations
        overlay = draw_table_lines_overlay(image, line_info)
        morph_vis = create_morphological_visualization(line_info)
        stats_plot = create_line_statistics_plot(line_info)
        
        # Create comparison
        comparison = create_table_lines_comparison(image, overlay, morph_vis, stats_plot, line_info)
        
        # Save outputs
        base_name = image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = {
            'original': str(output_dir / f"{base_name}_original.jpg"),
            'overlay': str(output_dir / f"{base_name}_table_lines.jpg"),
            'morphology': str(output_dir / f"{base_name}_morphology.jpg"),
            'statistics': str(output_dir / f"{base_name}_line_stats.jpg"),
            'comparison': str(output_dir / f"{base_name}_table_lines_comparison.jpg")
        }
        
        cv2.imwrite(output_files['original'], image)
        cv2.imwrite(output_files['overlay'], overlay)
        cv2.imwrite(output_files['morphology'], morph_vis)
        cv2.imwrite(output_files['statistics'], stats_plot)
        cv2.imwrite(output_files['comparison'], comparison)
        
        # Save analysis data
        analysis_file = output_dir / f"{base_name}_table_lines_analysis.json"
        analysis_data = {
            'line_detection': {k: v for k, v in line_info.items() 
                             if k not in ['horizontal_morph', 'vertical_morph']},
            'config_used': {
                'min_line_length': config.min_line_length,
                'max_line_gap': config.max_line_gap,
                'kernel_h_size': kernel_h_size,
                'kernel_v_size': kernel_v_size,
                'hough_threshold': hough_threshold
            }
        }
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        output_files['analysis'] = str(analysis_file)
        
        result = {
            'image_name': image_path.name,
            'success': True,
            'line_info': line_info,
            'output_files': output_files
        }
        
        print(f"  SUCCESS: H lines: {line_info['h_line_count']}, V lines: {line_info['v_line_count']}")
        print(f"  SUCCESS: Table structure detected: {line_info['has_table_structure']}")
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            'image_name': image_path.name,
            'success': False,
            'error': str(e)
        }


def main():
    """Main function for table lines visualization."""
    parser = argparse.ArgumentParser(description="Visualize table line detection results")
    parser.add_argument("images", nargs="*", 
                       default=["input/raw_images/Wang2017_Page_001.jpg"],
                       help="Images to visualize")
    parser.add_argument("--output-dir", default="table_lines_visualization",
                       help="Output directory for visualizations")
    
    # Parameter options
    parser.add_argument("--min-line-length", type=int, default=100,
                       help="Minimum line length for detection")
    parser.add_argument("--max-line-gap", type=int, default=10,
                       help="Maximum gap in line detection")
    parser.add_argument("--kernel-h-size", type=int, default=40,
                       help="Horizontal morphological kernel size")
    parser.add_argument("--kernel-v-size", type=int, default=40,
                       help="Vertical morphological kernel size")
    parser.add_argument("--hough-threshold", type=int, default=50,
                       help="Hough transform threshold")
    
    args = parser.parse_args()
    
    # Resolve image paths
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
        min_line_length=args.min_line_length,
        max_line_gap=args.max_line_gap,
        verbose=False
    )
    output_dir = Path(args.output_dir)
    
    print(f"Visualizing table line detection on {len(image_paths)} images")
    print(f"Parameters:")
    print(f"  - Min line length: {config.min_line_length}px")
    print(f"  - Max line gap: {config.max_line_gap}px")
    print(f"  - H kernel size: {args.kernel_h_size}px")
    print(f"  - V kernel size: {args.kernel_v_size}px")
    print(f"  - Hough threshold: {args.hough_threshold}")
    
    # Process all images
    results = []
    for image_path in image_paths:
        result = process_image_table_lines_visualization(
            image_path, config, output_dir, 
            args.kernel_h_size, args.kernel_v_size, args.hough_threshold
        )
        results.append(result)
    
    # Summary
    successful_results = [r for r in results if r['success']]
    print(f"\n{'='*60}")
    print("TABLE LINES VISUALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {len(successful_results)}/{len(image_paths)} images")
    
    if successful_results:
        table_structure_count = sum(1 for r in successful_results if r['line_info']['has_table_structure'])
        print(f"Images with table structure: {table_structure_count}/{len(successful_results)}")
        
        total_h_lines = sum(r['line_info']['h_line_count'] for r in successful_results)
        total_v_lines = sum(r['line_info']['v_line_count'] for r in successful_results)
        print(f"Total horizontal lines: {total_h_lines}")
        print(f"Total vertical lines: {total_v_lines}")
        
        if table_structure_count > 0:
            avg_h_length = sum(r['line_info']['h_avg_length'] for r in successful_results if r['line_info']['h_avg_length'] > 0) / max(1, sum(1 for r in successful_results if r['line_info']['h_avg_length'] > 0))
            avg_v_length = sum(r['line_info']['v_avg_length'] for r in successful_results if r['line_info']['v_avg_length'] > 0) / max(1, sum(1 for r in successful_results if r['line_info']['v_avg_length'] > 0))
            print(f"Average H line length: {avg_h_length:.1f}px")
            print(f"Average V line length: {avg_v_length:.1f}px")
    
    print(f"\nOutput files saved to: {output_dir}")
    print(f"Review the '_table_lines_comparison.jpg' files to assess line detection quality")
    
    # Save summary
    summary_file = output_dir / "table_lines_visualization_summary.json"
    summary_data = {
        'timestamp': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
        'config_parameters': {
            'min_line_length': config.min_line_length,
            'max_line_gap': config.max_line_gap,
            'kernel_h_size': args.kernel_h_size,
            'kernel_v_size': args.kernel_v_size,
            'hough_threshold': args.hough_threshold
        },
        'results': results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()