#!/usr/bin/env python3
"""
Deskew Visualization Script

This script visualizes the deskewing process, helping you assess angle detection
quality and adjust deskewing parameters.
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

from src.ocr_pipeline.config import Stage1Config, Stage2Config
import src.ocr_pipeline.utils as ocr_utils
from output_manager import get_default_output_manager, organize_visualization_output, get_test_images, convert_numpy_types


def load_config_from_file(config_path: Path = None, stage: int = 1):
    """Load configuration from JSON file or use defaults."""
    if config_path is None:
        # Use default config based on stage
        if stage == 2:
            config_path = Path("configs/stage2_default.json")
        else:
            config_path = Path("configs/stage1_default.json")
    
    if config_path.exists():
        if stage == 2:
            return Stage2Config.from_json(config_path)
        else:
            return Stage1Config.from_json(config_path)
    else:
        print(f"Warning: Config file {config_path} not found, using hardcoded defaults")
        if stage == 2:
            return Stage2Config()
        else:
            return Stage1Config()


def analyze_skew_detailed(image: np.ndarray, angle_range: int = 10, 
                         angle_step: float = 0.2) -> Dict[str, Any]:
    """Enhanced skew analysis with detailed line detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Find edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is None:
        return {
            'has_lines': False,
            'rotation_angle': 0,
            'line_count': 0,
            'angles': [],
            'confidence': 0,
            'edges': edges
        }
    
    # Calculate angles of detected lines
    angles = []
    for rho, theta in lines[:, 0]:
        angle = theta * 180 / np.pi
        # Convert to -45 to 45 degree range
        if angle > 90:
            angle = angle - 180
        elif angle > 45:
            angle = angle - 90
        elif angle < -45:
            angle = angle + 90
        angles.append(angle)
    
    # Calculate rotation angle and confidence
    if not angles:
        rotation_angle = 0
        confidence = 0
    else:
        # Use median angle for rotation
        rotation_angle = np.median(angles)
        # Confidence based on consistency of angles
        angle_std = np.std(angles)
        confidence = max(0, 1 - (angle_std / 10))  # Normalize std to 0-1 scale
    
    # Test different angle ranges for sensitivity analysis
    angle_histogram = {}
    for test_angle in np.arange(-angle_range, angle_range + angle_step, angle_step):
        nearby_angles = [a for a in angles if abs(a - test_angle) <= angle_step]
        angle_histogram[round(test_angle, 1)] = len(nearby_angles)
    
    return {
        'has_lines': True,
        'rotation_angle': rotation_angle,
        'line_count': len(lines),
        'angles': angles,
        'confidence': confidence,
        'angle_std': angle_std if angles else 0,
        'angle_histogram': angle_histogram,
        'edges': edges,
        'will_rotate': abs(rotation_angle) >= 0.5
    }


def draw_line_detection_overlay(image: np.ndarray, skew_info: Dict[str, Any]) -> np.ndarray:
    """Draw overlay showing detected lines and skew analysis."""
    overlay = image.copy()
    
    if not skew_info['has_lines']:
        # Draw "No lines detected" message
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay, "No lines detected", (50, 50), font, 1.0, (0, 0, 255), 2)
        return overlay
    
    # Recreate line detection for visualization
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is not None:
        # Draw detected lines
        for i, (rho, theta) in enumerate(lines[:, 0]):
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            # Color based on angle (red for more skewed lines)
            angle = theta * 180 / np.pi
            if angle > 90:
                angle = angle - 180
            color_intensity = min(255, int(abs(angle) * 5))
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255 - color_intensity, color_intensity), 1)
    
    # Add analysis text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    text_lines = [
        f"Lines detected: {skew_info['line_count']}",
        f"Rotation angle: {skew_info['rotation_angle']:.2f}°",
        f"Confidence: {skew_info['confidence']:.3f}",
        f"Angle std: {skew_info['angle_std']:.2f}°",
        f"Will rotate: {skew_info['will_rotate']}"
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
        color = (255, 255, 255) if i < 4 else ((0, 255, 0) if skew_info['will_rotate'] else (0, 0, 255))
        cv2.putText(overlay, line, (15, text_y_start + i*35), 
                   font, font_scale, color, thickness)
    
    return overlay


def create_angle_histogram_plot(skew_info: Dict[str, Any]) -> np.ndarray:
    """Create a plot showing the distribution of detected angles."""
    if not skew_info['has_lines']:
        plot_img = np.ones((300, 600, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(plot_img, "No lines detected", (200, 150), font, 1.0, (0, 0, 0), 2)
        return plot_img
    
    angles = skew_info['angles']
    histogram = skew_info['angle_histogram']
    
    plot_width = 600
    plot_height = 300
    plot_img = np.ones((plot_height, plot_width, 3), dtype=np.uint8) * 255
    
    if not angles:
        return plot_img
    
    # Create histogram visualization
    angle_range = max(histogram.keys()) - min(histogram.keys())
    if angle_range == 0:
        return plot_img
    
    max_count = max(histogram.values()) if histogram.values() else 1
    
    # Draw bars
    bar_width = plot_width // len(histogram)
    for i, (angle, count) in enumerate(sorted(histogram.items())):
        bar_height = int((count / max_count) * (plot_height - 80)) if max_count > 0 else 0
        x = i * bar_width
        y = plot_height - 40 - bar_height
        
        # Color based on angle value
        color_intensity = min(255, int(abs(angle) * 5))
        color = (color_intensity, 0, 255 - color_intensity)
        
        cv2.rectangle(plot_img, (x, y), (x + bar_width - 1, plot_height - 40), color, -1)
        cv2.rectangle(plot_img, (x, y), (x + bar_width - 1, plot_height - 40), (0, 0, 0), 1)
    
    # Mark the selected angle
    rotation_angle = skew_info['rotation_angle']
    angles_list = sorted(histogram.keys())
    if rotation_angle in angles_list:
        angle_index = angles_list.index(rotation_angle)
        x = angle_index * bar_width + bar_width // 2
        cv2.line(plot_img, (x, 0), (x, plot_height - 40), (0, 255, 0), 3)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(plot_img, "Angle Distribution", (10, 25), font, 0.7, (0, 0, 0), 2)
    cv2.putText(plot_img, f"Selected: {rotation_angle:.2f}°", (10, plot_height - 10), font, 0.6, (0, 255, 0), 1)
    cv2.putText(plot_img, f"Range: {min(angles):.1f}° to {max(angles):.1f}°", (300, plot_height - 10), font, 0.6, (0, 0, 0), 1)
    
    return plot_img


def create_edge_visualization(edges: np.ndarray) -> np.ndarray:
    """Create a colored visualization of the edge detection."""
    # Convert edges to 3-channel for visualization
    edges_colored = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
    return edges_colored


def create_deskew_comparison(original: np.ndarray, deskewed: np.ndarray, 
                           overlay: np.ndarray, angle_plot: np.ndarray,
                           edges_vis: np.ndarray, skew_info: Dict[str, Any]) -> np.ndarray:
    """Create a comprehensive comparison showing all deskew results."""
    target_height = 500
    
    # Resize main images
    scale = target_height / original.shape[0]
    new_width = int(original.shape[1] * scale)
    
    orig_resized = cv2.resize(original, (new_width, target_height))
    deskewed_resized = cv2.resize(deskewed, (new_width, target_height))
    overlay_resized = cv2.resize(overlay, (new_width, target_height))
    edges_resized = cv2.resize(edges_vis, (new_width, target_height))
    
    # Resize plots to match width
    plot_scale = new_width / angle_plot.shape[1]
    plot_height = int(angle_plot.shape[0] * plot_scale)
    plot_resized = cv2.resize(angle_plot, (new_width, plot_height))
    
    # Create labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_height = 40
    
    def create_label(text: str, width: int, color: tuple = (255, 255, 255)) -> np.ndarray:
        label_img = np.zeros((label_height, width, 3), dtype=np.uint8)
        cv2.putText(label_img, text, (10, 25), font, 0.8, color, 2)
        return label_img
    
    # Create top row (analysis)
    orig_label = create_label("Original", new_width)
    overlay_label = create_label("Line Detection", new_width)
    edges_label = create_label("Edge Detection", new_width)
    
    top_row = np.vstack([
        orig_label, orig_resized,
        overlay_label, overlay_resized,
        edges_label, edges_resized
    ])
    
    # Create middle row (result and analysis)
    result_color = (0, 255, 0) if skew_info['will_rotate'] else (255, 255, 255)
    deskewed_label = create_label(f"Deskewed ({skew_info['rotation_angle']:.2f}°)", new_width, result_color)
    plot_label = create_label("Angle Analysis", new_width)
    
    middle_row = np.vstack([
        deskewed_label, deskewed_resized,
        plot_label, plot_resized
    ])
    
    # Combine rows
    max_width = max(top_row.shape[1], middle_row.shape[1])
    
    # Pad to same width if needed
    if top_row.shape[1] < max_width:
        padding = np.zeros((top_row.shape[0], max_width - top_row.shape[1], 3), dtype=np.uint8)
        top_row = np.hstack([top_row, padding])
    if middle_row.shape[1] < max_width:
        padding = np.zeros((middle_row.shape[0], max_width - middle_row.shape[1], 3), dtype=np.uint8)
        middle_row = np.hstack([middle_row, padding])
    
    comparison = np.vstack([top_row, middle_row])
    
    return comparison


def process_image_deskew_visualization(image_path: Path, config, 
                                     output_dir: Path, use_organized_output: bool = True) -> Dict[str, Any]:
    """Process a single image and create deskew visualization."""
    print(f"Processing: {image_path.name}")
    
    try:
        # Load image
        image = ocr_utils.load_image(image_path)
        
        # Analyze skew
        skew_info = analyze_skew_detailed(image, config.angle_range, config.angle_step)
        
        # Deskew the image
        deskewed, detected_angle = ocr_utils.deskew_image(image, config.angle_range, config.angle_step, config.min_angle_correction)
        
        # Create visualizations
        overlay = draw_line_detection_overlay(image, skew_info)
        angle_plot = create_angle_histogram_plot(skew_info)
        edges_vis = create_edge_visualization(skew_info['edges'])
        
        # Create comparison
        comparison = create_deskew_comparison(image, deskewed, overlay, angle_plot, edges_vis, skew_info)
        
        # Save outputs
        base_name = image_path.stem
        
        if use_organized_output:
            # Create temporary files first
            temp_dir = Path(output_dir) / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            temp_files = {
                'original': str(temp_dir / f"{base_name}_original.jpg"),
                'deskewed': str(temp_dir / f"{base_name}_deskewed.jpg"),
                'overlay': str(temp_dir / f"{base_name}_line_detection.jpg"),
                'edges': str(temp_dir / f"{base_name}_edges.jpg"),
                'angle_plot': str(temp_dir / f"{base_name}_angle_analysis.jpg"),
                'comparison': str(temp_dir / f"{base_name}_deskew_comparison.jpg")
            }
            
            cv2.imwrite(temp_files['original'], image)
            cv2.imwrite(temp_files['deskewed'], deskewed)
            cv2.imwrite(temp_files['overlay'], overlay)
            cv2.imwrite(temp_files['edges'], edges_vis)
            cv2.imwrite(temp_files['angle_plot'], angle_plot)
            cv2.imwrite(temp_files['comparison'], comparison)
            
            # Prepare analysis data
            analysis_data = {
                'image_name': image_path.name,
                'skew_info': {k: float(v) if isinstance(v, np.floating) else 
                             int(v) if isinstance(v, np.integer) else
                             v.tolist() if isinstance(v, np.ndarray) and k != 'edges' else
                             v for k, v in skew_info.items() if k != 'edges'},
                'config_used': {
                    'angle_range': config.angle_range,
                    'angle_step': config.angle_step,
                    'min_angle_correction': config.min_angle_correction
                }
            }
            
            # Organize into structured output
            output_files = organize_visualization_output(
                'deskew', temp_files, analysis_data, output_dir
            )
            
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        else:
            # Use old flat structure
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_files = {
                'original': str(output_dir / f"{base_name}_original.jpg"),
                'deskewed': str(output_dir / f"{base_name}_deskewed.jpg"),
                'overlay': str(output_dir / f"{base_name}_line_detection.jpg"),
                'edges': str(output_dir / f"{base_name}_edges.jpg"),
                'angle_plot': str(output_dir / f"{base_name}_angle_analysis.jpg"),
                'comparison': str(output_dir / f"{base_name}_deskew_comparison.jpg")
            }
            
            cv2.imwrite(output_files['original'], image)
            cv2.imwrite(output_files['deskewed'], deskewed)
            cv2.imwrite(output_files['overlay'], overlay)
            cv2.imwrite(output_files['edges'], edges_vis)
            cv2.imwrite(output_files['angle_plot'], angle_plot)
            cv2.imwrite(output_files['comparison'], comparison)
            
            # Save analysis data
            analysis_file = output_dir / f"{base_name}_deskew_analysis.json"
            analysis_data = {
                'skew_info': {k: float(v) if isinstance(v, np.floating) else 
                             int(v) if isinstance(v, np.integer) else
                             v.tolist() if isinstance(v, np.ndarray) and k != 'edges' else
                             v for k, v in skew_info.items() if k != 'edges'},
                'config_used': {
                    'angle_range': config.angle_range,
                    'angle_step': config.angle_step,
                    'min_angle_correction': config.min_angle_correction
                }
            }
            
            with open(analysis_file, 'w') as f:
                json.dump(convert_numpy_types(analysis_data), f, indent=2)
            
            output_files['analysis'] = str(analysis_file)
        
        result = {
            'image_name': image_path.name,
            'success': True,
            'skew_info': skew_info,
            'output_files': output_files
        }
        
        print(f"  SUCCESS: Rotation angle: {skew_info['rotation_angle']:.2f}°, confidence: {skew_info['confidence']:.3f}")
        print(f"  SUCCESS: Lines detected: {skew_info['line_count']}, will rotate: {skew_info['will_rotate']}")
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            'image_name': image_path.name,
            'success': False,
            'error': str(e)
        }


def main():
    """Main function for deskew visualization."""
    parser = argparse.ArgumentParser(description="Visualize deskewing results")
    parser.add_argument("images", nargs="*", 
                       default=["input/raw_images/Wang2017_Page_001.jpg"],
                       help="Images to visualize")
    parser.add_argument("--test-images", action="store_true",
                       help="Process all images in input/test_images directory")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory for visualizations (default: organized structure)")
    parser.add_argument("--flat-output", action="store_true",
                       help="Use flat output structure instead of organized folders")
    
    # Config file options
    parser.add_argument("--config-file", type=Path, default=None,
                       help="JSON config file to use (default: configs/stage1_default.json)")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1,
                       help="Use stage1 or stage2 default config (default: 1)")
    
    # Parameter overrides (take precedence over config file)
    parser.add_argument("--angle-range", type=int, default=None,
                       help="Maximum angle range for detection (degrees)")
    parser.add_argument("--angle-step", type=float, default=None,
                       help="Angle step for detection (degrees)")
    parser.add_argument("--min-angle-correction", type=float, default=None,
                       help="Minimum angle to trigger rotation (degrees)")
    
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
    
    # Load configuration from JSON file
    config = load_config_from_file(args.config_file, args.stage)
    
    # Apply command line parameter overrides
    if args.angle_range is not None:
        config.angle_range = args.angle_range
    if args.angle_step is not None:
        config.angle_step = args.angle_step
    if args.min_angle_correction is not None:
        config.min_angle_correction = args.min_angle_correction
    
    config.verbose = False  # Keep visualization quiet
    
    # Handle output directory
    if args.flat_output or args.output_dir:
        # Use specified directory or flat structure
        output_dir = Path(args.output_dir) if args.output_dir else Path("data/output/visualization/deskew")
        use_organized = False
    else:
        # Use organized structure
        manager = get_default_output_manager()
        output_dir = manager.create_run_directory("deskew")
        use_organized = True
    
    print(f"Visualizing deskewing on {len(image_paths)} images")
    if args.test_images:
        print(f"Batch mode: Processing all images from test_images directory")
    print(f"Parameters:")
    print(f"  - Angle range: ±{config.angle_range}°")
    print(f"  - Angle step: {config.angle_step}°")
    print(f"  - Min correction: {config.min_angle_correction}°")
    print(f"Output directory: {output_dir}")
    print()
    
    # Process all images
    results = []
    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Processing: {image_path.name}")
        result = process_image_deskew_visualization(image_path, config, output_dir, use_organized)
        results.append(result)
    
    # Summary
    successful_results = [r for r in results if r['success']]
    print(f"\n{'='*60}")
    print("DESKEW VISUALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {len(successful_results)}/{len(image_paths)} images")
    
    if successful_results:
        rotated_count = sum(1 for r in successful_results if r['skew_info']['will_rotate'])
        print(f"Images that will be rotated: {rotated_count}/{len(successful_results)}")
        
        if rotated_count > 0:
            avg_angle = sum(abs(r['skew_info']['rotation_angle']) for r in successful_results if r['skew_info']['will_rotate']) / rotated_count
            print(f"Average rotation angle: {avg_angle:.2f}°")
        
        avg_confidence = sum(r['skew_info']['confidence'] for r in successful_results) / len(successful_results)
        print(f"Average detection confidence: {avg_confidence:.3f}")
    
    print(f"\nOutput files saved to: {output_dir}")
    
    if use_organized:
        print(f"Use 'python visualization/check_results.py latest deskew --view' to view results")
        print(f"Use 'python visualization/check_results.py list' to see all runs")
    else:
        print(f"Review the '_deskew_comparison.jpg' files to assess deskewing quality")
    
    # Save summary (only for flat structure, organized structure handles this automatically)
    if not use_organized:
        summary_file = output_dir / "deskew_visualization_summary.json"
        summary_data = {
            'timestamp': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
            'config_parameters': {
                'angle_range': config.angle_range,
                'angle_step': config.angle_step,
                'min_angle_correction': config.min_angle_correction
            },
            'results': results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(convert_numpy_types(summary_data), f, indent=2)
        
        print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()