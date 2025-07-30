#!/usr/bin/env python3
"""
ROI Visualization Script

This script creates visual overlays showing detected ROI regions on your images,
helping you assess the quality of ROI detection and adjust parameters.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import sys
from typing import Dict, Any

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.ocr_pipeline.config import Stage1Config, Stage2Config
import src.ocr_pipeline.utils as ocr_utils
from output_manager import get_test_images, convert_numpy_types


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


def draw_roi_overlay(image: np.ndarray, roi_coords: Dict[str, Any], 
                    show_gabor: bool = False, config = None) -> np.ndarray:
    """Draw ROI detection overlay on image."""
    overlay = image.copy()
    height, width = image.shape[:2]
    
    # Create semi-transparent overlay
    roi_overlay = np.zeros_like(overlay)
    
    # Extract ROI coordinates
    left = roi_coords['roi_left']
    right = roi_coords['roi_right']
    top = roi_coords['roi_top']
    bottom = roi_coords['roi_bottom']
    
    # Draw ROI rectangle (green for included region)
    cv2.rectangle(roi_overlay, (left, top), (right, bottom), (0, 255, 0), -1)
    
    # Draw excluded regions (red overlay)
    if left > 0:  # Left exclusion
        cv2.rectangle(roi_overlay, (0, 0), (left, height), (0, 0, 255), -1)
    if right < width:  # Right exclusion
        cv2.rectangle(roi_overlay, (right, 0), (width, height), (0, 0, 255), -1)
    if top > 0:  # Top exclusion
        cv2.rectangle(roi_overlay, (0, 0), (width, top), (0, 0, 255), -1)
    if bottom < height:  # Bottom exclusion
        cv2.rectangle(roi_overlay, (0, bottom), (width, height), (0, 0, 255), -1)
    
    # Blend overlay with original image
    alpha = 0.3
    overlay = cv2.addWeighted(overlay, 1-alpha, roi_overlay, alpha, 0)
    
    # Draw ROI boundary lines
    cv2.rectangle(overlay, (left, top), (right, bottom), (0, 255, 0), 3)
    
    # Add coordinate text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    # ROI dimensions text
    roi_width = right - left
    roi_height = bottom - top
    coverage = (roi_width * roi_height) / (width * height) * 100
    
    text_lines = [
        f"ROI: ({left}, {top}) to ({right}, {bottom})",
        f"Size: {roi_width} x {roi_height}",
        f"Coverage: {coverage:.1f}%"
    ]
    
    # Draw text background
    text_y_start = 30
    for i, line in enumerate(text_lines):
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        cv2.rectangle(overlay, (10, text_y_start + i*40 - 25), 
                     (20 + text_size[0], text_y_start + i*40 + 10), 
                     (0, 0, 0), -1)
    
    # Draw text
    for i, line in enumerate(text_lines):
        cv2.putText(overlay, line, (15, text_y_start + i*40), 
                   font, font_scale, (255, 255, 255), thickness)
    
    # Show Gabor filter response if requested
    if show_gabor and config:
        gabor_mask = detect_roi_gabor(
            image, config.gabor_kernel_size, config.gabor_sigma,
            config.gabor_lambda, config.gabor_gamma, config.gabor_binary_threshold
        )
        
        # Create a small overlay for Gabor response
        gabor_small = cv2.resize(gabor_mask, (200, 150))
        gabor_colored = cv2.applyColorMap(gabor_small, cv2.COLORMAP_JET)
        
        # Overlay Gabor response in top-right corner
        y_offset = 10
        x_offset = width - 210
        overlay[y_offset:y_offset+150, x_offset:x_offset+200] = gabor_colored
        
        # Label for Gabor filter
        cv2.putText(overlay, "Gabor Response", (x_offset, y_offset-5), 
                   font, 0.6, (255, 255, 255), 2)
    
    return overlay


def create_comparison_grid(original: np.ndarray, roi_overlay: np.ndarray, 
                          cropped_roi: np.ndarray = None) -> np.ndarray:
    """Create a comparison grid showing original, overlay, and cropped versions."""
    # Resize images to same height for comparison
    target_height = 600
    
    # Resize original and overlay
    scale = target_height / original.shape[0]
    new_width = int(original.shape[1] * scale)
    
    orig_resized = cv2.resize(original, (new_width, target_height))
    overlay_resized = cv2.resize(roi_overlay, (new_width, target_height))
    
    # Create labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_height = 40
    label_img = np.zeros((label_height, new_width, 3), dtype=np.uint8)
    
    # Label for original
    orig_label = label_img.copy()
    cv2.putText(orig_label, "Original", (10, 25), font, 0.8, (255, 255, 255), 2)
    
    # Label for overlay
    overlay_label = label_img.copy()
    cv2.putText(overlay_label, "ROI Detection", (10, 25), font, 0.8, (0, 255, 0), 2)
    
    # Combine original and overlay
    left_panel = np.vstack([orig_label, orig_resized, overlay_label, overlay_resized])
    
    # Add cropped ROI if provided
    if cropped_roi is not None and cropped_roi.size > 0:
        # Resize cropped ROI to fit
        crop_scale = target_height / cropped_roi.shape[0]
        crop_width = int(cropped_roi.shape[1] * crop_scale)
        cropped_resized = cv2.resize(cropped_roi, (crop_width, target_height))
        
        # Create label for cropped
        crop_label = np.zeros((label_height, crop_width, 3), dtype=np.uint8)
        cv2.putText(crop_label, "Cropped ROI", (10, 25), font, 0.8, (0, 255, 255), 2)
        
        # Combine with cropped
        right_panel = np.vstack([crop_label, cropped_resized])
        
        # Pad to match height if needed
        height_diff = left_panel.shape[0] - right_panel.shape[0]
        if height_diff > 0:
            padding = np.zeros((height_diff, crop_width, 3), dtype=np.uint8)
            right_panel = np.vstack([right_panel, padding])
        
        # Combine horizontally
        comparison = np.hstack([left_panel, right_panel])
    else:
        comparison = left_panel
    
    return comparison


def save_debug_images(image: np.ndarray, config, output_dir: Path, base_name: str):
    """Save debug images for binarization and gabor steps."""
    # Convert to grayscale for processing
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Step 1: Save grayscale image
    cv2.imwrite(str(output_dir / f"{base_name}_01_grayscale.jpg"), gray_img)
    
    # Step 2: Create and save Gabor kernels visualization
    kernels = []
    for theta in [0, np.pi/2]:  # 0° (vertical), 90° (horizontal)
        kernel = cv2.getGaborKernel((config.gabor_kernel_size, config.gabor_kernel_size), 
                                   config.gabor_sigma, float(theta), 
                                   config.gabor_lambda, config.gabor_gamma, 0, ktype=cv2.CV_32F)
        kernels.append(kernel)
    
    # Visualize kernels
    kernel_vis = np.hstack([cv2.normalize(k, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) for k in kernels])
    cv2.imwrite(str(output_dir / f"{base_name}_02_gabor_kernels.jpg"), kernel_vis)
    
    # Step 3: Apply individual Gabor filters and save responses
    combined_response = np.zeros_like(gray_img, dtype=np.float32)
    for i, kernel in enumerate(kernels):
        filtered_img = cv2.filter2D(gray_img, cv2.CV_8UC3, kernel)
        combined_response += filtered_img.astype(np.float32)
        
        # Save individual filter response
        filter_normalized = cv2.normalize(filtered_img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        orientation = "vertical" if i == 0 else "horizontal"
        cv2.imwrite(str(output_dir / f"{base_name}_03_gabor_{orientation}.jpg"), filter_normalized)
    
    # Step 4: Save combined Gabor response
    gabor_response_map = cv2.normalize(combined_response, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(str(output_dir / f"{base_name}_04_gabor_combined.jpg"), gabor_response_map)
    
    # Step 5: Save binarized result
    _, binary_mask = cv2.threshold(gabor_response_map, config.gabor_binary_threshold, 255, cv2.THRESH_BINARY)
    cv2.imwrite(str(output_dir / f"{base_name}_05_gabor_binary.jpg"), binary_mask)
    
    # Step 6: Save threshold comparison
    threshold_comparison = np.hstack([gabor_response_map, binary_mask])
    cv2.imwrite(str(output_dir / f"{base_name}_06_threshold_comparison.jpg"), threshold_comparison)
    
    return {
        'grayscale': str(output_dir / f"{base_name}_01_grayscale.jpg"),
        'gabor_kernels': str(output_dir / f"{base_name}_02_gabor_kernels.jpg"),
        'gabor_vertical': str(output_dir / f"{base_name}_03_gabor_vertical.jpg"),
        'gabor_horizontal': str(output_dir / f"{base_name}_03_gabor_horizontal.jpg"),
        'gabor_combined': str(output_dir / f"{base_name}_04_gabor_combined.jpg"),
        'gabor_binary': str(output_dir / f"{base_name}_05_gabor_binary.jpg"),
        'threshold_comparison': str(output_dir / f"{base_name}_06_threshold_comparison.jpg")
    }


def process_image_visualization(image_path: Path, config, 
                              output_dir: Path, show_gabor: bool = False, save_debug: bool = False) -> Dict[str, Any]:
    """Process a single image and create visualization."""
    print(f"Processing: {image_path.name}")
    
    try:
        # Load image
        image = ocr_utils.load_image(image_path)
        
        # Detect ROI
        roi_coords = ocr_utils.detect_roi_for_image(image, config)
        
        # Create ROI overlay
        roi_overlay = draw_roi_overlay(image, roi_coords, show_gabor, config)
        
        # Crop to ROI
        left = roi_coords['roi_left']
        right = roi_coords['roi_right']
        top = roi_coords['roi_top']
        bottom = roi_coords['roi_bottom']
        cropped_roi = image[top:bottom, left:right]
        
        # Create comparison grid
        comparison = create_comparison_grid(image, roi_overlay, cropped_roi)
        
        # Save outputs
        base_name = image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual images
        cv2.imwrite(str(output_dir / f"{base_name}_original.jpg"), image)
        cv2.imwrite(str(output_dir / f"{base_name}_roi_overlay.jpg"), roi_overlay)
        cv2.imwrite(str(output_dir / f"{base_name}_roi_cropped.jpg"), cropped_roi)
        cv2.imwrite(str(output_dir / f"{base_name}_comparison.jpg"), comparison)
        
        # Save debug images if requested
        debug_files = {}
        if save_debug:
            debug_files = save_debug_images(image, config, output_dir, base_name)
        
        # Save ROI coordinates
        roi_file = output_dir / f"{base_name}_roi_coords.json"
        with open(roi_file, 'w') as f:
            json.dump(convert_numpy_types(roi_coords), f, indent=2)
        
        # Calculate metrics
        original_area = roi_coords['image_width'] * roi_coords['image_height']
        roi_area = (right - left) * (bottom - top)
        coverage = roi_area / original_area if original_area > 0 else 0
        
        output_files = {
            'original': str(output_dir / f"{base_name}_original.jpg"),
            'overlay': str(output_dir / f"{base_name}_roi_overlay.jpg"),
            'cropped': str(output_dir / f"{base_name}_roi_cropped.jpg"),
            'comparison': str(output_dir / f"{base_name}_comparison.jpg"),
            'coords': str(roi_file)
        }
        
        # Add debug files if they were saved
        if debug_files:
            output_files.update(debug_files)
        
        result = {
            'image_name': image_path.name,
            'success': True,
            'roi_coords': roi_coords,
            'coverage': coverage,
            'output_files': output_files
        }
        
        print(f"  SUCCESS: ROI Coverage: {coverage*100:.1f}%")
        print(f"  SUCCESS: ROI Size: {right-left} x {bottom-top}")
        print(f"  SUCCESS: Saved to: {output_dir}")
        
        return result
        
    except Exception as e:
        print(f"  ERROR: Error: {e}")
        return {
            'image_name': image_path.name,
            'success': False,
            'error': str(e)
        }


def main():
    """Main function for ROI visualization."""
    parser = argparse.ArgumentParser(description="Visualize ROI detection results")
    parser.add_argument("images", nargs="*", 
                       default=["input/raw_images/Wang2017_Page_001.jpg"],
                       help="Images to visualize")
    parser.add_argument("--test-images", action="store_true",
                       help="Process all images in input/test_images directory")
    parser.add_argument("--output-dir", default="data/output/visualization/roi",
                       help="Output directory for visualizations")
    parser.add_argument("--show-gabor", action="store_true",
                       help="Show Gabor filter response overlay")
    parser.add_argument("--save-debug", action="store_true",
                       help="Save debug images showing binarization and gabor steps")
    parser.add_argument("--config-params", type=str,
                       help="JSON string of config parameters to test")
    
    # Config file options
    parser.add_argument("--config-file", type=Path, default=None,
                       help="JSON config file to use (default: configs/stage1_default.json)")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1,
                       help="Use stage1 or stage2 default config (default: 1)")
    
    # Parameter overrides (take precedence over config file)
    parser.add_argument("--gabor-threshold", type=int, default=None,
                       help="Gabor binary threshold")
    parser.add_argument("--cut-strength", type=float, default=None,
                       help="Minimum cut strength")
    parser.add_argument("--confidence-threshold", type=float, default=None,
                       help="Minimum confidence threshold")
    
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
    
    # Handle config-params JSON override if provided
    config_params = {}
    if args.config_params:
        try:
            config_params = json.loads(args.config_params)
            for key, value in config_params.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        except json.JSONDecodeError:
            print("Warning: Invalid JSON in config-params, using command line args")
    
    # Apply command line parameter overrides
    if args.gabor_threshold is not None:
        config.gabor_binary_threshold = args.gabor_threshold
    if args.cut_strength is not None:
        config.roi_min_cut_strength = args.cut_strength
    if args.confidence_threshold is not None:
        config.roi_min_confidence_threshold = args.confidence_threshold
    
    config.verbose = False  # Keep visualization quiet
    config.enable_roi_detection = True  # Ensure ROI detection is enabled
    output_dir = Path(args.output_dir)
    
    print(f"Visualizing ROI detection on {len(image_paths)} images")
    if args.test_images:
        print(f"Batch mode: Processing all images from test_images directory")
    print(f"Parameters:")
    print(f"  - Gabor threshold: {config.gabor_binary_threshold}")
    print(f"  - Cut strength: {config.roi_min_cut_strength}")
    print(f"  - Confidence threshold: {config.roi_min_confidence_threshold}")
    print(f"  - Show Gabor filter: {args.show_gabor}")
    print(f"  - Save debug images: {args.save_debug}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Process all images
    results = []
    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Processing: {image_path.name}")
        result = process_image_visualization(
            image_path, config, output_dir, args.show_gabor, args.save_debug
        )
        results.append(result)
    
    # Summary
    successful_results = [r for r in results if r['success']]
    print(f"\n{'='*60}")
    print("VISUALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {len(successful_results)}/{len(image_paths)} images")
    
    if successful_results:
        avg_coverage = sum(r['coverage'] for r in successful_results) / len(successful_results)
        print(f"Average ROI coverage: {avg_coverage*100:.1f}%")
        
        coverage_range = [r['coverage'] for r in successful_results]
        print(f"Coverage range: {min(coverage_range)*100:.1f}% - {max(coverage_range)*100:.1f}%")
    
    print(f"\nOutput files saved to: {output_dir}")
    print(f"Review the '_comparison.jpg' files to assess ROI quality")
    
    # Save summary
    summary_file = output_dir / "visualization_summary.json"
    summary_data = {
        'timestamp': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
        'config_parameters': config_params,
        'results': results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(convert_numpy_types(summary_data), f, indent=2)
    
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()