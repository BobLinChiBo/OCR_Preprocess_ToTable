#!/usr/bin/env python3
"""
Pipeline Visualization Script

This is the master script that runs all pipeline step visualizations in sequence,
allowing you to see the complete processing workflow and adjust parameters for each step.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import sys
import time
from typing import Dict, Any, List

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from ocr.config import Config
from ocr.utils import (
    load_image, split_two_page_image, deskew_image, 
    detect_table_lines, crop_table_region, detect_roi_for_image, crop_to_roi
)

# Import individual visualization functions
from visualize_page_split import find_gutter_detailed, draw_split_overlay
from visualize_deskew import analyze_skew_detailed, draw_line_detection_overlay
from visualize_table_lines import detect_table_lines_detailed, draw_table_lines_overlay
from visualize_table_crop import analyze_table_crop_detailed, draw_crop_overlay
from visualize_roi import draw_roi_overlay
from visualization.output_manager import get_test_images, convert_numpy_types


def process_complete_pipeline(image_path: Path, config: Config, 
                            output_dir: Path, save_intermediates: bool = True) -> Dict[str, Any]:
    """Process a single image through the complete pipeline with visualization."""
    print(f"Processing complete pipeline for: {image_path.name}")
    
    try:
        results = {}
        base_name = image_path.stem
        
        # Step 1: Load original image
        print("  Step 1: Loading image...")
        original_image = load_image(image_path)
        results['original'] = {
            'image': original_image,
            'size': original_image.shape[:2]
        }
        
        # Step 2: Page splitting
        print("  Step 2: Page splitting...")
        gutter_info = find_gutter_detailed(
            original_image, 
            config.gutter_search_start, 
            config.gutter_search_end,
            config.min_gutter_width
        )
        left_page, right_page = split_two_page_image(
            original_image,
            config.gutter_search_start,
            config.gutter_search_end
        )
        split_overlay = draw_split_overlay(original_image, gutter_info)
        
        results['page_split'] = {
            'left_page': left_page,
            'right_page': right_page,
            'gutter_info': gutter_info,
            'overlay': split_overlay
        }
        
        # Process each page separately
        for page_idx, page in enumerate([left_page, right_page], 1):
            page_key = f'page_{page_idx}'
            print(f"  Processing {page_key}...")
            
            # Step 3: Deskewing
            print(f"    Step 3.{page_idx}: Deskewing...")
            skew_info = analyze_skew_detailed(page, config.angle_range, config.angle_step)
            deskewed_page = deskew_image(page, config.angle_range, config.angle_step)
            deskew_overlay = draw_line_detection_overlay(page, skew_info)
            
            # Step 4: ROI Detection (if enabled)
            processing_image = deskewed_page
            roi_overlay = None
            roi_info = None
            if config.enable_roi_detection:
                print(f"    Step 4.{page_idx}: ROI detection...")
                roi_info = detect_roi_for_image(deskewed_page, config)
                processing_image = crop_to_roi(deskewed_page, roi_info)
                roi_overlay = draw_roi_overlay(deskewed_page, roi_info, show_gabor=False, config=config)
            
            # Step 5: Table line detection
            print(f"    Step 5.{page_idx}: Table line detection...")
            line_info = detect_table_lines_detailed(
                processing_image, 
                config.min_line_length, 
                config.max_line_gap
            )
            lines_overlay = draw_table_lines_overlay(processing_image, line_info)
            
            # Step 6: Table cropping
            print(f"    Step 6.{page_idx}: Table cropping...")
            crop_info = analyze_table_crop_detailed(
                processing_image,
                config.min_line_length,
                config.max_line_gap
            )
            
            # Final result
            if line_info['h_lines'] and line_info['v_lines']:
                final_result = crop_table_region(processing_image, line_info['h_lines'], line_info['v_lines'])
            else:
                final_result = processing_image
            
            crop_overlay = draw_crop_overlay(processing_image, crop_info)
            
            results[page_key] = {
                'original_page': page,
                'deskewed': deskewed_page,
                'roi_processed': processing_image,
                'final_result': final_result,
                'skew_info': skew_info,
                'roi_info': roi_info,
                'line_info': line_info,
                'crop_info': crop_info,
                'overlays': {
                    'deskew': deskew_overlay,
                    'roi': roi_overlay,
                    'lines': lines_overlay,
                    'crop': crop_overlay
                }
            }
        
        # Save intermediate results if requested
        if save_intermediates:
            save_pipeline_intermediates(results, output_dir, base_name)
        
        # Create comprehensive comparison
        comparison = create_pipeline_comparison(results)
        
        # Save final comparison
        comparison_file = output_dir / f"{base_name}_pipeline_comparison.jpg"
        cv2.imwrite(str(comparison_file), comparison)
        
        # Save pipeline summary
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_file = output_dir / f"{base_name}_pipeline_summary.json"
        summary_data = create_pipeline_summary(results, config)
        
        with open(summary_file, 'w') as f:
            json.dump(convert_numpy_types(summary_data), f, indent=2)
        
        print(f"  SUCCESS: Pipeline complete for {image_path.name}")
        
        return {
            'image_name': image_path.name,
            'success': True,
            'results': results,
            'comparison_file': str(comparison_file),
            'summary_file': str(summary_file)
        }
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            'image_name': image_path.name,
            'success': False,
            'error': str(e)
        }


def save_pipeline_intermediates(results: Dict[str, Any], output_dir: Path, base_name: str):
    """Save all intermediate processing steps."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original and split results
    cv2.imwrite(str(output_dir / f"{base_name}_00_original.jpg"), results['original']['image'])
    cv2.imwrite(str(output_dir / f"{base_name}_01_split_overlay.jpg"), results['page_split']['overlay'])
    cv2.imwrite(str(output_dir / f"{base_name}_02_left_page.jpg"), results['page_split']['left_page'])
    cv2.imwrite(str(output_dir / f"{base_name}_03_right_page.jpg"), results['page_split']['right_page'])
    
    # Save each page's processing steps
    for page_idx in [1, 2]:
        page_key = f'page_{page_idx}'
        if page_key in results:
            page_data = results[page_key]
            
            # Save intermediate images
            cv2.imwrite(str(output_dir / f"{base_name}_04_{page_idx}_deskewed.jpg"), page_data['deskewed'])
            cv2.imwrite(str(output_dir / f"{base_name}_05_{page_idx}_deskew_overlay.jpg"), page_data['overlays']['deskew'])
            
            if page_data['roi_info']:
                cv2.imwrite(str(output_dir / f"{base_name}_06_{page_idx}_roi_processed.jpg"), page_data['roi_processed'])
                cv2.imwrite(str(output_dir / f"{base_name}_07_{page_idx}_roi_overlay.jpg"), page_data['overlays']['roi'])
            
            cv2.imwrite(str(output_dir / f"{base_name}_08_{page_idx}_lines_overlay.jpg"), page_data['overlays']['lines'])
            cv2.imwrite(str(output_dir / f"{base_name}_09_{page_idx}_crop_overlay.jpg"), page_data['overlays']['crop'])
            cv2.imwrite(str(output_dir / f"{base_name}_10_{page_idx}_final_result.jpg"), page_data['final_result'])


def create_pipeline_comparison(results: Dict[str, Any]) -> np.ndarray:
    """Create a comprehensive visual comparison of the entire pipeline."""
    target_height = 400
    
    # Get main images for comparison
    original = results['original']['image']
    left_page = results['page_1']['final_result']
    right_page = results['page_2']['final_result']
    
    # Resize images
    orig_scale = target_height / original.shape[0]
    orig_width = int(original.shape[1] * orig_scale)
    orig_resized = cv2.resize(original, (orig_width, target_height))
    
    left_scale = target_height / left_page.shape[0]
    left_width = int(left_page.shape[1] * left_scale)
    left_resized = cv2.resize(left_page, (left_width, target_height))
    
    right_scale = target_height / right_page.shape[0]
    right_width = int(right_page.shape[1] * right_scale)
    right_resized = cv2.resize(right_page, (right_width, target_height))
    
    # Create labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_height = 40
    
    def create_label(text: str, width: int) -> np.ndarray:
        label_img = np.zeros((label_height, width, 3), dtype=np.uint8)
        cv2.putText(label_img, text, (10, 25), font, 0.8, (255, 255, 255), 2)
        return label_img
    
    # Create labeled panels
    orig_panel = np.vstack([create_label("Original Document", orig_width), orig_resized])
    left_panel = np.vstack([create_label("Left Page (Processed)", left_width), left_resized])
    right_panel = np.vstack([create_label("Right Page (Processed)", right_width), right_resized])
    
    # Combine results horizontally
    results_row = np.hstack([left_panel, right_panel])
    
    # Pad to match widths
    max_width = max(orig_panel.shape[1], results_row.shape[1])
    
    if orig_panel.shape[1] < max_width:
        padding = np.zeros((orig_panel.shape[0], max_width - orig_panel.shape[1], 3), dtype=np.uint8)
        orig_panel = np.hstack([orig_panel, padding])
    
    if results_row.shape[1] < max_width:
        padding = np.zeros((results_row.shape[0], max_width - results_row.shape[1], 3), dtype=np.uint8)
        results_row = np.hstack([results_row, padding])
    
    # Combine vertically
    comparison = np.vstack([orig_panel, results_row])
    
    # Add summary statistics
    stats_height = 100
    stats_panel = np.zeros((stats_height, max_width, 3), dtype=np.uint8)
    
    # Add pipeline statistics
    stats_text = [
        f"Pipeline Summary:",
        f"Original size: {results['original']['size']}",
        f"Pages processed: 2",
        f"ROI enabled: {results['page_1']['roi_info'] is not None}"
    ]
    
    for i, text in enumerate(stats_text):
        cv2.putText(stats_panel, text, (20, 25 + i*20), font, 0.6, (255, 255, 255), 1)
    
    final_comparison = np.vstack([comparison, stats_panel])
    
    return final_comparison


def create_pipeline_summary(results: Dict[str, Any], config: Config) -> Dict[str, Any]:
    """Create a comprehensive summary of pipeline processing."""
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config_parameters': {
            'gutter_search_start': config.gutter_search_start,
            'gutter_search_end': config.gutter_search_end,
            'min_gutter_width': config.min_gutter_width,
            'angle_range': config.angle_range,
            'angle_step': config.angle_step,
            'min_angle_correction': config.min_angle_correction,
            'enable_roi_detection': config.enable_roi_detection,
            'min_line_length': config.min_line_length,
            'max_line_gap': config.max_line_gap
        },
        'processing_steps': {},
        'results_summary': {}
    }
    
    # Summarize each step
    if 'page_split' in results:
        summary['processing_steps']['page_split'] = {
            'gutter_strength': results['page_split']['gutter_info']['gutter_strength'],
            'gutter_width': results['page_split']['gutter_info']['gutter_width'],
            'meets_min_width': results['page_split']['gutter_info']['meets_min_width']
        }
    
    # Summarize each page
    for page_idx in [1, 2]:
        page_key = f'page_{page_idx}'
        if page_key in results:
            page_data = results[page_key]
            
            page_summary = {
                'deskew': {
                    'rotation_angle': page_data['skew_info']['rotation_angle'],
                    'confidence': page_data['skew_info']['confidence'],
                    'will_rotate': page_data['skew_info']['will_rotate']
                },
                'line_detection': {
                    'h_line_count': page_data['line_info']['h_line_count'],
                    'v_line_count': page_data['line_info']['v_line_count'],
                    'has_table_structure': page_data['line_info']['has_table_structure']
                },
                'cropping': {
                    'has_crop_region': page_data['crop_info']['has_crop_region'],
                    'coverage_ratio': page_data['crop_info']['coverage_ratio']
                }
            }
            
            if page_data['roi_info']:
                roi_coords = page_data['roi_info']
                roi_area = (roi_coords['roi_right'] - roi_coords['roi_left']) * (roi_coords['roi_bottom'] - roi_coords['roi_top'])
                total_area = roi_coords['image_width'] * roi_coords['image_height']
                page_summary['roi'] = {
                    'coordinates': roi_coords,
                    'coverage': roi_area / total_area if total_area > 0 else 0
                }
            
            summary['results_summary'][page_key] = page_summary
    
    return summary


def main():
    """Main function for complete pipeline visualization."""
    parser = argparse.ArgumentParser(description="Visualize complete OCR pipeline")
    parser.add_argument("images", nargs="*", 
                       default=["input/raw_images/Wang2017_Page_001.jpg"],
                       help="Images to process through pipeline")
    parser.add_argument("--test-images", action="store_true",
                       help="Process all images in input/test_images directory")
    parser.add_argument("--output-dir", default="pipeline_visualization",
                       help="Output directory for visualizations")
    parser.add_argument("--save-intermediates", action="store_true",
                       help="Save all intermediate processing steps")
    
    # Configuration parameters (using config defaults but allowing overrides)
    parser.add_argument("--config-file", type=str,
                       help="JSON file with configuration parameters")
    
    # Key parameter overrides
    parser.add_argument("--enable-roi", action="store_true", default=None,
                       help="Enable ROI detection")
    parser.add_argument("--disable-roi", action="store_true", default=None,
                       help="Disable ROI detection")
    parser.add_argument("--gutter-start", type=float,
                       help="Gutter search start position")
    parser.add_argument("--gutter-end", type=float,
                       help="Gutter search end position")
    parser.add_argument("--min-line-length", type=int,
                       help="Minimum line length for detection")
    
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
    config_params = {}
    
    # Load from file if specified
    if args.config_file and Path(args.config_file).exists():
        with open(args.config_file, 'r') as f:
            config_params = json.load(f)
    
    # Apply command line overrides
    if args.enable_roi:
        config_params['enable_roi_detection'] = True
    elif args.disable_roi:
        config_params['enable_roi_detection'] = False
    
    if args.gutter_start is not None:
        config_params['gutter_search_start'] = args.gutter_start
    if args.gutter_end is not None:
        config_params['gutter_search_end'] = args.gutter_end
    if args.min_line_length is not None:
        config_params['min_line_length'] = args.min_line_length
    
    config = Config(**config_params)
    output_dir = Path(args.output_dir)
    
    print(f"Processing complete pipeline on {len(image_paths)} images")
    if args.test_images:
        print(f"Batch mode: Processing all images from test_images directory")
    print(f"Configuration:")
    print(f"  - ROI detection: {config.enable_roi_detection}")
    print(f"  - Gutter search: {config.gutter_search_start:.2f} - {config.gutter_search_end:.2f}")
    print(f"  - Min line length: {config.min_line_length}px")
    print(f"  - Save intermediates: {args.save_intermediates}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Process all images
    results = []
    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Processing pipeline for: {image_path.name}")
        result = process_complete_pipeline(image_path, config, output_dir, args.save_intermediates)
        results.append(result)
    
    # Create master summary
    successful_results = [r for r in results if r['success']]
    print(f"\n{'='*60}")
    print("COMPLETE PIPELINE VISUALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {len(successful_results)}/{len(image_paths)} images")
    
    if successful_results:
        # Aggregate statistics
        total_pages = len(successful_results) * 2
        successful_crops = 0
        roi_processed = 0
        
        for result in successful_results:
            if 'results' in result:
                for page_idx in [1, 2]:
                    page_key = f'page_{page_idx}'
                    if page_key in result['results']:
                        page_data = result['results'][page_key]
                        if page_data['crop_info']['has_crop_region']:
                            successful_crops += 1
                        if page_data['roi_info']:
                            roi_processed += 1
        
        print(f"Total pages processed: {total_pages}")
        print(f"Successful table crops: {successful_crops}/{total_pages}")
        print(f"Pages with ROI processing: {roi_processed}/{total_pages}")
    
    print(f"\nOutput files saved to: {output_dir}")
    print(f"Review the '_pipeline_comparison.jpg' files for complete workflow assessment")
    print(f"Check '_pipeline_summary.json' files for detailed processing metrics")
    
    if args.save_intermediates:
        print(f"All intermediate steps saved with numbered prefixes (00_original through 10_final_result)")
    
    # Save master summary
    output_dir.mkdir(parents=True, exist_ok=True)
    master_summary_file = output_dir / "master_pipeline_summary.json"
    master_summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'processing_info': {
            'total_images': len(image_paths),
            'successful_images': len(successful_results),
            'config_used': config_params,
            'save_intermediates': args.save_intermediates
        },
        'results': results
    }
    
    with open(master_summary_file, 'w') as f:
        json.dump(convert_numpy_types(master_summary), f, indent=2)
    
    print(f"Master summary saved to: {master_summary_file}")


if __name__ == "__main__":
    main()