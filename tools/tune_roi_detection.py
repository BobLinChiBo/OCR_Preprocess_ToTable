#!/usr/bin/env python3
"""
ROI Detection Parameter Tuning Tool

This script helps tune ROI (Region of Interest) detection parameters by testing
different combinations of Gabor filter settings and cut detection thresholds.

Usage:
    python tools/tune_roi_detection.py

Input: Deskewed images from previous tuning stage
Output: ROI-cropped images with different parameter combinations for evaluation
"""

import sys
from pathlib import Path
import itertools
import json

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr_pipeline.utils import (
    load_image, save_image, get_image_files, 
    detect_roi_for_image, crop_to_roi
)


class ROIConfig:
    """Simple config class for ROI detection parameters."""
    def __init__(self, **kwargs):
        # Gabor filter parameters
        self.gabor_kernel_size = kwargs.get('gabor_kernel_size', 31)
        self.gabor_sigma = kwargs.get('gabor_sigma', 4.0)
        self.gabor_lambda = kwargs.get('gabor_lambda', 8.0)
        self.gabor_gamma = kwargs.get('gabor_gamma', 0.2)
        self.gabor_binary_threshold = kwargs.get('gabor_binary_threshold', 127)
        
        # ROI cut detection parameters
        self.roi_vertical_mode = kwargs.get('roi_vertical_mode', 'single_best')
        self.roi_horizontal_mode = kwargs.get('roi_horizontal_mode', 'both_sides')
        self.roi_window_size_divisor = kwargs.get('roi_window_size_divisor', 20)
        self.roi_min_window_size = kwargs.get('roi_min_window_size', 10)
        self.roi_min_cut_strength = kwargs.get('roi_min_cut_strength', 20.0)
        self.roi_min_confidence_threshold = kwargs.get('roi_min_confidence_threshold', 5.0)




def test_roi_detection_parameters():
    """Test different ROI detection parameter combinations."""
    
    # Input and output directories
    input_dir = Path("data/output/tuning/03_roi_input")
    output_base = Path("data/output/tuning/03_roi_detection")
    
    # Parameter ranges to test - EXPANDED CONSERVATIVE VALUES for 65-95% area ratios
    # Progressively higher cut_strength and confidence_threshold = minimal cropping
    gabor_kernel_sizes = [21, 31, 41, 51]  # Expanded range for comprehensive testing
    gabor_sigmas = [2.0, 3.0, 4.0, 5.0]   # Broader sigma range
    gabor_lambdas = [6.0, 8.0, 10.0, 12.0, 15.0]  # Extended lambda range
    cut_strengths = [500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0]  # VERY high = minimal cropping
    confidence_thresholds = [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0]  # VERY high = ultra conservative
    
    print("[CONFIG] ROI DETECTION PARAMETER TUNING")
    print("=" * 50)
    print(f"[DIR] Input: {input_dir}")
    print(f"[DIR] Output: {output_base}")
    print()
    
    # Validate input directory
    if not input_dir.exists():
        print(f"[ERROR] Error: Input directory not found: {input_dir}")
        print("Please first run deskewing tuning and copy the best results to:")
        print(f"   {input_dir}")
        return
    
    # Get deskewed images
    image_files = get_image_files(input_dir)
    if not image_files:
        print(f"[ERROR] Error: No image files found in {input_dir}")
        print("Please copy deskewed images from deskewing tuning results")
        return
    
    print(f"Found {len(image_files)} deskewed images:")
    for img_file in image_files:
        print(f"  - {img_file.name}")
    print()
    
    # Clean up output directory
    if output_base.exists():
        import shutil
        shutil.rmtree(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Create parameter combinations (reduced set for manageability)
    param_combinations = list(itertools.product(
        gabor_kernel_sizes,
        gabor_sigmas, 
        gabor_lambdas,
        cut_strengths,
        confidence_thresholds
    ))
    
    # Calculate total combinations and provide runtime estimate
    total_combinations = len(param_combinations)
    estimated_runtime_min = (total_combinations * len(image_files) * 0.5) / 60  # ~0.5 sec per image
    
    print(f"Testing {total_combinations} parameter combinations:")
    print("Parameters: kernel_size, sigma, lambda, cut_strength, confidence")
    print(f"Estimated runtime: {estimated_runtime_min:.1f} minutes ({total_combinations * len(image_files)} image processes)")
    print("Press Ctrl+C to interrupt and save partial results")
    print()
    
    results_summary = []
    successful_combinations = []  # Track combinations achieving 65-95% target
    
    # Test each parameter combination
    try:
        for i, (kernel_size, sigma, lambda_val, cut_strength, confidence) in enumerate(param_combinations, 1):
            param_name = f"k{kernel_size}_s{sigma}_l{lambda_val}_cs{cut_strength}_ct{confidence}"
            param_dir = output_base / param_name
            param_dir.mkdir(exist_ok=True)
            
            # Progress indicator with percentage
            progress_pct = (i / total_combinations) * 100
            print(f"\n[{i}/{total_combinations}] ({progress_pct:.1f}%) Testing: {param_name}")
            print(f"  gabor_kernel_size: {kernel_size}")
            print(f"  gabor_sigma: {sigma}")
            print(f"  gabor_lambda: {lambda_val}")
            print(f"  roi_min_cut_strength: {cut_strength}")
            print(f"  roi_min_confidence_threshold: {confidence}")
            
            # Create config for this parameter set
            config = ROIConfig(
                gabor_kernel_size=kernel_size,
                gabor_sigma=sigma,
                gabor_lambda=lambda_val,
                roi_min_cut_strength=cut_strength,
                roi_min_confidence_threshold=confidence
            )
            
            roi_log = []  # Track ROI detection results
            
            # Process each image with current parameters
            for image_path in image_files:
                try:
                    # Load image
                    image = load_image(image_path)
                    
                    # Detect ROI with current parameters
                    roi_coords, analysis = detect_roi_for_image(image, config, return_analysis=True)
                    binary_mask = analysis['binary_mask']
                    
                    # Convert analysis format to match expected structure
                    analysis_stats = {
                        'vertical_info': analysis['vertical_info'],
                        'horizontal_info': analysis['horizontal_info'],
                        'area_ratio': analysis['roi_area_percentage'] / 100,
                        'cropped_size': (analysis['roi_width'], analysis['roi_height']),
                        'original_size': (roi_coords['image_width'], roi_coords['image_height'])
                    }
                    
                    # Crop to ROI
                    cropped_image = crop_to_roi(image, roi_coords)
                    
                    # Save results
                    base_name = image_path.stem
                    
                    # Save cropped ROI image
                    roi_path = param_dir / f"{base_name}_roi.jpg"
                    save_image(cropped_image, roi_path)
                    
                    # Save Gabor response for visualization (optional)
                    gabor_path = param_dir / f"{base_name}_gabor.jpg"
                    save_image(binary_mask, gabor_path)
                    
                    # Save ROI coordinates as JSON
                    coords_path = param_dir / f"{base_name}_roi.json"
                    with open(coords_path, 'w') as f:
                        json.dump({
                            'roi_coordinates': roi_coords,
                            'analysis': {
                                'area_ratio': analysis_stats['area_ratio'],
                                'cropped_size': analysis_stats['cropped_size'],
                                'original_size': analysis_stats['original_size']
                            }
                        }, f, indent=2)
                    
                    # Log results
                    roi_log.append({
                        'image': image_path.name,
                        'roi_coords': roi_coords,
                        'area_ratio': analysis_stats['area_ratio'],
                        'cropped_size': analysis_stats['cropped_size']
                    })
                    
                    print(f"    [OK] {image_path.name} -> area: {analysis_stats['area_ratio']:.2%}")
                    
                except Exception as e:
                    print(f"    [ERROR] Error processing {image_path.name}: {e}")
            
            # Analyze results for this parameter combination
            if roi_log:
                avg_area_ratio = sum(entry['area_ratio'] for entry in roi_log) / len(roi_log)
                in_target_count = sum(1 for entry in roi_log if 0.65 <= entry['area_ratio'] <= 0.95)
                success_rate = (in_target_count / len(roi_log)) * 100
                
                print(f"    [RESULT] Average area: {avg_area_ratio:.2%}, Target range (65-95%): {in_target_count}/{len(roi_log)} ({success_rate:.1f}%)")
                
                # Track successful combinations (80%+ images in target range)
                if success_rate >= 80:
                    successful_combinations.append({
                        'params': param_name,
                        'avg_area_ratio': avg_area_ratio,
                        'success_rate': success_rate,
                        'in_target_count': in_target_count,
                        'total_count': len(roi_log)
                    })
                    print(f"    [SUCCESS] *** PROMISING COMBINATION - {success_rate:.1f}% success rate ***")
            
            # Save analysis for this parameter set
            analysis_file = param_dir / "roi_analysis.txt"
            with open(analysis_file, 'w') as f:
                f.write(f"ROI DETECTION ANALYSIS - {param_name}\n")
                f.write("=" * 50 + "\n")
                f.write(f"Parameters:\n")
                f.write(f"  gabor_kernel_size: {kernel_size}\n")
                f.write(f"  gabor_sigma: {sigma}\n")
                f.write(f"  gabor_lambda: {lambda_val}\n")
                f.write(f"  roi_min_cut_strength: {cut_strength}\n")
                f.write(f"  roi_min_confidence_threshold: {confidence}\n\n")
                
                f.write("Per-image results:\n")
                f.write("-" * 30 + "\n")
                for entry in roi_log:
                    coords = entry['roi_coords']
                    f.write(f"{entry['image']}:\n")
                    f.write(f"  ROI: ({coords['roi_left']}, {coords['roi_top']}) to ")
                    f.write(f"({coords['roi_right']}, {coords['roi_bottom']})\n")
                    f.write(f"  Size: {entry['cropped_size'][0]}x{entry['cropped_size'][1]}\n")
                    f.write(f"  Area ratio: {entry['area_ratio']:.2%}\n\n")
                
                # Statistics
                if roi_log:
                    avg_area_ratio = sum(entry['area_ratio'] for entry in roi_log) / len(roi_log)
                    in_target_count = sum(1 for entry in roi_log if 0.65 <= entry['area_ratio'] <= 0.95)
                    success_rate = (in_target_count / len(roi_log)) * 100
                    f.write(f"Statistics:\n")
                    f.write(f"  Images processed: {len(roi_log)}\n")
                    f.write(f"  Average area ratio: {avg_area_ratio:.2%}\n")
                    f.write(f"  Target range (65-95%): {in_target_count}/{len(roi_log)} ({success_rate:.1f}%)\n")
            
            # Record results for summary
            results_summary.append({
                'params': param_name,
                'kernel_size': kernel_size,
                'sigma': sigma,
                'lambda': lambda_val,
                'cut_strength': cut_strength,
                'confidence': confidence,
                'output_dir': param_dir,
                'roi_log': roi_log
            })
    
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] Processing stopped by user after {len(results_summary)} combinations")
        print(f"[INFO] Partial results will be saved...")
    
    # Generate comprehensive summary report
    summary_file = output_base / "roi_detection_test_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("ROI DETECTION PARAMETER TUNING RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test images: {len(image_files)}\n")
        f.write(f"Parameter combinations tested: {len(results_summary)}/{total_combinations}\n")
        f.write(f"Successful combinations (80%+ in 65-95% range): {len(successful_combinations)}\n\n")
        
        # Top successful combinations first
        if successful_combinations:
            f.write("★ TOP SUCCESSFUL COMBINATIONS (Target: 65-95% area ratio):\n")
            f.write("-" * 60 + "\n")
            # Sort by success rate, then by average area ratio closest to 80%
            successful_combinations.sort(key=lambda x: (x['success_rate'], -abs(x['avg_area_ratio'] - 0.80)), reverse=True)
            
            for i, success in enumerate(successful_combinations[:10], 1):  # Top 10
                f.write(f"{i}. {success['params']}\n")
                f.write(f"   Success rate: {success['success_rate']:.1f}% ({success['in_target_count']}/{success['total_count']} images)\n")
                f.write(f"   Average area: {success['avg_area_ratio']:.1%}\n")
                f.write(f"   → RECOMMENDED for 65-95% target\n\n")
        
        f.write("ALL PARAMETER COMBINATIONS:\n")
        f.write("-" * 30 + "\n")
        # Sort all results by average area ratio (descending)
        results_summary.sort(key=lambda x: sum(entry['area_ratio'] for entry in x['roi_log']) / len(x['roi_log']) if x['roi_log'] else 0, reverse=True)
        
        for result in results_summary:
            if result['roi_log']:
                avg_area = sum(entry['area_ratio'] for entry in result['roi_log']) / len(result['roi_log'])
                in_target_count = sum(1 for entry in result['roi_log'] if 0.65 <= entry['area_ratio'] <= 0.95)
                success_rate = (in_target_count / len(result['roi_log'])) * 100
            else:
                avg_area = 0
                in_target_count = 0
                success_rate = 0
            
            status = "★★★" if success_rate >= 80 else "★★" if success_rate >= 50 else "★" if success_rate >= 20 else "   "
            
            f.write(f"{status} Folder: {result['params']}\n")
            f.write(f"  gabor_kernel_size: {result['kernel_size']}\n")
            f.write(f"  gabor_sigma: {result['sigma']}\n")
            f.write(f"  gabor_lambda: {result['lambda']}\n")
            f.write(f"  roi_min_cut_strength: {result['cut_strength']}\n")
            f.write(f"  roi_min_confidence_threshold: {result['confidence']}\n")
            f.write(f"  Average area preserved: {avg_area:.2%}\n")
            f.write(f"  Target range success: {success_rate:.1f}% ({in_target_count}/{len(result['roi_log']) if result['roi_log'] else 0})\n")
            f.write(f"  Results in: {result['output_dir']}\n\n")
        
        f.write("EVALUATION GUIDELINES:\n")
        f.write("-" * 30 + "\n")
        f.write("1. Look for ROI crops that focus on table content\n")
        f.write("2. Check that important content is not cut off\n")
        f.write("3. Ensure headers, footers, and margins are appropriately removed\n")
        f.write("4. Balance between aggressive cropping and content preservation\n")
        f.write("5. Compare Gabor response images to understand edge detection\n\n")
        
        f.write("AREA RATIO GUIDELINES:\n")
        f.write("-" * 30 + "\n")
        f.write("• 90-100%: Very conservative, minimal cropping\n")
        f.write("• 70-90%: Moderate cropping, good for mixed content\n")
        f.write("• 50-70%: Aggressive cropping, good for clean table extraction\n")
        f.write("• <50%: Very aggressive, check for over-cropping\n\n")
        
        f.write("NEXT STEPS:\n")
        f.write("-" * 30 + "\n")
        f.write("1. Choose the best parameter combination\n")
        f.write("2. Copy the ROI images from that folder to:\n")
        f.write("   data/output/tuning/04_line_input/\n")
        f.write("3. Run: python tools/tune_line_detection.py\n")
    
    print(f"\n[SUCCESS] ROI DETECTION PARAMETER TUNING COMPLETE!")
    print(f"[TESTED] {len(results_summary)}/{total_combinations} parameter combinations")
    print(f"[TARGET] {len(successful_combinations)} combinations achieved 65-95% target range")
    print(f"[DIR] Results saved in: {output_base}")
    print(f"[FILE] Summary report: {summary_file}")
    print()
    
    if successful_combinations:
        print("🎯 TOP SUCCESSFUL COMBINATIONS:")
        successful_combinations.sort(key=lambda x: (x['success_rate'], -abs(x['avg_area_ratio'] - 0.80)), reverse=True)
        for i, success in enumerate(successful_combinations[:3], 1):  # Top 3
            print(f"  {i}. {success['params']} - {success['success_rate']:.1f}% success, {success['avg_area_ratio']:.1%} avg area")
    else:
        print("⚠️  NO COMBINATIONS achieved 80%+ success rate for 65-95% target")
        print("   Consider disabling ROI detection or using fixed margins")
    
    print()
    print("EVALUATION GUIDELINES:")
    print("[TARGET] Look for crops that focus on table content")
    print("[CUT] Balance between aggressive cropping and content preservation") 
    print("[CHECK] Check both ROI images and Gabor response visualizations")
    print()
    print("NEXT STEPS:")
    print("1. [REVIEW] Check summary report for successful combinations")
    print("2. [CHOOSE] Select best parameter combination from ★★★ rated results")
    print("3. [COPY] Copy best results to: data/output/tuning/04_line_input/")
    print("4. [RUN] Execute: python tools/tune_line_detection.py")


if __name__ == "__main__":
    test_roi_detection_parameters()