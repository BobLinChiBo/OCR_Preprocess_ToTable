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
    
    # Parameter ranges to test - focusing on most impactful parameters
    gabor_kernel_sizes = [21, 31, 41]
    gabor_sigmas = [3.0, 4.0, 5.0]
    gabor_lambdas = [6.0, 8.0, 10.0]
    cut_strengths = [10.0, 20.0, 30.0]
    confidence_thresholds = [3.0, 5.0, 7.0]
    
    print("🔧 ROI DETECTION PARAMETER TUNING")
    print("=" * 50)
    print(f"📂 Input: {input_dir}")
    print(f"📁 Output: {output_base}")
    print()
    
    # Validate input directory
    if not input_dir.exists():
        print(f"❌ Error: Input directory not found: {input_dir}")
        print("Please first run deskewing tuning and copy the best results to:")
        print(f"   {input_dir}")
        return
    
    # Get deskewed images
    image_files = get_image_files(input_dir)
    if not image_files:
        print(f"❌ Error: No image files found in {input_dir}")
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
    
    # Limit to a reasonable number for manual evaluation
    if len(param_combinations) > 20:
        print(f"⚠️  Reducing {len(param_combinations)} combinations to 20 most representative...")
        # Select a diverse subset
        step = len(param_combinations) // 20
        param_combinations = param_combinations[::step][:20]
    
    print(f"Testing {len(param_combinations)} parameter combinations:")
    print("Parameters: kernel_size, sigma, lambda, cut_strength, confidence")
    
    results_summary = []
    
    # Test each parameter combination
    for i, (kernel_size, sigma, lambda_val, cut_strength, confidence) in enumerate(param_combinations, 1):
        param_name = f"k{kernel_size}_s{sigma}_l{lambda_val}_cs{cut_strength}_ct{confidence}"
        param_dir = output_base / param_name
        param_dir.mkdir(exist_ok=True)
        
        print(f"\n[{i}/{len(param_combinations)}] Testing: {param_name}")
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
                
                print(f"    ✅ {image_path.name} -> area: {analysis_stats['area_ratio']:.2%}")
                
            except Exception as e:
                print(f"    ❌ Error processing {image_path.name}: {e}")
        
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
                f.write(f"Statistics:\n")
                f.write(f"  Images processed: {len(roi_log)}\n")
                f.write(f"  Average area ratio: {avg_area_ratio:.2%}\n")
        
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
    
    # Generate comprehensive summary report
    summary_file = output_base / "roi_detection_test_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("ROI DETECTION PARAMETER TUNING RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test images: {len(image_files)}\n")
        f.write(f"Parameter combinations tested: {len(param_combinations)}\n\n")
        
        f.write("PARAMETER COMBINATIONS:\n")
        f.write("-" * 30 + "\n")
        for result in results_summary:
            if result['roi_log']:
                avg_area = sum(entry['area_ratio'] for entry in result['roi_log']) / len(result['roi_log'])
            else:
                avg_area = 0
            
            f.write(f"Folder: {result['params']}\n")
            f.write(f"  gabor_kernel_size: {result['kernel_size']}\n")
            f.write(f"  gabor_sigma: {result['sigma']}\n")
            f.write(f"  gabor_lambda: {result['lambda']}\n")
            f.write(f"  roi_min_cut_strength: {result['cut_strength']}\n")
            f.write(f"  roi_min_confidence_threshold: {result['confidence']}\n")
            f.write(f"  Average area preserved: {avg_area:.2%}\n")
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
    
    print(f"\n🎉 ROI DETECTION PARAMETER TUNING COMPLETE!")
    print(f"📁 Results saved in: {output_base}")
    print(f"📄 Summary report: {summary_file}")
    print()
    print("EVALUATION GUIDELINES:")
    print("🎯 Look for crops that focus on table content")
    print("✂️  Balance between aggressive cropping and content preservation")
    print("👀 Check both ROI images and Gabor response visualizations")
    print()
    print("NEXT STEPS:")
    print("1. 📋 Choose the best parameter combination")
    print("2. 📂 Copy best results to: data/output/tuning/04_line_input/")
    print("3. 🚀 Run: python tools/tune_line_detection.py")


if __name__ == "__main__":
    test_roi_detection_parameters()