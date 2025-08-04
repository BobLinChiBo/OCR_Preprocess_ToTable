#!/usr/bin/env python3
"""
Mark Removal Visualization Script (Version 2)

Visualizes the mark removal process for watermark/stamp/artifact removal.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import sys
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.ocr_pipeline import utils
from src.ocr_pipeline.config import Stage1Config
from config_utils import load_config, get_command_args_dict
from output_manager import (
    OutputManager,
    get_test_images,
    convert_numpy_types,
    save_step_parameters,
)


def analyze_mark_removal(image: np.ndarray, dilate_iter: int = 2) -> Dict[str, Any]:
    """Analyze mark removal process and return detailed information."""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Build protection mask
    protect_mask = utils.build_protect_mask(gray, dilate_iter)
    
    # Calculate statistics
    total_pixels = gray.size
    protected_pixels = np.sum(protect_mask > 0)
    removed_pixels = total_pixels - protected_pixels
    
    # Detect potential marks (areas that will be removed)
    inverted_mask = cv2.bitwise_not(protect_mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted_mask, connectivity=8)
    
    # Filter out background (label 0) and very small components
    min_area = 50  # Minimum pixels to consider as a mark
    marks = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            marks.append({
                'x': stats[i, cv2.CC_STAT_LEFT],
                'y': stats[i, cv2.CC_STAT_TOP],
                'width': stats[i, cv2.CC_STAT_WIDTH],
                'height': stats[i, cv2.CC_STAT_HEIGHT],
                'area': area,
                'centroid': (centroids[i][0], centroids[i][1])
            })
    
    analysis = {
        'total_pixels': total_pixels,
        'protected_pixels': protected_pixels,
        'removed_pixels': removed_pixels,
        'protection_ratio': protected_pixels / total_pixels,
        'removal_ratio': removed_pixels / total_pixels,
        'num_marks_detected': len(marks),
        'marks': marks,
        'dilate_iterations': dilate_iter
    }
    
    return analysis


def create_mark_removal_visualization(
    image: np.ndarray,
    analysis: Dict[str, Any],
    dilate_iter: int = 2
) -> np.ndarray:
    """Create a comprehensive visualization of the mark removal process."""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Mark Removal Visualization', fontsize=16)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        display_original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        gray = image.copy()
        display_original = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    # 1. Original image
    axes[0, 0].imshow(display_original)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 2. Protection mask
    protect_mask = utils.build_protect_mask(gray, dilate_iter)
    axes[0, 1].imshow(protect_mask, cmap='gray')
    axes[0, 1].set_title(f'Protection Mask (dilate={dilate_iter})')
    axes[0, 1].axis('off')
    
    # 3. Marks to be removed (inverted mask)
    marks_mask = cv2.bitwise_not(protect_mask)
    axes[0, 2].imshow(marks_mask, cmap='gray')
    axes[0, 2].set_title('Marks to Remove')
    axes[0, 2].axis('off')
    
    # 4. Cleaned result
    cleaned = utils.remove_marks(image, dilate_iter)
    if len(cleaned.shape) == 2:
        cleaned_display = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
    else:
        cleaned_display = cleaned
    axes[1, 0].imshow(cleaned_display)
    axes[1, 0].set_title('Cleaned Result')
    axes[1, 0].axis('off')
    
    # 5. Difference map
    diff = cv2.absdiff(gray, cleaned)
    diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    axes[1, 1].imshow(cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Difference Map')
    axes[1, 1].axis('off')
    
    # 6. Mark detection overlay
    overlay = display_original.copy()
    for mark in analysis['marks']:
        x, y, w, h = mark['x'], mark['y'], mark['width'], mark['height']
        # Draw rectangle
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Draw centroid
        cx, cy = int(mark['centroid'][0]), int(mark['centroid'][1])
        cv2.circle(overlay, (cx, cy), 3, (0, 255, 0), -1)
    
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title(f"Detected Marks ({analysis['num_marks_detected']})")
    axes[1, 2].axis('off')
    
    # Add statistics text
    stats_text = f"Protection Ratio: {analysis['protection_ratio']:.1%}\n"
    stats_text += f"Removal Ratio: {analysis['removal_ratio']:.1%}\n"
    stats_text += f"Marks Detected: {analysis['num_marks_detected']}"
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    
    plt.tight_layout()
    
    # Convert matplotlib figure to numpy array
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    # Convert RGB to BGR for OpenCV
    visualization = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    
    return visualization


def process_single_image(
    image_path: Path,
    output_dir: Path,
    config: Stage1Config,
    save_debug: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Process a single image through mark removal visualization."""
    manager = OutputManager(output_dir / "mark_removal")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"\nProcessing: {image_path.name}")
    print(f"  Image shape: {image.shape}")
    
    # Get dilate iterations from config or kwargs
    dilate_iter = kwargs.get('dilate_iter', config.mark_removal_dilate_iter)
    
    # Analyze mark removal
    analysis = analyze_mark_removal(image, dilate_iter)
    print(f"  Protection ratio: {analysis['protection_ratio']:.1%}")
    print(f"  Marks detected: {analysis['num_marks_detected']}")
    
    # Apply mark removal
    cleaned = utils.remove_marks(image, dilate_iter)
    
    # Save cleaned result
    result_path = manager.get_output_path(image_path, "_cleaned", subdir="cleaned_images")
    cv2.imwrite(str(result_path), cleaned)
    print(f"  Saved cleaned: {result_path.name}")
    
    # Create and save visualization
    visualization = create_mark_removal_visualization(image, analysis, dilate_iter)
    vis_path = manager.get_output_path(image_path, "_visualization", subdir="visualizations")
    cv2.imwrite(str(vis_path), visualization)
    print(f"  Saved visualization: {vis_path.name}")
    
    # Save debug images if requested
    if save_debug:
        debug_dir = manager.get_output_dir("debug")
        
        # Save protection mask
        protect_mask = utils.build_protect_mask(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image,
            dilate_iter
        )
        mask_path = debug_dir / f"{image_path.stem}_protect_mask.png"
        cv2.imwrite(str(mask_path), protect_mask)
        
        # Save difference map
        gray_orig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        diff = cv2.absdiff(gray_orig, cleaned)
        diff_path = debug_dir / f"{image_path.stem}_difference.png"
        cv2.imwrite(str(diff_path), diff)
    
    # Prepare results
    results = {
        'image_path': str(image_path),
        'output_path': str(result_path),
        'visualization_path': str(vis_path),
        'analysis': convert_numpy_types(analysis),
        'parameters': {
            'dilate_iter': dilate_iter
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Visualize mark removal process for watermark/stamp/artifact removal"
    )
    
    # Input arguments
    parser.add_argument(
        "images",
        nargs="*",
        help="Input image paths (if not using --test-images or --input-dir)"
    )
    parser.add_argument(
        "--test-images",
        action="store_true",
        help="Process all images in input/test_images directory"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Process all images in specified directory"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: output/[timestamp]_mark_removal)"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage1_default.json"),
        help="Configuration file path"
    )
    
    # Mark removal specific arguments
    parser.add_argument(
        "--dilate-iter",
        type=int,
        default=None,
        help="Number of dilation iterations for protection mask (overrides config)"
    )
    
    # Debug options
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save debug images (masks, difference maps)"
    )
    parser.add_argument(
        "--save-params",
        action="store_true",
        help="Save processing parameters to JSON"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config, Stage1Config)
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        manager = OutputManager()
        output_dir = manager.create_output_dir("mark_removal")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Determine input images
    if args.test_images:
        images = get_test_images()
        print(f"Processing {len(images)} test images")
    elif args.input_dir:
        images = list(args.input_dir.glob("*.jpg")) + list(args.input_dir.glob("*.png"))
        print(f"Processing {len(images)} images from {args.input_dir}")
    elif args.images:
        images = [Path(img) for img in args.images]
    else:
        print("No input images specified. Use --test-images, --input-dir, or provide image paths.")
        return
    
    # Process options
    kwargs = {}
    if args.dilate_iter is not None:
        kwargs['dilate_iter'] = args.dilate_iter
    
    # Process each image
    all_results = []
    for image_path in images:
        try:
            results = process_single_image(
                image_path,
                output_dir,
                config,
                save_debug=args.save_debug,
                **kwargs
            )
            all_results.append(results)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Save parameters if requested
    if args.save_params:
        params = {
            'config': str(args.config),
            'dilate_iter': kwargs.get('dilate_iter', config.mark_removal_dilate_iter),
            'images_processed': len(all_results),
            'command_args': get_command_args_dict(args)
        }
        save_step_parameters(output_dir, "mark_removal", params)
    
    # Save results summary
    summary_path = output_dir / "mark_removal_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nProcessing complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()