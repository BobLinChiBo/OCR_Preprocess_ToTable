#!/usr/bin/env python3
"""
Debug Margin Analysis Script

Shows detailed step-by-step analysis of margin detection including:
1. Original image
2. Binary image (Otsu threshold)
3. Edge regions being analyzed
4. Intensity profiles (grayscale and binary)
5. Cutting decisions and thresholds
6. Final result with detailed annotations
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
from typing import Dict, Any, Tuple

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.ocr_pipeline.processors import remove_margins_gradient


def create_debug_visualizations(image_path: str, output_dir: str = None):
    """Create detailed debug visualizations for margin analysis."""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Set up output directory
    if output_dir is None:
        output_dir = Path("data/output/debug_margin_analysis")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_name = Path(image_path).stem
    
    # Run gradient margin removal with analysis
    result, analysis = remove_margins_gradient(
        image,
        edge_percentage=0.20,
        gradient_window_size=21,
        intensity_shift_threshold=50.0,
        return_analysis=True
    )
    
    # Create debug plots
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle(f'Margin Analysis Debug: {image_name}', fontsize=16)
    
    # Convert to grayscale and binary for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    height, width = gray.shape
    edge_height = int(height * 0.20)
    edge_width = int(width * 0.20)
    
    # 1. Original image
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 2. Binary image
    axes[0, 1].imshow(binary, cmap='gray')
    axes[0, 1].set_title('Binary Image (Otsu)')
    axes[0, 1].axis('off')
    
    # 3. Edge regions overlay
    edge_overlay = image.copy()
    # Draw edge regions
    cv2.rectangle(edge_overlay, (0, 0), (width, edge_height), (0, 255, 0), 2)  # Top
    cv2.rectangle(edge_overlay, (0, height-edge_height), (width, height), (0, 255, 0), 2)  # Bottom
    cv2.rectangle(edge_overlay, (0, 0), (edge_width, height), (255, 0, 0), 2)  # Left
    cv2.rectangle(edge_overlay, (width-edge_width, 0), (width, height), (255, 0, 0), 2)  # Right
    
    axes[0, 2].imshow(cv2.cvtColor(edge_overlay, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Edge Analysis Regions (20%)')
    axes[0, 2].axis('off')
    
    # Calculate profiles
    top_profile = np.mean(gray[:edge_height, :], axis=1)
    bottom_profile = np.mean(gray[height-edge_height:, :], axis=1)
    left_profile = np.mean(gray[:, :edge_width], axis=0)
    right_profile = np.mean(gray[:, width-edge_width:], axis=0)
    
    # Binary profiles - match main code logic with correct edge directions
    top_binary_profile = np.mean(binary[:edge_height, :], axis=1)  # Top edge, index 0 = top
    bottom_binary_profile = np.mean(binary[height-edge_height:, :], axis=1)[::-1]  # Reverse: index 0 = bottom edge
    left_binary_profile = np.mean(binary[:, :edge_width], axis=0)  # Left edge, index 0 = left  
    right_binary_profile = np.mean(binary[:, width-edge_width:], axis=0)[::-1]  # Reverse: index 0 = right edge
    
    # 4-7. Intensity profiles
    binary_threshold = 200  # Aggressive threshold for more cutting
    
    # Top profile
    axes[1, 0].plot(top_profile, label='Grayscale', color='blue')
    axes[1, 0].plot(top_binary_profile, label='Binary', color='red')
    axes[1, 0].axhline(y=binary_threshold, color='green', linestyle='--', label=f'Threshold {binary_threshold}')
    axes[1, 0].set_title('Top Edge Profile')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Bottom profile  
    axes[1, 1].plot(bottom_profile, label='Grayscale', color='blue')
    axes[1, 1].plot(bottom_binary_profile, label='Binary', color='red')
    axes[1, 1].axhline(y=binary_threshold, color='green', linestyle='--', label=f'Threshold {binary_threshold}')
    axes[1, 1].set_title('Bottom Edge Profile')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Left profile
    axes[1, 2].plot(left_profile, label='Grayscale', color='blue')
    axes[1, 2].plot(left_binary_profile, label='Binary', color='red')
    axes[1, 2].axhline(y=binary_threshold, color='green', linestyle='--', label=f'Threshold {binary_threshold}')
    axes[1, 2].set_title('Left Edge Profile')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    # Right profile
    axes[2, 0].plot(right_profile, label='Grayscale', color='blue')
    axes[2, 0].plot(right_binary_profile, label='Binary', color='red')
    axes[2, 0].axhline(y=binary_threshold, color='green', linestyle='--', label=f'Threshold {binary_threshold}')
    axes[2, 0].set_title('Right Edge Profile')
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    
    # 8. Boundary position analysis (new clean edge detection logic)
    boundaries, debug_info = detect_gradient_boundaries(
        gray, edge_percentage=0.20, gradient_window_size=21, 
        intensity_shift_threshold=50.0, margin_confidence_threshold=0.7, 
        return_debug=True
    )
    
    # Get boundary positions
    gradient_analyses = {
        'top': debug_info.get('top_analysis'),
        'bottom': debug_info.get('bottom_analysis'), 
        'left': debug_info.get('left_analysis'),
        'right': debug_info.get('right_analysis')
    }
    
    boundary_positions = []
    sides = ['Top', 'Bottom', 'Left', 'Right']
    for side_key in ['top', 'bottom', 'left', 'right']:
        analysis = gradient_analyses[side_key]
        if analysis:
            boundary_positions.append(analysis.get('boundary_pos', 0))
        else:
            boundary_positions.append(0)
    
    # Color based on clean edge rule: boundary_pos < 5 = clean (don't cut)
    colors = ['green' if pos < 5 else 'red' for pos in boundary_positions]
    
    bars = axes[2, 1].bar(sides, boundary_positions, color=colors, alpha=0.7)
    axes[2, 1].axhline(y=5, color='black', linestyle='--', label='Clean Edge Threshold (5px)')
    axes[2, 1].set_title('Boundary Positions (Clean Edge Detection)')
    axes[2, 1].set_ylabel('Boundary Position (pixels)')
    axes[2, 1].legend()
    
    # Add text annotations
    for i, (side, pos, color) in enumerate(zip(sides, boundary_positions, colors)):
        decision = "Don't Cut (Clean)" if pos < 5 else "Cut (Margin)"
        axes[2, 1].text(i, pos + 1, f'{pos:.1f}px\n{decision}', 
                        ha='center', va='bottom', fontsize=8)
    
    # 9. Final result
    axes[2, 2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[2, 2].set_title('Final Result')
    axes[2, 2].axis('off')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_dir / f'{image_name}_debug_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Debug analysis saved: {output_dir / f'{image_name}_debug_analysis.png'}")
    
    # Save detailed analysis data
    debug_data = {
        'image_name': image_name,
        'clean_edge_threshold': 5,  # Boundary position threshold for clean edges
        'boundary_positions': {
            'top': float(boundary_positions[0]),
            'bottom': float(boundary_positions[1]),
            'left': float(boundary_positions[2]),
            'right': float(boundary_positions[3])
        },
        'cutting_decisions': {
            'top': bool(boundary_positions[0] >= 5),
            'bottom': bool(boundary_positions[1] >= 5),
            'left': bool(boundary_positions[2] >= 5),
            'right': bool(boundary_positions[3] >= 5)
        },
        'profiles': {
            'top_grayscale': top_profile.tolist(),
            'top_binary': top_binary_profile.tolist(),
            'bottom_grayscale': bottom_profile.tolist(),
            'bottom_binary': bottom_binary_profile.tolist(),
            'left_grayscale': left_profile.tolist(),
            'left_binary': left_binary_profile.tolist(),
            'right_grayscale': right_profile.tolist(),
            'right_binary': right_binary_profile.tolist()
        }
    }
    
    with open(output_dir / f'{image_name}_debug_data.json', 'w') as f:
        import json
        json.dump(debug_data, f, indent=2)
    
    print(f"Debug data saved: {output_dir / f'{image_name}_debug_data.json'}")
    
    return debug_data


def main():
    parser = argparse.ArgumentParser(description='Debug margin analysis with detailed visualizations')
    parser.add_argument('image_path', help='Path to image file to analyze')
    parser.add_argument('--output-dir', help='Output directory for debug files')
    
    args = parser.parse_args()
    
    debug_data = create_debug_visualizations(args.image_path, args.output_dir)
    
    print("\n=== DEBUG ANALYSIS SUMMARY ===")
    print(f"Clean edge threshold: {debug_data['clean_edge_threshold']} pixels")
    print("\nBoundary positions:")
    for side, position in debug_data['boundary_positions'].items():
        decision = "Cut (has margin)" if debug_data['cutting_decisions'][side] else "Don't cut (clean edge)"
        print(f"  {side:6}: {position:6.1f}px -> {decision}")


if __name__ == "__main__":
    main()