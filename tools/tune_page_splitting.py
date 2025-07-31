#!/usr/bin/env python3
"""
Page Splitting Parameter Tuning Tool

This script helps tune page splitting parameters by testing different combinations
of gutter search range and minimum gutter width on your test images.

Usage:
    python tools/tune_page_splitting.py

The script will process all images in data/input/test_images/ and save results
for each parameter combination, allowing you to visually inspect and compare.
"""

import sys
from pathlib import Path
import itertools

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr_pipeline.utils import (
    load_image,
    save_image,
    split_two_page_image,
    get_image_files,
)


def test_page_splitting_parameters():
    """Test different page splitting parameter combinations."""

    # Input and output directories
    input_dir = Path("data/input/test_images")
    output_base = Path("data/output/tuning/01_split_pages")

    # Parameter ranges to test
    gutter_start_values = [0.35, 0.4, 0.45]
    gutter_end_values = [0.55, 0.6, 0.65]
    min_gutter_width_values = [
        30,
        50,
        70,
    ]  # Not used in current implementation, but for future

    print("[CONFIG] PAGE SPLITTING PARAMETER TUNING")
    print("=" * 50)
    print(f"[DIR] Input: {input_dir}")
    print(f"[DIR] Output: {output_base}")
    print()

    # Validate input directory
    if not input_dir.exists():
        print(f"[ERROR] Error: Input directory not found: {input_dir}")
        return

    # Get test images
    image_files = get_image_files(input_dir)
    if not image_files:
        print(f"[ERROR] Error: No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} test images:")
    for img_file in image_files:
        print(f"  - {img_file.name}")
    print()

    # Clean up output directory
    if output_base.exists():
        import shutil

        shutil.rmtree(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    # Create parameter combinations
    param_combinations = list(
        itertools.product(
            gutter_start_values, gutter_end_values, min_gutter_width_values
        )
    )

    print(f"Testing {len(param_combinations)} parameter combinations:")
    print("Parameters: gutter_start, gutter_end, min_gutter_width")

    results_summary = []

    # Test each parameter combination
    for i, (gutter_start, gutter_end, min_gutter_width) in enumerate(
        param_combinations, 1
    ):
        param_name = f"start{gutter_start}_end{gutter_end}_width{min_gutter_width}"
        param_dir = output_base / param_name
        param_dir.mkdir(exist_ok=True)

        print(f"\n[{i}/{len(param_combinations)}] Testing: {param_name}")
        print(f"  gutter_search_start: {gutter_start}")
        print(f"  gutter_search_end: {gutter_end}")
        print(f"  min_gutter_width: {min_gutter_width}")

        # Process each test image with current parameters
        for image_path in image_files:
            try:
                # Load image
                image = load_image(image_path)

                # Split with current parameters
                left_page, right_page = split_two_page_image(
                    image, gutter_start=gutter_start, gutter_end=gutter_end
                )

                # Save split results
                base_name = image_path.stem
                left_path = param_dir / f"{base_name}_page_1.jpg"
                right_path = param_dir / f"{base_name}_page_2.jpg"

                save_image(left_page, left_path)
                save_image(right_page, right_path)

                print(
                    f"    [OK] {image_path.name} -> {left_path.name}, {right_path.name}"
                )

            except Exception as e:
                print(f"    [ERROR] Error processing {image_path.name}: {e}")

        # Record results for summary
        results_summary.append(
            {
                "params": param_name,
                "gutter_start": gutter_start,
                "gutter_end": gutter_end,
                "min_gutter_width": min_gutter_width,
                "output_dir": param_dir,
            }
        )

    # Generate summary report
    summary_file = output_base / "parameter_test_summary.txt"
    with open(summary_file, "w") as f:
        f.write("PAGE SPLITTING PARAMETER TUNING RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test images: {len(image_files)}\n")
        f.write(f"Parameter combinations tested: {len(param_combinations)}\n\n")

        f.write("PARAMETER COMBINATIONS:\n")
        f.write("-" * 30 + "\n")
        for result in results_summary:
            f.write(f"Folder: {result['params']}\n")
            f.write(f"  gutter_search_start: {result['gutter_start']}\n")
            f.write(f"  gutter_search_end: {result['gutter_end']}\n")
            f.write(f"  min_gutter_width: {result['min_gutter_width']}\n")
            f.write(f"  Results in: {result['output_dir']}\n\n")

        f.write("EVALUATION INSTRUCTIONS:\n")
        f.write("-" * 30 + "\n")
        f.write("1. Visually inspect each parameter combination folder\n")
        f.write("2. Look for clean page separation at the gutter\n")
        f.write("3. Check that content is not cut off from either page\n")
        f.write("4. Note which parameters work best for your document type\n")
        f.write("5. Use the best parameters for the next tuning stage\n\n")

        f.write("NEXT STEPS:\n")
        f.write("-" * 30 + "\n")
        f.write("1. Choose the best parameter combination\n")
        f.write("2. Copy the split images from that folder to:\n")
        f.write("   data/output/tuning/02_deskewed_input/\n")
        f.write("3. Run: python tools/tune_deskewing.py\n")

    print(f"\n[SUCCESS] PARAMETER TUNING COMPLETE!")
    print(f"[DIR] Results saved in: {output_base}")
    print(f"[FILE] Summary report: {summary_file}")
    print()
    print("NEXT STEPS:")
    print("1. [CHECK] Visually inspect all parameter combination folders")
    print("2. [LIST] Choose the best parameter combination")
    print("3. [DIR] Copy the best results to: data/output/tuning/02_deskewed_input/")
    print("4. [START] Run: python tools/tune_deskewing.py")


if __name__ == "__main__":
    test_page_splitting_parameters()
