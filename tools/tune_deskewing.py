#!/usr/bin/env python3
"""
Deskewing Parameter Tuning Tool

This script helps tune deskewing parameters by testing different combinations
of angle range, angle step, and minimum angle correction on split page images.

Usage:
    python tools/tune_deskewing.py

Input: Split page images from previous tuning stage
Output: Deskewed images with different parameter combinations for evaluation
"""

import sys
from pathlib import Path
import itertools

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr_pipeline.utils import (
    load_image,
    save_image,
    get_image_files,
    deskew_image,
)  # noqa: E402
import numpy as np  # noqa: E402


def test_deskewing_parameters():
    """Test different deskewing parameter combinations."""

    # Input and output directories
    input_dir = Path("data/output/tuning/02_deskewed_input")
    output_base = Path("data/output/tuning/02_deskewed")

    # Parameter ranges to test
    angle_range_values = [5, 10, 15, 20]  # Maximum angle range in degrees
    angle_step_values = [
        0.1,
        0.2,
        0.5,
    ]  # Angle search step (not used in basic implementation)
    min_angle_correction_values = [
        0.1,
        0.2,
        0.5,
        1.0,
    ]  # Minimum angle to apply correction

    print("[CONFIG] DESKEWING PARAMETER TUNING")
    print("=" * 50)
    print(f"[DIR] Input: {input_dir}")
    print(f"[DIR] Output: {output_base}")
    print()

    # Validate input directory
    if not input_dir.exists():
        print(f"[ERROR] Error: Input directory not found: {input_dir}")
        print("Please first run page splitting tuning and copy the best results to:")
        print(f"   {input_dir}")
        return

    # Get split page images
    image_files = get_image_files(input_dir)
    if not image_files:
        print(f"[ERROR] Error: No image files found in {input_dir}")
        print("Please copy split page images from page splitting tuning results")
        return

    print(f"Found {len(image_files)} split page images:")
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
            angle_range_values, angle_step_values, min_angle_correction_values
        )
    )

    print(f"Testing {len(param_combinations)} parameter combinations:")
    print("Parameters: angle_range, angle_step, min_angle_correction")

    results_summary = []

    # Test each parameter combination
    for i, (angle_range, angle_step, min_angle_correction) in enumerate(
        param_combinations, 1
    ):
        param_name = f"range{angle_range}_step{angle_step}_min{min_angle_correction}"
        param_dir = output_base / param_name
        param_dir.mkdir(exist_ok=True)

        print(f"\n[{i}/{len(param_combinations)}] Testing: {param_name}")
        print(f"  angle_range: ±{angle_range}°")
        print(f"  angle_step: {angle_step}°")
        print(f"  min_angle_correction: {min_angle_correction}°")

        angle_log = []  # Track detected angles for analysis

        # Process each image with current parameters
        for image_path in image_files:
            try:
                # Load image
                image = load_image(image_path)

                # Deskew with current parameters
                deskewed, detected_angle = deskew_image(
                    image,
                    angle_range=angle_range,
                    angle_step=angle_step,
                    min_angle_correction=min_angle_correction,
                )

                # Save deskewed result
                base_name = image_path.stem
                output_path = param_dir / f"{base_name}_deskewed.jpg"
                save_image(deskewed, output_path)

                # Log angle information
                angle_log.append(
                    {
                        "image": image_path.name,
                        "detected_angle": detected_angle,
                        "correction_applied": abs(detected_angle)
                        >= min_angle_correction,
                    }
                )

                status = (
                    "[OK]" if abs(detected_angle) >= min_angle_correction else "[  ]"
                )
                print(f"    {status} {image_path.name} -> angle: {detected_angle:.2f}°")

            except Exception as e:
                print(f"    [ERROR] Error processing {image_path.name}: {e}")

        # Save angle analysis for this parameter set
        angle_log_file = param_dir / "angle_analysis.txt"
        with open(angle_log_file, "w") as f:
            f.write(f"DESKEWING ANALYSIS - {param_name}\n")
            f.write("=" * 50 + "\n")
            f.write("Parameters:\n")
            f.write(f"  angle_range: ±{angle_range}°\n")
            f.write(f"  angle_step: {angle_step}°\n")
            f.write(f"  min_angle_correction: {min_angle_correction}°\n\n")

            f.write("Per-image results:\n")
            f.write("-" * 30 + "\n")
            for entry in angle_log:
                status = "CORRECTED" if entry["correction_applied"] else "NO CHANGE"
                f.write(
                    f"{entry['image']}: {entry['detected_angle']:.2f}° ({status})\n"
                )

            # Statistics
            corrected_count = sum(
                1 for entry in angle_log if entry["correction_applied"]
            )
            avg_angle = np.mean([abs(entry["detected_angle"]) for entry in angle_log])

            f.write("\nStatistics:\n")
            f.write(f"  Images processed: {len(angle_log)}\n")
            f.write(f"  Images corrected: {corrected_count}\n")
            f.write(f"  Images unchanged: {len(angle_log) - corrected_count}\n")
            f.write(f"  Average detected angle: {avg_angle:.2f}°\n")

        # Record results for summary
        results_summary.append(
            {
                "params": param_name,
                "angle_range": angle_range,
                "angle_step": angle_step,
                "min_angle_correction": min_angle_correction,
                "output_dir": param_dir,
                "angle_log": angle_log,
            }
        )

    # Generate comprehensive summary report
    summary_file = output_base / "deskewing_test_summary.txt"
    with open(summary_file, "w") as f:
        f.write("DESKEWING PARAMETER TUNING RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test images: {len(image_files)}\n")
        f.write(f"Parameter combinations tested: {len(param_combinations)}\n\n")

        f.write("PARAMETER COMBINATIONS:\n")
        f.write("-" * 30 + "\n")
        for result in results_summary:
            corrected = sum(
                1 for entry in result["angle_log"] if entry["correction_applied"]
            )
            avg_angle = np.mean(
                [abs(entry["detected_angle"]) for entry in result["angle_log"]]
            )

            f.write(f"Folder: {result['params']}\n")
            f.write(f"  angle_range: ±{result['angle_range']}°\n")
            f.write(f"  angle_step: {result['angle_step']}°\n")
            f.write(f"  min_angle_correction: {result['min_angle_correction']}°\n")
            f.write(f"  Images corrected: {corrected}/{len(image_files)}\n")
            f.write(f"  Average detected angle: {avg_angle:.2f}°\n")
            f.write(f"  Results in: {result['output_dir']}\n\n")

        f.write("EVALUATION GUIDELINES:\n")
        f.write("-" * 30 + "\n")
        f.write("1. Look for images that appear properly aligned horizontally\n")
        f.write("2. Check that text lines are straight\n")
        f.write("3. Ensure table borders are horizontal/vertical\n")
        f.write("4. Avoid over-correction (unnecessary rotation of straight images)\n")
        f.write("5. Balance between correction sensitivity and stability\n\n")

        f.write("RECOMMENDED EVALUATION ORDER:\n")
        f.write("-" * 30 + "\n")
        f.write("1. Start with moderate parameters (range10_step0.2_min0.2)\n")
        f.write("2. Compare with more sensitive settings (min0.1)\n")
        f.write("3. Check more conservative settings (min0.5 or min1.0)\n")
        f.write("4. Choose based on your image quality needs\n\n")

        f.write("NEXT STEPS:\n")
        f.write("-" * 30 + "\n")
        f.write("1. Choose the best parameter combination\n")
        f.write("2. Copy the deskewed images from that folder to:\n")
        f.write("   data/output/tuning/03_roi_input/\n")
        f.write("3. Run: python tools/tune_roi_detection.py\n")

    print("\n[SUCCESS] DESKEWING PARAMETER TUNING COMPLETE!")
    print(f"[DIR] Results saved in: {output_base}")
    print(f"[FILE] Summary report: {summary_file}")
    print()
    print("EVALUATION GUIDELINES:")
    print("[CHECK] Look for properly aligned text and table borders")
    print("[WARNING] Balance between correction sensitivity and stability")
    print("[WARNING] Avoid over-correction of already straight images")
    print()
    print("NEXT STEPS:")
    print("1. [LIST] Choose the best parameter combination")
    print("2. [DIR] Copy best results to: data/output/tuning/03_roi_input/")
    print("3. [START] Run: python tools/tune_roi_detection.py")


if __name__ == "__main__":
    test_deskewing_parameters()
