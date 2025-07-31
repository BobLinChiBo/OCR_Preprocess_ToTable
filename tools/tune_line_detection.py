#!/usr/bin/env python3
"""
Line Detection Parameter Tuning Tool

This script helps tune line detection parameters by testing different combinations
of minimum line length and maximum line gap for table line detection.

Usage:
    python tools/tune_line_detection.py

Input: ROI-cropped images from previous tuning stage
Output: Line detection visualizations and table crops with different parameters
"""

import sys
from pathlib import Path
import itertools
import json

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr_pipeline.utils import (  # noqa: E402
    load_image,
    save_image,
    get_image_files,
    detect_table_lines,
    crop_table_region,
    visualize_detected_lines,
)


def test_line_detection_parameters():
    """Test different line detection parameter combinations."""

    # Input and output directories
    input_dir = Path("data/output/tuning/04_line_input")
    output_base = Path("data/output/tuning/04_line_detection")

    # Parameter ranges to test
    min_line_lengths = [20, 30, 40, 50, 60, 80]
    max_line_gaps = [5, 10, 15, 20, 25]

    print("[CONFIG] LINE DETECTION PARAMETER TUNING")
    print("=" * 50)
    print(f"[DIR] Input: {input_dir}")
    print(f"[DIR] Output: {output_base}")
    print()

    # Validate input directory
    if not input_dir.exists():
        print(f"[ERROR] Error: Input directory not found: {input_dir}")
        print("Please first run ROI detection tuning and copy the best results to:")
        print(f"   {input_dir}")
        return

    # Get ROI-cropped images
    image_files = get_image_files(input_dir)
    if not image_files:
        print(f"[ERROR] Error: No image files found in {input_dir}")
        print("Please copy ROI-cropped images from ROI detection tuning results")
        return

    print(f"Found {len(image_files)} ROI-cropped images:")
    for img_file in image_files:
        print(f"  - {img_file.name}")
    print()

    # Clean up output directory
    if output_base.exists():
        import shutil

        shutil.rmtree(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    # Create parameter combinations
    param_combinations = list(itertools.product(min_line_lengths, max_line_gaps))

    print(f"Testing {len(param_combinations)} parameter combinations:")
    print("Parameters: min_line_length, max_line_gap")

    results_summary = []

    # Test each parameter combination
    for i, (min_line_length, max_line_gap) in enumerate(param_combinations, 1):
        param_name = f"minlen{min_line_length}_maxgap{max_line_gap}"
        param_dir = output_base / param_name
        param_dir.mkdir(exist_ok=True)

        print(f"\n[{i}/{len(param_combinations)}] Testing: {param_name}")
        print(f"  min_line_length: {min_line_length}px")
        print(f"  max_line_gap: {max_line_gap}px")

        line_log = []  # Track line detection results

        # Process each image with current parameters
        for image_path in image_files:
            try:
                # Load image
                image = load_image(image_path)

                # Detect lines with current parameters
                h_lines, v_lines, analysis = detect_table_lines(
                    image,
                    min_line_length=min_line_length,
                    max_line_gap=max_line_gap,
                    return_analysis=True,
                )

                base_name = image_path.stem

                # Create line visualization
                line_vis = visualize_detected_lines(image, h_lines, v_lines)
                vis_path = param_dir / f"{base_name}_lines.jpg"
                save_image(line_vis, vis_path)

                # Crop to table region if lines detected
                if h_lines and v_lines:
                    cropped_table = crop_table_region(image, h_lines, v_lines)
                    crop_path = param_dir / f"{base_name}_table.jpg"
                    save_image(cropped_table, crop_path)

                    # Calculate table crop area
                    height, width = image.shape[:2]
                    crop_height, crop_width = cropped_table.shape[:2]
                    crop_area_ratio = (crop_width * crop_height) / (width * height)
                else:
                    # No lines detected, save original as fallback
                    crop_path = param_dir / f"{base_name}_table.jpg"
                    save_image(image, crop_path)
                    crop_area_ratio = 1.0

                # Save line detection data as JSON
                data_path = param_dir / f"{base_name}_lines.json"
                with open(data_path, "w") as f:
                    json.dump(
                        {
                            "parameters": {
                                "min_line_length": min_line_length,
                                "max_line_gap": max_line_gap,
                            },
                            "analysis": analysis,
                            "crop_area_ratio": crop_area_ratio,
                            "lines_detected": len(h_lines) > 0 and len(v_lines) > 0,
                        },
                        f,
                        indent=2,
                    )

                # Log results
                line_log.append(
                    {
                        "image": image_path.name,
                        "h_lines": len(h_lines),
                        "v_lines": len(v_lines),
                        "total_lines": analysis["total_lines"],
                        "lines_detected": len(h_lines) > 0 and len(v_lines) > 0,
                        "crop_area_ratio": crop_area_ratio,
                    }
                )

                status = "[OK]" if len(h_lines) > 0 and len(v_lines) > 0 else "[  ]"
                print(
                    f"    {status} {image_path.name} -> H:{len(h_lines)} V:{len(v_lines)}"
                )

            except Exception as e:
                print(f"    [ERROR] Error processing {image_path.name}: {e}")

        # Save analysis for this parameter set
        analysis_file = param_dir / "line_detection_analysis.txt"
        with open(analysis_file, "w") as f:
            f.write(f"LINE DETECTION ANALYSIS - {param_name}\n")
            f.write("=" * 50 + "\n")
            f.write("Parameters:\n")
            f.write(f"  min_line_length: {min_line_length}px\n")
            f.write(f"  max_line_gap: {max_line_gap}px\n\n")

            f.write("Per-image results:\n")
            f.write("-" * 30 + "\n")
            for entry in line_log:
                status = "DETECTED" if entry["lines_detected"] else "NO LINES"
                f.write(f"{entry['image']}:\n")
                f.write(f"  Horizontal lines: {entry['h_lines']}\n")
                f.write(f"  Vertical lines: {entry['v_lines']}\n")
                f.write(f"  Total lines: {entry['total_lines']}\n")
                f.write(f"  Status: {status}\n")
                f.write(f"  Crop area ratio: {entry['crop_area_ratio']:.2%}\n\n")

            # Statistics
            if line_log:
                detected_count = sum(1 for entry in line_log if entry["lines_detected"])
                avg_h_lines = sum(entry["h_lines"] for entry in line_log) / len(
                    line_log
                )
                avg_v_lines = sum(entry["v_lines"] for entry in line_log) / len(
                    line_log
                )
                avg_crop_ratio = sum(
                    entry["crop_area_ratio"] for entry in line_log
                ) / len(line_log)

                f.write("Statistics:\n")
                f.write(f"  Images processed: {len(line_log)}\n")
                f.write(f"  Images with lines detected: {detected_count}\n")
                f.write(f"  Detection rate: {detected_count/len(line_log):.1%}\n")
                f.write(f"  Average horizontal lines: {avg_h_lines:.1f}\n")
                f.write(f"  Average vertical lines: {avg_v_lines:.1f}\n")
                f.write(f"  Average crop area ratio: {avg_crop_ratio:.2%}\n")

        # Record results for summary
        results_summary.append(
            {
                "params": param_name,
                "min_line_length": min_line_length,
                "max_line_gap": max_line_gap,
                "output_dir": param_dir,
                "line_log": line_log,
            }
        )

    # Generate comprehensive summary report
    summary_file = output_base / "line_detection_test_summary.txt"
    with open(summary_file, "w") as f:
        f.write("LINE DETECTION PARAMETER TUNING RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test images: {len(image_files)}\n")
        f.write(f"Parameter combinations tested: {len(param_combinations)}\n\n")

        f.write("PARAMETER COMBINATIONS:\n")
        f.write("-" * 30 + "\n")
        for result in results_summary:
            if result["line_log"]:
                detected = sum(
                    1 for entry in result["line_log"] if entry["lines_detected"]
                )
                detection_rate = detected / len(result["line_log"])
                avg_h = sum(entry["h_lines"] for entry in result["line_log"]) / len(
                    result["line_log"]
                )
                avg_v = sum(entry["v_lines"] for entry in result["line_log"]) / len(
                    result["line_log"]
                )
                avg_crop = sum(
                    entry["crop_area_ratio"] for entry in result["line_log"]
                ) / len(result["line_log"])
            else:
                detection_rate = 0
                avg_h = avg_v = avg_crop = 0

            f.write(f"Folder: {result['params']}\n")
            f.write(f"  min_line_length: {result['min_line_length']}px\n")
            f.write(f"  max_line_gap: {result['max_line_gap']}px\n")
            f.write(f"  Detection rate: {detection_rate:.1%}\n")
            f.write(f"  Avg H lines: {avg_h:.1f}, Avg V lines: {avg_v:.1f}\n")
            f.write(f"  Avg crop area: {avg_crop:.2%}\n")
            f.write(f"  Results in: {result['output_dir']}\n\n")

        f.write("EVALUATION GUIDELINES:\n")
        f.write("-" * 30 + "\n")
        f.write("1. Look for consistent line detection across similar table types\n")
        f.write("2. Check line visualization images for accuracy\n")
        f.write("3. Ensure table boundaries are properly detected\n")
        f.write("4. Balance between sensitivity (detecting faint lines) and noise\n")
        f.write("5. Verify that table crops contain the complete table structure\n\n")

        f.write("PARAMETER SELECTION TIPS:\n")
        f.write("-" * 30 + "\n")
        f.write(
            "• Lower min_line_length: More sensitive, detects shorter line segments\n"
        )
        f.write("• Higher min_line_length: More selective, reduces noise\n")
        f.write("• Lower max_line_gap: Requires continuous lines\n")
        f.write("• Higher max_line_gap: Bridges gaps, connects broken lines\n\n")

        f.write("NEXT STEPS:\n")
        f.write("-" * 30 + "\n")
        f.write("1. Choose the best parameter combination\n")
        f.write("2. Note the optimal parameters for your document type\n")
        f.write("3. Run: python tools/run_tuned_pipeline.py\n")
        f.write("4. Use the optimal parameters in your production pipeline\n")

    print("\n[SUCCESS] LINE DETECTION PARAMETER TUNING COMPLETE!")
    print(f"[DIR] Results saved in: {output_base}")
    print(f"[FILE] Summary report: {summary_file}")
    print()
    print("EVALUATION GUIDELINES:")
    print("[CHECK] Check line visualization images for detection accuracy")
    print("[TARGET] Look for complete table structure detection")
    print("[WARNING] Balance sensitivity vs noise reduction")
    print()
    print("NEXT STEPS:")
    print("1. [LIST] Choose the best parameter combination")
    print("2. [FILE] Note optimal parameters for your document type")
    print("3. [START] Run: python tools/run_tuned_pipeline.py")


if __name__ == "__main__":
    test_line_detection_parameters()
