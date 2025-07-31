#!/usr/bin/env python3
"""
Parameter Tuning Results Comparison Tool

This script helps compare results from different parameter combinations
by generating side-by-side comparisons and analysis reports.

Usage:
    python tools/compare_results.py [stage] [options]

Examples:
    python tools/compare_results.py page_splitting
    python tools/compare_results.py deskewing --top 5
    python tools/compare_results.py roi_detection --image Wang2017_Page_001
    python tools/compare_results.py line_detection --metric detection_rate
"""

import sys
import argparse
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr_pipeline.utils import load_image, save_image, get_image_files
import cv2
import numpy as np


def create_comparison_grid(image_paths, labels, output_path, grid_cols=3):
    """Create a grid comparison of multiple images."""
    if not image_paths:
        print("No images to compare")
        return False

    # Load all images
    images = []
    for img_path in image_paths:
        try:
            img = load_image(img_path)
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
            continue

    if not images:
        print("No valid images found")
        return False

    # Calculate grid dimensions
    num_images = len(images)
    grid_rows = (num_images + grid_cols - 1) // grid_cols

    # Resize all images to same height for better comparison
    target_height = 400
    resized_images = []
    for img in images:
        height, width = img.shape[:2]
        new_width = int(width * target_height / height)
        resized = cv2.resize(img, (new_width, target_height))
        resized_images.append(resized)

    # Find maximum width for padding
    max_width = max(img.shape[1] for img in resized_images)

    # Pad images to same width
    padded_images = []
    for i, img in enumerate(resized_images):
        height, width = img.shape[:2]
        if width < max_width:
            padding = max_width - width
            padded = cv2.copyMakeBorder(
                img, 0, 0, 0, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
        else:
            padded = img

        # Add label text
        label_height = 30
        label_img = np.ones((label_height, max_width, 3), dtype=np.uint8) * 255
        cv2.putText(
            label_img,
            labels[i] if i < len(labels) else f"Image {i+1}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )

        # Combine label and image
        combined = np.vstack([label_img, padded])
        padded_images.append(combined)

    # Create grid
    grid_height = target_height + 30  # Include label height
    grid_width = max_width

    rows = []
    for row in range(grid_rows):
        row_images = []
        for col in range(grid_cols):
            idx = row * grid_cols + col
            if idx < len(padded_images):
                row_images.append(padded_images[idx])
            else:
                # Empty placeholder
                empty = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 240
                row_images.append(empty)

        if row_images:
            row_combined = np.hstack(row_images)
            rows.append(row_combined)

    if rows:
        final_grid = np.vstack(rows)

        # Save comparison grid
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(final_grid, output_path)
        return True

    return False


def compare_page_splitting(top_n=None, specific_image=None):
    """Compare page splitting results."""
    print("🔍 COMPARING PAGE SPLITTING RESULTS")
    print("=" * 50)

    results_dir = Path("data/output/tuning/01_split_pages")
    comparison_dir = Path("data/output/tuning/comparisons/page_splitting")

    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return

    # Get parameter folders
    param_folders = [
        d for d in results_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]

    if not param_folders:
        print("❌ No parameter folders found")
        return

    # Limit to top N if specified
    if top_n:
        param_folders = param_folders[:top_n]

    print(f"📂 Found {len(param_folders)} parameter combinations")

    # Get test images
    test_images = []
    if param_folders:
        first_folder = param_folders[0]
        image_files = get_image_files(first_folder)
        test_images = list(set([f.name.split("_page_")[0] for f in image_files]))

    if specific_image:
        test_images = [img for img in test_images if specific_image in img]

    print(f"📷 Comparing {len(test_images)} base images")

    # Create comparisons for each test image
    for base_name in test_images:
        print(f"\nComparing: {base_name}")

        for page_num in [1, 2]:
            page_images = []
            labels = []

            for folder in param_folders:
                page_file = folder / f"{base_name}_page_{page_num}.jpg"
                if page_file.exists():
                    page_images.append(page_file)
                    labels.append(folder.name)

            if page_images:
                output_path = (
                    comparison_dir / f"{base_name}_page_{page_num}_comparison.jpg"
                )
                success = create_comparison_grid(page_images, labels, output_path)
                if success:
                    print(f"  ✅ Created comparison: {output_path}")

    print(f"\n📁 Comparisons saved to: {comparison_dir}")


def compare_deskewing(top_n=None, specific_image=None):
    """Compare deskewing results."""
    print("🔍 COMPARING DESKEWING RESULTS")
    print("=" * 50)

    results_dir = Path("data/output/tuning/02_deskewed")
    comparison_dir = Path("data/output/tuning/comparisons/deskewing")

    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return

    # Get parameter folders sorted by a metric (e.g., correction rate)
    param_folders = []
    for d in results_dir.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            analysis_file = d / "angle_analysis.txt"
            if analysis_file.exists():
                param_folders.append(d)

    # Sort by folder name for now (could add metric-based sorting)
    param_folders.sort(key=lambda x: x.name)

    if top_n:
        param_folders = param_folders[:top_n]

    print(f"📂 Found {len(param_folders)} parameter combinations")

    # Get test images
    test_images = []
    if param_folders:
        first_folder = param_folders[0]
        image_files = get_image_files(first_folder)
        test_images = [f.stem.replace("_deskewed", "") for f in image_files]

    if specific_image:
        test_images = [img for img in test_images if specific_image in img]

    # Create comparisons
    for base_name in test_images:
        print(f"\nComparing: {base_name}")

        deskewed_images = []
        labels = []

        for folder in param_folders:
            deskewed_file = folder / f"{base_name}_deskewed.jpg"
            if deskewed_file.exists():
                deskewed_images.append(deskewed_file)
                labels.append(folder.name)

        if deskewed_images:
            output_path = comparison_dir / f"{base_name}_deskewing_comparison.jpg"
            success = create_comparison_grid(deskewed_images, labels, output_path)
            if success:
                print(f"  ✅ Created comparison: {output_path}")

    print(f"\n📁 Comparisons saved to: {comparison_dir}")


def compare_roi_detection(top_n=None, specific_image=None):
    """Compare ROI detection results."""
    print("🔍 COMPARING ROI DETECTION RESULTS")
    print("=" * 50)

    results_dir = Path("data/output/tuning/03_roi_detection")
    comparison_dir = Path("data/output/tuning/comparisons/roi_detection")

    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return

    # Get parameter folders
    param_folders = [
        d for d in results_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]
    param_folders.sort(key=lambda x: x.name)

    if top_n:
        param_folders = param_folders[:top_n]

    print(f"📂 Found {len(param_folders)} parameter combinations")

    # Get test images
    test_images = []
    if param_folders:
        first_folder = param_folders[0]
        roi_files = list(first_folder.glob("*_roi.jpg"))
        test_images = [f.stem.replace("_roi", "") for f in roi_files]

    if specific_image:
        test_images = [img for img in test_images if specific_image in img]

    # Create comparisons for ROI crops
    for base_name in test_images:
        print(f"\nComparing ROI crops: {base_name}")

        roi_images = []
        labels = []

        for folder in param_folders:
            roi_file = folder / f"{base_name}_roi.jpg"
            if roi_file.exists():
                roi_images.append(roi_file)
                labels.append(folder.name)

        if roi_images:
            output_path = comparison_dir / f"{base_name}_roi_comparison.jpg"
            success = create_comparison_grid(roi_images, labels, output_path)
            if success:
                print(f"  ✅ Created ROI comparison: {output_path}")

        # Also compare Gabor responses
        print(f"Comparing Gabor responses: {base_name}")

        gabor_images = []
        gabor_labels = []

        for folder in param_folders:
            gabor_file = folder / f"{base_name}_gabor.jpg"
            if gabor_file.exists():
                gabor_images.append(gabor_file)
                gabor_labels.append(folder.name)

        if gabor_images:
            output_path = comparison_dir / f"{base_name}_gabor_comparison.jpg"
            success = create_comparison_grid(gabor_images, gabor_labels, output_path)
            if success:
                print(f"  ✅ Created Gabor comparison: {output_path}")

    print(f"\n📁 Comparisons saved to: {comparison_dir}")


def compare_line_detection(top_n=None, specific_image=None, metric=None):
    """Compare line detection results."""
    print("🔍 COMPARING LINE DETECTION RESULTS")
    print("=" * 50)

    results_dir = Path("data/output/tuning/04_line_detection")
    comparison_dir = Path("data/output/tuning/comparisons/line_detection")

    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return

    # Get parameter folders
    param_folders = [
        d for d in results_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]

    # Sort by metric if specified
    if metric == "detection_rate":
        # Sort by detection rate (would need to parse analysis files)
        param_folders.sort(key=lambda x: x.name)  # Simplified sorting
    else:
        param_folders.sort(key=lambda x: x.name)

    if top_n:
        param_folders = param_folders[:top_n]

    print(f"📂 Found {len(param_folders)} parameter combinations")

    # Get test images
    test_images = []
    if param_folders:
        first_folder = param_folders[0]
        line_files = list(first_folder.glob("*_lines.jpg"))
        test_images = [f.stem.replace("_lines", "") for f in line_files]

    if specific_image:
        test_images = [img for img in test_images if specific_image in img]

    # Create comparisons for line visualizations
    for base_name in test_images:
        print(f"\nComparing line detection: {base_name}")

        line_images = []
        labels = []

        for folder in param_folders:
            line_file = folder / f"{base_name}_lines.jpg"
            if line_file.exists():
                line_images.append(line_file)
                labels.append(folder.name)

        if line_images:
            output_path = comparison_dir / f"{base_name}_lines_comparison.jpg"
            success = create_comparison_grid(line_images, labels, output_path)
            if success:
                print(f"  ✅ Created lines comparison: {output_path}")

        # Also compare final table crops
        print(f"Comparing table crops: {base_name}")

        table_images = []
        table_labels = []

        for folder in param_folders:
            table_file = folder / f"{base_name}_table.jpg"
            if table_file.exists():
                table_images.append(table_file)
                table_labels.append(folder.name)

        if table_images:
            output_path = comparison_dir / f"{base_name}_table_comparison.jpg"
            success = create_comparison_grid(table_images, table_labels, output_path)
            if success:
                print(f"  ✅ Created table comparison: {output_path}")

    print(f"\n📁 Comparisons saved to: {comparison_dir}")


def generate_summary_report():
    """Generate a comprehensive summary report of all tuning results."""
    print("📊 GENERATING SUMMARY REPORT")
    print("=" * 50)

    report_path = Path("data/output/tuning/TUNING_SUMMARY_REPORT.md")

    with open(report_path, "w") as f:
        f.write("# OCR Parameter Tuning Summary Report\n\n")
        f.write("Generated by compare_results.py\n\n")

        # Page Splitting Summary
        f.write("## Page Splitting Results\n\n")
        split_dir = Path("data/output/tuning/01_split_pages")
        if split_dir.exists():
            folders = [d for d in split_dir.iterdir() if d.is_dir()]
            f.write(f"- Parameter combinations tested: {len(folders)}\n")
            f.write(f"- Results directory: `{split_dir}`\n")
            f.write(
                "- Key evaluation criteria: Clean gutter separation, no content loss\n\n"
            )

        # Deskewing Summary
        f.write("## Deskewing Results\n\n")
        deskew_dir = Path("data/output/tuning/02_deskewed")
        if deskew_dir.exists():
            folders = [d for d in deskew_dir.iterdir() if d.is_dir()]
            f.write(f"- Parameter combinations tested: {len(folders)}\n")
            f.write(f"- Results directory: `{deskew_dir}`\n")
            f.write(
                "- Key evaluation criteria: Horizontal text, straight table borders\n\n"
            )

        # ROI Detection Summary
        f.write("## ROI Detection Results\n\n")
        roi_dir = Path("data/output/tuning/03_roi_detection")
        if roi_dir.exists():
            folders = [d for d in roi_dir.iterdir() if d.is_dir()]
            f.write(f"- Parameter combinations tested: {len(folders)}\n")
            f.write(f"- Results directory: `{roi_dir}`\n")
            f.write(
                "- Key evaluation criteria: Content focus, margin removal, preservation\n\n"
            )

        # Line Detection Summary
        f.write("## Line Detection Results\n\n")
        line_dir = Path("data/output/tuning/04_line_detection")
        if line_dir.exists():
            folders = [d for d in line_dir.iterdir() if d.is_dir()]
            f.write(f"- Parameter combinations tested: {len(folders)}\n")
            f.write(f"- Results directory: `{line_dir}`\n")
            f.write(
                "- Key evaluation criteria: Table line detection, noise filtering\n\n"
            )

        # Comparison Images
        f.write("## Visual Comparisons\n\n")
        comp_dir = Path("data/output/tuning/comparisons")
        if comp_dir.exists():
            f.write(f"Side-by-side comparison images generated in: `{comp_dir}`\n\n")
            for stage_dir in comp_dir.iterdir():
                if stage_dir.is_dir():
                    images = list(stage_dir.glob("*.jpg"))
                    f.write(f"- {stage_dir.name}: {len(images)} comparison images\n")

        f.write("\n## Next Steps\n\n")
        f.write("1. Review visual comparisons in the comparisons directory\n")
        f.write("2. Select optimal parameters for each stage\n")
        f.write("3. Update TUNED_PARAMETERS in `tools/run_tuned_pipeline.py`\n")
        f.write("4. Run final tuned pipeline and validate results\n")

    print(f"✅ Summary report generated: {report_path}")


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(
        description="Compare OCR parameter tuning results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/compare_results.py page_splitting
  python tools/compare_results.py deskewing --top 5
  python tools/compare_results.py roi_detection --image Wang2017_Page_001
  python tools/compare_results.py line_detection --metric detection_rate
  python tools/compare_results.py summary
        """,
    )

    parser.add_argument(
        "stage",
        choices=[
            "page_splitting",
            "deskewing",
            "roi_detection",
            "line_detection",
            "summary",
        ],
        help="Tuning stage to compare",
    )

    parser.add_argument(
        "--top", type=int, help="Compare only top N parameter combinations"
    )

    parser.add_argument(
        "--image", help="Compare only specific image (partial name match)"
    )

    parser.add_argument(
        "--metric",
        choices=["detection_rate", "area_ratio", "correction_rate"],
        help="Sort parameter combinations by metric",
    )

    args = parser.parse_args()

    print("🔍 OCR PARAMETER TUNING RESULTS COMPARISON")
    print("=" * 60)

    if args.stage == "summary":
        generate_summary_report()
    elif args.stage == "page_splitting":
        compare_page_splitting(args.top, args.image)
    elif args.stage == "deskewing":
        compare_deskewing(args.top, args.image)
    elif args.stage == "roi_detection":
        compare_roi_detection(args.top, args.image)
    elif args.stage == "line_detection":
        compare_line_detection(args.top, args.image, args.metric)

    print("\n✅ Comparison complete!")
    print("💡 Review the generated comparison images to select optimal parameters")


if __name__ == "__main__":
    main()
