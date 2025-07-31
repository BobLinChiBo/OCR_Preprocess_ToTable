#!/usr/bin/env python3
"""
Tuned Pipeline Runner

This script runs the complete OCR pipeline with your manually tuned parameters.
After completing the parameter tuning process, update the TUNED_PARAMETERS
section below with your optimal values.

Usage:
    python tools/run_tuned_pipeline.py [input_dir] [options]

Example:
    python tools/run_tuned_pipeline.py data/input/test_images/ --verbose
    python tools/run_tuned_pipeline.py data/input/test_images/ -o custom_output/
"""

import sys
import argparse
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr_pipeline.config import Stage1Config, Stage2Config  # noqa: E402
from src.ocr_pipeline.pipeline import TwoStageOCRPipeline  # noqa: E402


# =============================================================================
# TUNED PARAMETERS - UPDATE THESE WITH YOUR OPTIMAL VALUES
# =============================================================================

TUNED_PARAMETERS = {
    # Page Splitting Parameters (from tune_page_splitting.py results)
    "page_splitting": {
        "gutter_search_start": 0.4,  # Update with your optimal value
        "gutter_search_end": 0.6,  # Update with your optimal value
        # Note: min_gutter_width not used in current implementation
    },
    # Deskewing Parameters (from tune_deskewing.py results)
    "deskewing": {
        "angle_range": 10,  # Update with your optimal value
        "angle_step": 0.2,  # Update with your optimal value
        "min_angle_correction": 0.2,  # Update with your optimal value
    },
    # ROI Detection Parameters (from tune_roi_detection.py results)
    "roi_detection": {
        "gabor_kernel_size": 31,  # Update with your optimal value
        "gabor_sigma": 4.0,  # Update with your optimal value
        "gabor_lambda": 8.0,  # Update with your optimal value
        "gabor_gamma": 0.2,  # Usually keep default
        "gabor_binary_threshold": 127,  # Usually keep default
        "roi_min_cut_strength": 20.0,  # Update with your optimal value
        "roi_min_confidence_threshold": 5.0,  # Update with your optimal value
        # Keep these defaults unless you have specific needs
        "roi_vertical_mode": "single_best",
        "roi_horizontal_mode": "both_sides",
        "roi_window_size_divisor": 20,
        "roi_min_window_size": 10,
    },
    # Line Detection Parameters (from tune_line_detection.py results)
    "line_detection": {
        "stage1": {
            "min_line_length": 40,  # Update with your optimal value for Stage 1
            "max_line_gap": 15,  # Update with your optimal value for Stage 1
        },
        "stage2": {
            "min_line_length": 30,  # Update with your optimal value for Stage 2
            "max_line_gap": 5,  # Update with your optimal value for Stage 2
        },
    },
}


def create_tuned_stage1_config(input_dir, output_base, verbose=False, debug=False):
    """Create Stage 1 configuration with tuned parameters."""
    params = TUNED_PARAMETERS

    return Stage1Config(
        input_dir=input_dir,
        output_dir=output_base / "stage1_initial_processing",
        verbose=verbose,
        save_debug_images=debug,
        # Page splitting parameters
        gutter_search_start=params["page_splitting"]["gutter_search_start"],
        gutter_search_end=params["page_splitting"]["gutter_search_end"],
        # Deskewing parameters
        angle_range=params["deskewing"]["angle_range"],
        angle_step=params["deskewing"]["angle_step"],
        min_angle_correction=params["deskewing"]["min_angle_correction"],
        # Line detection parameters (Stage 1)
        min_line_length=params["line_detection"]["stage1"]["min_line_length"],
        max_line_gap=params["line_detection"]["stage1"]["max_line_gap"],
        # ROI detection parameters
        enable_roi_detection=True,
        gabor_kernel_size=params["roi_detection"]["gabor_kernel_size"],
        gabor_sigma=params["roi_detection"]["gabor_sigma"],
        gabor_lambda=params["roi_detection"]["gabor_lambda"],
        gabor_gamma=params["roi_detection"]["gabor_gamma"],
        gabor_binary_threshold=params["roi_detection"]["gabor_binary_threshold"],
        roi_vertical_mode=params["roi_detection"]["roi_vertical_mode"],
        roi_horizontal_mode=params["roi_detection"]["roi_horizontal_mode"],
        roi_window_size_divisor=params["roi_detection"]["roi_window_size_divisor"],
        roi_min_window_size=params["roi_detection"]["roi_min_window_size"],
        roi_min_cut_strength=params["roi_detection"]["roi_min_cut_strength"],
        roi_min_confidence_threshold=params["roi_detection"][
            "roi_min_confidence_threshold"
        ],
    )


def create_tuned_stage2_config(output_base, verbose=False, debug=False):
    """Create Stage 2 configuration with tuned parameters."""
    params = TUNED_PARAMETERS

    return Stage2Config(
        input_dir=output_base / "stage1_initial_processing" / "05_cropped_tables",
        output_dir=output_base / "stage2_refinement",
        verbose=verbose,
        save_debug_images=debug,
        # Deskewing parameters (fine-tuning for Stage 2)
        angle_range=params["deskewing"]["angle_range"],
        angle_step=params["deskewing"]["angle_step"],
        min_angle_correction=params["deskewing"]["min_angle_correction"],
        # Line detection parameters (Stage 2 - more precise)
        min_line_length=params["line_detection"]["stage2"]["min_line_length"],
        max_line_gap=params["line_detection"]["stage2"]["max_line_gap"],
        # ROI detection disabled for Stage 2 (already cropped)
        enable_roi_detection=False,
    )


def print_parameter_summary():
    """Print a summary of the tuned parameters being used."""
    print("🔧 TUNED PARAMETERS SUMMARY")
    print("=" * 50)

    params = TUNED_PARAMETERS

    print("📄 Page Splitting:")
    print(f"   gutter_search_start: {params['page_splitting']['gutter_search_start']}")
    print(f"   gutter_search_end: {params['page_splitting']['gutter_search_end']}")

    print("\n🔄 Deskewing:")
    print(f"   angle_range: ±{params['deskewing']['angle_range']}°")
    print(f"   angle_step: {params['deskewing']['angle_step']}°")
    print(f"   min_angle_correction: {params['deskewing']['min_angle_correction']}°")

    print("\n🎯 ROI Detection:")
    print(f"   gabor_kernel_size: {params['roi_detection']['gabor_kernel_size']}")
    print(f"   gabor_sigma: {params['roi_detection']['gabor_sigma']}")
    print(f"   gabor_lambda: {params['roi_detection']['gabor_lambda']}")
    print(f"   roi_min_cut_strength: {params['roi_detection']['roi_min_cut_strength']}")
    print(
        f"   roi_min_confidence_threshold: {params['roi_detection']['roi_min_confidence_threshold']}"
    )

    print("\n📏 Line Detection:")
    print(
        f"   Stage 1 - min_line_length: {params['line_detection']['stage1']['min_line_length']}px"
    )
    print(
        f"   Stage 1 - max_line_gap: {params['line_detection']['stage1']['max_line_gap']}px"
    )
    print(
        f"   Stage 2 - min_line_length: {params['line_detection']['stage2']['min_line_length']}px"
    )
    print(
        f"   Stage 2 - max_line_gap: {params['line_detection']['stage2']['max_line_gap']}px"
    )
    print()


def main():
    """Main entry point for tuned pipeline."""
    parser = argparse.ArgumentParser(
        description="OCR Pipeline with Tuned Parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/run_tuned_pipeline.py data/input/test_images/         # Use test images
  python tools/run_tuned_pipeline.py data/input/raw_images/          # Use full dataset
  python tools/run_tuned_pipeline.py input/ -o results/ --verbose    # Custom paths
  python tools/run_tuned_pipeline.py input/ --stage1-only            # Only Stage 1
Note: Update TUNED_PARAMETERS in this script with your optimal values first!
        """,
    )

    parser.add_argument(
        "input",
        nargs="?",
        default="data/input/test_images",
        help="Input directory with images (default: data/input/test_images/)",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="data/output/tuned_pipeline",
        help="Base output directory (default: data/output/tuned_pipeline/)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--debug", action="store_true", help="Save debug images during processing"
    )

    parser.add_argument(
        "--stage1-only",
        action="store_true",
        help="Run only Stage 1 (initial processing)",
    )

    parser.add_argument(
        "--stage2-only",
        action="store_true",
        help="Run only Stage 2 (refinement) - requires Stage 1 output",
    )

    parser.add_argument(
        "--show-params", action="store_true", help="Show parameter summary and exit"
    )

    args = parser.parse_args()

    # Show parameters and exit if requested
    if args.show_params:
        print_parameter_summary()
        return

    # Validate arguments
    if args.stage1_only and args.stage2_only:
        print("❌ Error: Cannot specify both --stage1-only and --stage2-only")
        sys.exit(1)

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input path does not exist: {input_path}")
        sys.exit(1)

    # Create output base directory
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    print("🏃 TUNED OCR PIPELINE")
    print("=" * 50)
    print(f"📂 Input: {input_path}")
    print(f"📁 Output: {output_base}")

    if args.verbose:
        print()
        print_parameter_summary()

    # Create configurations with tuned parameters
    stage1_config = create_tuned_stage1_config(
        input_path if input_path.is_dir() else input_path.parent,
        output_base,
        verbose=args.verbose,
        debug=args.debug,
    )

    stage2_config = create_tuned_stage2_config(
        output_base, verbose=args.verbose, debug=args.debug
    )

    try:
        # Create pipeline
        pipeline = TwoStageOCRPipeline(stage1_config, stage2_config)

        if args.stage2_only:
            # Run only Stage 2
            if args.verbose:
                print("⏭️  Running Stage 2 only (refinement processing)")
            results = pipeline.run_stage2()

        elif args.stage1_only:
            # Run only Stage 1
            if args.verbose:
                print("🚀 Running Stage 1 only (initial processing)")
            results = pipeline.run_stage1(input_path if input_path.is_dir() else None)

        else:
            # Run complete pipeline
            if args.verbose:
                print("🚀 Running complete two-stage pipeline")
            results = pipeline.run_complete_pipeline(
                input_path if input_path.is_dir() else None
            )

        # Print final results
        print("\n🎉 TUNED PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"📊 Processed {len(results)} images")

        if args.stage1_only:
            print(
                f"📁 Stage 1 results: {stage1_config.output_dir / '05_cropped_tables'}"
            )
            print("💡 Next: Run with --stage2-only or run complete pipeline")
        elif args.stage2_only:
            print(
                f"📁 Stage 2 results: {stage2_config.output_dir / '04_fitted_tables'}"
            )
        else:
            print(f"📁 Final results: {stage2_config.output_dir / '04_fitted_tables'}")

        print("\n📝 PARAMETER TUNING RECORD:")
        print("   These results used the tuned parameters in this script.")
        print("   Compare with default pipeline to evaluate improvements.")

        return True

    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
