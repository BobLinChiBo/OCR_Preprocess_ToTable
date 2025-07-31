#!/usr/bin/env python3
"""
Stage 2: Refinement Processing CLI

Runs the refinement stage of the two-stage OCR pipeline.
Takes cropped table images from Stage 1 and refines them through
re-deskewing, precise line detection, table reconstruction, and fitting.

Usage:
    python run_stage2.py [input_dir] [options]

Example:
    python run_stage2.py  # Use default Stage 1 output
    python run_stage2.py output/stage1_initial_processing/05_cropped_tables/ --verbose
    python run_stage2.py cropped_tables/ -o output/stage2 --debug
"""

import argparse
import sys
import traceback
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr_pipeline.pipeline import TwoStageOCRPipeline  # noqa: E402
from src.ocr_pipeline.config import get_stage2_config  # noqa: E402


def main():
    """Main entry point for Stage 2 processing."""
    parser = argparse.ArgumentParser(
        description="OCR Stage 2: Refinement Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_stage2.py                                      # Use default Stage 1 output
  python run_stage2.py cropped_tables/                      # Process custom directory
  python run_stage2.py stage1_output/05_cropped_tables/     # Specific Stage 1 output
  python run_stage2.py cropped/ -o refined/ --verbose       # Custom input/output with verbose
        """,
    )

    parser.add_argument(
        "input",
        nargs="?",
        default="data/output/stage1_initial_processing/05_cropped_tables",
        help=(
            "Input directory with cropped tables from Stage 1 "
            "(default: data/output/stage1_initial_processing/05_cropped_tables/)"
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        default="data/output/stage2_refinement",
        help="Output directory for Stage 2 results (default: data/output/stage2_refinement/)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--debug", action="store_true", help="Save debug images during processing"
    )

    parser.add_argument(
        "--angle-range",
        type=int,
        default=10,
        help="Refinement deskewing angle range in degrees (default: 10)",
    )

    parser.add_argument(
        "--angle-step",
        type=float,
        default=0.2,
        help="Refinement deskewing angle step in degrees (default: 0.2)",
    )

    parser.add_argument(
        "--min-line-length",
        type=int,
        default=30,
        help="Minimum line length for refined table detection (default: 30)",
    )

    parser.add_argument(
        "--max-line-gap",
        type=int,
        default=5,
        help="Maximum line gap for refined table detection (default: 5)",
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to JSON configuration file (default: configs/stage2_default.json)",
    )

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_path}")
        print("Hint: Run Stage 1 first with: python run_stage1.py")
        sys.exit(1)

    if not input_path.is_dir():
        print(
            f"Error: Stage 2 requires a directory of cropped table images: {input_path}"
        )
        sys.exit(1)

    # Check if input directory has images
    from src.ocr_pipeline import utils

    image_files = utils.get_image_files(input_path)
    if not image_files:
        print(f"Error: No image files found in: {input_path}")
        print("Hint: Make sure Stage 1 completed successfully")
        sys.exit(1)

    # Create Stage 2 configuration
    # Start with JSON config if provided, otherwise use defaults
    stage2_config = get_stage2_config(args.config)

    # Override with command line arguments
    stage2_config.input_dir = input_path
    stage2_config.output_dir = Path(args.output)
    stage2_config.verbose = args.verbose
    stage2_config.save_debug_images = args.debug
    stage2_config.angle_range = args.angle_range
    stage2_config.angle_step = args.angle_step
    stage2_config.min_line_length = args.min_line_length
    stage2_config.max_line_gap = args.max_line_gap

    try:
        if args.verbose:
            print("*** OCR STAGE 2: REFINEMENT PROCESSING ***")
            print("=" * 60)
            print(f"Input: {input_path}")
            print(f"Output: {stage2_config.output_dir}")
            print(f"Found: {len(image_files)} cropped table images")
            print("Parameters:")
            print(f"   - Angle range: ±{stage2_config.angle_range}°")
            print(f"   - Angle step: {stage2_config.angle_step}°")
            print(f"   - Min line length: {stage2_config.min_line_length}px")
            print(f"   - Max line gap: {stage2_config.max_line_gap}px")
            print(
                f"   - Debug images: {'enabled' if stage2_config.save_debug_images else 'disabled'}"
            )
            print()

        # Create and run Stage 2 pipeline
        pipeline = TwoStageOCRPipeline(stage2_config=stage2_config)
        results = pipeline.run_stage2(input_path)

        # Print results
        if args.verbose:
            print("\n*** STAGE 2 COMPLETED SUCCESSFULLY! ***")
            print(f"Generated {len(results)} publication-ready table images")
            print(f"Results saved to: {stage2_config.output_dir}")
            print(f"Final tables: {stage2_config.output_dir / '04_fitted_tables'}")
            print()
            print("Two-stage pipeline complete!")
            print(
                "   Your tables are ready for use in publications, reports, or further analysis."
            )
        else:
            output_path = stage2_config.output_dir / '04_fitted_tables'
            print(f"Stage 2 complete: {len(results)} refined tables -> {output_path}")

        return True

    except KeyboardInterrupt:
        print("\nStage 2 interrupted by user")
        return False
    except Exception as e:
        print(f"Stage 2 failed: {e}")
        if args.verbose:
            traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
