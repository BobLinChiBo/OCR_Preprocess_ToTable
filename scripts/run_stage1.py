#!/usr/bin/env python3
"""
Stage 1: Initial Processing CLI

Runs the initial processing stage of the two-stage OCR pipeline.
Processes raw scanned images through page splitting, deskewing, ROI detection,
line detection, table reconstruction, and cropping.

Usage:
    python run_stage1.py [input_dir] [options]

Example:
    python run_stage1.py input/ -o output/stage1 --verbose
    python run_stage1.py image.jpg -o output/stage1 --debug
"""

import argparse
import sys
from pathlib import Path
import shutil
import traceback

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr_pipeline.config import get_stage1_config  # noqa: E402
from src.ocr_pipeline.pipeline import TwoStageOCRPipeline  # noqa: E402


def main():
    """Main entry point for Stage 1 processing."""
    parser = argparse.ArgumentParser(
        description="OCR Stage 1: Initial Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_stage1.py                           # Use default (data/input/raw_images/)
  python run_stage1.py input/                    # Process input/ directory
  python run_stage1.py image.jpg                 # Process single image
  python run_stage1.py --test-images             # Process test images
  python run_stage1.py --test-images --verbose   # Test images with verbose output
  python run_stage1.py input/ -o stage1_output/  # Custom output directory
        """,
    )

    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help=(
            "Input directory with raw images or single image file "
            "(default: from config file or data/input/raw_images/)"
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output directory for Stage 1 results "
            "(default: from config file)"
        ),
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--debug", action="store_true", help="Save debug images during processing"
    )

    parser.add_argument(
        "--test-images",
        action="store_true",
        help="Use test images directory (data/input/test_images/) instead of default input"
    )

    parser.add_argument(
        "--angle-range",
        type=int,
        default=None,
        help="Deskewing angle range in degrees (default: from config file)",
    )

    parser.add_argument(
        "--angle-step",
        type=float,
        default=None,
        help="Deskewing angle step in degrees (default: from config file)",
    )

    parser.add_argument(
        "--min-line-length",
        type=int,
        default=None,
        help="Minimum line length for table detection (default: from config file)",
    )

    parser.add_argument(
        "--disable-roi",
        action="store_true",
        help="Disable ROI detection preprocessing",
    )

    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Path to JSON configuration file " "(default: configs/stage1_default.json)"
        ),
    )

    args = parser.parse_args()

    # Create Stage 1 configuration
    # Start with JSON config if provided, otherwise use defaults
    stage1_config = get_stage1_config(args.config)

    # Handle input path
    if args.test_images:
        input_path = Path("data/input/test_images")
    elif args.input is not None:
        input_path = Path(args.input)
    else:
        # Use config file's input_dir
        input_path = stage1_config.input_dir
    
    # Validate input
    if not input_path.exists():
        print(f"❌ Error: Input path does not exist: {input_path}")
        sys.exit(1)

    # Override config with command line arguments only if provided
    stage1_config.input_dir = input_path  # Always set the resolved input path
    
    if args.output is not None:
        stage1_config.output_dir = Path(args.output)
    
    # Always override these flags
    stage1_config.verbose = args.verbose
    stage1_config.save_debug_images = args.debug
    stage1_config.enable_roi_detection = not args.disable_roi
    
    # Only override numeric parameters if provided
    if args.angle_range is not None:
        stage1_config.angle_range = args.angle_range
    if args.angle_step is not None:
        stage1_config.angle_step = args.angle_step
    if args.min_line_length is not None:
        stage1_config.min_line_length = args.min_line_length

    try:
        if args.verbose:
            print("*** OCR STAGE 1: INITIAL PROCESSING ***")
            print("=" * 60)
            print(f"Input: {input_path}")
            print(f"Output: {stage1_config.output_dir}")
            print("Parameters:")
            print(f"   - Angle range: ±{stage1_config.angle_range}°")
            print(f"   - Angle step: {stage1_config.angle_step}°")
            print(f"   - Min line length: {stage1_config.min_line_length}px")
            roi_status = "enabled" if stage1_config.enable_roi_detection else "disabled"
            print(f"   - ROI detection: {roi_status}")
            debug_status = "enabled" if stage1_config.save_debug_images else "disabled"
            print(f"   - Debug images: {debug_status}")
            print()

        # Create and run Stage 1 pipeline
        pipeline = TwoStageOCRPipeline(stage1_config=stage1_config)

        # Pass the input path directly - pipeline now handles both files and directories
        results = pipeline.run_stage1(input_path)

        # Print results
        if args.verbose:
            print("\n*** STAGE 1 COMPLETED SUCCESSFULLY! ***")
            print(f"Generated {len(results)} cropped table images")
            print(f"Results saved to: {stage1_config.output_dir}")
            print(
                "Cropped tables ready for Stage 2: "
                f"{stage1_config.output_dir / '07_border_cropped'}"
            )
            print()
            print("Next steps:")
            print("   python run_stage2.py  # Run Stage 2 refinement")
            print(f"   python run_complete.py {args.input}  # Run both stages")
        else:
            print(
                f"Stage 1 complete: {len(results)} cropped tables -> "
                f"{stage1_config.output_dir / '07_border_cropped'}"
            )

        return True

    except KeyboardInterrupt:
        print("\nStage 1 interrupted by user")
        return False
    except Exception as e:
        print(f"Stage 1 failed: {e}")
        if args.verbose:
            traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
