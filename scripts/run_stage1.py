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

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr_pipeline.pipeline import TwoStageOCRPipeline
from src.ocr_pipeline.config import Stage1Config


def main():
    """Main entry point for Stage 1 processing."""
    parser = argparse.ArgumentParser(
        description="OCR Stage 1: Initial Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_stage1.py input/                    # Process input/ directory
  python run_stage1.py image.jpg                 # Process single image
  python run_stage1.py input/ -o stage1_output/  # Custom output directory
  python run_stage1.py input/ --verbose --debug  # Verbose output with debug images
        """
    )
    
    parser.add_argument(
        "input",
        nargs="?",
        default="data/input",
        help="Input directory with raw images or single image file (default: data/input/)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="data/output/stage1_initial_processing",
        help="Output directory for Stage 1 results (default: data/output/stage1_initial_processing/)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save debug images during processing"
    )
    
    parser.add_argument(
        "--angle-range",
        type=int,
        default=10,
        help="Deskewing angle range in degrees (default: 10)"
    )
    
    parser.add_argument(
        "--angle-step",
        type=float,
        default=0.2,
        help="Deskewing angle step in degrees (default: 0.2)"
    )
    
    parser.add_argument(
        "--min-line-length",
        type=int,
        default=40,
        help="Minimum line length for table detection (default: 40)"
    )
    
    parser.add_argument(
        "--disable-roi",
        action="store_true",
        help="Disable ROI detection preprocessing"
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Create Stage 1 configuration
    stage1_config = Stage1Config(
        input_dir=input_path if input_path.is_dir() else input_path.parent,
        output_dir=Path(args.output),
        verbose=args.verbose,
        save_debug_images=args.debug,
        angle_range=args.angle_range,
        angle_step=args.angle_step,
        min_line_length=args.min_line_length,
        enable_roi_detection=not args.disable_roi
    )
    
    try:
        if args.verbose:
            print("*** OCR STAGE 1: INITIAL PROCESSING ***")
            print("=" * 60)
            print(f"Input: {input_path}")
            print(f"Output: {stage1_config.output_dir}")
            print(f"Parameters:")
            print(f"   - Angle range: ±{stage1_config.angle_range}°")
            print(f"   - Angle step: {stage1_config.angle_step}°")
            print(f"   - Min line length: {stage1_config.min_line_length}px")
            print(f"   - ROI detection: {'enabled' if stage1_config.enable_roi_detection else 'disabled'}")
            print(f"   - Debug images: {'enabled' if stage1_config.save_debug_images else 'disabled'}")
            print()
        
        # Create and run Stage 1 pipeline
        pipeline = TwoStageOCRPipeline(stage1_config=stage1_config)
        
        # Handle single file vs directory
        if input_path.is_file():
            # For single file, temporarily move it to input directory structure
            temp_input_dir = stage1_config.output_dir / "temp_input"
            temp_input_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy file to temp directory (or create symlink)
            import shutil
            temp_file = temp_input_dir / input_path.name
            shutil.copy2(input_path, temp_file)
            
            try:
                results = pipeline.run_stage1(temp_input_dir)
            finally:
                # Clean up temp directory
                shutil.rmtree(temp_input_dir, ignore_errors=True)
        else:
            results = pipeline.run_stage1(input_path)
        
        # Print results
        if args.verbose:
            print(f"\n*** STAGE 1 COMPLETED SUCCESSFULLY! ***")
            print(f"Generated {len(results)} cropped table images")
            print(f"Results saved to: {stage1_config.output_dir}")
            print(f"Cropped tables ready for Stage 2: {stage1_config.output_dir / '05_cropped_tables'}")
            print()
            print("Next steps:")
            print(f"   python run_stage2.py  # Run Stage 2 refinement")
            print(f"   python run_complete.py {args.input}  # Run both stages")
        else:
            print(f"Stage 1 complete: {len(results)} cropped tables -> {stage1_config.output_dir / '05_cropped_tables'}")
        
        return True
        
    except KeyboardInterrupt:
        print("\nStage 1 interrupted by user")
        return False
    except Exception as e:
        print(f"Stage 1 failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)