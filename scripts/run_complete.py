#!/usr/bin/env python3
"""
Complete Two-Stage Pipeline CLI

Runs both Stage 1 and Stage 2 of the OCR pipeline sequentially.
This is the most convenient way to process raw scanned images
all the way to publication-ready table extractions.

Usage:
    python run_complete.py [input_dir] [options]

Example:
    python run_complete.py input/                     # Process input/ directory
    python run_complete.py image.jpg                  # Process single image
    python run_complete.py input/ --verbose --debug   # Full pipeline with detailed output
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr_pipeline.pipeline import TwoStageOCRPipeline
from src.ocr_pipeline.config import Stage1Config, Stage2Config


def main():
    """Main entry point for complete two-stage processing."""
    parser = argparse.ArgumentParser(
        description="OCR Complete Pipeline: Stage 1 + Stage 2 Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_complete.py input/                       # Process all images in input/
  python run_complete.py image.jpg                    # Process single image
  python run_complete.py input/ -o custom_output/     # Custom output base directory
  python run_complete.py input/ --verbose --debug     # Full verbose output with debug images
  python run_complete.py scans/ --stage1-only         # Run only Stage 1
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
        default="data/output",
        help="Base output directory (stages will create subdirectories) (default: data/output/)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output for both stages"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save debug images during processing"
    )
    
    parser.add_argument(
        "--stage1-only",
        action="store_true",
        help="Run only Stage 1 (initial processing)"
    )
    
    parser.add_argument(
        "--stage2-only",
        action="store_true",
        help="Run only Stage 2 (refinement) - requires Stage 1 output"
    )
    
    # Stage 1 specific options
    stage1_group = parser.add_argument_group('Stage 1 Options')
    stage1_group.add_argument(
        "--s1-angle-range",
        type=int,
        default=10,
        help="Stage 1 deskewing angle range (default: 10)"
    )
    stage1_group.add_argument(
        "--s1-min-line-length",
        type=int,
        default=40,
        help="Stage 1 minimum line length (default: 40)"
    )
    stage1_group.add_argument(
        "--disable-roi",
        action="store_true",
        help="Disable ROI detection in Stage 1"
    )
    
    # Stage 2 specific options
    stage2_group = parser.add_argument_group('Stage 2 Options')
    stage2_group.add_argument(
        "--s2-angle-range",
        type=int,
        default=10,
        help="Stage 2 refinement angle range (default: 10)"
    )
    stage2_group.add_argument(
        "--s2-min-line-length",
        type=int,
        default=30,
        help="Stage 2 minimum line length (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.stage1_only and args.stage2_only:
        print("Error: Cannot specify both --stage1-only and --stage2-only")
        sys.exit(1)
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Create output base directory
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Create stage configurations
    stage1_config = Stage1Config(
        input_dir=input_path if input_path.is_dir() else input_path.parent,
        output_dir=output_base / "stage1_initial_processing",
        verbose=args.verbose,
        save_debug_images=args.debug,
        angle_range=args.s1_angle_range,
        min_line_length=args.s1_min_line_length,
        enable_roi_detection=not args.disable_roi
    )
    
    stage2_config = Stage2Config(
        input_dir=output_base / "stage1_initial_processing" / "05_cropped_tables",
        output_dir=output_base / "stage2_refinement",
        verbose=args.verbose,
        save_debug_images=args.debug,
        angle_range=args.s2_angle_range,
        min_line_length=args.s2_min_line_length
    )
    
    try:
        start_time = time.time()
        
        if args.verbose:
            print("🏃 COMPLETE TWO-STAGE OCR PIPELINE")
            print("=" * 80)
            print(f"📂 Input: {input_path}")
            print(f"📁 Output base: {output_base}")
            
            if not args.stage2_only:
                print(f"🚀 Stage 1 output: {stage1_config.output_dir}")
            if not args.stage1_only:
                print(f"🔄 Stage 2 output: {stage2_config.output_dir}")
            print()
        
        # Create pipeline
        pipeline = TwoStageOCRPipeline(stage1_config, stage2_config)
        
        if args.stage2_only:
            # Run only Stage 2
            if args.verbose:
                print("⏭️  Skipping Stage 1 (--stage2-only specified)")
            
            results = pipeline.run_stage2()
            
        elif args.stage1_only:
            # Run only Stage 1
            results = pipeline.run_stage1(input_path if input_path.is_dir() else None)
            
            if args.verbose:
                print("⏸️  Stopping after Stage 1 (--stage1-only specified)")
                
        else:
            # Run complete pipeline
            results = pipeline.run_complete_pipeline(input_path if input_path.is_dir() else None)
        
        # Calculate timing
        total_time = time.time() - start_time
        
        # Print final results
        if args.verbose:
            print(f"\n🎊 PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"⏱️  Total processing time: {total_time:.1f} seconds")
            print(f"📊 Final output: {len(results)} processed images")
            
            if not args.stage1_only:
                print(f"📁 Publication-ready tables: {stage2_config.output_dir / '04_fitted_tables'}")
            else:
                print(f"📁 Cropped tables ready for Stage 2: {stage1_config.output_dir / '05_cropped_tables'}")
                print(f"🔄 Next: python run_stage2.py  # Run refinement stage")
            
            print(f"\n📋 Processing summary:")
            if not args.stage2_only:
                print(f"   ✅ Stage 1: Initial processing and table cropping")
            if not args.stage1_only:
                print(f"   ✅ Stage 2: Refinement and publication preparation")
                
        else:
            if args.stage1_only:
                print(f"Stage 1 complete: {len(results)} cropped tables → {stage1_config.output_dir / '05_cropped_tables'}")
            elif args.stage2_only:
                print(f"Stage 2 complete: {len(results)} refined tables → {stage2_config.output_dir / '04_fitted_tables'}")
            else:
                print(f"Complete pipeline: {len(results)} refined tables → {stage2_config.output_dir / '04_fitted_tables'}")
        
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