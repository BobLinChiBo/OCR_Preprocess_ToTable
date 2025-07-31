#!/usr/bin/env python3
"""
Complete Two-Stage OCR Pipeline CLI

Runs the complete two-stage OCR pipeline, processing raw scanned images 
through both initial processing (Stage 1) and refinement (Stage 2) stages.

Usage:
    python run_complete.py [input_dir] [options]

Example:
    python run_complete.py input/ --verbose
    python run_complete.py image.jpg --verbose
    python run_complete.py input/ -o output/ --debug
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr_pipeline.pipeline import TwoStageOCRPipeline
from src.ocr_pipeline.config import Stage1Config, Stage2Config, get_stage1_config, get_stage2_config


def main():
    """Main entry point for complete two-stage OCR processing."""
    parser = argparse.ArgumentParser(
        description="Complete Two-Stage OCR Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_complete.py input/                    # Process input/ directory through both stages
  python run_complete.py image.jpg                 # Process single image through both stages
  python run_complete.py input/ -o output/         # Custom output directory
  python run_complete.py input/ --verbose --debug  # Verbose output with debug images
  python run_complete.py input/ --stage1-only      # Run only Stage 1
  python run_complete.py --stage2-only             # Run only Stage 2 (requires Stage 1 output)
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
        help="Base output directory (default: data/output/)"
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
        "--stage1-only",
        action="store_true",
        help="Run only Stage 1 (initial processing)"
    )
    
    parser.add_argument(
        "--stage2-only",
        action="store_true",
        help="Run only Stage 2 (refinement). Requires Stage 1 output."
    )
    
    # Stage 1 specific arguments
    parser.add_argument(
        "--s1-angle-range",
        type=int,
        default=10,
        help="Stage 1 deskewing angle range in degrees (default: 10)"
    )
    
    parser.add_argument(
        "--s1-angle-step",
        type=float,
        default=0.2,
        help="Stage 1 deskewing angle step in degrees (default: 0.2)"
    )
    
    parser.add_argument(
        "--s1-min-line-length",
        type=int,
        default=40,
        help="Stage 1 minimum line length for table detection (default: 40)"
    )
    
    parser.add_argument(
        "--disable-roi",
        action="store_true",
        help="Disable ROI detection preprocessing in Stage 1"
    )
    
    # Stage 2 specific arguments
    parser.add_argument(
        "--s2-angle-range",
        type=int,
        default=10,
        help="Stage 2 refinement angle range in degrees (default: 10)"
    )
    
    parser.add_argument(
        "--s2-angle-step",
        type=float,
        default=0.2,
        help="Stage 2 refinement angle step in degrees (default: 0.2)"
    )
    
    parser.add_argument(
        "--s2-min-line-length",
        type=int,
        default=30,
        help="Stage 2 minimum line length for refined detection (default: 30)"
    )
    
    parser.add_argument(
        "--s2-max-line-gap",
        type=int,
        default=5,
        help="Stage 2 maximum line gap for refined detection (default: 5)"
    )
    
    # Configuration files
    parser.add_argument(
        "--stage1-config",
        type=Path,
        help="Path to Stage 1 JSON configuration file (default: configs/stage1_default.json)"
    )
    
    parser.add_argument(
        "--stage2-config",
        type=Path,
        help="Path to Stage 2 JSON configuration file (default: configs/stage2_default.json)"
    )
    
    args = parser.parse_args()
    
    # Validate stage selection
    if args.stage1_only and args.stage2_only:
        print("❌ Error: Cannot specify both --stage1-only and --stage2-only")
        sys.exit(1)
    
    # Validate input for Stage 1 or complete pipeline
    if not args.stage2_only:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"❌ Error: Input path does not exist: {input_path}")
            sys.exit(1)
    
    # Create configurations
    base_output = Path(args.output)
    
    # Stage 1 configuration
    stage1_config = get_stage1_config(args.stage1_config)
    if not args.stage2_only:
        stage1_config.input_dir = Path(args.input) if Path(args.input).is_dir() else Path(args.input).parent
        stage1_config.output_dir = base_output / "stage1_initial_processing"
        stage1_config.verbose = args.verbose
        stage1_config.save_debug_images = args.debug
        stage1_config.angle_range = args.s1_angle_range
        stage1_config.angle_step = args.s1_angle_step
        stage1_config.min_line_length = args.s1_min_line_length
        stage1_config.enable_roi_detection = not args.disable_roi
    
    # Stage 2 configuration
    stage2_config = get_stage2_config(args.stage2_config)
    if not args.stage1_only:
        stage2_config.input_dir = stage1_config.output_dir / "05_cropped_tables" if not args.stage2_only else Path("data/output/stage1_initial_processing/05_cropped_tables")
        stage2_config.output_dir = base_output / "stage2_refinement"
        stage2_config.verbose = args.verbose
        stage2_config.save_debug_images = args.debug
        stage2_config.angle_range = args.s2_angle_range
        stage2_config.angle_step = args.s2_angle_step
        stage2_config.min_line_length = args.s2_min_line_length
        stage2_config.max_line_gap = args.s2_max_line_gap
    
    try:
        if args.verbose:
            print("*** COMPLETE TWO-STAGE OCR PIPELINE ***")
            print("=" * 70)
            if not args.stage2_only:
                print(f"Input: {args.input}")
            print(f"Output Base: {base_output}")
            
            if args.stage1_only:
                print("Mode: Stage 1 Only (Initial Processing)")
            elif args.stage2_only:
                print("Mode: Stage 2 Only (Refinement)")
            else:
                print("Mode: Complete Two-Stage Pipeline")
            print()
        
        # Create pipeline
        pipeline = TwoStageOCRPipeline(
            stage1_config=stage1_config if not args.stage2_only else None,
            stage2_config=stage2_config if not args.stage1_only else None
        )
        
        # Execute pipeline based on mode
        if args.stage1_only:
            # Run only Stage 1
            input_dir = Path(args.input) if Path(args.input).is_dir() else None
            if Path(args.input).is_file():
                # Handle single file for Stage 1
                temp_input_dir = stage1_config.output_dir / "temp_input"
                temp_input_dir.mkdir(parents=True, exist_ok=True)
                
                import shutil
                temp_file = temp_input_dir / Path(args.input).name
                shutil.copy2(Path(args.input), temp_file)
                
                try:
                    results = pipeline.run_stage1(temp_input_dir)
                finally:
                    shutil.rmtree(temp_input_dir, ignore_errors=True)
            else:
                results = pipeline.run_stage1(input_dir)
            
            if args.verbose:
                print(f"\n*** STAGE 1 COMPLETED! ***")
                print(f"Generated {len(results)} cropped table images")
                print(f"Results: {stage1_config.output_dir / '05_cropped_tables'}")
                print("\nNext step: python run_stage2.py --verbose")
        
        elif args.stage2_only:
            # Run only Stage 2
            if not stage2_config.input_dir.exists():
                print(f"❌ Error: Stage 1 output not found: {stage2_config.input_dir}")
                print("Hint: Run Stage 1 first or use complete pipeline")
                sys.exit(1)
            
            results = pipeline.run_stage2()
            
            if args.verbose:
                print(f"\n*** STAGE 2 COMPLETED! ***")
                print(f"Generated {len(results)} publication-ready table images")
                print(f"Final Results: {stage2_config.output_dir / '04_fitted_tables'}")
        
        else:
            # Run complete two-stage pipeline
            input_dir = Path(args.input) if Path(args.input).is_dir() else None
            if Path(args.input).is_file():
                # Handle single file for complete pipeline
                temp_input_dir = stage1_config.output_dir / "temp_input"
                temp_input_dir.mkdir(parents=True, exist_ok=True)
                
                import shutil
                temp_file = temp_input_dir / Path(args.input).name
                shutil.copy2(Path(args.input), temp_file)
                
                try:
                    results = pipeline.run_complete_pipeline(temp_input_dir)
                finally:
                    shutil.rmtree(temp_input_dir, ignore_errors=True)
            else:
                results = pipeline.run_complete_pipeline(input_dir)
            
            if args.verbose:
                print(f"\n*** COMPLETE PIPELINE FINISHED! ***")
                print(f"Generated {len(results)} publication-ready table images")
                print(f"Stage 1 Output: {stage1_config.output_dir}")
                print(f"Final Results: {stage2_config.output_dir / '04_fitted_tables'}")
                print("\nYour tables are ready for use in publications, reports, or further analysis!")
        
        return True
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        return False
    except Exception as e:
        print(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)