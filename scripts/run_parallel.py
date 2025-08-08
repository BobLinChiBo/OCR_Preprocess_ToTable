#!/usr/bin/env python3
"""Run parallel batch processing on Windows.

This script provides proper multiprocessing support for Windows by using
the if __name__ == '__main__' guard required by Windows.

Usage:
    python run_parallel.py [input_dir] [options]

Example:
    python run_parallel.py data/input/raw_images/ --workers 4
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Main entry point for parallel processing."""
    parser = argparse.ArgumentParser(
        description="OCR Parallel Batch Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input directory with images (default: from config)",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory base path",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count - 1)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of images to process per batch (default: 4)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--memory-mode", action="store_true", help="Use memory-efficient mode"
    )

    args = parser.parse_args()

    # Import after parsing args to avoid issues
    from src.ocr_pipeline.config import get_stage1_config, get_stage2_config
    from src.ocr_pipeline.pipeline import TwoStageOCRPipeline
    from src.ocr_pipeline.processors import get_image_files

    # Create configurations
    stage1_config = get_stage1_config()
    stage2_config = get_stage2_config()

    # Override with command line arguments
    if args.input:
        stage1_config.input_dir = Path(args.input)
    
    if args.output:
        base_output = Path(args.output)
        stage1_config.output_dir = base_output / "stage1"
        stage2_config.output_dir = base_output / "stage2"
    
    if args.workers:
        stage1_config.max_workers = args.workers
    
    stage1_config.batch_size = args.batch_size
    stage1_config.verbose = args.verbose
    stage2_config.verbose = args.verbose
    
    # Enable parallel processing
    stage1_config.parallel_processing = True
    stage2_config.parallel_processing = True
    
    # Memory mode settings
    if args.memory_mode:
        stage1_config.memory_mode = True
        stage2_config.memory_mode = True

    # Check input
    if not stage1_config.input_dir.exists():
        print(f"Error: Input directory not found: {stage1_config.input_dir}")
        return 1

    # Get images
    image_files = get_image_files(stage1_config.input_dir)
    if not image_files:
        print(f"No images found in {stage1_config.input_dir}")
        return 1

    print(f"Found {len(image_files)} images to process")
    print(f"Workers: {stage1_config.max_workers or 'auto'}")
    print(f"Batch size: {stage1_config.batch_size}")
    print(f"Memory mode: {args.memory_mode}")
    print()

    # Create pipeline
    pipeline = TwoStageOCRPipeline(stage1_config, stage2_config)

    try:
        # Run parallel batch processing
        results = pipeline.run_batch_optimized(
            stage1_config.input_dir,
            use_parallel=True,
            use_memory_mode=args.memory_mode
        )
        
        print(f"\nProcessing complete: {len(results)} output files generated")
        return 0
        
    except Exception as e:
        print(f"Error during parallel processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    # This guard is required for multiprocessing on Windows
    from multiprocessing import freeze_support
    freeze_support()
    
    sys.exit(main())