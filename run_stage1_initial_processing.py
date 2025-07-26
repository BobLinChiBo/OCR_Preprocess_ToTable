#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1: Initial Processing Pipeline

Processes raw scanned images through the complete initial processing workflow
using the reorganized modular architecture.

Usage:
    python run_stage1_initial_processing.py [config_file]

Default config: configs/stage1_config.json
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config_loader import load_config, setup_logging
from src.utils.file_utils import ensure_directory_exists


class Stage1Pipeline:
    """
    Stage 1 Initial Processing Pipeline
    
    Orchestrates the complete initial processing workflow from raw scanned
    images to cropped table regions ready for Stage 2 refinement.
    """
    
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration."""
        self.config_path = config_path
        self.config = load_config(config_path)
        setup_logging(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Processing statistics
        self.start_time = None
        self.step_times = {}
    
    def validate_setup(self) -> bool:
        """Validate that all requirements are met before processing."""
        # Check input directory
        input_dir = self.config.get_directory('raw_images')
        if not os.path.isdir(input_dir):
            self.logger.error(f"Input directory not found: {input_dir}")
            return False
        
        # Check for input images
        from src.utils.image_utils import get_image_files
        image_files = get_image_files(input_dir)
        if not image_files:
            self.logger.error(f"No image files found in {input_dir}")
            return False
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        # Ensure all output directories exist  
        try:
            self.config.create_output_directories()
            return True
        except Exception as e:
            self.logger.error(f"Failed to create output directories: {e}")
            return False
    
    def run_step(self, step_name: str, module_name: str, description: str) -> bool:
        """
        Run a single processing step.
        
        Args:
            step_name: Name of the step for logging
            module_name: Name of the processing module
            description: Human-readable description
            
        Returns:
            True if step completed successfully
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"STEP: {description}")
        self.logger.info(f"{'='*60}")
        
        step_start = time.time()
        
        try:
            # For now, use the legacy scripts with updated config paths
            # In a full implementation, these would be replaced with the new modular imports
            import subprocess
            
            # Map module names to current script files
            script_mapping = {
                'page_splitting': 'split_pages.py',
                'deskewing': 'deskew.py', 
                'edge_detection': 'detect_edges.py',
                'line_detection': 'detect_curved_lines.py',
                'table_reconstruction': 'reconstruct_table_full.py',
                'table_cropping': 'crop_to_table.py'
            }
            
            script_name = script_mapping.get(module_name, f"{module_name}.py")
            
            # Run the processing step
            cmd = [sys.executable, script_name, self.config_path]
            self.logger.info(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if result.stdout:
                # Log output at debug level to avoid clutter
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        self.logger.debug(f"[{module_name}] {line}")
            
            step_time = time.time() - step_start
            self.step_times[step_name] = step_time
            
            self.logger.info(f"✓ {description} completed successfully ({step_time:.1f}s)")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"✗ {description} failed:")
            if e.stdout:
                self.logger.error(f"STDOUT: {e.stdout}")
            if e.stderr:
                self.logger.error(f"STDERR: {e.stderr}")
            
            return False
        except Exception as e:
            self.logger.error(f"✗ {description} failed with exception: {e}")
            return False
    
    def run_pipeline(self) -> bool:
        """
        Execute the complete Stage 1 pipeline.
        
        Returns:
            True if all steps completed successfully
        """
        self.start_time = time.time()
        
        self.logger.info("🚀 STARTING STAGE 1: INITIAL PROCESSING PIPELINE")
        self.logger.info(f"Configuration: {self.config_path}")
        self.logger.info(f"Input directory: {self.config.get_directory('raw_images')}")
        self.logger.info(f"Output directory: {self.config.get_directory('splited_images')}")
        
        # Validate setup
        if not self.validate_setup():
            self.logger.error("❌ Pipeline validation failed")
            return False
        
        # Define processing steps
        steps = [
            ("step1", "page_splitting", "Page Splitting - Separate double-page scans"),
            ("step2", "deskewing", "Deskewing - Correct image rotation"),
            ("step3", "edge_detection", "Edge Detection - Detect content boundaries"),
            ("step4", "line_detection", "Line Detection - Find table lines with curve support"),
            ("step5", "table_reconstruction", "Table Reconstruction - Build complete table grids"),
            ("step6", "table_cropping", "Table Cropping - Extract table regions for refinement")
        ]
        
        # Execute each step
        failed_steps = []
        for step_id, module, description in steps:
            if not self.run_step(step_id, module, description):
                failed_steps.append(description)
                self.logger.error(f"❌ Pipeline failed at: {description}")
                return False
        
        # Pipeline completed successfully
        total_time = time.time() - self.start_time
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("🎉 STAGE 1 PIPELINE COMPLETED SUCCESSFULLY!")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total processing time: {total_time:.1f} seconds")
        
        # Show step timing breakdown
        self.logger.info("\nStep timing breakdown:")
        for step_name, step_time in self.step_times.items():
            percentage = (step_time / total_time) * 100
            self.logger.info(f"  {step_name}: {step_time:.1f}s ({percentage:.1f}%)")
        
        # Show output directories
        self.logger.info("\nOutput directories created:")
        output_dirs = [
            ("Split pages", "output/stage1_initial_processing/01_split_pages/"),
            ("Deskewed images", "output/stage1_initial_processing/02_deskewed/"),
            ("Edge detection", "output/stage1_initial_processing/02.5_edge_detection/"),
            ("Line detection", "output/stage1_initial_processing/03_line_detection/"),
            ("Table reconstruction", "output/stage1_initial_processing/04_table_reconstruction/"),
            ("Cropped tables", "output/stage1_initial_processing/05_cropped_tables/")
        ]
        
        for desc, path in output_dirs:
            file_count = len(os.listdir(path)) if os.path.exists(path) else 0
            self.logger.info(f"  📁 {desc}: {path} ({file_count} files)")
        
        self.logger.info(f"\n🔄 Ready for Stage 2 refinement!")
        self.logger.info(f"   Run: python run_stage2_refinement.py")
        
        return True


def main():
    """Main entry point for Stage 1 pipeline."""
    
    # Determine config file
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "configs/stage1_config.json"
    
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"❌ Configuration file not found: {config_file}")
        print(f"Usage: python run_stage1_initial_processing.py [config_file]")
        print(f"Default config: configs/stage1_config.json")
        return False
    
    try:
        # Create and run pipeline
        pipeline = Stage1Pipeline(config_file)
        success = pipeline.run_pipeline()
        
        return success
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Pipeline failed with exception: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)