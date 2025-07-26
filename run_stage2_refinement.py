#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2: Refinement Pipeline

Refines the cropped table images from Stage 1 through additional processing steps
to achieve higher precision and better final table extraction results.

Usage:
    python run_stage2_refinement.py [config_file]

Default config: configs/stage2_config.json
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


class Stage2Pipeline:
    """
    Stage 2 Refinement Pipeline
    
    Orchestrates refined processing on cropped table images to produce
    publication-ready table structures with high precision.
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
        # Check that Stage 1 output exists
        stage1_output = self.config.get_directory('splited_images')  # This points to Stage 1 cropped tables
        
        if not os.path.isdir(stage1_output):
            self.logger.error(f"Stage 1 output directory not found: {stage1_output}")
            self.logger.error("Please run Stage 1 first: python run_stage1_initial_processing.py")
            return False
        
        # Check for input images from Stage 1
        from src.utils.image_utils import get_image_files
        image_files = get_image_files(stage1_output)
        if not image_files:
            self.logger.error(f"No cropped table images found in {stage1_output}")
            self.logger.error("Stage 1 must be completed successfully before running Stage 2")
            return False
        
        self.logger.info(f"Found {len(image_files)} cropped table images from Stage 1")
        
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
            
            # Map module names to current script files (Stage 2 specific)
            script_mapping = {
                'deskewing': 'deskew.py',
                'line_detection': 'detect_curved_lines.py',
                'table_reconstruction': 'reconstruct_table_full.py',
                'table_fitting': 'reconstruct_table_fit.py'
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
        Execute the complete Stage 2 pipeline.
        
        Returns:
            True if all steps completed successfully
        """
        self.start_time = time.time()
        
        self.logger.info("🔄 STARTING STAGE 2: REFINEMENT PIPELINE")
        self.logger.info(f"Configuration: {self.config_path}")
        self.logger.info(f"Input: {self.config.get_directory('splited_images')} (Stage 1 cropped tables)")
        self.logger.info(f"Output: {self.config.get_directory('table_fit_images')} (Final fitted tables)")
        
        # Validate setup
        if not self.validate_setup():
            self.logger.error("❌ Pipeline validation failed")
            return False
        
        # Define refinement steps
        steps = [
            ("refine1", "deskewing", "Re-deskewing Cropped Tables - Fine-tune rotation"),
            ("refine2", "line_detection", "Refined Line Detection - Optimized for table content"),
            ("refine3", "table_reconstruction", "Final Table Reconstruction - Precise grid creation"),
            ("refine4", "table_fitting", "Table Fitting - Publication-ready cell structure")
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
        self.logger.info("🎯 STAGE 2 PIPELINE COMPLETED SUCCESSFULLY!")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total refinement time: {total_time:.1f} seconds")
        
        # Show step timing breakdown
        self.logger.info("\nRefinement step timing:")
        for step_name, step_time in self.step_times.items():
            percentage = (step_time / total_time) * 100
            self.logger.info(f"  {step_name}: {step_time:.1f}s ({percentage:.1f}%)")
        
        # Show output directories
        self.logger.info("\nRefinement output directories:")
        output_dirs = [
            ("Re-deskewed tables", "output/stage2_refinement/01_deskewed/"),
            ("Refined line detection", "output/stage2_refinement/02_line_detection/"),
            ("Final table reconstruction", "output/stage2_refinement/03_table_reconstruction/"),
            ("Publication-ready tables", "output/stage2_refinement/04_fitted_tables/")
        ]
        
        for desc, path in output_dirs:
            file_count = len(os.listdir(path)) if os.path.exists(path) else 0
            self.logger.info(f"  📁 {desc}: {path} ({file_count} files)")
        
        # Show final results summary
        final_output = "output/stage2_refinement/04_fitted_tables/"
        if os.path.exists(final_output):
            final_count = len([f for f in os.listdir(final_output) if f.endswith('.jpg')])
            self.logger.info(f"\n🎉 PROCESSING COMPLETE!")
            self.logger.info(f"   📊 {final_count} publication-ready table images generated")
            self.logger.info(f"   📁 Check results in: {final_output}")
            self.logger.info(f"   📄 Cell data available in corresponding JSON files")
        
        return True


def main():
    """Main entry point for Stage 2 pipeline."""
    
    # Determine config file
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "configs/stage2_config.json"
    
    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"❌ Configuration file not found: {config_file}")
        print(f"Usage: python run_stage2_refinement.py [config_file]")
        print(f"Default config: configs/stage2_config.json")
        return False
    
    try:
        # Create and run pipeline
        pipeline = Stage2Pipeline(config_file)
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