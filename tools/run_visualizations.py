#!/usr/bin/env python3
"""
Master Visualization Runner

A unified script to run multiple visualization scripts and manage their outputs.
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from visualization.output_manager import OutputManager


class VisualizationRunner:
    """Manages running multiple visualization scripts."""
    
    def __init__(self):
        self.manager = OutputManager()
        self.available_scripts = {
            'deskew': {
                'script': 'visualize_deskew.py',
                'description': 'Deskewing analysis and visualization',
                'default_args': ['--angle-range', '45', '--angle-step', '0.5']
            },
            'page-split': {
                'script': 'visualize_page_split.py',
                'description': 'Page splitting visualization',
                'default_args': ['--gutter-start', '0.4', '--gutter-end', '0.6']
            },
            'roi': {
                'script': 'visualize_roi.py',
                'description': 'Region of interest detection',
                'default_args': ['--gabor-threshold', '127']
            },
            'table-lines': {
                'script': 'visualize_table_lines.py',
                'description': 'Table line detection visualization',
                'default_args': ['--min-line-length', '100']
            },
            'table-crop': {
                'script': 'visualize_table_crop.py',
                'description': 'Table cropping visualization',
                'default_args': ['--crop-padding', '10']
            },
            'pipeline': {
                'script': 'visualize_pipeline.py',
                'description': 'Complete pipeline visualization',
                'default_args': ['--save-intermediates']
            }
        }
    
    def run_single_script(self, script_name: str, images: List[str], 
                         extra_args: List[str] = None, use_test_images: bool = False) -> Dict[str, Any]:
        """Run a single visualization script."""
        if script_name not in self.available_scripts:
            raise ValueError(f"Unknown script: {script_name}")
        
        script_info = self.available_scripts[script_name]
        script_path = script_dir / script_info['script']
        
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        # Build command
        cmd = [sys.executable, str(script_path)]
        
        # Add --test-images flag if requested (don't add individual images if using test images)
        if use_test_images:
            cmd.append('--test-images')
        else:
            cmd.extend(images)
        
        cmd.extend(script_info['default_args'])
        
        if extra_args:
            cmd.extend(extra_args)
        
        print(f"\n{'='*60}")
        print(f"Running {script_name}: {script_info['description']}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        try:
            # Run the script
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            success = result.returncode == 0
            
            if success:
                print(f"SUCCESS: {script_name} completed successfully in {duration:.1f}s")
            else:
                print(f"FAILED: {script_name} failed after {duration:.1f}s")
                print(f"Error: {result.stderr}")
            
            return {
                'script': script_name,
                'success': success,
                'duration': duration,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"CRASHED: {script_name} crashed after {duration:.1f}s: {e}")
            
            return {
                'script': script_name,
                'success': False,
                'duration': duration,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'error': str(e),
                'returncode': -1
            }
    
    def run_multiple_scripts(self, script_names: List[str], images: List[str],
                           args_per_script: Dict[str, List[str]] = None, use_test_images: bool = False) -> List[Dict[str, Any]]:
        """Run multiple visualization scripts."""
        results = []
        
        if use_test_images:
            print(f"\nRunning {len(script_names)} visualization scripts on all test images")
            print(f"Using batch mode: processing images from input/test_images/")
        else:
            print(f"\nRunning {len(script_names)} visualization scripts on {len(images)} images")
        print(f"Scripts: {', '.join(script_names)}")
        
        total_start = datetime.now()
        
        for script_name in script_names:
            extra_args = args_per_script.get(script_name, []) if args_per_script else []
            result = self.run_single_script(script_name, images, extra_args, use_test_images)
            results.append(result)
            
            # Print status
            if result['success']:
                print(f"  SUCCESS: {script_name}")
            else:
                print(f"  FAILED: {script_name}")
        
        total_end = datetime.now()
        total_duration = (total_end - total_start).total_seconds()
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        print(f"\n{'='*60}")
        print(f"BATCH RUN SUMMARY")
        print(f"{'='*60}")
        print(f"Completed: {successful}/{len(results)} scripts")
        print(f"Total time: {total_duration:.1f}s")
        print(f"Average time per script: {total_duration/len(results):.1f}s")
        
        if successful < len(results):
            failed_scripts = [r['script'] for r in results if not r['success']]
            print(f"Failed scripts: {', '.join(failed_scripts)}")
        
        return results
    
    def list_available_scripts(self):
        """List all available visualization scripts."""
        print(f"\n{'='*60}")
        print("AVAILABLE VISUALIZATION SCRIPTS")
        print(f"{'='*60}")
        
        for name, info in self.available_scripts.items():
            print(f"{name:<15} - {info['description']}")
            print(f"{'':15}   Script: {info['script']}")
            print(f"{'':15}   Args: {' '.join(info['default_args'])}")
            print()


def parse_script_args(script_args: List[str]) -> Dict[str, List[str]]:
    """Parse per-script arguments from command line."""
    result = {}
    current_script = None
    
    for arg in script_args:
        if arg.startswith('--') and arg.endswith('-args'):
            # Extract script name (e.g., --deskew-args -> deskew)
            current_script = arg[2:-5]  # Remove -- and -args
            result[current_script] = []
        elif current_script:
            result[current_script].append(arg)
        else:
            # Global argument - ignore for now
            pass
    
    return result


def main():
    """Main function for the visualization runner."""
    parser = argparse.ArgumentParser(
        description="Run OCR visualization scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run deskew visualization
  python run_visualizations.py deskew image1.jpg image2.jpg

  # Run multiple scripts
  python run_visualizations.py deskew page-split roi image.jpg

  # Run with custom arguments per script
  python run_visualizations.py deskew page-split image.jpg \\
    --deskew-args --angle-range 30 --angle-step 1.0 \\
    --page-split-args --gutter-start 0.3 --gutter-end 0.7

  # Run all available scripts
  python run_visualizations.py all image.jpg

  # Process all images in test_images directory (batch mode)
  python run_visualizations.py deskew --test-images
  python run_visualizations.py all --test-images

  # List available scripts
  python run_visualizations.py --list
        """
    )
    
    parser.add_argument('scripts', nargs='*', 
                       help='Script names to run (or "all" for all scripts)')
    parser.add_argument('images', nargs='*',
                       help='Image files to process')
    
    parser.add_argument('--list', action='store_true',
                       help='List available scripts and exit')
    parser.add_argument('--test-images', action='store_true',
                       help='Process all images in input/test_images directory (passes --test-images to all scripts)')
    parser.add_argument('--save-report', action='store_true',
                       help='Save detailed execution report')
    
    # Per-script arguments (parsed manually)
    parser.add_argument('--deskew-args', nargs='*', dest='_ignore1', 
                       help='Arguments for deskew script')
    parser.add_argument('--page-split-args', nargs='*', dest='_ignore2',
                       help='Arguments for page-split script')
    parser.add_argument('--roi-args', nargs='*', dest='_ignore3',
                       help='Arguments for roi script')
    parser.add_argument('--table-lines-args', nargs='*', dest='_ignore4',
                       help='Arguments for table-lines script')
    parser.add_argument('--table-crop-args', nargs='*', dest='_ignore5',
                       help='Arguments for table-crop script')
    parser.add_argument('--pipeline-args', nargs='*', dest='_ignore6',
                       help='Arguments for pipeline script')
    
    # Parse known args to handle script-specific arguments
    args, remaining = parser.parse_known_args()
    
    runner = VisualizationRunner()
    
    if args.list:
        runner.list_available_scripts()
        return
    
    # Parse script-specific arguments from remaining args
    script_args = parse_script_args(remaining)
    
    # Determine which scripts to run
    script_names = args.scripts if args.scripts else []
    
    if 'all' in script_names:
        script_names = list(runner.available_scripts.keys())
    
    if not script_names:
        print("No scripts specified. Use --list to see available scripts.")
        parser.print_help()
        return
    
    # Validate script names
    invalid_scripts = [s for s in script_names if s not in runner.available_scripts]
    if invalid_scripts:
        print(f"Invalid script names: {', '.join(invalid_scripts)}")
        print("Use --list to see available scripts.")
        return
    
    # Handle images vs test-images mode
    if args.test_images:
        # In test-images mode, we don't need to validate individual images
        valid_images = []  # Will be ignored anyway
        print("Using --test-images mode: Individual image validation skipped")
    else:
        # Get images (from args.images or remaining args)
        images = args.images if args.images else []
        if not images:
            # Try to find images in remaining args (after script args)
            potential_images = [arg for arg in remaining if not arg.startswith('--') 
                              and Path(arg).suffix.lower() in ['.jpg', '.png', '.jpeg']]
            images = potential_images
        
        if not images:
            print("No images specified.")
            return
        
        # Validate image paths
        valid_images = []
        for img in images:
            img_path = Path(img)
            if img_path.exists():
                valid_images.append(img)
            else:
                print(f"Warning: Image not found: {img}")
        
        if not valid_images:
            print("No valid images found.")
            return
    
    if args.test_images:
        print(f"Processing test images with {len(script_names)} scripts")
    else:
        print(f"Processing {len(valid_images)} images with {len(script_names)} scripts")
    
    # Run the scripts
    if len(script_names) == 1:
        result = runner.run_single_script(script_names[0], valid_images, 
                                        script_args.get(script_names[0], []), args.test_images)
        results = [result]
    else:
        results = runner.run_multiple_scripts(script_names, valid_images, script_args, args.test_images)
    
    # Save report if requested
    if args.save_report:
        report_dir = runner.manager.base_dir / "reports"
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"visualization_run_{timestamp}.json"
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'scripts_run': script_names,
            'images_processed': valid_images,
            'script_args': script_args,
            'results': results
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\nExecution report saved to: {report_file}")
    
    print(f"\nUse 'python visualization/check_results.py list' to view all results")


if __name__ == "__main__":
    main()