#!/usr/bin/env python3
"""
Master Visualization Runner

A unified script to run multiple visualization scripts and manage their outputs.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from output_manager import OutputManager  # noqa: E402


class VisualizationRunner:
    """Manages running multiple visualization scripts."""

    def __init__(self, use_v2=True):
        self.manager = OutputManager()
        self.use_v2 = use_v2
        
        # V2 scripts are now the only supported version
        self.available_scripts = {
            "page-split": {
                "script": "visualize_page_split_v2.py",
                "description": "Page splitting visualization",
                "default_args": [],
            },
            "margin-removal": {
                "script": "visualize_margin_removal_v2.py",
                "description": "Margin removal visualization (supports multiple methods)",
                "default_args": ["--save-debug"],
            },
            "deskew": {
                "script": "visualize_deskew_v2.py",
                "description": "Deskewing analysis and visualization",
                "default_args": [],
            },
            "table-lines": {
                "script": "visualize_table_lines_v2.py",
                "description": "Table line detection visualization",
                "default_args": ["--save-debug", "--show-filtering-steps"],
            },
            "table-structure": {
                "script": "visualize_table_structure_v2.py",
                "description": "Table structure detection visualization",
                "default_args": [],
            },
            "table-crop": {
                "script": "visualize_table_crop_v2.py",
                "description": "Table cropping visualization",
                "default_args": [],
            },
        }

    def run_single_script(
        self,
        script_name: str,
        images: List[str],
        extra_args: List[str] = None,
        use_test_images: bool = False,
        no_filtering_steps: bool = False,
        save_intermediates: bool = False,
    ) -> Dict[str, Any]:
        """Run a single visualization script."""
        if script_name not in self.available_scripts:
            raise ValueError(f"Unknown script: {script_name}")

        script_info = self.available_scripts[script_name]
        script_path = script_dir / script_info["script"]

        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        # Build command
        cmd = [sys.executable, str(script_path)]

        # Add --test-images flag if requested (don't add individual images if using test images)
        if use_test_images:
            cmd.append("--test-images")
        else:
            cmd.extend(images)

        # Add default args, but handle filtering steps override
        default_args = script_info["default_args"].copy()
        
        # For table-lines script, handle --no-filtering-steps override
        if script_name == "table-lines" and no_filtering_steps:
            # Remove --show-filtering-steps from default args and add --no-filtering-steps
            if "--show-filtering-steps" in default_args:
                default_args.remove("--show-filtering-steps")
            default_args.append("--no-filtering-steps")
        
        cmd.extend(default_args)
        
        # Add --save-intermediates if requested and supported by the script
        if save_intermediates:
            # V2 scripts use --save-debug for intermediate saving
            if script_name == "deskew":
                cmd.append("--save-debug")

        if extra_args:
            cmd.extend(extra_args)

        print(f"\n{'='*60}")
        print(f"Running {script_name}: {script_info['description']}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")

        start_time = datetime.now()

        try:
            # Run the script from project root directory for proper path resolution
            project_root = script_dir.parent
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=project_root, timeout=600
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            success = result.returncode == 0

            if success:
                print(
                    f"SUCCESS: {script_name} completed successfully in {duration:.1f}s"
                )
            else:
                print(f"FAILED: {script_name} failed after {duration:.1f}s")
                print(f"Error: {result.stderr}")

            return {
                "script": script_name,
                "success": success,
                "duration": duration,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

        except subprocess.TimeoutExpired as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            print(f"TIMEOUT: {script_name} timed out after {duration:.1f}s (600s limit)")
            print(f"Command: {' '.join(cmd)}")

            return {
                "script": script_name,
                "success": False,
                "duration": duration,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "error": f"Script timed out after 600 seconds",
                "returncode": -1,
            }

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            print(f"CRASHED: {script_name} crashed after {duration:.1f}s: {e}")

            return {
                "script": script_name,
                "success": False,
                "duration": duration,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "error": str(e),
                "returncode": -1,
            }

    def run_multiple_scripts(
        self,
        script_names: List[str],
        images: List[str],
        args_per_script: Dict[str, List[str]] = None,
        use_test_images: bool = False,
        no_filtering_steps: bool = False,
        save_intermediates: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run multiple visualization scripts."""
        results = []

        if use_test_images:
            print(
                f"\nRunning {len(script_names)} visualization scripts on all test images"
            )
            print("Using batch mode: processing images from input/test_images/")
        else:
            print(
                f"\nRunning {len(script_names)} visualization scripts on {len(images)} images"
            )
        print(f"Scripts: {', '.join(script_names)}")

        total_start = datetime.now()

        for script_name in script_names:
            extra_args = args_per_script.get(script_name, []) if args_per_script else []
            result = self.run_single_script(
                script_name, images, extra_args, use_test_images, no_filtering_steps, save_intermediates
            )
            results.append(result)

            # Print status
            if result["success"]:
                print(f"  SUCCESS: {script_name}")
            else:
                print(f"  FAILED: {script_name}")

        total_end = datetime.now()
        total_duration = (total_end - total_start).total_seconds()

        # Summary
        successful = sum(1 for r in results if r["success"])
        print(f"\n{'='*60}")
        print("BATCH RUN SUMMARY")
        print(f"{'='*60}")
        print(f"Completed: {successful}/{len(results)} scripts")
        print(f"Total time: {total_duration:.1f}s")
        print(f"Average time per script: {total_duration/len(results):.1f}s")

        if successful < len(results):
            failed_scripts = [r["script"] for r in results if not r["success"]]
            print(f"Failed scripts: {', '.join(failed_scripts)}")

        return results

    def run_pipeline(
        self,
        script_names: List[str],
        args_per_script: Dict[str, List[str]] = None,
        use_test_images: bool = False,
        no_filtering_steps: bool = False,
        save_intermediates: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run scripts in pipeline mode where each stage uses previous stage's output as input."""
        results = []

        # Create pipeline output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pipeline_dir = self.manager.base_dir / f"pipeline_{timestamp}"
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nRunning {len(script_names)} scripts in pipeline mode")
        print(f"Pipeline directory: {pipeline_dir}")
        print(f"Scripts: {' -> '.join(script_names)}")

        total_start = datetime.now()
        current_input_dir = None

        for i, script_name in enumerate(script_names):
            stage_num = i + 1
            stage_dir = pipeline_dir / f"{stage_num:02d}_{script_name}"
            stage_dir.mkdir(exist_ok=True)

            print(f"\n{'='*60}")
            print(f"PIPELINE STAGE {stage_num}/{len(script_names)}: {script_name}")
            print(f"{'='*60}")

            # Determine input for this stage
            if i == 0:
                # First stage uses test images or specified input
                if use_test_images:
                    print("Stage 1: Using test images as input")
                    extra_args = ["--test-images"]
                else:
                    print("Stage 1: Using individual image arguments")
                    extra_args = []
            else:
                # Special handling for table-crop which needs both table structure and deskewed images
                if script_name == "table-crop":
                    # table-crop needs to process deskewed images using table structure information
                    # Find the deskew stage output (usually stage 3)
                    deskew_stage_idx = None
                    for idx, stage_name in enumerate(script_names):
                        if stage_name == "deskew":
                            deskew_stage_idx = idx
                            break
                    
                    if deskew_stage_idx is not None:
                        deskew_stage_num = deskew_stage_idx + 1
                        deskew_dir = pipeline_dir / f"{deskew_stage_num:02d}_deskew" / "processed_images"
                        if deskew_dir.exists():
                            print(
                                f"Stage {stage_num}: Using deskewed images from stage {deskew_stage_num} with table structure from {script_names[i-1]}"
                            )
                            extra_args = ["--input-dir", str(deskew_dir.resolve())]
                        else:
                            print(
                                f"Stage {stage_num}: Warning - could not find deskewed images, using output from {script_names[i-1]}"
                            )
                            # Fallback to using files from previous stage
                            extra_args = ["--input-dir", str(current_input_dir.resolve())]
                    else:
                        print(
                            f"Stage {stage_num}: Using output from {script_names[i-1]}"
                        )
                        extra_args = ["--input-dir", str(current_input_dir.resolve())]
                else:
                    # Special handling for table-structure which needs table lines images
                    if script_name == "table-structure" and i > 0 and script_names[i-1] == "table-lines":
                        # Look for table_lines_images directory first
                        table_lines_dir = current_input_dir / "table_lines_images"
                        if table_lines_dir.exists():
                            print(
                                f"Stage {stage_num}: Using table lines images from {script_names[i-1]}"
                            )
                            extra_args = ["--input-dir", str(table_lines_dir.resolve())]
                        else:
                            # Fallback to main directory
                            print(
                                f"Stage {stage_num}: Using table lines output from {script_names[i-1]}"
                            )
                            extra_args = ["--input-dir", str(current_input_dir.resolve())]
                    else:
                        # Subsequent stages use previous stage's processed images
                        processed_input_dir = current_input_dir / "processed_images"
                        if processed_input_dir.exists():
                            print(
                                f"Stage {stage_num}: Using processed images from {script_names[i-1]} as input"
                            )
                            extra_args = ["--input-dir", str(processed_input_dir.resolve())]
                        else:
                            print(
                                f"Stage {stage_num}: Using all output from {script_names[i-1]} as input"
                            )
                            extra_args = ["--input-dir", str(current_input_dir.resolve())]

            # Add output directory and any custom args (use absolute paths)
            extra_args.extend(["--output-dir", str(stage_dir.resolve())])
            
            # Add --no-params flag for intermediate pipeline stages to prevent hanging
            if script_name == "deskew":
                extra_args.append("--no-params")
            
            if args_per_script and script_name in args_per_script:
                extra_args.extend(args_per_script[script_name])

            # Run the script
            result = self.run_single_script(
                script_name,
                [],
                extra_args,
                False,  # Don't use test_images flag for individual scripts
                no_filtering_steps,
                save_intermediates,
            )
            results.append(result)

            if not result["success"]:
                print(f"PIPELINE FAILED: Stage {stage_num} ({script_name}) failed!")
                print("Stopping pipeline execution.")
                break

            # Set up for next stage
            current_input_dir = stage_dir
            print(
                f"Stage {stage_num} completed successfully. Output saved to: {stage_dir}"
            )

        total_end = datetime.now()
        total_duration = (total_end - total_start).total_seconds()

        # Pipeline summary
        successful = sum(1 for r in results if r["success"])
        print(f"\n{'='*60}")
        print("PIPELINE SUMMARY")
        print(f"{'='*60}")
        print(f"Completed: {successful}/{len(script_names)} stages")
        print(f"Total time: {total_duration:.1f}s")
        print(
            f"Average time per stage: {total_duration/len(results):.1f}s"
            if results
            else "N/A"
        )
        print(f"Pipeline directory: {pipeline_dir}")

        if successful < len(script_names):
            failed_scripts = [r["script"] for r in results if not r["success"]]
            print(f"Failed stages: {', '.join(failed_scripts)}")
        else:
            print(f"SUCCESS: All {len(script_names)} stages completed successfully!")
            print(f"Final output available in: {current_input_dir}")

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
        if arg.startswith("--") and arg.endswith("-args"):
            # Extract script name (e.g., --deskew-args -> deskew)
            current_script = arg[2:-5]  # Remove -- and -args
            result[current_script] = []
        elif current_script:
            result[current_script].append(arg)
        else:
            # Global argument - ignore for now
            pass

    return result


def parse_script_args_from_argv(argv: List[str]) -> Dict[str, List[str]]:
    """Parse script arguments directly from sys.argv, handling argparse limitations."""
    result = {}
    current_script = None

    i = 0
    while i < len(argv):
        arg = argv[i]

        if arg.startswith("--") and arg.endswith("-args"):
            # Extract script name (e.g., --roi-args -> roi)
            current_script = arg[2:-5]  # Remove -- and -args
            result[current_script] = []
            i += 1

            # Collect arguments until next --flag or end
            while i < len(argv) and not (
                argv[i].startswith("--")
                and not argv[i].startswith("--margin-removal-")
                and not argv[i].startswith("--deskew-")
                and not argv[i].startswith("--page-split-")
            ):
                if argv[i] in [
                    "--test-images",
                    "--pipeline",
                    "--save-report",
                    "--list",
                ]:
                    break
                result[current_script].append(argv[i])
                i += 1
        else:
            i += 1

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
  python run_visualizations.py deskew page-split margin-removal image.jpg

  # Run with custom arguments per script
  python run_visualizations.py deskew page-split image.jpg \\
    --deskew-args --angle-range 30 --angle-step 1.0 \\
    --page-split-args --gutter-start 0.3 --gutter-end 0.7

  # Run all available scripts
  python run_visualizations.py all image.jpg

  # Process all images in test_images directory (batch mode)
  python run_visualizations.py deskew --test-images
  python run_visualizations.py all --test-images

  # Pipeline mode: each stage uses previous stage's output as input
  python run_visualizations.py page-split margin-removal deskew --test-images --pipeline

  # Test new table structure detection
  python run_visualizations.py table-structure --test-images

  # Test all scripts including table structure
  python run_visualizations.py all --test-images --pipeline

  # List available scripts
  python run_visualizations.py --list
        """,
    )

    parser.add_argument(
        "scripts", nargs="*", help='Script names to run (or "all" for all scripts)'
    )

    parser.add_argument(
        "--list", action="store_true", help="List available scripts and exit"
    )
    parser.add_argument(
        "--test-images",
        action="store_true",
        help="Process all images in input/test_images directory (passes --test-images to all scripts)",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Run scripts in pipeline mode where each stage uses previous stage's output as input",
    )
    parser.add_argument(
        "--save-report", action="store_true", help="Save detailed execution report"
    )
    parser.add_argument(
        "--no-filtering-steps",
        action="store_true", 
        help="Disable step-by-step filtering visualization for table-lines (default: enabled)"
    )
    parser.add_argument(
        "--use-v2",
        action="store_true",
        default=True,
        help="Use v2 visualization scripts (default: True, v1 scripts have been removed)"
    )
    parser.add_argument(
        "--method",
        type=str,
        help="Method to use for margin removal (aggressive, bounding_box, smart, curved_black_background)"
    )
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Save intermediate processing steps (supported by deskew and other compatible tools)"
    )

    # Per-script arguments (parsed manually)
    parser.add_argument(
        "--deskew-args", nargs="*", dest="_ignore1", help="Arguments for deskew script"
    )
    parser.add_argument(
        "--page-split-args",
        nargs="*",
        dest="_ignore2",
        help="Arguments for page-split script",
    )
    parser.add_argument(
        "--margin-removal-args", nargs="*", dest="_ignore3", help="Arguments for margin-removal script"
    )
    parser.add_argument(
        "--table-lines-args",
        nargs="*",
        dest="_ignore4",
        help="Arguments for table-lines script",
    )
    parser.add_argument(
        "--table-crop-args",
        nargs="*",
        dest="_ignore5",
        help="Arguments for table-crop script",
    )
    parser.add_argument(
        "--table-structure-args",
        nargs="*",
        dest="_ignore6",
        help="Arguments for table-structure script",
    )
    parser.add_argument(
        "--pipeline-args",
        nargs="*",
        dest="_ignore7",
        help="Arguments for pipeline script",
    )

    # Parse known args to handle script-specific arguments
    args, remaining = parser.parse_known_args()

    runner = VisualizationRunner(use_v2=args.use_v2)

    if args.list:
        runner.list_available_scripts()
        return

    # Parse script-specific arguments using custom parser that handles argparse limitations
    script_args = parse_script_args_from_argv(sys.argv[1:])

    # Also try the original method for any arguments in remaining
    remaining_script_args = parse_script_args(remaining)

    # Merge the results
    for script, args_list in remaining_script_args.items():
        if script in script_args:
            script_args[script].extend(args_list)
        else:
            script_args[script] = args_list

    # Determine which scripts to run
    script_names = args.scripts if args.scripts else []

    if "all" in script_names:
        script_names = list(runner.available_scripts.keys())
    
    # Handle margin-removal variants by mapping to single script
    # Map all margin-removal variants to the unified v2 script
    mapped_names = []
    for script in script_names:
        if script in ["margin-removal-fast", "margin-removal-bbox"]:
            # Map to base margin-removal, but preserve args
            if "margin-removal" not in mapped_names:
                mapped_names.append("margin-removal")
            # Transfer args from variant to base
            if script in script_args:
                if "margin-removal" not in script_args:
                    script_args["margin-removal"] = []
                script_args["margin-removal"].extend(script_args[script])
                # Add method selection based on variant
                if script == "margin-removal-fast":
                    script_args["margin-removal"].append("--use-optimized")
                elif script == "margin-removal-bbox":
                    script_args["margin-removal"].extend(["--method", "bounding_box"])
        else:
            mapped_names.append(script)
    script_names = mapped_names

    # Add method argument to margin-removal if specified
    if args.method and "margin-removal" in script_names:
        if "margin-removal" not in script_args:
            script_args["margin-removal"] = []
        script_args["margin-removal"].extend(["--method", args.method])

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
        # Get images from remaining args (after script args are parsed)
        images = []
        # Try to find images in remaining args (after script args)
        potential_images = [
            arg
            for arg in remaining
            if not arg.startswith("--")
            and Path(arg).suffix.lower() in [".jpg", ".png", ".jpeg"]
        ]
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
    if args.pipeline:
        # Pipeline mode: each stage uses previous stage's output as input
        if len(script_names) < 2:
            print(
                "Pipeline mode requires at least 2 scripts. Use regular mode for single script."
            )
            return
        results = runner.run_pipeline(script_names, script_args, args.test_images, args.no_filtering_steps, args.save_intermediates)
    elif len(script_names) == 1:
        result = runner.run_single_script(
            script_names[0],
            valid_images,
            script_args.get(script_names[0], []),
            args.test_images,
            args.no_filtering_steps,
            args.save_intermediates,
        )
        results = [result]
    else:
        results = runner.run_multiple_scripts(
            script_names, valid_images, script_args, args.test_images, args.no_filtering_steps, args.save_intermediates
        )

    # Save report if requested
    if args.save_report:
        report_dir = runner.manager.base_dir / "reports"
        report_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"visualization_run_{timestamp}.json"

        report_data = {
            "timestamp": datetime.now().isoformat(),
            "scripts_run": script_names,
            "images_processed": valid_images,
            "script_args": script_args,
            "results": results,
        }

        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)

        print(f"\nExecution report saved to: {report_file}")

    print("\nUse 'python tools/check_results.py list' to view all results")


if __name__ == "__main__":
    main()
