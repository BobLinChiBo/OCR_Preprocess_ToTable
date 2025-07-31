#!/usr/bin/env python3
"""
Visualization Results Checker

A comprehensive tool for viewing, comparing, and managing OCR visualization results.
"""

import argparse
import json
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from output_manager import (
    OutputManager,
    create_html_viewer,
    create_parameter_comparison,
    validate_parameter_usage,
)


def print_run_summary(runs: List[Dict[str, Any]], limit: int = 10):
    """Print a formatted summary of visualization runs."""
    if not runs:
        print("No visualization runs found.")
        return

    print(f"\n{'='*80}")
    print(f"VISUALIZATION RUNS SUMMARY (showing latest {min(limit, len(runs))})")
    print(f"{'='*80}")

    # Headers
    print(
        f"{'Script':<20} {'Date':<12} {'Time':<8} {'Images':<7} {'Status':<10} {'Directory'}"
    )
    print(f"{'-'*20} {'-'*12} {'-'*8} {'-'*7} {'-'*10} {'-'*20}")

    for i, run in enumerate(runs[:limit]):
        script = run["script"][:19]
        date_str = run["timestamp"].strftime("%Y-%m-%d")
        time_str = run["timestamp"].strftime("%H:%M:%S")
        image_count = run["image_count"]

        # Determine status
        status = "✓ Complete" if image_count > 0 else "✗ Failed"

        dir_name = run["path"].name[:20]

        print(
            f"{script:<20} {date_str:<12} {time_str:<8} {image_count:<7} {status:<10} {dir_name}"
        )


def show_run_details(run: Dict[str, Any]):
    """Show detailed information about a specific run."""
    print(f"\n{'='*60}")
    print(f"RUN DETAILS: {run['script']}")
    print(f"{'='*60}")
    print(f"Timestamp: {run['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Directory: {run['path']}")
    print(f"Images: {run['image_count']}")
    print(f"Analysis files: {run['analysis_count']}")

    # Show file structure
    print(f"\nFile Structure:")
    for subdir in ["images", "analysis", "comparisons"]:
        subdir_path = run["path"] / subdir
        if subdir_path.exists():
            files = list(subdir_path.glob("*"))
            print(f"  {subdir}/: {len(files)} files")
            for file in sorted(files)[:5]:  # Show first 5 files
                print(f"    - {file.name}")
            if len(files) > 5:
                print(f"    ... and {len(files) - 5} more")

    # Show summary data if available
    if run["summary"]:
        print(f"\nSummary Data:")
        if "config_parameters" in run["summary"]:
            print(f"  Config Parameters:")
            for key, value in run["summary"]["config_parameters"].items():
                print(f"    {key}: {value}")

        if "results" in run["summary"]:
            results = run["summary"]["results"]
            successful = sum(1 for r in results if r.get("success", False))
            print(f"  Processing Results: {successful}/{len(results)} successful")


def compare_runs(runs: List[Dict[str, Any]], script_name: str):
    """Compare multiple runs of the same script."""
    script_runs = [r for r in runs if r["script"] == script_name]

    if len(script_runs) < 2:
        print(
            f"Need at least 2 runs of {script_name} to compare. Found {len(script_runs)}."
        )
        return

    print(f"\n{'='*80}")
    print(f"COMPARING {script_name.upper()} RUNS")
    print(f"{'='*80}")

    # Sort by timestamp
    script_runs.sort(key=lambda x: x["timestamp"], reverse=True)

    print(
        f"{'Run':<5} {'Date':<12} {'Time':<8} {'Images':<7} {'Success Rate':<12} {'Notes'}"
    )
    print(f"{'-'*5} {'-'*12} {'-'*8} {'-'*7} {'-'*12} {'-'*20}")

    for i, run in enumerate(script_runs[:5]):  # Show latest 5
        date_str = run["timestamp"].strftime("%m-%d")
        time_str = run["timestamp"].strftime("%H:%M")
        image_count = run["image_count"]

        # Calculate success rate from summary if available
        success_rate = "N/A"
        notes = ""
        if run["summary"] and "results" in run["summary"]:
            results = run["summary"]["results"]
            successful = sum(1 for r in results if r.get("success", False))
            success_rate = f"{successful}/{len(results)}"
            if successful < len(results):
                notes = f"{len(results) - successful} failed"

        print(
            f"#{i+1:<4} {date_str:<12} {time_str:<8} {image_count:<7} {success_rate:<12} {notes}"
        )


def open_results_viewer(run: Dict[str, Any]):
    """Open the HTML results viewer for a run."""
    try:
        html_file = create_html_viewer(run["path"])
        if html_file and html_file.exists():
            print(f"Opening results viewer: {html_file}")
            webbrowser.open(f"file://{html_file.absolute()}")
            return True
        else:
            print("Could not create HTML viewer.")
            return False
    except Exception as e:
        print(f"Error opening viewer: {e}")
        return False


def cleanup_old_runs(
    manager: OutputManager, keep_latest: int, script_name: Optional[str] = None
):
    """Clean up old visualization runs."""
    print(f"\nCleaning up old runs (keeping latest {keep_latest} per script)...")

    # Get current counts
    runs_before = manager.list_runs(script_name)
    scripts_before = {}
    for run in runs_before:
        script = run["script"]
        scripts_before[script] = scripts_before.get(script, 0) + 1

    # Perform cleanup
    manager.cleanup_old_runs(keep_latest, script_name)

    # Get counts after cleanup
    runs_after = manager.list_runs(script_name)
    scripts_after = {}
    for run in runs_after:
        script = run["script"]
        scripts_after[script] = scripts_after.get(script, 0) + 1

    # Report results
    print("Cleanup results:")
    for script in scripts_before:
        before = scripts_before[script]
        after = scripts_after.get(script, 0)
        removed = before - after
        if removed > 0:
            print(f"  {script}: {before} → {after} runs (removed {removed})")
        else:
            print(f"  {script}: {after} runs (no change)")


def show_parameter_info(
    runs: List[Dict[str, Any]],
    script_name: Optional[str] = None,
    run_index: Optional[int] = None,
):
    """Show parameter information for runs."""
    if script_name:
        runs = [r for r in runs if r["script"] == script_name]

    if not runs:
        print(f"No runs found" + (f" for {script_name}" if script_name else ""))
        return

    if run_index is not None:
        if 0 <= run_index < len(runs):
            runs = [runs[run_index]]
        else:
            print(f"Run index {run_index} out of range (0-{len(runs)-1})")
            return

    print(f"\n{'='*80}")
    print("PARAMETER INFORMATION")
    print(f"{'='*80}")

    for i, run in enumerate(runs[:5]):  # Limit to first 5 runs
        print(
            f"\n[{i}] {run['script']} - {run['timestamp'].strftime('%Y-%m-%d %H:%M')}"
        )
        print(f"    Path: {run['path']}")

        # Look for parameter files
        param_dir = run["path"] / "parameters"
        if param_dir.exists():
            param_files = list(param_dir.glob("*_parameters.json"))
            if param_files:
                print(f"    Parameter files found: {len(param_files)}")
                for param_file in param_files:
                    try:
                        with open(param_file) as f:
                            param_data = json.load(f)

                        step_info = param_data.get("step_info", {})
                        config = param_data.get("configuration", {})
                        params = config.get("parameters", {})

                        print(f"      Step: {step_info.get('step_name', 'unknown')}")
                        print(f"      Config source: {config.get('source', 'unknown')}")
                        print(f"      Parameters: {len(params)} values")

                        # Show key parameters
                        key_params = [
                            "angle_range",
                            "min_line_length",
                            "enable_roi_detection",
                            "gutter_search_start",
                            "roi_min_cut_strength",
                        ]
                        for key in key_params:
                            if key in params:
                                print(f"        {key}: {params[key]}")

                        # Show success indicators
                        indicators = param_data.get("processing_results", {}).get(
                            "success_indicators", {}
                        )
                        if indicators:
                            success_keys = [
                                "overall_success",
                                "deskew_confidence",
                                "roi_coverage",
                                "has_table_structure",
                            ]
                            for key in success_keys:
                                if key in indicators:
                                    value = indicators[key]
                                    if isinstance(value, float):
                                        print(f"        {key}: {value:.3f}")
                                    else:
                                        print(f"        {key}: {value}")

                    except Exception as e:
                        print(f"      Error reading {param_file.name}: {e}")
            else:
                print("    No parameter files found")
        else:
            print("    No parameters directory found")


def compare_parameters(runs: List[Dict[str, Any]], script_name: str):
    """Compare parameters across runs of the same script."""
    script_runs = [r for r in runs if r["script"] == script_name]

    if len(script_runs) < 2:
        print(
            f"Need at least 2 runs of {script_name} to compare parameters. Found {len(script_runs)}."
        )
        return

    print(f"\nComparing parameters for {script_name} across {len(script_runs)} runs...")

    # Collect all parameter files from these runs
    param_files = []
    for run in script_runs:
        param_dir = run["path"] / "parameters"
        if param_dir.exists():
            step_files = list(param_dir.glob(f"{script_name}_parameters.json"))
            param_files.extend(step_files)

    if not param_files:
        print(f"No parameter files found for {script_name}")
        return

    # Create comparison using utility function
    try:
        comparison_file = create_parameter_comparison(
            param_files, Path("data/output/temp")
        )
        if comparison_file:
            print(f"Parameter comparison saved to: {comparison_file}")

            # Display key findings
            with open(comparison_file) as f:
                comparison_data = json.load(f)

            variations = comparison_data.get("parameter_variations", {})
            if script_name in variations:
                step_comparison = variations[script_name]
                print(f"\nComparison Summary:")
                print(f"  Runs compared: {step_comparison['runs_compared']}")

                param_diffs = step_comparison.get("parameter_differences", {})
                if param_diffs:
                    print(f"  Parameters that varied:")
                    for param, info in param_diffs.items():
                        values = info["unique_values"]
                        print(f"    {param}: {len(values)} different values - {values}")
                else:
                    print(f"  All parameters were identical across runs")

                success_rates = step_comparison.get("success_rates", {})
                success_rate = success_rates.get("success_rate", 0)
                print(f"  Success rate: {success_rate:.1%}")

    except Exception as e:
        print(f"Error creating parameter comparison: {e}")


def validate_parameters(runs: List[Dict[str, Any]], script_name: Optional[str] = None):
    """Validate parameter effectiveness for runs."""
    if script_name:
        runs = [r for r in runs if r["script"] == script_name]

    if not runs:
        print(f"No runs found" + (f" for {script_name}" if script_name else ""))
        return

    print(f"\n{'='*80}")
    print("PARAMETER VALIDATION")
    print(f"{'='*80}")

    for run in runs[:3]:  # Limit to first 3 runs
        param_dir = run["path"] / "parameters"
        if not param_dir.exists():
            continue

        param_files = list(param_dir.glob("*_parameters.json"))
        if not param_files:
            continue

        print(f"\n{run['script']} - {run['timestamp'].strftime('%Y-%m-%d %H:%M')}")

        for param_file in param_files:
            try:
                validation = validate_parameter_usage(param_file)
                step = validation["file_info"]["step"]
                score = validation["effectiveness_score"]
                warnings = validation["parameter_warnings"]
                suggestions = validation["suggestions"]

                print(f"  Step: {step}")
                print(f"  Effectiveness Score: {score:.2f}")

                if warnings:
                    print(f"  Warnings:")
                    for warning in warnings:
                        print(f"    - {warning}")

                if suggestions:
                    print(f"  Suggestions:")
                    for suggestion in suggestions:
                        print(f"    - {suggestion}")

            except Exception as e:
                print(f"  Error validating {param_file.name}: {e}")


def main():
    """Main function for the results checker."""
    parser = argparse.ArgumentParser(
        description="Check and manage OCR visualization results"
    )

    # Commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List visualization runs")
    list_parser.add_argument("--script", help="Filter by script name")
    list_parser.add_argument(
        "--limit", type=int, default=10, help="Limit number of runs shown"
    )
    list_parser.add_argument("--since", help="Show runs since date (YYYY-MM-DD)")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show detailed run information")
    show_parser.add_argument(
        "run_index", type=int, help="Run index from list (0-based)"
    )
    show_parser.add_argument("--script", help="Filter by script name first")

    # View command
    view_parser = subparsers.add_parser("view", help="Open HTML viewer for a run")
    view_parser.add_argument(
        "run_index", type=int, help="Run index from list (0-based)"
    )
    view_parser.add_argument("--script", help="Filter by script name first")

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare", help="Compare runs of the same script"
    )
    compare_parser.add_argument("script", help="Script name to compare")

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old runs")
    cleanup_parser.add_argument(
        "--keep", type=int, default=5, help="Number of latest runs to keep per script"
    )
    cleanup_parser.add_argument("--script", help="Clean up specific script only")
    cleanup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting",
    )

    # Latest command
    latest_parser = subparsers.add_parser(
        "latest", help="Show and optionally view latest run"
    )
    latest_parser.add_argument("script", help="Script name")
    latest_parser.add_argument("--view", action="store_true", help="Open HTML viewer")

    # Parameters command
    params_parser = subparsers.add_parser(
        "parameters", help="Show parameter information for runs"
    )
    params_parser.add_argument("script", nargs="?", help="Script name (optional)")
    params_parser.add_argument(
        "--compare", action="store_true", help="Compare parameters across runs"
    )
    params_parser.add_argument(
        "--validate", action="store_true", help="Validate parameter effectiveness"
    )
    params_parser.add_argument(
        "--run-index", type=int, help="Show parameters for specific run index"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize output manager
    manager = OutputManager()

    # Handle commands
    if args.command == "list":
        runs = manager.list_runs(args.script)

        # Filter by date if specified
        if args.since:
            try:
                since_date = datetime.strptime(args.since, "%Y-%m-%d")
                runs = [r for r in runs if r["timestamp"] >= since_date]
            except ValueError:
                print(f"Invalid date format: {args.since}. Use YYYY-MM-DD")
                return

        print_run_summary(runs, args.limit)

        if runs:
            print(f"\nUse 'check-results show <index>' to see details")
            print(f"Use 'check-results view <index>' to open HTML viewer")

    elif args.command == "show":
        runs = manager.list_runs(args.script)
        if args.run_index < len(runs):
            show_run_details(runs[args.run_index])
        else:
            print(
                f"Run index {args.run_index} not found. Use 'list' command to see available runs."
            )

    elif args.command == "view":
        runs = manager.list_runs(args.script)
        if args.run_index < len(runs):
            if not open_results_viewer(runs[args.run_index]):
                print("Failed to open viewer. Try 'show' command to see run details.")
        else:
            print(
                f"Run index {args.run_index} not found. Use 'list' command to see available runs."
            )

    elif args.command == "compare":
        runs = manager.list_runs()
        compare_runs(runs, args.script)

    elif args.command == "cleanup":
        if args.dry_run:
            print("DRY RUN - no files will be deleted")
            runs = manager.list_runs(args.script)
            scripts = {}
            for run in runs:
                script = run["script"]
                scripts[script] = scripts.get(script, 0) + 1

            print("Current run counts:")
            for script, count in scripts.items():
                if count > args.keep:
                    print(
                        f"  {script}: {count} runs (would remove {count - args.keep})"
                    )
                else:
                    print(f"  {script}: {count} runs (no change)")
        else:
            cleanup_old_runs(manager, args.keep, args.script)

    elif args.command == "latest":
        latest_run = manager.get_latest_run(args.script)
        if latest_run:
            runs = [r for r in manager.list_runs() if r["path"] == latest_run]
            if runs:
                show_run_details(runs[0])
                if args.view:
                    open_results_viewer(runs[0])
        else:
            print(f"No runs found for script: {args.script}")

    elif args.command == "parameters":
        runs = manager.list_runs(args.script)

        if args.compare and args.script:
            compare_parameters(runs, args.script)
        elif args.validate:
            validate_parameters(runs, args.script)
        else:
            show_parameter_info(runs, args.script, args.run_index)


if __name__ == "__main__":
    main()
