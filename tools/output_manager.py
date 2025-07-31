#!/usr/bin/env python3
"""
Output Manager for OCR Visualization Scripts

Provides organized output structure and utilities for managing visualization results.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import webbrowser
import tempfile
import numpy as np
from dataclasses import MISSING
import sys
import inspect


class OutputManager:
    """Manages organized output structure for visualization scripts."""
    
    def __init__(self, base_output_dir: str = "data/output/visualization"):
        self.base_dir = Path(base_output_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def create_run_directory(self, script_name: str, custom_suffix: str = "") -> Path:
        """Create a timestamped directory for a visualization run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{script_name}_{timestamp}"
        if custom_suffix:
            run_name += f"_{custom_suffix}"
        
        run_dir = self.base_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (run_dir / "images").mkdir(exist_ok=True)
        (run_dir / "analysis").mkdir(exist_ok=True)
        (run_dir / "comparisons").mkdir(exist_ok=True)
        (run_dir / "parameters").mkdir(exist_ok=True)
        
        return run_dir
    
    def get_latest_run(self, script_name: str) -> Optional[Path]:
        """Get the most recent run directory for a script."""
        pattern = f"{script_name}_*"
        run_dirs = sorted([d for d in self.base_dir.glob(pattern) if d.is_dir()], reverse=True)
        return run_dirs[0] if run_dirs else None
    
    def list_runs(self, script_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all visualization runs."""
        pattern = f"{script_name}_*" if script_name else "*"
        run_dirs = sorted([d for d in self.base_dir.glob(pattern) if d.is_dir()], reverse=True)
        
        runs = []
        for run_dir in run_dirs:
            # Parse run info from directory name
            parts = run_dir.name.split('_')
            if len(parts) >= 3:
                script = parts[0]
                timestamp_str = f"{parts[1]}_{parts[2]}"
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                except ValueError:
                    continue
                
                # Get summary info if available
                summary_files = list(run_dir.glob("*summary*.json"))
                summary_info = {}
                if summary_files:
                    try:
                        with open(summary_files[0]) as f:
                            summary_info = json.load(f)
                    except:
                        pass
                
                # Count output files
                image_count = len(list((run_dir / "images").glob("*.jpg"))) if (run_dir / "images").exists() else 0
                analysis_count = len(list((run_dir / "analysis").glob("*.json"))) if (run_dir / "analysis").exists() else 0
                
                runs.append({
                    'path': run_dir,
                    'script': script,
                    'timestamp': timestamp,
                    'image_count': image_count,
                    'analysis_count': analysis_count,
                    'summary': summary_info
                })
        
        return runs
    
    def cleanup_old_runs(self, keep_latest: int = 5, script_name: Optional[str] = None):
        """Remove old run directories, keeping only the latest N."""
        runs = self.list_runs(script_name)
        
        if script_name:
            # Group by script and clean each separately
            script_runs = [r for r in runs if r['script'] == script_name]
            if len(script_runs) > keep_latest:
                for run in script_runs[keep_latest:]:
                    shutil.rmtree(run['path'])
        else:
            # Group by script type
            by_script = {}
            for run in runs:
                script = run['script']
                if script not in by_script:
                    by_script[script] = []
                by_script[script].append(run)
            
            # Clean each script type
            for script, script_runs in by_script.items():
                if len(script_runs) > keep_latest:
                    for run in script_runs[keep_latest:]:
                        shutil.rmtree(run['path'])


def organize_visualization_output(script_name: str, output_files: Dict[str, str], 
                                analysis_data: Dict[str, Any], run_dir: Path,
                                parameter_file: Optional[Path] = None):
    """Organize output files into structured directories."""
    
    # Move image files to images directory
    images_dir = run_dir / "images"
    analysis_dir = run_dir / "analysis"
    comparisons_dir = run_dir / "comparisons"
    parameters_dir = run_dir / "parameters"
    
    organized_files = {}
    
    for file_type, file_path in output_files.items():
        if not file_path or not Path(file_path).exists():
            continue
            
        source_path = Path(file_path)
        
        # Determine destination based on file type
        if 'comparison' in file_type or 'overlay' in file_type:
            dest_dir = comparisons_dir
        elif 'parameter' in file_type or source_path.name.endswith('_parameters.json'):
            dest_dir = parameters_dir
        elif source_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            dest_dir = images_dir
        else:
            dest_dir = analysis_dir
        
        dest_path = dest_dir / source_path.name
        
        # Move file (or copy if cross-device)
        try:
            shutil.move(str(source_path), str(dest_path))
            organized_files[file_type] = str(dest_path)
        except:
            try:
                shutil.copy2(str(source_path), str(dest_path))
                organized_files[file_type] = str(dest_path)
                source_path.unlink()  # Remove original
            except:
                organized_files[file_type] = str(source_path)  # Keep original location
    
    # Copy parameter file if provided separately
    if parameter_file and parameter_file.exists():
        param_dest = parameters_dir / parameter_file.name
        try:
            if parameter_file != param_dest:  # Don't copy to itself
                shutil.copy2(str(parameter_file), str(param_dest))
                organized_files['parameters'] = str(param_dest)
        except:
            organized_files['parameters'] = str(parameter_file)
    
    # Save analysis data
    analysis_file = analysis_dir / f"{script_name}_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis_data, f, indent=2, default=str)
    
    # Create run metadata
    metadata = {
        'script_name': script_name,
        'timestamp': datetime.now().isoformat(),
        'output_files': organized_files,
        'analysis_summary': analysis_data,
        'has_parameter_documentation': parameter_file is not None or 'parameters' in organized_files
    }
    
    metadata_file = run_dir / "run_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return organized_files


def create_html_viewer(run_dir: Path) -> Path:
    """Create an HTML viewer for visualization results."""
    
    metadata_file = run_dir / "run_metadata.json"
    if not metadata_file.exists():
        return None
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    script_name = metadata['script_name']
    timestamp = metadata['timestamp']
    output_files = metadata['output_files']
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{script_name} Visualization Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
            .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }}
            .image-item {{ border: 1px solid #ddd; padding: 10px; border-radius: 5px; }}
            .image-item img {{ max-width: 100%; height: auto; }}
            .analysis-section {{ background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            pre {{ background: #f5f5f5; padding: 10px; border-radius: 3px; overflow-x: auto; }}
            .file-type {{ font-weight: bold; color: #333; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{script_name} Visualization Results</h1>
            <p><strong>Generated:</strong> {timestamp}</p>
            <p><strong>Run Directory:</strong> {run_dir}</p>
        </div>
    """
    
    # Add images
    if output_files:
        html_content += '<div class="image-grid">'
        
        for file_type, file_path in output_files.items():
            if Path(file_path).suffix.lower() in ['.jpg', '.png', '.jpeg']:
                rel_path = Path(file_path).relative_to(run_dir)
                html_content += f'''
                <div class="image-item">
                    <div class="file-type">{file_type.replace('_', ' ').title()}</div>
                    <img src="{rel_path}" alt="{file_type}">
                    <p><small>{rel_path}</small></p>
                </div>
                '''
        
        html_content += '</div>'
    
    # Add analysis data
    if 'analysis_summary' in metadata:
        html_content += '''
        <div class="analysis-section">
            <h2>Analysis Data</h2>
            <pre>{}</pre>
        </div>
        '''.format(json.dumps(metadata['analysis_summary'], indent=2))
    
    html_content += '''
    </body>
    </html>
    '''
    
    # Save HTML file
    html_file = run_dir / f"{script_name}_results.html"
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    return html_file


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def get_test_images(test_images_dir: str = "data/input/test_images") -> List[Path]:
    """Discover all supported image files in the test images directory.
    
    Args:
        test_images_dir: Path to the test images directory
        
    Returns:
        List of Path objects for found image files
    """
    test_dir = Path(test_images_dir)
    
    if not test_dir.exists():
        print(f"Warning: Test images directory not found: {test_dir}")
        return []
    
    # Supported image extensions
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Find all image files
    image_files = []
    for ext in supported_extensions:
        image_files.extend(test_dir.glob(f"*{ext}"))
        image_files.extend(test_dir.glob(f"*{ext.upper()}"))
    
    # Sort for consistent processing order
    image_files = sorted(set(image_files))
    
    if not image_files:
        print(f"Warning: No image files found in {test_dir}")
        print(f"Looking for extensions: {', '.join(supported_extensions)}")
    else:
        print(f"Found {len(image_files)} test images in {test_dir}")
    
    return image_files


def get_default_output_manager() -> OutputManager:
    """Get the default output manager instance."""
    return OutputManager()


def save_step_parameters(step_name: str, config_obj: Any, command_args: Dict[str, Any], 
                        processing_results: Dict[str, Any], output_dir: Path,
                        config_source: str = "default") -> Path:
    """Save detailed parameter documentation for a processing step.
    
    Args:
        step_name: Name of the processing step (e.g., 'deskew', 'roi_detection')
        config_obj: Configuration object with parameter values
        command_args: Command line arguments and overrides
        processing_results: Results/metrics from processing 
        output_dir: Directory to save parameter files
        config_source: Source of configuration ('default', 'file', 'override')
        
    Returns:
        Path to the saved parameter file
    """
    parameters_dir = output_dir / "parameters"
    parameters_dir.mkdir(exist_ok=True)
    
    # Extract configuration parameters
    if hasattr(config_obj, '__dict__'):
        config_params = {k: v for k, v in config_obj.__dict__.items() 
                        if not k.startswith('_') and not callable(v)}
    else:
        config_params = dict(config_obj) if isinstance(config_obj, dict) else {}
    
    # Get parameter documentation from config class if available
    param_docs = {}
    if hasattr(config_obj, '__class__') and hasattr(config_obj.__class__, '__dataclass_fields__'):
        for field_name, field in config_obj.__class__.__dataclass_fields__.items():
            param_docs[field_name] = {
                'type': str(field.type),
                'default': field.default if field.default is not MISSING else (field.default_factory() if field.default_factory is not MISSING else None),
                'description': getattr(field, 'metadata', {}).get('description', '')
            }
    
    # Determine parameter sources and overrides
    param_sources = {}
    overrides = {}
    for key, value in config_params.items():
        if key in command_args and command_args[key] is not None:
            param_sources[key] = "command_line"
            overrides[key] = {"from": "default", "to": command_args[key]}
        else:
            param_sources[key] = config_source
    
    # Environment information
    env_info = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'script_path': inspect.stack()[-1].filename if inspect.stack() else '',
        'working_directory': str(Path.cwd())
    }
    
    # Create comprehensive parameter documentation
    parameter_doc = {
        'step_info': {
            'step_name': step_name,
            'description': f"Parameter documentation for {step_name} processing step",
            'timestamp': datetime.now().isoformat()
        },
        'configuration': {
            'source': config_source,
            'parameters': convert_numpy_types(config_params),
            'parameter_sources': param_sources,
            'overrides': overrides,
            'parameter_documentation': param_docs
        },
        'processing_results': {
            'metrics': convert_numpy_types(processing_results),
            'success_indicators': extract_success_indicators(processing_results)
        },
        'environment': env_info
    }
    
    # Save parameter documentation
    param_file = parameters_dir / f"{step_name}_parameters.json"
    with open(param_file, 'w') as f:
        json.dump(parameter_doc, f, indent=2, default=str)
    
    return param_file


def extract_success_indicators(processing_results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract success indicators and quality metrics from processing results."""
    indicators = {}
    
    # Common success patterns
    if 'success' in processing_results:
        indicators['overall_success'] = processing_results['success']
    
    if 'error' in processing_results:
        indicators['has_errors'] = True
        indicators['error_message'] = processing_results['error']
    else:
        indicators['has_errors'] = False
    
    # Step-specific indicators
    if 'skew_info' in processing_results:
        skew = processing_results['skew_info']
        indicators['deskew_confidence'] = skew.get('confidence', 0)
        indicators['will_rotate'] = skew.get('will_rotate', False)
        indicators['line_count'] = skew.get('line_count', 0)
    
    if 'roi_info' in processing_results:
        roi = processing_results['roi_info']
        if roi:
            roi_area = (roi.get('roi_right', 0) - roi.get('roi_left', 0)) * (roi.get('roi_bottom', 0) - roi.get('roi_top', 0))
            total_area = roi.get('image_width', 1) * roi.get('image_height', 1)
            indicators['roi_coverage'] = roi_area / total_area if total_area > 0 else 0
    
    if 'line_info' in processing_results:
        lines = processing_results['line_info']
        indicators['horizontal_lines'] = lines.get('h_line_count', 0)
        indicators['vertical_lines'] = lines.get('v_line_count', 0)
        indicators['has_table_structure'] = lines.get('has_table_structure', False)
    
    if 'gutter_info' in processing_results:
        gutter = processing_results['gutter_info']
        indicators['gutter_strength'] = gutter.get('gutter_strength', 0)
        indicators['meets_min_width'] = gutter.get('meets_min_width', False)
    
    return indicators


def create_parameter_comparison(param_files: List[Path], output_dir: Path) -> Path:
    """Create a comparison report across multiple parameter files.
    
    Args:
        param_files: List of parameter JSON files to compare
        output_dir: Directory to save comparison report
        
    Returns:
        Path to the comparison report
    """
    parameters_dir = output_dir / "parameters"  
    parameters_dir.mkdir(exist_ok=True)
    
    comparison_data = {
        'comparison_info': {
            'created': datetime.now().isoformat(),
            'files_compared': [str(f) for f in param_files],
            'comparison_count': len(param_files)
        },
        'parameter_variations': {},
        'success_comparison': {},
        'recommendations': []
    }
    
    # Load all parameter files
    param_data = []
    for param_file in param_files:
        try:
            with open(param_file) as f:
                data = json.load(f)
                param_data.append({
                    'file': param_file,
                    'data': data,
                    'step': data.get('step_info', {}).get('step_name', 'unknown')
                })
        except Exception as e:
            print(f"Warning: Could not load {param_file}: {e}")
    
    if not param_data:
        return None
    
    # Group by step type
    by_step = {}
    for item in param_data:
        step = item['step']
        if step not in by_step:
            by_step[step] = []
        by_step[step].append(item)
    
    # Compare parameters within each step type
    for step, items in by_step.items():
        if len(items) < 2:
            continue
            
        step_comparison = {
            'runs_compared': len(items),
            'parameter_differences': {},
            'success_rates': {}
        }
        
        # Find parameter variations
        all_params = {}
        for item in items:
            params = item['data'].get('configuration', {}).get('parameters', {})
            for key, value in params.items():
                if key not in all_params:
                    all_params[key] = []
                all_params[key].append(value)
        
        # Identify varying parameters
        for param, values in all_params.items():
            if len(set(str(v) for v in values)) > 1:  # Parameter varies
                step_comparison['parameter_differences'][param] = {
                    'values': values,
                    'unique_values': list(set(str(v) for v in values))
                }
        
        # Compare success rates
        successes = []
        for item in items:
            indicators = item['data'].get('processing_results', {}).get('success_indicators', {})
            successes.append(indicators.get('overall_success', True))
        
        step_comparison['success_rates'] = {
            'total_runs': len(successes),
            'successful_runs': sum(successes),
            'success_rate': sum(successes) / len(successes) if successes else 0
        }
        
        comparison_data['parameter_variations'][step] = step_comparison
    
    # Generate recommendations
    for step, comparison in comparison_data['parameter_variations'].items():
        if comparison['success_rates']['success_rate'] < 1.0:
            comparison_data['recommendations'].append({
                'step': step,
                'issue': 'Some runs failed',
                'suggestion': 'Review parameter variations for failed runs'
            })
        
        if len(comparison['parameter_differences']) > 5:
            comparison_data['recommendations'].append({
                'step': step,
                'issue': 'Many parameter variations',
                'suggestion': 'Consider standardizing frequently changed parameters'
            })
    
    # Save comparison report
    comparison_file = parameters_dir / "parameter_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2, default=str)
    
    return comparison_file


def validate_parameter_usage(param_file: Path) -> Dict[str, Any]:
    """Validate parameter effectiveness and suggest improvements.
    
    Args:
        param_file: Path to parameter JSON file
        
    Returns:
        Dictionary with validation results and suggestions
    """
    with open(param_file) as f:
        param_data = json.load(f)
    
    validation = {
        'file_info': {
            'file': str(param_file),
            'step': param_data.get('step_info', {}).get('step_name', 'unknown')
        },
        'parameter_warnings': [],
        'effectiveness_score': 0,
        'suggestions': []
    }
    
    config = param_data.get('configuration', {})
    params = config.get('parameters', {})
    results = param_data.get('processing_results', {})
    indicators = results.get('success_indicators', {})
    
    step_name = validation['file_info']['step']
    
    # Step-specific validation
    if step_name == 'deskew':
        # Check deskewing parameters
        if indicators.get('deskew_confidence', 0) < 0.5:
            validation['parameter_warnings'].append("Low deskew confidence - consider adjusting angle_range or angle_step")
        
        if indicators.get('line_count', 0) < 10:
            validation['parameter_warnings'].append("Few lines detected - image may not be suitable for skew detection")
        
        validation['effectiveness_score'] = indicators.get('deskew_confidence', 0)
        
    elif step_name == 'roi_detection':
        # Check ROI parameters
        coverage = indicators.get('roi_coverage', 1.0)
        if coverage < 0.1:
            validation['parameter_warnings'].append("Very small ROI detected - parameters may be too aggressive")
        elif coverage > 0.95:
            validation['parameter_warnings'].append("ROI covers almost entire image - parameters may be too conservative")
        
        validation['effectiveness_score'] = min(1.0, coverage * 2) if coverage < 0.5 else (2 - coverage * 2)
        
    elif step_name in ['table_lines', 'table_detection']:
        # Check line detection parameters  
        h_lines = indicators.get('horizontal_lines', 0)
        v_lines = indicators.get('vertical_lines', 0)
        
        if h_lines == 0 or v_lines == 0:
            validation['parameter_warnings'].append("Missing horizontal or vertical lines - adjust line detection parameters")
        
        has_structure = indicators.get('has_table_structure', False)
        validation['effectiveness_score'] = 1.0 if has_structure else 0.3
    
    # General parameter checks
    overrides = config.get('overrides', {})
    if len(overrides) > 3:
        validation['suggestions'].append("Many parameter overrides used - consider updating default config")
    
    if not indicators.get('overall_success', True):
        validation['effectiveness_score'] = 0
        validation['suggestions'].append("Processing failed - review all parameters")
    
    return validation


if __name__ == "__main__":
    # Demo usage
    manager = OutputManager()
    print(f"Output base directory: {manager.base_dir}")
    
    # List recent runs
    runs = manager.list_runs()
    if runs:
        print("\nRecent visualization runs:")
        for run in runs[:10]:  # Show latest 10
            print(f"  {run['script']} - {run['timestamp'].strftime('%Y-%m-%d %H:%M')} - {run['image_count']} images")
    else:
        print("\nNo visualization runs found.")