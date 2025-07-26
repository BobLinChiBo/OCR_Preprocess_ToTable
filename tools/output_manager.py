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


class OutputManager:
    """Manages organized output structure for visualization scripts."""
    
    def __init__(self, base_output_dir: str = "visualization_outputs"):
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
                                analysis_data: Dict[str, Any], run_dir: Path):
    """Organize output files into structured directories."""
    
    # Move image files to images directory
    images_dir = run_dir / "images"
    analysis_dir = run_dir / "analysis"
    comparisons_dir = run_dir / "comparisons"
    
    organized_files = {}
    
    for file_type, file_path in output_files.items():
        if not file_path or not Path(file_path).exists():
            continue
            
        source_path = Path(file_path)
        
        # Determine destination based on file type
        if 'comparison' in file_type or 'overlay' in file_type:
            dest_dir = comparisons_dir
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
    
    # Save analysis data
    analysis_file = analysis_dir / f"{script_name}_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis_data, f, indent=2, default=str)
    
    # Create run metadata
    metadata = {
        'script_name': script_name,
        'timestamp': datetime.now().isoformat(),
        'output_files': organized_files,
        'analysis_summary': analysis_data
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


def get_test_images(test_images_dir: str = "input/test_images") -> List[Path]:
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