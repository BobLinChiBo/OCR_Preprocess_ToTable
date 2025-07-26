"""
Configuration loader with support for multiple formats and validation.

Provides comprehensive configuration loading with support for JSON, YAML,
and TOML formats, plus environment variable substitution and validation.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import ValidationError

# Handle tomli import for Python < 3.11
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

from .models import Config
from ..exceptions import ConfigurationError

PathLike = Union[str, Path]


def load_config(config_path: PathLike) -> Config:
    """
    Load configuration from file with automatic format detection.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Validated configuration object
        
    Raises:
        ConfigurationError: If configuration cannot be loaded or is invalid
    """
    path = Path(config_path)
    
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")
    
    try:
        # Detect format from extension
        suffix = path.suffix.lower()
        
        if suffix == '.json':
            config_data = _load_json(path)
        elif suffix in ['.yaml', '.yml']:
            config_data = _load_yaml(path)
        elif suffix == '.toml':
            config_data = _load_toml(path)
        else:
            # Try to auto-detect format
            config_data = _load_auto_detect(path)
        
        # Apply environment variable substitution
        config_data = _substitute_env_vars(config_data)
        
        # Load and validate with Pydantic
        return load_config_from_dict(config_data)
        
    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise
        raise ConfigurationError(f"Error loading configuration from {path}: {e}")


def load_config_from_dict(config_data: Dict[str, Any]) -> Config:
    """
    Load configuration from dictionary.
    
    Args:
        config_data: Configuration dictionary
        
    Returns:
        Validated configuration object
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    try:
        return Config(**config_data)
    except ValidationError as e:
        error_details = []
        for error in e.errors():
            location = " -> ".join(str(x) for x in error['loc'])
            error_details.append(f"{location}: {error['msg']}")
        
        raise ConfigurationError(
            f"Configuration validation failed:\n" + "\n".join(error_details)
        )


def save_config(config: Config, output_path: PathLike, format_type: Optional[str] = None) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration object to save
        output_path: Output file path
        format_type: Format to save in ('json', 'yaml', 'toml'). Auto-detected if None.
        
    Raises:
        ConfigurationError: If configuration cannot be saved
    """
    path = Path(output_path)
    
    # Ensure output directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine format
    if format_type is None:
        format_type = path.suffix.lower().lstrip('.')
    
    try:
        config_dict = config.dict()
        
        if format_type == 'json':
            _save_json(config_dict, path)
        elif format_type in ['yaml', 'yml']:
            _save_yaml(config_dict, path)
        elif format_type == 'toml':
            _save_toml(config_dict, path)
        else:
            raise ConfigurationError(f"Unsupported format: {format_type}")
            
    except Exception as e:
        raise ConfigurationError(f"Error saving configuration to {path}: {e}")


def get_default_config() -> Config:
    """
    Get default configuration object.
    
    Returns:
        Default configuration with all default values
    """
    return Config()


def validate_config_file(config_path: PathLike) -> bool:
    """
    Validate configuration file without loading it fully.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if configuration is valid
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    try:
        load_config(config_path)
        return True
    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(f"Validation failed: {e}")


def _load_json(path: Path) -> Dict[str, Any]:
    """Load JSON configuration file."""
    try:
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in {path}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error reading JSON file {path}: {e}")


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    try:
        with path.open('r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in {path}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error reading YAML file {path}: {e}")


def _load_toml(path: Path) -> Dict[str, Any]:
    """Load TOML configuration file."""
    if tomllib is None:
        raise ConfigurationError("TOML support requires tomli package for Python < 3.11")
    
    try:
        with path.open('rb') as f:
            return tomllib.load(f)
    except Exception as e:
        raise ConfigurationError(f"Error reading TOML file {path}: {e}")


def _load_auto_detect(path: Path) -> Dict[str, Any]:
    """Auto-detect configuration file format."""
    content = path.read_text(encoding='utf-8').strip()
    
    # Try JSON first
    if content.startswith('{'):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
    
    # Try YAML
    try:
        data = yaml.safe_load(content)
        if isinstance(data, dict):
            return data
    except yaml.YAMLError:
        pass
    
    # Try TOML
    if tomllib is not None:
        try:
            return tomllib.loads(content)
        except Exception:
            pass
    
    raise ConfigurationError(f"Unable to detect format for {path}")


def _save_json(config_dict: Dict[str, Any], path: Path) -> None:
    """Save configuration as JSON."""
    with path.open('w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)


def _save_yaml(config_dict: Dict[str, Any], path: Path) -> None:
    """Save configuration as YAML."""
    with path.open('w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)


def _save_toml(config_dict: Dict[str, Any], path: Path) -> None:
    """Save configuration as TOML."""
    # Note: Python's tomllib is read-only, so we'd need toml or tomli_w for writing
    # This is a placeholder - in practice, you'd use a library like toml or tomli_w
    raise ConfigurationError("TOML writing not implemented - use JSON or YAML instead")


def _substitute_env_vars(data: Any, prefix: str = "OCR_") -> Any:
    """
    Recursively substitute environment variables in configuration data.
    
    Looks for strings in format ${ENV_VAR} or ${ENV_VAR:default_value}
    and replaces them with environment variable values.
    
    Args:
        data: Configuration data (dict, list, or primitive)
        prefix: Prefix to add to environment variable names
        
    Returns:
        Data with environment variables substituted
    """
    if isinstance(data, dict):
        return {key: _substitute_env_vars(value, prefix) for key, value in data.items()}
    elif isinstance(data, list):
        return [_substitute_env_vars(item, prefix) for item in data]
    elif isinstance(data, str):
        return _substitute_env_var_string(data, prefix)
    else:
        return data


def _substitute_env_var_string(text: str, prefix: str) -> str:
    """
    Substitute environment variables in a string.
    
    Supports formats:
    - ${VAR} - substitute with environment variable VAR
    - ${VAR:default} - substitute with VAR or use default if not set
    - ${OCR_VAR} - with prefix
    """
    import re
    
    def replace_env_var(match):
        var_expr = match.group(1)
        
        if ':' in var_expr:
            var_name, default_value = var_expr.split(':', 1)
        else:
            var_name, default_value = var_expr, None
        
        # Try with prefix first, then without
        for name in [f"{prefix}{var_name}", var_name]:
            if name in os.environ:
                return os.environ[name]
        
        # Return default if provided, otherwise keep original
        if default_value is not None:
            return default_value
        else:
            return match.group(0)  # Keep original ${...}
    
    # Pattern to match ${VAR} or ${VAR:default}
    pattern = r'\$\{([^}]+)\}'
    return re.sub(pattern, replace_env_var, text)


def create_config_from_legacy(legacy_config_path: PathLike) -> Config:
    """
    Create modern configuration from legacy JSON format.
    
    Args:
        legacy_config_path: Path to legacy configuration file
        
    Returns:
        Modern configuration object
        
    Raises:
        ConfigurationError: If legacy config cannot be converted
    """
    path = Path(legacy_config_path)
    
    if not path.exists():
        raise ConfigurationError(f"Legacy configuration file not found: {path}")
    
    try:
        legacy_data = _load_json(path)
        
        # Map legacy structure to modern structure
        modern_data = {}
        
        # Direct mappings
        if 'directories' in legacy_data:
            modern_data['directories'] = legacy_data['directories']
        
        # Page splitting mapping
        if 'page_splitting' in legacy_data:
            ps = legacy_data['page_splitting']
            modern_data['page_splitting'] = {
                'gutter_search_start_percent': ps.get('GUTTER_SEARCH_START_PERCENT', 0.4),
                'gutter_search_end_percent': ps.get('GUTTER_SEARCH_END_PERCENT', 0.6),
                'split_threshold': ps.get('SPLIT_THRESHOLD', 0.8),
                'left_page_suffix': ps.get('LEFT_PAGE_SUFFIX', '_page_2.jpg'),
                'right_page_suffix': ps.get('RIGHT_PAGE_SUFFIX', '_page_1.jpg'),
            }
        
        # Deskewing mapping
        if 'deskewing' in legacy_data:
            ds = legacy_data['deskewing']
            modern_data['deskewing'] = {
                'angle_range': ds.get('ANGLE_RANGE', 10.0),
                'angle_step': ds.get('ANGLE_STEP', 0.2),
                'min_angle_for_correction': ds.get('MIN_ANGLE_FOR_CORRECTION', 0.2),
            }
        
        # Line detection mapping (more complex)
        if 'line_detection' in legacy_data:
            ld = legacy_data['line_detection']
            modern_data['line_detection'] = {
                'save_debug_images': ld.get('SAVE_DEBUG_IMAGES', True),
                'roi_margins_page_1': ld.get('ROI_MARGINS_PAGE_1', {}),
                'roi_margins_page_2': ld.get('ROI_MARGINS_PAGE_2', {}),
                'roi_margins_default': ld.get('ROI_MARGINS_DEFAULT', {}),
                'v_params': ld.get('V_PARAMS', {}),
                'h_params': ld.get('H_PARAMS', {}),
                'output_viz_line_thickness': ld.get('OUTPUT_VIZ_LINE_THICKNESS', 2),
                'output_viz_v_color_bgr': ld.get('OUTPUT_VIZ_V_COLOR_BGR', [0, 0, 255]),
                'output_viz_h_color_bgr': ld.get('OUTPUT_VIZ_H_COLOR_BGR', [0, 255, 0]),
                'output_json_suffix': ld.get('OUTPUT_JSON_SUFFIX', '_lines.json'),
                'output_viz_suffix': ld.get('OUTPUT_VIZ_SUFFIX', '_visualization.jpg'),
            }
        
        # Other direct mappings
        for section in ['edge_detection', 'table_reconstruction', 'table_fitting', 'table_cropping']:
            if section in legacy_data:
                modern_data[section] = legacy_data[section]
        
        return load_config_from_dict(modern_data)
        
    except Exception as e:
        raise ConfigurationError(f"Error converting legacy configuration: {e}")


def export_example_configs(output_dir: PathLike) -> None:
    """
    Export example configuration files for different stages.
    
    Args:
        output_dir: Directory to save example configs
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Default configuration
    default_config = get_default_config()
    save_config(default_config, output_path / "default_config.yaml")
    
    # Stage 1 configuration
    stage1_config = get_default_config()
    stage1_config.directories.splited_images = "output/stage1_initial_processing/01_split_pages"
    stage1_config.directories.deskewed_images = "output/stage1_initial_processing/02_deskewed"
    stage1_config.directories.lines_images = "output/stage1_initial_processing/03_line_detection"
    stage1_config.directories.table_images = "output/stage1_initial_processing/04_table_reconstruction"
    stage1_config.directories.table_fit_images = "output/stage1_initial_processing/05_cropped_tables"
    stage1_config.directories.debug_output_dir = "debug/stage1_debug/line_detection"
    save_config(stage1_config, output_path / "stage1_config.yaml")
    
    # Stage 2 configuration
    stage2_config = get_default_config()
    stage2_config.directories.splited_images = "output/stage1_initial_processing/05_cropped_tables"
    stage2_config.directories.deskewed_images = "output/stage2_refinement/01_deskewed"
    stage2_config.directories.lines_images = "output/stage2_refinement/02_line_detection"
    stage2_config.directories.table_images = "output/stage2_refinement/03_table_reconstruction"
    stage2_config.directories.table_fit_images = "output/stage2_refinement/04_fitted_tables"
    stage2_config.directories.debug_output_dir = "debug/stage2_debug/line_detection"
    
    # Zero margins for stage 2 (already cropped)
    zero_margins = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    stage2_config.line_detection.roi_margins_page_1.top = 0
    stage2_config.line_detection.roi_margins_page_1.bottom = 0
    stage2_config.line_detection.roi_margins_page_1.left = 0
    stage2_config.line_detection.roi_margins_page_1.right = 0
    stage2_config.line_detection.roi_margins_page_2.top = 0
    stage2_config.line_detection.roi_margins_page_2.bottom = 0
    stage2_config.line_detection.roi_margins_page_2.left = 0
    stage2_config.line_detection.roi_margins_page_2.right = 0
    stage2_config.line_detection.roi_margins_default.top = 0
    stage2_config.line_detection.roi_margins_default.bottom = 0
    stage2_config.line_detection.roi_margins_default.left = 0
    stage2_config.line_detection.roi_margins_default.right = 0
    
    save_config(stage2_config, output_path / "stage2_config.yaml")
    
    # Also save as JSON for backward compatibility
    save_config(stage1_config, output_path / "stage1_config.json")
    save_config(stage2_config, output_path / "stage2_config.json")