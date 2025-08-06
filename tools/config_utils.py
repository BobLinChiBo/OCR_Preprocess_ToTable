"""
Unified configuration utilities for visualization tools.

This module provides common functions for loading configurations and
adding command-line arguments across all visualization scripts.
"""

from pathlib import Path
from typing import Dict, Any, Type, Optional, Tuple
import argparse

from src.ocr_pipeline.config import Stage1Config, Stage2Config, Config


# Default parameter values for different processors
DEFAULT_PARAMS = {
    'page_split': {
        'split_method': 'auto',
        'min_page_width_ratio': 0.25,
        'page_overlap_ratio': 0.02,
        'center_margin_ratio': 0.1,
    },
    'margin_removal': {
        'blur_kernel_size': 9,
        'black_threshold': 45,
        'content_threshold': 180,
        'morph_kernel_size': 30,
        'min_content_area_ratio': 0.005,
        'margin_padding': 10,
        'histogram_threshold': 0.05,
        'projection_smoothing': 3,
        'min_contour_area': 1000,
        'fill_method': 'color_fill',
    },
    'deskew': {
        'deskew_angle_range': 5,
        'deskew_angle_step': 0.1,
        'deskew_method': 'projection',
        'edge_detection_method': 'canny',
        'canny_sigma': 2.0,
    },
    'table_lines': {
        # Core parameters
        'threshold': 40,
        'horizontal_kernel_size': 10,
        'vertical_kernel_size': 10,
        'alignment_threshold': 3,
        'min_aspect_ratio': 5,
        # New H/V separated parameters
        'h_min_length_image_ratio': 0.3,
        'h_min_length_relative_ratio': 0.4,
        'v_min_length_image_ratio': 0.3,
        'v_min_length_relative_ratio': 0.4,
        # Post-processing parameters
        'max_h_length_ratio': 1.0,
        'max_v_length_ratio': 1.0,
        'close_line_distance': 45,
    },
    'table_crop': {
        'table_crop_margin': 10,
        'min_table_width': 100,
        'min_table_height': 100,
        'eps': 10,
        'kernel_ratio': 0.05,
        'padding': 20,
    },
    'table_structure': {
        'eps': 10,
        'kernel_ratio': 0.05,
    }
}


def load_config(args: argparse.Namespace, 
                config_class: Type[Config] = Stage1Config,
                processor_type: Optional[str] = None) -> Tuple[Config, str]:
    """
    Load configuration from file or command line arguments.
    
    Args:
        args: Parsed command line arguments
        config_class: Configuration class to use (Stage1Config or Stage2Config)
        processor_type: Type of processor to get defaults for
        
    Returns:
        Tuple of (config_object, config_source_string)
    """
    if hasattr(args, 'config') and args.config:
        # Load from JSON file
        config = config_class.from_json(Path(args.config))
        config_source = f"file:{args.config}"
    else:
        # Create from command line arguments
        # Only include parameters that are valid for the config class
        valid_params = set(config_class.__annotations__.keys()) if hasattr(config_class, '__annotations__') else set()
        config_params = {}
        
        # Get default parameters for this processor type
        if processor_type and processor_type in DEFAULT_PARAMS:
            defaults = DEFAULT_PARAMS[processor_type]
            
            # Map command line args to config parameters (only if valid for config class)
            for param_name, default_value in defaults.items():
                if param_name not in valid_params:
                    continue  # Skip parameters not in config class
                    
                # Convert parameter name to command line format (e.g., margin_padding -> margin-padding)
                arg_name = param_name.replace('_', '-')
                
                # Check if argument was provided
                if hasattr(args, param_name):
                    config_params[param_name] = getattr(args, param_name)
                elif hasattr(args, arg_name.replace('-', '_')):
                    config_params[param_name] = getattr(args, arg_name.replace('-', '_'))
                else:
                    config_params[param_name] = default_value
        
        # Add common parameters
        if hasattr(args, 'verbose'):
            config_params['verbose'] = args.verbose
            
        # Handle special parameters that might not be in defaults
        for attr in ['enable_margin_removal', 'enable_deskewing']:
            if hasattr(args, attr):
                config_params[attr] = getattr(args, attr)
                
        config = config_class(**config_params)
        config_source = "command_line"
        
    return config, config_source


def add_config_arguments(parser: argparse.ArgumentParser, 
                        processor_type: str,
                        include_config_file: bool = True):
    """
    Add configuration arguments to argument parser.
    
    Args:
        parser: Argument parser to add arguments to
        processor_type: Type of processor to add arguments for
        include_config_file: Whether to include --config argument
    """
    if include_config_file:
        parser.add_argument(
            "--config",
            help="Path to configuration JSON file",
        )
    
    if processor_type in DEFAULT_PARAMS:
        defaults = DEFAULT_PARAMS[processor_type]
        
        for param_name, default_value in defaults.items():
            # Convert to command line format
            arg_name = f"--{param_name.replace('_', '-')}"
            
            # Determine type
            if isinstance(default_value, bool):
                parser.add_argument(
                    arg_name,
                    action='store_true' if not default_value else 'store_false',
                    default=default_value,
                    help=f"Default: {default_value}"
                )
            elif isinstance(default_value, (list, tuple)):
                # Handle tuples/lists as comma-separated values
                parser.add_argument(
                    arg_name,
                    type=lambda x: eval(x) if x.startswith('(') else x,
                    default=default_value,
                    help=f"Default: {default_value}"
                )
            else:
                parser.add_argument(
                    arg_name,
                    type=type(default_value),
                    default=default_value,
                    help=f"Default: {default_value}"
                )
    
    # Add common arguments
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )


def add_processor_specific_arguments(parser: argparse.ArgumentParser, processor_type: str):
    """
    Add processor-specific arguments that aren't in config.
    
    Args:
        parser: Argument parser
        processor_type: Type of processor
    """
    if processor_type == 'margin_removal':
        parser.add_argument(
            "--method",
            choices=['inscribed'],
            default='inscribed',
            help="Margin removal method (only 'inscribed' is supported)"
        )
        parser.add_argument(
            "--padding-top",
            type=int,
            help="Top padding for smart method (defaults to margin-padding)"
        )
        parser.add_argument(
            "--padding-bottom",
            type=int,
            help="Bottom padding for smart method (defaults to margin-padding)"
        )
        parser.add_argument(
            "--padding-left",
            type=int,
            help="Left padding for smart method (defaults to margin-padding)"
        )
        parser.add_argument(
            "--padding-right",
            type=int,
            help="Right padding for smart method (defaults to margin-padding)"
        )
        # Hybrid method parameters
        parser.add_argument(
            "--edge-margin-width",
            type=int,
            default=50,
            help="Width of edge regions to analyze for hybrid method"
        )
        parser.add_argument(
            "--texture-threshold",
            type=float,
            default=10.0,
            help="Variance threshold for uniform areas in hybrid method"
        )
        parser.add_argument(
            "--black-intensity-max",
            type=int,
            default=75,
            help="Maximum intensity for dark areas in hybrid method"
        )
        # Edge transition method parameters
        parser.add_argument(
            "--edge-percentage",
            type=float,
            default=0.20,
            help="Percentage of image edge to scan for transitions (gradient method: 0.20, edge_transition method: 0.15)"
        )
        parser.add_argument(
            "--intensity-jump-threshold",
            type=int,
            default=30,
            help="Minimum intensity jump to consider a boundary (edge_transition method)"
        )
        # Gradient method parameters
        parser.add_argument(
            "--gradient-window-size",
            type=int,
            default=21,
            help="Size of smoothing window for gradient analysis (gradient method)"
        )
        parser.add_argument(
            "--intensity-shift-threshold",
            type=float,
            default=50.0,
            help="Minimum sustained intensity change for boundary (gradient method)"
        )
        parser.add_argument(
            "--margin-confidence-threshold",
            type=float,
            default=0.7,
            help="Statistical confidence required for margin detection (gradient method)"
        )
        # Inscribed rectangle method parameters
        parser.add_argument(
            "--inscribed-blur-ksize",
            type=int,
            default=7,
            help="Gaussian blur kernel size for inscribed rectangle method"
        )
        parser.add_argument(
            "--inscribed-close-ksize",
            type=int,
            default=30,
            help="Morphological closing kernel size for inscribed rectangle method"
        )
        parser.add_argument(
            "--inscribed-close-iter",
            type=int,
            default=3,
            help="Number of closing iterations for inscribed rectangle method"
        )
        parser.add_argument(
            "--direct-gradient-cutting",
            action="store_true",
            default=True,
            help="Use direct gradient-based cutting point calculation (gradient method)"
        )
        parser.add_argument(
            "--max-content-cut-ratio",
            type=float,
            default=0.25,
            help="Maximum ratio of image dimension to cut into content (gradient method)"
        )
        parser.add_argument(
            "--strict-margin-detection",
            action="store_true",
            default=True,
            help="Use strict criteria to avoid false positives on clean sides (gradient method)"
        )
        parser.add_argument(
            "--gradient-aggressive-cutting-threshold",
            type=float,
            default=80.0,
            help="Edge intensity threshold for aggressive cutting (gradient method)"
        )
        parser.add_argument(
            "--gradient-min-margin-intensity",
            type=int,
            default=80,
            help="Minimum intensity to consider a margin - lower values require darker margins (gradient method)"
        )
        parser.add_argument(
            "--gradient-contrast-threshold",
            type=float,
            default=40.0,
            help="Minimum contrast between edge and content for margin detection (gradient method)"
        )
    elif processor_type == 'deskew':
        parser.add_argument(
            "--angle-range",
            type=int,
            default=5,
            help="Angle range for deskewing (±degrees)"
        )
        parser.add_argument(
            "--angle-step",
            type=float,
            default=0.1,
            help="Angle step for deskewing"
        )


def get_command_args_dict(args: argparse.Namespace, processor_type: str) -> Dict[str, Any]:
    """
    Extract relevant command arguments for a processor type.
    
    Args:
        args: Parsed arguments
        processor_type: Type of processor
        
    Returns:
        Dictionary of relevant arguments
    """
    command_args = {}
    
    # Get default parameters for this processor
    if processor_type in DEFAULT_PARAMS:
        for param_name in DEFAULT_PARAMS[processor_type]:
            if hasattr(args, param_name):
                command_args[param_name] = getattr(args, param_name)
            # Also check with underscores replaced by hyphens
            attr_name = param_name.replace('-', '_')
            if hasattr(args, attr_name):
                command_args[param_name] = getattr(args, attr_name)
    
    # Add processor-specific arguments
    if processor_type == 'margin_removal':
        for param in ['method', 'expansion_factor', 'use_min_area_rect', 
                      'gradient_aggressive_cutting_threshold', 'gradient_min_margin_intensity', 
                      'gradient_contrast_threshold']:
            if hasattr(args, param):
                command_args[param] = getattr(args, param)
    
    # Add common arguments
    for param in ['verbose', 'save_debug', 'output_dir']:
        if hasattr(args, param):
            command_args[param] = getattr(args, param)
            
    return command_args