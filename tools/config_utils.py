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
    },
    'deskew': {
        'deskew_angle_range': (-5, 5),
        'deskew_angle_step': 0.1,
        'deskew_method': 'projection',
        'edge_detection_method': 'canny',
        'canny_sigma': 2.0,
    },
    'table_lines': {
        # Note: min_line_length, max_line_gap, and hough_threshold are handled specially
        'horizontal_kernel_ratio': 30,
        'vertical_kernel_ratio': 30,
        'h_erode_iterations': 1,
        'h_dilate_iterations': 1,
        'v_erode_iterations': 1,
        'v_dilate_iterations': 1,
        'min_table_coverage': 0.15,
        'max_parallel_distance': 12,
        'angle_tolerance': 5.0,
        'h_length_filter_ratio': 0.6,
        'v_length_filter_ratio': 0.6,
        'line_merge_distance_h': 15,
        'line_merge_distance_v': 15,
        'line_extension_tolerance': 20,
        'max_merge_iterations': 3,
    },
    'table_crop': {
        'table_crop_margin': 10,
        'min_table_width': 100,
        'min_table_height': 100,
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
    if processor_type == 'table_lines':
        # Add dynamic parameters
        parser.add_argument(
            "--min-line-length",
            type=int,
            help="Minimum line length (calculated dynamically if not specified)"
        )
        parser.add_argument(
            "--max-line-gap",
            type=int,
            help="Maximum line gap (calculated dynamically if not specified)"
        )
        parser.add_argument(
            "--hough-threshold",
            type=int,
            default=60,
            help="Hough transform threshold"
        )
    elif processor_type == 'margin_removal':
        parser.add_argument(
            "--method",
            choices=['aggressive', 'bounding_box'],
            default='aggressive',
            help="Margin removal method"
        )
        parser.add_argument(
            "--expansion-factor",
            type=float,
            default=0.0,
            help="Expansion factor for bounding box method (0.1 = 10%%)"
        )
        parser.add_argument(
            "--use-min-area-rect",
            action="store_true",
            help="Use minimum area rectangle for bounding box method"
        )
    elif processor_type == 'deskew':
        parser.add_argument(
            "--angle-range",
            type=lambda x: eval(x),
            default="(-5, 5)",
            help="Angle range for deskewing as tuple"
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
    if processor_type == 'table_lines':
        for param in ['min_line_length', 'max_line_gap', 'hough_threshold']:
            if hasattr(args, param):
                value = getattr(args, param)
                if value is not None:
                    command_args[param] = value
    elif processor_type == 'margin_removal':
        for param in ['method', 'expansion_factor', 'use_min_area_rect']:
            if hasattr(args, param):
                command_args[param] = getattr(args, param)
    
    # Add common arguments
    for param in ['verbose', 'save_debug', 'output_dir']:
        if hasattr(args, param):
            command_args[param] = getattr(args, param)
            
    return command_args