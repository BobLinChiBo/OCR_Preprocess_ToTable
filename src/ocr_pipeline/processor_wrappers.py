"""
Processor classes that wrap utility functions with config management.

This module provides a clean interface between configuration objects and utility functions,
handling parameter mapping and default values in a centralized way.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from pathlib import Path

from .config import Stage1Config, Stage2Config
from . import utils


class BaseProcessor:
    """Base class for all processors."""
    
    def __init__(self, config):
        self.config = config
        
    def get_config_value(self, key: str, default: Any) -> Any:
        """Safely get a config value with a default."""
        return getattr(self.config, key, default)


class PageSplitProcessor(BaseProcessor):
    """Processor for page splitting operations."""
    
    def __init__(self, config: Stage1Config):
        super().__init__(config)
        
    def process(self, image: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split a page image into left and right pages.
        
        Args:
            image: Input image
            **kwargs: Additional parameters to override config
            
        Returns:
            Tuple of (left_page, right_page)
        """
        params = {
            'split_method': kwargs.get('split_method', self.config.split_method),
            'min_page_width_ratio': kwargs.get('min_page_width_ratio', self.config.min_page_width_ratio),
            'page_overlap_ratio': kwargs.get('page_overlap_ratio', self.config.page_overlap_ratio),
            'center_margin_ratio': kwargs.get('center_margin_ratio', self.config.center_margin_ratio),
            'verbose': kwargs.get('verbose', self.config.verbose),
        }
        
        return utils.split_page(image, **params)


class MarginRemovalProcessor(BaseProcessor):
    """Processor for margin removal operations."""
    
    def __init__(self, config: Stage1Config):
        super().__init__(config)
        
    def process(self, image: np.ndarray, method: str = "aggressive", **kwargs) -> Tuple:
        """
        Remove margins from an image.
        
        Args:
            image: Input image
            method: "aggressive" (inscribed rectangle) or "bounding_box"
            **kwargs: Additional parameters to override config
            
        Returns:
            Cropped image or (cropped_image, analysis) if return_analysis=True
        """
        base_params = {
            'blur_kernel_size': kwargs.get('blur_kernel_size', self.config.blur_kernel_size),
            'black_threshold': kwargs.get('black_threshold', self.config.black_threshold),
            'content_threshold': kwargs.get('content_threshold', self.config.content_threshold),
            'morph_kernel_size': kwargs.get('morph_kernel_size', self.config.morph_kernel_size),
            'min_content_area_ratio': kwargs.get('min_content_area_ratio', self.config.min_content_area_ratio),
            'padding': kwargs.get('padding', self.config.margin_padding),
            'return_analysis': kwargs.get('return_analysis', False),
        }
        
        if method == "bounding_box":
            # Additional parameters for bounding box method
            base_params.update({
                'expansion_factor': kwargs.get('expansion_factor', 0.0),
                'use_min_area_rect': kwargs.get('use_min_area_rect', False),
            })
            return utils.remove_margin_bounding_box(image, **base_params)
        else:
            # Default to aggressive method
            return utils.remove_margin_aggressive(image, **base_params)


class DeskewProcessor(BaseProcessor):
    """Processor for deskewing operations."""
    
    def __init__(self, config: Stage1Config):
        super().__init__(config)
        
    def process(self, image: np.ndarray, **kwargs) -> Tuple:
        """
        Deskew an image.
        
        Args:
            image: Input image
            **kwargs: Additional parameters to override config
            
        Returns:
            Deskewed image or (deskewed_image, angle) if return_angle=True
        """
        params = {
            'angle_range': kwargs.get('angle_range', self.get_config_value('deskew_angle_range', (-5, 5))),
            'angle_step': kwargs.get('angle_step', self.get_config_value('deskew_angle_step', 0.1)),
            'method': kwargs.get('method', self.get_config_value('deskew_method', 'projection')),
            'edge_detection_method': kwargs.get('edge_detection_method', self.get_config_value('edge_detection_method', 'canny')),
            'return_angle': kwargs.get('return_angle', False),
        }
        
        # Handle special parameter names
        if 'sigma' in kwargs:
            params['sigma'] = kwargs['sigma']
        elif hasattr(self.config, 'canny_sigma'):
            params['sigma'] = self.config.canny_sigma
        else:
            params['sigma'] = 2.0
            
        return utils.deskew_image(image, **params)


class TableLineProcessor(BaseProcessor):
    """Processor for table line detection operations."""
    
    def __init__(self, config: Stage1Config):
        super().__init__(config)
        
    def process(self, image: np.ndarray, **kwargs) -> Tuple[List, List, Dict]:
        """
        Detect table lines in an image.
        
        Args:
            image: Input image
            **kwargs: Additional parameters to override config or provide dynamic values
            
        Returns:
            Tuple of (h_lines, v_lines, analysis)
        """
        # Build parameters with proper defaults
        params = {
            # Dynamic parameters (calculated if None)
            'min_line_length': kwargs.get('min_line_length', None),
            'max_line_gap': kwargs.get('max_line_gap', None),
            
            # Config parameters with defaults
            'hough_threshold': kwargs.get('hough_threshold', 60),
            'horizontal_kernel_ratio': self.get_config_value('horizontal_kernel_ratio', 30),
            'vertical_kernel_ratio': self.get_config_value('vertical_kernel_ratio', 30),
            'h_erode_iterations': self.get_config_value('h_erode_iterations', 1),
            'h_dilate_iterations': self.get_config_value('h_dilate_iterations', 1),
            'v_erode_iterations': self.get_config_value('v_erode_iterations', 1),
            'v_dilate_iterations': self.get_config_value('v_dilate_iterations', 1),
            'min_table_coverage': self.get_config_value('min_table_coverage', 0.15),
            'max_parallel_distance': self.get_config_value('max_parallel_distance', 12),
            'angle_tolerance': self.get_config_value('angle_tolerance', 5.0),
            'h_length_filter_ratio': self.get_config_value('h_length_filter_ratio', 0.6),
            'v_length_filter_ratio': self.get_config_value('v_length_filter_ratio', 0.6),
            'line_merge_distance_h': self.get_config_value('line_merge_distance_h', 15),
            'line_merge_distance_v': self.get_config_value('line_merge_distance_v', 15),
            'line_extension_tolerance': self.get_config_value('line_extension_tolerance', 20),
            'max_merge_iterations': self.get_config_value('max_merge_iterations', 3),
            'return_analysis': kwargs.get('return_analysis', True),
        }
        
        # Override with any additional kwargs
        for key, value in kwargs.items():
            if key in params:
                params[key] = value
                
        return utils.detect_table_lines(image, **params)
    
    def process_with_enhanced_analysis(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Process image and return enhanced analysis dict for visualization.
        
        This method adds additional fields expected by visualization scripts.
        """
        h_lines, v_lines, analysis = self.process(image, **kwargs)
        
        # Enhance analysis for visualization compatibility
        analysis["h_lines"] = h_lines
        analysis["v_lines"] = v_lines
        analysis["h_line_count"] = analysis.get("h_lines_count", len(h_lines))
        analysis["v_line_count"] = analysis.get("v_lines_count", len(v_lines))
        analysis["h_avg_length"] = analysis.get("avg_h_length", 0)
        analysis["v_avg_length"] = analysis.get("avg_v_length", 0)
        analysis["has_table_structure"] = len(h_lines) > 0 and len(v_lines) > 0
        
        return analysis


class TableCropProcessor(BaseProcessor):
    """Processor for table cropping operations."""
    
    def __init__(self, config: Stage2Config):
        super().__init__(config)
        
    def process(self, image: np.ndarray, h_lines: List, v_lines: List, **kwargs) -> Tuple:
        """
        Crop table region from an image based on detected lines.
        
        Args:
            image: Input image
            h_lines: Horizontal lines
            v_lines: Vertical lines
            **kwargs: Additional parameters to override config
            
        Returns:
            Cropped table image or (cropped_image, crop_info) if return_crop_info=True
        """
        params = {
            'margin': kwargs.get('margin', self.get_config_value('table_crop_margin', 10)),
            'min_width': kwargs.get('min_width', self.get_config_value('min_table_width', 100)),
            'min_height': kwargs.get('min_height', self.get_config_value('min_table_height', 100)),
            'return_crop_info': kwargs.get('return_crop_info', False),
        }
        
        return utils.crop_table_region(image, h_lines, v_lines, **params)


# Factory function to create processors
def create_processor(processor_type: str, config) -> BaseProcessor:
    """
    Create a processor instance based on type.
    
    Args:
        processor_type: Type of processor ("page_split", "margin_removal", etc.)
        config: Configuration object
        
    Returns:
        Processor instance
    """
    processors = {
        'page_split': PageSplitProcessor,
        'margin_removal': MarginRemovalProcessor,
        'deskew': DeskewProcessor,
        'table_lines': TableLineProcessor,
        'table_crop': TableCropProcessor,
    }
    
    processor_class = processors.get(processor_type)
    if not processor_class:
        raise ValueError(f"Unknown processor type: {processor_type}")
        
    return processor_class(config)