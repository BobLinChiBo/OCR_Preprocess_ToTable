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
        Split a page image into left and right pages using robust algorithm.
        
        Args:
            image: Input image
            **kwargs: Additional parameters to override config
            
        Returns:
            Tuple of (left_page, right_page)
        """
        params = {
            'search_ratio': kwargs.get('search_ratio', self.config.search_ratio),
            'blur_k': kwargs.get('blur_k', self.config.blur_k),
            'open_k': kwargs.get('open_k', self.config.open_k),
            'width_min': kwargs.get('width_min', self.config.width_min),
        }
        
        return utils.split_two_page_image(image, **params)


class MarginRemovalProcessor(BaseProcessor):
    """Processor for margin removal operations."""
    
    def __init__(self, config: Stage1Config):
        super().__init__(config)
        
    def process(self, image: np.ndarray, method: str = "inscribed", **kwargs) -> Tuple:
        """
        Remove margins from an image.
        
        Args:
            image: Input image
            method: "inscribed" (inscribed rectangle), "aggressive" (legacy), "bounding_box", "smart" (asymmetric projection-based), "curved_black_background" (book page black background removal), "hybrid" (position + texture analysis), "edge_transition" (simple edge intensity jump detection), or "gradient" (gradient-based sustained transition detection)
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
            'return_analysis': kwargs.get('return_analysis', False),
        }
        
        if method == "inscribed":
            # Inscribed rectangle method with new parameters
            inscribed_params = {
                'blur_ksize': kwargs.get('blur_ksize', getattr(self.config, 'inscribed_blur_ksize', 5)),
                'close_ksize': kwargs.get('close_ksize', getattr(self.config, 'inscribed_close_ksize', 25)),
                'close_iter': kwargs.get('close_iter', getattr(self.config, 'inscribed_close_iter', 2)),
                'return_analysis': kwargs.get('return_analysis', False),
            }
            return utils.remove_margin_inscribed(image, **inscribed_params)
        elif method == "bounding_box":
            # Additional parameters for bounding box method
            base_params.update({
                'padding': kwargs.get('padding', self.config.margin_padding),
                'expansion_factor': kwargs.get('expansion_factor', 0.0),
                'use_min_area_rect': kwargs.get('use_min_area_rect', False),
            })
            return utils.remove_margin_bounding_box(image, **base_params)
        elif method == "smart":
            # Smart asymmetric margin removal with projection histograms
            base_params.update({
                'padding_top': kwargs.get('padding_top', self.config.margin_padding),
                'padding_bottom': kwargs.get('padding_bottom', self.config.margin_padding),
                'padding_left': kwargs.get('padding_left', self.config.margin_padding),
                'padding_right': kwargs.get('padding_right', self.config.margin_padding),
                'histogram_threshold': kwargs.get('histogram_threshold', 0.05),
                'projection_smoothing': kwargs.get('projection_smoothing', 3),
            })
            return utils.remove_margin_smart(image, **base_params)
        elif method == "curved_black_background":
            # Curved black background removal for book pages
            curved_params = {
                'black_threshold': kwargs.get('black_threshold', self.config.black_threshold),
                'min_contour_area': kwargs.get('min_contour_area', 1000),
                'padding': kwargs.get('padding', self.config.margin_padding),
                'fill_method': kwargs.get('fill_method', 'color_fill'),
                'return_analysis': kwargs.get('return_analysis', False),
            }
            return utils.remove_curved_black_background(image, **curved_params)
        elif method == "hybrid":
            # Hybrid position + texture analysis for margin removal
            hybrid_params = {
                'edge_margin_width': kwargs.get('edge_margin_width', 50),
                'texture_threshold': kwargs.get('texture_threshold', 10.0),
                'black_intensity_max': kwargs.get('black_intensity_max', 75),
                'fill_method': kwargs.get('fill_method', 'color_fill'),
                'return_analysis': kwargs.get('return_analysis', False),
            }
            return utils.remove_margins_hybrid(image, **hybrid_params)
        elif method == "edge_transition":
            # Simple edge transition detection for margin removal
            edge_params = {
                'edge_percentage': kwargs.get('edge_percentage', 0.15),
                'intensity_jump_threshold': kwargs.get('intensity_jump_threshold', 100),
                'fill_method': kwargs.get('fill_method', 'color_fill'),
                'return_analysis': kwargs.get('return_analysis', False),
            }
            return utils.remove_margins_edge_transition(image, **edge_params)
        elif method == "gradient":
            # Gradient-based sustained transition detection for margin removal
            gradient_params = {
                'edge_percentage': kwargs.get('edge_percentage', 0.20),
                'gradient_window_size': kwargs.get('gradient_window_size', 21),
                'intensity_shift_threshold': kwargs.get('intensity_shift_threshold', 50.0),
                'margin_confidence_threshold': kwargs.get('margin_confidence_threshold', 0.7),
                'fill_method': kwargs.get('fill_method', 'crop'),
                'direct_gradient_cutting': kwargs.get('direct_gradient_cutting', True),
                'max_content_cut_ratio': kwargs.get('max_content_cut_ratio', 0.25),
                'strict_margin_detection': kwargs.get('strict_margin_detection', True),
                'gradient_aggressive_cutting_threshold': kwargs.get('gradient_aggressive_cutting_threshold', 80.0),
                'gradient_min_margin_intensity': kwargs.get('gradient_min_margin_intensity', 80),
                'gradient_contrast_threshold': kwargs.get('gradient_contrast_threshold', 40.0),
                'use_binarization': kwargs.get('use_binarization', False),
                'return_analysis': kwargs.get('return_analysis', False),
            }
            return utils.remove_margins_gradient(image, **gradient_params)
        else:
            # Default to aggressive method
            base_params.update({
                'padding': kwargs.get('padding', self.config.margin_padding),
            })
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
            Deskewed image or (deskewed_image, angle) if return_analysis_data=True
        """
        params = {
            'angle_range': kwargs.get('angle_range', self.config.angle_range),
            'angle_step': kwargs.get('angle_step', self.config.angle_step),
            'min_angle_correction': kwargs.get('min_angle_correction', self.config.min_angle_correction),
            'return_analysis_data': kwargs.get('return_analysis_data', False),
        }
        
        result = utils.deskew_image(image, **params)
        
        # Handle different return formats
        if kwargs.get('return_angle', False):
            # If caller wants angle, return just (image, angle)
            if len(result) == 3:
                return result[0], result[1]  # (deskewed_image, angle)
            else:
                return result  # Already (deskewed_image, angle)
        else:
            # If caller doesn't want angle, return just the image
            return result[0] if isinstance(result, tuple) else result


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
            
            # New post-processing parameters
            'max_h_length_ratio': kwargs.get('max_h_length_ratio', self.get_config_value('max_h_length_ratio', 1.0)),
            'max_v_length_ratio': kwargs.get('max_v_length_ratio', self.get_config_value('max_v_length_ratio', 0.9)),
            'close_line_distance': kwargs.get('close_line_distance', self.get_config_value('close_line_distance', 20)),
            
            # Backward compatibility for old parameter name
            'max_length_ratio': kwargs.get('max_length_ratio', None),
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


class TableDetectionProcessor(BaseProcessor):
    """Processor for table structure detection operations."""
    
    def __init__(self, config: Stage1Config):
        super().__init__(config)
        
    def process(self, lines_image: np.ndarray, **kwargs) -> Dict:
        """
        Detect table structure from a lines image.
        
        Args:
            lines_image: Image containing detected table lines
            **kwargs: Additional parameters to override config
            
        Returns:
            Dict with table structure information
        """
        params = {
            'eps': kwargs.get('eps', self.get_config_value('table_detection_eps', 10)),
            'kernel_ratio': kwargs.get('kernel_ratio', self.get_config_value('table_detection_kernel_ratio', 0.05)),
            'return_analysis': kwargs.get('return_analysis', True),
        }
        
        return utils.detect_table_structure(lines_image, **params)


class TableCropProcessor(BaseProcessor):
    """Processor for table cropping operations using detected table structure."""
    
    def __init__(self, config: Stage1Config):
        super().__init__(config)
        
    def process(self, deskewed_image: np.ndarray, table_structure: Dict, **kwargs) -> Tuple:
        """
        Crop deskewed image to table borders with padding.
        
        Args:
            deskewed_image: The deskewed image to crop
            table_structure: Result from TableDetectionProcessor
            **kwargs: Additional parameters to override config
            
        Returns:
            Cropped image or (cropped_image, analysis) if return_analysis=True
        """
        params = {
            'padding': kwargs.get('padding', self.get_config_value('table_crop_padding', 20)),
            'return_analysis': kwargs.get('return_analysis', False),
        }
        
        return utils.crop_to_table_borders(deskewed_image, table_structure, **params)
    
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
        'table_detection': TableDetectionProcessor,
        'table_crop': TableCropProcessor,
    }
    
    processor_class = processors.get(processor_type)
    if not processor_class:
        raise ValueError(f"Unknown processor type: {processor_type}")
        
    return processor_class(config)