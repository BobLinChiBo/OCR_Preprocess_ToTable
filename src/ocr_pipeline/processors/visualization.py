"""Visualization utilities for debugging OCR pipeline."""

from typing import List, Tuple, Optional
import cv2
import numpy as np
from .base import BaseProcessor


class VisualizationProcessor(BaseProcessor):
    """Processor for creating visualizations of pipeline operations."""
    
    def process(
        self,
        image: np.ndarray,
        visualization_type: str = "lines",
        **kwargs
    ) -> np.ndarray:
        """Create a visualization of the specified type.
        
        Args:
            image: Input image
            visualization_type: Type of visualization ("lines", etc.)
            **kwargs: Visualization-specific parameters
            
        Returns:
            Visualization image
        """
        self.validate_image(image)
        
        if visualization_type == "lines":
            h_lines = kwargs.get("h_lines", [])
            v_lines = kwargs.get("v_lines", [])
            return visualize_detected_lines(image, h_lines, v_lines, **kwargs)
        else:
            raise ValueError(f"Unknown visualization type: {visualization_type}")


def visualize_detected_lines(
    image: np.ndarray,
    h_lines: List[Tuple[int, int, int, int]],
    v_lines: List[Tuple[int, int, int, int]],
    line_color: Tuple[int, int, int] = (0, 0, 255),
    line_thickness: int = 2,
    h_line_color: Optional[Tuple[int, int, int]] = None,
    v_line_color: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """Create visualization of detected lines on the image.
    
    Args:
        image: Base image for visualization
        h_lines: List of horizontal lines as (x1, y1, x2, y2) tuples
        v_lines: List of vertical lines as (x1, y1, x2, y2) tuples
        line_color: Default color for lines (B, G, R)
        line_thickness: Thickness of drawn lines
        h_line_color: Color for horizontal lines (defaults to line_color)
        v_line_color: Color for vertical lines (defaults to green)
        
    Returns:
        Image with lines overlay
    """
    vis_image = image.copy()
    
    # Set default colors
    if h_line_color is None:
        h_line_color = line_color
    if v_line_color is None:
        v_line_color = (0, 255, 0)  # Green for vertical lines

    # Draw horizontal lines
    for x1, y1, x2, y2 in h_lines:
        cv2.line(vis_image, (x1, y1), (x2, y2), h_line_color, line_thickness)

    # Draw vertical lines
    for x1, y1, x2, y2 in v_lines:
        cv2.line(vis_image, (x1, y1), (x2, y2), v_line_color, line_thickness)

    return vis_image