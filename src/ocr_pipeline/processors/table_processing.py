"""Table processing operations including cropping and cell enumeration."""

from typing import Dict, List, Tuple, Union, Any, Optional
import cv2
import numpy as np
from .base import BaseProcessor


class TableProcessingProcessor(BaseProcessor):
    """Processor for table cropping and processing operations."""
    
    def process(
        self,
        image: np.ndarray,
        table_structure: Dict,
        operation: str = "crop",
        **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """Process table-related operations on an image.
        
        Args:
            image: Input image
            table_structure: Table structure from detect_table_structure
            operation: Operation to perform ("crop" or "visualize")
            **kwargs: Operation-specific parameters
            
        Returns:
            Processed image or (processed_image, analysis) if return_analysis=True
        """
        self.validate_image(image)
        
        # Clear previous debug images
        self.clear_debug_images()
        
        # Pass processor instance for debug saving
        kwargs['_processor'] = self
        
        if operation == "crop":
            return crop_to_table_borders(image, table_structure, **kwargs)
        elif operation == "visualize":
            return visualize_table_structure(image, table_structure, **kwargs)
        else:
            raise ValueError(f"Unknown table processing operation: {operation}")


def crop_to_table_borders(
    deskewed_image: np.ndarray,
    table_structure: Dict,
    padding: int = 20,
    return_analysis: bool = False,
    **kwargs
) -> Union[np.ndarray, Tuple]:
    """
    Crop deskewed image to table borders with padding.

    Args:
        deskewed_image: The deskewed image to crop
        table_structure: Result from detect_table_structure()
        padding: Padding around detected table borders
        return_analysis: If True, return analysis info

    Returns:
        Cropped image, or (cropped_image, analysis) if return_analysis=True
    """
    # Get processor instance if available for debug saving
    processor = kwargs.get('_processor', None)
    cells = table_structure.get("cells", [])
    xs = table_structure.get("xs", [])
    ys = table_structure.get("ys", [])
    
    # Save input image
    if processor:
        processor.save_debug_image('input_image', deskewed_image)

    # Prefer using cell boundaries if available
    if cells:
        # Calculate boundaries from cells
        min_x = min(x1 for x1, y1, x2, y2 in cells)
        max_x = max(x2 for x1, y1, x2, y2 in cells)
        min_y = min(y1 for x1, y1, x2, y2 in cells)
        max_y = max(y2 for x1, y1, x2, y2 in cells)
        crop_method = "cells"
    elif xs and ys:
        # Fall back to grid lines if no cells detected
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        crop_method = "grid_lines"
    else:
        # No table detected, return original image
        if processor:
            # Save visualization showing no table detected
            vis = deskewed_image.copy() if len(deskewed_image.shape) == 3 else cv2.cvtColor(deskewed_image, cv2.COLOR_GRAY2BGR)
            cv2.putText(vis, 'NO TABLE DETECTED', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            processor.save_debug_image('no_table_detected', vis)
        
        if return_analysis:
            analysis = {
                "cropped": False,
                "reason": "No table structure detected",
                "original_shape": deskewed_image.shape,
                "cropped_shape": deskewed_image.shape,
            }
            return deskewed_image, analysis
        return deskewed_image

    # Save visualization of detected bounds before padding
    if processor:
        vis = deskewed_image.copy() if len(deskewed_image.shape) == 3 else cv2.cvtColor(deskewed_image, cv2.COLOR_GRAY2BGR)
        
        # Draw detected table boundary (before padding)
        cv2.rectangle(vis, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        cv2.putText(vis, 'Detected Table', (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw all cells if using cell method
        if crop_method == "cells" and cells:
            for x1, y1, x2, y2 in cells:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
        # Draw grid lines if using grid method
        elif crop_method == "grid_lines":
            h, w = vis.shape[:2]
            for x in xs:
                cv2.line(vis, (x, 0), (x, h), (0, 255, 255), 1)
            for y in ys:
                cv2.line(vis, (0, y), (w, y), (0, 255, 255), 1)
        
        processor.save_debug_image('table_bounds_detected', vis)

    # Apply padding and ensure bounds are within image
    min_x = max(0, min_x - padding)
    max_x = min(deskewed_image.shape[1], max_x + padding)
    min_y = max(0, min_y - padding)
    max_y = min(deskewed_image.shape[0], max_y + padding)
    
    # Save visualization with padding
    if processor:
        vis = deskewed_image.copy() if len(deskewed_image.shape) == 3 else cv2.cvtColor(deskewed_image, cv2.COLOR_GRAY2BGR)
        
        # Draw crop region with padding
        cv2.rectangle(vis, (min_x, min_y), (max_x, max_y), (0, 0, 255), 3)
        cv2.putText(vis, f'Crop Region (padding={padding})', (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Highlight areas that will be removed
        overlay = vis.copy()
        h, w = vis.shape[:2]
        # Top margin
        if min_y > 0:
            cv2.rectangle(overlay, (0, 0), (w, min_y), (128, 128, 128), -1)
        # Bottom margin  
        if max_y < h:
            cv2.rectangle(overlay, (0, max_y), (w, h), (128, 128, 128), -1)
        # Left margin
        if min_x > 0:
            cv2.rectangle(overlay, (0, min_y), (min_x, max_y), (128, 128, 128), -1)
        # Right margin
        if max_x < w:
            cv2.rectangle(overlay, (max_x, min_y), (w, max_y), (128, 128, 128), -1)
        
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        processor.save_debug_image('crop_region_with_padding', vis)

    # Crop the image
    cropped = deskewed_image[min_y:max_y, min_x:max_x]
    
    # Save cropped result
    if processor:
        processor.save_debug_image('cropped_result', cropped)

    if return_analysis:
        # Calculate actual table bounds based on method used
        if crop_method == "cells":
            table_min_x = min(x1 for x1, y1, x2, y2 in cells)
            table_max_x = max(x2 for x1, y1, x2, y2 in cells)
            table_min_y = min(y1 for x1, y1, x2, y2 in cells)
            table_max_y = max(y2 for x1, y1, x2, y2 in cells)
        else:
            table_min_x = min(xs)
            table_max_x = max(xs)
            table_min_y = min(ys)
            table_max_y = max(ys)

        analysis = {
            "cropped": True,
            "crop_method": crop_method,
            "original_shape": deskewed_image.shape,
            "cropped_shape": cropped.shape,
            "crop_bounds": {
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
            },
            "padding_applied": padding,
            "table_bounds": {
                "min_x": table_min_x,
                "max_x": table_max_x,
                "min_y": table_min_y,
                "max_y": table_max_y,
            },
            "size_reduction": {
                "width": deskewed_image.shape[1] - cropped.shape[1],
                "height": deskewed_image.shape[0] - cropped.shape[0],
            },
            "num_cells_used": len(cells) if crop_method == "cells" else 0,
        }
        return cropped, analysis

    return cropped


def visualize_table_structure(
    image: np.ndarray,
    table_structure: Dict,
    cell_color: Tuple[int, int, int] = (0, 0, 255),
    line_color: Tuple[int, int, int] = (0, 255, 0),
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Create visualization of detected table structure.

    Args:
        image: Base image for visualization
        table_structure: Result from detect_table_structure()
        cell_color: Color for cell rectangles (B, G, R)
        line_color: Color for grid lines (B, G, R)
        line_thickness: Thickness of drawn lines

    Returns:
        Image with table structure overlay
    """
    vis = image.copy()

    xs = table_structure.get("xs", [])
    ys = table_structure.get("ys", [])
    cells = table_structure.get("cells", [])

    # Draw grid lines
    height, width = image.shape[:2]
    for x in xs:
        cv2.line(vis, (x, 0), (x, height), line_color, line_thickness)
    for y in ys:
        cv2.line(vis, (0, y), (width, y), line_color, line_thickness)

    # Draw cell rectangles
    for x1, y1, x2, y2 in cells:
        cv2.rectangle(vis, (x1, y1), (x2, y2), cell_color, line_thickness)

    return vis