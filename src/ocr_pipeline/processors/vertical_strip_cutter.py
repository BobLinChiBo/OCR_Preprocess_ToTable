"""Vertical strip cutter processor for cutting images into columns based on table structure."""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from PIL import Image
import cv2

from .base import BaseProcessor


class VerticalStripCutterProcessor(BaseProcessor):
    """Processor for cutting images into vertical strips based on table structure."""

    def _maybe_rescale(self, xs: List[int], img_width: int, tolerance: float = 0.05) -> List[int]:
        """If max(xs) differs from img_width by > tolerance, rescale xs."""
        if not xs:
            return xs
            
        max_xs = max(xs)
        if max_xs == 0:
            return xs
            
        if max_xs > img_width:
            if abs(max_xs - img_width) / max_xs > tolerance:
                scale = img_width / max_xs
                xs = [int(round(x * scale)) for x in xs]
                if self.config and self.config.verbose:
                    print(f"    Rescaled xs coordinates by factor {scale:.3f}")
        
        return xs

    def process(
        self,
        image: np.ndarray,
        structure_data: Dict[str, Any] = None,
        structure_json_path: str = None,
        padding: int = 20,
        min_width: int = 1,
        output_dir: Path = None,
        base_name: str = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Cut image into vertical strips based on table structure.

        Args:
            image: Input image (numpy array, BGR format)
            structure_data: Dictionary with 'xs' array defining column boundaries
            structure_json_path: Path to JSON file containing structure data
            padding: Horizontal padding in pixels to add to each strip
            min_width: Minimum column width to keep (skip very thin strips)
            output_dir: Directory to save strip images
            base_name: Base name for output files

        Returns:
            Dictionary with:
                - strips: List of dictionaries with strip info (image, bounds, path)
                - num_strips: Number of strips created
                - xs: The (possibly rescaled) x coordinates used
        """
        # Get structure data
        if structure_json_path:
            with open(structure_json_path, 'r', encoding='utf-8') as f:
                structure_data = json.load(f)
        
        if structure_data is None:
            raise ValueError("Either structure_data or structure_json_path must be provided")
        
        xs = structure_data.get("xs", [])
        if not xs or len(xs) < 2:
            raise ValueError("Structure data must contain 'xs' array with at least 2 values")
        
        # Convert image to PIL format for easier manipulation
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Convert BGR to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        w, h = pil_image.size
        
        # Maybe rescale xs to match image width
        xs = self._maybe_rescale(xs, w)
        
        strips = []
        
        # Cut strips between consecutive x coordinates
        for i in range(len(xs) - 1):
            left = max(xs[i] - padding, 0)
            right = min(xs[i + 1] + padding, w)
            
            # Skip very thin strips
            if right - left < min_width:
                if self.config and self.config.verbose:
                    print(f"    Skipping thin strip {i+1}: width={right-left}px < {min_width}px")
                continue
            
            # Crop the strip
            strip_pil = pil_image.crop((left, 0, right, h))
            
            # Convert back to numpy array (RGB)
            strip_np = np.array(strip_pil)
            
            # Convert RGB back to BGR for OpenCV compatibility
            strip_bgr = cv2.cvtColor(strip_np, cv2.COLOR_RGB2BGR)
            
            strip_info = {
                "image": strip_bgr,
                "column_index": i + 1,
                "bounds": {
                    "left": left,
                    "right": right,
                    "original_left": xs[i],
                    "original_right": xs[i + 1],
                    "padding": padding
                }
            }
            
            # Save strip if output directory provided
            if output_dir and base_name:
                output_dir.mkdir(parents=True, exist_ok=True)
                strip_filename = f"{base_name}_col{i+1:02d}.png"
                strip_path = output_dir / strip_filename
                
                # Save using cv2 to maintain BGR format
                cv2.imwrite(str(strip_path), strip_bgr)
                strip_info["path"] = strip_path
                
                if self.config and self.config.verbose:
                    print(f"    Saved strip {i+1}: {strip_filename} ({right-left}x{h}px)")
            
            strips.append(strip_info)
        
        return {
            "strips": strips,
            "num_strips": len(strips),
            "xs": xs,
            "image_size": (w, h)
        }


def cut_vertical_strips(
    image: np.ndarray,
    structure_data: Dict[str, Any] = None,
    structure_json_path: str = None,
    padding: int = 20,
    min_width: int = 1,
    use_longest_lines_only: bool = False,
    min_length_ratio: float = 0.9,
    v_lines: List[Tuple[int, int, int, int]] = None,
    output_dir: Path = None,
    base_name: str = None,
    verbose: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function for cutting image into vertical strips.

    Args:
        image: Input image (numpy array, BGR format)
        structure_data: Dictionary with 'xs' array defining column boundaries
        structure_json_path: Path to JSON file containing structure data
        padding: Horizontal padding in pixels to add to each strip
        min_width: Minimum column width to keep (skip very thin strips)
        use_longest_lines_only: If True, use only the longest vertical lines
        min_length_ratio: Minimum length ratio relative to longest line (0.0-1.0)
        v_lines: List of vertical lines (x1, y1, x2, y2) for filtering
        output_dir: Directory to save strip images
        base_name: Base name for output files
        verbose: Whether to print progress messages

    Returns:
        Dictionary with strip information
    """
    from ..config import Config
    
    # Create a minimal config for verbose output
    config = Config(verbose=verbose) if verbose else None
    
    # If we need to filter lines and v_lines is provided
    if use_longest_lines_only and v_lines is not None and len(v_lines) > 0:
        # Calculate line lengths
        line_lengths = []
        for x1, y1, x2, y2 in v_lines:
            length = abs(y2 - y1)  # Vertical line length
            line_lengths.append((length, (x1, y1, x2, y2)))
        
        # Find the maximum length
        max_length = max(line_lengths, key=lambda x: x[0])[0]
        
        # Filter lines based on min_length_ratio
        min_length = max_length * min_length_ratio
        filtered_lines = []
        
        for length, line in line_lengths:
            if length >= min_length:
                filtered_lines.append(line)
        
        if verbose:
            print(f"  Filtering vertical lines: {len(v_lines)} -> {len(filtered_lines)} lines")
            print(f"  Using lines >= {min_length_ratio:.0%} of max length ({max_length}px)")
        
        # Extract unique x positions from filtered lines
        x_positions = []
        for x1, y1, x2, y2 in filtered_lines:
            x_positions.append(x1)
        
        # Sort and remove duplicates (with small tolerance)
        x_positions = sorted(set(x_positions))
        
        # Create structure_data with filtered xs
        if structure_data is None:
            structure_data = {}
        structure_data['xs'] = x_positions
        
        if verbose:
            print(f"  Extracted {len(x_positions)} unique x-positions from filtered lines")
    
    processor = VerticalStripCutterProcessor(config=config)
    return processor.process(
        image=image,
        structure_data=structure_data,
        structure_json_path=structure_json_path,
        padding=padding,
        min_width=min_width,
        output_dir=output_dir,
        base_name=base_name,
        **kwargs,
    )