"""Page splitting functionality for two-page scanned images."""

from typing import Dict, Tuple, Any, Optional
import cv2
import numpy as np
from .base import BaseProcessor


class PageSplitProcessor(BaseProcessor):
    """Processor for splitting two-page scanned images."""
    
    def process(
        self,
        image: np.ndarray,
        search_ratio: float = 0.5,
        line_len_frac: float = 0.3,
        line_thick: int = 3,
        peak_thr: float = 0.3,
        return_analysis: bool = False,
        **kwargs
    ) -> tuple:
        """Split a two-page scanned image into separate pages.
        
        Args:
            image: Input two-page image
            search_ratio: Fraction of width, centered, to search for gutter (0.0-1.0)
            line_len_frac: Vertical kernel = fraction of image height
            line_thick: Kernel width in pixels
            peak_thr: Peak threshold (fraction of max response)
            return_analysis: If True, returns detailed analysis information
            
        Returns:
            tuple: (right_page, left_page) or (right_page, left_page, analysis) if return_analysis=True
        """
        self.validate_image(image)
        
        # Clear previous debug images
        self.clear_debug_images()
        
        # Pass processor instance for debug saving
        kwargs['_processor'] = self
        
        return split_two_page_image(
            image,
            search_ratio=search_ratio,
            line_len_frac=line_len_frac,
            line_thick=line_thick,
            peak_thr=peak_thr,
            return_analysis=return_analysis,
            **kwargs
        )


def split_two_page_image(
    image: np.ndarray,
    search_ratio: float = 0.5,
    line_len_frac: float = 0.3,
    line_thick: int = 3,
    peak_thr: float = 0.3,
    return_analysis: bool = False,
    **kwargs
) -> tuple:
    """Split a two-page scanned image into separate pages using V2 algorithm.

    Args:
        image: Input two-page image
        search_ratio: Fraction of width, centered, to search for gutter (0.0-1.0)
        line_len_frac: Vertical kernel = fraction of image height
        line_thick: Kernel width in pixels
        peak_thr: Peak threshold (fraction of max response)
        return_analysis: If True, returns detailed analysis information

    Returns:
        tuple: (right_page, left_page) or (right_page, left_page, analysis) if return_analysis=True
    """
    # Get processor instance if available for debug saving
    processor = kwargs.get('_processor', None)
    
    h, w = image.shape[:2]
    
    # Save original input image
    if processor and processor.config and getattr(processor.config, 'save_debug_images', False):
        processor.save_debug_image('input_image', image)
    
    # 1) extract long vertical lines
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, bw  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    vert   = 255 - bw
    k_len  = max(15, int(line_len_frac * h))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_thick, k_len))
    lines  = cv2.morphologyEx(vert, cv2.MORPH_OPEN, kernel)
    
    # Save debug images
    if processor and processor.config and getattr(processor.config, 'save_debug_images', False):
        processor.save_debug_image('binary_threshold', bw)
        processor.save_debug_image('inverted_binary', vert)
        processor.save_debug_image('vertical_lines', lines)

    # 2) column response & search window
    col_sum = lines.sum(axis=0)
    centre  = w // 2
    margin  = int((1 - search_ratio) / 2 * w)
    window  = col_sum[margin:w - margin]
    
    # Create column response visualization
    if processor and processor.config and getattr(processor.config, 'save_debug_images', False):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(col_sum, 'b-', linewidth=1)
        ax.axvline(margin, color='r', linestyle='--', label=f'Search start ({margin})')
        ax.axvline(w - margin, color='r', linestyle='--', label=f'Search end ({w-margin})')
        ax.axvline(centre, color='g', linestyle='-', alpha=0.7, label=f'Center ({centre})')
        ax.set_xlabel('Column Index')
        ax.set_ylabel('Sum of White Pixels')
        ax.set_title('Column Response (Vertical Line Density)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Convert plot to image
        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        
        processor.save_debug_image('column_response', plot_img)

    peaks = np.where(window >= peak_thr * window.max())[0] + margin
    if len(peaks) < 2:                     # fallback if only one border found
        split_x = centre
        left_page = image[:, :split_x]
        right_page = image[:, split_x:]
        
        # Debug: Print the result only in debug mode
        if processor and processor.config and getattr(processor.config, 'save_debug_images', False):
            print(f"    [DEBUG] Page split: Center fallback at x={split_x} (no vertical lines detected)")
        
        # Save split visualization even for fallback
        if processor and processor.config and getattr(processor.config, 'save_debug_images', False):
            vis = image.copy() if len(image.shape) == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.line(vis, (split_x, 0), (split_x, h), (0, 0, 255), 3)
            cv2.putText(vis, 'FALLBACK: Center Split', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            processor.save_debug_image('split_visualization', vis)
        
        if not return_analysis:
            return right_page, left_page
            
        # Build analysis for fallback case
        analysis = {
            "gutter_x": split_x,
            "gutter_strength": 0.0,
            "gutter_width": 0,
            "search_start": margin,
            "search_end": w - margin,
            "segments": [],
            "selected_segment": None,
            "fallback_used": True,
            "meets_min_width": False,
            "has_two_pages": False,
            "col_dark": None,
            "darkmask": None,
            "thresh": None,
            "widest_segment_width": 0,
            # Legacy compatibility fields
            "vertical_sums": None,
            "min_sum": 0,
            "avg_sum": 0,
        }
        return right_page, left_page, analysis

    # 3) contiguous-peak grouping
    segments, s = [], peaks[0]
    for p, q in zip(peaks, peaks[1:]):
        if q != p + 1:
            segments.append((s, p)); s = q
    segments.append((s, peaks[-1]))

    # 4) choose the two segments nearest the centre
    segs = sorted(segments, key=lambda ab: abs(((ab[0]+ab[1])//2) - centre))[:2]
    if len(segs) < 2:  # If only one segment found, split at that vertical line
        if len(segs) == 1 and segments:
            # Use the single detected vertical line as the split point
            seg_start, seg_end = segs[0]
            split_x = (seg_start + seg_end) // 2
            segment_width = seg_end - seg_start + 1
            
            # Determine if this is likely a gutter or edge based on position
            # If within middle 60% of image, likely a gutter; otherwise might be edge
            is_likely_gutter = margin < split_x < (w - margin)
            
            left_page = image[:, :split_x]
            right_page = image[:, split_x:]
            
            # Save split visualization for single segment
            if processor and processor.config and getattr(processor.config, 'save_debug_images', False):
                vis = image.copy() if len(image.shape) == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                # Draw the detected segment
                cv2.rectangle(vis, (seg_start, 0), (seg_end, h), (0, 255, 0), 2)
                # Draw the split line
                cv2.line(vis, (split_x, 0), (split_x, h), (0, 0, 255), 3)
                split_type = 'Single Vertical Line Split' if is_likely_gutter else 'Edge Detection Split'
                cv2.putText(vis, split_type, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(vis, f'Split at: {split_x}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                processor.save_debug_image('split_visualization', vis)
            
            if not return_analysis:
                return right_page, left_page
                
            # Build analysis for single segment case with actual detected values
            analysis = {
                "gutter_x": split_x,
                "gutter_strength": window[split_x - margin] / window.max() if (margin <= split_x < w - margin and window.max() > 0) else 0.5,
                "gutter_width": segment_width,
                "search_start": margin,
                "search_end": w - margin,
                "segments": segments,
                "selected_segment": segs[0],
                "fallback_used": False,  # We're using the detected line, not fallback
                "meets_min_width": segment_width >= 3,
                "has_two_pages": is_likely_gutter,
                "col_dark": None,
                "darkmask": None,
                "thresh": None,
                "widest_segment_width": max(b - a + 1 for a, b in segments) if segments else 0,
                # Legacy compatibility fields
                "vertical_sums": None,
                "min_sum": 0,
                "avg_sum": 0,
            }
            return right_page, left_page, analysis
        else:
            # No segments at all, use center fallback
            split_x = centre
            left_page = image[:, :split_x]
            right_page = image[:, split_x:]
            
            # Save split visualization for no segments case
            if processor and processor.config and getattr(processor.config, 'save_debug_images', False):
                vis = image.copy() if len(image.shape) == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.line(vis, (split_x, 0), (split_x, h), (0, 0, 255), 3)
                cv2.putText(vis, 'FALLBACK: No Segments Found', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                processor.save_debug_image('split_visualization', vis)
            
            if not return_analysis:
                return right_page, left_page
                
            # Build analysis for fallback case
            analysis = {
                "gutter_x": split_x,
                "gutter_strength": 0.0,
                "gutter_width": 0,
                "search_start": margin,
                "search_end": w - margin,
                "segments": [],
                "selected_segment": None,
                "fallback_used": True,
                "meets_min_width": False,
                "has_two_pages": False,
                "col_dark": None,
                "darkmask": None,
                "thresh": None,
                "widest_segment_width": 0,
                # Legacy compatibility fields
                "vertical_sums": None,
                "min_sum": 0,
                "avg_sum": 0,
            }
            return right_page, left_page, analysis
    
    (a1, b1), (a2, b2) = sorted(segs, key=lambda ab: ab[0])
    split_x = (b1 + a2) // 2
    
    left_page = image[:, :split_x]
    right_page = image[:, split_x:]
    
    # Debug: Print the result only in debug mode
    if processor and processor.config and getattr(processor.config, 'save_debug_images', False):
        gutter_width = a2 - b1
        print(f"    [DEBUG] Page split: Gutter at x={split_x} (width: {gutter_width}px, {len(segments)} vertical lines detected)")
    
    # Save final split visualization
    if processor and processor.config and getattr(processor.config, 'save_debug_images', False):
        vis = image.copy() if len(image.shape) == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Draw search window
        overlay = vis.copy()
        cv2.rectangle(overlay, (margin, 0), (w-margin, h), (255, 255, 0), -1)
        vis = cv2.addWeighted(vis, 0.85, overlay, 0.15, 0)
        
        # Draw detected segments
        for a, b in segments:
            cv2.rectangle(vis, (a, 0), (b, h), (0, 255, 0), 2)
        
        # Draw selected segments in blue
        cv2.rectangle(vis, (a1, 0), (b1, h), (255, 0, 0), 3)
        cv2.rectangle(vis, (a2, 0), (b2, h), (255, 0, 0), 3)
        
        # Draw final split line
        cv2.line(vis, (split_x, 0), (split_x, h), (0, 0, 255), 3)
        
        # Add labels
        cv2.putText(vis, f'Gutter: {split_x}', (split_x + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(vis, 'Left Page', (50, h-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis, 'Right Page', (split_x + 50, h-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        processor.save_debug_image('split_visualization', vis)
    
    if not return_analysis:
        return right_page, left_page
    
    # Build full analysis dict for compatibility
    analysis = {
        "gutter_x": split_x,
        "gutter_strength": 1.0,  # V2 doesn't calculate this exactly
        "gutter_width": a2 - b1,
        "search_start": margin,
        "search_end": w - margin,
        "segments": segments,
        "selected_segment": (b1, a2),
        "fallback_used": False,
        "meets_min_width": True,
        "has_two_pages": True,
        "col_dark": None,
        "darkmask": None,
        "thresh": None,
        "widest_segment_width": max(b - a + 1 for a, b in segments) if segments else 0,
        # Legacy compatibility fields
        "vertical_sums": None,
        "min_sum": 0,
        "avg_sum": 0,
    }
    
    return right_page, left_page, analysis