"""Image deskewing operations for OCR pipeline."""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
from .base import BaseProcessor


class DeskewProcessor(BaseProcessor):
    """Processor for deskewing images."""
    
    def process(
        self,
        image: np.ndarray,
        angle_range: int = 15,
        angle_step: float = 0.5,
        min_angle_correction: float = 0.5,
        return_analysis_data: bool = False,
        **kwargs
    ) -> tuple:
        """Deskew an image using histogram variance optimization.
        
        Args:
            image: Input image to deskew
            angle_range: Maximum rotation angle in degrees (±)
            angle_step: Step size for fine search in degrees
            min_angle_correction: Minimum angle threshold to apply correction
            return_analysis_data: If True, return additional analysis data
            
        Returns:
            tuple: (deskewed_image, detected_angle) or (deskewed_image, detected_angle, analysis_data)
        """
        self.validate_image(image)
        
        # Clear previous debug images
        self.clear_debug_images()
        
        # Pass processor instance for debug saving
        kwargs['_processor'] = self
        
        return deskew_image(
            image,
            angle_range=angle_range,
            angle_step=angle_step,
            min_angle_correction=min_angle_correction,
            return_analysis_data=return_analysis_data,
            **kwargs
        )


def deskew_image(
    image: np.ndarray,
    angle_range: int = 15,
    angle_step: float = 0.5,
    min_angle_correction: float = 0.5,
    return_analysis_data: bool = False,
    **kwargs
) -> tuple:
    """Deskew image using optimized coarse-to-fine histogram variance optimization.

    Performance optimizations:
    - Coarse-to-fine search: reduces iterations from ~100 to ~40
    - Multi-resolution: coarse search on downsampled image (16x faster)
    - Fast interpolation for search phase, high quality for final rotation
    - Early termination when improvement is minimal

    Philosophy: Well-aligned text creates sharp peaks in horizontal projection.
    This approach maximizes the variance of adjacent histogram differences,
    which occurs when text lines are perfectly horizontal.

    Args:
        image: Input image to deskew
        angle_range: Maximum rotation angle in degrees (±)
        angle_step: Step size for fine search in degrees
        min_angle_correction: Minimum angle threshold to apply correction
        return_analysis_data: If True, return additional analysis data for visualization
    Returns:
        tuple: (deskewed_image, detected_angle) or (deskewed_image, detected_angle, analysis_data)
    """
    # Get processor instance if available for debug saving
    processor = kwargs.get('_processor', None)
    
    # Convert to binary for optimal histogram analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Save debug images
    if processor:
        processor.save_debug_image('gray_input', gray)
        processor.save_debug_image('binary_threshold', binary)

    def histogram_variance_score(
        binary_img: np.ndarray, angle: float, use_fast_interpolation: bool = True
    ) -> float:
        """Calculate sharpness of horizontal projection after rotation.
        When text is well-aligned, the horizontal projection shows sharp peaks
        (text rows) and valleys (white space). This maximizes the variance of
        adjacent differences in the projection histogram.
        """
        h, w = binary_img.shape
        center = (w // 2, h // 2)
        # Rotate binary image - use fast interpolation for search phase
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        interpolation = cv2.INTER_LINEAR if use_fast_interpolation else cv2.INTER_CUBIC
        rotated = cv2.warpAffine(
            binary_img,
            rotation_matrix,
            (w, h),
            flags=interpolation,
            borderMode=cv2.BORDER_REPLICATE,
        )
        # Calculate horizontal projection (sum of pixels in each row)
        horizontal_projection = np.sum(rotated, axis=1)
        # Calculate variance of adjacent differences (sharpness measure)
        # Sharp transitions between text and whitespace maximize this value
        differences = horizontal_projection[1:] - horizontal_projection[:-1]
        return np.sum(differences**2)

    # Phase 1: Coarse search on downsampled image for speed
    # Downsample by 4x for ~16x speedup (4x4 pixels -> 1 pixel)
    h, w = binary.shape
    small_binary = cv2.resize(binary, (w // 4, h // 4), interpolation=cv2.INTER_AREA)

    # Coarse search with 1-degree steps
    coarse_step = 1.0
    coarse_angles = np.arange(-angle_range, angle_range + coarse_step, coarse_step)
    coarse_scores = []

    for angle in coarse_angles:
        score = histogram_variance_score(
            small_binary, angle, use_fast_interpolation=True
        )
        coarse_scores.append(score)

        # Early termination: if we have enough samples and last few scores are decreasing
        if len(coarse_scores) >= 5:
            recent_scores = coarse_scores[-5:]
            if all(
                recent_scores[i] >= recent_scores[i + 1]
                for i in range(len(recent_scores) - 1)
            ):
                # Scores are consistently decreasing, likely past the optimum
                break

    # Find best coarse angle
    best_coarse_idx = np.argmax(coarse_scores)
    best_coarse_angle = coarse_angles[best_coarse_idx]

    # Phase 2: Fine search around best coarse candidate on full resolution
    # Search ±2 degrees around best coarse angle with original step size
    fine_range = 2.0
    fine_start = max(-angle_range, best_coarse_angle - fine_range)
    fine_end = min(angle_range, best_coarse_angle + fine_range)
    fine_angles = np.arange(fine_start, fine_end + angle_step, angle_step)

    fine_scores = []
    for angle in fine_angles:
        score = histogram_variance_score(binary, angle, use_fast_interpolation=True)
        fine_scores.append(score)

    # Find best fine angle
    best_fine_idx = np.argmax(fine_scores)
    best_angle = fine_angles[best_fine_idx]
    
    # Create angle histogram visualization
    if processor:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Coarse search plot
        ax1.plot(coarse_angles[:len(coarse_scores)], coarse_scores, 'b-', label='Coarse Search')
        ax1.axvline(x=best_coarse_angle, color='r', linestyle='--', label=f'Best Coarse: {best_coarse_angle:.1f}°')
        ax1.set_xlabel('Angle (degrees)')
        ax1.set_ylabel('Variance Score')
        ax1.set_title('Coarse Angle Search')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Fine search plot
        ax2.plot(fine_angles, fine_scores, 'g-', label='Fine Search')
        ax2.axvline(x=best_angle, color='r', linestyle='--', label=f'Best Angle: {best_angle:.2f}°')
        ax2.set_xlabel('Angle (degrees)')
        ax2.set_ylabel('Variance Score')
        ax2.set_title('Fine Angle Search')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert plot to image
        fig.canvas.draw()
        # Use the correct method name for getting the buffer
        plot_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert RGBA to BGR
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
        processor.save_debug_image('angle_histogram', plot_img)
        plt.close(fig)

    # Apply rotation only if significant
    if abs(best_angle) < min_angle_correction:
        if return_analysis_data:
            # Create analysis data even when no rotation is applied
            all_angles = list(coarse_angles[: len(coarse_scores)]) + list(fine_angles)
            all_scores = coarse_scores + fine_scores
            analysis_data = {
                "has_lines": True,
                "rotation_angle": 0.0,
                "angles": all_angles,
                "scores": all_scores,
                "coarse_angles": list(coarse_angles[: len(coarse_scores)]),
                "coarse_scores": coarse_scores,
                "fine_angles": list(fine_angles),
                "fine_scores": fine_scores,
                "best_score": max(all_scores) if all_scores else 0,
                "confidence": 1.0,
                "will_rotate": False,
                "method": "histogram_variance",
                "gray": gray,
                "binary": binary,
                "line_count": len(all_angles),
                "angle_std": 0.0,
            }
            return image, 0.0, analysis_data
        return image, 0.0

    # Apply optimal rotation to original color image with high quality interpolation
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    deskewed = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,  # High quality for final result
        borderMode=cv2.BORDER_REPLICATE,
    )
    
    # Create rotation comparison visualization
    if processor:
        # Create side-by-side comparison
        comparison = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
        comparison[:, :w] = image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        comparison[:, w+20:] = deskewed if len(deskewed.shape) == 3 else cv2.cvtColor(deskewed, cv2.COLOR_GRAY2BGR)
        
        # Add divider line
        comparison[:, w:w+20] = 255
        
        # Add text labels
        cv2.putText(comparison, f"Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, f"Deskewed ({best_angle:.2f} deg)", (w+30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        processor.save_debug_image('rotation_comparison', comparison)

    if return_analysis_data:
        # Create comprehensive analysis data for visualization
        all_angles = list(coarse_angles[: len(coarse_scores)]) + list(fine_angles)
        all_scores = coarse_scores + fine_scores

        # Calculate confidence based on score distribution
        score_std = np.std(all_scores)
        best_score = max(all_scores) if all_scores else 0
        confidence = min(1.0, best_score / (np.mean(all_scores) + score_std + 1e-6))

        analysis_data = {
            "has_lines": True,
            "rotation_angle": best_angle,
            "angles": all_angles,
            "scores": all_scores,
            "coarse_angles": list(coarse_angles[: len(coarse_scores)]),
            "coarse_scores": coarse_scores,
            "fine_angles": list(fine_angles),
            "fine_scores": fine_scores,
            "best_score": best_score,
            "confidence": confidence,
            "angle_std": score_std,
            "will_rotate": True,
            "method": "histogram_variance",
            "gray": gray,
            "binary": binary,
            "line_count": len(all_angles),
        }
        return deskewed, best_angle, analysis_data

    return deskewed, best_angle