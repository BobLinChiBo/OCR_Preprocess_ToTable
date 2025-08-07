"""Image deskewing operations for OCR pipeline."""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
from .base import BaseProcessor

try:
    from skimage import transform
    from skimage.transform import radon, rotate
    from skimage.feature import canny
    from skimage import img_as_float
    RADON_AVAILABLE = True
except ImportError:
    RADON_AVAILABLE = False
    
try:
    from deskew import determine_skew
    DESKEW_LIBRARY_AVAILABLE = True
except ImportError:
    DESKEW_LIBRARY_AVAILABLE = False


class DeskewProcessor(BaseProcessor):
    """Processor for deskewing images."""
    
    @staticmethod
    def get_available_methods():
        """Get list of available deskewing methods based on installed dependencies."""
        methods = ['histogram_variance']  # Always available
        if RADON_AVAILABLE:
            methods.append('radon')
        if DESKEW_LIBRARY_AVAILABLE:
            methods.append('deskew_library')
        return methods
    
    @staticmethod
    def validate_and_fallback_method(method: str) -> str:
        """Validate the requested method and provide fallback if unavailable.
        
        Args:
            method: Requested deskewing method
            
        Returns:
            str: Valid method to use (original or fallback)
        """
        available_methods = DeskewProcessor.get_available_methods()
        
        if method in available_methods:
            return method
        
        # Provide fallback with warning
        fallback_method = 'histogram_variance'  # Always available
        
        # Construct warning message
        if method == 'radon' and not RADON_AVAILABLE:
            print(f"Warning: Radon method requested but scikit-image is not installed. "
                  f"Falling back to '{fallback_method}' method. "
                  f"Install scikit-image with: pip install scikit-image>=0.19.0")
        elif method == 'deskew_library' and not DESKEW_LIBRARY_AVAILABLE:
            print(f"Warning: Deskew library method requested but deskew is not installed. "
                  f"Falling back to '{fallback_method}' method. "
                  f"Install deskew with: pip install deskew")
        else:
            print(f"Warning: Unknown deskewing method '{method}'. "
                  f"Falling back to '{fallback_method}' method. "
                  f"Available methods: {', '.join(available_methods)}")
        
        return fallback_method
    
    def process(
        self,
        image: np.ndarray,
        method: str = "histogram_variance",
        angle_range: int = 15,
        angle_step: float = 0.5,
        coarse_range: float = None,
        coarse_step: float = 1.0,
        fine_range: float = 2.0,
        fine_step: float = None,
        min_angle_correction: float = 0.5,
        return_analysis_data: bool = False,
        **kwargs
    ) -> tuple:
        """Deskew an image using the specified method.
        
        Args:
            image: Input image to deskew
            method: Deskewing method ("radon", "histogram_variance", or "deskew_library")
                   - "radon": Uses Canny edge detection + Radon transform (recommended)
                   - "histogram_variance": Optimized coarse-to-fine histogram variance
                   - "deskew_library": Legacy method using deskew library
            angle_range: Maximum rotation angle in degrees (±) - legacy parameter, use coarse_range instead
            angle_step: Step size for fine search in degrees - legacy parameter, use fine_step instead
            coarse_range: Coarse search range in degrees (±). If None, uses angle_range
            coarse_step: Step size for coarse search in degrees (default: 1.0 for histogram, 0.5 for radon)
            fine_range: Fine search range around best coarse angle in degrees (±) (default: 2.0)
            fine_step: Step size for fine search in degrees. If None, uses angle_step (default: 0.5 for histogram, 0.2 for radon)
            min_angle_correction: Minimum angle threshold to apply correction
            return_analysis_data: If True, return additional analysis data
            
        Returns:
            tuple: (deskewed_image, detected_angle) or (deskewed_image, detected_angle, analysis_data)
        """
        self.validate_image(image)
        
        # Clear previous debug images
        self.clear_debug_images()
        
        # Validate method and provide fallback if necessary
        method = self.validate_and_fallback_method(method)
        
        # Pass processor instance for debug saving
        kwargs['_processor'] = self
        
        # Handle legacy parameters for backward compatibility
        if coarse_range is None:
            coarse_range = angle_range
        if fine_step is None:
            fine_step = angle_step
        
        if method == "radon":
            # Double-check availability (should already be validated)
            if not RADON_AVAILABLE:
                # This should not happen due to validation, but add safety check
                print("Error: Radon method selected but scikit-image is not available. Using histogram_variance instead.")
                return deskew_image(
                    image,
                    coarse_range=coarse_range,
                    coarse_step=coarse_step,
                    fine_range=fine_range,
                    fine_step=fine_step if fine_step is not None else 0.5,
                    min_angle_correction=min_angle_correction,
                    return_analysis_data=return_analysis_data,
                    **kwargs
                )
            return radon_method(
                image,
                coarse_range=coarse_range,
                coarse_step=coarse_step if coarse_step != 1.0 else 0.5,  # Default 0.5 for radon
                fine_range=fine_range,
                fine_step=fine_step if fine_step is not None else 0.2,  # Default 0.2 for radon
                min_angle_correction=min_angle_correction,
                return_analysis_data=return_analysis_data,
                **kwargs
            )
        elif method == "deskew_library":
            # Legacy method kept for backward compatibility
            if not DESKEW_LIBRARY_AVAILABLE:
                # This should not happen due to validation, but add safety check
                print("Error: Deskew library method selected but deskew is not available. Using histogram_variance instead.")
                return deskew_image(
                    image,
                    coarse_range=coarse_range,
                    coarse_step=coarse_step,
                    fine_range=fine_range,
                    fine_step=fine_step if fine_step is not None else 0.5,
                    min_angle_correction=min_angle_correction,
                    return_analysis_data=return_analysis_data,
                    **kwargs
                )
            return deskew_library_method(
                image,
                min_angle_correction=min_angle_correction,
                return_analysis_data=return_analysis_data,
                **kwargs
            )
        elif method == "histogram_variance":
            return deskew_image(
                image,
                coarse_range=coarse_range,
                coarse_step=coarse_step,
                fine_range=fine_range,
                fine_step=fine_step if fine_step is not None else 0.5,  # Default 0.5 for histogram
                min_angle_correction=min_angle_correction,
                return_analysis_data=return_analysis_data,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown deskewing method: {method}. Available methods: 'radon', 'histogram_variance', 'deskew_library'")


def deskew_image(
    image: np.ndarray,
    coarse_range: float = 15,
    coarse_step: float = 1.0,
    fine_range: float = 2.0,
    fine_step: float = 0.5,
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
        coarse_range: Coarse search range in degrees (±)
        coarse_step: Step size for coarse search in degrees
        fine_range: Fine search range around best coarse angle in degrees (±)
        fine_step: Step size for fine search in degrees
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

    # Coarse search with configurable range and step
    coarse_angles = np.arange(-coarse_range, coarse_range + coarse_step, coarse_step)
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
    # Search within fine_range around best coarse angle with fine_step
    fine_start = max(-coarse_range, best_coarse_angle - fine_range)
    fine_end = min(coarse_range, best_coarse_angle + fine_range)
    fine_angles = np.arange(fine_start, fine_end + fine_step, fine_step)

    fine_scores = []
    for angle in fine_angles:
        score = histogram_variance_score(binary, angle, use_fast_interpolation=True)
        fine_scores.append(score)

    # Find best fine angle
    best_fine_idx = np.argmax(fine_scores)
    best_angle = fine_angles[best_fine_idx]
    
    # Debug: Print the result only in debug mode
    if processor and processor.config and getattr(processor.config, 'save_debug_images', False):
        print(f"    [DEBUG] Histogram deskew: Detected angle: {best_angle:.2f}° (coarse: {best_coarse_angle:.1f}°, searched ±{coarse_range}°)")
    
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


def radon_method(
    image: np.ndarray,
    coarse_range: float = 5.0,
    coarse_step: float = 0.5,
    fine_range: float = 1.0,
    fine_step: float = 0.2,
    min_angle_correction: float = 0.5,
    return_analysis_data: bool = False,
    **kwargs
) -> tuple:
    """Deskew image using Radon transform on edge-detected image with coarse-to-fine search.
    
    This method uses Canny edge detection followed by Radon transform
    to find the optimal rotation angle for text alignment.
    
    Args:
        image: Input image to deskew
        coarse_range: Coarse search range in degrees (±)
        coarse_step: Step size for coarse search in degrees
        fine_range: Fine search range around best coarse angle in degrees (±)
        fine_step: Step size for fine search in degrees
        min_angle_correction: Minimum angle threshold to apply correction
        return_analysis_data: If True, return additional analysis data
        
    Returns:
        tuple: (deskewed_image, detected_angle) or (deskewed_image, detected_angle, analysis_data)
    """
    if not RADON_AVAILABLE:
        raise ImportError("scikit-image is not available. Install with: pip install scikit-image")
    
    # Get processor instance if available for debug saving
    processor = kwargs.get('_processor', None)
    
    # Convert to grayscale for skew detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Save debug images
    if processor:
        processor.save_debug_image('gray_input', gray)
    
    # 1. Edge detection (Canny gives clean lines for Radon)
    edges = canny(img_as_float(gray), sigma=1.0)
    
    # Save edge detection debug image
    if processor:
        edges_vis = (edges * 255).astype(np.uint8)
        processor.save_debug_image('canny_edges', edges_vis)
    
    # 2. Coarse Radon scan
    coarse_angles = np.arange(-coarse_range, coarse_range + coarse_step, coarse_step)
    coarse_scores = []
    
    # Will print debug result after finding best angle
    
    for a in coarse_angles:
        sinogram = radon(edges, [a], circle=False)
        # Higher variance == more energy in fewer bins -> better alignment
        coarse_scores.append(np.var(sinogram))
    
    # Find best coarse angle
    best_coarse_idx = int(np.argmax(coarse_scores))
    best_coarse_angle = coarse_angles[best_coarse_idx]
    
    # 3. Fine Radon scan around best coarse angle
    fine_start = max(-coarse_range, best_coarse_angle - fine_range)
    fine_end = min(coarse_range, best_coarse_angle + fine_range)
    fine_angles = np.arange(fine_start, fine_end + fine_step, fine_step)
    fine_scores = []
    
    for a in fine_angles:
        sinogram = radon(edges, [a], circle=False)
        fine_scores.append(np.var(sinogram))
    
    # Find best fine angle
    best_fine_idx = int(np.argmax(fine_scores))
    best_angle = fine_angles[best_fine_idx]
    
    # Debug: Print the result only in debug mode
    if processor and processor.config and getattr(processor.config, 'save_debug_images', False):
        print(f"    [DEBUG] Radon deskew: Detected angle: {best_angle:.2f}° (coarse: {best_coarse_angle:.1f}°, searched ±{coarse_range}°)")
    
    # Create angle histogram visualization
    if processor:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Coarse search plot
        ax1.plot(coarse_angles, coarse_scores, 'b-', label='Coarse Radon Search')
        ax1.axvline(x=best_coarse_angle, color='r', linestyle='--', label=f'Best Coarse: {best_coarse_angle:.2f}°')
        ax1.set_xlabel('Angle (degrees)')
        ax1.set_ylabel('Variance Score')
        ax1.set_title('Coarse Radon Transform Search')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Fine search plot
        ax2.plot(fine_angles, fine_scores, 'g-', label='Fine Radon Search')
        ax2.axvline(x=best_angle, color='r', linestyle='--', label=f'Best Angle: {best_angle:.2f}°')
        ax2.set_xlabel('Angle (degrees)')
        ax2.set_ylabel('Variance Score')
        ax2.set_title('Fine Radon Transform Search')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert plot to image
        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert RGBA to BGR
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
        processor.save_debug_image('radon_angle_histogram', plot_img)
        plt.close(fig)
    
    # Apply rotation only if significant
    if abs(best_angle) < min_angle_correction:
        if return_analysis_data:
            all_angles = list(coarse_angles) + list(fine_angles)
            all_scores = coarse_scores + fine_scores
            analysis_data = {
                "has_lines": True,
                "rotation_angle": 0.0,
                "angles": all_angles,
                "scores": all_scores,
                "coarse_angles": list(coarse_angles),
                "coarse_scores": coarse_scores,
                "fine_angles": list(fine_angles),
                "fine_scores": fine_scores,
                "best_score": max(all_scores),
                "confidence": 1.0,
                "will_rotate": False,
                "method": "radon",
                "gray": gray,
                "binary": edges_vis if processor else None,
                "line_count": len(all_angles),
                "angle_std": np.std(all_scores),
            }
            return image, 0.0, analysis_data
        return image, 0.0
    
    # 3. Rotate (negative of detected angle to deskew)
    # Convert to float for rotation if needed
    if image.dtype == np.uint8:
        img_for_rotation = image.astype(np.float64) / 255.0
    else:
        img_for_rotation = image.astype(np.float64)
    
    # Apply rotation using scikit-image rotate
    deskewed = rotate(img_for_rotation, -best_angle, resize=False, mode="edge")
    
    # Convert back to original format
    if image.dtype == np.uint8:
        deskewed = (deskewed * 255).astype(np.uint8)
    else:
        deskewed = deskewed.astype(image.dtype)
    
    # Create rotation comparison visualization
    if processor:
        h, w = image.shape[:2]
        # Create side-by-side comparison
        comparison = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
        orig_bgr = image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        desk_bgr = deskewed if len(deskewed.shape) == 3 else cv2.cvtColor(deskewed, cv2.COLOR_GRAY2BGR)
        
        comparison[:, :w] = orig_bgr
        comparison[:, w+20:] = desk_bgr
        
        # Add divider line
        comparison[:, w:w+20] = 255
        
        # Add text labels
        cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, f"Deskewed ({best_angle:.2f} deg)", (w+30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        processor.save_debug_image('rotation_comparison', comparison)
    
    if return_analysis_data:
        # Calculate confidence based on score distribution
        all_angles = list(coarse_angles) + list(fine_angles)
        all_scores = coarse_scores + fine_scores
        score_std = np.std(all_scores)
        best_score = max(all_scores)
        confidence = min(1.0, best_score / (np.mean(all_scores) + score_std + 1e-6))
        
        analysis_data = {
            "has_lines": True,
            "rotation_angle": best_angle,
            "angles": all_angles,
            "scores": all_scores,
            "coarse_angles": list(coarse_angles),
            "coarse_scores": coarse_scores,
            "fine_angles": list(fine_angles),
            "fine_scores": fine_scores,
            "best_score": best_score,
            "confidence": confidence,
            "will_rotate": True,
            "method": "radon",
            "gray": gray,
            "binary": edges_vis if processor else None,
            "line_count": len(all_angles),
            "angle_std": score_std,
        }
        return deskewed, best_angle, analysis_data
    
    return deskewed, best_angle


def deskew_library_method(
    image: np.ndarray,
    min_angle_correction: float = 0.5,
    return_analysis_data: bool = False,
    **kwargs
) -> tuple:
    """Legacy deskew method using the deskew library (moments + Radon hybrid).
    
    This method is kept for backward compatibility.
    
    Args:
        image: Input image to deskew
        min_angle_correction: Minimum angle threshold to apply correction
        return_analysis_data: If True, return additional analysis data
        
    Returns:
        tuple: (deskewed_image, detected_angle) or (deskewed_image, detected_angle, analysis_data)
    """
    if not DESKEW_LIBRARY_AVAILABLE:
        raise ImportError("deskew library is not available. Install with: pip install deskew")
    
    # Get processor instance if available for debug saving
    processor = kwargs.get('_processor', None)
    
    # Convert to grayscale for skew detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Save debug images
    if processor:
        processor.save_debug_image('gray_input', gray)
    
    # Detect skew angle using deskew library
    detected_angle = determine_skew(gray)
    
    # Handle case where no angle is detected (returns None)
    if detected_angle is None:
        detected_angle = 0.0
    
    # Apply rotation only if significant
    if abs(detected_angle) < min_angle_correction:
        if return_analysis_data:
            analysis_data = {
                "has_lines": True,
                "rotation_angle": 0.0,
                "angles": [detected_angle],
                "scores": [1.0],
                "best_score": 1.0,
                "confidence": 1.0,
                "will_rotate": False,
                "method": "deskew_library",
                "gray": gray,
                "binary": None,
                "line_count": 1,
                "angle_std": 0.0,
            }
            return image, 0.0, analysis_data
        return image, 0.0
    
    # Apply rotation using scikit-image for high quality
    # Note: scikit-image expects values in [0,1] for float images
    if image.dtype == np.uint8:
        img_for_rotation = image.astype(np.float64) / 255.0
    else:
        img_for_rotation = image.astype(np.float64)
    
    # Rotate the image (negative angle because deskew library returns clockwise positive)
    rotated = transform.rotate(
        img_for_rotation, 
        -detected_angle, 
        resize=False,  # Keep same size
        mode='edge',   # Fill with edge values
        preserve_range=False  # Output in [0,1] range
    )
    
    # Convert back to original format
    if image.dtype == np.uint8:
        deskewed = (rotated * 255).astype(np.uint8)
    else:
        deskewed = rotated.astype(image.dtype)
    
    # Create rotation comparison visualization
    if processor:
        h, w = image.shape[:2]
        # Create side-by-side comparison
        comparison = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
        orig_bgr = image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        desk_bgr = deskewed if len(deskewed.shape) == 3 else cv2.cvtColor(deskewed, cv2.COLOR_GRAY2BGR)
        
        comparison[:, :w] = orig_bgr
        comparison[:, w+20:] = desk_bgr
        
        # Add divider line
        comparison[:, w:w+20] = 255
        
        # Add text labels
        cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, f"Deskewed ({detected_angle:.2f} deg)", (w+30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        processor.save_debug_image('rotation_comparison', comparison)
    
    if return_analysis_data:
        analysis_data = {
            "has_lines": True,
            "rotation_angle": detected_angle,
            "angles": [detected_angle],
            "scores": [1.0],
            "best_score": 1.0,
            "confidence": 1.0 if detected_angle is not None else 0.0,
            "will_rotate": True,
            "method": "deskew_library",
            "gray": gray,
            "binary": None,
            "line_count": 1,
            "angle_std": 0.0,
        }
        return deskewed, detected_angle, analysis_data
    
    return deskewed, detected_angle