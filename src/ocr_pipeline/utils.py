"""Simple utilities for OCR pipeline."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def load_image(image_path: Path) -> np.ndarray:
    """Load image from file."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image


def save_image(image: np.ndarray, output_path: Path) -> None:
    """Save image to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from directory."""
    extensions = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
    image_files = []
    
    for ext in extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(image_files)


def split_two_page_image(image: np.ndarray, gutter_start: float = 0.4, 
                        gutter_end: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
    """Split a two-page scanned image into separate pages."""
    height, width = image.shape[:2]
    
    # Search for gutter in the middle region
    search_start = int(width * gutter_start)
    search_end = int(width * gutter_end)
    search_region = image[:, search_start:search_end]
    
    # Find the darkest vertical line (gutter)
    gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY) if len(search_region.shape) == 3 else search_region
    vertical_sums = np.sum(gray, axis=0)
    gutter_offset = np.argmin(vertical_sums)
    gutter_x = search_start + gutter_offset
    
    # Split the image
    left_page = image[:, :gutter_x]
    right_page = image[:, gutter_x:]
    
    return left_page, right_page


def deskew_image(image: np.ndarray, angle_range: int = 45, 
                angle_step: float = 0.5) -> np.ndarray:
    """Deskew image by finding optimal rotation angle."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Find edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is None:
        return image
    
    # Calculate angles of detected lines
    angles = []
    for rho, theta in lines[:, 0]:
        angle = theta * 180 / np.pi
        # Convert to -45 to 45 degree range
        if angle > 90:
            angle = angle - 180
        elif angle > 45:
            angle = angle - 90
        elif angle < -45:
            angle = angle + 90
        angles.append(angle)
    
    # Find the most common angle
    if not angles:
        return image
    
    # Use median angle for rotation
    rotation_angle = np.median(angles)
    
    # Only rotate if angle is significant
    if abs(rotation_angle) < 0.5:
        return image
    
    # Rotate image
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                           flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated


def detect_table_lines(image: np.ndarray, min_line_length: int = 100, 
                      max_line_gap: int = 10) -> Tuple[List[Tuple], List[Tuple]]:
    """Detect horizontal and vertical lines in image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Apply morphological operations to enhance lines
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    
    # Detect horizontal lines
    horizontal = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_h)
    h_lines = cv2.HoughLinesP(horizontal, 1, np.pi/180, threshold=50,
                             minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    # Detect vertical lines
    vertical = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_v)
    v_lines = cv2.HoughLinesP(vertical, 1, np.pi/180, threshold=50,
                             minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    # Convert to list of tuples
    h_lines = [tuple(line[0]) for line in h_lines] if h_lines is not None else []
    v_lines = [tuple(line[0]) for line in v_lines] if v_lines is not None else []
    
    return h_lines, v_lines


def crop_table_region(image: np.ndarray, h_lines: List[Tuple], 
                     v_lines: List[Tuple]) -> np.ndarray:
    """Crop image to table region based on detected lines."""
    if not h_lines or not v_lines:
        return image
    
    height, width = image.shape[:2]
    
    # Find boundaries
    min_x = min([min(x1, x2) for x1, y1, x2, y2 in v_lines])
    max_x = max([max(x1, x2) for x1, y1, x2, y2 in v_lines])
    min_y = min([min(y1, y2) for x1, y1, x2, y2 in h_lines])
    max_y = max([max(y1, y2) for x1, y1, x2, y2 in h_lines])
    
    # Add some padding
    padding = 10
    min_x = max(0, min_x - padding)
    max_x = min(width, max_x + padding)
    min_y = max(0, min_y - padding)
    max_y = min(height, max_y + padding)
    
    # Crop the image
    cropped = image[min_y:max_y, min_x:max_x]
    return cropped


def find_vertical_cuts(binary_mask: np.ndarray, mode: str = 'single_best', 
                      window_size_divisor: int = 20, min_window_size: int = 50,
                      min_cut_strength: float = 10.0, min_confidence_threshold: float = 5.0) -> Tuple[int, int, Dict]:
    """Find vertical cuts in the binary mask using sliding window approach."""
    height, width = binary_mask.shape
    vertical_projection = np.sum(binary_mask // 255, axis=0)
    
    window_size = max(width // window_size_divisor, min_window_size)
    
    # Find left cut
    max_drop_left = 0
    cut_x_left = 0
    for i in range(window_size, width // 2):
#        left_avg = np.mean(vertical_projection[i-window_size:i])
#        right_avg = np.mean(vertical_projection[i:i+window_size]) if i+window_size < width else 0
#        drop = abs(left_avg - right_avg)
        drop = abs(vertical_projection[i] - vertical_projection[i+window_size]) if i + window_size < width else 0
        if drop > max_drop_left:
            max_drop_left = drop
            cut_x_left = i+window_size
    
    # Find right cut
    max_drop_right = 0
    cut_x_right = width
    for i in range(width - window_size, width // 2, -1):
#        right_avg = np.mean(vertical_projection[i:i+window_size])
#        left_avg = np.mean(vertical_projection[i-window_size:i]) if i-window_size >= 0 else 0
#        drop = abs(right_avg - left_avg)
        drop = abs(vertical_projection[i] - vertical_projection[i-window_size]) if i + window_size < width else 0
        if drop > max_drop_right:
            max_drop_right = drop
            cut_x_right = i-window_size

    # Evaluate whether cuts should be applied
    apply_left_cut = (max_drop_left >= min_cut_strength and max_drop_left >= min_confidence_threshold)
    apply_right_cut = (max_drop_right >= min_cut_strength and max_drop_right >= min_confidence_threshold)
    
    # Decide what to return based on mode
    if mode == 'both_sides':
        final_left = cut_x_left if apply_left_cut else 0
        final_right = cut_x_right if apply_right_cut else width
        return final_left, final_right, {
            "projection": vertical_projection,
            "left_cut_strength": max_drop_left,
            "right_cut_strength": max_drop_right,
            "left_cut_applied": apply_left_cut,
            "right_cut_applied": apply_right_cut
        }

    # For single_best mode, choose the stronger cut
    midpoint = width // 2
    left_density = np.sum(vertical_projection[:midpoint])
    right_density = np.sum(vertical_projection[midpoint:])
    
    if right_density < left_density:
        # Prefer right cut
        final_right = cut_x_right if apply_right_cut else width
        return 0, final_right, {
            "projection": vertical_projection,
            "right_cut_strength": max_drop_right,
            "right_cut_applied": apply_right_cut,
            "cut_side": "right"
        }
    else:
        # Prefer left cut
        final_left = cut_x_left if apply_left_cut else 0
        return final_left, width, {
            "projection": vertical_projection,
            "left_cut_strength": max_drop_left,
            "left_cut_applied": apply_left_cut,
            "cut_side": "left"
        }


def find_horizontal_cuts(binary_mask: np.ndarray, mode: str = 'single_best',
                        window_size_divisor: int = 20, min_window_size: int = 50,
                        min_cut_strength: float = 10.0, min_confidence_threshold: float = 5.0) -> Tuple[int, int, Dict]:
    """Find horizontal cuts in the binary mask using sliding window approach."""
    height, width = binary_mask.shape
    horizontal_projection = np.sum(binary_mask // 255, axis=1)
    
    window_size = max(height // window_size_divisor, min_window_size)
    
    # Find top cut
    max_drop_top = 0
    cut_y_top = 0
    for i in range(window_size, height // 2):
#        top_avg = np.mean(horizontal_projection[i-window_size:i])
#        bottom_avg = np.mean(horizontal_projection[i:i+window_size]) if i+window_size < height else 0
#        drop = top_avg - bottom_avg
         drop = abs(horizontal_projection[i] - horizontal_projection[i+window_size]) if i + window_size < height else 0
         if drop > max_drop_top:
            max_drop_top = drop
            cut_y_top = i+window_size
    
    # Find bottom cut
    max_drop_bottom = 0
    cut_y_bottom = height
    for i in range(height - window_size, height // 2, -1):
#        bottom_avg = np.mean(horizontal_projection[i:i+window_size])
#        top_avg = np.mean(horizontal_projection[i-window_size:i]) if i-window_size >= 0 else 0
#        drop = bottom_avg - top_avg
         drop = abs(horizontal_projection[i] - horizontal_projection[i-window_size]) if i + window_size < height else 0
         if drop > max_drop_bottom:
            max_drop_bottom = drop
            cut_y_bottom = i-window_size

    # Evaluate whether cuts should be applied
    apply_top_cut = (max_drop_top >= min_cut_strength and max_drop_top >= min_confidence_threshold)
    apply_bottom_cut = (max_drop_bottom >= min_cut_strength and max_drop_bottom >= min_confidence_threshold)

    if mode == 'both_sides':
        final_top = cut_y_top if apply_top_cut else 0
        final_bottom = cut_y_bottom if apply_bottom_cut else height
        return final_top, final_bottom, {
            "projection": horizontal_projection,
            "top_cut_strength": max_drop_top,
            "bottom_cut_strength": max_drop_bottom,
            "top_cut_applied": apply_top_cut,
            "bottom_cut_applied": apply_bottom_cut
        }

    # For single_best mode, choose the stronger cut
    midpoint_y = height // 2
    top_density = np.sum(horizontal_projection[:midpoint_y])
    bottom_density = np.sum(horizontal_projection[midpoint_y:])

    if bottom_density < top_density:
        # Prefer bottom cut
        final_bottom = cut_y_bottom if apply_bottom_cut else height
        return 0, final_bottom, {
            "projection": horizontal_projection,
            "bottom_cut_strength": max_drop_bottom,
            "bottom_cut_applied": apply_bottom_cut,
            "cut_side": "bottom"
        }
    else:
        # Prefer top cut
        final_top = cut_y_top if apply_top_cut else 0
        return final_top, height, {
            "projection": horizontal_projection,
            "top_cut_strength": max_drop_top,
            "top_cut_applied": apply_top_cut,
            "cut_side": "top"
        }


def detect_roi_gabor(image: np.ndarray, kernel_size: int = 31, sigma: float = 4.0, 
                    lambda_val: float = 10.0, gamma: float = 0.5, 
                    binary_threshold: int = 127) -> np.ndarray:
    """Apply Gabor filters to detect edge structures."""
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Create Gabor kernels for vertical and horizontal orientations only
    kernels = []
    for theta in [0, np.pi/2]:  # 0° (vertical), 90° (horizontal)
        kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, float(theta), 
                                   lambda_val, gamma, 0, ktype=cv2.CV_32F)
        kernels.append(kernel)

    # Apply all kernels and combine responses
    combined_response = np.zeros_like(gray_img, dtype=np.float32)
    for kernel in kernels:
        filtered_img = cv2.filter2D(gray_img, cv2.CV_8UC3, kernel)
        combined_response += filtered_img.astype(np.float32)

    # Normalize and threshold
    gabor_response_map = cv2.normalize(combined_response, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, binary_mask = cv2.threshold(gabor_response_map, binary_threshold, 255, cv2.THRESH_BINARY)
    
    return binary_mask


def detect_roi_for_image(image: np.ndarray, config) -> Dict:
    """Detect ROI for an image using Gabor filters and sliding window analysis."""
    # Apply Gabor filtering
    binary_mask = detect_roi_gabor(
        image, 
        config.gabor_kernel_size,
        config.gabor_sigma,
        config.gabor_lambda,
        config.gabor_gamma,
        config.gabor_binary_threshold
    )
    
    # Find cuts
    cut_x_left, cut_x_right, vertical_info = find_vertical_cuts(
        binary_mask, 
        mode=config.roi_vertical_mode,
        window_size_divisor=config.roi_window_size_divisor,
        min_window_size=config.roi_min_window_size,
        min_cut_strength=config.roi_min_cut_strength,
        min_confidence_threshold=config.roi_min_confidence_threshold
    )
    
    cut_y_top, cut_y_bottom, horizontal_info = find_horizontal_cuts(
        binary_mask,
        mode=config.roi_horizontal_mode,
        window_size_divisor=config.roi_window_size_divisor,
        min_window_size=config.roi_min_window_size,
        min_cut_strength=config.roi_min_cut_strength,
        min_confidence_threshold=config.roi_min_confidence_threshold
    )
    
    return {
        "roi_left": int(cut_x_left),
        "roi_right": int(cut_x_right),
        "roi_top": int(cut_y_top),
        "roi_bottom": int(cut_y_bottom),
        "image_width": image.shape[1],
        "image_height": image.shape[0]
    }


def crop_to_roi(image: np.ndarray, roi_coords: Dict) -> np.ndarray:
    """Crop image to ROI coordinates."""
    left = roi_coords["roi_left"]
    right = roi_coords["roi_right"]
    top = roi_coords["roi_top"]
    bottom = roi_coords["roi_bottom"]
    
    return image[top:bottom, left:right]