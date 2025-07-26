"""Simple utilities for OCR pipeline."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple


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
    h_lines = [tuple(line[0]) for line in (h_lines or [])]
    v_lines = [tuple(line[0]) for line in (v_lines or [])]
    
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