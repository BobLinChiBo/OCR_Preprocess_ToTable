"""Image I/O utilities for loading and saving images."""

from pathlib import Path
from typing import List

import cv2
import numpy as np


def load_image(image_path: Path) -> np.ndarray:
    """Load image from file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        numpy array containing the image
        
    Raises:
        ValueError: If image cannot be loaded
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image


def save_image(image: np.ndarray, output_path: Path) -> None:
    """Save image to file.
    
    Args:
        image: Image array to save
        output_path: Path where to save the image
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from directory.
    
    Args:
        directory: Directory to search for images
        
    Returns:
        List of paths to image files, sorted
    """
    extensions = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
    image_files = []

    for ext in extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))

    return sorted(image_files)