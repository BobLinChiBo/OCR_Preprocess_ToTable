"""Basic tests for OCR pipeline."""

from pathlib import Path

import numpy as np
import pytest

from src.ocr_pipeline.config import Config, get_default_config
from src.ocr_pipeline.pipeline import OCRPipeline
from src.ocr_pipeline.processors import (
    deskew_image,
    detect_table_lines,
    split_two_page_image,
)


def test_default_config():
    """Test default configuration creation."""
    config = get_default_config()
    assert isinstance(config, Config)
    assert config.input_dir == Path("data/input")
    assert config.output_dir == Path("data/output")


def test_config_path_conversion():
    """Test that string paths are converted to Path objects."""
    config = Config(input_dir="test_input", output_dir="test_output")
    assert isinstance(config.input_dir, Path)
    assert isinstance(config.output_dir, Path)


def test_split_two_page_image():
    """Test two-page image splitting."""
    # Create a simple test image (white with black gutter in middle)
    image = np.ones((100, 200, 3), dtype=np.uint8) * 255
    # Add black gutter in the middle
    image[:, 95:105] = 0

    left, right = split_two_page_image(image)

    # Check that images were split
    assert left.shape[1] < image.shape[1]
    assert right.shape[1] < image.shape[1]
    assert left.shape[1] + right.shape[1] <= image.shape[1]


def test_deskew_image():
    """Test image deskewing."""
    # Create a simple test image
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255

    # Deskewing should return an image of the same size
    deskewed, angle = deskew_image(image)
    assert deskewed.shape == image.shape


def test_detect_table_lines():
    """Test table line detection."""
    # Create a simple image with lines
    image = np.ones((100, 100), dtype=np.uint8) * 255

    # Add some horizontal and vertical lines
    image[25, :] = 0  # Horizontal line
    image[:, 25] = 0  # Vertical line

    h_lines, v_lines = detect_table_lines(image)

    # Should detect some lines (exact count depends on parameters)
    assert isinstance(h_lines, list)
    assert isinstance(v_lines, list)


def test_pipeline_initialization():
    """Test pipeline initialization."""
    pipeline = OCRPipeline()
    assert isinstance(pipeline.config, Config)

    # Test with custom config
    custom_config = Config(verbose=True)
    pipeline2 = OCRPipeline(custom_config)
    assert pipeline2.config.verbose is True


def test_pipeline_with_nonexistent_directory():
    """Test pipeline with non-existent input directory."""
    config = Config(input_dir="nonexistent_directory")
    pipeline = OCRPipeline(config)

    with pytest.raises(ValueError, match="Input directory does not exist"):
        pipeline.process_directory()


"""
TODO: These tests need to be updated to work with the new processor architecture.
The following functions no longer exist in the refactored code:
- detect_roi_gabor
- find_vertical_cuts  
- find_horizontal_cuts
- detect_roi_for_image
- crop_to_roi

def test_detect_roi_gabor():
    # Test Gabor filter ROI detection.
    # Create a test image with edge-like structures
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255

    # Add some edge structures
    image[20:80, 20:25] = 0  # Vertical edge
    image[20:25, 20:80] = 0  # Horizontal edge

    binary_mask = detect_roi_gabor(image)

    assert binary_mask.shape == (100, 100)
    assert binary_mask.dtype == np.uint8
    assert np.max(binary_mask) <= 255


# def test_find_vertical_cuts():
    # Test vertical cut detection.
    # Create a binary mask with content on left and right
    binary_mask = np.zeros((100, 100), dtype=np.uint8)
    binary_mask[:, 10:30] = 255  # Left content
    binary_mask[:, 70:90] = 255  # Right content

    left, right, info = find_vertical_cuts(binary_mask, mode="single_best")

    assert isinstance(left, int)
    assert isinstance(right, int)
    assert isinstance(info, dict)
    assert 0 <= left <= right <= 100


# def test_find_horizontal_cuts():
    # Test horizontal cut detection.
    # Create a binary mask with content on top and bottom
    binary_mask = np.zeros((100, 100), dtype=np.uint8)
    binary_mask[10:30, :] = 255  # Top content
    binary_mask[70:90, :] = 255  # Bottom content

    top, bottom, info = find_horizontal_cuts(binary_mask, mode="single_best")

    assert isinstance(top, int)
    assert isinstance(bottom, int)
    assert isinstance(info, dict)
    assert 0 <= top <= bottom <= 100


# def test_detect_roi_for_image():
    # Test full ROI detection pipeline.
    # Create a test image
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255

    # Add some content in a specific region
    image[30:70, 30:70] = 0

    config = get_default_config()
    roi_coords = detect_roi_for_image(image, config)

    assert isinstance(roi_coords, dict)
    assert "roi_left" in roi_coords
    assert "roi_right" in roi_coords
    assert "roi_top" in roi_coords
    assert "roi_bottom" in roi_coords
    assert "image_width" in roi_coords
    assert "image_height" in roi_coords

    # Check coordinate validity
    assert 0 <= roi_coords["roi_left"] <= roi_coords["roi_right"] <= 100
    assert 0 <= roi_coords["roi_top"] <= roi_coords["roi_bottom"] <= 100
    assert roi_coords["image_width"] == 100
    assert roi_coords["image_height"] == 100


# def test_crop_to_roi():
    # Test ROI cropping.
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255

    roi_coords = {"roi_left": 20, "roi_right": 80, "roi_top": 30, "roi_bottom": 70}

    cropped = crop_to_roi(image, roi_coords)

    expected_height = roi_coords["roi_bottom"] - roi_coords["roi_top"]
    expected_width = roi_coords["roi_right"] - roi_coords["roi_left"]

    assert cropped.shape == (expected_height, expected_width, 3)


# def test_roi_detection_config():
    # Test that ROI detection configuration is included.
    config = get_default_config()

    # Check that ROI detection parameters exist
    assert hasattr(config, "enable_roi_detection")
    assert hasattr(config, "gabor_kernel_size")
    assert hasattr(config, "gabor_sigma")
    assert hasattr(config, "roi_vertical_mode")
    assert hasattr(config, "roi_horizontal_mode")

    # Check default values
    assert config.enable_roi_detection is True
    assert config.gabor_kernel_size == 31
    assert config.roi_vertical_mode == "single_best"
"""


if __name__ == "__main__":
    pytest.main([__file__])
