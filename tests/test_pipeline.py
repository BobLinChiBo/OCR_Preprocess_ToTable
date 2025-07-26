"""Basic tests for OCR pipeline."""

import numpy as np
import pytest
from pathlib import Path

from ocr.config import Config, get_default_config
from ocr.utils import split_two_page_image, deskew_image, detect_table_lines
from ocr.pipeline import OCRPipeline


def test_default_config():
    """Test default configuration creation."""
    config = get_default_config()
    assert isinstance(config, Config)
    assert config.input_dir == Path("input")
    assert config.output_dir == Path("output")


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
    deskewed = deskew_image(image)
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


if __name__ == "__main__":
    pytest.main([__file__])