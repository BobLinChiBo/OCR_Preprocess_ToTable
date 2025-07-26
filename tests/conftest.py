"""
Pytest configuration and shared fixtures for OCR pipeline tests.

Provides common test fixtures, mock data, and configuration for all test modules.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator
import numpy as np
import cv2

from ocr_pipeline.config import Config, get_default_config
from ocr_pipeline.utils.logging_utils import setup_logging


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample test image."""
    # Create a simple test image with some structure
    image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Add some lines to simulate table structure
    cv2.line(image, (100, 100), (500, 100), (0, 0, 0), 2)  # Horizontal line
    cv2.line(image, (100, 200), (500, 200), (0, 0, 0), 2)  # Horizontal line
    cv2.line(image, (100, 300), (500, 300), (0, 0, 0), 2)  # Horizontal line
    
    cv2.line(image, (150, 50), (150, 350), (0, 0, 0), 2)   # Vertical line
    cv2.line(image, (300, 50), (300, 350), (0, 0, 0), 2)   # Vertical line
    cv2.line(image, (450, 50), (450, 350), (0, 0, 0), 2)   # Vertical line
    
    # Add some text-like noise
    for i in range(10):
        for j in range(20):
            x = 160 + j * 10
            y = 110 + i * 5
            if x < 290 and y < 190:
                cv2.circle(image, (x, y), 1, (0, 0, 0), -1)
    
    return image


@pytest.fixture
def sample_grayscale_image() -> np.ndarray:
    """Create a sample grayscale test image."""
    image = np.ones((300, 400), dtype=np.uint8) * 255
    
    # Add some structure
    cv2.rectangle(image, (50, 50), (350, 250), (0), 2)
    cv2.line(image, (50, 150), (350, 150), (0), 1)
    cv2.line(image, (200, 50), (200, 250), (0), 1)
    
    return image


@pytest.fixture
def sample_binary_image() -> np.ndarray:
    """Create a sample binary test image."""
    image = np.zeros((200, 300), dtype=np.uint8)
    
    # Add white lines on black background
    cv2.line(image, (50, 50), (250, 50), (255), 3)
    cv2.line(image, (50, 100), (250, 100), (255), 3)
    cv2.line(image, (50, 150), (250, 150), (255), 3)
    
    cv2.line(image, (100, 25), (100, 175), (255), 2)
    cv2.line(image, (200, 25), (200, 175), (255), 2)
    
    return image


@pytest.fixture
def sample_config() -> Config:
    """Create a sample configuration for testing."""
    config = get_default_config()
    
    # Override some settings for testing
    config.logging.level = "DEBUG"
    config.logging.use_rich = False  # Disable rich for cleaner test output
    config.line_detection.save_debug_images = False  # Don't save debug images in tests
    
    return config


@pytest.fixture
def sample_config_dict() -> Dict[str, Any]:
    """Create a sample configuration dictionary."""
    return {
        "directories": {
            "raw_images": "test_input",
            "splited_images": "test_output/split",
            "deskewed_images": "test_output/deskewed",
            "lines_images": "test_output/lines",
            "table_images": "test_output/tables",
            "table_fit_images": "test_output/fitted",
            "debug_output_dir": "test_debug"
        },
        "page_splitting": {
            "gutter_search_start_percent": 0.4,
            "gutter_search_end_percent": 0.6,
            "split_threshold": 0.8
        },
        "deskewing": {
            "angle_range": 5.0,
            "angle_step": 0.5,
            "min_angle_for_correction": 0.5
        },
        "line_detection": {
            "save_debug_images": False,
            "roi_margins_page_1": {
                "top": 50, "bottom": 50, "left": 50, "right": 50
            }
        }
    }


@pytest.fixture
def test_image_files(temp_dir: Path, sample_image: np.ndarray) -> Dict[str, Path]:
    """Create test image files in temporary directory."""
    files = {}
    
    # Create input directory
    input_dir = temp_dir / "input"
    input_dir.mkdir()
    
    # Save sample images
    for i, suffix in enumerate(["_001.jpg", "_002.png", "_003.jpg"]):
        filename = f"test_image{suffix}"
        filepath = input_dir / filename
        
        # Create slightly different images
        test_img = sample_image.copy()
        test_img[10:50, 10:50] = [i * 80, i * 80, i * 80]
        
        cv2.imwrite(str(filepath), test_img)
        files[f"image_{i+1}"] = filepath
    
    return files


@pytest.fixture
def mock_line_detection_result() -> Dict[str, Any]:
    """Create mock line detection result data."""
    return {
        "vertical_lines": [
            {"x1": 100, "y1": 50, "x2": 100, "y2": 350, "confidence": 0.9},
            {"x1": 200, "y1": 50, "x2": 200, "y2": 350, "confidence": 0.85},
            {"x1": 300, "y1": 50, "x2": 300, "y2": 350, "confidence": 0.88},
        ],
        "horizontal_lines": [
            {"x1": 50, "y1": 100, "x2": 350, "y2": 100, "confidence": 0.92},
            {"x1": 50, "y1": 200, "x2": 350, "y2": 200, "confidence": 0.87},
            {"x1": 50, "y1": 300, "x2": 350, "y2": 300, "confidence": 0.90},
        ],
        "image_width": 400,
        "image_height": 400,
        "processing_time": 1.23
    }


@pytest.fixture
def mock_table_structure() -> Dict[str, Any]:
    """Create mock table structure data."""
    return {
        "cells": [
            {"row": 0, "col": 0, "x1": 50, "y1": 50, "x2": 150, "y2": 100},
            {"row": 0, "col": 1, "x1": 150, "y1": 50, "x2": 250, "y2": 100},
            {"row": 0, "col": 2, "x1": 250, "y1": 50, "x2": 350, "y2": 100},
            {"row": 1, "col": 0, "x1": 50, "y1": 100, "x2": 150, "y2": 150},
            {"row": 1, "col": 1, "x1": 150, "y1": 100, "x2": 250, "y2": 150},
            {"row": 1, "col": 2, "x1": 250, "y1": 100, "x2": 350, "y2": 150},
        ],
        "rows": 2,
        "cols": 3,
        "table_bounds": {"x1": 50, "y1": 50, "x2": 350, "y2": 150}
    }


@pytest.fixture(autouse=True)
def setup_test_logging(sample_config: Config):
    """Setup logging for tests."""
    # Configure minimal logging for tests
    setup_logging(
        level="WARNING",  # Only show warnings and errors in tests
        use_rich=False,   # Disable rich formatting for cleaner test output
        include_performance=False,  # Disable performance logging
        format_style="minimal"
    )


@pytest.fixture
def roi_margins() -> Dict[str, int]:
    """Sample ROI margins for testing."""
    return {
        "top": 50,
        "bottom": 50,
        "left": 30,
        "right": 30
    }


@pytest.fixture
def morphological_params() -> Dict[str, float]:
    """Sample morphological parameters for testing."""
    return {
        "morph_open_kernel_ratio": 0.02,
        "morph_close_kernel_ratio": 0.015,
        "hough_threshold": 8,
        "hough_min_line_length": 50,
        "cluster_distance_threshold": 10,
        "qualify_length_ratio": 0.6
    }


# Marks for different test categories
pytestmark = pytest.mark.filterwarnings("ignore:.*:DeprecationWarning")


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (deselect with '-m \"not unit\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add unit marker to tests in unit directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to tests in integration directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to tests with "slow" in the name
        if "slow" in item.name.lower():
            item.add_marker(pytest.mark.slow)


# Helper functions for tests
def assert_image_similar(img1: np.ndarray, img2: np.ndarray, threshold: float = 0.95):
    """Assert that two images are similar within a threshold."""
    if img1.shape != img2.shape:
        raise AssertionError(f"Image shapes don't match: {img1.shape} vs {img2.shape}")
    
    # Calculate correlation coefficient
    correlation = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)[0][0]
    
    if correlation < threshold:
        raise AssertionError(f"Images not similar enough: {correlation} < {threshold}")


def create_test_image_with_lines(
    width: int = 400, 
    height: int = 300, 
    num_h_lines: int = 3, 
    num_v_lines: int = 4
) -> np.ndarray:
    """Create a test image with specified number of lines."""
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add horizontal lines
    for i in range(num_h_lines):
        y = int((i + 1) * height / (num_h_lines + 1))
        cv2.line(image, (50, y), (width - 50, y), (0, 0, 0), 2)
    
    # Add vertical lines  
    for i in range(num_v_lines):
        x = int((i + 1) * width / (num_v_lines + 1))
        cv2.line(image, (x, 50), (x, height - 50), (0, 0, 0), 2)
    
    return image