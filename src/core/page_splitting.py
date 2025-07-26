"""
Page Splitting Module

Intelligently separates double-page book/document scans into individual pages
by detecting the central gutter (binding area) between pages.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, List
import os

from ..utils.config_loader import Config, get_config_value
from ..utils.image_utils import load_image, save_image, get_image_files, convert_to_grayscale
from ..utils.file_utils import get_base_filename, ensure_directory_exists

logger = logging.getLogger(__name__)


class PageSplitter:
    """
    Handles intelligent splitting of double-page scans into individual pages.
    
    Uses gutter detection in the central region of the image to determine
    if the image contains two pages that should be split.
    """
    
    def __init__(self, config: Config):
        """
        Initialize page splitter with configuration.
        
        Args:
            config: Configuration object containing page splitting parameters
        """
        self.config = config
        self.page_splitting = getattr(config, 'page_splitting', {})
        
        # Extract parameters with defaults
        self.gutter_start_percent = self.page_splitting.get('GUTTER_SEARCH_START_PERCENT', 0.4)
        self.gutter_end_percent = self.page_splitting.get('GUTTER_SEARCH_END_PERCENT', 0.6)
        self.split_threshold = self.page_splitting.get('SPLIT_THRESHOLD', 0.8)
        self.left_page_suffix = self.page_splitting.get('LEFT_PAGE_SUFFIX', '_page_2.jpg')
        self.right_page_suffix = self.page_splitting.get('RIGHT_PAGE_SUFFIX', '_page_1.jpg')
        
        logger.info(f"PageSplitter initialized with threshold: {self.split_threshold}")
    
    def detect_gutter(self, image: np.ndarray) -> Tuple[bool, Optional[int]]:
        """
        Detect gutter (binding area) in the central region of the image.
        
        Args:
            image: Input image (color or grayscale)
            
        Returns:
            Tuple of (should_split: bool, gutter_position: Optional[int])
        """
        # Convert to grayscale for analysis
        gray = convert_to_grayscale(image)
        height, width = gray.shape
        
        # Define gutter search region
        search_start = int(width * self.gutter_start_percent)
        search_end = int(width * self.gutter_end_percent)
        
        logger.debug(f"Searching for gutter in region: {search_start}-{search_end} (image width: {width})")
        
        # Calculate vertical projection in the search region
        search_region = gray[:, search_start:search_end]
        vertical_projection = np.sum(search_region, axis=0)
        
        # Normalize projection
        if len(vertical_projection) == 0:
            logger.warning("Empty search region for gutter detection")
            return False, None
        
        normalized_projection = vertical_projection / np.max(vertical_projection)
        
        # Find the minimum (darkest vertical line - potential gutter)
        min_index = np.argmin(normalized_projection)
        min_value = normalized_projection[min_index]
        
        # Calculate gutter confidence
        # A strong gutter should be significantly darker than surrounding areas
        surrounding_mean = np.mean(normalized_projection)
        gutter_confidence = (surrounding_mean - min_value) / surrounding_mean
        
        logger.debug(f"Gutter confidence: {gutter_confidence:.3f} (threshold: {self.split_threshold})")
        
        should_split = gutter_confidence >= self.split_threshold
        gutter_position = search_start + min_index if should_split else None
        
        return should_split, gutter_position
    
    def split_image(self, image: np.ndarray, gutter_position: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split image at the gutter position into left and right pages.
        
        Args:
            image: Input image to split
            gutter_position: X coordinate where to split the image
            
        Returns:
            Tuple of (left_page, right_page) images
        """
        height, width = image.shape[:2]
        
        # Split the image
        left_page = image[:, :gutter_position]
        right_page = image[:, gutter_position:]
        
        logger.debug(f"Split image at position {gutter_position}: "
                    f"left={left_page.shape}, right={right_page.shape}")
        
        return left_page, right_page
    
    def process_single_image(self, image_path: str) -> bool:
        """
        Process a single image file for page splitting.
        
        Args:
            image_path: Path to input image
            
        Returns:
            True if processing was successful
        """
        logger.info(f"Processing image: {os.path.basename(image_path)}")
        
        # Load image
        image = load_image(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return False
        
        # Get base filename for output
        base_filename = get_base_filename(image_path)
        output_dir = self.config.get_directory('splited_images')
        
        # Detect gutter
        should_split, gutter_position = self.detect_gutter(image)
        
        if should_split and gutter_position is not None:
            # Split the image
            left_page, right_page = self.split_image(image, gutter_position)
            
            # Save split pages
            left_path = os.path.join(output_dir, f"{base_filename}{self.left_page_suffix}")
            right_path = os.path.join(output_dir, f"{base_filename}{self.right_page_suffix}")
            
            left_success = save_image(left_page, left_path)
            right_success = save_image(right_page, right_path)
            
            if left_success and right_success:
                logger.info(f"Successfully split '{os.path.basename(image_path)}' into two pages")
                return True
            else:
                logger.error(f"Failed to save split pages for {image_path}")
                return False
        else:
            # Save as single page
            single_path = os.path.join(output_dir, os.path.basename(image_path))
            success = save_image(image, single_path)
            
            if success:
                logger.info(f"Saved '{os.path.basename(image_path)}' as single page")
                return True
            else:
                logger.error(f"Failed to save single page for {image_path}")
                return False
    
    def process_batch(self) -> bool:
        """
        Process all images in the input directory.
        
        Returns:
            True if all images were processed successfully
        """
        input_dir = self.config.get_directory('raw_images')
        output_dir = self.config.get_directory('splited_images')
        
        logger.info(f"Starting batch page splitting: {input_dir} -> {output_dir}")
        
        # Validate input directory
        if not os.path.isdir(input_dir):
            logger.error(f"Input directory not found: {input_dir}")
            return False
        
        # Ensure output directory exists
        if not ensure_directory_exists(output_dir):
            logger.error(f"Failed to create output directory: {output_dir}")
            return False
        
        # Get image files
        image_files = get_image_files(input_dir)
        if not image_files:
            logger.warning(f"No image files found in {input_dir}")
            return True  # Not an error, just nothing to process
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process each image
        successful_count = 0
        for image_file in image_files:
            try:
                if self.process_single_image(image_file):
                    successful_count += 1
                else:
                    logger.warning(f"Failed to process: {os.path.basename(image_file)}")
            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
        
        logger.info(f"Page splitting completed: {successful_count}/{len(image_files)} images processed successfully")
        
        return successful_count == len(image_files)


def run_page_splitting(config_path: str) -> bool:
    """
    Run page splitting process with given configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if successful
    """
    try:
        from ..utils.config_loader import load_config, setup_logging
        
        # Load configuration
        config = load_config(config_path)
        setup_logging(config)
        
        logger.info("Starting page splitting process")
        
        # Create and run page splitter
        splitter = PageSplitter(config)
        success = splitter.process_batch()
        
        if success:
            logger.info("Page splitting completed successfully")
        else:
            logger.error("Page splitting failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in page splitting: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    # Setup basic logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) != 2:
        print("Usage: python page_splitting.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    success = run_page_splitting(config_file)
    sys.exit(0 if success else 1)