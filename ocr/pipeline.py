"""Main OCR pipeline."""

import argparse
from pathlib import Path
from typing import List

from .config import Config, get_default_config
from .utils import (
    get_image_files, load_image, save_image, split_two_page_image,
    deskew_image, detect_table_lines, crop_table_region,
    detect_roi_for_image, crop_to_roi
)


class OCRPipeline:
    """Simple OCR table extraction pipeline."""
    
    def __init__(self, config: Config = None):
        """Initialize pipeline with configuration."""
        self.config = config or get_default_config()
        self.config.create_output_dirs()
    
    def process_image(self, image_path: Path) -> List[Path]:
        """Process a single image through the pipeline."""
        if self.config.verbose:
            print(f"Processing: {image_path}")
        
        # Load image
        image = load_image(image_path)
        
        # Split into two pages
        left_page, right_page = split_two_page_image(
            image, 
            self.config.gutter_search_start, 
            self.config.gutter_search_end
        )
        
        output_paths = []
        
        # Process each page
        for i, page in enumerate([left_page, right_page], 1):
            # Deskew
            deskewed = deskew_image(
                page, 
                self.config.angle_range, 
                self.config.angle_step
            )
            
            # ROI detection (optional preprocessing step)
            processing_image = deskewed
            roi_coords = None
            if self.config.enable_roi_detection:
                roi_coords = detect_roi_for_image(deskewed, self.config)
                processing_image = crop_to_roi(deskewed, roi_coords)
                
                if self.config.verbose:
                    print(f"    ROI detected: ({roi_coords['roi_left']}, {roi_coords['roi_top']}) to "
                          f"({roi_coords['roi_right']}, {roi_coords['roi_bottom']})")
            
            # Detect table lines
            h_lines, v_lines = detect_table_lines(
                processing_image,
                self.config.min_line_length,
                self.config.max_line_gap
            )
            
            # Crop to table region
            if h_lines and v_lines:
                cropped = crop_table_region(processing_image, h_lines, v_lines)
            else:
                cropped = processing_image
            
            # Save result
            output_name = f"{image_path.stem}_page_{i}.jpg"
            output_path = self.config.output_dir / output_name
            save_image(cropped, output_path)
            output_paths.append(output_path)
            
            if self.config.verbose:
                print(f"  Saved: {output_path}")
        
        return output_paths
    
    def process_directory(self, input_dir: Path = None) -> List[Path]:
        """Process all images in input directory."""
        input_dir = input_dir or self.config.input_dir
        
        if not input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        image_files = get_image_files(input_dir)
        
        if not image_files:
            print(f"No image files found in: {input_dir}")
            return []
        
        print(f"Found {len(image_files)} images to process")
        
        all_outputs = []
        for image_path in image_files:
            try:
                outputs = self.process_image(image_path)
                all_outputs.extend(outputs)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        print(f"Processing complete. {len(all_outputs)} files created.")
        return all_outputs


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="OCR Table Extraction Pipeline")
    parser.add_argument("input", nargs="?", default="input", 
                       help="Input directory or file")
    parser.add_argument("-o", "--output", default="output", 
                       help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("--debug", action="store_true",
                       help="Save debug images")
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        verbose=args.verbose,
        save_debug_images=args.debug
    )
    
    # Run pipeline
    pipeline = OCRPipeline(config)
    
    input_path = Path(args.input)
    if input_path.is_file():
        # Process single file
        outputs = pipeline.process_image(input_path)
        print(f"Processed {input_path} -> {len(outputs)} output files")
    else:
        # Process directory
        outputs = pipeline.process_directory(input_path)
        print(f"Processed {len(outputs)} files total")


if __name__ == "__main__":
    main()