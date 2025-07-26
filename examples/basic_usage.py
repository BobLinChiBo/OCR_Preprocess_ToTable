"""Basic usage example for OCR pipeline."""

from pathlib import Path
from ocr.config import Config
from ocr.pipeline import OCRPipeline

def main():
    """Example of basic pipeline usage."""
    
    # Create a custom configuration
    config = Config(
        input_dir="input",
        output_dir="output", 
        verbose=True,
        min_line_length=50
    )
    
    # Initialize pipeline
    pipeline = OCRPipeline(config)
    
    print("OCR Table Extraction Pipeline - Example")
    print("=" * 40)
    
    # Check if input directory exists
    if not config.input_dir.exists():
        print(f"Creating input directory: {config.input_dir}")
        config.input_dir.mkdir(parents=True, exist_ok=True)
        print("Please add some images to the input directory and run again.")
        return
    
    # Process all images in input directory
    try:
        output_files = pipeline.process_directory()
        print(f"\nSuccess! Created {len(output_files)} output files:")
        for output_file in output_files:
            print(f"  - {output_file}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()