# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import json
import glob

# --- INTERNAL DEFAULT CONFIGURATION ---
class Config:
    TABLE_DETAILS_INPUT_DIR = "table_images"
    BASE_IMAGE_DIR = "deskewed_images"
    OUTPUT_DIR = "cropped_table_images"
    INPUT_DETAILS_SUFFIX = "_details.json"
    OUTPUT_IMAGE_SUFFIX = "_cropped.jpg"
    OUTPUT_LINES_SUFFIX = "_cropped_lines.json"
    PADDING = 10  # Extra pixels around table boundaries

def load_config_or_use_class(config_path='config.json'):
    """Loads config from JSON if it exists, otherwise uses the internal Config class."""
    config = Config()
    if os.path.exists(config_path):
        print(f"Loading configuration from '{config_path}'...")
        with open(config_path, 'r', encoding='utf-8') as f:
            ext = json.load(f)
        
        dirs = ext.get('directories', {})
        config.TABLE_DETAILS_INPUT_DIR = dirs.get('table_images', config.TABLE_DETAILS_INPUT_DIR)
        config.BASE_IMAGE_DIR = dirs.get('deskewed_images', config.BASE_IMAGE_DIR)
        
        # Add cropping config if it exists in JSON
        crop_params = ext.get('table_cropping', {})
        config.OUTPUT_DIR = crop_params.get('OUTPUT_DIR', config.OUTPUT_DIR)
        config.INPUT_DETAILS_SUFFIX = crop_params.get('INPUT_DETAILS_SUFFIX', config.INPUT_DETAILS_SUFFIX)
        config.OUTPUT_IMAGE_SUFFIX = crop_params.get('OUTPUT_IMAGE_SUFFIX', config.OUTPUT_IMAGE_SUFFIX)
        config.OUTPUT_LINES_SUFFIX = crop_params.get('OUTPUT_LINES_SUFFIX', config.OUTPUT_LINES_SUFFIX)
        config.PADDING = crop_params.get('PADDING', config.PADDING)
    else:
        print("External 'config.json' not found. Using internal default configuration.")
    return config

def get_table_boundaries(reconstructed_lines):
    """Extract table boundaries from reconstructed grid lines."""
    vertical_lines = reconstructed_lines.get("vertical_lines", [])
    horizontal_lines = reconstructed_lines.get("horizontal_lines", [])
    
    if not vertical_lines or not horizontal_lines:
        return None
    
    # Get the outermost coordinates
    min_x = min(line[0][0] for line in vertical_lines)      # leftmost vertical line
    max_x = max(line[0][0] for line in vertical_lines)      # rightmost vertical line
    min_y = min(line[0][1] for line in horizontal_lines)    # topmost horizontal line
    max_y = max(line[0][1] for line in horizontal_lines)    # bottommost horizontal line
    
    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y
    }

def adjust_lines_for_crop(table_data, crop_x1, crop_y1):
    """Adjust line coordinates relative to the cropped image."""
    adjusted_data = table_data.copy()
    
    # Adjust original detected lines
    if "original_detected_lines" in adjusted_data:
        original_lines = adjusted_data["original_detected_lines"]
        
        # Adjust vertical lines (subtract crop_x1 from avg_coord)
        if "vertical_lines" in original_lines:
            for line in original_lines["vertical_lines"]:
                line["avg_coord"] -= crop_x1
        
        # Adjust horizontal lines (subtract crop_y1 from avg_coord)
        if "horizontal_lines" in original_lines:
            for line in original_lines["horizontal_lines"]:
                line["avg_coord"] -= crop_y1
    
    # Adjust reconstructed grid lines
    if "reconstructed_grid_lines" in adjusted_data:
        reconstructed_lines = adjusted_data["reconstructed_grid_lines"]
        
        # Adjust vertical lines (subtract crop_x1 from x-coordinates, crop_y1 from y-coordinates)
        if "vertical_lines" in reconstructed_lines:
            for line in reconstructed_lines["vertical_lines"]:
                for point in line:
                    point[0] -= crop_x1  # x-coordinate
                    point[1] -= crop_y1  # y-coordinate
        
        # Adjust horizontal lines (subtract crop_x1 from x-coordinates, crop_y1 from y-coordinates)
        if "horizontal_lines" in reconstructed_lines:
            for line in reconstructed_lines["horizontal_lines"]:
                for point in line:
                    point[0] -= crop_x1  # x-coordinate
                    point[1] -= crop_y1  # y-coordinate
    
    return adjusted_data

def crop_table_content(base_image_path, table_details_path, output_cropped_path, config):
    """Crops a single image to show only content within table boundaries and saves adjusted line data."""
    try:
        with open(table_details_path, 'r', encoding='utf-8') as f:
            table_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Table details file not found at {table_details_path}")
        return False

    image = cv2.imread(base_image_path)
    if image is None:
        print(f"Error: Could not read base image at {base_image_path}")
        return False

    # Get table boundaries from reconstructed grid lines
    reconstructed_lines = table_data.get("reconstructed_grid_lines", {})
    boundaries = get_table_boundaries(reconstructed_lines)
    
    if boundaries is None:
        print(f"Error: Could not determine table boundaries from {table_details_path}")
        return False

    # Add padding and ensure boundaries are within image limits
    height, width = image.shape[:2]
    
    crop_x1 = max(0, boundaries["min_x"] - config.PADDING)
    crop_y1 = max(0, boundaries["min_y"] - config.PADDING)
    crop_x2 = min(width, boundaries["max_x"] + config.PADDING)
    crop_y2 = min(height, boundaries["max_y"] + config.PADDING)
    
    # Crop the image
    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # Save the cropped image
    cv2.imwrite(output_cropped_path, cropped_image)
    
    # Adjust line coordinates for the cropped image
    adjusted_table_data = adjust_lines_for_crop(table_data, crop_x1, crop_y1)
    
    # Update source image path to point to the cropped image
    adjusted_table_data["source_image_path"] = output_cropped_path
    adjusted_table_data["crop_info"] = {
        "original_image_path": base_image_path,
        "crop_coordinates": {
            "x1": crop_x1,
            "y1": crop_y1,
            "x2": crop_x2,
            "y2": crop_y2
        },
        "original_size": {"width": width, "height": height},
        "cropped_size": {"width": crop_x2-crop_x1, "height": crop_y2-crop_y1}
    }
    
    # Save adjusted line data
    base_name = os.path.splitext(os.path.basename(output_cropped_path))[0]
    if base_name.endswith('_cropped'):
        base_name = base_name[:-8]  # Remove '_cropped' suffix
    output_lines_path = os.path.join(config.OUTPUT_DIR, f"{base_name}{config.OUTPUT_LINES_SUFFIX}")
    
    try:
        with open(output_lines_path, 'w', encoding='utf-8') as f:
            json.dump(adjusted_table_data, f, indent=4, ensure_ascii=False)
        print(f"  - Successfully saved adjusted line data: {os.path.basename(output_lines_path)}")
    except Exception as e:
        print(f"  - Warning: Could not save line data: {e}")
    
    print(f"  - Successfully cropped and saved: {os.path.basename(base_image_path)}")
    print(f"    Original size: {width}x{height}, Cropped size: {crop_x2-crop_x1}x{crop_y2-crop_y1}")
    
    return True

def batch_crop_table_content(config):
    """Finds all table details files and crops corresponding images."""
    print("\n--- Starting Table Content Cropping ---")
    
    if not os.path.isdir(config.TABLE_DETAILS_INPUT_DIR) or not os.path.isdir(config.BASE_IMAGE_DIR):
        print(f"Error: Input directories not found.")
        return
        
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    details_files = glob.glob(os.path.join(config.TABLE_DETAILS_INPUT_DIR, f'*{config.INPUT_DETAILS_SUFFIX}'))

    if not details_files:
        print(f"Error: No '*{config.INPUT_DETAILS_SUFFIX}' files found in '{config.TABLE_DETAILS_INPUT_DIR}'.")
        return
    
    print(f"Found {len(details_files)} table details file(s) to process.")
    
    processed_count = 0
    for details_path in details_files:
        base_name = os.path.basename(details_path).replace(config.INPUT_DETAILS_SUFFIX, '')
        
        # Find corresponding base image
        image_path_candidates = glob.glob(os.path.join(config.BASE_IMAGE_DIR, f"{base_name}.*"))
        if not image_path_candidates:
            print(f"Warning: Skipping {os.path.basename(details_path)}: no matching base image found.")
            continue
        
        base_image_path = image_path_candidates[0]
        output_cropped_path = os.path.join(config.OUTPUT_DIR, f"{base_name}{config.OUTPUT_IMAGE_SUFFIX}")

        if crop_table_content(base_image_path, details_path, output_cropped_path, config):
            processed_count += 1

    print(f"\n--- Table Content Cropping Complete ---")
    print(f"Successfully processed {processed_count} out of {len(details_files)} files.")

if __name__ == '__main__':
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
    config = load_config_or_use_class(config_path)
    batch_crop_table_content(config)