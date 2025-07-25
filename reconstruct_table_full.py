# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import json
import glob

# --- INTERNAL DEFAULT CONFIGURATION ---
class Config:
    LINE_DATA_INPUT_DIR = "lines_images"
    BASE_IMAGE_DIR = "deskewed_images"
    OUTPUT_DIR = "table_images"
    LINE_JSON_SUFFIX = "_lines.json"
    OUTPUT_IMAGE_SUFFIX = "_reconstructed.jpg"
    OUTPUT_DETAILS_SUFFIX = "_details.json"
    VERTICAL_LINE_COLOR = (0, 0, 255)
    HORIZONTAL_LINE_COLOR = (0, 255, 0)
    LINE_THICKNESS = 2

def load_config_or_use_class(config_path='config.json'):
    """Loads config from JSON if it exists, otherwise uses the internal Config class."""
    config = Config()
    if os.path.exists(config_path):
        print(f"Loading configuration from '{config_path}'...")
        with open(config_path, 'r', encoding='utf-8') as f:
            ext = json.load(f)
        
        dirs, params = ext.get('directories', {}), ext.get('table_reconstruction', {})
        config.LINE_DATA_INPUT_DIR = dirs.get('lines_images', config.LINE_DATA_INPUT_DIR)
        config.BASE_IMAGE_DIR = dirs.get('deskewed_images', config.BASE_IMAGE_DIR)
        config.OUTPUT_DIR = dirs.get('table_images', config.OUTPUT_DIR)
        
        config.LINE_JSON_SUFFIX = params.get('LINE_JSON_SUFFIX', config.LINE_JSON_SUFFIX)
        config.OUTPUT_IMAGE_SUFFIX = params.get('OUTPUT_IMAGE_SUFFIX', config.OUTPUT_IMAGE_SUFFIX)
        config.OUTPUT_DETAILS_SUFFIX = params.get('OUTPUT_DETAILS_SUFFIX', config.OUTPUT_DETAILS_SUFFIX)
        config.LINE_THICKNESS = params.get('LINE_THICKNESS', config.LINE_THICKNESS)
        
        if 'VERTICAL_LINE_COLOR' in params: config.VERTICAL_LINE_COLOR = tuple(params['VERTICAL_LINE_COLOR'])
        if 'HORIZONTAL_LINE_COLOR' in params: config.HORIZONTAL_LINE_COLOR = tuple(params['HORIZONTAL_LINE_COLOR'])
    else:
        print("External 'config.json' not found. Using internal default configuration.")
    return config

def reconstruct_table(base_image_path, line_data_path, output_table_path, output_details_path, config):
    """Reconstructs a single table grid from line data and saves the results."""
    try:
        with open(line_data_path, 'r', encoding='utf-8') as f:
            original_line_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Line data file not found at {line_data_path}"); return

    image = cv2.imread(base_image_path)
    if image is None:
        print(f"Error: Could not read base image at {base_image_path}"); return

    vertical_lines_input = original_line_data.get("vertical_lines", [])
    horizontal_lines_input = original_line_data.get("horizontal_lines", [])

    def create_perfect_grid(vertical_lines, horizontal_lines):
        if not vertical_lines or not horizontal_lines: return [], []
        v_coords = sorted([v['avg_coord'] for v in vertical_lines])
        h_coords = sorted([h['avg_coord'] for h in horizontal_lines])
        grid_min_x, grid_max_x = v_coords[0], v_coords[-1]
        grid_min_y, grid_max_y = h_coords[0], h_coords[-1]
        grid_v_lines = [[(int(vx), int(grid_min_y)), (int(vx), int(grid_max_y))] for vx in v_coords]
        grid_h_lines = [[(int(grid_min_x), int(hy)), (int(grid_max_x), int(hy))] for hy in h_coords]
        return grid_v_lines, grid_h_lines

    reconstructed_v_lines, reconstructed_h_lines = create_perfect_grid(vertical_lines_input, horizontal_lines_input)
    
    output_data = {
        "source_image_path": base_image_path,
        "input_line_data_path": line_data_path,
        "original_detected_lines": {"vertical_lines": vertical_lines_input, "horizontal_lines": horizontal_lines_input},
        "reconstructed_grid_lines": {"vertical_lines": reconstructed_v_lines, "horizontal_lines": reconstructed_h_lines}
    }

    with open(output_details_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    output_image_table = image.copy()
    for p1, p2 in reconstructed_v_lines:
        cv2.line(output_image_table, p1, p2, config.VERTICAL_LINE_COLOR, config.LINE_THICKNESS)
    for p1, p2 in reconstructed_h_lines:
        cv2.line(output_image_table, p1, p2, config.HORIZONTAL_LINE_COLOR, config.LINE_THICKNESS)

    cv2.imwrite(output_table_path, output_image_table)
    print(f"  - Successfully processed and saved results for: {os.path.basename(base_image_path)}")

def batch_reconstruct_tables(config):
    """Finds all line data files and processes them in a batch."""
    print("\n--- Starting Step 4: Table Reconstruction ---")
    if not os.path.isdir(config.LINE_DATA_INPUT_DIR) or not os.path.isdir(config.BASE_IMAGE_DIR):
        print(f"Error: Input directories not found."); return
        
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    json_files = glob.glob(os.path.join(config.LINE_DATA_INPUT_DIR, f'*{config.LINE_JSON_SUFFIX}'))

    if not json_files:
        print(f"Error: No '*{config.LINE_JSON_SUFFIX}' files found in '{config.LINE_DATA_INPUT_DIR}'."); return
    
    print(f"Found {len(json_files)} line data file(s) to process.")
    
    for line_data_path in json_files:
        base_name = os.path.basename(line_data_path).replace(config.LINE_JSON_SUFFIX, '')
        image_path_candidates = glob.glob(os.path.join(config.BASE_IMAGE_DIR, f"{base_name}.*"))
        if not image_path_candidates:
            print(f"Warning: Skipping {os.path.basename(line_data_path)}: no matching base image found."); continue
        
        base_image_path = image_path_candidates[0]
        output_table_path = os.path.join(config.OUTPUT_DIR, f"{base_name}{config.OUTPUT_IMAGE_SUFFIX}")
        output_details_path = os.path.join(config.OUTPUT_DIR, f"{base_name}{config.OUTPUT_DETAILS_SUFFIX}")

        reconstruct_table(base_image_path, line_data_path, output_table_path, output_details_path, config)

    print(f"\n--- Step 4 Complete ---")

if __name__ == '__main__':
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
    config = load_config_or_use_class(config_path)
    batch_reconstruct_tables(config)