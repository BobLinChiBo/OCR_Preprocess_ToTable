# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import json

# --- INTERNAL DEFAULT CONFIGURATION ---
# This class is used ONLY if 'config.json' is not found.
class Config:
    INPUT_DIR = "raw_images"
    OUTPUT_DIR = "splited_images"
    GUTTER_SEARCH_START_PERCENT = 0.40
    GUTTER_SEARCH_END_PERCENT = 0.60
    SPLIT_THRESHOLD = 0.8
    LEFT_PAGE_SUFFIX = "_page_2.jpg"
    RIGHT_PAGE_SUFFIX = "_page_1.jpg"

def load_config_or_use_class(config_path='config.json'):
    """Loads config from JSON if it exists, otherwise uses the internal Config class."""
    config = Config() # Start with internal defaults
    if os.path.exists(config_path):
        print(f"Loading configuration from '{config_path}'...")
        with open(config_path, 'r', encoding='utf-8') as f:
            external_config = json.load(f)
        
        # Override defaults with values from JSON
        dirs = external_config.get('directories', {})
        params = external_config.get('page_splitting', {})
        config.INPUT_DIR = dirs.get('raw_images', config.INPUT_DIR)
        config.OUTPUT_DIR = dirs.get('splited_images', config.OUTPUT_DIR)
        config.GUTTER_SEARCH_START_PERCENT = params.get('GUTTER_SEARCH_START_PERCENT', config.GUTTER_SEARCH_START_PERCENT)
        config.GUTTER_SEARCH_END_PERCENT = params.get('GUTTER_SEARCH_END_PERCENT', config.GUTTER_SEARCH_END_PERCENT)
        config.SPLIT_THRESHOLD = params.get('SPLIT_THRESHOLD', config.SPLIT_THRESHOLD)
        config.LEFT_PAGE_SUFFIX = params.get('LEFT_PAGE_SUFFIX', config.LEFT_PAGE_SUFFIX)
        config.RIGHT_PAGE_SUFFIX = params.get('RIGHT_PAGE_SUFFIX', config.RIGHT_PAGE_SUFFIX)
    else:
        print("External 'config.json' not found. Using internal default configuration.")
    return config

def split_image_batch_intelligent(config):
    """Processes a batch of images to intelligently split two-page spreads."""
    if not os.path.isdir(config.INPUT_DIR):
        print(f"Error: Input directory '{config.INPUT_DIR}' not found.")
        return
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    image_files = [f for f in os.listdir(config.INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"No image files found in '{config.INPUT_DIR}'.")
        return

    print("\n--- Starting Step 1: Intelligent Page Splitting ---")
    for filename in image_files:
        image_path = os.path.join(config.INPUT_DIR, filename)
        image = cv2.imread(image_path)
        if image is None: continue

        h, w, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        vertical_projection = np.sum(gray, axis=0)
        center_start = int(w * config.GUTTER_SEARCH_START_PERCENT)
        center_end = int(w * config.GUTTER_SEARCH_END_PERCENT)
        center_projection = vertical_projection[center_start:center_end]

        if len(center_projection) == 0:
            output_path = os.path.join(config.OUTPUT_DIR, filename)
            cv2.imwrite(output_path, image)
            continue

        split_x_relative = np.argmin(center_projection)
        split_x_absolute = center_start + split_x_relative
        avg_projection_value = np.mean(vertical_projection)
        gutter_value = vertical_projection[split_x_absolute]

        if gutter_value < avg_projection_value * config.SPLIT_THRESHOLD:
            print(f"  - Processing '{filename}': Two-page spread detected.")
            left_page, right_page = image[:, :split_x_absolute], image[:, split_x_absolute:]
            base_filename = os.path.splitext(filename)[0]
            left_path = os.path.join(config.OUTPUT_DIR, f"{base_filename}{config.LEFT_PAGE_SUFFIX}")
            right_path = os.path.join(config.OUTPUT_DIR, f"{base_filename}{config.RIGHT_PAGE_SUFFIX}")
            cv2.imwrite(left_path, left_page); cv2.imwrite(right_path, right_page)
        else:
            print(f"  - Processing '{filename}': Single page detected.")
            cv2.imwrite(os.path.join(config.OUTPUT_DIR, filename), image)

    print(f"--- Step 1 Complete ---")

if __name__ == '__main__':
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
    config = load_config_or_use_class(config_path)
    split_image_batch_intelligent(config)