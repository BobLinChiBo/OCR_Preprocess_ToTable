# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import json
import sys

# --- INTERNAL DEFAULT CONFIGURATION ---
class Config:
    INPUT_DIR = "output/stage1_initial_processing/01_split_pages"
    OUTPUT_DIR = "output/stage1_initial_processing/02.5_edge_detection"
    OUTPUT_JSON_SUFFIX = "_roi.json"
    
    # Gabor filter parameters
    GABOR_KERNEL_SIZE = 31
    GABOR_SIGMA = 4.0
    GABOR_LAMBDA = 10.0
    GABOR_GAMMA = 0.5
    BINARY_THRESHOLD = 127
    
    # Cut detection parameters
    VERTICAL_MODE = 'single_best'  # 'both_sides' or 'single_best'
    HORIZONTAL_MODE = 'single_best'  # 'both_sides' or 'single_best'
    
    # Sliding window parameters (as fractions of image dimensions)
    WINDOW_SIZE_DIVISOR = 20
    MIN_WINDOW_SIZE = 50
    MIN_CUT_STRENGTH = 10.0
    MIN_CONFIDENCE_THRESHOLD = 5.0

def load_config(config_path='config.json'):
    """Loads config from JSON if it exists, otherwise uses the internal Config class."""
    config = Config()
    if os.path.exists(config_path):
        print(f"Loading configuration from '{config_path}'...")
        with open(config_path, 'r', encoding='utf-8') as f:
            external_config = json.load(f)
        
        # Override defaults with values from JSON
        edge_detection = external_config.get('edge_detection', {})
        if edge_detection:
            # Directory configuration
            dirs = external_config.get('directories', {})
            config.INPUT_DIR = dirs.get('splited_images', config.INPUT_DIR)
            config.OUTPUT_DIR = edge_detection.get('output_dir', config.OUTPUT_DIR)
            
            # Gabor parameters
            gabor = edge_detection.get('gabor_params', {})
            config.GABOR_KERNEL_SIZE = gabor.get('kernel_size', config.GABOR_KERNEL_SIZE)
            config.GABOR_SIGMA = gabor.get('sigma', config.GABOR_SIGMA)
            config.GABOR_LAMBDA = gabor.get('lambda', config.GABOR_LAMBDA)
            config.GABOR_GAMMA = gabor.get('gamma', config.GABOR_GAMMA)
            config.BINARY_THRESHOLD = gabor.get('binary_threshold', config.BINARY_THRESHOLD)
            
            # Cut detection parameters
            cut_params = edge_detection.get('cut_detection', {})
            config.VERTICAL_MODE = cut_params.get('vertical_mode', config.VERTICAL_MODE)
            config.HORIZONTAL_MODE = cut_params.get('horizontal_mode', config.HORIZONTAL_MODE)
            
            # Window sizing parameters
            window_sizing = edge_detection.get('window_sizing', {})
            config.WINDOW_SIZE_DIVISOR = window_sizing.get('window_size_divisor', config.WINDOW_SIZE_DIVISOR)
            config.MIN_WINDOW_SIZE = window_sizing.get('min_window_size', config.MIN_WINDOW_SIZE)
            
    else:
        print("External config not found. Using internal default configuration.")
    return config

def find_vertical_cuts(binary_mask, mode='both_sides', config=None):
    """Find vertical cuts in the binary mask using sliding window approach for robustness."""
    height, width = binary_mask.shape
    vertical_projection = np.sum(binary_mask // 255, axis=0)
    
    window_size = max(width // config.WINDOW_SIZE_DIVISOR, config.MIN_WINDOW_SIZE)
    min_cut_strength = config.MIN_CUT_STRENGTH
    min_confidence_threshold = config.MIN_CONFIDENCE_THRESHOLD
    
    # Find left cut
    max_drop_left = 0
    cut_x_left = 0
    for i in range(window_size, width // 2):
        left_avg = np.mean(vertical_projection[i-window_size:i])
        right_avg = np.mean(vertical_projection[i:i+window_size]) if i+window_size < width else 0
        drop = left_avg - right_avg
        if drop > max_drop_left:
            max_drop_left = drop
            cut_x_left = i
    
    # Find right cut
    max_drop_right = 0
    cut_x_right = width
    for i in range(width - window_size, width // 2, -1):
        right_avg = np.mean(vertical_projection[i:i+window_size])
        left_avg = np.mean(vertical_projection[i-window_size:i]) if i-window_size >= 0 else 0
        drop = right_avg - left_avg
        if drop > max_drop_right:
            max_drop_right = drop
            cut_x_right = i

    # --- Evaluate whether cuts should be applied ---
    apply_left_cut = (max_drop_left >= min_cut_strength and max_drop_left >= min_confidence_threshold)
    apply_right_cut = (max_drop_right >= min_cut_strength and max_drop_right >= min_confidence_threshold)
    
    # --- Decide what to return based on mode ---
    if mode == 'both_sides':
        final_left = cut_x_left if apply_left_cut else 0
        final_right = cut_x_right if apply_right_cut else width
        return final_left, final_right, {
            "projection": vertical_projection,
            "left_cut_strength": max_drop_left,
            "right_cut_strength": max_drop_right,
            "left_cut_applied": apply_left_cut,
            "right_cut_applied": apply_right_cut
        }

    # For single_best mode, choose the stronger cut
    midpoint = width // 2
    left_density = np.sum(vertical_projection[:midpoint])
    right_density = np.sum(vertical_projection[midpoint:])
    
    if right_density < left_density:
        # Prefer right cut
        final_right = cut_x_right if apply_right_cut else width
        return 0, final_right, {
            "projection": vertical_projection,
            "right_cut_strength": max_drop_right,
            "right_cut_applied": apply_right_cut,
            "cut_side": "right"
        }
    else:
        # Prefer left cut
        final_left = cut_x_left if apply_left_cut else 0
        return final_left, width, {
            "projection": vertical_projection,
            "left_cut_strength": max_drop_left,
            "left_cut_applied": apply_left_cut,
            "cut_side": "left"
        }

def find_horizontal_cuts(binary_mask, mode='both_sides', config=None):
    """Find horizontal cuts in the binary mask using sliding window approach for robustness."""
    height, width = binary_mask.shape
    horizontal_projection = np.sum(binary_mask // 255, axis=1)
    
    window_size = max(height // config.WINDOW_SIZE_DIVISOR, config.MIN_WINDOW_SIZE)
    min_cut_strength = config.MIN_CUT_STRENGTH
    min_confidence_threshold = config.MIN_CONFIDENCE_THRESHOLD
    
    # Find top cut
    max_drop_top = 0
    cut_y_top = 0
    for i in range(window_size, height // 2):
        top_avg = np.mean(horizontal_projection[i-window_size:i])
        bottom_avg = np.mean(horizontal_projection[i:i+window_size]) if i+window_size < height else 0
        drop = top_avg - bottom_avg
        if drop > max_drop_top:
            max_drop_top = drop
            cut_y_top = i
    
    # Find bottom cut
    max_drop_bottom = 0
    cut_y_bottom = height
    for i in range(height - window_size, height // 2, -1):
        bottom_avg = np.mean(horizontal_projection[i:i+window_size])
        top_avg = np.mean(horizontal_projection[i-window_size:i]) if i-window_size >= 0 else 0
        drop = bottom_avg - top_avg
        if drop > max_drop_bottom:
            max_drop_bottom = drop
            cut_y_bottom = i

    # --- Evaluate whether cuts should be applied ---
    apply_top_cut = (max_drop_top >= min_cut_strength and max_drop_top >= min_confidence_threshold)
    apply_bottom_cut = (max_drop_bottom >= min_cut_strength and max_drop_bottom >= min_confidence_threshold)

    if mode == 'both_sides':
        final_top = cut_y_top if apply_top_cut else 0
        final_bottom = cut_y_bottom if apply_bottom_cut else height
        return final_top, final_bottom, {
            "projection": horizontal_projection,
            "top_cut_strength": max_drop_top,
            "bottom_cut_strength": max_drop_bottom,
            "top_cut_applied": apply_top_cut,
            "bottom_cut_applied": apply_bottom_cut
        }

    # For single_best mode, choose the stronger cut
    midpoint_y = height // 2
    top_density = np.sum(horizontal_projection[:midpoint_y])
    bottom_density = np.sum(horizontal_projection[midpoint_y:])

    if bottom_density < top_density:
        # Prefer bottom cut
        final_bottom = cut_y_bottom if apply_bottom_cut else height
        return 0, final_bottom, {
            "projection": horizontal_projection,
            "bottom_cut_strength": max_drop_bottom,
            "bottom_cut_applied": apply_bottom_cut,
            "cut_side": "bottom"
        }
    else:
        # Prefer top cut
        final_top = cut_y_top if apply_top_cut else 0
        return final_top, height, {
            "projection": horizontal_projection,
            "top_cut_strength": max_drop_top,
            "top_cut_applied": apply_top_cut,
            "cut_side": "top"
        }

def detect_roi_for_image(image_path, config):
    """Detects ROI for a single image and returns the coordinates."""
    try:
        original_img = cv2.imread(image_path)
        if original_img is None:
            print(f"Error: Could not load the image at {image_path}")
            return None
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    kernels = []
    for theta in np.arange(0, np.pi, np.pi / 4):
        kernel = cv2.getGaborKernel((config.GABOR_KERNEL_SIZE, config.GABOR_KERNEL_SIZE), 
                                   config.GABOR_SIGMA, float(theta), config.GABOR_LAMBDA, 
                                   config.GABOR_GAMMA, 0, ktype=cv2.CV_32F)
        kernels.append(kernel)

    combined_response = np.zeros_like(gray_img, dtype=np.float32)
    for kernel in kernels:
        filtered_img = cv2.filter2D(gray_img, cv2.CV_8UC3, kernel)
        combined_response += filtered_img.astype(np.float32)

    gabor_response_map = cv2.normalize(combined_response, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, binary_mask = cv2.threshold(gabor_response_map, config.BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)

    cut_x_left, cut_x_right, vertical_info = find_vertical_cuts(binary_mask, mode=config.VERTICAL_MODE, config=config)
    cut_y_top, cut_y_bottom, horizontal_info = find_horizontal_cuts(binary_mask, mode=config.HORIZONTAL_MODE, config=config)

    # Convert all values to JSON-serializable types
    def make_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        else:
            return obj
    
    return {
        "image_width": original_img.shape[1],
        "image_height": original_img.shape[0],
        "roi_left": int(cut_x_left),
        "roi_right": int(cut_x_right),
        "roi_top": int(cut_y_top),
        "roi_bottom": int(cut_y_bottom),
        "vertical_cut_info": make_json_serializable({k: v for k, v in vertical_info.items() if k != "projection"}),
        "horizontal_cut_info": make_json_serializable({k: v for k, v in horizontal_info.items() if k != "projection"})
    }

def process_edge_detection(config):
    """Processes a batch of images to detect ROI using edge detection."""
    if not os.path.isdir(config.INPUT_DIR):
        print(f"Error: Input directory '{config.INPUT_DIR}' not found.")
        return

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    image_files = [f for f in os.listdir(config.INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No image files found in '{config.INPUT_DIR}'.")
        return
        
    print("\n--- Starting Edge Detection for ROI ---")
    for filename in image_files:
        base_filename = os.path.splitext(filename)[0]
        image_path = os.path.join(config.INPUT_DIR, filename)
        
        roi_data = detect_roi_for_image(image_path, config)
        if roi_data:
            output_path = os.path.join(config.OUTPUT_DIR, f"{base_filename}{config.OUTPUT_JSON_SUFFIX}")
            with open(output_path, 'w') as f:
                json.dump(roi_data, f, indent=4)
            print(f"  - Processed '{filename}': ROI saved to {base_filename}{config.OUTPUT_JSON_SUFFIX}")
        else:
            print(f"  - Failed to process '{filename}'")

    print(f"--- Edge Detection Complete ---")


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
    config = load_config(config_path)
    process_edge_detection(config)