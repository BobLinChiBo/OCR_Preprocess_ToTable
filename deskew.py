# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import json
from scipy.ndimage import rotate

# --- INTERNAL DEFAULT CONFIGURATION ---
class Config:
    INPUT_DIR = "splited_images"
    OUTPUT_DIR = "deskewed_images"
    ANGLE_RANGE = 10.0
    ANGLE_STEP = 0.2
    MIN_ANGLE_FOR_CORRECTION = 0.2

def load_config_or_use_class(config_path='config.json'):
    """Loads config from JSON if it exists, otherwise uses the internal Config class."""
    config = Config()
    if os.path.exists(config_path):
        print(f"Loading configuration from '{config_path}'...")
        with open(config_path, 'r', encoding='utf-8') as f:
            external_config = json.load(f)
        
        dirs = external_config.get('directories', {})
        params = external_config.get('deskewing', {})
        config.INPUT_DIR = dirs.get('splited_images', config.INPUT_DIR)
        config.OUTPUT_DIR = dirs.get('deskewed_images', config.OUTPUT_DIR)
        config.ANGLE_RANGE = params.get('ANGLE_RANGE', config.ANGLE_RANGE)
        config.ANGLE_STEP = params.get('ANGLE_STEP', config.ANGLE_STEP)
        config.MIN_ANGLE_FOR_CORRECTION = params.get('MIN_ANGLE_FOR_CORRECTION', config.MIN_ANGLE_FOR_CORRECTION)
    else:
        print("External 'config.json' not found. Using internal default configuration.")
    return config

def deskew_image_batch(config):
    """Processes a batch of images to correct for skew."""
    if not os.path.isdir(config.INPUT_DIR):
        print(f"Error: Input directory '{config.INPUT_DIR}' not found."); return
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    image_files = [f for f in os.listdir(config.INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"No image files found in '{config.INPUT_DIR}'."); return

    print("\n--- Starting Step 2: Skew Correction ---")
    for filename in image_files:
        image_path = os.path.join(config.INPUT_DIR, filename)
        image = cv2.imread(image_path)
        if image is None: continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        def determine_score(arr, angle):
            data = rotate(arr, angle, reshape=False, order=0)
            histogram = np.sum(data, axis=1)
            return np.sum((histogram[1:] - histogram[:-1]) ** 2)

        angles = np.arange(-config.ANGLE_RANGE, config.ANGLE_RANGE + config.ANGLE_STEP, config.ANGLE_STEP)
        scores = [determine_score(thresh, angle) for angle in angles]
        best_angle = angles[np.argmax(scores)]

        if abs(best_angle) > config.MIN_ANGLE_FOR_CORRECTION:
            print(f"  - Processing '{filename}': Correcting skew of {best_angle:.2f} degrees.")
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
            deskewed_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        else:
            print(f"  - Processing '{filename}': No significant skew detected.")
            deskewed_image = image

        cv2.imwrite(os.path.join(config.OUTPUT_DIR, filename), deskewed_image)
        
    print(f"--- Step 2 Complete ---")

if __name__ == '__main__':
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
    config = load_config_or_use_class(config_path)
    deskew_image_batch(config)