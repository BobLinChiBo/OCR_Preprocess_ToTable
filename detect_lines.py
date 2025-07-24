# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import json

# --- INTERNAL DEFAULT CONFIGURATION ---
class Config:
    INPUT_DIR = "deskewed_images"
    OUTPUT_DIR = "lines_images"
    SAVE_DEBUG_IMAGES = True
    DEBUG_OUTPUT_DIR = "lines_debug"
    ROI_MARGINS_PAGE_1 = {"top": 60, "bottom": 60, "left": 0, "right": 60}
    ROI_MARGINS_PAGE_2 = {"top": 60, "bottom": 60, "left": 30, "right": 5}
    ROI_MARGINS_DEFAULT = {"top": 60, "bottom": 60, "left": 5, "right": 5}
    V_PARAMS = {
        "morph_open_kernel_ratio": 1/60.0, "morph_close_kernel_ratio": 1/60.0,
        "hough_threshold": 5, "hough_min_line_length": 40,
        "hough_max_line_gap_ratio": 0.001, "cluster_distance_threshold": 15,
        "qualify_length_ratio": 0.5, "final_selection_ratio": 0.5,
        "solid_check_std_threshold": 30.0,
    }
    H_PARAMS = {
        "morph_open_kernel_ratio": 1/20.0, "morph_close_kernel_ratio": 1/50.0,
        "hough_threshold": 10, "hough_min_line_length": 30,
        "hough_max_line_gap_ratio": 0.01, "cluster_distance_threshold": 5,
        "qualify_length_ratio": 0.5, "final_selection_ratio": 0.7,
        "solid_check_std_threshold": 50.0,
    }
    OUTPUT_VIZ_LINE_THICKNESS = 2
    OUTPUT_VIZ_V_COLOR_BGR = (0, 0, 255)
    OUTPUT_VIZ_H_COLOR_BGR = (0, 255, 0)
    OUTPUT_JSON_SUFFIX = "_lines.json"
    OUTPUT_VIZ_SUFFIX = "_visualization.jpg"

def load_config_or_use_class(config_path='config.json'):
    """Loads config from JSON if it exists, otherwise uses the internal Config class."""
    config = Config()
    if os.path.exists(config_path):
        print(f"Loading configuration from '{config_path}'...")
        with open(config_path, 'r', encoding='utf-8') as f:
            ext = json.load(f)
        
        dirs, params = ext.get('directories', {}), ext.get('line_detection', {})
        config.INPUT_DIR = dirs.get('deskewed_images', config.INPUT_DIR)
        config.OUTPUT_DIR = dirs.get('lines_images', config.OUTPUT_DIR)
        config.DEBUG_OUTPUT_DIR = dirs.get('debug_output_dir', config.DEBUG_OUTPUT_DIR)
        
        for key in ['SAVE_DEBUG_IMAGES', 'ROI_MARGINS_PAGE_1', 'ROI_MARGINS_PAGE_2', 'ROI_MARGINS_DEFAULT', 
                    'OUTPUT_VIZ_LINE_THICKNESS', 'OUTPUT_JSON_SUFFIX', 'OUTPUT_VIZ_SUFFIX']:
            if key in params: setattr(config, key, params[key])

        if 'V_PARAMS' in params: config.V_PARAMS.update(params['V_PARAMS'])
        if 'H_PARAMS' in params: config.H_PARAMS.update(params['H_PARAMS'])
        if 'OUTPUT_VIZ_V_COLOR_BGR' in params: config.OUTPUT_VIZ_V_COLOR_BGR = tuple(params['OUTPUT_VIZ_V_COLOR_BGR'])
        if 'OUTPUT_VIZ_H_COLOR_BGR' in params: config.OUTPUT_VIZ_H_COLOR_BGR = tuple(params['OUTPUT_VIZ_H_COLOR_BGR'])
    else:
        print("External 'config.json' not found. Using internal default configuration.")
    return config

def find_lines(img, orientation, params, roi_mask, debug_path=None):
    # This function is unchanged as it receives parameters directly
    img_h, img_w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    binary = cv2.bitwise_and(binary, binary, mask=roi_mask)
    if debug_path: cv2.imwrite(os.path.join(debug_path, "01_binary_masked.jpg"), binary)

    if orientation == 'vertical':
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(img_h * params["morph_open_kernel_ratio"])))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(img_h * params["morph_close_kernel_ratio"])))
    else:
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(img_w * params["morph_open_kernel_ratio"]), 1))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(img_w * params["morph_close_kernel_ratio"]), 1))
    
    isolated = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=2)
    repaired = cv2.morphologyEx(isolated, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    if debug_path:
        cv2.imwrite(os.path.join(debug_path, "02_morph_isolated.jpg"), isolated)
        cv2.imwrite(os.path.join(debug_path, "03_morph_repaired.jpg"), repaired)
    
    num_labels, labels_matrix, stats, _ = cv2.connectedComponentsWithStats(repaired, connectivity=8)
    repaired_filtered = repaired.copy()
    for i in range(1, num_labels):
        x,y,w,h = stats[i,0], stats[i,1], stats[i,2], stats[i,3]
        if gray[y:y+h, x:x+w].size > 0 and np.std(gray[y:y+h, x:x+w]) < params["solid_check_std_threshold"]:
            repaired_filtered[labels_matrix == i] = 0
    if debug_path: cv2.imwrite(os.path.join(debug_path, "04_repaired_filtered.jpg"), repaired_filtered)
            
    edges = cv2.Canny(repaired_filtered, 50, 150, apertureSize=3)
    if debug_path: cv2.imwrite(os.path.join(debug_path, "05_canny_edges.jpg"), edges)
        
    max_line_gap = int(img_h * params["hough_max_line_gap_ratio"]) if orientation == 'vertical' else int(img_w * params["hough_max_line_gap_ratio"])
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=params["hough_threshold"], minLineLength=params["hough_min_line_length"], maxLineGap=max_line_gap)
    
    if debug_path and lines is not None:
        hough_viz = img.copy()
        for line in lines: cv2.line(hough_viz, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255, 0, 255), 1)
        cv2.imwrite(os.path.join(debug_path, "06_hough_raw_lines.jpg"), hough_viz)

    if lines is None: return []

    coord_index = 0 if orientation == 'vertical' else 1
    sorted_lines = sorted(lines, key=lambda line: line[0][coord_index])
    
    clusters = []
    if sorted_lines:
        current_cluster = [sorted_lines[0]]
        for i in range(1, len(sorted_lines)):
            if abs(sorted_lines[i][0][coord_index] - current_cluster[-1][0][coord_index]) < params["cluster_distance_threshold"]:
                current_cluster.append(sorted_lines[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sorted_lines[i]]
        clusters.append(current_cluster)

    detailed_clusters = []
    for cluster in clusters:
        if not cluster: continue
        points = np.vstack([line[0].reshape(-1, 2) for line in cluster])
        if orientation == 'vertical':
            avg_coord = int(np.mean(points[:, 0])); min_span, max_span = int(np.min(points[:, 1])), int(np.max(points[:, 1])); span_dim = img_h
        else:
            avg_coord = int(np.mean(points[:, 1])); min_span, max_span = int(np.min(points[:, 0])), int(np.max(points[:, 0])); span_dim = img_w
        detailed_clusters.append({'avg_coord': avg_coord, 'min_span': min_span, 'max_span': max_span})

    qualified = [c for c in detailed_clusters if (c['max_span'] - c['min_span']) > (span_dim * params["qualify_length_ratio"])]
    
    final_selection = []
    if qualified:
        max_len = max(c['max_span'] - c['min_span'] for c in qualified)
        final_selection = [c for c in qualified if (c['max_span'] - c['min_span']) >= (max_len * params["final_selection_ratio"])]
    
    if debug_path and final_selection:
        final_viz = img.copy()
        line_color = (0,0,255) if orientation == 'vertical' else (0,255,0)
        for line in final_selection:
            if orientation == 'vertical': cv2.line(final_viz, (line['avg_coord'], line['min_span']), (line['avg_coord'], line['max_span']), line_color, 2)
            else: cv2.line(final_viz, (line['min_span'], line['avg_coord']), (line['max_span'], line['avg_coord']), line_color, 2)
        cv2.imwrite(os.path.join(debug_path, "07_final_clustered_lines.jpg"), final_viz)
            
    return final_selection

def detect_lines_batch(config):
    """Processes a batch of images to detect lines."""
    if not os.path.isdir(config.INPUT_DIR):
        print(f"Error: Input directory '{config.INPUT_DIR}' not found."); return

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    if config.SAVE_DEBUG_IMAGES: os.makedirs(config.DEBUG_OUTPUT_DIR, exist_ok=True)

    image_files = [f for f in os.listdir(config.INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No image files found in '{config.INPUT_DIR}'."); return
        
    print("\n--- Starting Step 3: Line Detection ---")
    for filename in image_files:
        base_filename = os.path.splitext(filename)[0]
        image = cv2.imread(os.path.join(config.INPUT_DIR, filename))
        if image is None: continue

        if "_page_1" in filename: roi_margins, page_type = config.ROI_MARGINS_PAGE_1, "Right Page"
        elif "_page_2" in filename: roi_margins, page_type = config.ROI_MARGINS_PAGE_2, "Left Page"
        else: roi_margins, page_type = config.ROI_MARGINS_DEFAULT, "Default/Single Page"
        
        debug_path_v, debug_path_h = None, None
        if config.SAVE_DEBUG_IMAGES:
            image_debug_dir = os.path.join(config.DEBUG_OUTPUT_DIR, base_filename)
            debug_path_v, debug_path_h = os.path.join(image_debug_dir, "vertical"), os.path.join(image_debug_dir, "horizontal")
            os.makedirs(debug_path_v, exist_ok=True); os.makedirs(debug_path_h, exist_ok=True)

        h, w, _ = image.shape
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(roi_mask, (roi_margins["left"], roi_margins["top"]), (w - roi_margins["right"], h - roi_margins["bottom"]), 255, -1)
        
        vertical_lines = find_lines(image, 'vertical', config.V_PARAMS, roi_mask, debug_path_v)
        horizontal_lines = find_lines(image, 'horizontal', config.H_PARAMS, roi_mask, debug_path_h)
        print(f"  - Processing '{filename}' ({page_type}): Found {len(vertical_lines)} vertical and {len(horizontal_lines)} horizontal lines.")

        line_data = {"image_width": w, "image_height": h, "vertical_lines": vertical_lines, "horizontal_lines": horizontal_lines}
        with open(os.path.join(config.OUTPUT_DIR, f"{base_filename}{config.OUTPUT_JSON_SUFFIX}"), 'w') as f: json.dump(line_data, f, indent=4)

        viz_image = image.copy()
        for line in vertical_lines: cv2.line(viz_image, (line['avg_coord'], line['min_span']), (line['avg_coord'], line['max_span']), config.OUTPUT_VIZ_V_COLOR_BGR, config.OUTPUT_VIZ_LINE_THICKNESS)
        for line in horizontal_lines: cv2.line(viz_image, (line['min_span'], line['avg_coord']), (line['max_span'], line['avg_coord']), config.OUTPUT_VIZ_H_COLOR_BGR, config.OUTPUT_VIZ_LINE_THICKNESS)
        cv2.imwrite(os.path.join(config.OUTPUT_DIR, f"{base_filename}{config.OUTPUT_VIZ_SUFFIX}"), viz_image)

    print(f"--- Step 3 Complete ---")

if __name__ == '__main__':
    config = load_config_or_use_class()
    detect_lines_batch(config)