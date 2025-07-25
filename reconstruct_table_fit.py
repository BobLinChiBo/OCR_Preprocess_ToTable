# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import json
import glob
import os
from collections import namedtuple

# --- INTERNAL DEFAULT CONFIGURATION ---
class Config:
    INPUT_DIR = "table_images"
    OUTPUT_DIR = "table_fit_images"
    INPUT_JSON_SUFFIX = "_details.json"
    OUTPUT_IMAGE_SUFFIX = "_fitted.jpg"
    OUTPUT_CELLS_SUFFIX = "_fitted_cells.json"
    TOLERANCE = 15
    COVERAGE_THRESHOLD = 0.4
    FIGURE_SIZE = (10, 12)
    LINE_COLOR = 'blue'
    LINE_WIDTH = 2

def load_config_or_use_class(config_path='config.json'):
    """Loads config from JSON if it exists, otherwise uses the internal Config class."""
    config = Config()
    if os.path.exists(config_path):
        print(f"Loading configuration from '{config_path}'...")
        with open(config_path, 'r', encoding='utf-8') as f:
            ext = json.load(f)
        
        dirs, params = ext.get('directories', {}), ext.get('table_fitting', {})
        config.INPUT_DIR = dirs.get('table_images', config.INPUT_DIR)
        config.OUTPUT_DIR = dirs.get('table_fit_images', config.OUTPUT_DIR)
        
        for key, default_val in vars(Config).items():
            if key.isupper() and key in params:
                setattr(config, key, params[key])
    else:
        print("External 'config.json' not found. Using internal default configuration.")
    return config

# --- Data Structures & Helper Functions ---
Line = namedtuple('Line', ['x1', 'y1', 'x2', 'y2'])

class Cell:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    def __repr__(self): return f"Cell({self.x1:.0f},{self.y1:.0f},{self.x2:.0f},{self.y2:.0f})"
    def __eq__(self, other): return all(abs(a - b) < 1e-3 for a, b in zip(vars(self).values(), vars(other).values()))
    def __hash__(self): return hash(tuple(round(v) for v in vars(self).values()))

def reconstruct_fitted_table(grid_h_lines, grid_v_lines, original_h_lines, original_v_lines, config):
    if not grid_h_lines or not grid_v_lines: return []

    x_coords = sorted(list(set([line.x1 for line in grid_v_lines])))
    y_coords = sorted(list(set([line.y1 for line in grid_h_lines])))
    
    atomic_cells = [Cell(x_coords[j], y_coords[i], x_coords[j+1], y_coords[i+1])
                    for i in range(len(y_coords) - 1) for j in range(len(x_coords) - 1)
                    if (x_coords[j+1] - x_coords[j]) > config.TOLERANCE / 2 and (y_coords[i+1] - y_coords[i]) > config.TOLERANCE / 2]

    def has_sufficient_coverage(p1, p2, lines, is_horizontal, config):
        boundary_length = abs(p2[0] - p1[0]) if is_horizontal else abs(p2[1] - p1[1])
        if boundary_length < 1e-6: return True
        total_overlap = sum(max(0, min(line.x2, p2[0]) - max(line.x1, p1[0])) for line in lines if is_horizontal and abs(line.y1 - p1[1]) < config.TOLERANCE)
        total_overlap += sum(max(0, min(line.y2, p2[1]) - max(line.y1, p1[1])) for line in lines if not is_horizontal and abs(line.x1 - p1[0]) < config.TOLERANCE)
        return total_overlap / boundary_length >= config.COVERAGE_THRESHOLD

    merged_cells = set(atomic_cells)
    while True:
        merged_in_pass, cell_to_remove1, cell_to_remove2, cell_to_add = False, None, None, None
        cells_list = sorted(list(merged_cells), key=lambda c: (c.y1, c.x1))

        for i in range(len(cells_list)):
            for j in range(i + 1, len(cells_list)):
                c1, c2 = cells_list[i], cells_list[j]
                if abs(c1.x2 - c2.x1) < config.TOLERANCE and abs(c1.y1 - c2.y1) < config.TOLERANCE and abs(c1.y2 - c2.y2) < config.TOLERANCE:
                    if not has_sufficient_coverage((c1.x2, c1.y1), (c1.x2, c1.y2), original_v_lines, False, config):
                        cell_to_add = Cell(c1.x1, c1.y1, c2.x2, c1.y2); merged_in_pass = True
                elif abs(c1.y2 - c2.y1) < config.TOLERANCE and abs(c1.x1 - c2.x1) < config.TOLERANCE and abs(c1.x2 - c2.x2) < config.TOLERANCE:
                    if not has_sufficient_coverage((c1.x1, c1.y2), (c1.x2, c1.y2), original_h_lines, True, config):
                        cell_to_add = Cell(c1.x1, c1.y1, c1.x2, c2.y2); merged_in_pass = True
                if merged_in_pass: cell_to_remove1, cell_to_remove2 = c1, c2; break
            if merged_in_pass: break
        
        if merged_in_pass:
            merged_cells.remove(cell_to_remove1); merged_cells.remove(cell_to_remove2); merged_cells.add(cell_to_add)
        else:
            break
            
    return list(merged_cells)

def extract_lines_from_cells(cells):
    h_lines, v_lines = set(), set()
    for cell in cells:
        h_lines.add((cell.x1, cell.y1, cell.x2, cell.y1)); h_lines.add((cell.x1, cell.y2, cell.x2, cell.y2))
        v_lines.add((cell.x1, cell.y1, cell.x1, cell.y2)); v_lines.add((cell.x2, cell.y1, cell.x2, cell.y2))
    return [Line(*t) for t in h_lines], [Line(*t) for t in v_lines]

def batch_process_fitted_tables(config):
    print("\n--- Starting Step 5: Fitted Table Reconstruction ---")
    if not os.path.isdir(config.INPUT_DIR):
        print(f"Error: Input directory not found at '{config.INPUT_DIR}'"); return
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    json_files = glob.glob(os.path.join(config.INPUT_DIR, f'*{config.INPUT_JSON_SUFFIX}'))
    if not json_files:
        print(f"Error: No '*{config.INPUT_JSON_SUFFIX}' files found in '{config.INPUT_DIR}'."); return
    
    print(f"Found {len(json_files)} details file(s) to process.")
    
    for json_path in json_files:
        print(f"  - Processing: {os.path.basename(json_path)}")
        try:
            with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
        except Exception as e:
            print(f"    Error loading {os.path.basename(json_path)}: {e}"); continue

        original_v_data = data.get('original_detected_lines', {}).get('vertical_lines', [])
        original_h_data = data.get('original_detected_lines', {}).get('horizontal_lines', [])
        original_v_lines = [Line(d['avg_coord'], d['min_span'], d['avg_coord'], d['max_span']) for d in original_v_data]
        original_h_lines = [Line(d['min_span'], d['avg_coord'], d['max_span'], d['avg_coord']) for d in original_h_data]
        grid_v_data = data.get('reconstructed_grid_lines', {}).get('vertical_lines', [])
        grid_h_data = data.get('reconstructed_grid_lines', {}).get('horizontal_lines', [])
        grid_v_lines = [Line(p1[0], p1[1], p2[0], p2[1]) for p1, p2 in grid_v_data]
        grid_h_lines = [Line(p1[0], p1[1], p2[0], p2[1]) for p1, p2 in grid_h_data]
        image_path = data.get('source_image_path')

        if not all((original_h_lines, original_v_lines, grid_h_lines, grid_v_lines, image_path)):
            print(f"    Warning: Skipping due to missing data in JSON file."); continue

        final_cells = reconstruct_fitted_table(grid_h_lines, grid_v_lines, original_h_lines, original_v_lines, config)
        
        base_name = os.path.basename(json_path).replace(config.INPUT_JSON_SUFFIX, '')
        cells_as_dicts = [{'x1': c.x1, 'y1': c.y1, 'x2': c.x2, 'y2': c.y2} for c in final_cells]
        with open(os.path.join(config.OUTPUT_DIR, f"{base_name}{config.OUTPUT_CELLS_SUFFIX}"), 'w') as f:
            json.dump({'fitted_cells': cells_as_dicts}, f, indent=4)
        
        final_h_lines, final_v_lines = extract_lines_from_cells(final_cells)
        try:
            img = plt.imread(image_path)
            fig, ax = plt.subplots(1, figsize=config.FIGURE_SIZE)
            ax.imshow(img)
            for line in final_h_lines: ax.plot([line.x1, line.x2], [line.y1, line.y2], color=config.LINE_COLOR, linewidth=config.LINE_WIDTH)
            for line in final_v_lines: ax.plot([line.x1, line.x2], [line.y1, line.y2], color=config.LINE_COLOR, linewidth=config.LINE_WIDTH)
            plt.axis('off')
            plt.savefig(os.path.join(config.OUTPUT_DIR, f"{base_name}{config.OUTPUT_IMAGE_SUFFIX}"), bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
        except FileNotFoundError:
            print(f"    Warning: Could not find image at '{image_path}' for visualization.")

    print(f"\n--- Step 5 Complete ---")

if __name__ == '__main__':
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.json'
    config = load_config_or_use_class(config_path)
    batch_process_fitted_tables(config)