"""Table recovery processor for reconstructing table structure with merged cells."""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
import cv2

from .base import BaseProcessor

# Constants from recover_table.py
TOL = 15  # px tolerance when grouping lines from the detector


# ------------------------------------------------------------------ #
# Helper functions from recover_table.py                             #
# ------------------------------------------------------------------ #


def _cluster_lines(
    lines: List[List[int]], coord_idx: int, tol: int = TOL
) -> Dict[int, List[Tuple[int, int]]]:
    """
    coord_idx = 0 -> verticals (use X), 1 -> horizontals (use Y)

    returns {line_coord : [(span_start, span_end), ...]}
    """
    clusters: Dict[int, List[Tuple[int, int]]] = {}
    for x1, y1, x2, y2 in lines:
        key = x1 if coord_idx == 0 else y1
        # span along the orthogonal axis
        start, end = (y1, y2) if coord_idx == 0 else (x1, x2)
        if start > end:
            start, end = end, start
        # snap to an existing cluster within +/-tol
        anchor = next((g for g in clusters if abs(g - key) <= tol), key)
        clusters.setdefault(anchor, []).append((start, end))
    return clusters


# coverage of interval [a,b] by union of detector-spans
def _coverage(intvls: List[Tuple[int, int]], a: int, b: int) -> float:
    if a >= b:  # degenerate
        return 1.0
    covered = 0
    for s, e in intvls:
        s, e = max(a, s), min(b, e)
        if e > s:
            covered += e - s
    return covered / (b - a)


def _boundary_present(
    coord: int,
    a: int,
    b: int,
    clusters: Dict[int, List[Tuple[int, int]]],
    thresh: float,
    tol: int = TOL,
) -> bool:
    anchor = next((g for g in clusters if abs(g - coord) <= tol), None)
    if anchor is None:
        return False
    return _coverage(clusters[anchor], a, b) >= thresh


# simple disjoint-set union
class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.p[ry] = rx


# ------------------------------------------------------------------ #
# Recovery function from recover_table.py                             #
# ------------------------------------------------------------------ #


def recover_table_impl(
    lines_path, struct_path, out_json, out_img, bg=None, coverage_ratio=0.8
):
    """Core table recovery logic from recover_table.py"""
    with open(lines_path, "r", encoding="utf-8") as f:
        line_data = json.load(f)
    with open(struct_path, "r", encoding="utf-8") as f:
        S = json.load(f)

    xs, ys, raw_cells = S["xs"], S["ys"], S["cells"]
    n_rows, n_cols = len(ys) - 1, len(xs) - 1

    # Map each cell to its (row, col) position based on bounding box
    # raw_cells contains [x1, y1, x2, y2] for each cell
    idx = {}
    for i, (x1, y1, x2, y2) in enumerate(raw_cells):
        # Find which row this cell belongs to
        row = None
        for r in range(n_rows):
            if ys[r] <= y1 < ys[r + 1] and ys[r] < y2 <= ys[r + 1]:
                row = r
                break

        # Find which column this cell belongs to
        col = None
        for c in range(n_cols):
            if xs[c] <= x1 < xs[c + 1] and xs[c] < x2 <= xs[c + 1]:
                col = c
                break

        if row is not None and col is not None:
            idx[(row, col)] = i

    v = _cluster_lines(line_data["vertical_lines"], 0)
    h = _cluster_lines(line_data["horizontal_lines"], 1)

    dsu = DSU(len(raw_cells))

    # ---- missing vertical boundaries -> horizontal merges
    for r in range(n_rows):
        y1, y2 = ys[r], ys[r + 1]
        for c in range(n_cols - 1):
            if (r, c) in idx and (r, c + 1) in idx:
                left, right = idx[(r, c)], idx[(r, c + 1)]
                x = xs[c + 1]
                if not _boundary_present(x, y1, y2, v, coverage_ratio):
                    dsu.union(left, right)

    # ---- missing horizontal boundaries -> vertical merges
    for r in range(n_rows - 1):
        y = ys[r + 1]
        for c in range(n_cols):
            if (r, c) in idx and (r + 1, c) in idx:
                up, down = idx[(r, c)], idx[(r + 1, c)]
                x1, x2 = xs[c], xs[c + 1]
                if not _boundary_present(y, x1, x2, h, coverage_ratio):
                    dsu.union(up, down)

    # ---- collect merged groups
    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(raw_cells)):
        groups[dsu.find(i)].append(i)

    # Create reverse mapping from cell index to (row, col)
    cell_to_rowcol = {v: k for k, v in idx.items()}

    merged = []
    for members in groups.values():
        # Get bounding boxes for all cells in this group
        bboxes = [raw_cells[m] for m in members]
        x1s, y1s, x2s, y2s = zip(*bboxes)

        # Get row/col positions for all cells in this group
        rowcols = [cell_to_rowcol.get(m) for m in members if m in cell_to_rowcol]
        if rowcols:
            rows, cols = zip(*rowcols)
            min_row, max_row = min(rows), max(rows)
            min_col, max_col = min(cols), max(cols)

            merged.append(
                {
                    "row": min_row,
                    "col": min_col,
                    "rowspan": max_row - min_row + 1,
                    "colspan": max_col - min_col + 1,
                    "bbox": [min(x1s), min(y1s), max(x2s), max(y2s)],
                }
            )

    # ---- write JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {"rows": n_rows, "cols": n_cols, "cells": merged},
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ---- draw PNG
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    W, H = max_x - min_x + 4, max_y - min_y + 4
    canvas = (
        Image.open(bg).convert("RGB").crop((min_x, min_y, max_x, max_y))
        if bg
        else Image.new("RGB", (W, H), "white")
    )
    draw = ImageDraw.Draw(canvas, "RGBA")
    for c in merged:
        x1, y1, x2, y2 = c["bbox"]
        draw.rectangle(
            [x1 - min_x, y1 - min_y, x2 - min_x, y2 - min_y],
            outline=(0, 0, 0, 255),
            width=3,
        )
    canvas.save(out_img)
    # Only show debug info if explicitly requested
    if os.environ.get('DEBUG_TABLE_RECOVERY'):
        print(f"Recovered {len(merged)} merged cells -> {out_json}, {out_img}")


class TableRecoveryProcessor(BaseProcessor):
    """Processor for recovering table structure with merged cells."""

    def process(
        self,
        line_data: Dict[str, List[List[int]]] = None,
        structure_data: Dict[str, Any] = None,
        lines_json_path: str = None,
        structure_json_path: str = None,
        coverage_ratio: float = 0.8,
        background_image: np.ndarray = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process table lines and structure to recover merged cells.

        Args:
            line_data: Dictionary with 'horizontal_lines' and 'vertical_lines' arrays
            structure_data: Dictionary with 'xs', 'ys', and 'cells' arrays
            lines_json_path: Path to JSON file containing line data
            structure_json_path: Path to JSON file containing structure data
            coverage_ratio: Minimum fraction of border that must be covered by detected lines
            background_image: Optional background image for visualization

        Returns:
            Dictionary with recovered table structure including merged cells
        """
        import os
        import tempfile

        # Handle both file path and data inputs
        if lines_json_path and structure_json_path:
            # Use provided file paths directly
            lines_path = lines_json_path
            struct_path = structure_json_path
            temp_files_created = False
        else:
            # Use temporary files for data inputs
            if line_data is None or structure_data is None:
                raise ValueError("Either provide file paths or data dictionaries")

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f_lines:
                json.dump(line_data, f_lines)
                lines_path = f_lines.name

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f_struct:
                json.dump(structure_data, f_struct)
                struct_path = f_struct.name

            temp_files_created = True

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f_out_json:
            out_json_path = f_out_json.name

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_out_img:
            out_img_path = f_out_img.name

        try:
            # Call the recovery function
            recover_table_impl(
                lines_path=lines_path,
                struct_path=struct_path,
                out_json=out_json_path,
                out_img=out_img_path,
                bg=None,  # We'll handle visualization separately
                coverage_ratio=coverage_ratio,
            )

            # Read the results
            with open(out_json_path, "r") as f:
                recovered_data = json.load(f)

            # Create custom visualization if background image provided
            if background_image is not None:
                # If we used file paths, we need to load the structure data
                if lines_json_path and structure_json_path and structure_data is None:
                    with open(structure_json_path, "r") as f:
                        structure_data = json.load(f)

                visualization = self._create_visualization(
                    background_image,
                    recovered_data,
                    structure_data,
                    highlight_merged=kwargs.get("highlight_merged", True),
                    show_grid=kwargs.get("show_grid", True),
                    label_cells=kwargs.get("label_cells", True),
                )
            else:
                # Read the generated visualization
                visualization = np.array(Image.open(out_img_path))

            return {
                "rows": recovered_data["rows"],
                "cols": recovered_data["cols"],
                "cells": recovered_data["cells"],
                "visualization": visualization,
            }

        finally:
            # Clean up temporary files only if we created them
            if temp_files_created:
                for path in [lines_path, struct_path]:
                    if os.path.exists(path):
                        os.unlink(path)
            # Always clean up output temp files
            for path in [out_json_path, out_img_path]:
                if os.path.exists(path):
                    os.unlink(path)

    def _create_visualization(
        self,
        background: np.ndarray,
        recovered_data: Dict[str, Any],
        structure_data: Dict[str, Any],
        highlight_merged: bool = True,
        show_grid: bool = True,
        label_cells: bool = True,
    ) -> np.ndarray:
        """Create visualization of recovered table structure."""
        # Convert numpy array to PIL Image
        if background.dtype != np.uint8:
            background = (background * 255).astype(np.uint8)
        img = Image.fromarray(background).convert("RGB")
        
        # Create a separate layer for semi-transparent fills
        overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Main drawing context for opaque elements
        draw = ImageDraw.Draw(img)

        # First, draw faint grid if requested
        if show_grid:
            xs = structure_data.get("xs", [])
            ys = structure_data.get("ys", [])

            # Draw vertical grid lines
            for x in xs:
                draw.line(
                    [(x, min(ys)), (x, max(ys))], fill=(200, 200, 200), width=1
                )

            # Draw horizontal grid lines
            for y in ys:
                draw.line(
                    [(min(xs), y), (max(xs), y)], fill=(200, 200, 200), width=1
                )

        # First draw all cell fills on the overlay
        for cell in recovered_data["cells"]:
            x1, y1, x2, y2 = cell["bbox"]
            is_merged = cell["rowspan"] > 1 or cell["colspan"] > 1
            
            if not is_merged:
                # Normal cells - light blue fill
                overlay_draw.rectangle(
                    [x1 + 2, y1 + 2, x2 - 2, y2 - 2],
                    fill=(204, 229, 255, 60)  # Light blue with transparency
                )
            elif is_merged and highlight_merged:
                # Merged cells - light red fill
                overlay_draw.rectangle(
                    [x1 + 2, y1 + 2, x2 - 2, y2 - 2],
                    fill=(255, 200, 200, 100)  # Light red with transparency
                )
        
        # Composite the overlay onto the main image
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Then draw all borders
        for cell in recovered_data["cells"]:
            x1, y1, x2, y2 = cell["bbox"]
            is_merged = cell["rowspan"] > 1 or cell["colspan"] > 1
            
            if not is_merged:
                # Normal cells - blue border
                draw.rectangle(
                    [x1, y1, x2, y2], outline=(0, 102, 204), width=3  # Dark blue
                )
        
            elif is_merged and highlight_merged:
                # Merged cells - prominent visualization with red border
                # Thick red border
                draw.rectangle(
                    [x1, y1, x2, y2], outline=(255, 0, 0), width=8  # Bright red, thicker
                )

                # Add label if requested
                if label_cells:
                    label = f"{cell['rowspan']}x{cell['colspan']}"
                    # Calculate text position (centered)
                    text_x = (x1 + x2) // 2 - 20
                    text_y = (y1 + y2) // 2 - 10
                    # Draw text background for readability
                    draw.rectangle(
                        [text_x - 5, text_y - 2, text_x + 40, text_y + 15],
                        fill=(255, 255, 255),
                    )
                    draw.text((text_x, text_y), label, fill=(204, 0, 0))
        
        # Convert RGB to BGR for OpenCV
        result = np.array(img)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        return result


def table_recovery(
    line_data: Dict[str, List[List[int]]] = None,
    structure_data: Dict[str, Any] = None,
    lines_json_path: str = None,
    structure_json_path: str = None,
    coverage_ratio: float = 0.8,
    background_image: np.ndarray = None,
    highlight_merged: bool = True,
    show_grid: bool = True,
    label_cells: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function for table recovery.

    Args:
        line_data: Dictionary with 'horizontal_lines' and 'vertical_lines' arrays
        structure_data: Dictionary with 'xs', 'ys', and 'cells' arrays
        lines_json_path: Path to JSON file containing line data
        structure_json_path: Path to JSON file containing structure data
        coverage_ratio: Minimum fraction of border that must be covered by detected lines
        background_image: Optional background image for visualization
        highlight_merged: Whether to highlight merged cells prominently
        show_grid: Whether to show the underlying grid structure
        label_cells: Whether to add text labels to merged cells

    Returns:
        Dictionary with recovered table structure including merged cells
    """
    processor = TableRecoveryProcessor()
    return processor.process(
        line_data=line_data,
        structure_data=structure_data,
        lines_json_path=lines_json_path,
        structure_json_path=structure_json_path,
        coverage_ratio=coverage_ratio,
        background_image=background_image,
        highlight_merged=highlight_merged,
        show_grid=show_grid,
        label_cells=label_cells,
        **kwargs,
    )
