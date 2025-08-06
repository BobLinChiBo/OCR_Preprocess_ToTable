"""Utilities for extracting information from recovered table data."""

from typing import Dict, List, Tuple, Any


def extract_vertical_lines_from_recovered_table(
    recovered_data: Dict[str, Any],
    min_row_coverage: float = 1.0
) -> List[Tuple[int, int, int, int]]:
    """
    Extract vertical lines from recovered table data, considering merged cells.
    
    Args:
        recovered_data: Dictionary containing 'rows', 'cols', and 'cells' with bbox info
        min_row_coverage: Minimum fraction of rows a vertical line must span (1.0 = full height)
    
    Returns:
        List of vertical lines as (x1, y1, x2, y2) tuples
    """
    if not recovered_data or 'cells' not in recovered_data:
        return []
    
    cells = recovered_data['cells']
    if not cells:
        return []
    
    # Get table dimensions
    num_rows = recovered_data.get('rows', 0)
    num_cols = recovered_data.get('cols', 0)
    
    # Find overall y-extent of the table
    y_min = min(cell['bbox'][1] for cell in cells)
    y_max = max(cell['bbox'][3] for cell in cells)
    
    # Track x-coordinates and which rows they appear in
    x_coord_rows = {}  # x -> set of rows where this x appears
    
    for cell in cells:
        bbox = cell['bbox']  # [x1, y1, x2, y2]
        row = cell['row']
        colspan = cell.get('colspan', 1)
        
        # Left edge of the cell
        x_left = bbox[0]
        if x_left not in x_coord_rows:
            x_coord_rows[x_left] = set()
        x_coord_rows[x_left].add(row)
        
        # Right edge of the cell
        x_right = bbox[2]
        if x_right not in x_coord_rows:
            x_coord_rows[x_right] = set()
        x_coord_rows[x_right].add(row)
        
        # For merged cells (colspan > 1), the internal boundaries don't count
        # as full vertical lines, so we don't add them to all rows
    
    # Create vertical lines based on row coverage
    vertical_lines = []
    min_rows_required = int(num_rows * min_row_coverage)
    
    for x, rows in x_coord_rows.items():
        if len(rows) >= min_rows_required:
            # This x-coordinate appears in enough rows to be considered a vertical line
            
            # Find the actual y-extent for this x-coordinate
            y_coords = []
            for cell in cells:
                bbox = cell['bbox']
                # If this x is a boundary of this cell
                if x == bbox[0] or x == bbox[2]:
                    y_coords.extend([bbox[1], bbox[3]])
            
            if y_coords:
                line_y_min = min(y_coords)
                line_y_max = max(y_coords)
                vertical_lines.append((x, line_y_min, x, line_y_max))
    
    # Sort by x-coordinate
    vertical_lines.sort(key=lambda line: line[0])
    
    return vertical_lines


def get_major_vertical_boundaries(
    recovered_data: Dict[str, Any]
) -> List[Tuple[int, int, int, int]]:
    """
    Get only the major vertical boundaries (full-height lines) from recovered table data.
    These are the lines that span all rows, ignoring internal merged cell boundaries.
    
    Args:
        recovered_data: Dictionary containing recovered table structure
    
    Returns:
        List of major vertical lines as (x1, y1, x2, y2) tuples
    """
    return extract_vertical_lines_from_recovered_table(recovered_data, min_row_coverage=1.0)


def get_all_vertical_boundaries(
    recovered_data: Dict[str, Any]
) -> List[Tuple[int, int, int, int]]:
    """
    Get all vertical boundaries from recovered table data, including internal ones.
    
    Args:
        recovered_data: Dictionary containing recovered table structure
    
    Returns:
        List of all vertical lines as (x1, y1, x2, y2) tuples
    """
    if not recovered_data or 'cells' not in recovered_data:
        return []
    
    cells = recovered_data['cells']
    if not cells:
        return []
    
    # Collect all unique x-coordinates and their extents
    x_extents = {}  # x -> (y_min, y_max)
    
    for cell in cells:
        bbox = cell['bbox']  # [x1, y1, x2, y2]
        
        # Process left edge
        x_left = bbox[0]
        if x_left not in x_extents:
            x_extents[x_left] = [bbox[1], bbox[3]]
        else:
            x_extents[x_left][0] = min(x_extents[x_left][0], bbox[1])
            x_extents[x_left][1] = max(x_extents[x_left][1], bbox[3])
        
        # Process right edge
        x_right = bbox[2]
        if x_right not in x_extents:
            x_extents[x_right] = [bbox[1], bbox[3]]
        else:
            x_extents[x_right][0] = min(x_extents[x_right][0], bbox[1])
            x_extents[x_right][1] = max(x_extents[x_right][1], bbox[3])
    
    # Create vertical lines
    vertical_lines = []
    for x, (y_min, y_max) in x_extents.items():
        vertical_lines.append((x, y_min, x, y_max))
    
    # Sort by x-coordinate
    vertical_lines.sort(key=lambda line: line[0])
    
    return vertical_lines