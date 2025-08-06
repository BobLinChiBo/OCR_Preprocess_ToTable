"""OCR Pipeline Processors Module.

This module provides modular image processing components for the OCR pipeline.
Each processor handles a specific aspect of the image processing workflow.
"""

# Base processor
from .base import BaseProcessor

# Image I/O
from .image_io import (
    load_image,
    save_image,
    get_image_files,
)

# Page splitting
from .page_split import (
    PageSplitProcessor,
    split_two_page_image,
)

# Deskewing
from .deskew import (
    DeskewProcessor,
    deskew_image,
)

# Margin removal
from .margin_removal import (
    MarginRemovalProcessor,
    remove_margin_inscribed,
    paper_mask,
    largest_inside_rect,
)

# Table detection
from .table_detection import (
    TableDetectionProcessor,
    detect_table_lines,
    filter_long_lines,
    merge_close_parallel_lines,
    cluster_line_positions,
    detect_table_structure,
    enumerate_table_cells,
)

# Table processing
from .table_processing import (
    TableProcessingProcessor,
    crop_to_table_borders,
    visualize_table_structure,
)

# Mark removal
from .mark_removal import (
    MarkRemovalProcessor,
    remove_marks,
    build_protect_mask,
    create_table_lines_mask,
)

# Visualization
from .visualization import (
    VisualizationProcessor,
    visualize_detected_lines,
)

# Table recovery
from .table_recovery import (
    TableRecoveryProcessor,
    table_recovery,
)

# Vertical strip cutting
from .vertical_strip_cutter import (
    VerticalStripCutterProcessor,
    cut_vertical_strips,
)

# Binarization
from .binarize import (
    BinarizeProcessor,
    binarize_image,
)

__all__ = [
    # Base
    "BaseProcessor",
    
    # Image I/O
    "load_image",
    "save_image",
    "get_image_files",
    
    # Page splitting
    "PageSplitProcessor",
    "split_two_page_image",
    
    # Deskewing
    "DeskewProcessor",
    "deskew_image",
    
    # Margin removal
    "MarginRemovalProcessor",
    "remove_margin_inscribed",
    "paper_mask",
    "largest_inside_rect",
    
    # Table detection
    "TableDetectionProcessor",
    "detect_table_lines",
    "filter_long_lines",
    "merge_close_parallel_lines",
    "cluster_line_positions",
    "detect_table_structure",
    "enumerate_table_cells",
    
    # Table processing
    "TableProcessingProcessor",
    "crop_to_table_borders",
    "visualize_table_structure",
    
    # Mark removal
    "MarkRemovalProcessor",
    "remove_marks",
    "build_protect_mask",
    "create_table_lines_mask",
    
    # Visualization
    "VisualizationProcessor",
    "visualize_detected_lines",
    
    # Table recovery
    "TableRecoveryProcessor",
    "table_recovery",
    
    # Vertical strip cutting
    "VerticalStripCutterProcessor",
    "cut_vertical_strips",
    
    # Binarization
    "BinarizeProcessor",
    "binarize_image",
]
