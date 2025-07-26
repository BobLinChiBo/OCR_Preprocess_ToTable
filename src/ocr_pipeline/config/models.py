"""
Pydantic models for OCR pipeline configuration.

Defines comprehensive configuration schemas with validation,
defaults, and documentation for all pipeline components.
"""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from pydantic import BaseModel, Field, validator, root_validator
import logging


class LogLevel(str, Enum):
    """Supported logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class CutDetectionMode(str, Enum):
    """Cut detection modes for edge detection."""
    BOTH_SIDES = "both_sides"
    SINGLE_BEST = "single_best"


class DirectoryConfig(BaseModel):
    """Directory configuration for pipeline stages."""
    
    raw_images: str = Field(
        default="input/raw_images",
        description="Directory containing raw scanned images"
    )
    splited_images: str = Field(
        default="output/stage1_initial_processing/01_split_pages",
        description="Directory for split page images"
    )
    deskewed_images: str = Field(
        default="output/stage1_initial_processing/02_deskewed",
        description="Directory for deskewed images"
    )
    lines_images: str = Field(
        default="output/stage1_initial_processing/03_line_detection",
        description="Directory for line detection results"
    )
    table_images: str = Field(
        default="output/stage1_initial_processing/04_table_reconstruction",
        description="Directory for table reconstruction results"
    )
    table_fit_images: str = Field(
        default="output/stage1_initial_processing/05_cropped_tables",
        description="Directory for cropped table images"
    )
    debug_output_dir: str = Field(
        default="debug/stage1_debug/line_detection",
        description="Directory for debug output"
    )
    
    @validator('*')
    def validate_directory_path(cls, v):
        """Validate directory path format."""
        if not v or not isinstance(v, str):
            raise ValueError("Directory path must be a non-empty string")
        return v.replace('\\', '/')  # Normalize path separators


class PageSplittingConfig(BaseModel):
    """Configuration for page splitting operations."""
    
    gutter_search_start_percent: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Start position for gutter search as fraction of image width"
    )
    gutter_search_end_percent: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="End position for gutter search as fraction of image width"
    )
    split_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for determining if image should be split"
    )
    left_page_suffix: str = Field(
        default="_page_2.jpg",
        description="Suffix for left page (verso) images"
    )
    right_page_suffix: str = Field(
        default="_page_1.jpg",
        description="Suffix for right page (recto) images"
    )
    
    @root_validator
    def validate_gutter_search_range(cls, values):
        """Validate that gutter search range is logical."""
        start = values.get('gutter_search_start_percent', 0.4)
        end = values.get('gutter_search_end_percent', 0.6)
        if start >= end:
            raise ValueError("gutter_search_start_percent must be less than gutter_search_end_percent")
        return values


class DeskewingConfig(BaseModel):
    """Configuration for image deskewing operations."""
    
    angle_range: float = Field(
        default=10.0,
        gt=0.0,
        le=45.0,
        description="Maximum angle range for skew detection (degrees)"
    )
    angle_step: float = Field(
        default=0.2,
        gt=0.0,
        le=1.0,
        description="Step size for angle search (degrees)"
    )
    min_angle_for_correction: float = Field(
        default=0.2,
        ge=0.0,
        description="Minimum detected angle to apply correction (degrees)"
    )


class GaborFilterConfig(BaseModel):
    """Configuration for Gabor filters in edge detection."""
    
    kernel_size: int = Field(
        default=31,
        ge=3,
        description="Size of Gabor kernel (must be odd)"
    )
    sigma: float = Field(
        default=4.0,
        gt=0.0,
        description="Standard deviation of Gaussian envelope"
    )
    lambda_param: float = Field(
        default=10.0,
        gt=0.0,
        alias="lambda",
        description="Wavelength of sinusoidal component"
    )
    gamma: float = Field(
        default=0.5,
        gt=0.0,
        description="Aspect ratio of Gabor kernel"
    )
    binary_threshold: int = Field(
        default=127,
        ge=0,
        le=255,
        description="Binary threshold for Gabor response"
    )
    
    @validator('kernel_size')
    def validate_odd_kernel_size(cls, v):
        """Ensure kernel size is odd."""
        if v % 2 == 0:
            raise ValueError("Kernel size must be odd")
        return v


class WindowSizingConfig(BaseModel):
    """Configuration for sliding window operations."""
    
    window_size_divisor: int = Field(
        default=20,
        gt=0,
        description="Divisor for calculating window size from image dimensions"
    )
    min_window_size: int = Field(
        default=50,
        gt=0,
        description="Minimum window size in pixels"
    )
    min_cut_strength: float = Field(
        default=10.0,
        ge=0.0,
        description="Minimum strength required for cut detection"
    )
    min_confidence_threshold: float = Field(
        default=5.0,
        ge=0.0,
        description="Minimum confidence threshold for cut application"
    )


class EdgeDetectionConfig(BaseModel):
    """Configuration for edge detection operations."""
    
    output_dir: str = Field(
        default="output/stage1_initial_processing/02.5_edge_detection",
        description="Output directory for edge detection results"
    )
    gabor_params: GaborFilterConfig = Field(
        default_factory=GaborFilterConfig,
        description="Gabor filter parameters"
    )
    cut_detection: Dict[str, CutDetectionMode] = Field(
        default={
            "vertical_mode": CutDetectionMode.SINGLE_BEST,
            "horizontal_mode": CutDetectionMode.SINGLE_BEST
        },
        description="Cut detection modes for vertical and horizontal cuts"
    )
    window_sizing: WindowSizingConfig = Field(
        default_factory=WindowSizingConfig,
        description="Window sizing parameters"
    )


class ROIMargins(BaseModel):
    """Region of Interest margins configuration."""
    
    top: int = Field(default=120, ge=0, description="Top margin in pixels")
    bottom: int = Field(default=120, ge=0, description="Bottom margin in pixels")
    left: int = Field(default=0, ge=0, description="Left margin in pixels")
    right: int = Field(default=100, ge=0, description="Right margin in pixels")


class LineDetectionParams(BaseModel):
    """Parameters for line detection algorithms."""
    
    morph_open_kernel_ratio: float = Field(
        default=0.0166,
        gt=0.0,
        le=1.0,
        description="Morphological opening kernel size as ratio of image dimension"
    )
    morph_close_kernel_ratio: float = Field(
        default=0.0166,
        gt=0.0,
        le=1.0,
        description="Morphological closing kernel size as ratio of image dimension"
    )
    hough_threshold: int = Field(
        default=5,
        gt=0,
        description="Threshold for Hough line detection"
    )
    hough_min_line_length: int = Field(
        default=40,
        gt=0,
        description="Minimum line length for Hough detection"
    )
    hough_max_line_gap_ratio: float = Field(
        default=0.001,
        ge=0.0,
        le=1.0,
        description="Maximum line gap as ratio of image dimension"
    )
    cluster_distance_threshold: int = Field(
        default=15,
        gt=0,
        description="Distance threshold for line clustering"
    )
    qualify_length_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum length ratio for line qualification"
    )
    final_selection_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Ratio for final line selection"
    )
    solid_check_std_threshold: float = Field(
        default=30.0,
        ge=0.0,
        description="Standard deviation threshold for solid region check"
    )
    contour_min_length_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum contour length ratio for curved line detection"
    )
    contour_aspect_ratio_threshold: float = Field(
        default=5.0,
        gt=0.0,
        description="Aspect ratio threshold for contour validation"
    )


class LineDetectionConfig(BaseModel):
    """Configuration for line detection operations."""
    
    save_debug_images: bool = Field(
        default=True,
        description="Whether to save debug images"
    )
    roi_margins_page_1: ROIMargins = Field(
        default=ROIMargins(top=120, bottom=120, left=0, right=100),
        description="ROI margins for page 1 (recto)"
    )
    roi_margins_page_2: ROIMargins = Field(
        default=ROIMargins(top=120, bottom=120, left=60, right=5),
        description="ROI margins for page 2 (verso)"
    )
    roi_margins_default: ROIMargins = Field(
        default=ROIMargins(top=60, bottom=60, left=5, right=5),
        description="Default ROI margins"
    )
    v_params: LineDetectionParams = Field(
        default_factory=LineDetectionParams,
        description="Parameters for vertical line detection"
    )
    h_params: LineDetectionParams = Field(
        default=LineDetectionParams(
            morph_open_kernel_ratio=0.033,
            morph_close_kernel_ratio=0.02,
            hough_threshold=10,
            hough_min_line_length=30,
            hough_max_line_gap_ratio=0.01,
            cluster_distance_threshold=5,
            qualify_length_ratio=0.5,
            final_selection_ratio=0.7,
            solid_check_std_threshold=50.0,
            contour_min_length_ratio=0.8,
            contour_aspect_ratio_threshold=5.0
        ),
        description="Parameters for horizontal line detection"
    )
    output_viz_line_thickness: int = Field(
        default=2,
        gt=0,
        description="Line thickness for visualization output"
    )
    output_viz_v_color_bgr: Tuple[int, int, int] = Field(
        default=(0, 0, 255),
        description="BGR color for vertical lines in visualization"
    )
    output_viz_h_color_bgr: Tuple[int, int, int] = Field(
        default=(0, 255, 0),
        description="BGR color for horizontal lines in visualization"
    )
    output_json_suffix: str = Field(
        default="_lines.json",
        description="Suffix for JSON output files"
    )
    output_viz_suffix: str = Field(
        default="_visualization.jpg",
        description="Suffix for visualization output files"
    )
    
    @validator('output_viz_v_color_bgr', 'output_viz_h_color_bgr')
    def validate_bgr_color(cls, v):
        """Validate BGR color values."""
        if len(v) != 3:
            raise ValueError("BGR color must have exactly 3 values")
        if not all(0 <= c <= 255 for c in v):
            raise ValueError("BGR color values must be between 0 and 255")
        return v


class TableReconstructionConfig(BaseModel):
    """Configuration for table reconstruction operations."""
    
    output_image_suffix: str = Field(
        default="_reconstructed.jpg",
        description="Suffix for reconstructed table images"
    )
    output_details_suffix: str = Field(
        default="_details.json",
        description="Suffix for table details JSON files"
    )
    vertical_line_color: Tuple[int, int, int] = Field(
        default=(0, 0, 255),
        description="BGR color for vertical lines"
    )
    horizontal_line_color: Tuple[int, int, int] = Field(
        default=(0, 255, 0),
        description="BGR color for horizontal lines"
    )
    line_thickness: int = Field(
        default=2,
        gt=0,
        description="Thickness of drawn lines"
    )
    
    @validator('vertical_line_color', 'horizontal_line_color')
    def validate_line_color(cls, v):
        """Validate line color values."""
        if len(v) != 3:
            raise ValueError("Line color must have exactly 3 values")
        if not all(0 <= c <= 255 for c in v):
            raise ValueError("Line color values must be between 0 and 255")
        return v


class TableFittingConfig(BaseModel):
    """Configuration for table fitting operations."""
    
    input_json_suffix: str = Field(
        default="_details.json",
        description="Suffix for input JSON files"
    )
    output_image_suffix: str = Field(
        default="_fitted.jpg",
        description="Suffix for fitted table images"
    )
    output_cells_suffix: str = Field(
        default="_fitted_cells.json",
        description="Suffix for cell data JSON files"
    )
    tolerance: int = Field(
        default=15,
        ge=0,
        description="Tolerance for line fitting in pixels"
    )
    coverage_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum coverage threshold for line validation"
    )
    figure_size: Tuple[int, int] = Field(
        default=(10, 12),
        description="Figure size for visualization (width, height)"
    )
    line_color: str = Field(
        default="blue",
        description="Color for fitted lines"
    )
    line_width: int = Field(
        default=2,
        gt=0,
        description="Width of fitted lines"
    )
    
    @validator('figure_size')
    def validate_figure_size(cls, v):
        """Validate figure size."""
        if len(v) != 2:
            raise ValueError("Figure size must have exactly 2 values (width, height)")
        if not all(s > 0 for s in v):
            raise ValueError("Figure size values must be positive")
        return v


class TableCroppingConfig(BaseModel):
    """Configuration for table cropping operations."""
    
    output_dir: str = Field(
        default="output/stage1_initial_processing/05_cropped_tables",
        description="Output directory for cropped tables"
    )
    input_details_suffix: str = Field(
        default="_details.json",
        description="Suffix for input details JSON files"
    )
    output_image_suffix: str = Field(
        default="_cropped.jpg",
        description="Suffix for cropped table images"
    )
    output_lines_suffix: str = Field(
        default="_cropped_lines.json",
        description="Suffix for cropped lines JSON files"
    )
    padding: int = Field(
        default=10,
        ge=0,
        description="Padding around cropped table in pixels"
    )


class LoggingConfig(BaseModel):
    """Configuration for logging setup."""
    
    level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Base logging level"
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Optional log file path"
    )
    use_rich: bool = Field(
        default=True,
        description="Whether to use rich console output"
    )
    include_performance: bool = Field(
        default=True,
        description="Whether to include performance logging"
    )
    format_style: str = Field(
        default="detailed",
        regex="^(simple|detailed|minimal)$",
        description="Logging format style"
    )
    debug_dir: Optional[str] = Field(
        default=None,
        description="Directory for debug logs"
    )


class Config(BaseModel):
    """Main configuration model for the OCR pipeline."""
    
    # Core configuration sections
    directories: DirectoryConfig = Field(
        default_factory=DirectoryConfig,
        description="Directory configuration"
    )
    page_splitting: PageSplittingConfig = Field(
        default_factory=PageSplittingConfig,
        description="Page splitting configuration"
    )
    deskewing: DeskewingConfig = Field(
        default_factory=DeskewingConfig,
        description="Deskewing configuration"
    )
    edge_detection: EdgeDetectionConfig = Field(
        default_factory=EdgeDetectionConfig,
        description="Edge detection configuration"
    )
    line_detection: LineDetectionConfig = Field(
        default_factory=LineDetectionConfig,
        description="Line detection configuration"
    )
    table_reconstruction: TableReconstructionConfig = Field(
        default_factory=TableReconstructionConfig,
        description="Table reconstruction configuration"
    )
    table_fitting: TableFittingConfig = Field(
        default_factory=TableFittingConfig,
        description="Table fitting configuration"
    )
    table_cropping: TableCroppingConfig = Field(
        default_factory=TableCroppingConfig,
        description="Table cropping configuration"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration"
    )
    
    # Meta configuration
    version: str = Field(
        default="0.2.0",
        description="Configuration version"
    )
    description: Optional[str] = Field(
        default=None,
        description="Configuration description"
    )
    
    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Prevent extra fields
        validate_assignment = True  # Validate on assignment
        use_enum_values = True  # Use enum values in serialization
    
    def get_directory(self, key: str) -> str:
        """Get directory path by key."""
        return getattr(self.directories, key, "")
    
    def create_output_directories(self) -> None:
        """Create all output directories."""
        from pathlib import Path
        
        for field_name, field_value in self.directories.dict().items():
            if field_name != 'raw_images':  # Don't create input directory
                Path(field_value).mkdir(parents=True, exist_ok=True)
    
    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convert to legacy dictionary format for backward compatibility."""
        return {
            "directories": self.directories.dict(),
            "page_splitting": {
                "GUTTER_SEARCH_START_PERCENT": self.page_splitting.gutter_search_start_percent,
                "GUTTER_SEARCH_END_PERCENT": self.page_splitting.gutter_search_end_percent,
                "SPLIT_THRESHOLD": self.page_splitting.split_threshold,
                "LEFT_PAGE_SUFFIX": self.page_splitting.left_page_suffix,
                "RIGHT_PAGE_SUFFIX": self.page_splitting.right_page_suffix,
            },
            "deskewing": {
                "ANGLE_RANGE": self.deskewing.angle_range,
                "ANGLE_STEP": self.deskewing.angle_step,
                "MIN_ANGLE_FOR_CORRECTION": self.deskewing.min_angle_for_correction,
            },
            "edge_detection": self.edge_detection.dict(),
            "line_detection": {
                "SAVE_DEBUG_IMAGES": self.line_detection.save_debug_images,
                "ROI_MARGINS_PAGE_1": self.line_detection.roi_margins_page_1.dict(),
                "ROI_MARGINS_PAGE_2": self.line_detection.roi_margins_page_2.dict(),
                "ROI_MARGINS_DEFAULT": self.line_detection.roi_margins_default.dict(),
                "V_PARAMS": self.line_detection.v_params.dict(),
                "H_PARAMS": self.line_detection.h_params.dict(),
                "OUTPUT_VIZ_LINE_THICKNESS": self.line_detection.output_viz_line_thickness,
                "OUTPUT_VIZ_V_COLOR_BGR": list(self.line_detection.output_viz_v_color_bgr),
                "OUTPUT_VIZ_H_COLOR_BGR": list(self.line_detection.output_viz_h_color_bgr),
                "OUTPUT_JSON_SUFFIX": self.line_detection.output_json_suffix,
                "OUTPUT_VIZ_SUFFIX": self.line_detection.output_viz_suffix,
            },
            "table_reconstruction": self.table_reconstruction.dict(),
            "table_fitting": self.table_fitting.dict(),
            "table_cropping": self.table_cropping.dict(),
        }