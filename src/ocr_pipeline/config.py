"""Configuration classes for OCR pipeline."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Config:
    """Simple configuration class."""

    # Input/Output directories
    input_dir: Path = Path("data/input")
    output_dir: Path = Path("data/output")
    debug_dir: Optional[Path] = None

    # Page splitting parameters
    search_ratio: float = 0.5  # Fraction of width, centered, to search for gutter
    line_len_frac: float = 0.3  # Vertical kernel = 30% of image height
    line_thick: int = 3  # Kernel width in pixels
    peak_thr: float = 0.3  # Peak threshold (30% of max response)

    # Deskewing
    angle_range: int = 5
    angle_step: float = 0.1
    min_angle_correction: float = 0.1

    # Table line detection (new connected components method)
    threshold: int = 40
    horizontal_kernel_size: int = 10
    vertical_kernel_size: int = 10
    alignment_threshold: int = 3
    
    # Horizontal line length filtering
    h_min_length_image_ratio: float = 0.3  # Min length as ratio of image width
    h_min_length_relative_ratio: float = 0.4  # Min length relative to longest h-line
    
    # Vertical line length filtering  
    v_min_length_image_ratio: float = 0.3  # Min length as ratio of image height
    v_min_length_relative_ratio: float = 0.4  # Min length relative to longest v-line
    
    min_aspect_ratio: int = 5

    # Table line post-processing parameters
    max_h_length_ratio: float = (
        1.0  # Max horizontal line length ratio (1.0 = disable filtering)
    )
    max_v_length_ratio: float = (
        1.0  # Max vertical line length ratio (1.0 = disable filtering)
    )
    close_line_distance: int = 45  # Distance for merging close lines (0 = disable)

    # Table structure detection parameters
    table_detection_eps: int = 10  # Clustering tolerance in pixels
    table_detection_kernel_ratio: float = (
        0.05  # Morphology kernel length as fraction of width/height
    )
    table_crop_padding: int = 20  # Padding around detected table borders
    enable_table_border_cropping: bool = True  # Enable new table border cropping

    # Margin removal (replaces ROI detection)
    enable_margin_removal: bool = True
    blur_kernel_size: int = 7
    black_threshold: int = 50
    content_threshold: int = 200
    morph_kernel_size: int = 25
    min_content_area_ratio: float = 0.01
    margin_padding: int = 5

    # Smart margin removal parameters
    histogram_threshold: float = 0.05
    projection_smoothing: int = 3

    # Curved black background removal parameters
    min_contour_area: int = 1000
    fill_method: str = "color_fill"

    # Inscribed rectangle margin removal parameters
    inscribed_blur_ksize: int = 5
    inscribed_close_ksize: int = 25
    inscribed_close_iter: int = 2

    # Debug options
    save_debug_images: bool = False
    verbose: bool = False

    def __post_init__(self) -> None:
        """Ensure paths are Path objects and validate parameters."""
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        if self.debug_dir:
            self.debug_dir = Path(self.debug_dir)

        # Validate parameters
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate configuration parameters."""
        # Page split parameters
        if not (0.0 <= self.search_ratio <= 1.0):
            raise ValueError(
                f"search_ratio must be between 0 and 1, got {self.search_ratio}"
            )
        if not (0.0 <= self.line_len_frac <= 1.0):
            raise ValueError(
                f"line_len_frac must be between 0 and 1, got {self.line_len_frac}"
            )
        if self.line_thick <= 0:
            raise ValueError(f"line_thick must be positive, got {self.line_thick}")
        if not (0.0 <= self.peak_thr <= 1.0):
            raise ValueError(
                f"peak_thr must be between 0 and 1, got {self.peak_thr}"
            )
        if self.angle_range <= 0:
            raise ValueError(f"angle_range must be positive, got {self.angle_range}")
        if self.angle_step <= 0:
            raise ValueError(f"angle_step must be positive, got {self.angle_step}")
        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive, got {self.threshold}")
        if self.horizontal_kernel_size <= 0:
            raise ValueError(
                f"horizontal_kernel_size must be positive, got {self.horizontal_kernel_size}"
            )
        if self.vertical_kernel_size <= 0:
            raise ValueError(
                f"vertical_kernel_size must be positive, got {self.vertical_kernel_size}"
            )
        if self.max_h_length_ratio <= 0:
            raise ValueError(
                f"max_h_length_ratio must be positive, got {self.max_h_length_ratio}"
            )
        if self.max_v_length_ratio <= 0:
            raise ValueError(
                f"max_v_length_ratio must be positive, got {self.max_v_length_ratio}"
            )
        if self.close_line_distance < 0:
            raise ValueError(
                f"close_line_distance must be non-negative, got {self.close_line_distance}"
            )

    def create_output_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_json(cls, json_path):
        """Load configuration from JSON file."""
        json_path = Path(json_path)  # Convert string to Path if needed
        if not json_path.exists():
            raise FileNotFoundError(f"Config file not found: {json_path}")

        with open(json_path, "r") as f:
            config_data = json.load(f)

        return cls.from_dict(config_data)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary, applying only relevant fields."""
        # Get the dataclass fields for this class
        field_names = {field.name for field in cls.__dataclass_fields__.values()}

        # Filter config_dict to only include fields that exist in this class
        filtered_config = {}
        for key, value in config_dict.items():
            if key in field_names:
                # Convert path strings to Path objects
                if key.endswith("_dir") and isinstance(value, str):
                    value = Path(value)
                filtered_config[key] = value
            elif key == "line_detection" and isinstance(value, dict):
                # Handle nested line_detection parameters
                for line_key, line_value in value.items():
                    if line_key in field_names:
                        filtered_config[line_key] = line_value

        return cls(**filtered_config)


@dataclass
class Stage1Config(Config):
    """Stage 1 Configuration: Initial Processing with aggressive parameters."""

    # Override defaults for Stage 1 - more aggressive initial detection
    output_dir: Path = Path("data/output/stage1_initial_processing")

    # Stage 1 specific deskewing (wider range)
    angle_range: int = 10
    angle_step: float = 0.2
    min_angle_correction: float = 0.2

    # Stage 1 table line detection - more permissive
    threshold: int = 35  # Lower threshold for more permissive detection
    horizontal_kernel_size: int = 15  # Larger kernels for initial pass
    vertical_kernel_size: int = 15
    alignment_threshold: int = 5  # More permissive clustering
    
    # Stage 1 horizontal line filtering - more permissive
    h_min_length_image_ratio: float = 0.2  # Lower ratio to catch more segments
    h_min_length_relative_ratio: float = 0.3  # More permissive final filtering
    
    # Stage 1 vertical line filtering - more permissive
    v_min_length_image_ratio: float = 0.2  # Lower ratio to catch more segments
    v_min_length_relative_ratio: float = 0.3  # More permissive final filtering
    
    min_aspect_ratio: int = 3  # Lower aspect ratio for initial detection

    # Stage 1 mark removal - remove watermarks/stamps/artifacts
    enable_mark_removal: bool = True
    mark_removal_dilate_iter: int = 2

    # Stage 1 margin removal - more aggressive
    enable_margin_removal: bool = True
    blur_kernel_size: int = 9
    black_threshold: int = 45
    content_threshold: int = 180
    morph_kernel_size: int = 30
    min_content_area_ratio: float = 0.005
    margin_padding: int = 10

    # Stage 1 inscribed rectangle margin removal - more aggressive
    inscribed_blur_ksize: int = 7
    inscribed_close_ksize: int = 30
    inscribed_close_iter: int = 3

    def __post_init__(self) -> None:
        """Initialize Stage 1 specific settings."""
        super().__post_init__()

    def create_output_dirs(self) -> None:
        """Create Stage 1 specific output directories."""
        super().create_output_dirs()

        # Create Stage 1 subdirectories
        stage1_dirs = [
            "01_marks_removed",
            "02_margin_removed",
            "03_split_pages",
            "04_deskewed",
            "05_table_lines",
            "06_table_structure",
            "07_border_cropped",
        ]

        for subdir in stage1_dirs:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Stage1Config":
        """Create Stage1Config from dictionary with special handling for nested structures."""
        # Handle nested structures
        if "page_splitting" in config_dict:
            page_split = config_dict["page_splitting"]
            config_dict.update(
                {
                    "search_ratio": page_split.get("search_ratio", 0.5),
                    "line_len_frac": page_split.get("line_len_frac", 0.3),
                    "line_thick": page_split.get("line_thick", 3),
                    "peak_thr": page_split.get("peak_thr", 0.3),
                }
            )
            del config_dict["page_splitting"]

        if "deskewing" in config_dict:
            deskew = config_dict["deskewing"]
            config_dict.update(
                {
                    "angle_range": deskew.get("angle_range"),
                    "angle_step": deskew.get("angle_step"),
                    "min_angle_correction": deskew.get("min_angle_correction"),
                }
            )
            del config_dict["deskewing"]

        if "table_line_detection" in config_dict:
            line_det = config_dict["table_line_detection"]
            
            config_dict.update(
                {
                    "threshold": line_det.get("threshold"),
                    "horizontal_kernel_size": line_det.get("horizontal_kernel_size"),
                    "vertical_kernel_size": line_det.get("vertical_kernel_size"),
                    "alignment_threshold": line_det.get("alignment_threshold"),
                    "min_aspect_ratio": line_det.get("min_aspect_ratio"),
                    
                    # New separated parameters
                    "h_min_length_image_ratio": line_det.get("h_min_length_image_ratio"),
                    "h_min_length_relative_ratio": line_det.get("h_min_length_relative_ratio"),
                    "v_min_length_image_ratio": line_det.get("v_min_length_image_ratio"),
                    "v_min_length_relative_ratio": line_det.get("v_min_length_relative_ratio"),
                }
            )
            del config_dict["table_line_detection"]

        if "mark_removal" in config_dict:
            mark = config_dict["mark_removal"]
            config_dict.update(
                {
                    "enable_mark_removal": mark.get("enable_mark_removal", True),
                    "mark_removal_dilate_iter": mark.get("dilate_iter", 2),
                }
            )
            del config_dict["mark_removal"]

        if "margin_removal" in config_dict:
            margin = config_dict["margin_removal"]
            config_dict.update(
                {
                    "enable_margin_removal": margin.get("enable_margin_removal", True),
                    "blur_kernel_size": margin.get("blur_kernel_size", 7),
                    "black_threshold": margin.get("black_threshold", 50),
                    "content_threshold": margin.get("content_threshold", 200),
                    "morph_kernel_size": margin.get("morph_kernel_size", 25),
                    "min_content_area_ratio": margin.get(
                        "min_content_area_ratio", 0.01
                    ),
                    "margin_padding": margin.get("margin_padding", 5),
                }
            )
            del config_dict["margin_removal"]

        # Handle inscribed_margin_removal specially
        if "inscribed_margin_removal" in config_dict:
            inscribed = config_dict["inscribed_margin_removal"]
            config_dict.update(
                {
                    "inscribed_blur_ksize": inscribed.get("inscribed_blur_ksize"),
                    "inscribed_close_ksize": inscribed.get("inscribed_close_ksize"),
                    "inscribed_close_iter": inscribed.get("inscribed_close_iter"),
                }
            )
            del config_dict["inscribed_margin_removal"]

        # Handle table_detection specially
        if "table_detection" in config_dict:
            table_det = config_dict["table_detection"]
            config_dict.update(
                {
                    "table_detection_eps": table_det.get("table_detection_eps"),
                    "table_detection_kernel_ratio": table_det.get(
                        "table_detection_kernel_ratio"
                    ),
                    "table_crop_padding": table_det.get("table_crop_padding"),
                    "enable_table_border_cropping": table_det.get(
                        "enable_table_border_cropping"
                    ),
                }
            )
            del config_dict["table_detection"]

        # Handle roi_margins specially
        if "roi_margins" in config_dict:
            margins = config_dict["roi_margins"]
            config_dict["roi_margins_page_1"] = margins.get("page_1")
            config_dict["roi_margins_page_2"] = margins.get("page_2")
            config_dict["roi_margins_default"] = margins.get("default")
            del config_dict["roi_margins"]

        # Remove any unrecognized sections
        if "table_fitting" in config_dict:
            del config_dict["table_fitting"]

        return cls(**config_dict)


@dataclass
class Stage2Config(Config):
    """Stage 2 Configuration: Refinement processing with precise parameters."""

    # Stage 2 takes input from Stage 1 output
    input_dir: Path = Path("data/output/stage1_initial_processing/07_border_cropped")
    output_dir: Path = Path("data/output/stage2_refinement")

    # Stage 2 deskewing (fine-tuning)
    angle_range: int = 10
    angle_step: float = 0.2
    min_angle_correction: float = 0.2

    # Stage 2 table line detection - more precise
    threshold: int = 45  # Higher threshold for more precise detection
    horizontal_kernel_size: int = 8  # Smaller kernels for refinement pass
    vertical_kernel_size: int = 8
    alignment_threshold: int = 2  # More strict clustering
    
    # Stage 2 horizontal line filtering - more precise
    h_min_length_image_ratio: float = 0.4  # Higher ratio for quality filtering
    h_min_length_relative_ratio: float = 0.5  # More strict final filtering
    
    # Stage 2 vertical line filtering - more precise
    v_min_length_image_ratio: float = 0.4  # Higher ratio for quality filtering
    v_min_length_relative_ratio: float = 0.5  # More strict final filtering
    
    min_aspect_ratio: int = 7  # Higher aspect ratio for refined detection

    # Stage 2 margin removal - more conservative (minimal additional cropping)
    enable_margin_removal: bool = (
        False  # Usually disabled for Stage 2 (already cropped)
    )

    # Stage 2 inscribed rectangle margin removal - conservative
    inscribed_blur_ksize: int = 3
    inscribed_close_ksize: int = 15
    inscribed_close_iter: int = 1

    def __post_init__(self) -> None:
        """Initialize Stage 2 specific settings."""
        super().__post_init__()

    def create_output_dirs(self) -> None:
        """Create Stage 2 specific output directories."""
        super().create_output_dirs()

        # Create Stage 2 subdirectories
        stage2_dirs = [
            "01_deskewed",
            "02_margin_removed",
            "03_table_lines",
            "04_fitted_tables",
        ]

        for subdir in stage2_dirs:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Stage2Config":
        """Create Stage2Config from dictionary with special handling for nested structures."""
        # Handle nested structures
        if "deskewing" in config_dict:
            deskew = config_dict["deskewing"]
            config_dict.update(
                {
                    "angle_range": deskew.get("angle_range"),
                    "angle_step": deskew.get("angle_step"),
                    "min_angle_correction": deskew.get("min_angle_correction"),
                }
            )
            del config_dict["deskewing"]

        if "table_line_detection" in config_dict:
            line_det = config_dict["table_line_detection"]
            
            config_dict.update(
                {
                    "threshold": line_det.get("threshold"),
                    "horizontal_kernel_size": line_det.get("horizontal_kernel_size"),
                    "vertical_kernel_size": line_det.get("vertical_kernel_size"),
                    "alignment_threshold": line_det.get("alignment_threshold"),
                    "min_aspect_ratio": line_det.get("min_aspect_ratio"),
                    
                    # New separated parameters
                    "h_min_length_image_ratio": line_det.get("h_min_length_image_ratio"),
                    "h_min_length_relative_ratio": line_det.get("h_min_length_relative_ratio"),
                    "v_min_length_image_ratio": line_det.get("v_min_length_image_ratio"),
                    "v_min_length_relative_ratio": line_det.get("v_min_length_relative_ratio"),
                }
            )
            del config_dict["table_line_detection"]

        if "margin_removal" in config_dict:
            margin = config_dict["margin_removal"]
            config_dict.update(
                {
                    "enable_margin_removal": margin.get("enable_margin_removal", False),
                    "blur_kernel_size": margin.get("blur_kernel_size", 7),
                    "black_threshold": margin.get("black_threshold", 50),
                    "content_threshold": margin.get("content_threshold", 200),
                    "morph_kernel_size": margin.get("morph_kernel_size", 25),
                    "min_content_area_ratio": margin.get(
                        "min_content_area_ratio", 0.01
                    ),
                    "margin_padding": margin.get("margin_padding", 5),
                }
            )
            del config_dict["margin_removal"]

        # Handle inscribed_margin_removal specially
        if "inscribed_margin_removal" in config_dict:
            inscribed = config_dict["inscribed_margin_removal"]
            config_dict.update(
                {
                    "inscribed_blur_ksize": inscribed.get("inscribed_blur_ksize"),
                    "inscribed_close_ksize": inscribed.get("inscribed_close_ksize"),
                    "inscribed_close_iter": inscribed.get("inscribed_close_iter"),
                }
            )
            del config_dict["inscribed_margin_removal"]

        # Remove any unrecognized sections
        if "table_fitting" in config_dict:
            del config_dict["table_fitting"]

        return cls(**config_dict)


def get_default_config(config_path: Optional[Path] = None) -> Config:
    """Get default configuration, optionally from JSON file."""
    if config_path and config_path.exists():
        return Config.from_json(config_path)
    return Config()


def get_stage1_config(config_path: Optional[Path] = None) -> Stage1Config:
    """Get Stage 1 configuration, optionally from JSON file."""
    if config_path is None:
        # Try default config path
        default_path = Path("configs/stage1_default.json")
        if default_path.exists():
            config_path = default_path

    if config_path and config_path.exists():
        return Stage1Config.from_json(config_path)
    return Stage1Config()


def get_stage2_config(config_path: Optional[Path] = None) -> Stage2Config:
    """Get Stage 2 configuration, optionally from JSON file."""
    if config_path is None:
        # Try default config path
        default_path = Path("configs/stage2_default.json")
        if default_path.exists():
            config_path = default_path

    if config_path and config_path.exists():
        return Stage2Config.from_json(config_path)
    return Stage2Config()
