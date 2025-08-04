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

    # Page splitting
    gutter_search_start: float = 0.4  # DEPRECATED - use search_ratio
    gutter_search_end: float = 0.6  # DEPRECATED - use search_ratio
    min_gutter_width: int = 50  # DEPRECATED - use width_min

    # New page split parameters (robust algorithm)
    search_ratio: float = 0.3  # Fraction of width, centered, to search for gutter
    blur_k: int = 21  # Gaussian blur kernel size (odd number)
    open_k: int = 9  # Morphological opening kernel width
    width_min: int = 20  # Minimum gutter width in pixels

    # Deskewing
    angle_range: int = 5
    angle_step: float = 0.1
    min_angle_correction: float = 0.1

    # Table line detection (new connected components method)
    threshold: int = 40
    horizontal_kernel_size: int = 10
    vertical_kernel_size: int = 10
    alignment_threshold: int = 3
    pre_merge_length_ratio: float = 0.3
    post_merge_length_ratio: float = 0.4
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
        # Legacy page split parameters
        if not (0.0 <= self.gutter_search_start <= 1.0):
            raise ValueError(
                f"gutter_search_start must be between 0 and 1, got {self.gutter_search_start}"
            )
        if not (0.0 <= self.gutter_search_end <= 1.0):
            raise ValueError(
                f"gutter_search_end must be between 0 and 1, got {self.gutter_search_end}"
            )
        if self.gutter_search_start >= self.gutter_search_end:
            raise ValueError("gutter_search_start must be less than gutter_search_end")
        if self.min_gutter_width <= 0:
            raise ValueError(
                f"min_gutter_width must be positive, got {self.min_gutter_width}"
            )

        # New page split parameters
        if not (0.0 <= self.search_ratio <= 1.0):
            raise ValueError(
                f"search_ratio must be between 0 and 1, got {self.search_ratio}"
            )
        if self.blur_k <= 0 or self.blur_k % 2 == 0:
            raise ValueError(f"blur_k must be positive and odd, got {self.blur_k}")
        if self.open_k <= 0:
            raise ValueError(f"open_k must be positive, got {self.open_k}")
        if self.width_min <= 0:
            raise ValueError(f"width_min must be positive, got {self.width_min}")
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
    pre_merge_length_ratio: float = 0.2  # Lower ratio to catch more segments
    post_merge_length_ratio: float = 0.3  # More permissive final filtering
    min_aspect_ratio: int = 3  # Lower aspect ratio for initial detection

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
            "01_split_pages",
            "02_margin_removed",
            "03_deskewed",
            "04_table_lines",
            "05_table_structure",
            "06_border_cropped",
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
                    "gutter_search_start": page_split.get("gutter_search_start"),
                    "gutter_search_end": page_split.get("gutter_search_end"),
                    "min_gutter_width": page_split.get("min_gutter_width"),
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
                    "pre_merge_length_ratio": line_det.get("pre_merge_length_ratio"),
                    "post_merge_length_ratio": line_det.get("post_merge_length_ratio"),
                    "min_aspect_ratio": line_det.get("min_aspect_ratio"),
                }
            )
            del config_dict["table_line_detection"]

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
    input_dir: Path = Path("data/output/stage1_initial_processing/06_border_cropped")
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
    pre_merge_length_ratio: float = 0.4  # Higher ratio for quality filtering
    post_merge_length_ratio: float = 0.5  # More strict final filtering
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
                    "pre_merge_length_ratio": line_det.get("pre_merge_length_ratio"),
                    "post_merge_length_ratio": line_det.get("post_merge_length_ratio"),
                    "min_aspect_ratio": line_det.get("min_aspect_ratio"),
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
