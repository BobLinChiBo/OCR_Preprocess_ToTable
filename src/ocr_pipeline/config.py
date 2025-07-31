"""Configuration classes for OCR pipeline."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class Config:
    """Simple configuration class."""

    # Input/Output directories
    input_dir: Path = Path("data/input")
    output_dir: Path = Path("data/output")
    debug_dir: Optional[Path] = None

    # Page splitting
    gutter_search_start: float = 0.4
    gutter_search_end: float = 0.6
    min_gutter_width: int = 50

    # Deskewing
    angle_range: int = 5
    angle_step: float = 0.1
    min_angle_correction: float = 0.1

    # Line detection
    min_line_length: int = 100
    max_line_gap: int = 10

    # ROI detection (edge detection preprocessing)
    # CONSERVATIVE SETTINGS for 65-95% area ratio preservation
    enable_roi_detection: bool = True
    roi_detection_method: str = "gabor"  # 'gabor', 'canny_sobel', or 'adaptive_threshold'
    
    # Gabor filter parameters (original method)
    gabor_kernel_size: int = 31
    gabor_sigma: float = 3.0
    gabor_lambda: float = 8.0
    gabor_gamma: float = 0.2
    gabor_binary_threshold: int = 127
    
    # Canny + Sobel edge detection parameters (new method)
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    sobel_kernel_size: int = 3
    gaussian_blur_size: int = 5
    edge_binary_threshold: int = 127
    morphology_kernel_size: int = 3
    
    # Adaptive threshold parameters (new method)
    adaptive_method: str = "gaussian"  # 'mean' or 'gaussian'
    adaptive_block_size: int = 11
    adaptive_C: float = 2.0
    edge_enhancement: bool = True
    
    # ROI cut detection parameters (common to all methods)
    roi_vertical_mode: str = "single_best"  # 'both_sides' or 'single_best'
    roi_horizontal_mode: str = "both_sides"  # 'both_sides' or 'single_best'
    roi_window_size_divisor: int = 20
    roi_min_window_size: int = 10
    roi_min_cut_strength: float = 1000.0  # VERY HIGH = minimal cropping for 65-95% area ratio
    roi_min_confidence_threshold: float = 50.0  # VERY HIGH = very conservative cropping

    # Debug options
    save_debug_images: bool = False
    verbose: bool = False

    def __post_init__(self):
        """Ensure paths are Path objects and validate parameters."""
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        if self.debug_dir:
            self.debug_dir = Path(self.debug_dir)
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate configuration parameters."""
        if not (0.0 <= self.gutter_search_start <= 1.0):
            raise ValueError(f"gutter_search_start must be between 0 and 1, got {self.gutter_search_start}")
        if not (0.0 <= self.gutter_search_end <= 1.0):
            raise ValueError(f"gutter_search_end must be between 0 and 1, got {self.gutter_search_end}")
        if self.gutter_search_start >= self.gutter_search_end:
            raise ValueError("gutter_search_start must be less than gutter_search_end")
        if self.min_gutter_width <= 0:
            raise ValueError(f"min_gutter_width must be positive, got {self.min_gutter_width}")
        if self.angle_range <= 0:
            raise ValueError(f"angle_range must be positive, got {self.angle_range}")
        if self.angle_step <= 0:
            raise ValueError(f"angle_step must be positive, got {self.angle_step}")
        if self.min_line_length <= 0:
            raise ValueError(f"min_line_length must be positive, got {self.min_line_length}")
        if self.max_line_gap < 0:
            raise ValueError(f"max_line_gap must be non-negative, got {self.max_line_gap}")

    def create_output_dirs(self):
        """Create output directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_json(cls, json_path: Path) -> "Config":
        """Load configuration from JSON file."""
        if not json_path.exists():
            raise FileNotFoundError(f"Config file not found: {json_path}")
        
        with open(json_path, 'r') as f:
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
                if key.endswith('_dir') and isinstance(value, str):
                    value = Path(value)
                filtered_config[key] = value
        
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

    # Stage 1 line detection - more permissive
    min_line_length: int = 40
    max_line_gap: int = 15

    # Stage 1 ROI margins (more generous)
    roi_margins_page_1: dict = None
    roi_margins_page_2: dict = None
    roi_margins_default: dict = None

    def __post_init__(self):
        """Initialize Stage 1 specific settings."""
        super().__post_init__()

        # Set default ROI margins for Stage 1
        if self.roi_margins_page_1 is None:
            self.roi_margins_page_1 = {
                "top": 120,
                "bottom": 120,
                "left": 0,
                "right": 100,
            }
        if self.roi_margins_page_2 is None:
            self.roi_margins_page_2 = {
                "top": 120,
                "bottom": 120,
                "left": 60,
                "right": 5,
            }
        if self.roi_margins_default is None:
            self.roi_margins_default = {"top": 60, "bottom": 60, "left": 5, "right": 5}

    def create_output_dirs(self):
        """Create Stage 1 specific output directories."""
        super().create_output_dirs()

        # Create Stage 1 subdirectories
        stage1_dirs = [
            "01_split_pages",
            "02_deskewed",
            "02.5_edge_detection",
            "03_line_detection",
            "04_table_reconstruction",
            "05_cropped_tables",
        ]

        for subdir in stage1_dirs:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Stage1Config":
        """Create Stage1Config from dictionary with special handling for nested structures."""
        # Handle nested structures
        if 'page_splitting' in config_dict:
            page_split = config_dict['page_splitting']
            config_dict.update({
                'gutter_search_start': page_split.get('gutter_search_start'),
                'gutter_search_end': page_split.get('gutter_search_end'),
                'min_gutter_width': page_split.get('min_gutter_width')
            })
        
        if 'deskewing' in config_dict:
            deskew = config_dict['deskewing']
            config_dict.update({
                'angle_range': deskew.get('angle_range'),
                'angle_step': deskew.get('angle_step'),
                'min_angle_correction': deskew.get('min_angle_correction')
            })
        
        if 'line_detection' in config_dict:
            line_det = config_dict['line_detection']
            config_dict.update({
                'min_line_length': line_det.get('min_line_length'),
                'max_line_gap': line_det.get('max_line_gap')
            })
        
        if 'roi_detection' in config_dict:
            roi = config_dict['roi_detection']
            config_dict.update({
                'enable_roi_detection': roi.get('enable_roi_detection'),
                'gabor_kernel_size': roi.get('gabor_kernel_size'),
                'gabor_sigma': roi.get('gabor_sigma'),
                'gabor_lambda': roi.get('gabor_lambda'),
                'gabor_gamma': roi.get('gabor_gamma'),
                'gabor_binary_threshold': roi.get('gabor_binary_threshold'),
                'roi_vertical_mode': roi.get('roi_vertical_mode'),
                'roi_horizontal_mode': roi.get('roi_horizontal_mode'),
                'roi_window_size_divisor': roi.get('roi_window_size_divisor'),
                'roi_min_window_size': roi.get('roi_min_window_size'),
                'roi_min_cut_strength': roi.get('roi_min_cut_strength'),
                'roi_min_confidence_threshold': roi.get('roi_min_confidence_threshold')
            })
        
        # Handle roi_margins specially
        if 'roi_margins' in config_dict:
            margins = config_dict['roi_margins']
            config_dict['roi_margins_page_1'] = margins.get('page_1')
            config_dict['roi_margins_page_2'] = margins.get('page_2')
            config_dict['roi_margins_default'] = margins.get('default')
        
        return super().from_dict(config_dict)


@dataclass
class Stage2Config(Config):
    """Stage 2 Configuration: Refinement processing with precise parameters."""

    # Stage 2 takes input from Stage 1 output
    input_dir: Path = Path("data/output/stage1_initial_processing/05_cropped_tables")
    output_dir: Path = Path("data/output/stage2_refinement")

    # Stage 2 deskewing (fine-tuning)
    angle_range: int = 10
    angle_step: float = 0.2
    min_angle_correction: float = 0.2

    # Stage 2 line detection - more precise
    min_line_length: int = 30
    max_line_gap: int = 5

    # Stage 2 ROI margins (minimal, already cropped)
    roi_margins_page_1: dict = None
    roi_margins_page_2: dict = None
    roi_margins_default: dict = None

    # Disable ROI detection for Stage 2 (already cropped)
    enable_roi_detection: bool = False

    def __post_init__(self):
        """Initialize Stage 2 specific settings."""
        super().__post_init__()

        # Set minimal ROI margins for Stage 2 (images already cropped)
        if self.roi_margins_page_1 is None:
            self.roi_margins_page_1 = {"top": 0, "bottom": 0, "left": 0, "right": 0}
        if self.roi_margins_page_2 is None:
            self.roi_margins_page_2 = {"top": 0, "bottom": 0, "left": 0, "right": 0}
        if self.roi_margins_default is None:
            self.roi_margins_default = {"top": 0, "bottom": 0, "left": 0, "right": 0}

    def create_output_dirs(self):
        """Create Stage 2 specific output directories."""
        super().create_output_dirs()

        # Create Stage 2 subdirectories
        stage2_dirs = [
            "01_deskewed",
            "02_line_detection",
            "03_table_reconstruction",
            "04_fitted_tables",
        ]

        for subdir in stage2_dirs:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Stage2Config":
        """Create Stage2Config from dictionary with special handling for nested structures."""
        # Handle nested structures
        if 'deskewing' in config_dict:
            deskew = config_dict['deskewing']
            config_dict.update({
                'angle_range': deskew.get('angle_range'),
                'angle_step': deskew.get('angle_step'),
                'min_angle_correction': deskew.get('min_angle_correction')
            })
        
        if 'line_detection' in config_dict:
            line_det = config_dict['line_detection']
            config_dict.update({
                'min_line_length': line_det.get('min_line_length'),
                'max_line_gap': line_det.get('max_line_gap')
            })
        
        if 'roi_detection' in config_dict:
            roi = config_dict['roi_detection']
            config_dict.update({
                'enable_roi_detection': roi.get('enable_roi_detection')
            })
        
        # Handle roi_margins specially
        if 'roi_margins' in config_dict:
            margins = config_dict['roi_margins']
            config_dict['roi_margins_page_1'] = margins.get('page_1')
            config_dict['roi_margins_page_2'] = margins.get('page_2')
            config_dict['roi_margins_default'] = margins.get('default')
        
        return super().from_dict(config_dict)


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
