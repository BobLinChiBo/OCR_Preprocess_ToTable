"""Configuration classes for OCR pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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
    enable_roi_detection: bool = True
    gabor_kernel_size: int = 31
    gabor_sigma: float = 4.0
    gabor_lambda: float = 8.0
    gabor_gamma: float = 0.2
    gabor_binary_threshold: int = 127
    roi_vertical_mode: str = "single_best"  # 'both_sides' or 'single_best'
    roi_horizontal_mode: str = "both_sides"  # 'both_sides' or 'single_best'
    roi_window_size_divisor: int = 20
    roi_min_window_size: int = 10
    roi_min_cut_strength: float = 20.0  # Reduced from 10.0 to work with lambda=1.0
    roi_min_confidence_threshold: float = 5.0

    # Debug options
    save_debug_images: bool = False
    verbose: bool = False

    def __post_init__(self):
        """Ensure paths are Path objects."""
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        if self.debug_dir:
            self.debug_dir = Path(self.debug_dir)

    def create_output_dirs(self):
        """Create output directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)


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


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def get_stage1_config() -> Stage1Config:
    """Get Stage 1 configuration."""
    return Stage1Config()


def get_stage2_config() -> Stage2Config:
    """Get Stage 2 configuration."""
    return Stage2Config()
