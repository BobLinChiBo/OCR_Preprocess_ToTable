"""Simple configuration for OCR pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Simple configuration class."""
    
    # Input/Output directories
    input_dir: Path = Path("input")
    output_dir: Path = Path("output")
    debug_dir: Optional[Path] = None
    
    # Page splitting
    gutter_search_start: float = 0.4
    gutter_search_end: float = 0.6
    min_gutter_width: int = 50
    
    # Deskewing
    angle_range: int = 45
    angle_step: float = 0.5
    min_angle_correction: float = 0.5
    
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
    roi_vertical_mode: str = 'single_best'  # 'both_sides' or 'single_best'
    roi_horizontal_mode: str = 'both_sides'  # 'both_sides' or 'single_best'
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


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()