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