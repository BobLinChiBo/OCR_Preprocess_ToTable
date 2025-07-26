"""
Configuration loading and validation utilities.

Provides consistent configuration loading across all processing modules
with validation and error handling.
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Base configuration class with common settings."""
    
    # Directory paths
    directories: Dict[str, str] = field(default_factory=dict)
    
    # Common settings
    debug_enabled: bool = False
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_directories()
    
    def _validate_directories(self):
        """Validate that required directories are specified."""
        if not self.directories:
            raise ValueError("No directories specified in configuration")
    
    def get_directory(self, key: str, default: Optional[str] = None) -> str:
        """Get directory path with optional default."""
        return self.directories.get(key, default or "")
    
    def create_output_directories(self):
        """Create all output directories if they don't exist."""
        for key, path in self.directories.items():
            if key != 'raw_images' and path:  # Don't create input directory
                try:
                    os.makedirs(path, exist_ok=True)
                    logger.debug(f"Created directory: {path}")
                except OSError as e:
                    logger.error(f"Failed to create directory {path}: {e}")
                    raise


def load_config(config_path: str) -> Config:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        Config object with loaded settings
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
        ValueError: If configuration is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        logger.info(f"Loaded configuration from: {config_path}")
        
        # Create Config object
        config = Config()
        
        # Load directories
        if 'directories' in config_data:
            config.directories = config_data['directories']
        
        # Load additional sections as attributes
        for section_name, section_data in config_data.items():
            if section_name != 'directories':
                setattr(config, section_name, section_data)
        
        # Create output directories
        config.create_output_directories()
        
        return config
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        raise


def validate_config_section(config: Config, section_name: str, required_keys: list) -> bool:
    """
    Validate that a configuration section contains required keys.
    
    Args:
        config: Configuration object
        section_name: Name of the section to validate
        required_keys: List of required keys in the section
        
    Returns:
        True if all required keys are present
        
    Raises:
        ValueError: If required keys are missing
    """
    if not hasattr(config, section_name):
        raise ValueError(f"Missing configuration section: {section_name}")
    
    section = getattr(config, section_name)
    if not isinstance(section, dict):
        raise ValueError(f"Configuration section {section_name} must be a dictionary")
    
    missing_keys = [key for key in required_keys if key not in section]
    if missing_keys:
        raise ValueError(f"Missing required keys in {section_name}: {missing_keys}")
    
    return True


def get_config_value(config: Config, section: str, key: str, default: Any = None) -> Any:
    """
    Safely get a configuration value with optional default.
    
    Args:
        config: Configuration object
        section: Section name
        key: Key within section
        default: Default value if key is missing
        
    Returns:
        Configuration value or default
    """
    if not hasattr(config, section):
        return default
    
    section_data = getattr(config, section)
    if not isinstance(section_data, dict):
        return default
    
    return section_data.get(key, default)


def setup_logging(config: Config):
    """
    Setup logging based on configuration.
    
    Args:
        config: Configuration object
    """
    log_level = getattr(config, 'log_level', 'INFO')
    debug_enabled = getattr(config, 'debug_enabled', False)
    
    if debug_enabled:
        log_level = 'DEBUG'
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Reduce opencv logging noise
    logging.getLogger('cv2').setLevel(logging.WARNING)