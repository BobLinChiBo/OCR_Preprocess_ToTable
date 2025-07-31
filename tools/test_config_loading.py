#!/usr/bin/env python3
"""
Test script to verify config loading functionality for visualization scripts.
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ocr_pipeline.config import Stage1Config, Stage2Config  # noqa: E402


def load_config_from_file(config_path: Path = None, stage: int = 1):
    """Load configuration from JSON file or use defaults."""
    if config_path is None:
        # Use default config based on stage
        if stage == 2:
            config_path = Path("configs/stage2_default.json")
        else:
            config_path = Path("configs/stage1_default.json")

    if config_path.exists():
        if stage == 2:
            return Stage2Config.from_json(config_path)
        else:
            return Stage1Config.from_json(config_path)
    else:
        print(f"Warning: Config file {config_path} not found, using hardcoded defaults")
        if stage == 2:
            return Stage2Config()
        else:
            return Stage1Config()


def test_config_loading():
    """Test config loading with different configurations."""

    print("Testing Config Loading for Visualization Scripts")
    print("=" * 60)

    # Test Stage 1 default config
    print("\n1. Testing Stage 1 Default Config:")
    config1 = load_config_from_file(stage=1)
    print(f"   - deskewing.angle_range: {config1.angle_range}")
    print(f"   - page_splitting.gutter_search_start: {config1.gutter_search_start}")
    print(f"   - page_splitting.gutter_search_end: {config1.gutter_search_end}")
    print(f"   - line_detection.min_line_length: {config1.min_line_length}")
    print(
        f"   - roi_detection.gabor_binary_threshold: {config1.gabor_binary_threshold}"
    )

    # Test Stage 2 default config
    print("\n2. Testing Stage 2 Default Config:")
    config2 = load_config_from_file(stage=2)
    print(f"   - deskewing.angle_range: {config2.angle_range}")
    print(f"   - line_detection.min_line_length: {config2.min_line_length}")
    print(f"   - roi_detection.enable_roi_detection: {config2.enable_roi_detection}")

    # Test explicit config file path
    print("\n3. Testing Explicit Config File Path:")
    config3 = load_config_from_file(Path("configs/stage1_default.json"))
    print("   - Config loaded from: configs/stage1_default.json")
    print(f"   - deskewing.angle_range: {config3.angle_range}")

    # Test parameter override simulation
    print("\n4. Testing Parameter Override Simulation:")
    config4 = load_config_from_file(stage=1)
    original_angle_range = config4.angle_range
    config4.angle_range = 15  # Override from command line
    print(f"   - Original angle_range: {original_angle_range}")
    print(f"   - After override: {config4.angle_range}")

    print("\nConfig loading test completed successfully!")
    print("\nKey differences between Stage 1 and Stage 2:")
    print(f"   - Stage 1 deskewing angle_range: {config1.angle_range}")
    print(f"   - Stage 2 deskewing angle_range: {config2.angle_range}")
    print(f"   - Stage 1 line_detection min_line_length: {config1.min_line_length}")
    print(f"   - Stage 2 line_detection min_line_length: {config2.min_line_length}")
    print(f"   - Stage 1 ROI detection enabled: {config1.enable_roi_detection}")
    print(f"   - Stage 2 ROI detection enabled: {config2.enable_roi_detection}")


if __name__ == "__main__":
    test_config_loading()
