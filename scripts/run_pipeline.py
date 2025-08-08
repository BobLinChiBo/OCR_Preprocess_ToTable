#!/usr/bin/env python3
"""Clean entry point for running the OCR pipeline.

This is the main CLI for the OCR pipeline. It respects config file defaults
and only overrides them when explicitly requested via command line arguments.

Usage:
    python scripts/run_pipeline.py              # Use all config defaults
    python scripts/run_pipeline.py input_dir    # Override input only
    python scripts/run_pipeline.py -o output    # Override output only
    python scripts/run_pipeline.py --help       # Show all options
"""

import sys
from pathlib import Path

# Add project root to Python path (parent of scripts directory)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ocr_pipeline.pipeline import main

if __name__ == "__main__":
    main()