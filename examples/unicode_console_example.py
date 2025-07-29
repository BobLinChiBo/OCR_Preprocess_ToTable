#!/usr/bin/env python3
"""
Example: Using Unicode-Safe Console Utilities
==============================================

This example demonstrates how to use the OCR Pipeline's Unicode-safe console
utilities for cross-platform compatible output.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ocr_pipeline.utils.console import (
    print_success, print_error, print_warning, print_info,
    print_header, print_separator, safe_print,
    get_symbol, ConsoleSymbols, can_display_unicode,
    configure_console_encoding, print_console_info
)


def demonstrate_basic_usage():
    """Demonstrate basic console utility functions."""
    print_header("Basic Console Utilities Demo")
    
    print_success("Successfully loaded image dataset")
    print_info("Processing 150 images in batch mode")
    print_warning("Low memory detected - using conservative settings")
    print_error("Failed to process corrupted image: sample_001.jpg")
    
    print_separator()


def demonstrate_symbol_usage():
    """Demonstrate manual symbol usage with fallbacks."""
    print_header("Symbol Usage with Fallbacks")
    
    # Manual symbol usage
    check = get_symbol(ConsoleSymbols.SUCCESS, ConsoleSymbols.SUCCESS_FALLBACK)
    arrow = get_symbol(ConsoleSymbols.ARROW_RIGHT, ConsoleSymbols.ARROW_RIGHT_FALLBACK)
    gear = get_symbol(ConsoleSymbols.GEAR, ConsoleSymbols.GEAR_FALLBACK)
    
    safe_print(f"{check} Image preprocessing complete")
    safe_print(f"{arrow} Moving to deskewing stage")
    safe_print(f"{gear} Applying histogram variance optimization")
    
    print_separator()


def demonstrate_progress_indicators():
    """Demonstrate progress indication with Unicode-safe symbols."""
    print_header("Progress Indicators")
    
    import time
    
    stages = [
        "Loading configuration",
        "Initializing OCR pipeline", 
        "Processing page splitting",
        "Applying deskew correction",
        "Detecting table lines",
        "Cropping table regions"
    ]
    
    for i, stage in enumerate(stages, 1):
        # Simulate some work
        time.sleep(0.1)
        
        if i == len(stages):
            print_success(f"Stage {i}/{len(stages)}: {stage}")
        else:
            bullet = get_symbol(ConsoleSymbols.BULLET, ConsoleSymbols.BULLET_FALLBACK)
            safe_print(f"{bullet} Stage {i}/{len(stages)}: {stage}")
    
    print_separator()


def demonstrate_error_handling():
    """Demonstrate error handling with different message types."""
    print_header("Error Handling Demo")
    
    # Simulate different types of messages
    scenarios = [
        ("success", "OCR pipeline completed successfully", "All 25 images processed"),
        ("info", "Configuration loaded", "Using stage1_default.json"),
        ("warning", "Suboptimal image quality detected", "Consider increasing resolution"),
        ("error", "Critical error in line detection", "No table lines found in image"),
    ]
    
    for msg_type, title, details in scenarios:
        if msg_type == "success":
            print_success(f"{title}: {details}")
        elif msg_type == "info":
            print_info(f"{title}: {details}")
        elif msg_type == "warning":
            print_warning(f"{title}: {details}")
        elif msg_type == "error":
            print_error(f"{title}: {details}")
    
    print_separator()


def demonstrate_console_detection():
    """Demonstrate console capability detection."""
    print_header("Console Capability Detection")
    
    unicode_supported = can_display_unicode()
    
    if unicode_supported:
        safe_print("✓ Unicode symbols are fully supported")
        safe_print("🎯 Using enhanced visual indicators")
    else:
        safe_print("+ Unicode symbols not supported")
        safe_print("[TARGET] Using ASCII fallback indicators")
    
    # Show encoding configuration
    print_info("Current console configuration:")
    print_console_info()


def demonstrate_real_world_usage():
    """Demonstrate real-world usage in OCR pipeline context."""
    print_header("Real-World OCR Pipeline Usage")
    
    # Simulate OCR pipeline stages
    try:
        print_info("Starting OCR table extraction pipeline")
        
        # Stage 1: Page splitting
        safe_print("Stage 1: Page splitting")
        arrow = get_symbol(ConsoleSymbols.ARROW_RIGHT, ConsoleSymbols.ARROW_RIGHT_FALLBACK)
        safe_print(f"  {arrow} Detecting gutter region")
        safe_print(f"  {arrow} Splitting double-page scan")
        print_success("Page splitting completed: 2 pages extracted")
        
        # Stage 2: Deskewing  
        safe_print("Stage 2: Deskewing correction")
        safe_print(f"  {arrow} Analyzing histogram variance")
        safe_print(f"  {arrow} Testing rotation angles: -10° to +10°")
        print_success("Deskewing completed: -2.3° correction applied")
        
        # Stage 3: Line detection
        safe_print("Stage 3: Table line detection")
        safe_print(f"  {arrow} Applying morphological operations")
        safe_print(f"  {arrow} Using Hough line detection")
        print_success("Line detection completed: 15 horizontal, 8 vertical lines")
        
        # Final result
        gear = get_symbol(ConsoleSymbols.GEAR, ConsoleSymbols.GEAR_FALLBACK)
        target = get_symbol(ConsoleSymbols.TARGET, ConsoleSymbols.TARGET_FALLBACK)
        
        print_separator(width=40)
        safe_print(f"{gear} Pipeline execution complete!")
        safe_print(f"{target} Output saved to: output/processed_tables/")
        print_success("OCR table extraction successful")
        
    except Exception as e:
        print_error(f"Pipeline failed: {str(e)}")
    
    print_separator()


def main():
    """Main example function."""
    try:
        # Try to configure console encoding
        configure_console_encoding()
        
        # Run all demonstrations
        demonstrate_basic_usage()
        demonstrate_symbol_usage()
        demonstrate_progress_indicators()
        demonstrate_error_handling()
        demonstrate_console_detection()
        demonstrate_real_world_usage()
        
        # Final summary
        print_header("Example Complete", width=50)
        print_success("All Unicode console utility examples completed successfully!")
        print_info("Check the output above to see Unicode fallback behavior")
        
        if can_display_unicode():
            safe_print("🎉 Your console fully supports Unicode symbols!")
        else:
            safe_print("[PARTY] Your console uses ASCII fallbacks (works perfectly!)")
            
    except KeyboardInterrupt:
        print_warning("Example interrupted by user")
    except Exception as e:
        print_error(f"Example failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())