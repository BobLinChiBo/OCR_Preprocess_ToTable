#!/usr/bin/env python3
"""
Parameter Tuning Setup Script - Windows Compatible

This script sets up the directory structure and validates prerequisites
for the OCR parameter tuning process on Windows systems.

Usage:
    python tools/setup_tuning_windows.py
"""

import sys
import subprocess
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    required_packages = ['cv2', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [MISSING] {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install opencv-python numpy")
        return False
    
    print("  [OK] All dependencies found")
    return True


def check_test_images():
    """Check if test images exist."""
    print("\nChecking test images...")
    
    test_dir = Path("data/input/test_images")
    if not test_dir.exists():
        print(f"  [ERROR] Test image directory not found: {test_dir}")
        return False
    
    from src.ocr_pipeline.utils import get_image_files
    image_files = get_image_files(test_dir)
    
    if not image_files:
        print(f"  [ERROR] No image files found in {test_dir}")
        return False
    
    print(f"  [OK] Found {len(image_files)} test images:")
    for img_file in image_files:
        print(f"     - {img_file.name}")
    
    return True


def create_directory_structure():
    """Create all required directories for the tuning process."""
    print("\nCreating directory structure...")
    
    directories = [
        "data/output/tuning",
        "data/output/tuning/01_split_pages",
        "data/output/tuning/02_deskewed_input",
        "data/output/tuning/02_deskewed", 
        "data/output/tuning/03_roi_input",
        "data/output/tuning/03_roi_detection",
        "data/output/tuning/04_line_input",
        "data/output/tuning/04_line_detection",
        "data/output/tuned_pipeline",
        "tools/logs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  [OK] {dir_path}")
    
    print("  [OK] All directories created")


def check_existing_results():
    """Check if any tuning results already exist."""
    print("\nChecking for existing tuning results...")
    
    result_dirs = [
        "data/output/tuning/01_split_pages",
        "data/output/tuning/02_deskewed",
        "data/output/tuning/03_roi_detection", 
        "data/output/tuning/04_line_detection"
    ]
    
    existing_results = []
    for result_dir in result_dirs:
        path = Path(result_dir)
        if path.exists() and any(path.iterdir()):
            existing_results.append(result_dir)
    
    if existing_results:
        print("  [WARNING] Found existing tuning results:")
        for result_dir in existing_results:
            print(f"     - {result_dir}")
        print("  You can continue from where you left off or clean up to start fresh")
        return True
    else:
        print("  [OK] No existing results found - clean slate for tuning")
        return False


def print_usage_instructions():
    """Print step-by-step usage instructions."""
    print("\n" + "="*60)
    print("PARAMETER TUNING SETUP COMPLETE!")
    print("="*60)
    
    print("\nNEXT STEPS - Follow this sequence:")
    print("-" * 40)
    
    print("\n1. PAGE SPLITTING TUNING:")
    print("   python tools\\tune_page_splitting.py")
    print("   -> Evaluate results in data\\output\\tuning\\01_split_pages\\")
    print("   -> Copy best results to data\\output\\tuning\\02_deskewed_input\\")
    
    print("\n2. DESKEWING TUNING:")
    print("   python tools\\tune_deskewing.py")
    print("   -> Evaluate results in data\\output\\tuning\\02_deskewed\\")
    print("   -> Copy best results to data\\output\\tuning\\03_roi_input\\")
    
    print("\n3. ROI DETECTION TUNING:")
    print("   python tools\\tune_roi_detection.py")
    print("   -> Evaluate results in data\\output\\tuning\\03_roi_detection\\")
    print("   -> Copy best ROI images to data\\output\\tuning\\04_line_input\\")
    
    print("\n4. LINE DETECTION TUNING:")
    print("   python tools\\tune_line_detection.py")
    print("   -> Evaluate results in data\\output\\tuning\\04_line_detection\\")
    print("   -> Note optimal parameters")
    
    print("\n5. RUN TUNED PIPELINE:")
    print("   -> Update TUNED_PARAMETERS in tools\\run_tuned_pipeline.py")
    print("   -> python tools\\run_tuned_pipeline.py data\\input\\test_images\\ --verbose")
    
    print("\nTIPS:")
    print("   • Each stage builds on the previous one")
    print("   • Manually evaluate and copy the best results between stages")
    print("   • Save your optimal parameters for future use")
    print("   • Compare tuned results with default pipeline")


def print_file_copy_examples():
    """Print examples of how to copy files between stages."""
    print("\nFILE COPYING EXAMPLES (Windows):")
    print("-" * 30)
    
    print("\nAfter Page Splitting:")
    print('copy "data\\output\\tuning\\01_split_pages\\best_folder\\*.jpg" "data\\output\\tuning\\02_deskewed_input\\"')
    
    print("\nAfter Deskewing:")
    print('copy "data\\output\\tuning\\02_deskewed\\best_folder\\*.jpg" "data\\output\\tuning\\03_roi_input\\"')
    
    print("\nAfter ROI Detection:")
    print('copy "data\\output\\tuning\\03_roi_detection\\best_folder\\*_roi.jpg" "data\\output\\tuning\\04_line_input\\"')


def create_helper_scripts():
    """Create helper scripts for common operations."""
    print("\nCreating helper scripts...")
    
    # Create a Windows batch file for copying
    copy_script_content = '''@echo off
echo OCR Parameter Tuning - File Copy Helper
echo ========================================
echo.
echo This script helps copy files between tuning stages.
echo.
echo Usage Examples:
echo   After Page Splitting:
echo   copy "data\\output\\tuning\\01_split_pages\\best_folder\\*.jpg" "data\\output\\tuning\\02_deskewed_input\\"
echo.
echo   After Deskewing:
echo   copy "data\\output\\tuning\\02_deskewed\\best_folder\\*.jpg" "data\\output\\tuning\\03_roi_input\\"
echo.
echo   After ROI Detection:
echo   copy "data\\output\\tuning\\03_roi_detection\\best_folder\\*_roi.jpg" "data\\output\\tuning\\04_line_input\\"
echo.
echo Replace 'best_folder' with the actual folder name of your best results.
echo.
pause
'''
    
    copy_script_path = Path("tools/windows_copy_helper.bat")
    with open(copy_script_path, 'w') as f:
        f.write(copy_script_content)
    
    print(f"  [OK] Created {copy_script_path}")
    
    # Create a progress tracker
    progress_content = '''# OCR Parameter Tuning Progress Tracker

## Tuning Progress
- [ ] Page Splitting - tune_page_splitting.py
- [ ] Deskewing - tune_deskewing.py  
- [ ] ROI Detection - tune_roi_detection.py
- [ ] Line Detection - tune_line_detection.py
- [ ] Final Pipeline - run_tuned_pipeline.py

## Best Parameters Found
### Page Splitting
- gutter_search_start: 
- gutter_search_end: 

### Deskewing  
- angle_range: 
- angle_step:
- min_angle_correction:

### ROI Detection
- gabor_kernel_size:
- gabor_sigma:
- gabor_lambda:
- roi_min_cut_strength:
- roi_min_confidence_threshold:

### Line Detection
- Stage 1 min_line_length:
- Stage 1 max_line_gap:
- Stage 2 min_line_length:
- Stage 2 max_line_gap:

## Notes
- Best page splitting folder:
- Best deskewing folder:
- Best ROI detection folder:
- Best line detection folder:

## Windows Commands Used
### Page Splitting -> Deskewing
copy "data\\output\\tuning\\01_split_pages\\BEST_FOLDER\\*.jpg" "data\\output\\tuning\\02_deskewed_input\\"

### Deskewing -> ROI Detection  
copy "data\\output\\tuning\\02_deskewed\\BEST_FOLDER\\*.jpg" "data\\output\\tuning\\03_roi_input\\"

### ROI Detection -> Line Detection
copy "data\\output\\tuning\\03_roi_detection\\BEST_FOLDER\\*_roi.jpg" "data\\output\\tuning\\04_line_input\\"
'''
    
    progress_path = Path("tools/tuning_progress_windows.md")
    with open(progress_path, 'w') as f:
        f.write(progress_content)
    
    print(f"  [OK] Created {progress_path}")


def main():
    """Main setup function."""
    print("OCR PARAMETER TUNING SETUP - WINDOWS")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n[ERROR] Setup failed: Missing dependencies")
        return False
    
    # Check test images
    if not check_test_images():
        print("\n[ERROR] Setup failed: Test images not found")
        print("Please ensure test images are in data\\input\\test_images\\")
        return False
    
    # Create directory structure
    create_directory_structure()
    
    # Check existing results
    has_existing = check_existing_results()
    
    # Create helper scripts
    create_helper_scripts()
    
    # Print instructions
    print_usage_instructions()
    print_file_copy_examples()
    
    if has_existing:
        print("\n[WARNING] Existing tuning results detected.")
        print("   You can either continue from where you left off or")
        print("   clean up the directories to start fresh.")
    
    print("\n[SUCCESS] Setup complete! Ready to start parameter tuning.")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)