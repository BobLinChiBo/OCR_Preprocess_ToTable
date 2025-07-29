@echo off
REM OCR Pipeline Unicode Setup Script for Windows
REM ==============================================
REM This script configures Windows console for proper Unicode/UTF-8 display

echo.
echo ========================================
echo OCR Pipeline Unicode Setup
echo ========================================
echo.

REM Set console code page to UTF-8 (65001)
echo Setting console code page to UTF-8...
chcp 65001 >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Console code page set to UTF-8 ^(65001^)
) else (
    echo [WARNING] Failed to set console code page
)

REM Set Python environment variables for UTF-8
echo.
echo Setting Python environment variables...
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=0
set PYTHONUNBUFFERED=1

echo [OK] PYTHONIOENCODING=utf-8
echo [OK] PYTHONLEGACYWINDOWSSTDIO=0
echo [OK] PYTHONUNBUFFERED=1

REM Load .env file if it exists
if exist ".env" (
    echo.
    echo Loading .env configuration...
    for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
        if not "%%a"=="" if not "%%b"=="" (
            set "%%a=%%b"
            echo [OK] %%a=%%b
        )
    )
) else (
    echo.
    echo [INFO] No .env file found - using default settings
)

REM Test Unicode support
echo.
echo Testing Unicode support...
python -c "
import sys
try:
    # Test basic Unicode characters
    test_chars = ['✓', 'ℹ', '→']
    for char in test_chars:
        print(f'Testing {char}: OK')
    print('[SUCCESS] Unicode display is working!')
except UnicodeEncodeError:
    print('[INFO] Unicode display not supported - will use ASCII fallbacks')
except Exception as e:
    print(f'[ERROR] Test failed: {e}')
"

echo.
echo Testing OCR Pipeline console utilities...
python -c "
import sys
sys.path.insert(0, 'src')
try:
    from ocr_pipeline.utils.console import print_success, print_error, print_warning, print_info
    print_success('Unicode console utilities loaded successfully!')
    print_info('Console configuration complete')
except ImportError as e:
    print(f'[WARNING] Could not import console utilities: {e}')
except Exception as e:
    print(f'[ERROR] Console utilities test failed: {e}')
"

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To make these settings permanent:
echo 1. Add environment variables to your system/user PATH
echo 2. Use Windows Terminal instead of Command Prompt for better Unicode support
echo 3. Consider setting your IDE/editor to use UTF-8 encoding
echo.
echo For troubleshooting, run: python src/ocr_pipeline/utils/console.py
echo.