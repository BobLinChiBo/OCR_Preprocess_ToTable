# OCR Pipeline Unicode Setup Script for PowerShell
# ================================================
# This script configures Windows PowerShell for proper Unicode/UTF-8 display

Write-Host ""
Write-Host "========================================"
Write-Host "OCR Pipeline Unicode Setup (PowerShell)"
Write-Host "========================================"
Write-Host ""

# Set console output encoding to UTF-8
Write-Host "Setting console encoding to UTF-8..."
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    [Console]::InputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
    Write-Host "[OK] Console encoding set to UTF-8" -ForegroundColor Green
} catch {
    Write-Host "[WARNING] Failed to set console encoding: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Set Python environment variables
Write-Host ""
Write-Host "Setting Python environment variables..."
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONLEGACYWINDOWSSTDIO = "0"
$env:PYTHONUNBUFFERED = "1"

Write-Host "[OK] PYTHONIOENCODING=utf-8" -ForegroundColor Green
Write-Host "[OK] PYTHONLEGACYWINDOWSSTDIO=0" -ForegroundColor Green
Write-Host "[OK] PYTHONUNBUFFERED=1" -ForegroundColor Green

# Load .env file if it exists
if (Test-Path ".env") {
    Write-Host ""
    Write-Host "Loading .env configuration..."
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^([^=]+)=(.*)$") {
            $key = $matches[1]
            $value = $matches[2]
            Set-Item -Path "env:$key" -Value $value
            Write-Host "[OK] $key=$value" -ForegroundColor Green
        }
    }
} else {
    Write-Host ""
    Write-Host "[INFO] No .env file found - using default settings" -ForegroundColor Cyan
}

# Test Unicode support
Write-Host ""
Write-Host "Testing Unicode support..."
try {
    python -c @"
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
"@
} catch {
    Write-Host "[ERROR] Python Unicode test failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test OCR Pipeline console utilities
Write-Host ""
Write-Host "Testing OCR Pipeline console utilities..."
try {
    python -c @"
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
"@
} catch {
    Write-Host "[ERROR] Console utilities test failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================"
Write-Host "Setup Complete!"
Write-Host "========================================"
Write-Host ""
Write-Host "To make these settings permanent:" -ForegroundColor Yellow
Write-Host "1. Add environment variables to your PowerShell profile"
Write-Host "2. Use Windows Terminal for best Unicode support"
Write-Host "3. Set your IDE/editor to use UTF-8 encoding"
Write-Host ""
Write-Host "For troubleshooting, run: python src/ocr_pipeline/utils/console.py" -ForegroundColor Cyan
Write-Host ""