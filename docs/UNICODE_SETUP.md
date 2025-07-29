# Unicode Display Setup Guide

This guide resolves Unicode display issues in the OCR Pipeline project on Windows systems.

## Problem

Windows Command Prompt uses `cp1252` encoding by default, which cannot display Unicode symbols like ✓, ❌, 🔧, causing `UnicodeEncodeError` when running Python scripts.

## Quick Solution

### Option 1: Automatic Setup (Recommended)

Run the setup script to configure your environment:

```bash
# For Command Prompt
scripts\setup_unicode.bat

# For PowerShell  
scripts\setup_unicode.ps1
```

### Option 2: Manual Environment Setup

Set environment variables before running Python:

```bash
# Windows Command Prompt
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=0
python your_script.py

# PowerShell
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONLEGACYWINDOWSSTDIO="0"
python your_script.py

# Bash/Git Bash
export PYTHONIOENCODING=utf-8
export PYTHONLEGACYWINDOWSSTDIO=0
python your_script.py
```

### Option 3: Use Unicode-Safe Functions

The project includes Unicode-safe console utilities that automatically fall back to ASCII:

```python
from ocr_pipeline.utils.console import print_success, print_error, print_warning

# These work on any system
print_success("Operation completed successfully!")
print_error("Something went wrong!")
print_warning("This is a warning message")
```

## Permanent Solution

### Method 1: .env File (Recommended)

The project includes a `.env` file with proper encoding settings. Make sure it's in your project root:

```bash
# .env file contents
PYTHONIOENCODING=utf-8
PYTHONLEGACYWINDOWSSTDIO=0
PYTHONUNBUFFERED=1
```

### Method 2: System Environment Variables

Add these to your Windows system environment variables:

1. Open "System Properties" → "Advanced" → "Environment Variables"
2. Add these user/system variables:
   - `PYTHONIOENCODING` = `utf-8`
   - `PYTHONLEGACYWINDOWSSTDIO` = `0`

### Method 3: Python Profile

Add to your Python startup script or IDE configuration:

```python
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONLEGACYWINDOWSSTDIO'] = '0'
```

## Terminal Recommendations

For best Unicode support on Windows:

1. **Windows Terminal** (Recommended) - Excellent Unicode support
2. **PowerShell 7+** - Good Unicode support
3. **Git Bash** - Good Unicode support  
4. **Command Prompt** - Limited, but works with proper encoding setup
5. **VS Code Terminal** - Inherits from your default terminal

## Troubleshooting

### Test Unicode Support

Run this diagnostic script:

```bash
python src/ocr_pipeline/utils/console.py
```

This will show:
- Current console encoding
- Unicode support status
- Symbol display test
- Fallback behavior

### Common Issues

**Issue**: `UnicodeEncodeError: 'charmap' codec can't encode character`

**Solution**: Run setup script or set `PYTHONIOENCODING=utf-8`

**Issue**: Symbols display as question marks

**Solution**: Use Windows Terminal or ensure console font supports Unicode

**Issue**: Setup script doesn't work

**Solution**: Run PowerShell as Administrator or use manual environment setup

### Verification

After setup, this should work without errors:

```python
python -c "print('✓ Unicode test successful!')"
```

## Development Guidelines

When adding new Unicode symbols to the project:

1. **Always provide ASCII fallbacks**:
   ```python
   from ocr_pipeline.utils.console import get_symbol, ConsoleSymbols
   
   symbol = get_symbol(ConsoleSymbols.SUCCESS, ConsoleSymbols.SUCCESS_FALLBACK)
   print(f"{symbol} Operation complete")
   ```

2. **Use safe print functions**:
   ```python
   from ocr_pipeline.utils.console import safe_print, print_success
   
   safe_print("This handles Unicode safely")
   print_success("This is even better")
   ```

3. **Test on different terminals**:
   - Windows Command Prompt
   - PowerShell
   - Windows Terminal
   - Git Bash

## Technical Details

### Encoding Hierarchy

1. **PYTHONIOENCODING** environment variable (highest priority)
2. **Console encoding** (Windows: cp1252, Linux: utf-8)
3. **System locale** (fallback)

### Why This Works

- `PYTHONIOENCODING=utf-8` forces Python to use UTF-8 for stdin/stdout
- `PYTHONLEGACYWINDOWSSTDIO=0` enables proper Unicode handling on Windows
- Console utilities detect encoding support and provide fallbacks
- Setup scripts configure both Python and console environments

### Cross-Platform Compatibility

The solution works on:
- ✅ Windows 10/11 (Command Prompt, PowerShell, Windows Terminal)
- ✅ Linux (most terminals support UTF-8 by default)
- ✅ macOS (Terminal.app supports UTF-8 by default)
- ✅ WSL/WSL2 (inherits Linux UTF-8 support)

## References

- [Python Unicode HOWTO](https://docs.python.org/3/howto/unicode.html)
- [Windows Console Unicode Support](https://docs.microsoft.com/en-us/windows/console/)
- [PYTHONIOENCODING documentation](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONIOENCODING)