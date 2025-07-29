"""Unicode-safe console output utilities for cross-platform compatibility."""

import sys
import os
from typing import Dict, Any


class ConsoleSymbols:
    """Unicode symbols with ASCII fallbacks for cross-platform compatibility."""
    
    # Success/Status symbols
    SUCCESS = "✓"
    SUCCESS_FALLBACK = "OK"
    
    ERROR = "❌"
    ERROR_FALLBACK = "ERROR"
    
    WARNING = "⚠"
    WARNING_FALLBACK = "WARNING"
    
    INFO = "ℹ"
    INFO_FALLBACK = "INFO"
    
    # Progress symbols
    ARROW_RIGHT = "→"
    ARROW_RIGHT_FALLBACK = "->"
    
    BULLET = "•"
    BULLET_FALLBACK = "*"
    
    CHECK = "✓"
    CHECK_FALLBACK = "+"
    
    CROSS = "✗"
    CROSS_FALLBACK = "x"
    
    # Decorative symbols
    GEAR = "🔧"
    GEAR_FALLBACK = "[TOOL]"
    
    MEMO = "📝"
    MEMO_FALLBACK = "[NOTE]"
    
    TARGET = "🎯"
    TARGET_FALLBACK = "[TARGET]"
    
    ROCKET = "🚀"
    ROCKET_FALLBACK = "[START]"


def can_display_unicode() -> bool:
    """Check if the current console can display Unicode characters."""
    try:
        # Try to encode a Unicode character
        "✓".encode(sys.stdout.encoding or 'utf-8')
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def get_symbol(unicode_symbol: str, fallback: str) -> str:
    """Get Unicode symbol if supported, otherwise return ASCII fallback."""
    if can_display_unicode():
        return unicode_symbol
    return fallback


def safe_print(*args, **kwargs) -> None:
    """Print with Unicode fallback support."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Convert args to safe ASCII equivalents
        safe_args = []
        for arg in args:
            if isinstance(arg, str):
                # Replace common Unicode symbols with ASCII equivalents
                safe_arg = (arg
                           .replace("✓", ConsoleSymbols.SUCCESS_FALLBACK)
                           .replace("❌", ConsoleSymbols.ERROR_FALLBACK)
                           .replace("⚠", ConsoleSymbols.WARNING_FALLBACK)
                           .replace("→", ConsoleSymbols.ARROW_RIGHT_FALLBACK)
                           .replace("•", ConsoleSymbols.BULLET_FALLBACK)
                           .replace("🔧", ConsoleSymbols.GEAR_FALLBACK)
                           .replace("📝", ConsoleSymbols.MEMO_FALLBACK)
                           .replace("🎯", ConsoleSymbols.TARGET_FALLBACK)
                           .replace("🚀", ConsoleSymbols.ROCKET_FALLBACK))
                safe_args.append(safe_arg)
            else:
                safe_args.append(str(arg))
        print(*safe_args, **kwargs)


def print_success(message: str, **kwargs) -> None:
    """Print success message with appropriate symbol."""
    symbol = get_symbol(ConsoleSymbols.SUCCESS, ConsoleSymbols.SUCCESS_FALLBACK)
    safe_print(f"{symbol} {message}", **kwargs)


def print_error(message: str, **kwargs) -> None:
    """Print error message with appropriate symbol."""
    symbol = get_symbol(ConsoleSymbols.ERROR, ConsoleSymbols.ERROR_FALLBACK)
    safe_print(f"{symbol} {message}", **kwargs)


def print_warning(message: str, **kwargs) -> None:
    """Print warning message with appropriate symbol."""
    symbol = get_symbol(ConsoleSymbols.WARNING, ConsoleSymbols.WARNING_FALLBACK)
    safe_print(f"{symbol} {message}", **kwargs)


def print_info(message: str, **kwargs) -> None:
    """Print info message with appropriate symbol."""
    symbol = get_symbol(ConsoleSymbols.INFO, ConsoleSymbols.INFO_FALLBACK)
    safe_print(f"{symbol} {message}", **kwargs)


def print_header(title: str, width: int = 60, char: str = "=") -> None:
    """Print a formatted header."""
    safe_print(title.upper())
    safe_print(char * width)


def print_separator(width: int = 60, char: str = "-") -> None:
    """Print a separator line."""
    safe_print(char * width)


def configure_console_encoding() -> bool:
    """Attempt to configure console for UTF-8 encoding."""
    try:
        # Set environment variables for UTF-8
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        # On Windows, try to set console code page
        if sys.platform == 'win32':
            try:
                import subprocess
                subprocess.run(['chcp', '65001'], shell=True, 
                             capture_output=True, check=False)
            except Exception:
                pass  # Ignore if chcp fails
        
        return True
    except Exception:
        return False


def get_console_info() -> Dict[str, Any]:
    """Get information about current console configuration."""
    return {
        'platform': sys.platform,
        'stdout_encoding': getattr(sys.stdout, 'encoding', 'unknown'),
        'stderr_encoding': getattr(sys.stderr, 'encoding', 'unknown'),
        'unicode_support': can_display_unicode(),
        'pythonioencoding': os.environ.get('PYTHONIOENCODING', 'not set'),
        'python_version': sys.version,
    }


def print_console_info() -> None:
    """Print current console configuration information."""
    info = get_console_info()
    
    print_header("Console Configuration")
    safe_print(f"Platform: {info['platform']}")
    safe_print(f"Python Version: {info['python_version'].split()[0]}")
    safe_print(f"Stdout Encoding: {info['stdout_encoding']}")
    safe_print(f"Stderr Encoding: {info['stderr_encoding']}")
    safe_print(f"PYTHONIOENCODING: {info['pythonioencoding']}")
    
    if info['unicode_support']:
        print_success("Unicode display supported")
    else:
        print_warning("Unicode display not supported - using ASCII fallbacks")
    
    print_separator()


# Example usage and testing
if __name__ == "__main__":
    print_header("Unicode Console Utility Test")
    
    print_success("This is a success message")
    print_error("This is an error message")
    print_warning("This is a warning message")
    print_info("This is an info message")
    
    print_separator()
    
    # Test various symbols
    symbols_to_test = [
        ("Success", ConsoleSymbols.SUCCESS, ConsoleSymbols.SUCCESS_FALLBACK),
        ("Error", ConsoleSymbols.ERROR, ConsoleSymbols.ERROR_FALLBACK),
        ("Warning", ConsoleSymbols.WARNING, ConsoleSymbols.WARNING_FALLBACK),
        ("Arrow", ConsoleSymbols.ARROW_RIGHT, ConsoleSymbols.ARROW_RIGHT_FALLBACK),
        ("Gear", ConsoleSymbols.GEAR, ConsoleSymbols.GEAR_FALLBACK),
    ]
    
    safe_print("Symbol Test:")
    for name, unicode_sym, fallback in symbols_to_test:
        displayed = get_symbol(unicode_sym, fallback)
        safe_print(f"  {name}: {displayed}")
    
    print_separator()
    print_console_info()