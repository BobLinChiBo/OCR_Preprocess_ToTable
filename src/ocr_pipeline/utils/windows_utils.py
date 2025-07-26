"""
Windows-specific utilities for OCR pipeline.

Provides Windows-compatible path handling, environment detection,
and system integration for the OCR table extraction pipeline.
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path, PureWindowsPath, PurePosixPath
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

logger = logging.getLogger(__name__)

# Windows-specific constants
WINDOWS_RESERVED_NAMES = {
    'CON', 'PRN', 'AUX', 'NUL',
    'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
    'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
}

WINDOWS_INVALID_CHARS = '<>:"|?*'
MAX_WINDOWS_PATH_LENGTH = 260
MAX_WINDOWS_FILENAME_LENGTH = 255


def is_windows() -> bool:
    """Check if running on Windows."""
    return platform.system().lower() == 'windows'


def get_windows_version() -> Optional[str]:
    """Get Windows version information."""
    if not is_windows():
        return None
    
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                           r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
        version = winreg.QueryValueEx(key, "ProductName")[0]
        winreg.CloseKey(key)
        return version
    except Exception:
        return platform.platform()


def normalize_path_for_windows(path: Union[str, Path]) -> str:
    """
    Normalize path for Windows compatibility.
    
    Args:
        path: Input path (string or Path object)
        
    Returns:
        Windows-compatible path string
    """
    if isinstance(path, Path):
        path = str(path)
    
    # Convert forward slashes to backslashes on Windows
    if is_windows():
        path = path.replace('/', '\\')
    else:
        # On non-Windows, keep forward slashes but normalize
        path = path.replace('\\', '/')
    
    # Remove duplicate separators
    if is_windows():
        while '\\\\' in path:
            path = path.replace('\\\\', '\\')
    else:
        while '//' in path:
            path = path.replace('//', '/')
    
    return path


def make_windows_safe_filename(filename: str) -> str:
    """
    Make filename safe for Windows file system.
    
    Args:
        filename: Original filename
        
    Returns:
        Windows-safe filename
    """
    # Remove or replace invalid characters
    safe_filename = filename
    for char in WINDOWS_INVALID_CHARS:
        safe_filename = safe_filename.replace(char, '_')
    
    # Handle reserved names
    name_part = Path(safe_filename).stem.upper()
    if name_part in WINDOWS_RESERVED_NAMES:
        safe_filename = f"_{safe_filename}"
    
    # Truncate if too long
    if len(safe_filename) > MAX_WINDOWS_FILENAME_LENGTH:
        name = Path(safe_filename).stem
        ext = Path(safe_filename).suffix
        max_name_len = MAX_WINDOWS_FILENAME_LENGTH - len(ext)
        safe_filename = name[:max_name_len] + ext
    
    # Remove trailing dots and spaces
    safe_filename = safe_filename.rstrip('. ')
    
    return safe_filename


def create_windows_compatible_path(base_path: Union[str, Path], 
                                 relative_path: str) -> Path:
    """
    Create a Windows-compatible path by joining base and relative paths.
    
    Args:
        base_path: Base directory path
        relative_path: Relative path to join
        
    Returns:
        Windows-compatible Path object
    """
    base = Path(base_path)
    
    # Normalize the relative path
    rel_parts = relative_path.replace('\\', '/').split('/')
    safe_parts = [make_windows_safe_filename(part) for part in rel_parts if part]
    
    result_path = base
    for part in safe_parts:
        result_path = result_path / part
    
    return result_path


def ensure_directory_exists(directory: Union[str, Path], 
                          create_parents: bool = True) -> bool:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        directory: Directory path to check/create
        create_parents: Whether to create parent directories
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        path = Path(directory)
        
        if path.exists():
            if path.is_dir():
                return True
            else:
                logger.error(f"Path exists but is not a directory: {path}")
                return False
        
        # Create directory
        path.mkdir(parents=create_parents, exist_ok=True)
        logger.debug(f"Created directory: {path}")
        return True
        
    except PermissionError:
        logger.error(f"Permission denied creating directory: {directory}")
        return False
    except OSError as e:
        logger.error(f"Failed to create directory {directory}: {e}")
        return False


def get_available_drive_letters() -> List[str]:
    """
    Get list of available drive letters on Windows.
    
    Returns:
        List of available drive letters
    """
    if not is_windows():
        return []
    
    used_drives = [drive.split(':')[0].upper() for drive in 
                   subprocess.check_output(['wmic', 'logicaldisk', 'get', 'size,freespace,caption'], 
                                         shell=True, text=True).strip().split('\n')[1:] 
                   if drive.strip()]
    
    all_drives = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    return [drive for drive in all_drives if drive not in used_drives]


def get_disk_usage(path: Union[str, Path]) -> Dict[str, int]:
    """
    Get disk usage statistics for a path.
    
    Args:
        path: Path to check
        
    Returns:
        Dictionary with 'total', 'used', and 'free' bytes
    """
    try:
        if is_windows():
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            total_bytes = ctypes.c_ulonglong(0)
            
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(str(path)),
                ctypes.pointer(free_bytes),
                ctypes.pointer(total_bytes),
                None
            )
            
            total = total_bytes.value
            free = free_bytes.value
            used = total - free
            
        else:
            # Unix-like systems
            stat = shutil.disk_usage(path)
            total, used, free = stat.total, stat.used, stat.free
        
        return {
            'total': total,
            'used': used,
            'free': free
        }
        
    except Exception as e:
        logger.error(f"Failed to get disk usage for {path}: {e}")
        return {'total': 0, 'used': 0, 'free': 0}


def format_windows_path_for_display(path: Union[str, Path]) -> str:
    """
    Format path for user-friendly display on Windows.
    
    Args:
        path: Path to format
        
    Returns:
        Formatted path string
    """
    path_str = str(path)
    
    if is_windows():
        # Convert to Windows-style path
        path_str = normalize_path_for_windows(path_str)
        
        # Make relative to current directory if possible
        try:
            abs_path = Path(path_str).resolve()
            current_dir = Path.cwd()
            
            if abs_path.is_relative_to(current_dir):
                rel_path = abs_path.relative_to(current_dir)
                return f".\\{rel_path}"
        except (ValueError, OSError):
            pass
    
    return path_str


def find_python_executable() -> Optional[str]:
    """
    Find Python executable on Windows system.
    
    Returns:
        Path to Python executable or None if not found
    """
    # Try common Python executables
    python_names = ['python', 'python3', 'py']
    
    for name in python_names:
        try:
            result = subprocess.run([name, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # Get full path
                which_result = subprocess.run(['where' if is_windows() else 'which', name],
                                            capture_output=True, text=True, timeout=10)
                if which_result.returncode == 0:
                    return which_result.stdout.strip().split('\n')[0]
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    return None


def detect_package_manager() -> Dict[str, Any]:
    """
    Detect available Python package managers and their preferences.
    
    Returns:
        Dictionary with package manager information
    """
    managers = {
        'poetry': {'available': False, 'version': None, 'path': None},
        'pip': {'available': False, 'version': None, 'path': None},
        'conda': {'available': False, 'version': None, 'path': None},
        'pipenv': {'available': False, 'version': None, 'path': None}
    }
    
    for manager in managers:
        try:
            result = subprocess.run([manager, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                managers[manager]['available'] = True
                managers[manager]['version'] = result.stdout.strip()
                
                # Get path
                which_result = subprocess.run(['where' if is_windows() else 'which', manager],
                                            capture_output=True, text=True, timeout=10)
                if which_result.returncode == 0:
                    managers[manager]['path'] = which_result.stdout.strip().split('\n')[0]
        except Exception:
            continue
    
    # Determine preference order
    preferred_order = ['poetry', 'pipenv', 'conda', 'pip']
    preferred = None
    
    for manager in preferred_order:
        if managers[manager]['available']:
            preferred = manager
            break
    
    return {
        'managers': managers,
        'preferred': preferred,
        'has_venv': 'VIRTUAL_ENV' in os.environ or 'CONDA_DEFAULT_ENV' in os.environ
    }


def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive Windows system information.
    
    Returns:
        Dictionary with system information
    """
    info = {
        'platform': platform.platform(),
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'python_implementation': platform.python_implementation(),
        'is_windows': is_windows(),
        'windows_version': get_windows_version() if is_windows() else None,
        'environment': {
            'virtual_env': os.environ.get('VIRTUAL_ENV'),
            'conda_env': os.environ.get('CONDA_DEFAULT_ENV'),
            'path_separator': os.pathsep,
            'line_separator': os.linesep
        }
    }
    
    # Add package manager information
    info['package_managers'] = detect_package_manager()
    
    # Add Python executable path
    info['python_executable'] = find_python_executable()
    
    return info


def validate_windows_path(path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
    """
    Validate path for Windows compatibility.
    
    Args:
        path: Path to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    path_str = str(path)
    
    # Check length
    if len(path_str) > MAX_WINDOWS_PATH_LENGTH:
        return False, f"Path too long ({len(path_str)} > {MAX_WINDOWS_PATH_LENGTH})"
    
    # Check for invalid characters
    for char in WINDOWS_INVALID_CHARS:
        if char in path_str:
            return False, f"Invalid character '{char}' in path"
    
    # Check individual path components
    parts = Path(path_str).parts
    for part in parts:
        if len(part) > MAX_WINDOWS_FILENAME_LENGTH:
            return False, f"Filename too long: {part}"
        
        if part.upper() in WINDOWS_RESERVED_NAMES:
            return False, f"Reserved filename: {part}"
        
        if part.endswith('.') or part.endswith(' '):
            return False, f"Filename cannot end with dot or space: {part}"
    
    return True, None


class WindowsPathHandler:
    """
    Windows-specific path handling utility class.
    """
    
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize Windows path handler.
        
        Args:
            base_path: Base directory for relative path operations
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.is_windows = is_windows()
        
    def normalize(self, path: Union[str, Path]) -> str:
        """Normalize path for current platform."""
        return normalize_path_for_windows(path)
    
    def join(self, *parts: str) -> Path:
        """Join path parts safely."""
        result = self.base_path
        for part in parts:
            safe_part = make_windows_safe_filename(part)
            result = result / safe_part
        return result
    
    def ensure_dir(self, path: Union[str, Path]) -> bool:
        """Ensure directory exists."""
        return ensure_directory_exists(path)
    
    def is_valid(self, path: Union[str, Path]) -> bool:
        """Check if path is valid for Windows."""
        valid, _ = validate_windows_path(path)
        return valid
    
    def make_relative(self, path: Union[str, Path]) -> Path:
        """Make path relative to base path."""
        abs_path = Path(path).resolve()
        try:
            return abs_path.relative_to(self.base_path.resolve())
        except ValueError:
            return abs_path
    
    def get_display_path(self, path: Union[str, Path]) -> str:
        """Get user-friendly display path."""
        return format_windows_path_for_display(path)