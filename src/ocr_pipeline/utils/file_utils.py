"""
Modern file handling utilities with pathlib and improved error handling.

Provides robust file operations used across the OCR pipeline with
proper type hints, logging, and error handling.
"""

import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional, Union, List, Dict
import logging

from ..exceptions import DirectoryError, ValidationError

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def ensure_directory_exists(directory_path: PathLike) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        directory_path: Path to directory
        
    Returns:
        Path object for the directory
        
    Raises:
        DirectoryError: If directory cannot be created
    """
    path = Path(directory_path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ensured: {path}")
        return path
    except OSError as e:
        raise DirectoryError(f"Failed to create directory {path}: {e}")


def get_base_filename(file_path: PathLike) -> str:
    """
    Get base filename without extension.
    
    Args:
        file_path: Full file path
        
    Returns:
        Base filename without extension
    """
    return Path(file_path).stem


def get_file_extension(file_path: PathLike) -> str:
    """
    Get file extension.
    
    Args:
        file_path: Full file path
        
    Returns:
        File extension including the dot (e.g., '.jpg')
    """
    return Path(file_path).suffix


def create_output_path(
    input_path: PathLike,
    output_dir: PathLike,
    suffix: str = "",
    extension: Optional[str] = None
) -> Path:
    """
    Create output path based on input path.
    
    Args:
        input_path: Input file path
        output_dir: Output directory
        suffix: Suffix to add to filename
        extension: New extension (keeps original if None)
        
    Returns:
        Complete output path
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    base_name = input_path.stem + suffix
    if extension is None:
        extension = input_path.suffix
    elif not extension.startswith('.'):
        extension = '.' + extension
    
    return output_dir / (base_name + extension)


@contextmanager
def safe_file_operation(file_path: PathLike, operation: str = "operation") -> Generator[Path, None, None]:
    """
    Context manager for safe file operations with logging.
    
    Args:
        file_path: Path to file
        operation: Description of operation for logging
        
    Yields:
        Path object
        
    Raises:
        Exception: Re-raises any exceptions with additional context
    """
    path = Path(file_path)
    logger.debug(f"Starting {operation} on {path}")
    
    try:
        yield path
        logger.debug(f"Completed {operation} on {path}")
    except Exception as e:
        logger.error(f"Failed {operation} on {path}: {e}")
        raise


def save_json(
    data: Dict[str, Any],
    output_path: PathLike,
    indent: int = 4,
    ensure_ascii: bool = False
) -> None:
    """
    Save data to JSON file with error handling.
    
    Args:
        data: Data to save
        output_path: Path to output file
        indent: JSON indentation
        ensure_ascii: Whether to escape non-ASCII characters
        
    Raises:
        IOError: If file cannot be saved
    """
    path = Path(output_path)
    
    with safe_file_operation(path, "save JSON"):
        # Ensure output directory exists
        ensure_directory_exists(path.parent)
        
        with path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def load_json(file_path: PathLike) -> Dict[str, Any]:
    """
    Load JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
        IOError: If file cannot be read
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with safe_file_operation(path, "load JSON"):
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)


def get_files_with_extensions(
    directory: PathLike,
    extensions: List[str],
    recursive: bool = False
) -> List[Path]:
    """
    Get all files with specific extensions in directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions (with or without dots)
        recursive: Whether to search recursively
        
    Returns:
        List of file paths sorted by name
        
    Raises:
        DirectoryError: If directory doesn't exist
    """
    path = Path(directory)
    
    if not path.is_dir():
        raise DirectoryError(f"Directory not found: {path}")
    
    # Normalize extensions
    normalized_extensions = []
    for ext in extensions:
        if not ext.startswith('.'):
            ext = '.' + ext
        normalized_extensions.append(ext.lower())
    
    files = []
    try:
        pattern = "**/*" if recursive else "*"
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in normalized_extensions:
                files.append(file_path)
        
        files.sort()
        logger.debug(f"Found {len(files)} files with extensions {extensions} in {path}")
        return files
        
    except Exception as e:
        raise DirectoryError(f"Error reading directory {path}: {e}")


def copy_file(source: PathLike, destination: PathLike, create_parents: bool = True) -> None:
    """
    Copy file with error handling.
    
    Args:
        source: Source file path
        destination: Destination file path
        create_parents: Whether to create parent directories
        
    Raises:
        FileNotFoundError: If source file doesn't exist
        IOError: If copy operation fails
    """
    source_path = Path(source)
    dest_path = Path(destination)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    
    with safe_file_operation(dest_path, f"copy from {source_path}"):
        if create_parents:
            ensure_directory_exists(dest_path.parent)
        
        shutil.copy2(source_path, dest_path)


def move_file(source: PathLike, destination: PathLike, create_parents: bool = True) -> None:
    """
    Move file with error handling.
    
    Args:
        source: Source file path
        destination: Destination file path
        create_parents: Whether to create parent directories
        
    Raises:
        FileNotFoundError: If source file doesn't exist
        IOError: If move operation fails
    """
    source_path = Path(source)
    dest_path = Path(destination)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    
    with safe_file_operation(dest_path, f"move from {source_path}"):
        if create_parents:
            ensure_directory_exists(dest_path.parent)
        
        shutil.move(str(source_path), str(dest_path))


def delete_file(file_path: PathLike, missing_ok: bool = True) -> None:
    """
    Delete file with error handling.
    
    Args:
        file_path: Path to file to delete
        missing_ok: Whether to ignore missing files
        
    Raises:
        FileNotFoundError: If file doesn't exist and missing_ok is False
        IOError: If delete operation fails
    """
    path = Path(file_path)
    
    if not path.exists():
        if not missing_ok:
            raise FileNotFoundError(f"File not found: {path}")
        return
    
    with safe_file_operation(path, "delete"):
        path.unlink()


def get_file_size(file_path: PathLike) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
        
    Raises:
        FileNotFoundError: If file doesn't exist
        OSError: If file stats cannot be read
    """
    path = Path(file_path)
    return path.stat().st_size


def clean_directory(
    directory: PathLike,
    keep_extensions: Optional[List[str]] = None,
    recursive: bool = False
) -> int:
    """
    Clean directory, optionally keeping files with specific extensions.
    
    Args:
        directory: Directory to clean
        keep_extensions: List of extensions to keep (e.g., ['.json', '.jpg'])
        recursive: Whether to clean recursively
        
    Returns:
        Number of files deleted
        
    Raises:
        DirectoryError: If directory operations fail
    """
    path = Path(directory)
    
    if not path.is_dir():
        return 0
    
    keep_extensions = keep_extensions or []
    keep_extensions = [ext.lower() if ext.startswith('.') else '.' + ext.lower() 
                      for ext in keep_extensions]
    
    deleted_count = 0
    try:
        pattern = "**/*" if recursive else "*"
        for file_path in path.glob(pattern):
            if file_path.is_file():
                if keep_extensions and file_path.suffix.lower() in keep_extensions:
                    continue
                
                file_path.unlink()
                logger.debug(f"Deleted: {file_path}")
                deleted_count += 1
        
        logger.info(f"Cleaned directory {path}: deleted {deleted_count} files")
        return deleted_count
        
    except Exception as e:
        raise DirectoryError(f"Error cleaning directory {path}: {e}")


def create_backup(
    file_path: PathLike,
    backup_suffix: str = "_backup"
) -> Path:
    """
    Create backup of file.
    
    Args:
        file_path: Path to file to backup
        backup_suffix: Suffix to add to backup filename
        
    Returns:
        Path to backup file
        
    Raises:
        FileNotFoundError: If source file doesn't exist
        IOError: If backup cannot be created
    """
    source_path = Path(file_path)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    
    # Generate unique backup filename
    backup_path = source_path.with_stem(source_path.stem + backup_suffix)
    counter = 1
    while backup_path.exists():
        backup_path = source_path.with_stem(f"{source_path.stem}{backup_suffix}_{counter}")
        counter += 1
    
    with safe_file_operation(backup_path, f"backup of {source_path}"):
        shutil.copy2(source_path, backup_path)
        return backup_path


def validate_path(
    file_path: PathLike,
    must_exist: bool = True,
    must_be_file: bool = True
) -> Path:
    """
    Validate file path.
    
    Args:
        file_path: Path to validate
        must_exist: Whether path must exist
        must_be_file: Whether path must be a file (vs directory)
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If validation fails
    """
    if not file_path:
        raise ValidationError("File path cannot be empty")
    
    path = Path(file_path)
    
    if must_exist:
        if not path.exists():
            raise ValidationError(f"Path does not exist: {path}")
        
        if must_be_file and not path.is_file():
            raise ValidationError(f"Path is not a file: {path}")
    else:
        # For new files, ensure parent directory can be created
        try:
            ensure_directory_exists(path.parent)
        except DirectoryError as e:
            raise ValidationError(f"Cannot create parent directory for {path}: {e}")
    
    return path


def get_directory_stats(directory: PathLike) -> Dict[str, Any]:
    """
    Get statistics about directory contents.
    
    Args:
        directory: Directory to analyze
        
    Returns:
        Dictionary with statistics
        
    Raises:
        DirectoryError: If directory doesn't exist or cannot be read
    """
    path = Path(directory)
    
    if not path.is_dir():
        raise DirectoryError(f"Directory not found: {path}")
    
    stats = {
        "total_files": 0,
        "total_size": 0,
        "file_types": {},
        "largest_file": None,
        "largest_size": 0,
    }
    
    try:
        for file_path in path.rglob("*"):
            if file_path.is_file():
                stats["total_files"] += 1
                
                file_size = file_path.stat().st_size
                stats["total_size"] += file_size
                
                if file_size > stats["largest_size"]:
                    stats["largest_size"] = file_size
                    stats["largest_file"] = str(file_path)
                
                extension = file_path.suffix.lower()
                stats["file_types"][extension] = stats["file_types"].get(extension, 0) + 1
        
        return stats
        
    except Exception as e:
        raise DirectoryError(f"Error analyzing directory {path}: {e}")