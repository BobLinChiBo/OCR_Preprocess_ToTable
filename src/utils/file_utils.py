"""
File handling utilities.

Common file operations and utilities used across the pipeline.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_directory_exists(directory_path: str) -> bool:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        directory_path: Path to directory
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        logger.debug(f"Directory ensured: {directory_path}")
        return True
    except OSError as e:
        logger.error(f"Failed to create directory {directory_path}: {e}")
        return False


def save_json(data: Dict[Any, Any], output_path: str, indent: int = 4) -> bool:
    """
    Save data to JSON file with error handling.
    
    Args:
        data: Data to save
        output_path: Path to output file
        indent: JSON indentation
        
    Returns:
        True if successful
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        
        logger.debug(f"Saved JSON: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving JSON {output_path}: {e}")
        return False


def load_json(file_path: str) -> Optional[Dict[Any, Any]]:
    """
    Load JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded data or None if failed
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"JSON file not found: {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.debug(f"Loaded JSON: {file_path}")
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading JSON {file_path}: {e}")
        return None


def get_base_filename(file_path: str) -> str:
    """
    Get base filename without extension.
    
    Args:
        file_path: Full file path
        
    Returns:
        Base filename without extension
    """
    return os.path.splitext(os.path.basename(file_path))[0]


def get_files_with_extension(directory: str, extension: str) -> List[str]:
    """
    Get all files with specific extension in directory.
    
    Args:
        directory: Directory to search
        extension: File extension (with or without dot)
        
    Returns:
        List of file paths
    """
    if not os.path.isdir(directory):
        logger.warning(f"Directory not found: {directory}")
        return []
    
    if not extension.startswith('.'):
        extension = '.' + extension
    
    files = []
    try:
        for filename in os.listdir(directory):
            if filename.lower().endswith(extension.lower()):
                files.append(os.path.join(directory, filename))
        
        files.sort()
        logger.debug(f"Found {len(files)} {extension} files in {directory}")
        return files
        
    except Exception as e:
        logger.error(f"Error reading directory {directory}: {e}")
        return []


def copy_file(source: str, destination: str) -> bool:
    """
    Copy file with error handling.
    
    Args:
        source: Source file path
        destination: Destination file path
        
    Returns:
        True if successful
    """
    try:
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        shutil.copy2(source, destination)
        logger.debug(f"Copied file: {source} -> {destination}")
        return True
        
    except Exception as e:
        logger.error(f"Error copying file {source} to {destination}: {e}")
        return False


def move_file(source: str, destination: str) -> bool:
    """
    Move file with error handling.
    
    Args:
        source: Source file path
        destination: Destination file path
        
    Returns:
        True if successful
    """
    try:
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        shutil.move(source, destination)
        logger.debug(f"Moved file: {source} -> {destination}")
        return True
        
    except Exception as e:
        logger.error(f"Error moving file {source} to {destination}: {e}")
        return False


def delete_file(file_path: str) -> bool:
    """
    Delete file with error handling.
    
    Args:
        file_path: Path to file to delete
        
    Returns:
        True if successful or file doesn't exist
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Deleted file: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        return False


def get_file_size(file_path: str) -> Optional[int]:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes or None if error
    """
    try:
        return os.path.getsize(file_path)
    except OSError:
        return None


def clean_directory(directory: str, keep_extensions: List[str] = None) -> bool:
    """
    Clean directory, optionally keeping files with specific extensions.
    
    Args:
        directory: Directory to clean
        keep_extensions: List of extensions to keep (e.g., ['.json', '.jpg'])
        
    Returns:
        True if successful
    """
    if not os.path.isdir(directory):
        return True  # Nothing to clean
    
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            if os.path.isfile(file_path):
                if keep_extensions:
                    # Check if file has an extension we want to keep
                    _, ext = os.path.splitext(filename)
                    if ext.lower() in [e.lower() for e in keep_extensions]:
                        continue
                
                os.remove(file_path)
                logger.debug(f"Deleted: {file_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error cleaning directory {directory}: {e}")
        return False


def create_backup(file_path: str, backup_suffix: str = "_backup") -> Optional[str]:
    """
    Create backup of file.
    
    Args:
        file_path: Path to file to backup
        backup_suffix: Suffix to add to backup filename
        
    Returns:
        Path to backup file or None if failed
    """
    if not os.path.exists(file_path):
        return None
    
    try:
        base, ext = os.path.splitext(file_path)
        backup_path = f"{base}{backup_suffix}{ext}"
        
        # If backup already exists, add a number
        counter = 1
        while os.path.exists(backup_path):
            backup_path = f"{base}{backup_suffix}_{counter}{ext}"
            counter += 1
        
        shutil.copy2(file_path, backup_path)
        logger.debug(f"Created backup: {backup_path}")
        return backup_path
        
    except Exception as e:
        logger.error(f"Error creating backup of {file_path}: {e}")
        return None


def validate_file_path(file_path: str, must_exist: bool = True) -> bool:
    """
    Validate file path.
    
    Args:
        file_path: Path to validate
        must_exist: Whether file must exist
        
    Returns:
        True if valid
    """
    if not file_path:
        return False
    
    if must_exist and not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return False
    
    # Check if parent directory exists for new files
    if not must_exist:
        parent_dir = os.path.dirname(file_path)
        if parent_dir and not os.path.exists(parent_dir):
            try:
                os.makedirs(parent_dir, exist_ok=True)
            except OSError as e:
                logger.error(f"Cannot create parent directory for {file_path}: {e}")
                return False
    
    return True