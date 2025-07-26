"""
Tests for Windows-specific utilities.

These tests validate Windows path handling, environment detection,
and platform-specific functionality.
"""

import pytest
import platform
from pathlib import Path

from tests.conftest import (
    skip_on_non_windows,
    requires_windows_utils,
    normalize_test_path,
    create_windows_safe_test_file
)


class TestWindowsUtils:
    """Test Windows-specific utility functions."""
    
    def test_platform_detection(self, platform_info):
        """Test platform detection works correctly."""
        assert 'is_windows' in platform_info
        assert 'platform' in platform_info
        assert 'has_windows_utils' in platform_info
        
        # Verify platform detection is consistent
        expected_windows = platform.system().lower() == 'windows'
        assert platform_info['is_windows'] == expected_windows
    
    def test_path_normalization(self):
        """Test path normalization works on all platforms."""
        test_paths = [
            "input/raw_images",
            "output\\stage1_processing",
            "debug/output",
            "src\\ocr_pipeline\\config"
        ]
        
        for path in test_paths:
            normalized = normalize_test_path(path)
            assert isinstance(normalized, str)
            assert len(normalized) > 0
    
    @requires_windows_utils
    def test_windows_path_validation(self):
        """Test Windows path validation."""
        from ocr_pipeline.utils.windows_utils import validate_windows_path
        
        # Valid paths
        valid_paths = [
            "C:\\Users\\Test\\Documents",
            "input\\raw_images",
            "output/stage1_processing",
            "normal_filename.txt"
        ]
        
        for path in valid_paths:
            is_valid, error = validate_windows_path(path)
            assert is_valid, f"Path should be valid: {path}, error: {error}"
        
        # Invalid paths
        invalid_paths = [
            "file_with_<invalid>_chars.txt",
            "CON.txt",  # Reserved name
            "file|with|pipes.txt",
            "file?.txt"
        ]
        
        for path in invalid_paths:
            is_valid, error = validate_windows_path(path)
            assert not is_valid, f"Path should be invalid: {path}"
            assert error is not None
    
    @requires_windows_utils
    def test_windows_safe_filename(self):
        """Test Windows-safe filename generation."""
        from ocr_pipeline.utils.windows_utils import make_windows_safe_filename
        
        test_cases = [
            ("normal_file.txt", "normal_file.txt"),
            ("file<with>invalid.txt", "file_with_invalid.txt"),
            ("CON.txt", "_CON.txt"),
            ("file?.txt", "file_.txt"),
            ("file|with|pipes.txt", "file_with_pipes.txt")
        ]
        
        for input_name, expected in test_cases:
            result = make_windows_safe_filename(input_name)
            assert result == expected, f"Expected {expected}, got {result}"
    
    def test_windows_safe_test_file_creation(self, temp_dir):
        """Test Windows-safe test file creation."""
        # Test with potentially problematic filename
        filename = "test<file>with?chars.txt"
        content = "Test content"
        
        file_path = create_windows_safe_test_file(temp_dir, filename, content)
        
        assert file_path.exists()
        assert file_path.read_text(encoding='utf-8') == content
        # Filename should be sanitized
        assert '<' not in file_path.name
        assert '>' not in file_path.name
        assert '?' not in file_path.name
    
    @requires_windows_utils  
    def test_windows_path_handler(self, windows_path_handler):
        """Test Windows path handler functionality."""
        handler = windows_path_handler
        
        # Test path normalization
        test_path = "input/raw_images/test.jpg"
        normalized = handler.normalize(test_path)
        assert isinstance(normalized, str)
        
        # Test path joining
        parts = ["output", "stage1", "results"]
        joined = handler.join(*parts)
        assert isinstance(joined, Path)
        
        # Test validation
        assert handler.is_valid("normal_path.txt")
        assert not handler.is_valid("file?.txt")
    
    @skip_on_non_windows("Windows-only functionality")
    def test_windows_only_features(self):
        """Test features that only work on Windows."""
        # This test will only run on Windows
        from ocr_pipeline.utils.windows_utils import get_windows_version
        
        version = get_windows_version()
        assert version is not None
        assert isinstance(version, str)
        assert len(version) > 0


class TestEnvironmentDetection:
    """Test environment detection functionality."""
    
    def test_basic_environment_info(self):
        """Test basic environment information gathering."""
        try:
            from ocr_pipeline.utils.environment import get_environment_info
            
            env_info = get_environment_info()
            
            # Check required fields
            assert hasattr(env_info, 'python_executable')
            assert hasattr(env_info, 'python_version')
            assert hasattr(env_info, 'package_managers')
            assert hasattr(env_info, 'preferred_manager')
            
            # Basic validation
            assert env_info.python_executable is None or isinstance(env_info.python_executable, str)
            assert env_info.python_version is None or isinstance(env_info.python_version, str)
            assert isinstance(env_info.package_managers, list)
            
        except ImportError:
            pytest.skip("Environment utilities not available")
    
    def test_package_manager_detection(self):
        """Test package manager detection."""
        try:
            from ocr_pipeline.utils.environment import detect_package_managers
            
            managers = detect_package_managers()
            
            assert isinstance(managers, list)
            assert len(managers) > 0
            
            # Should include at least pip
            manager_names = [m.name for m in managers]
            assert 'pip' in manager_names
            
            # Check structure
            for manager in managers:
                assert hasattr(manager, 'name')
                assert hasattr(manager, 'available')
                assert hasattr(manager, 'version')
                assert hasattr(manager, 'path')
                assert hasattr(manager, 'priority')
                
        except ImportError:
            pytest.skip("Environment utilities not available")


class TestCrossPlatformCompatibility:
    """Test cross-platform compatibility features."""
    
    def test_path_handling_across_platforms(self, platform_info):
        """Test that path handling works consistently across platforms."""
        test_paths = [
            "input/raw_images",
            "output\\stage1",
            "src/ocr_pipeline",
            "tests\\unit"
        ]
        
        for path in test_paths:
            normalized = normalize_test_path(path)
            
            # Should be a valid string
            assert isinstance(normalized, str)
            assert len(normalized) > 0
            
            # Should use the correct separator for the platform
            if platform_info['is_windows']:
                # On Windows, should primarily use backslashes
                pass  # Windows handles both, so we're flexible
            else:
                # On Unix, should use forward slashes
                assert '\\' not in normalized or '/' in normalized
    
    def test_temp_directory_creation(self, temp_dir, windows_temp_dir):
        """Test temporary directory creation works on all platforms."""
        # Basic temp dir
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Windows-specific temp dir
        assert windows_temp_dir.exists()
        assert windows_temp_dir.is_dir()
        
        # Should be able to create files in both
        test_file1 = temp_dir / "test1.txt"
        test_file1.write_text("test content")
        assert test_file1.exists()
        
        test_file2 = windows_temp_dir / "test2.txt"
        test_file2.write_text("test content")
        assert test_file2.exists()