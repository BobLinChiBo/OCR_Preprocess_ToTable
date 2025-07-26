#!/usr/bin/env python3
"""
Python-based Makefile replacement for OCR Table Extraction Pipeline.

This script provides cross-platform development commands without requiring
make or Unix shell utilities. It uses Python-based environment detection
and works on Windows, macOS, and Linux.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional

# Add src to path so we can import our utilities
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from ocr_pipeline.utils.environment import (
        get_environment_info, 
        install_package_with_manager,
        run_command_safely,
        print_environment_status
    )
    from ocr_pipeline.utils.windows_utils import is_windows
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False
    def is_windows():
        return os.name == 'nt'


def run_cmd(cmd: List[str], description: str = "", timeout: int = 120) -> bool:
    """Run a command with error handling."""
    if description:
        print(f"Running: {description}")
    
    if HAS_UTILS:
        success, stdout, stderr = run_command_safely(cmd, timeout)
        if not success:
            print(f"Error: {stderr}")
            if stdout:
                print(f"Output: {stdout}")
        return success
    else:
        # Fallback implementation
        try:
            result = subprocess.run(cmd, check=True, timeout=timeout)
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            print(f"Error: Command failed with exit code {e.returncode}")
            return False
        except subprocess.TimeoutExpired:
            print("Error: Command timed out")
            return False
        except FileNotFoundError:
            print(f"Error: Command not found: {cmd[0]}")
            return False


def get_package_manager() -> str:
    """Get the preferred package manager."""
    if HAS_UTILS:
        env_info = get_environment_info()
        return env_info.preferred_manager or 'pip'
    else:
        # Simple fallback detection
        try:
            subprocess.run(['poetry', '--version'], capture_output=True, check=True)
            return 'poetry'
        except (subprocess.CalledProcessError, FileNotFoundError):
            return 'pip'


def cmd_help():
    """Show help information."""
    print("""
OCR Table Extraction Pipeline - Development Commands

Setup:
  install       Install package in development mode
  install-dev   Install with development dependencies
  setup-dev     Complete development setup (install + pre-commit)

Code Quality:
  format        Format code with black and isort
  lint          Run all linters (ruff, flake8, bandit)
  type-check    Run mypy type checking
  pre-commit    Run all pre-commit hooks

Testing:
  test          Run all tests
  test-fast     Run tests without slow integration tests
  test-cov      Run tests with coverage report

Pipeline Execution:
  stage1        Run OCR Stage 1 processing
  stage2        Run OCR Stage 2 processing
  pipeline      Run complete pipeline (stage1 + stage2)

Quick Development Workflows:
  quick         Run format + lint + test-fast
  dev-check     Run format + lint + type-check + test-fast

Maintenance:
  clean         Clean build artifacts and cache
  build         Build distribution packages
  status        Show environment status

Examples:
  python make.py format
  python make.py dev-check
  python make.py test-fast
""")


def cmd_install():
    """Install package in development mode."""
    print("Installing package in development mode...")
    
    manager = get_package_manager()
    
    if manager == 'poetry':
        return run_cmd(['poetry', 'install'], "Installing with Poetry")
    else:
        return run_cmd(['pip', 'install', '-e', '.'], "Installing with pip (editable)")


def cmd_install_dev():
    """Install with development dependencies."""
    print("Installing with development dependencies...")
    
    manager = get_package_manager()
    
    if manager == 'poetry':
        return run_cmd(['poetry', 'install', '--with', 'dev,test,docs'], 
                      "Installing dev dependencies with Poetry")
    else:
        # Install core dependencies
        success = run_cmd(['pip', 'install', '-r', 'requirements.txt'], 
                         "Installing core dependencies")
        if not success:
            return False
        
        # Install dev dependencies
        success = run_cmd(['pip', 'install', '-r', 'requirements-dev.txt'], 
                         "Installing dev dependencies")
        if not success:
            return False
        
        # Install package in editable mode
        return run_cmd(['pip', 'install', '-e', '.'], 
                      "Installing package in editable mode")


def cmd_setup_dev():
    """Complete development setup."""
    print("Setting up development environment...")
    
    # Install dependencies
    if not cmd_install_dev():
        return False
    
    # Setup pre-commit hooks
    print("Setting up pre-commit hooks...")
    success1 = run_cmd(['pre-commit', 'install'], "Installing pre-commit hooks")
    success2 = run_cmd(['pre-commit', 'install', '--hook-type', 'commit-msg'], 
                      "Installing commit-msg hooks")
    
    if success1 and success2:
        print("✓ Development environment setup complete!")
        return True
    else:
        print("⚠ Development environment setup completed with pre-commit issues")
        return True  # Don't fail the whole setup for pre-commit issues


def cmd_format():
    """Format code with black and isort."""
    print("Formatting code...")
    
    manager = get_package_manager()
    
    if manager == 'poetry':
        success1 = run_cmd(['poetry', 'run', 'black', 'src/', 'tests/'], 
                          "Running black formatter")
        success2 = run_cmd(['poetry', 'run', 'isort', 'src/', 'tests/'], 
                          "Running isort")
    else:
        success1 = run_cmd(['black', 'src/', 'tests/'], "Running black formatter")
        success2 = run_cmd(['isort', 'src/', 'tests/'], "Running isort")
    
    if success1 and success2:
        print("✓ Code formatting complete!")
        return True
    else:
        print("✗ Code formatting failed")
        return False


def cmd_lint():
    """Run all linters."""
    print("Running linters...")
    
    manager = get_package_manager()
    errors = 0
    
    # Run ruff
    if manager == 'poetry':
        if not run_cmd(['poetry', 'run', 'ruff', 'check', 'src/', 'tests/'], 
                      "Running ruff"):
            errors += 1
    else:
        if not run_cmd(['ruff', 'check', 'src/', 'tests/'], "Running ruff"):
            errors += 1
    
    # Run flake8
    if manager == 'poetry':
        if not run_cmd(['poetry', 'run', 'flake8', 'src/', 'tests/'], 
                      "Running flake8"):
            errors += 1
    else:
        if not run_cmd(['flake8', 'src/', 'tests/'], "Running flake8"):
            errors += 1
    
    # Run bandit
    if manager == 'poetry':
        if not run_cmd(['poetry', 'run', 'bandit', '-r', 'src/', '-c', 'pyproject.toml'], 
                      "Running bandit security check"):
            errors += 1
    else:
        if not run_cmd(['bandit', '-r', 'src/', '-c', 'pyproject.toml'], 
                      "Running bandit security check"):
            errors += 1
    
    if errors == 0:
        print("✓ All linting checks passed!")
        return True
    else:
        print(f"✗ {errors} linting check(s) failed")
        return False


def cmd_type_check():
    """Run mypy type checking."""
    print("Running type checking...")
    
    manager = get_package_manager()
    
    if manager == 'poetry':
        success = run_cmd(['poetry', 'run', 'mypy', 'src/'], "Running mypy")
    else:
        success = run_cmd(['mypy', 'src/'], "Running mypy")
    
    if success:
        print("✓ Type checking passed!")
        return True
    else:
        print("✗ Type checking failed")
        return False


def cmd_pre_commit():
    """Run all pre-commit hooks."""
    print("Running pre-commit hooks...")
    
    success = run_cmd(['pre-commit', 'run', '--all-files'], 
                     "Running pre-commit hooks")
    
    if success:
        print("✓ Pre-commit hooks passed!")
        return True
    else:
        print("✗ Pre-commit hooks failed")
        return False


def cmd_test():
    """Run all tests."""
    print("Running all tests...")
    
    manager = get_package_manager()
    
    if manager == 'poetry':
        success = run_cmd(['poetry', 'run', 'pytest'], "Running pytest")
    else:
        success = run_cmd(['pytest'], "Running pytest")
    
    return success


def cmd_test_fast():
    """Run tests without slow integration tests."""
    print("Running fast tests...")
    
    manager = get_package_manager()
    
    if manager == 'poetry':
        success = run_cmd(['poetry', 'run', 'pytest', '-m', 'not slow'], 
                         "Running fast tests")
    else:
        success = run_cmd(['pytest', '-m', 'not slow'], "Running fast tests")
    
    return success


def cmd_test_cov():
    """Run tests with coverage report."""
    print("Running tests with coverage...")
    
    manager = get_package_manager()
    
    if manager == 'poetry':
        success = run_cmd(['poetry', 'run', 'pytest', '--cov=ocr_pipeline', 
                          '--cov-report=html', '--cov-report=term-missing'], 
                         "Running tests with coverage")
    else:
        success = run_cmd(['pytest', '--cov=ocr_pipeline', 
                          '--cov-report=html', '--cov-report=term-missing'], 
                         "Running tests with coverage")
    
    if success:
        print("Coverage report generated in htmlcov/index.html")
    
    return success


def cmd_stage1():
    """Run OCR Stage 1 processing."""
    print("Running OCR Stage 1 processing...")
    
    manager = get_package_manager()
    
    if manager == 'poetry':
        success = run_cmd(['poetry', 'run', 'ocr-stage1'], 
                         "Running Stage 1", timeout=600)
    else:
        success = run_cmd(['ocr-stage1'], "Running Stage 1", timeout=600)
    
    return success


def cmd_stage2():
    """Run OCR Stage 2 processing."""
    print("Running OCR Stage 2 processing...")
    
    manager = get_package_manager()
    
    if manager == 'poetry':
        success = run_cmd(['poetry', 'run', 'ocr-stage2'], 
                         "Running Stage 2", timeout=600)
    else:
        success = run_cmd(['ocr-stage2'], "Running Stage 2", timeout=600)
    
    return success


def cmd_pipeline():
    """Run complete pipeline."""
    print("Running complete OCR pipeline...")
    
    print("Starting Stage 1...")
    if not cmd_stage1():
        print("✗ Pipeline failed at Stage 1")
        return False
    
    print("\nStarting Stage 2...")
    if not cmd_stage2():
        print("✗ Pipeline failed at Stage 2") 
        return False
    
    print("✓ Complete pipeline execution finished!")
    return True


def cmd_quick():
    """Run quick development workflow."""
    print("Running quick development check...")
    
    success = True
    success &= cmd_format()
    success &= cmd_lint()
    success &= cmd_test_fast()
    
    if success:
        print("✓ Quick development check complete!")
    else:
        print("✗ Quick development check failed")
    
    return success


def cmd_dev_check():
    """Run comprehensive development check."""
    print("Running comprehensive development check...")
    
    success = True
    success &= cmd_format()
    success &= cmd_lint()
    success &= cmd_type_check()
    success &= cmd_test_fast()
    
    if success:
        print("✓ Development checks complete!")
    else:
        print("✗ Development checks failed")
    
    return success


def cmd_clean():
    """Clean build artifacts and cache."""
    print("Cleaning build artifacts and cache...")
    
    dirs_to_remove = [
        'build',
        'dist', 
        '*.egg-info',
        '.pytest_cache',
        '.mypy_cache',
        'htmlcov',
        '.ruff_cache'
    ]
    
    files_to_remove = [
        '.coverage'
    ]
    
    import shutil
    import glob
    
    # Remove directories
    for pattern in dirs_to_remove:
        if '*' in pattern:
            for path in glob.glob(pattern):
                if Path(path).is_dir():
                    print(f"Removing {path}")
                    shutil.rmtree(path, ignore_errors=True)
        else:
            path = Path(pattern)
            if path.exists() and path.is_dir():
                print(f"Removing {path}")
                shutil.rmtree(path, ignore_errors=True)
    
    # Remove files
    for pattern in files_to_remove:
        if '*' in pattern:
            for path in glob.glob(pattern):
                if Path(path).is_file():
                    print(f"Removing {path}")
                    Path(path).unlink()
        else:
            path = Path(pattern)
            if path.exists() and path.is_file():
                print(f"Removing {path}")
                path.unlink()
    
    # Remove Python cache files
    print("Removing Python cache files...")
    for path in Path('.').rglob('*.pyc'):
        path.unlink()
    
    for path in Path('.').rglob('__pycache__'):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    
    print("✓ Cleanup complete!")
    return True


def cmd_build():
    """Build distribution packages."""
    print("Building distribution packages...")
    
    # Clean first
    cmd_clean()
    
    manager = get_package_manager()
    
    if manager == 'poetry':
        success = run_cmd(['poetry', 'build'], "Building with Poetry")
    else:
        success = run_cmd(['python', '-m', 'build'], "Building with build module")
    
    return success


def cmd_status():
    """Show environment status."""
    if HAS_UTILS:
        print_environment_status()
    else:
        print("Environment utilities not available")
        print("Run 'python make.py install-dev' to set up the environment")
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OCR Pipeline Development Commands",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        default='help',
        help='Command to run (use "help" for list of commands)'
    )
    
    args = parser.parse_args()
    
    # Command mapping
    commands = {
        'help': cmd_help,
        'install': cmd_install,
        'install-dev': cmd_install_dev,
        'setup-dev': cmd_setup_dev,
        'format': cmd_format,
        'lint': cmd_lint,
        'type-check': cmd_type_check,
        'pre-commit': cmd_pre_commit,
        'test': cmd_test,
        'test-fast': cmd_test_fast,
        'test-cov': cmd_test_cov,
        'stage1': cmd_stage1,
        'stage2': cmd_stage2,
        'pipeline': cmd_pipeline,
        'quick': cmd_quick,
        'dev-check': cmd_dev_check,
        'clean': cmd_clean,
        'build': cmd_build,
        'status': cmd_status,
    }
    
    command_func = commands.get(args.command)
    
    if command_func is None:
        print(f"Unknown command: {args.command}")
        print("Run 'python make.py help' for available commands")
        return 1
    
    try:
        success = command_func()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())