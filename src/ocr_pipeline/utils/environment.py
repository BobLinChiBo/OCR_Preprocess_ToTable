"""
Environment detection and management utilities.

Provides cross-platform environment detection, package manager discovery,
and development environment setup utilities.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
import logging

logger = logging.getLogger(__name__)


class PackageManagerInfo(NamedTuple):
    """Information about a package manager."""
    name: str
    available: bool
    version: Optional[str]
    path: Optional[str]
    priority: int


class EnvironmentInfo(NamedTuple):
    """Comprehensive environment information."""
    python_executable: Optional[str]
    python_version: Optional[str]
    virtual_env: Optional[str]
    package_managers: List[PackageManagerInfo]
    preferred_manager: Optional[str]
    is_development_mode: bool
    can_install_packages: bool


def run_command_safely(command: List[str], timeout: int = 10) -> Tuple[bool, str, str]:
    """
    Run a command safely with error handling.
    
    Args:
        command: Command and arguments as list
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (success, stdout, stderr)
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except FileNotFoundError:
        return False, "", "Command not found"
    except Exception as e:
        return False, "", str(e)


def find_executable(name: str) -> Optional[str]:
    """
    Find executable in PATH.
    
    Args:
        name: Executable name
        
    Returns:
        Full path to executable or None if not found
    """
    return shutil.which(name)


def get_python_info() -> Tuple[Optional[str], Optional[str]]:
    """
    Get Python executable path and version.
    
    Returns:
        Tuple of (executable_path, version_string)
    """
    # Try different Python executable names
    python_names = ['python', 'python3', 'py']
    
    for name in python_names:
        executable = find_executable(name)
        if executable:
            success, stdout, _ = run_command_safely([executable, '--version'])
            if success:
                return executable, stdout
    
    # Fallback to sys.executable if running in Python
    try:
        success, stdout, _ = run_command_safely([sys.executable, '--version'])
        if success:
            return sys.executable, stdout
    except Exception:
        pass
    
    return None, None


def detect_virtual_environment() -> Optional[str]:
    """
    Detect if running in a virtual environment.
    
    Returns:
        Virtual environment path or None
    """
    # Check common virtual environment indicators
    venv_indicators = [
        'VIRTUAL_ENV',
        'CONDA_DEFAULT_ENV',
        'PIPENV_ACTIVE',
        'POETRY_ACTIVE'
    ]
    
    for indicator in venv_indicators:
        if indicator in os.environ:
            env_path = os.environ[indicator]
            if indicator == 'CONDA_DEFAULT_ENV':
                # For conda, we need to construct the full path
                conda_prefix = os.environ.get('CONDA_PREFIX')
                if conda_prefix:
                    return conda_prefix
                return env_path
            return env_path
    
    # Check if Python executable is in a virtual environment
    python_exe, _ = get_python_info()
    if python_exe:
        python_path = Path(python_exe)
        
        # Check for common virtual environment patterns
        venv_patterns = ['venv', 'env', '.venv', '.env', 'Scripts', 'bin']
        
        for parent in python_path.parents:
            if any(pattern in parent.name.lower() for pattern in venv_patterns):
                return str(parent)
    
    return None


def detect_package_managers() -> List[PackageManagerInfo]:
    """
    Detect available package managers.
    
    Returns:
        List of package manager information, sorted by priority
    """
    managers = [
        ('poetry', 1),
        ('pipenv', 2),
        ('conda', 3),
        ('pip', 4)
    ]
    
    detected = []
    
    for name, priority in managers:
        executable = find_executable(name)
        if executable:
            success, version_output, _ = run_command_safely([name, '--version'])
            
            detected.append(PackageManagerInfo(
                name=name,
                available=True,
                version=version_output if success else None,
                path=executable,
                priority=priority
            ))
        else:
            detected.append(PackageManagerInfo(
                name=name,
                available=False,
                version=None,
                path=None,
                priority=priority
            ))
    
    # Sort by priority (lower number = higher priority)
    return sorted(detected, key=lambda x: x.priority)


def get_preferred_package_manager() -> Optional[str]:
    """
    Get the preferred package manager based on availability and priority.
    
    Returns:
        Name of preferred package manager or None
    """
    managers = detect_package_managers()
    
    for manager in managers:
        if manager.available:
            return manager.name
    
    return None


def check_development_mode() -> bool:
    """
    Check if the package is installed in development mode.
    
    Returns:
        True if in development mode
    """
    try:
        # Try to import the package and check if it's editable
        import pkg_resources
        
        try:
            dist = pkg_resources.get_distribution('ocr-table-extraction')
            # Check if it's an editable installation
            return hasattr(dist, 'location') and 'egg-link' in str(dist.location)
        except pkg_resources.DistributionNotFound:
            # Package not installed yet
            return False
            
    except ImportError:
        # pkg_resources not available, try alternative method
        try:
            import ocr_pipeline
            module_path = Path(ocr_pipeline.__file__).parent
            cwd = Path.cwd()
            
            # If module is in current directory tree, likely development mode
            try:
                module_path.relative_to(cwd)
                return True
            except ValueError:
                return False
                
        except ImportError:
            return False


def can_install_packages() -> bool:
    """
    Check if we can install packages in the current environment.
    
    Returns:
        True if package installation is possible
    """
    preferred_manager = get_preferred_package_manager()
    if not preferred_manager:
        return False
    
    # Test if we can run the package manager
    success, _, _ = run_command_safely([preferred_manager, '--help'])
    return success


def get_environment_info() -> EnvironmentInfo:
    """
    Get comprehensive environment information.
    
    Returns:
        EnvironmentInfo object with all detected information
    """
    python_exe, python_version = get_python_info()
    virtual_env = detect_virtual_environment()
    package_managers = detect_package_managers()
    preferred_manager = get_preferred_package_manager()
    is_dev_mode = check_development_mode()
    can_install = can_install_packages()
    
    return EnvironmentInfo(
        python_executable=python_exe,
        python_version=python_version,
        virtual_env=virtual_env,
        package_managers=package_managers,
        preferred_manager=preferred_manager,
        is_development_mode=is_dev_mode,
        can_install_packages=can_install
    )


def install_package_with_manager(manager: str, 
                                packages: List[str], 
                                dev: bool = False,
                                editable: bool = False) -> bool:
    """
    Install packages using specified package manager.
    
    Args:
        manager: Package manager name
        packages: List of packages to install
        dev: Install as development dependencies
        editable: Install in editable mode
        
    Returns:
        True if installation succeeded
    """
    if manager == 'poetry':
        cmd = ['poetry', 'install']
        if dev:
            cmd.extend(['--with', 'dev,test,docs'])
    elif manager == 'pipenv':
        cmd = ['pipenv', 'install']
        if dev:
            cmd.append('--dev')
        if packages != ['.']:
            cmd.extend(packages)
    elif manager == 'conda':
        cmd = ['conda', 'install', '-y']
        cmd.extend(packages)
    elif manager == 'pip':
        cmd = ['pip', 'install']
        if editable and '.' in packages:
            # Replace '.' with '-e .' for editable install
            packages = ['-e' if p == '.' else p for p in packages]
        cmd.extend(packages)
    else:
        logger.error(f"Unknown package manager: {manager}")
        return False
    
    success, stdout, stderr = run_command_safely(cmd, timeout=300)  # 5 minute timeout
    
    if success:
        logger.info(f"Successfully installed packages with {manager}")
        return True
    else:
        logger.error(f"Failed to install packages with {manager}: {stderr}")
        return False


def setup_development_environment(force_recreate: bool = False) -> bool:
    """
    Set up development environment automatically.
    
    Args:
        force_recreate: Force recreation of virtual environment
        
    Returns:
        True if setup succeeded
    """
    env_info = get_environment_info()
    
    logger.info("Setting up development environment...")
    logger.info(f"Python: {env_info.python_version}")
    logger.info(f"Virtual environment: {env_info.virtual_env or 'None'}")
    logger.info(f"Preferred package manager: {env_info.preferred_manager}")
    
    # Check if we're in a virtual environment
    if not env_info.virtual_env:
        logger.warning("No virtual environment detected")
        logger.info("Please activate a virtual environment or run scripts/setup.bat")
        return False
    
    # Install packages based on preferred manager
    if env_info.preferred_manager == 'poetry':
        success = install_package_with_manager('poetry', [], dev=True)
    elif env_info.preferred_manager in ['pip']:
        # Install core dependencies first
        success = install_package_with_manager('pip', ['-r', 'requirements.txt'])
        if success:
            # Install dev dependencies
            success = install_package_with_manager('pip', ['-r', 'requirements-dev.txt'])
        if success:
            # Install package in editable mode
            success = install_package_with_manager('pip', ['.'], editable=True)
    else:
        logger.error(f"Unsupported package manager: {env_info.preferred_manager}")
        return False
    
    if not success:
        logger.error("Failed to install dependencies")
        return False
    
    # Set up pre-commit hooks if available
    if find_executable('pre-commit'):
        logger.info("Setting up pre-commit hooks...")
        success1, _, _ = run_command_safely(['pre-commit', 'install'])
        success2, _, _ = run_command_safely(['pre-commit', 'install', '--hook-type', 'commit-msg'])
        
        if success1 and success2:
            logger.info("Pre-commit hooks installed successfully")
        else:
            logger.warning("Failed to install pre-commit hooks")
    
    logger.info("Development environment setup completed!")
    return True


def print_environment_status():
    """Print detailed environment status for debugging."""
    env_info = get_environment_info()
    
    print("Environment Status:")
    print("=" * 50)
    print(f"Python Executable: {env_info.python_executable}")
    print(f"Python Version: {env_info.python_version}")
    print(f"Virtual Environment: {env_info.virtual_env or 'Not Active'}")
    print(f"Development Mode: {env_info.is_development_mode}")
    print(f"Can Install Packages: {env_info.can_install_packages}")
    print()
    
    print("Package Managers:")
    print("-" * 30)
    for manager in env_info.package_managers:
        status = "✓" if manager.available else "✗"
        version = f" ({manager.version})" if manager.version else ""
        print(f"  {status} {manager.name}{version}")
    
    print(f"\nPreferred Manager: {env_info.preferred_manager or 'None'}")


if __name__ == "__main__":
    # Command line interface for environment detection
    import argparse
    
    parser = argparse.ArgumentParser(description="Environment detection utility")
    parser.add_argument('--status', action='store_true', 
                       help='Print environment status')
    parser.add_argument('--setup', action='store_true',
                       help='Set up development environment')
    parser.add_argument('--force', action='store_true',
                       help='Force recreation of environment')
    
    args = parser.parse_args()
    
    if args.status:
        print_environment_status()
    elif args.setup:
        success = setup_development_environment(args.force)
        sys.exit(0 if success else 1)
    else:
        print_environment_status()