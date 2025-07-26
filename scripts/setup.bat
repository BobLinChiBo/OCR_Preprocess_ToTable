@echo off
REM OCR Table Extraction Pipeline - Windows Setup Script
REM Handles environment setup and dependency installation

setlocal enabledelayedexpansion

echo.
echo ============================================
echo OCR Pipeline - Windows Environment Setup
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Display Python version
echo Python version:
python --version

REM Check for virtual environment
if not exist "venv" (
    echo.
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Check if Poetry is available and preferred
poetry --version >nul 2>&1
if not errorlevel 1 (
    echo.
    echo Poetry detected. Using Poetry for dependency management...
    poetry install
    if errorlevel 1 (
        echo ERROR: Poetry installation failed
        pause
        exit /b 1
    )
) else (
    echo.
    echo Poetry not found. Using pip for dependency management...
    
    REM Install main dependencies
    echo Installing core dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install core dependencies
        pause
        exit /b 1
    )
    
    REM Install development dependencies
    echo Installing development dependencies...
    pip install -r requirements-dev.txt
    if errorlevel 1 (
        echo ERROR: Failed to install development dependencies
        pause
        exit /b 1
    )
    
    REM Install package in development mode
    echo Installing package in development mode...
    pip install -e .
    if errorlevel 1 (
        echo ERROR: Failed to install package in development mode
        pause
        exit /b 1
    )
)

REM Setup pre-commit hooks
echo.
echo Setting up pre-commit hooks...
pre-commit install
pre-commit install --hook-type commit-msg

REM Create necessary directories
echo.
echo Creating necessary directories...
if not exist "input\raw_images" mkdir "input\raw_images"
if not exist "output\stage1_initial_processing" mkdir "output\stage1_initial_processing"
if not exist "debug\stage1_debug" mkdir "debug\stage1_debug"
if not exist "debug\stage2_debug" mkdir "debug\stage2_debug"

echo.
echo ============================================
echo Setup completed successfully!
echo ============================================
echo.
echo To activate the environment in future sessions, run:
echo   venv\Scripts\activate.bat
echo.
echo Available commands:
echo   scripts\dev.bat        - Development workflow commands
echo   scripts\test.bat       - Run tests
echo   scripts\run-pipeline.bat - Execute OCR pipeline
echo.
pause