@echo off
REM OCR Table Extraction Pipeline - Testing Commands
REM Windows testing script with various options

setlocal enabledelayedexpansion

if "%~1"=="" (
    set TEST_MODE=all
) else (
    set TEST_MODE=%~1
)

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo WARNING: Virtual environment not activated
    echo Run 'venv\Scripts\activate.bat' first
    echo.
)

REM Determine package manager
set USE_POETRY=0
poetry --version >nul 2>&1
if not errorlevel 1 (
    set USE_POETRY=1
)

echo.
echo OCR Pipeline - Running Tests (%TEST_MODE% mode)
echo ============================================

if "%TEST_MODE%"=="help" goto :help
if "%TEST_MODE%"=="all" goto :testall
if "%TEST_MODE%"=="fast" goto :testfast
if "%TEST_MODE%"=="unit" goto :testunit
if "%TEST_MODE%"=="integration" goto :testintegration
if "%TEST_MODE%"=="coverage" goto :testcoverage
if "%TEST_MODE%"=="cov" goto :testcoverage
if "%TEST_MODE%"=="slow" goto :testslow
if "%TEST_MODE%"=="gpu" goto :testgpu
if "%TEST_MODE%"=="specific" goto :testspecific

echo Unknown test mode: %TEST_MODE%
echo Run 'scripts\test.bat help' for available options
exit /b 1

:help
echo.
echo OCR Table Extraction Pipeline - Test Commands
echo.
echo Usage: scripts\test.bat [MODE] [OPTIONS]
echo.
echo Test Modes:
echo   all           Run all tests (default)
echo   fast          Run tests without slow integration tests
echo   unit          Run only unit tests
echo   integration   Run only integration tests
echo   coverage      Run tests with coverage report
echo   slow          Run only slow tests
echo   gpu           Run only GPU-dependent tests
echo   specific      Run specific test file (requires TEST_FILE env var)
echo.
echo Examples:
echo   scripts\test.bat fast
echo   scripts\test.bat coverage
echo   set TEST_FILE=tests\unit\test_config.py && scripts\test.bat specific
echo.
echo Pytest Markers Available:
echo   -m "unit"         Unit tests only
echo   -m "integration"  Integration tests only
echo   -m "slow"         Slow tests only
echo   -m "not slow"     Skip slow tests
echo   -m "gpu"          GPU tests only
echo.
exit /b 0

:testall
echo Running all tests...
if %USE_POETRY%==1 (
    poetry run pytest
) else (
    pytest
)
exit /b %errorlevel%

:testfast
echo Running fast tests (excluding slow integration tests)...
if %USE_POETRY%==1 (
    poetry run pytest -m "not slow"
) else (
    pytest -m "not slow"
)
exit /b %errorlevel%

:testunit
echo Running unit tests only...
if %USE_POETRY%==1 (
    poetry run pytest -m "unit"
) else (
    pytest -m "unit"
)
exit /b %errorlevel%

:testintegration
echo Running integration tests only...
if %USE_POETRY%==1 (
    poetry run pytest -m "integration"
) else (
    pytest -m "integration"
)
exit /b %errorlevel%

:testcoverage
echo Running tests with coverage report...
if %USE_POETRY%==1 (
    poetry run pytest --cov=ocr_pipeline --cov-report=html --cov-report=term-missing
) else (
    pytest --cov=ocr_pipeline --cov-report=html --cov-report=term-missing
)
echo.
echo Coverage report generated in htmlcov\index.html
exit /b %errorlevel%

:testslow
echo Running slow tests only...
if %USE_POETRY%==1 (
    poetry run pytest -m "slow"
) else (
    pytest -m "slow"
)
exit /b %errorlevel%

:testgpu
echo Running GPU tests only...
if %USE_POETRY%==1 (
    poetry run pytest -m "gpu"
) else (
    pytest -m "gpu"
)
exit /b %errorlevel%

:testspecific
if not defined TEST_FILE (
    echo ERROR: TEST_FILE environment variable not set
    echo Usage: set TEST_FILE=tests\unit\test_config.py && scripts\test.bat specific
    exit /b 1
)
echo Running specific test file: %TEST_FILE%
if %USE_POETRY%==1 (
    poetry run pytest "%TEST_FILE%"
) else (
    pytest "%TEST_FILE%"
)
exit /b %errorlevel%