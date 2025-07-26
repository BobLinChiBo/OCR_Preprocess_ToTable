@echo off
REM OCR Table Extraction Pipeline - Development Commands
REM Windows equivalent of Makefile development targets

setlocal enabledelayedexpansion

if "%~1"=="" (
    goto :help
)

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo WARNING: Virtual environment not activated
    echo Run 'venv\Scripts\activate.bat' first, or use scripts\setup.bat
    echo.
)

REM Determine package manager
set USE_POETRY=0
poetry --version >nul 2>&1
if not errorlevel 1 (
    set USE_POETRY=1
)

REM Parse command
if "%~1"=="help" goto :help
if "%~1"=="format" goto :format
if "%~1"=="lint" goto :lint
if "%~1"=="type-check" goto :typecheck
if "%~1"=="pre-commit" goto :precommit
if "%~1"=="install" goto :install
if "%~1"=="install-dev" goto :installdev
if "%~1"=="clean" goto :clean
if "%~1"=="build" goto :build
if "%~1"=="quick" goto :quick
if "%~1"=="dev-check" goto :devcheck

echo Unknown command: %~1
echo Run 'scripts\dev.bat help' for available commands
exit /b 1

:help
echo.
echo OCR Table Extraction Pipeline - Development Commands
echo.
echo Setup:
echo   install       Install package in development mode
echo   install-dev   Install with development dependencies
echo.
echo Code Quality:
echo   format        Format code with black and isort
echo   lint          Run all linters (ruff, flake8, bandit)
echo   type-check    Run mypy type checking
echo   pre-commit    Run all pre-commit hooks
echo.
echo Development Workflows:
echo   quick         Run format + lint + test-fast
echo   dev-check     Run format + lint + type-check + test-fast
echo.
echo Maintenance:
echo   clean         Clean build artifacts and cache
echo   build         Build distribution packages
echo.
echo Examples:
echo   scripts\dev.bat format
echo   scripts\dev.bat lint
echo   scripts\dev.bat quick
echo.
exit /b 0

:format
echo Formatting code with black and isort...
if %USE_POETRY%==1 (
    poetry run black src\ tests\
    poetry run isort src\ tests\
) else (
    black src\ tests\
    isort src\ tests\
)
echo Code formatting complete!
exit /b 0

:lint
echo Running linters...
if %USE_POETRY%==1 (
    poetry run ruff check src\ tests\
    poetry run flake8 src\ tests\
    poetry run bandit -r src\ -c pyproject.toml
) else (
    ruff check src\ tests\
    flake8 src\ tests\
    bandit -r src\ -c pyproject.toml
)
echo Linting complete!
exit /b 0

:typecheck
echo Running mypy type checking...
if %USE_POETRY%==1 (
    poetry run mypy src\
) else (
    mypy src\
)
echo Type checking complete!
exit /b 0

:precommit
echo Running pre-commit hooks...
pre-commit run --all-files
exit /b 0

:install
echo Installing package in development mode...
if %USE_POETRY%==1 (
    poetry install
) else (
    pip install -e .
)
exit /b 0

:installdev
echo Installing with development dependencies...
if %USE_POETRY%==1 (
    poetry install --with dev,test,docs
) else (
    pip install -e ".[dev,test,docs]"
)
exit /b 0

:clean
echo Cleaning build artifacts and cache...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.egg-info rmdir /s /q *.egg-info
if exist .pytest_cache rmdir /s /q .pytest_cache
if exist .mypy_cache rmdir /s /q .mypy_cache
if exist .coverage del .coverage
if exist htmlcov rmdir /s /q htmlcov
for /r . %%i in (*.pyc) do del "%%i"
for /d /r . %%i in (__pycache__) do rmdir /s /q "%%i" 2>nul
echo Cleanup complete!
exit /b 0

:build
echo Building distribution packages...
call :clean
if %USE_POETRY%==1 (
    poetry build
) else (
    python -m build
)
exit /b 0

:quick
echo Running quick development workflow...
call :format
call :lint
call scripts\test.bat fast
echo Quick development check complete!
exit /b 0

:devcheck
echo Running comprehensive development check...
call :format
call :lint
call :typecheck
call scripts\test.bat fast
echo Development checks complete!
exit /b 0