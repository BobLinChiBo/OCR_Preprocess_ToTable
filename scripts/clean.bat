@echo off
REM OCR Table Extraction Pipeline - Cleanup Script
REM Removes build artifacts, cache files, and temporary data

setlocal enabledelayedexpansion

if "%~1"=="" (
    set CLEAN_MODE=all
) else (
    set CLEAN_MODE=%~1
)

echo.
echo OCR Pipeline - Cleanup Utility
echo ===============================

if "%CLEAN_MODE%"=="help" goto :help
if "%CLEAN_MODE%"=="all" goto :cleanall
if "%CLEAN_MODE%"=="build" goto :cleanbuild
if "%CLEAN_MODE%"=="cache" goto :cleancache
if "%CLEAN_MODE%"=="output" goto :cleanoutput
if "%CLEAN_MODE%"=="debug" goto :cleandebug
if "%CLEAN_MODE%"=="venv" goto :cleanvenv

echo Unknown clean mode: %CLEAN_MODE%
echo Run 'scripts\clean.bat help' for available options
exit /b 1

:help
echo.
echo OCR Table Extraction Pipeline - Cleanup Commands
echo.
echo Usage: scripts\clean.bat [MODE]
echo.
echo Cleanup Modes:
echo   all       Clean everything (build, cache, output, debug)
echo   build     Clean build artifacts and distribution files
echo   cache     Clean Python cache files and pytest cache
echo   output    Clean pipeline output directories
echo   debug     Clean debug output directories
echo   venv      Remove virtual environment (requires confirmation)
echo.
echo Examples:
echo   scripts\clean.bat build
echo   scripts\clean.bat cache
echo   scripts\clean.bat all
echo.
exit /b 0

:cleanall
echo Cleaning all artifacts...
call :cleanbuild
call :cleancache
call :cleanoutput
call :cleandebug
echo.
echo Complete cleanup finished!
exit /b 0

:cleanbuild
echo Cleaning build artifacts...
if exist build (
    echo Removing build\
    rmdir /s /q build
)
if exist dist (
    echo Removing dist\
    rmdir /s /q dist
)
for /d %%i in (*.egg-info) do (
    if exist "%%i" (
        echo Removing %%i
        rmdir /s /q "%%i"
    )
)
echo Build artifacts cleaned!
exit /b 0

:cleancache
echo Cleaning cache files...
if exist .pytest_cache (
    echo Removing .pytest_cache\
    rmdir /s /q .pytest_cache
)
if exist .mypy_cache (
    echo Removing .mypy_cache\
    rmdir /s /q .mypy_cache
)
if exist .coverage (
    echo Removing .coverage
    del .coverage
)
if exist htmlcov (
    echo Removing htmlcov\
    rmdir /s /q htmlcov
)
if exist .ruff_cache (
    echo Removing .ruff_cache\
    rmdir /s /q .ruff_cache
)

REM Remove Python cache files
echo Removing Python cache files...
for /r . %%i in (*.pyc) do (
    del "%%i" 2>nul
)
for /d /r . %%i in (__pycache__) do (
    rmdir /s /q "%%i" 2>nul
)

echo Cache files cleaned!
exit /b 0

:cleanoutput
echo Cleaning output directories...
set /p confirm="This will delete all pipeline output data. Continue? (y/N): "
if /i not "%confirm%"=="y" (
    echo Cancelled.
    exit /b 0
)

if exist output (
    echo Removing output\
    rmdir /s /q output
)
echo Output directories cleaned!
exit /b 0

:cleandebug
echo Cleaning debug directories...
if exist debug (
    echo Removing debug\
    rmdir /s /q debug
)
echo Debug directories cleaned!
exit /b 0

:cleanvenv
echo This will completely remove the virtual environment.
echo You will need to run scripts\setup.bat to recreate it.
echo.
set /p confirm="Are you sure you want to remove the virtual environment? (y/N): "
if /i not "%confirm%"=="y" (
    echo Cancelled.
    exit /b 0
)

if exist venv (
    echo Removing virtual environment...
    rmdir /s /q venv
    echo Virtual environment removed!
    echo Run scripts\setup.bat to recreate the environment.
) else (
    echo No virtual environment found.
)
exit /b 0