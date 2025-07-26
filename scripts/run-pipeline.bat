@echo off
REM OCR Table Extraction Pipeline - Pipeline Execution Script
REM Windows script for running the OCR processing stages

setlocal enabledelayedexpansion

if "%~1"=="" (
    set PIPELINE_MODE=help
) else (
    set PIPELINE_MODE=%~1
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
echo OCR Pipeline - Execution Control
echo =================================

if "%PIPELINE_MODE%"=="help" goto :help
if "%PIPELINE_MODE%"=="stage1" goto :stage1
if "%PIPELINE_MODE%"=="stage2" goto :stage2
if "%PIPELINE_MODE%"=="pipeline" goto :pipeline
if "%PIPELINE_MODE%"=="full" goto :pipeline
if "%PIPELINE_MODE%"=="status" goto :status
if "%PIPELINE_MODE%"=="clean-output" goto :cleanoutput

echo Unknown pipeline command: %PIPELINE_MODE%
echo Run 'scripts\run-pipeline.bat help' for available commands
exit /b 1

:help
echo.
echo OCR Table Extraction Pipeline - Execution Commands
echo.
echo Usage: scripts\run-pipeline.bat [COMMAND]
echo.
echo Pipeline Commands:
echo   stage1        Run OCR Stage 1 processing (raw images to cropped tables)
echo   stage2        Run OCR Stage 2 processing (cropped tables to structured data)
echo   pipeline      Run complete pipeline (stage1 + stage2)
echo   full          Same as pipeline
echo.
echo Utility Commands:
echo   status        Show pipeline status and output directories
echo   clean-output  Clean all output directories
echo.
echo Configuration:
echo   Default config: src\ocr_pipeline\config\default_stage1.yaml
echo   Input directory: input\raw_images\
echo   Output directory: output\stage1_initial_processing\
echo.
echo Examples:
echo   scripts\run-pipeline.bat stage1
echo   scripts\run-pipeline.bat pipeline
echo   scripts\run-pipeline.bat status
echo.
exit /b 0

:stage1
echo Running OCR Stage 1 processing...
echo Input: input\raw_images\
echo Output: output\stage1_initial_processing\
echo.

REM Check if input directory exists and has files
if not exist "input\raw_images" (
    echo ERROR: Input directory 'input\raw_images' does not exist
    echo Please create the directory and add your scanned images
    exit /b 1
)

REM Count files in input directory
set file_count=0
for %%f in ("input\raw_images\*.*") do (
    set /a file_count+=1
)

if %file_count%==0 (
    echo ERROR: No files found in 'input\raw_images'
    echo Please add your scanned images to process
    exit /b 1
)

echo Found %file_count% files to process...
echo.

if %USE_POETRY%==1 (
    poetry run ocr-stage1
) else (
    ocr-stage1
)

if errorlevel 1 (
    echo ERROR: Stage 1 processing failed
    exit /b 1
)

echo.
echo Stage 1 processing completed successfully!
echo Check output in: output\stage1_initial_processing\
exit /b 0

:stage2
echo Running OCR Stage 2 processing...
echo Input: output\stage1_initial_processing\
echo Output: output\stage2_advanced_processing\
echo.

REM Check if stage1 output exists
if not exist "output\stage1_initial_processing" (
    echo ERROR: Stage 1 output not found
    echo Please run Stage 1 first: scripts\run-pipeline.bat stage1
    exit /b 1
)

if %USE_POETRY%==1 (
    poetry run ocr-stage2
) else (
    ocr-stage2
)

if errorlevel 1 (
    echo ERROR: Stage 2 processing failed
    exit /b 1
)

echo.
echo Stage 2 processing completed successfully!
echo Check output in: output\stage2_advanced_processing\
exit /b 0

:pipeline
echo Running complete OCR pipeline...
echo This will execute Stage 1 followed by Stage 2
echo.

call :stage1
if errorlevel 1 (
    echo Pipeline failed at Stage 1
    exit /b 1
)

echo.
echo =================================
echo Starting Stage 2...
echo =================================
echo.

call :stage2
if errorlevel 1 (
    echo Pipeline failed at Stage 2
    exit /b 1
)

echo.
echo =========================================
echo Complete pipeline execution finished!
echo =========================================
echo.
echo Final output available in:
echo   - output\stage1_initial_processing\
echo   - output\stage2_advanced_processing\
echo.
exit /b 0

:status
echo.
echo OCR Pipeline Status
echo ===================
echo.

REM Check input directory
if exist "input\raw_images" (
    set file_count=0
    for %%f in ("input\raw_images\*.*") do (
        set /a file_count+=1
    )
    echo Input Directory: input\raw_images\ (%file_count% files)
) else (
    echo Input Directory: input\raw_images\ (NOT FOUND)
)

REM Check stage1 output
if exist "output\stage1_initial_processing" (
    echo Stage 1 Output: output\stage1_initial_processing\ (EXISTS)
) else (
    echo Stage 1 Output: output\stage1_initial_processing\ (NOT FOUND)
)

REM Check stage2 output
if exist "output\stage2_advanced_processing" (
    echo Stage 2 Output: output\stage2_advanced_processing\ (EXISTS)
) else (
    echo Stage 2 Output: output\stage2_advanced_processing\ (NOT FOUND)
)

REM Check debug directories
if exist "debug" (
    echo Debug Directory: debug\ (EXISTS)
) else (
    echo Debug Directory: debug\ (NOT FOUND)
)

echo.
echo Virtual Environment: 
if defined VIRTUAL_ENV (
    echo   ACTIVATED (%VIRTUAL_ENV%)
) else (
    echo   NOT ACTIVATED
)

echo.
exit /b 0

:cleanoutput
echo Cleaning output directories...
echo.

set /p confirm="This will delete all output and debug data. Continue? (y/N): "
if /i not "%confirm%"=="y" (
    echo Cancelled.
    exit /b 0
)

if exist "output" (
    echo Removing output\...
    rmdir /s /q "output"
)

if exist "debug" (
    echo Removing debug\...
    rmdir /s /q "debug"
)

echo.
echo Output directories cleaned!
echo Run scripts\setup.bat to recreate necessary directories.
exit /b 0