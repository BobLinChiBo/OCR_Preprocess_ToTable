@echo off
echo OCR Parameter Tuning - File Copy Helper
echo ========================================
echo.
echo This script helps copy files between tuning stages.
echo.
echo Usage:
echo   copy_best_results.bat [stage] [source_folder]
echo.
echo Examples:
echo   copy_best_results.bat split start0.4_end0.6_width50
echo   copy_best_results.bat deskew range10_step0.2_min0.2
echo   copy_best_results.bat roi k31_s4.0_l8.0_cs20.0_ct5.0
echo.
echo After running, verify the files were copied correctly.
echo.
pause
