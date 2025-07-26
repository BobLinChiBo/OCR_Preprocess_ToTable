# OCR Table Extraction Pipeline - PowerShell Pipeline Management
# Advanced pipeline control with monitoring and reporting

param(
    [Parameter(Position=0)]
    [ValidateSet("help", "stage1", "stage2", "pipeline", "status", "monitor", "report", "config", "validate")]
    [string]$Command = "help",
    
    [string]$ConfigFile = "",
    [string]$InputDir = "input\raw_images",
    [string]$OutputDir = "output",
    [switch]$Watch,
    [switch]$Verbose,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# Configuration
$DefaultConfig = @{
    InputDirectory = "input\raw_images"
    Stage1Output = "output\stage1_initial_processing"
    Stage2Output = "output\stage2_advanced_processing"
    DebugDirectory = "debug"
    LogDirectory = "logs"
    ConfigFile = "src\ocr_pipeline\config\default_stage1.yaml"
}

function Write-PipelineHeader {
    param([string]$Title)
    Write-Host ""
    Write-Host "🔧 " -NoNewline -ForegroundColor Cyan
    Write-Host $Title -ForegroundColor Cyan
    Write-Host ("=" * ($Title.Length + 3)) -ForegroundColor Cyan
    Write-Host ""
}

function Write-Status {
    param([string]$Message, [string]$Status = "INFO")
    $color = switch ($Status) {
        "SUCCESS" { "Green" }
        "WARNING" { "Yellow" }
        "ERROR" { "Red" }
        "INFO" { "Blue" }
        default { "White" }
    }
    
    $icon = switch ($Status) {
        "SUCCESS" { "✓" }
        "WARNING" { "⚠" }
        "ERROR" { "✗" }
        "INFO" { "ℹ" }
        default { "•" }
    }
    
    Write-Host "$icon $Message" -ForegroundColor $color
}

function Test-PipelinePrerequisites {
    $issues = @()
    
    # Check Python
    try {
        $null = & python --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            $issues += "Python not found in PATH"
        }
    }
    catch {
        $issues += "Python not available"
    }
    
    # Check virtual environment
    if (-not $env:VIRTUAL_ENV) {
        $issues += "Virtual environment not activated"
    }
    
    # Check package installation
    try {
        & python -c "import ocr_pipeline" 2>$null
        if ($LASTEXITCODE -ne 0) {
            $issues += "OCR pipeline package not installed"
        }
    }
    catch {
        $issues += "OCR pipeline package not importable"
    }
    
    return $issues
}

function Get-DirectoryStats {
    param([string]$Path)
    
    if (-not (Test-Path $Path)) {
        return @{
            Exists = $false
            FileCount = 0
            TotalSize = 0
            LastModified = $null
        }
    }
    
    $files = Get-ChildItem -Path $Path -Recurse -File
    $totalSize = ($files | Measure-Object -Property Length -Sum).Sum
    $lastModified = ($files | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime
    
    return @{
        Exists = $true
        FileCount = $files.Count
        TotalSize = $totalSize
        LastModified = $lastModified
    }
}

function Format-FileSize {
    param([long]$Bytes)
    
    $sizes = @("B", "KB", "MB", "GB", "TB")
    $index = 0
    $size = [double]$Bytes
    
    while ($size -ge 1024 -and $index -lt $sizes.Length - 1) {
        $size /= 1024
        $index++
    }
    
    return "{0:N2} {1}" -f $size, $sizes[$index]
}

function Invoke-Help {
    Write-PipelineHeader "OCR Pipeline Management Commands"
    
    Write-Host "Pipeline Execution:" -ForegroundColor Yellow
    Write-Host "  stage1        Execute Stage 1 processing (raw images → cropped tables)"
    Write-Host "  stage2        Execute Stage 2 processing (cropped tables → structured data)"
    Write-Host "  pipeline      Execute complete pipeline (stage1 + stage2)"
    Write-Host ""
    
    Write-Host "Monitoring & Analysis:" -ForegroundColor Yellow
    Write-Host "  status        Show comprehensive pipeline status"
    Write-Host "  monitor       Monitor pipeline execution with real-time updates"
    Write-Host "  report        Generate detailed processing report"
    Write-Host "  validate      Validate input data and configuration"
    Write-Host ""
    
    Write-Host "Configuration:" -ForegroundColor Yellow
    Write-Host "  config        Show current configuration settings"
    Write-Host ""
    
    Write-Host "Parameters:" -ForegroundColor Yellow
    Write-Host "  -ConfigFile   Specify custom configuration file"
    Write-Host "  -InputDir     Override input directory"
    Write-Host "  -OutputDir    Override output directory"
    Write-Host "  -Watch        Monitor for file changes"
    Write-Host "  -Verbose      Enable verbose output"
    Write-Host "  -Force        Skip confirmation prompts"
    Write-Host ""
    
    Write-Host "Examples:" -ForegroundColor Cyan
    Write-Host "  .\scripts\manage-pipeline.ps1 status"
    Write-Host "  .\scripts\manage-pipeline.ps1 stage1 -Verbose"
    Write-Host "  .\scripts\manage-pipeline.ps1 pipeline -Watch"
    Write-Host "  .\scripts\manage-pipeline.ps1 validate -InputDir 'custom\input'"
}

function Invoke-Status {
    Write-PipelineHeader "Pipeline Status Report"
    
    # Check prerequisites
    $issues = Test-PipelinePrerequisites
    if ($issues.Count -gt 0) {
        Write-Host "⚠ Prerequisites Issues:" -ForegroundColor Yellow
        foreach ($issue in $issues) {
            Write-Host "  • $issue" -ForegroundColor Yellow
        }
        Write-Host ""
    }
    else {
        Write-Status "All prerequisites satisfied" "SUCCESS"
    }
    
    # Environment information
    Write-Host "Environment Information:" -ForegroundColor Blue
    Write-Host "  Python Version: $(try { & python --version } catch { 'Not Available' })"
    Write-Host "  Virtual Environment: $(if ($env:VIRTUAL_ENV) { $env:VIRTUAL_ENV } else { 'Not Active' })"
    Write-Host "  Working Directory: $(Get-Location)"
    Write-Host ""
    
    # Directory analysis
    Write-Host "Directory Analysis:" -ForegroundColor Blue
    
    $directories = @{
        "Input" = $DefaultConfig.InputDirectory
        "Stage 1 Output" = $DefaultConfig.Stage1Output
        "Stage 2 Output" = $DefaultConfig.Stage2Output
        "Debug Output" = $DefaultConfig.DebugDirectory
    }
    
    foreach ($dir in $directories.GetEnumerator()) {
        $stats = Get-DirectoryStats $dir.Value
        if ($stats.Exists) {
            $sizeStr = Format-FileSize $stats.TotalSize
            $modifiedStr = if ($stats.LastModified) { $stats.LastModified.ToString("yyyy-MM-dd HH:mm") } else { "N/A" }
            Write-Host "  $($dir.Key): $($stats.FileCount) files, $sizeStr (Last: $modifiedStr)" -ForegroundColor Green
        }
        else {
            Write-Host "  $($dir.Key): Directory not found" -ForegroundColor Yellow
        }
    }
    
    # Configuration status
    Write-Host ""
    Write-Host "Configuration:" -ForegroundColor Blue
    if (Test-Path $DefaultConfig.ConfigFile) {
        Write-Host "  Config File: $($DefaultConfig.ConfigFile) ✓" -ForegroundColor Green
    }
    else {
        Write-Host "  Config File: $($DefaultConfig.ConfigFile) ✗" -ForegroundColor Red
    }
}

function Invoke-Validate {
    Write-PipelineHeader "Pipeline Validation"
    
    $validationErrors = @()
    
    # Check input directory
    if (-not (Test-Path $InputDir)) {
        $validationErrors += "Input directory not found: $InputDir"
    }
    else {
        $inputFiles = Get-ChildItem -Path $InputDir -File
        if ($inputFiles.Count -eq 0) {
            $validationErrors += "Input directory is empty: $InputDir"
        }
        else {
            Write-Status "Found $($inputFiles.Count) input files" "SUCCESS"
            
            # Check file types
            $supportedExtensions = @(".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp")
            $unsupportedFiles = $inputFiles | Where-Object { $_.Extension.ToLower() -notin $supportedExtensions }
            
            if ($unsupportedFiles.Count -gt 0) {
                Write-Status "Warning: $($unsupportedFiles.Count) files may not be supported image formats" "WARNING"
                if ($Verbose) {
                    foreach ($file in $unsupportedFiles) {
                        Write-Host "    • $($file.Name)" -ForegroundColor Yellow
                    }
                }
            }
        }
    }
    
    # Check configuration file
    $configPath = if ($ConfigFile) { $ConfigFile } else { $DefaultConfig.ConfigFile }
    if (-not (Test-Path $configPath)) {
        $validationErrors += "Configuration file not found: $configPath"
    }
    else {
        Write-Status "Configuration file found: $configPath" "SUCCESS"
    }
    
    # Check output directories
    $outputDirs = @($DefaultConfig.Stage1Output, $DefaultConfig.Stage2Output, $DefaultConfig.DebugDirectory)
    foreach ($dir in $outputDirs) {
        if (-not (Test-Path $dir)) {
            Write-Status "Creating output directory: $dir" "INFO"
            try {
                New-Item -ItemType Directory -Path $dir -Force | Out-Null
            }
            catch {
                $validationErrors += "Failed to create output directory: $dir"
            }
        }
    }
    
    # Report validation results
    Write-Host ""
    if ($validationErrors.Count -eq 0) {
        Write-Status "Validation completed successfully - ready to run pipeline" "SUCCESS"
        return $true
    }
    else {
        Write-Status "Validation failed with $($validationErrors.Count) error(s):" "ERROR"
        foreach ($error in $validationErrors) {
            Write-Host "  • $error" -ForegroundColor Red
        }
        return $false
    }
}

function Invoke-Stage1 {
    Write-PipelineHeader "Executing Stage 1 Processing"
    
    if (-not (Invoke-Validate)) {
        Write-Status "Validation failed - aborting stage 1 execution" "ERROR"
        return $false
    }
    
    $packageManager = if (Get-Command poetry -ErrorAction SilentlyContinue) { "poetry" } else { "pip" }
    
    try {
        $startTime = Get-Date
        Write-Status "Starting Stage 1 processing..." "INFO"
        Write-Host "  Input: $InputDir"
        Write-Host "  Output: $($DefaultConfig.Stage1Output)"
        Write-Host ""
        
        if ($packageManager -eq "poetry") {
            & poetry run ocr-stage1
        }
        else {
            & ocr-stage1
        }
        
        if ($LASTEXITCODE -ne 0) {
            throw "Stage 1 processing failed with exit code $LASTEXITCODE"
        }
        
        $duration = (Get-Date) - $startTime
        Write-Status "Stage 1 completed successfully in $($duration.ToString('mm\:ss'))" "SUCCESS"
        
        # Show output statistics
        $outputStats = Get-DirectoryStats $DefaultConfig.Stage1Output
        Write-Host "  Generated $($outputStats.FileCount) output files ($(Format-FileSize $outputStats.TotalSize))"
        
        return $true
    }
    catch {
        Write-Status "Stage 1 processing failed: $_" "ERROR"
        return $false
    }
}

function Invoke-Stage2 {
    Write-PipelineHeader "Executing Stage 2 Processing"
    
    # Check if Stage 1 output exists
    if (-not (Test-Path $DefaultConfig.Stage1Output)) {
        Write-Status "Stage 1 output not found - please run Stage 1 first" "ERROR"
        return $false
    }
    
    $packageManager = if (Get-Command poetry -ErrorAction SilentlyContinue) { "poetry" } else { "pip" }
    
    try {
        $startTime = Get-Date
        Write-Status "Starting Stage 2 processing..." "INFO"
        Write-Host "  Input: $($DefaultConfig.Stage1Output)"
        Write-Host "  Output: $($DefaultConfig.Stage2Output)"
        Write-Host ""
        
        if ($packageManager -eq "poetry") {
            & poetry run ocr-stage2
        }
        else {
            & ocr-stage2
        }
        
        if ($LASTEXITCODE -ne 0) {
            throw "Stage 2 processing failed with exit code $LASTEXITCODE"
        }
        
        $duration = (Get-Date) - $startTime
        Write-Status "Stage 2 completed successfully in $($duration.ToString('mm\:ss'))" "SUCCESS"
        
        # Show output statistics
        $outputStats = Get-DirectoryStats $DefaultConfig.Stage2Output
        Write-Host "  Generated $($outputStats.FileCount) output files ($(Format-FileSize $outputStats.TotalSize))"
        
        return $true
    }
    catch {
        Write-Status "Stage 2 processing failed: $_" "ERROR"
        return $false
    }
}

function Invoke-Pipeline {
    Write-PipelineHeader "Executing Complete Pipeline"
    
    $pipelineStartTime = Get-Date
    
    # Execute Stage 1
    if (-not (Invoke-Stage1)) {
        Write-Status "Pipeline failed at Stage 1" "ERROR"
        return $false
    }
    
    Write-Host ""
    
    # Execute Stage 2
    if (-not (Invoke-Stage2)) {
        Write-Status "Pipeline failed at Stage 2" "ERROR"
        return $false
    }
    
    $totalDuration = (Get-Date) - $pipelineStartTime
    Write-Host ""
    Write-Status "Complete pipeline executed successfully in $($totalDuration.ToString('mm\:ss'))" "SUCCESS"
    
    # Final summary
    Write-Host ""
    Write-Host "Pipeline Summary:" -ForegroundColor Blue
    $stage1Stats = Get-DirectoryStats $DefaultConfig.Stage1Output
    $stage2Stats = Get-DirectoryStats $DefaultConfig.Stage2Output
    Write-Host "  Stage 1 Output: $($stage1Stats.FileCount) files ($(Format-FileSize $stage1Stats.TotalSize))"
    Write-Host "  Stage 2 Output: $($stage2Stats.FileCount) files ($(Format-FileSize $stage2Stats.TotalSize))"
    
    return $true
}

# Main execution
try {
    switch ($Command.ToLower()) {
        "help" { Invoke-Help }
        "status" { Invoke-Status }
        "validate" { Invoke-Validate }
        "stage1" { Invoke-Stage1 }
        "stage2" { Invoke-Stage2 }
        "pipeline" { Invoke-Pipeline }
        default {
            Write-Status "Unknown command: $Command" "ERROR"
            Write-Host "Run '.\scripts\manage-pipeline.ps1 help' for available commands" -ForegroundColor Yellow
            exit 1
        }
    }
}
catch {
    Write-Status "Command execution failed: $_" "ERROR"
    exit 1
}