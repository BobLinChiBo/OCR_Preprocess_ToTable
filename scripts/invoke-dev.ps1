# OCR Table Extraction Pipeline - PowerShell Development Module
# Advanced development workflow management with enhanced features

param(
    [Parameter(Position=0)]
    [ValidateSet("help", "format", "lint", "type-check", "pre-commit", "install", "install-dev", 
                 "clean", "build", "quick", "dev-check", "status", "profile", "security")]
    [string]$Command = "help",
    
    [switch]$Verbose,
    [switch]$Force,
    [string]$TestPattern = "",
    [string]$Profile = "default"
)

$ErrorActionPreference = "Stop"

# Color scheme
$Colors = @{
    Header = "Cyan"
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Info = "Blue"
    Emphasis = "Magenta"
}

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White",
        [switch]$NoNewline
    )
    $params = @{ Object = $Message; ForegroundColor = $Color }
    if ($NoNewline) { $params.NoNewline = $true }
    Write-Host @params
}

function Write-Header {
    param([string]$Title)
    Write-Host ""
    Write-ColorOutput ("=" * 60) $Colors.Header
    Write-ColorOutput $Title $Colors.Header
    Write-ColorOutput ("=" * 60) $Colors.Header
    Write-Host ""
}

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-ColorOutput ("─" * 40) $Colors.Info
    Write-ColorOutput $Title $Colors.Info
    Write-ColorOutput ("─" * 40) $Colors.Info
    Write-Host ""
}

function Test-VirtualEnvironment {
    if (-not $env:VIRTUAL_ENV) {
        Write-ColorOutput "⚠ Warning: Virtual environment not activated" $Colors.Warning
        Write-ColorOutput "  Run 'venv\Scripts\activate.bat' first" $Colors.Warning
        Write-Host ""
    }
    else {
        Write-ColorOutput "✓ Virtual environment active: $env:VIRTUAL_ENV" $Colors.Success
    }
}

function Test-PackageManager {
    try {
        $null = Get-Command poetry -ErrorAction Stop
        return "poetry"
    }
    catch {
        return "pip"
    }
}

function Invoke-WithPackageManager {
    param(
        [string]$PoetryCommand,
        [string]$PipCommand
    )
    
    $packageManager = Test-PackageManager
    
    if ($packageManager -eq "poetry") {
        Write-ColorOutput "Using Poetry..." $Colors.Info
        Invoke-Expression "poetry run $PoetryCommand"
    }
    else {
        Write-ColorOutput "Using pip..." $Colors.Info
        Invoke-Expression $PipCommand
    }
    
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

function Get-ProjectStatus {
    $status = @{
        VirtualEnv = if ($env:VIRTUAL_ENV) { "Active: $env:VIRTUAL_ENV" } else { "Not Active" }
        PackageManager = Test-PackageManager
        PythonVersion = try { & python --version } catch { "Not Found" }
        GitStatus = try { & git status --porcelain | Measure-Object -Line | Select-Object -ExpandProperty Lines } catch { "Not a git repo" }
        TestCoverage = if (Test-Path "htmlcov\index.html") { "Available" } else { "Not Generated" }
        PreCommit = if (Test-Path ".git\hooks\pre-commit") { "Installed" } else { "Not Installed" }
    }
    return $status
}

# Command implementations
function Invoke-Help {
    Write-Header "OCR Pipeline - PowerShell Development Commands"
    
    Write-Section "Code Quality Commands"
    Write-Host "  format        " -NoNewline; Write-ColorOutput "Format code with black and isort" $Colors.Info
    Write-Host "  lint          " -NoNewline; Write-ColorOutput "Run all linters (ruff, flake8, bandit)" $Colors.Info
    Write-Host "  type-check    " -NoNewline; Write-ColorOutput "Run mypy type checking" $Colors.Info
    Write-Host "  pre-commit    " -NoNewline; Write-ColorOutput "Run all pre-commit hooks" $Colors.Info
    Write-Host "  security      " -NoNewline; Write-ColorOutput "Run security audit with bandit" $Colors.Info
    
    Write-Section "Installation Commands"
    Write-Host "  install       " -NoNewline; Write-ColorOutput "Install package in development mode" $Colors.Info
    Write-Host "  install-dev   " -NoNewline; Write-ColorOutput "Install with development dependencies" $Colors.Info
    
    Write-Section "Development Workflows"
    Write-Host "  quick         " -NoNewline; Write-ColorOutput "Run format + lint + test-fast" $Colors.Info
    Write-Host "  dev-check     " -NoNewline; Write-ColorOutput "Run format + lint + type-check + test-fast" $Colors.Info
    Write-Host "  profile       " -NoNewline; Write-ColorOutput "Run performance profiling" $Colors.Info
    
    Write-Section "Maintenance Commands"
    Write-Host "  clean         " -NoNewline; Write-ColorOutput "Clean build artifacts and cache" $Colors.Info
    Write-Host "  build         " -NoNewline; Write-ColorOutput "Build distribution packages" $Colors.Info
    Write-Host "  status        " -NoNewline; Write-ColorOutput "Show project status" $Colors.Info
    
    Write-Section "Examples"
    Write-ColorOutput "  .\scripts\invoke-dev.ps1 format" $Colors.Emphasis
    Write-ColorOutput "  .\scripts\invoke-dev.ps1 dev-check -Verbose" $Colors.Emphasis
    Write-ColorOutput "  .\scripts\invoke-dev.ps1 lint -Force" $Colors.Emphasis
}

function Invoke-Format {
    Write-Header "Code Formatting"
    Test-VirtualEnvironment
    
    try {
        Invoke-WithPackageManager "black src\ tests\" "black src\ tests\"
        Write-ColorOutput "✓ Black formatting completed" $Colors.Success
        
        Invoke-WithPackageManager "isort src\ tests\" "isort src\ tests\"
        Write-ColorOutput "✓ Import sorting completed" $Colors.Success
        
        Write-ColorOutput "✓ Code formatting complete!" $Colors.Success
    }
    catch {
        Write-ColorOutput "✗ Formatting failed: $_" $Colors.Error
        exit 1
    }
}

function Invoke-Lint {
    Write-Header "Code Linting"
    Test-VirtualEnvironment
    
    $lintErrors = 0
    
    try {
        Write-Section "Running ruff..."
        Invoke-WithPackageManager "ruff check src\ tests\" "ruff check src\ tests\"
        Write-ColorOutput "✓ Ruff linting passed" $Colors.Success
    }
    catch {
        Write-ColorOutput "✗ Ruff linting failed" $Colors.Error
        $lintErrors++
    }
    
    try {
        Write-Section "Running flake8..."
        Invoke-WithPackageManager "flake8 src\ tests\" "flake8 src\ tests\"
        Write-ColorOutput "✓ Flake8 linting passed" $Colors.Success
    }
    catch {
        Write-ColorOutput "✗ Flake8 linting failed" $Colors.Error
        $lintErrors++
    }
    
    if ($lintErrors -eq 0) {
        Write-ColorOutput "✓ All linting checks passed!" $Colors.Success
    }
    else {
        Write-ColorOutput "✗ $lintErrors linting check(s) failed" $Colors.Error
        if (-not $Force) {
            exit 1
        }
    }
}

function Invoke-TypeCheck {
    Write-Header "Type Checking"
    Test-VirtualEnvironment
    
    try {
        Invoke-WithPackageManager "mypy src\" "mypy src\"
        Write-ColorOutput "✓ Type checking passed!" $Colors.Success
    }
    catch {
        Write-ColorOutput "✗ Type checking failed: $_" $Colors.Error
        exit 1
    }
}

function Invoke-Security {
    Write-Header "Security Audit"
    Test-VirtualEnvironment
    
    try {
        Invoke-WithPackageManager "bandit -r src\ -c pyproject.toml" "bandit -r src\ -c pyproject.toml"
        Write-ColorOutput "✓ Security audit passed!" $Colors.Success
    }
    catch {
        Write-ColorOutput "✗ Security audit failed: $_" $Colors.Error
        exit 1
    }
}

function Invoke-Status {
    Write-Header "Project Status"
    
    $status = Get-ProjectStatus
    
    Write-Section "Environment"
    Write-Host "  Python Version:   " -NoNewline; Write-ColorOutput $status.PythonVersion $Colors.Info
    Write-Host "  Virtual Env:      " -NoNewline; Write-ColorOutput $status.VirtualEnv $Colors.Info
    Write-Host "  Package Manager:  " -NoNewline; Write-ColorOutput $status.PackageManager $Colors.Info
    Write-Host "  Pre-commit:       " -NoNewline; Write-ColorOutput $status.PreCommit $Colors.Info
    
    Write-Section "Project State"
    Write-Host "  Uncommitted files:" -NoNewline; Write-ColorOutput " $($status.GitStatus)" $Colors.Info
    Write-Host "  Test Coverage:    " -NoNewline; Write-ColorOutput $status.TestCoverage $Colors.Info
    
    Write-Section "Directories"
    $directories = @(
        @{ Path = "input\raw_images"; Purpose = "Input files" },
        @{ Path = "output\stage1_initial_processing"; Purpose = "Stage 1 output" },
        @{ Path = "output\stage2_advanced_processing"; Purpose = "Stage 2 output" },
        @{ Path = "debug"; Purpose = "Debug output" }
    )
    
    foreach ($dir in $directories) {
        $exists = Test-Path $dir.Path
        $status = if ($exists) { "EXISTS" } else { "MISSING" }
        $color = if ($exists) { $Colors.Success } else { $Colors.Warning }
        Write-Host "  $($dir.Purpose): " -NoNewline
        Write-ColorOutput $status $color
    }
}

function Invoke-Quick {
    Write-Header "Quick Development Check"
    
    Invoke-Format
    Invoke-Lint
    
    Write-Section "Running fast tests..."
    & ".\scripts\test.bat" fast
    
    Write-ColorOutput "✓ Quick development check complete!" $Colors.Success
}

function Invoke-DevCheck {
    Write-Header "Comprehensive Development Check"
    
    Invoke-Format
    Invoke-Lint  
    Invoke-TypeCheck
    
    Write-Section "Running fast tests..."
    & ".\scripts\test.bat" fast
    
    Write-ColorOutput "✓ Development checks complete!" $Colors.Success
}

# Main execution
try {
    switch ($Command.ToLower()) {
        "help" { Invoke-Help }
        "format" { Invoke-Format }
        "lint" { Invoke-Lint }
        "type-check" { Invoke-TypeCheck }
        "security" { Invoke-Security }
        "status" { Invoke-Status }
        "quick" { Invoke-Quick }
        "dev-check" { Invoke-DevCheck }
        default {
            Write-ColorOutput "Unknown command: $Command" $Colors.Error
            Write-ColorOutput "Run '.\scripts\invoke-dev.ps1 help' for available commands" $Colors.Warning
            exit 1
        }
    }
}
catch {
    Write-ColorOutput "Command failed: $_" $Colors.Error
    exit 1
}