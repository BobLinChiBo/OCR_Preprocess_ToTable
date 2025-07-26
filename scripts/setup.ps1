# OCR Table Extraction Pipeline - PowerShell Setup Script
# Advanced Windows setup with error handling and validation

param(
    [switch]$Force,
    [switch]$SkipVenv,
    [switch]$DevOnly,
    [string]$PythonVersion = "3.8"
)

$ErrorActionPreference = "Stop"

function Write-Header {
    param([string]$Title)
    Write-Host ""
    Write-Host "=" * 50 -ForegroundColor Cyan
    Write-Host $Title -ForegroundColor Cyan
    Write-Host "=" * 50 -ForegroundColor Cyan
    Write-Host ""
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

function Test-PythonVersion {
    param([string]$RequiredVersion)
    
    try {
        $pythonOutput = & python --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            return $false
        }
        
        $versionMatch = $pythonOutput -match "Python (\d+\.\d+)"
        if (-not $versionMatch) {
            return $false
        }
        
        $installedVersion = [version]$matches[1]
        $required = [version]$RequiredVersion
        
        return $installedVersion -ge $required
    }
    catch {
        return $false
    }
}

function Test-Command {
    param([string]$Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function New-DirectoryIfNotExists {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
        Write-Success "Created directory: $Path"
    }
}

# Main setup process
try {
    Write-Header "OCR Pipeline - PowerShell Environment Setup"
    
    # Check Python installation
    Write-Host "Checking Python installation..." -ForegroundColor Blue
    
    if (-not (Test-Command "python")) {
        Write-Error "Python is not installed or not in PATH"
        Write-Host "Please install Python $PythonVersion+ from https://www.python.org/downloads/" -ForegroundColor Yellow
        Write-Host "Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Yellow
        exit 1
    }
    
    if (-not (Test-PythonVersion $PythonVersion)) {
        Write-Error "Python version $PythonVersion or higher is required"
        $currentVersion = & python --version
        Write-Host "Current version: $currentVersion" -ForegroundColor Yellow
        exit 1
    }
    
    $pythonVersion = & python --version
    Write-Success "Python installation verified: $pythonVersion"
    
    # Check for virtual environment
    if (-not $SkipVenv) {
        Write-Host "Setting up virtual environment..." -ForegroundColor Blue
        
        if ((Test-Path "venv") -and -not $Force) {
            Write-Warning "Virtual environment already exists. Use -Force to recreate."
        }
        else {
            if (Test-Path "venv") {
                Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
                Remove-Item -Recurse -Force "venv"
            }
            
            Write-Host "Creating virtual environment..."
            & python -m venv venv
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Failed to create virtual environment"
                exit 1
            }
            Write-Success "Virtual environment created"
        }
        
        # Activate virtual environment
        Write-Host "Activating virtual environment..."
        & "venv\Scripts\Activate.ps1"
        Write-Success "Virtual environment activated"
    }
    
    # Upgrade pip
    Write-Host "Upgrading pip..." -ForegroundColor Blue
    & python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to upgrade pip"
        exit 1
    }
    Write-Success "Pip upgraded successfully"
    
    # Detect package manager
    $usePoetry = Test-Command "poetry"
    
    if ($usePoetry) {
        Write-Host "Using Poetry for dependency management..." -ForegroundColor Blue
        & poetry install
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Poetry installation failed"
            exit 1
        }
        Write-Success "Dependencies installed with Poetry"
    }
    else {
        Write-Host "Using pip for dependency management..." -ForegroundColor Blue
        
        # Install core dependencies
        Write-Host "Installing core dependencies..."
        & pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to install core dependencies"
            exit 1
        }
        Write-Success "Core dependencies installed"
        
        # Install development dependencies
        if (-not $DevOnly) {
            Write-Host "Installing development dependencies..."
            & pip install -r requirements-dev.txt
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Failed to install development dependencies"
                exit 1
            }
            Write-Success "Development dependencies installed"
        }
        
        # Install package in development mode
        Write-Host "Installing package in development mode..."
        & pip install -e .
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to install package in development mode"
            exit 1
        }
        Write-Success "Package installed in development mode"
    }
    
    # Setup pre-commit hooks
    if (-not $DevOnly -and (Test-Command "pre-commit")) {
        Write-Host "Setting up pre-commit hooks..." -ForegroundColor Blue
        & pre-commit install
        & pre-commit install --hook-type commit-msg
        Write-Success "Pre-commit hooks installed"
    }
    
    # Create necessary directories
    Write-Host "Creating necessary directories..." -ForegroundColor Blue
    $directories = @(
        "input\raw_images",
        "output\stage1_initial_processing",
        "output\stage2_advanced_processing",
        "debug\stage1_debug",
        "debug\stage2_debug"
    )
    
    foreach ($dir in $directories) {
        New-DirectoryIfNotExists $dir
    }
    
    # Validate installation
    Write-Host "Validating installation..." -ForegroundColor Blue
    
    try {
        & python -c "import ocr_pipeline; print('Package import successful')"
        Write-Success "Package validation successful"
    }
    catch {
        Write-Warning "Package validation failed - this may be normal if CLI is not yet implemented"
    }
    
    Write-Header "Setup Completed Successfully!"
    
    Write-Host "Environment Information:" -ForegroundColor Blue
    Write-Host "  Python: $(& python --version)"
    Write-Host "  Package Manager: $(if ($usePoetry) { 'Poetry' } else { 'pip' })"
    Write-Host "  Virtual Environment: $(if (-not $SkipVenv) { 'venv\Scripts\activate.bat' } else { 'None' })"
    
    Write-Host ""
    Write-Host "Available Commands:" -ForegroundColor Blue
    Write-Host "  .\scripts\dev.bat help        - Development commands"
    Write-Host "  .\scripts\test.bat help       - Testing commands"
    Write-Host "  .\scripts\run-pipeline.bat help - Pipeline execution"
    
    Write-Host ""
    Write-Host "To activate the environment in future sessions:" -ForegroundColor Yellow
    Write-Host "  venv\Scripts\activate.bat"
    
}
catch {
    Write-Error "Setup failed: $_"
    Write-Host ""
    Write-Host "Common solutions:" -ForegroundColor Yellow
    Write-Host "  1. Ensure Python $PythonVersion+ is installed and in PATH"
    Write-Host "  2. Run PowerShell as Administrator if permission errors occur"
    Write-Host "  3. Check internet connection for package downloads"
    Write-Host "  4. Use -Force flag to recreate virtual environment"
    exit 1
}