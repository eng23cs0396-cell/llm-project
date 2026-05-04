param(
    [Parameter(Mandatory=$false)]
    [string]$PythonVersion = "3.10",

    [Parameter(Mandatory=$false)]
    [string]$VenvPath = ".venv",

    [Parameter(Mandatory=$false)]
    [int]$Port = 8000,

    [Parameter(Mandatory=$false)]
    [switch]$NoReload,

    [Parameter(Mandatory=$false)]
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$rootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $rootDir

$venvPython = Join-Path $VenvPath "Scripts\\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[1/4] Creating virtual environment at $VenvPath..."
    py -$PythonVersion -m venv $VenvPath
}

if (-not (Test-Path $venvPython)) {
    throw "Could not find virtual environment python at $venvPython"
}

Write-Host "[2/4] Upgrading pip..."
& $venvPython -m pip install --upgrade pip

if (-not $SkipInstall) {
    Write-Host "[3/4] Installing dependencies from requirements.txt..."
    & $venvPython -m pip install -r requirements.txt
} else {
    Write-Host "[3/4] Skipping dependency installation (--SkipInstall)."
}

Write-Host "[4/4] Starting FastAPI app on port $Port..."

$uvicornArgs = @("-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "$Port")
if (-not $NoReload) {
    $uvicornArgs += "--reload"
}

& $venvPython @uvicornArgs
