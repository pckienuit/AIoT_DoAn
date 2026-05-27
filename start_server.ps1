param(
    [string]$HostAddress = "0.0.0.0",
    [int]$Port = 8000,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = "python"
}

$UvicornArgs = @(
    "-m", "uvicorn",
    "server.main:app",
    "--host", $HostAddress,
    "--port", $Port
)

if ($Reload) {
    $UvicornArgs += "--reload"
}

Write-Host "Starting AIoT server at http://localhost:$Port"
Write-Host "Using Python: $Python"
& $Python @UvicornArgs
