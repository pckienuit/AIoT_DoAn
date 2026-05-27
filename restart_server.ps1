param(
    [string]$HostAddress = "0.0.0.0",
    [int]$Port = 8000,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "Restarting AIoT server on port $Port..."

& (Join-Path $ProjectRoot "stop_server.ps1") -Port $Port

$StartArgs = @(
    "-HostAddress", $HostAddress,
    "-Port", $Port
)

if ($Reload) {
    $StartArgs += "-Reload"
}

& (Join-Path $ProjectRoot "start_server.ps1") @StartArgs
