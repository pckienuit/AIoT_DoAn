param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$TargetPids = New-Object System.Collections.Generic.HashSet[int]

try {
    $Connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($Connection in $Connections) {
        if ($Connection.OwningProcess) {
            [void]$TargetPids.Add([int]$Connection.OwningProcess)
        }
    }
} catch {
    Write-Warning "Could not inspect port $Port with Get-NetTCPConnection: $($_.Exception.Message)"
}

$UvicornProcesses = Get-CimInstance Win32_Process |
    Where-Object {
        $_.CommandLine -and
        $_.CommandLine -match "uvicorn" -and
        $_.CommandLine -match "server\.main:app"
    }

foreach ($ProcessInfo in $UvicornProcesses) {
    [void]$TargetPids.Add([int]$ProcessInfo.ProcessId)
}

if ($TargetPids.Count -eq 0) {
    Write-Host "No AIoT server process found on port $Port."
    exit 0
}

foreach ($ProcessId in $TargetPids) {
    try {
        $Process = Get-Process -Id $ProcessId -ErrorAction Stop
        Write-Host "Stopping AIoT server process $ProcessId ($($Process.ProcessName))..."
        Stop-Process -Id $ProcessId -Force -ErrorAction Stop
    } catch {
        Write-Warning "Could not stop process ${ProcessId}: $($_.Exception.Message)"
    }
}

Start-Sleep -Milliseconds 500
Write-Host "AIoT server stopped."
