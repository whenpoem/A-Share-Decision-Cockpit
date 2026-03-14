$repoRoot = Split-Path -Parent $PSScriptRoot
$backendScript = Join-Path $PSScriptRoot "run_backend.ps1"
$frontendScript = Join-Path $PSScriptRoot "run_frontend.ps1"

Start-Process powershell.exe -WorkingDirectory $repoRoot -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    $backendScript
)

Start-Sleep -Seconds 2

Start-Process powershell.exe -WorkingDirectory $repoRoot -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    $frontendScript
)

Write-Host "Backend and frontend launch commands have been started."
Write-Host "API: http://127.0.0.1:8000"
Write-Host "Frontend: http://127.0.0.1:5173"
