$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = "C:\Users\whenpoem\.conda\envs\Introduction_to_Python\python.exe"
$ConfigPath = Join-Path $RepoRoot "configs\live_run.json"
$LogPath = Join-Path $RepoRoot "artifacts\reports\run_live.log"

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
"[run_live] started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $LogPath -Encoding utf8 -Append

Push-Location $RepoRoot
try {
    $env:PYTHONUNBUFFERED = "1"
    & $PythonExe "-u" "run_cli.py" "run-live" "--config" $ConfigPath "--fetch-retries" "10" "--retry-wait" "60" 2>&1 |
        Tee-Object -FilePath $LogPath -Append
}
finally {
    Pop-Location
}
