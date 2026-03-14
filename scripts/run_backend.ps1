$python = "D:\miniconda\envs\ashare-agent\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    & $python -m service.server.main
}
finally {
    Pop-Location
}
