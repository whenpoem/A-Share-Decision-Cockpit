$python = "D:\miniconda\envs\ashare-agent\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}
& $python -m service.server.main
