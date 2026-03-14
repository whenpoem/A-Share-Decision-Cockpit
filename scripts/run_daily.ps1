$python = "D:\miniconda\envs\ashare-agent\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}
& $python -c "import requests; print(requests.post('http://127.0.0.1:8000/api/sim/run-daily', timeout=120).json())"
