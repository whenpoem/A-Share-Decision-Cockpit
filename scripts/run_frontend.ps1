$repoRoot = Split-Path -Parent $PSScriptRoot
$frontendRoot = Join-Path $repoRoot "service\frontend"
$npm = "npm"

Push-Location $frontendRoot
try {
    & $npm install
    & $npm run dev
}
finally {
    Pop-Location
}
