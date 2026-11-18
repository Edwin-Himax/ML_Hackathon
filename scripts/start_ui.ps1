param([int]$Port = 8001)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { $pythonPath = 'python' }
Write-Host "Laptop 2 UI: http://localhost:$Port"
Write-Host 'Open API connection settings and enter Laptop 1''s LAN address on port 8000.'
& $pythonPath -m http.server $Port --bind 127.0.0.1 --directory (Join-Path $projectRoot 'ui')
