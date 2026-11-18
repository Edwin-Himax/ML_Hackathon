param(
    [int]$Port = 8000,
    [switch]$Lan,
    [ValidateSet('train','student_test','evaluator_test')][string]$Partition = 'student_test',
    [string]$DataRoot = '',
    [string]$ReferenceRoot = '',
    [string]$PointwiseRoot = '',
    [string]$AllowedOrigins = 'http://localhost:8001,http://127.0.0.1:8001'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { throw 'Run .\scripts\setup.ps1 first.' }
    $env:PATHSNAP_PARTITION = $Partition
    $env:PATHSNAP_ALLOWED_ORIGINS = $AllowedOrigins
    if ($DataRoot) { $env:PATHSNAP_DATA_ROOT = $DataRoot }
    if ($ReferenceRoot) { $env:PATHSNAP_REFERENCE_ROOT = $ReferenceRoot }
    if ($PointwiseRoot) { $env:PATHSNAP_POINTWISE_ROOT = $PointwiseRoot }
    $bindAddress = if ($Lan) { '0.0.0.0' } else { '127.0.0.1' }
    Write-Host "PathSnap API | partition=$Partition | bind=$bindAddress | port=$Port"
    Write-Host "Local UI: http://127.0.0.1:$Port/ui/"
    $defaultLog = if ($Partition -eq 'evaluator_test') { 'runtime/evaluator_requests.jsonl' } else { 'runtime/api_requests.jsonl' }
    $logPath = if ($env:PATHSNAP_LOG) { $env:PATHSNAP_LOG } else { $defaultLog }
    Write-Host "Prediction request logs appear below and in $logPath."
    & '.\.venv\Scripts\python.exe' -m uvicorn api.app:app --host $bindAddress --port $Port
    if ($LASTEXITCODE -ne 0) { throw 'The API did not exit successfully.' }
} finally { Pop-Location }
