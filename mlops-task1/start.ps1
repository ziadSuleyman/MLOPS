# Brings the whole Task 1 environment up: Docker -> Postgres -> data check -> JupyterLab.
#
# Usage:  .\start.ps1              database + Jupyter
#         .\start.ps1 -NoJupyter   database only
#
# ASCII only, on purpose: Windows PowerShell 5.1 reads a .ps1 with no BOM as
# ANSI, and a stray non-ASCII byte can end a string early and break parsing.

param([switch]$NoJupyter)

$ErrorActionPreference = "Stop"
$project = $PSScriptRoot
Set-Location $project

function Step($n, $msg) { Write-Host "`n[$n] $msg" -ForegroundColor Cyan }

# --- 1. Docker engine -------------------------------------------------------
Step 1 "Checking Docker..."
docker info *> $null
if (-not $?) {
    Write-Host "    Docker is not running. Starting Docker Desktop..." -ForegroundColor Yellow
    $exe = "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $exe) { Start-Process $exe } else { throw "Docker Desktop not found at $exe" }

    $waited = 0
    do {
        Start-Sleep -Seconds 3
        $waited += 3
        docker info *> $null
        if ($waited -ge 180) { throw "Docker did not start within 3 minutes." }
    } until ($?)
}
Write-Host "    Docker is running." -ForegroundColor Green

# --- 2. Database container --------------------------------------------------
Step 2 "Starting PostgreSQL..."
docker compose up -d
if (-not $?) { throw "docker compose up failed." }

$waited = 0
do {
    Start-Sleep -Seconds 2
    $waited += 2
    $health = docker inspect olist-postgres --format "{{.State.Health.Status}}"
    if ($waited -ge 120) { throw "Database did not become healthy within 2 minutes." }
} until ($health -eq "healthy")
Write-Host "    Database healthy on localhost:5433 after $waited seconds." -ForegroundColor Green

# --- 3. Data check ----------------------------------------------------------
Step 3 "Checking the data is still there..."
$rows = docker exec olist-postgres psql -U olist -d olist -tAc "SELECT count(*) FROM orders"
$rows = "$rows".Trim()

if ($rows -match '^\d+$' -and [int]$rows -gt 0) {
    Write-Host "    $rows orders present, no reload needed." -ForegroundColor Green
} else {
    Write-Host "    Tables are empty. Loading the CSVs, about one minute..." -ForegroundColor Yellow
    python "$project\ingest.py"
    if (-not $?) { throw "ingest.py failed." }
}

# --- 4. JupyterLab ----------------------------------------------------------
if ($NoJupyter) {
    Step 4 "Skipping Jupyter."
    Write-Host "`nReady. SQL shell:" -ForegroundColor Green
    Write-Host "  docker exec -it olist-postgres psql -U olist -d olist" -ForegroundColor Gray
} else {
    Step 4 "Starting JupyterLab. Your browser will open automatically."
    Write-Host "    Keep this window open. Closing it stops Jupyter; the database keeps running." -ForegroundColor DarkGray
    python -m jupyterlab --notebook-dir "$project"
}
