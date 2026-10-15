#!/usr/bin/env pwsh
# docker_build_and_test.ps1
# Run this from aml-priority/ once Docker Desktop is fully started.
# Usage:  .\docker_build_and_test.ps1

param(
    [string]$Tag = "latest"
)

Set-Location $PSScriptRoot
$ErrorActionPreference = "Stop"

Write-Host "=== AML Priority — Docker Build & Test ===" -ForegroundColor Cyan
Write-Host "Working directory: $(Get-Location)"

# ── 1. Verify Docker is running ────────────────────────────────────────────────
Write-Host "`n[1/6] Checking Docker engine..." -ForegroundColor Yellow
docker context use desktop-linux 2>&1 | Out-Null
$info = docker info 2>&1
if ($info -notmatch "Server Version") {
    Write-Host "ERROR: Docker engine not running. Start Docker Desktop first." -ForegroundColor Red
    exit 1
}
$sv = ($info | Select-String "Server Version").ToString().Trim()
Write-Host "     Docker $sv" -ForegroundColor Green

# ── 2. Build backend image ─────────────────────────────────────────────────────
Write-Host "`n[2/6] Building backend image (aml-priority-backend:$Tag)..." -ForegroundColor Yellow
Write-Host "     This installs: fastapi, uvicorn, xgboost, shap, scikit-learn (~800MB)"
docker build -t "aml-priority-backend:$Tag" -f backend/Dockerfile .
if ($LASTEXITCODE -ne 0) { Write-Host "Backend build FAILED" -ForegroundColor Red; exit 1 }
Write-Host "     Backend image built OK" -ForegroundColor Green

# ── 3. Build frontend image ────────────────────────────────────────────────────
Write-Host "`n[3/6] Building frontend image (aml-priority-frontend:$Tag)..." -ForegroundColor Yellow
docker build -t "aml-priority-frontend:$Tag" -f frontend/Dockerfile .
if ($LASTEXITCODE -ne 0) { Write-Host "Frontend build FAILED" -ForegroundColor Red; exit 1 }
Write-Host "     Frontend image built OK" -ForegroundColor Green

# ── 4. Start via docker-compose ────────────────────────────────────────────────
Write-Host "`n[4/6] Starting services with docker-compose..." -ForegroundColor Yellow
docker-compose up -d --build
if ($LASTEXITCODE -ne 0) { Write-Host "docker-compose up FAILED" -ForegroundColor Red; exit 1 }
Write-Host "     Containers started. Waiting 20s for startup..."
Start-Sleep -Seconds 20

# ── 5. Health check ────────────────────────────────────────────────────────────
Write-Host "`n[5/6] Testing backend health..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod http://localhost:8000/api/health -TimeoutSec 15
    Write-Host "     /api/health: $($health.status) | model_loaded=$($health.model_loaded)" -ForegroundColor Green
} catch {
    Write-Host "     /api/health FAILED: $_" -ForegroundColor Red
    Write-Host "     Check logs: docker-compose logs backend" -ForegroundColor Yellow
}

Write-Host "`n[6/6] Testing frontend..." -ForegroundColor Yellow
try {
    $fe = Invoke-WebRequest http://localhost:3000 -UseBasicParsing -TimeoutSec 10
    Write-Host "     Frontend HTTP $($fe.StatusCode) OK" -ForegroundColor Green
} catch {
    Write-Host "     Frontend FAILED: $_" -ForegroundColor Red
}

# ── Summary ────────────────────────────────────────────────────────────────────
Write-Host "`n=== Build Summary ===" -ForegroundColor Cyan
docker images --format "{{.Repository}}:{{.Tag}}  {{.Size}}  ({{.CreatedSince}})" | Select-String "aml-priority"
Write-Host "`nFrontend: http://localhost:3000"
Write-Host "Backend:  http://localhost:8000/api/docs"
Write-Host "`nTo stop:  docker-compose down"
Write-Host "To view logs: docker-compose logs -f"
