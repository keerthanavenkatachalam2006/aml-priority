#!/usr/bin/env pwsh
# AML Priority — Start backend and frontend dev servers
# Usage: .\aml-priority\start.ps1

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition

Write-Host "=== AML Priority Startup ===" -ForegroundColor Cyan

# ── Check model artifacts ──────────────────────────────────────────────────
$metaPath = Join-Path $ROOT "ml\artifacts\model_metadata.json"
if (-not (Test-Path $metaPath)) {
    Write-Host "[WARN] Model artifacts not found. Run the ML pipeline first:" -ForegroundColor Yellow
    Write-Host "  python ml/src/preprocess.py" -ForegroundColor Gray
    Write-Host "  python ml/src/train.py --trials 15" -ForegroundColor Gray
    Write-Host "  python ml/src/evaluate.py" -ForegroundColor Gray
    Write-Host "  python backend/seed_db.py" -ForegroundColor Gray
} else {
    Write-Host "[OK] Model artifacts found" -ForegroundColor Green
}

# ── Seed DB if empty ───────────────────────────────────────────────────────
$dbPath = Join-Path $ROOT "backend\data\aml_priority.db"
if (-not (Test-Path $dbPath)) {
    Write-Host "Seeding database..." -ForegroundColor Yellow
    Push-Location (Join-Path $ROOT "backend")
    python seed_db.py
    Pop-Location
}

# ── Start backend ──────────────────────────────────────────────────────────
Write-Host "Starting backend on http://localhost:8000 ..." -ForegroundColor Cyan
$backendDir = Join-Path $ROOT "backend"
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "Set-Location '$backendDir'; python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

Start-Sleep -Seconds 3

# ── Start frontend ─────────────────────────────────────────────────────────
Write-Host "Starting frontend on http://localhost:5173 ..." -ForegroundColor Cyan
$frontendDir = Join-Path $ROOT "frontend"
Start-Process powershell -ArgumentList "-NoExit", "-Command",
    "Set-Location '$frontendDir'; npm run dev"

Write-Host ""
Write-Host "Backend API:  http://localhost:8000/api/docs" -ForegroundColor Green
Write-Host "Frontend:     http://localhost:5173" -ForegroundColor Green
Write-Host ""
Write-Host "Run tests: python -m pytest aml-priority/tests/ -v" -ForegroundColor Gray
