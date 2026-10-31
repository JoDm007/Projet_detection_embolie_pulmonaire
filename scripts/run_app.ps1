# scripts/run_app.ps1 — lance PE-Seg AI en local (démo) en une commande.
#   .\scripts\run_app.ps1            # http://127.0.0.1:8000, navigateur ouvert automatiquement
#   .\scripts\run_app.ps1 -NoBrowser
#   .\scripts\run_app.ps1 -Port 8080
param(
    [int]$Port = 8000,
    [switch]$NoBrowser
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root "EP_env\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Environnement EP_env introuvable : $python" }

# Les logs de src/ contiennent des emojis : console en UTF-8
$env:PYTHONIOENCODING = "utf-8"
$env:PESEG_PORT = "$Port"

Write-Host "Vérification des modèles installés..."
& $python -m backend.cli check-models

if (-not $NoBrowser) {
    Start-Job -ScriptBlock { param($p) Start-Sleep -Seconds 4; Start-Process "http://127.0.0.1:$p" } -ArgumentList $Port | Out-Null
}

Write-Host "PE-Seg AI sur http://127.0.0.1:$Port  (Ctrl+C pour arrêter)"
& $python -m uvicorn backend.main:app --host 127.0.0.1 --port $Port
