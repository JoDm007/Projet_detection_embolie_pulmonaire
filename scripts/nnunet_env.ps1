# scripts/nnunet_env.ps1
# Variables d'environnement nnU-Net v2 — à "dot-sourcer" dans chaque terminal :
#     . .\scripts\nnunet_env.ps1
#
# Les trois dossiers sont sous data/ (gitignoré, volumineux).

$Root = Split-Path -Parent $PSScriptRoot
$Base = Join-Path $Root "data\nnunet"

$env:nnUNet_raw          = Join-Path $Base "nnUNet_raw"
$env:nnUNet_preprocessed = Join-Path $Base "nnUNet_preprocessed"
$env:nnUNet_results      = Join-Path $Base "nnUNet_results"

# Garde-fous RAM (PC 15 Go, gel du 03/11) : peu de workers d'augmentation.
# Valeur par défaut de nnU-Net = 12 → beaucoup trop pour cette machine.
$env:nnUNet_n_proc_DA = "3"

foreach ($d in @($env:nnUNet_raw, $env:nnUNet_preprocessed, $env:nnUNet_results)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
}

Write-Host "nnUNet_raw          = $env:nnUNet_raw"
Write-Host "nnUNet_preprocessed = $env:nnUNet_preprocessed"
Write-Host "nnUNet_results      = $env:nnUNet_results"
Write-Host "nnUNet_n_proc_DA    = $env:nnUNet_n_proc_DA"
