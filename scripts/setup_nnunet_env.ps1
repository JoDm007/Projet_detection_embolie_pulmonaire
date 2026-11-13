# scripts/setup_nnunet_env.ps1
# Étape 1 O5 — Crée l'environnement NN_env (Windows natif) pour nnU-Net v2 officiel.
#
# Séparé d'EP_env pour ne pas casser O1–O4 ni l'app.
# PyTorch a le support CUDA sous Windows natif (contrairement à TensorFlow) :
# pas besoin de WSL pour O5.
#
# Usage (depuis la racine du projet, dans PowerShell) :
#     .\scripts\setup_nnunet_env.ps1
#
# Durée : ~5–10 min (téléchargement de PyTorch CUDA ~2,5 Go).

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Env  = Join-Path $Root "NN_env"

Write-Host "== 1. Création du venv NN_env (Python 3.12) ==" -ForegroundColor Cyan
if (-not (Test-Path $Env)) {
    py -3.12 -m venv $Env
} else {
    Write-Host "NN_env existe déjà — réutilisé."
}
$Py = Join-Path $Env "Scripts\python.exe"

Write-Host "== 2. pip à jour ==" -ForegroundColor Cyan
& $Py -m pip install --upgrade pip

Write-Host "== 3. PyTorch + torchvision CUDA 13.0 (RTX 4060, pilote 592) ==" -ForegroundColor Cyan
# torch ET torchvision depuis le même index CUDA : sinon nnunetv2 → timm → torchvision (PyPI)
# exige un torch plus récent et pip le remplace par une version CPU (vu le 2026-11-13).
& $Py -m pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu130

Write-Host "== 4. nnU-Net v2 + dépendances de conversion ==" -ForegroundColor Cyan
# nnunetv2 apporte SimpleITK, nibabel, batchgenerators, etc.
# scipy/pydicom/pandas : nécessaires pour relire le DICOM + .mat via src/O1_preprocessing/loader.py
& $Py -m pip install nnunetv2 pydicom scipy pandas h5py

Write-Host "== 5. Vérification ==" -ForegroundColor Cyan
& $Py -c "import torch; print('torch', torch.__version__, '| CUDA dispo :', torch.cuda.is_available()); print('GPU :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'AUCUN'); import nnunetv2, importlib.metadata as m; print('nnunetv2', m.version('nnunetv2'))"

Write-Host ""
Write-Host "Terminé. Avant toute commande nnU-Net, dans chaque nouveau terminal :" -ForegroundColor Green
Write-Host "    .\NN_env\Scripts\Activate.ps1"
Write-Host "    . .\scripts\nnunet_env.ps1"
