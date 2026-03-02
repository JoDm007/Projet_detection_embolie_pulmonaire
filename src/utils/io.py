# io.py
from pathlib import Path

def get_project_root() -> Path:
    """Retourne le chemin absolu vers la racine du projet."""
    # Fonctionne que le code soit lancé depuis notebook, script ou test
    current = Path(__file__).resolve()
    while current.name != "FUMPE_Detection" and current.parent != current:
        current = current.parent
    return current

def get_data_path(subpath: str = "") -> Path:
    """Retourne le chemin vers un sous-dossier de data/."""
    return get_project_root() / "data" / subpath