# src/utils/io.py
from pathlib import Path

def get_project_root() -> Path:
    """Retourne le chemin absolu vers la racine du projet."""
    current = Path(__file__).resolve()
    
    # ✅ Option 1 : Chercher le dossier qui contient 'src' et 'data'
    while not (current / "src").exists() and not (current / "data").exists():
        current = current.parent
        if current == current.parent:  # Racine du disque atteinte
            break
    return current

def get_data_path(subpath: str = "") -> Path:
    """Retourne le chemin vers un sous-dossier de data/."""
    return get_project_root() / "data" / subpath

def get_src_path(subpath: str = "") -> Path:
    """Retourne le chemin vers un sous-dossier de src/."""
    return get_project_root() / "src" / subpath