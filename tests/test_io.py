# tests/test_io.py
from src.utils.io import get_project_root, get_data_path, get_src_path

def test_project_root():
    root = get_project_root()
    assert root.exists(), "Racine du projet introuvable"
    assert (root / "src").exists(), "src/ introuvable"
    assert (root / "data").exists(), "data/ introuvable"
    print(f"✅ get_project_root() : {root}")

def test_data_path():
    data = get_data_path()
    assert data.exists(), "data/ introuvable"
    assert (data / "raw").exists(), "data/raw/ introuvable"
    print(f"✅ get_data_path() : {data}")

def test_src_path():
    src = get_src_path()
    assert src.exists(), "src/ introuvable"
    assert (src / "O1_preprocessing").exists(), "src/O1_preprocessing/ introuvable"
    print(f"✅ get_src_path() : {src}")

if __name__ == "__main__":
    test_project_root()
    test_data_path()
    test_src_path()
    print("\n🎉 Tous les tests de chemins sont passés !")