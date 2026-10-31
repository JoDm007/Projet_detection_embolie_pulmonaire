# backend/tests/conftest.py
# Chaque test tourne avec un app_data/ temporaire : la vraie base n'est jamais touchée.
import pytest


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("PESEG_DATA_DIR", str(tmp_path / "app_data"))
    monkeypatch.setenv("PESEG_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("PESEG_FUMPE_DIR", str(tmp_path / "fumpe"))   # aucun cas de démo
    from backend.core.config import get_settings
    get_settings.cache_clear()
    s = get_settings()
    from backend.db.database import init_db
    init_db(s.db_path)
    yield s
    get_settings.cache_clear()


@pytest.fixture
def client(settings):
    from fastapi.testclient import TestClient
    from backend.main import create_app
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture
def make_user(settings):
    from backend.core.security import hash_password
    from backend.db.repositories import UserRepo

    def _make(username="dr_test", password="motdepasse-42", role="radiologue"):
        UserRepo.create(username, hash_password(password), role)
        return username, password
    return _make


CSRF = {"X-Requested-With": "PESeg"}


def login(client, username, password):
    return client.post("/api/auth/login", json={"username": username, "password": password}, headers=CSRF)
