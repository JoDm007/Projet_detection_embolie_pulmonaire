# backend/tests/test_api.py — bloc api : authentification, droits, anti-CSRF, en-têtes
import uuid

from backend.db.repositories import AnalysisRepo, UserRepo

from .conftest import CSRF, login


def test_health_is_public_and_lists_engines(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    names = {e["name"] for e in r.json()["engines"]}
    assert {"classique", "unet"} <= names


def test_security_headers(client):
    r = client.get("/api/health")
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["cache-control"] == "no-store"


def test_protected_routes_require_session(client):
    for path in ("/api/auth/me", "/api/analyses", "/api/cases/demo"):
        assert client.get(path).status_code == 401


def test_login_logout_flow(client, make_user):
    username, password = make_user()
    r = login(client, username, password)
    assert r.status_code == 200 and r.json()["username"] == username
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert client.get("/api/auth/me").status_code == 200
    assert client.post("/api/auth/logout", headers=CSRF).status_code == 204
    assert client.get("/api/auth/me").status_code == 401


def test_wrong_password_same_message_as_unknown_user(client, make_user):
    username, _ = make_user()
    r1 = login(client, username, "faux-mot-2-passe")
    r2 = login(client, "inconnu", "faux-mot-2-passe")
    assert r1.status_code == r2.status_code == 401
    assert r1.json()["detail"] == r2.json()["detail"]


def test_lockout_after_failures(client, make_user, settings):
    username, password = make_user()
    for _ in range(settings.login_max_failures):
        login(client, username, "faux-mot-2-passe")
    assert login(client, username, password).status_code == 429


def test_csrf_header_required(client, make_user):
    username, password = make_user()
    r = client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 403


def test_cannot_see_other_users_analysis(client, make_user):
    make_user("alice", "alice-pass-123")
    make_user("bob", "bob-pass-12345")
    alice = UserRepo.get_by_username("alice")
    analysis_id = str(uuid.uuid4())
    AnalysisRepo.create(analysis_id, alice["id"], "demo", "classique", "PAT001")

    login(client, "bob", "bob-pass-12345")
    assert client.get(f"/api/analyses/{analysis_id}").status_code == 404
    assert client.delete(f"/api/analyses/{analysis_id}", headers=CSRF).status_code == 404
    assert client.get("/api/analyses").json() == []

    login(client, "alice", "alice-pass-123")
    assert client.get(f"/api/analyses/{analysis_id}").status_code == 200


def test_file_route_rejects_unknown_names(client, make_user):
    username, password = make_user()
    user = UserRepo.get_by_username(username)
    analysis_id = str(uuid.uuid4())
    AnalysisRepo.create(analysis_id, user["id"], "demo", "classique", "PAT001")
    AnalysisRepo.set_done(analysis_id, {})
    login(client, username, password)
    for name in ("app.db", "..%2F..%2Fapp.db", "findings.json.bak"):
        assert client.get(f"/api/analyses/{analysis_id}/{name}").status_code == 404


def test_upload_rejects_non_zip(client, make_user):
    username, password = make_user()
    login(client, username, password)
    engine_ok = any(e["name"] == "classique" and e["available"] for e in client.get("/api/health").json()["engines"])
    r = client.post("/api/analyses", files={"file": ("x.zip", b"pas un zip", "application/zip")},
                    data={"engine": "classique"}, headers=CSRF)
    # Sans modèles installés, le moteur est refusé avant même de lire le fichier
    assert r.status_code == (400 if engine_ok else 409)


def test_unknown_demo_case(client, make_user):
    username, password = make_user()
    login(client, username, password)
    assert client.post("/api/analyses/demo/PAT999", json={"engine": "classique"},
                       headers=CSRF).status_code == 404
    # Chemin normalisé avant le routage : refusé (404/405), jamais accepté
    assert client.post("/api/analyses/demo/..%2Fetc", json={}, headers=CSRF).status_code in (404, 405)
