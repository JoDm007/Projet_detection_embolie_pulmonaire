# backend/api/deps.py
# Dépendances FastAPI partagées par les routes : utilisateur courant, rôle admin, anti-CSRF.
from fastapi import Depends, HTTPException, Request

from backend.core.security import CSRF_HEADER, CSRF_VALUE, SESSION_COOKIE, token_digest
from backend.db.repositories import SessionRepo


def current_user(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    user = SessionRepo.get_user(token_digest(token)) if token else None
    if user is None:
        raise HTTPException(status_code=401, detail="Session absente ou expirée : reconnectez-vous.")
    return user


def require_admin(user=Depends(current_user)):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Réservé aux administrateurs.")
    return user


def csrf_protect(request: Request):
    """À placer sur toute route qui modifie l'état (POST, DELETE)."""
    if request.headers.get(CSRF_HEADER) != CSRF_VALUE:
        raise HTTPException(status_code=403, detail="Requête refusée (en-tête anti-CSRF manquant).")


def job_manager(request: Request):
    return request.app.state.jobs
