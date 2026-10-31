# backend/api/routes/auth.py
from fastapi import APIRouter, Depends, HTTPException, Request, Response

from backend.api.deps import csrf_protect, current_user
from backend.core.config import get_settings
from backend.core.security import (SESSION_COOKIE, new_session_token, token_digest,
                                   verify_password)
from backend.db.database import utcnow
from backend.db.repositories import AuditRepo, SessionRepo, UserRepo
from backend.schemas.auth import LoginIn, UserOut

router = APIRouter(prefix="/api/auth", tags=["authentification"])

# Message unique : ne dit pas si c'est l'identifiant ou le mot de passe qui est faux
BAD_CREDENTIALS = "Identifiant ou mot de passe incorrect."


@router.post("/login", response_model=UserOut, dependencies=[Depends(csrf_protect)])
def login(body: LoginIn, response: Response):
    settings = get_settings()
    user = UserRepo.get_by_username(body.username.strip())

    if user and user["locked_until"] and user["locked_until"] > utcnow():
        AuditRepo.log("connexion_bloquee", user["username"])
        raise HTTPException(status_code=429, detail="Compte temporairement bloqué après plusieurs "
                                                    "échecs. Réessayez plus tard.")

    ok = verify_password(user["password_hash"] if user else None, body.password)
    if not ok or not user["active"]:
        if user:
            UserRepo.register_failure(user["id"], settings.login_max_failures, settings.lockout_minutes)
        AuditRepo.log("connexion_echec", body.username.strip()[:64])
        raise HTTPException(status_code=401, detail=BAD_CREDENTIALS)

    UserRepo.reset_failures(user["id"])
    token = new_session_token()
    SessionRepo.create(token_digest(token), user["id"], settings.session_hours)
    response.set_cookie(SESSION_COOKIE, token, max_age=settings.session_hours * 3600,
                        httponly=True, samesite="strict", secure=settings.secure_cookies, path="/")
    AuditRepo.log("connexion", user["username"])
    return UserOut(username=user["username"], role=user["role"])


@router.post("/logout", status_code=204, dependencies=[Depends(csrf_protect)])
def logout(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        SessionRepo.delete(token_digest(token))
    response = Response(status_code=204)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@router.get("/me", response_model=UserOut)
def me(user=Depends(current_user)):
    return UserOut(username=user["username"], role=user["role"])
