# backend/core/security.py
# Mots de passe (Argon2) et jetons de session (cf. plan2 § 4.2).
# Le jeton brut ne vit que dans le cookie du navigateur ; la base ne stocke que son SHA-256,
# donc une copie volée de app.db ne permet pas d'ouvrir une session.
import hashlib
import secrets

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

SESSION_COOKIE = "peseg_session"
# En-tête exigé sur toute requête qui modifie l'état : un site tiers ne peut pas le poser
# sans requête CORS préalable (refusée), ce qui bloque le CSRF en complément de SameSite=Strict.
CSRF_HEADER = "X-Requested-With"
CSRF_VALUE = "PESeg"
MIN_PASSWORD_LENGTH = 10

_hasher = PasswordHasher()
# Hash factice : quand l'utilisateur n'existe pas, on vérifie quand même un mot de passe
# pour que le temps de réponse ne révèle pas quels comptes existent.
_DUMMY_HASH = _hasher.hash(secrets.token_urlsafe(16))


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str | None, password: str) -> bool:
    try:
        return _hasher.verify(password_hash or _DUMMY_HASH, password) and password_hash is not None
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def password_problem(password: str) -> str | None:
    """Règle minimale de robustesse ; renvoie un message si le mot de passe est refusé."""
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Le mot de passe doit contenir au moins {MIN_PASSWORD_LENGTH} caractères."
    if password.isalpha() or password.isdigit():
        return "Le mot de passe doit mêler lettres et chiffres ou symboles."
    return None


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
