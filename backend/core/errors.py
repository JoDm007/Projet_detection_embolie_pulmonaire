# backend/core/errors.py
# Gestion centrale des erreurs (cf. plan2 § 4.10) : le navigateur ne reçoit jamais de trace
# Python, seulement un message lisible et, pour les erreurs imprévues, un identifiant à
# retrouver dans logs/app.log.
import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("backend.errors")


class AppError(Exception):
    """Erreur prévue, dont le message peut être montré tel quel à l'utilisateur."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        error_id = uuid.uuid4().hex[:8]
        logger.exception(f"Erreur interne {error_id} sur {request.method} {request.url.path}")
        return JSONResponse(status_code=500,
                            content={"detail": "Erreur interne du serveur.", "error_id": error_id})
