# backend/core/headers.py
# En-têtes de sécurité HTTP (cf. plan2 § 4.8). Tout est servi depuis la même origine :
# aucune source externe n'est autorisée.
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# NiiVue (WebGL) a besoin de blob:/data: pour ses textures et workers, et de WebAssembly
# pour certains décodeurs. 'unsafe-inline' n'est autorisé que pour les styles (NiiVue en pose).
CSP = "; ".join([
    "default-src 'self'",
    "script-src 'self' 'wasm-unsafe-eval'",
    "worker-src 'self' blob:",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "connect-src 'self' blob: data:",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if request.url.path.startswith("/api/"):
            # Résultats d'analyse : jamais en cache navigateur
            response.headers["Cache-Control"] = "no-store"
        return response
