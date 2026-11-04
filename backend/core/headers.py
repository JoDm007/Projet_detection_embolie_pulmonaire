# backend/core/headers.py
# En-têtes de sécurité HTTP (cf. plan2 § 4.8). Tout est servi depuis la même origine :
# aucune source externe n'est autorisée.
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

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


STATIC_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


class SecurityHeadersMiddleware:
    """Middleware ASGI pur : il n'ajoute que des en-têtes au message de début de réponse.
    (BaseHTTPMiddleware ferait transiter tout le corps par des flux en mémoire, ce qui
    ralentit l'envoi des volumes de ~200 Mo au viewer.)"""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        is_api = scope["path"].startswith("/api/")

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in STATIC_HEADERS.items():
                    headers[name] = value
                if is_api:
                    # Résultats d'analyse : jamais en cache navigateur
                    headers["Cache-Control"] = "no-store"
            await send(message)

        await self.app(scope, receive, send_with_headers)
