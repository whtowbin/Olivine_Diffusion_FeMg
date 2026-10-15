"""Lock the local server to this launch: 127.0.0.1, a random port, a secret token.

The app is a web server, so any local process could otherwise drive it (and through
the project actions read and write files). Each launch makes a token
(`secrets.token_urlsafe(32)`) and opens the window or browser at ``/?token=...``. The
middleware lets a request through only if its ``token`` query parameter or its
``forster_token`` cookie matches; the first page load sets that cookie (HttpOnly,
SameSite=Strict), which carries the token to the page's scripts, styles and its
socket.io connection (HTTP polling and the websocket). Anything else gets 403, and a
websocket is closed before it is accepted (uvicorn answers the upgrade with 403).
"""

from __future__ import annotations

import hmac
import secrets
import socket
from http.cookies import SimpleCookie
from typing import Any
from urllib.parse import parse_qs

HOST = "127.0.0.1"
COOKIE = "forster_token"


def free_port() -> int:
    """A port the OS picks at random from the free ephemeral ports."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def new_token() -> str:
    return secrets.token_urlsafe(32)


class TokenMiddleware:
    """Pure ASGI middleware, so it covers HTTP and websocket scopes alike."""

    def __init__(self, app: Any, token: str) -> None:
        self.app, self.token = app, token

    def _given(self, scope: dict[str, Any]) -> tuple[bool, bool]:
        """(allowed, came in the query string)."""
        query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
        for value in query.get("token", []):
            if hmac.compare_digest(value, self.token):
                return True, True
        for name, value in scope.get("headers", []):
            if name == b"cookie":
                morsel = SimpleCookie(value.decode("latin-1")).get(COOKIE)
                if morsel and hmac.compare_digest(morsel.value, self.token):
                    return True, False
        return False, False

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)  # lifespan
            return
        allowed, in_query = self._given(scope)
        if not allowed:
            if scope["type"] == "websocket":
                await receive()  # websocket.connect
                await send({"type": "websocket.close", "code": 1008})
                return
            await send(
                {
                    "type": "http.response.start",
                    "status": 403,
                    "headers": [(b"content-type", b"text/plain")],
                }
            )
            await send({"type": "http.response.body", "body": b"forbidden: no launch token"})
            return
        if not in_query or scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        cookie = f"{COOKIE}={self.token}; Path=/; HttpOnly; SameSite=Strict".encode()

        async def send_with_cookie(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                message["headers"] = [*message.get("headers", []), (b"set-cookie", cookie)]
            await send(message)

        await self.app(scope, receive, send_with_cookie)


def protect(app: Any) -> str:
    """Install the middleware on the NiceGUI (FastAPI) app; returns this launch's token."""
    token = new_token()
    app.add_middleware(TokenMiddleware, token=token)
    return token
