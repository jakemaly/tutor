"""Expose only the tutor's routes from the snapshot deployment's vLLM server.

The snapshot deployment runs vLLM with `VLLM_SERVER_DEV_MODE=1`, because `/sleep` and
`/wake_up` exist only in dev mode. Dev mode also adds administrative routes such as
`/collective_rpc`, `/server_info`, the `/reset_*` routes, and weight updates. vLLM warns against
exposing them, and Modal proxy auth only checks that a caller holds a workspace proxy token.

vLLM loads this ASGI middleware with `--middleware vllm_route_guard.RouteGuard`; it runs before
vLLM's routes. Requests to PUBLIC_ROUTES pass through. Every other request gets a 404 unless it
carries the token that the Modal container generated at startup in INTERNAL_HEADER. Only
the container's own process knows that token, so sleep and wake stay usable from inside it. The
client address cannot tell internal callers apart, because Modal's proxy may also connect
over loopback.

This module uses only the standard library and passes requests through unchanged, so streaming
responses and client disconnects (which cancel generation) behave as before.
"""

import hmac
import json
import os

TOKEN_ENV = "TUTOR_VLLM_INTERNAL_TOKEN"
INTERNAL_HEADER = "X-Tutor-Internal-Token"

# What the tutor backend and the deployment checks need from outside the container.
PUBLIC_ROUTES = frozenset(
    {
        ("POST", "/v1/chat/completions"),
        ("GET", "/v1/models"),
        ("GET", "/health"),
    }
)

_HEADER_KEY = INTERNAL_HEADER.lower().encode()
_NOT_FOUND = json.dumps({"error": "Not Found"}).encode()


class RouteGuard:
    """Pure ASGI middleware: allow PUBLIC_ROUTES, or any route with the internal token."""

    def __init__(self, app, token: str | None = None) -> None:
        self.app = app
        token = os.environ.get(TOKEN_ENV) if token is None else token
        if not token:
            # Fail closed: without a token the server would have no internal-only access path.
            raise RuntimeError(f"{TOKEN_ENV} must be set to start vLLM with RouteGuard.")
        self._token = token.encode()

    def _is_internal(self, scope) -> bool:
        return any(
            name == _HEADER_KEY and hmac.compare_digest(value, self._token)
            for name, value in scope.get("headers", ())
        )

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] not in ("http", "websocket"):
            # Lifespan events carry no request.
            await self.app(scope, receive, send)
            return
        # Same path handling as vLLM's own API-key middleware.
        path = scope["path"].removeprefix(scope.get("root_path", ""))
        if (scope.get("method"), path) in PUBLIC_ROUTES or self._is_internal(scope):
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            # Closing before accepting rejects the handshake.
            await send({"type": "websocket.close", "code": 1008})
            return
        await send(
            {
                "type": "http.response.start",
                "status": 404,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(_NOT_FOUND)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": _NOT_FOUND})
