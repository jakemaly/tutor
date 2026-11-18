"""The snapshot server's route guard, loaded the way vLLM loads `--middleware` classes."""

import asyncio

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from vllm_route_guard import INTERNAL_HEADER, PUBLIC_ROUTES, TOKEN_ENV, RouteGuard

TOKEN = "container-token"
# Representative vLLM dev-mode and non-tutor routes; none may be reachable from outside.
BLOCKED = [
    ("POST", "/sleep?level=1"),
    ("POST", "/wake_up"),
    ("GET", "/is_sleeping"),
    ("POST", "/collective_rpc"),
    ("GET", "/server_info"),
    ("POST", "/reset_prefix_cache"),
    ("POST", "/update_weights"),
    ("GET", "/metrics"),
    ("POST", "/v1/completions"),
    ("POST", "/tokenize"),
    ("GET", "/docs"),
    ("GET", "/v1/chat/completions"),  # wrong method on a public path
    ("POST", "/sleep/"),
    ("POST", "/SLEEP"),
]


def vllm_like_app() -> Starlette:
    async def chat(request):
        async def tokens():
            for token in ("Hello", " there"):
                yield f"data: {token}\n\n"

        return StreamingResponse(tokens(), media_type="text/event-stream")

    async def ok(request):
        return JSONResponse({"path": request.url.path})

    async def socket(websocket):
        await websocket.accept()
        await websocket.send_text("open")
        await websocket.close()

    paths = {path.split("?")[0].rstrip("/") or "/" for _, path in BLOCKED}
    routes = [Route("/v1/chat/completions", chat, methods=["POST"])]
    routes += [Route("/v1/models", ok), Route("/health", ok)]
    routes += [Route(path, ok, methods=["GET", "POST"]) for path in sorted(paths)]
    routes.append(WebSocketRoute("/v1/realtime", socket))
    app = Starlette(routes=routes)
    app.add_middleware(RouteGuard)  # vLLM: app.add_middleware(imported) with no arguments
    return app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    with TestClient(vllm_like_app()) as test_client:
        yield test_client


def test_public_routes_pass_through_and_stream_unchanged(client):
    assert PUBLIC_ROUTES == {
        ("POST", "/v1/chat/completions"),
        ("GET", "/v1/models"),
        ("GET", "/health"),
    }
    response = client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 200
    assert response.text == "data: Hello\n\ndata:  there\n\n"
    assert response.headers["content-type"].startswith("text/event-stream")
    assert client.get("/v1/models").status_code == 200
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize(("method", "path"), BLOCKED)
def test_other_routes_are_hidden_without_the_internal_token(client, method, path):
    for headers in ({}, {INTERNAL_HEADER: "wrong"}, {"Authorization": f"Bearer {TOKEN}"}):
        response = client.request(method, path, headers=headers)
        assert response.status_code == 404
        assert response.json() == {"error": "Not Found"}


@pytest.mark.parametrize(("method", "path"), BLOCKED[:7])
def test_internal_token_reaches_sleep_wake_and_dev_routes(client, method, path):
    response = client.request(method, path, headers={INTERNAL_HEADER: TOKEN})
    assert response.status_code == 200


def test_websockets_need_the_internal_token(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/v1/realtime") as websocket:
            websocket.receive_text()
    with client.websocket_connect("/v1/realtime", headers={INTERNAL_HEADER: TOKEN}) as websocket:
        assert websocket.receive_text() == "open"


def test_guard_refuses_to_start_without_a_token(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    with pytest.raises(RuntimeError, match=TOKEN_ENV):
        RouteGuard(PlainTextResponse("unused"))


def test_allowed_requests_keep_the_original_receive_and_send():
    """vLLM sees the client's own disconnect message, so cancellation still aborts generation."""
    seen = {}

    async def inner(scope, receive, send):
        seen.update(receive=receive, send=send)

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        pass

    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []}
    asyncio.run(RouteGuard(inner, token=TOKEN)(scope, receive, send))
    assert seen == {"receive": receive, "send": send}
