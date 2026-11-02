"""End-to-end checks for the FastAPI, LangGraph, and assistant-ui stream bridge."""

# asyncio drives async graph calls; json decodes the line-based streaming protocol in assertions.
import asyncio
import io
import json
from uuid import uuid4

# pytest and httpx test routes and emulate the model endpoint without a network.
import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr
from pypdf import PdfWriter

from app.config import Settings
from app.documents import UploadedSource, create_source
from app.graph import SYSTEM_PROMPT, create_graph
from app.main import (
    ChatRequest,
    _citation_link,
    _fallback_citations,
    _messages_for_graph,
    create_app,
)
from app.models import create_chat_model


def configured_settings(**overrides: object) -> Settings:
    """Create endpoint settings without reading a developer's local .env file."""
    defaults: dict[str, object] = {
        "llm_base_url": "https://example.invalid/v1",
        "llm_model": "test-model",
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def user_message(text: str) -> dict[str, object]:
    """Build a user message in the text-part format sent by assistant-ui."""
    return {"role": "user", "content": [{"type": "text", "text": text}]}


class RecordingFakeChatModel(FakeListChatModel):
    """Fake streaming model that records the history sent to the model."""

    _seen_messages: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    async def _astream(self, messages, **kwargs):
        """Remember the complete model input, then retain the fake model's chunk behavior."""
        self._seen_messages.append(list(messages))
        async for chunk in super()._astream(messages, **kwargs):
            yield chunk


class CancellableFakeChatModel(FakeListChatModel):
    """Fake model for checking that a closed response cancels model streaming."""

    _was_cancelled: bool = PrivateAttr(default=False)

    async def _astream(self, messages, **kwargs):
        """Mark cleanup when closing the response interrupts the model's async generator."""
        try:
            async for chunk in super()._astream(messages, **kwargs):
                yield chunk
        finally:
            self._was_cancelled = True


class StaticEventStream(httpx.AsyncByteStream):
    """Small OpenAI-compatible SSE response split across transport chunks."""

    def __init__(self, events: list[str]) -> None:
        """Save SSE frames that the fake HTTP response will yield in separate chunks."""
        self.events = events

    async def __aiter__(self):
        """Encode each event independently to exercise incremental HTTP streaming."""
        for event in self.events:
            yield event.encode()

    async def aclose(self) -> None:
        """Satisfy httpx's async stream interface; this static fixture owns no resources."""
        return None


def test_health_reports_configuration_without_disclosing_it() -> None:
    """Health is available and reveals configured status without returning endpoint details."""
    client = TestClient(create_app(settings=configured_settings()))

    response = client.get("/api/health")

    assert response.json() == {"status": "ok", "model_configured": True}
    assert "example.invalid" not in response.text


def test_missing_configuration_keeps_health_available_and_chat_actionable() -> None:
    """An unconfigured app stays inspectable and tells chat callers what to configure."""
    client = TestClient(create_app(settings=Settings(_env_file=None)))

    assert client.get("/api/health").json() == {
        "status": "ok",
        "model_configured": False,
    }
    response = client.post("/api/chat", json={"messages": [user_message("Hi")]})

    assert response.status_code == 503
    assert "LLM_BASE_URL and LLM_MODEL" in response.json()["detail"]


def test_streams_incremental_text_and_uses_the_full_supplied_history() -> None:
    """The bridge streams deltas and forwards prior turns while dropping the empty run stub."""
    model = RecordingFakeChatModel(responses=["Hello from the model."])
    client = TestClient(create_app(settings=configured_settings(), model=model))

    response = client.post(
        "/api/chat",
        json={
            "messages": [
                user_message("First turn"),
                {"role": "assistant", "content": [{"type": "text", "text": "Prior answer"}]},
                user_message("Second turn"),
                {"role": "assistant", "content": []},
            ]
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    deltas = [json.loads(line[2:]) for line in response.text.splitlines() if line.startswith("0:")]
    assert "".join(deltas) == "Hello from the model."
    assert len(deltas) > 1
    history = model._seen_messages[0]
    assert [message.content for message in history] == [
        SYSTEM_PROMPT,
        "First turn",
        "Prior answer",
        "Second turn",
    ]


def test_empty_assistant_placeholder_is_removed_but_nonempty_history_is_kept() -> None:
    """Only assistant-ui's empty final placeholder is removed from the supplied history."""
    messages = _messages_for_graph(
        [
            user_message("Question"),
            {"role": "assistant", "content": [{"type": "text", "text": "Earlier reply"}]},
            {"role": "assistant", "content": []},
        ]
    )

    assert [message.content for message in messages] == ["Question", "Earlier reply"]


def test_conversations_are_independent_when_each_request_supplies_its_history() -> None:
    """Separate requests use their own browser-supplied histories rather than server memory."""
    model = FakeListChatModel(responses=["First answer", "Other conversation"])
    client = TestClient(create_app(settings=configured_settings(), model=model))

    first = client.post("/api/chat", json={"messages": [user_message("A")]})
    second = client.post("/api/chat", json={"messages": [user_message("B")]})

    first_deltas = [
        json.loads(line[2:]) for line in first.text.splitlines() if line.startswith("0:")
    ]
    second_deltas = [
        json.loads(line[2:]) for line in second.text.splitlines() if line.startswith("0:")
    ]
    assert "".join(first_deltas) == "First answer"
    assert "".join(second_deltas) == "Other conversation"


def test_rejects_non_text_parts_and_unsupported_roles() -> None:
    """The text-only API rejects image/file content and roles the graph cannot represent."""
    client = TestClient(
        create_app(
            settings=configured_settings(),
            model=FakeListChatModel(responses=["unused"]),
        )
    )

    image = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": [{"type": "file", "data": "x"}]}]},
    )
    tool = client.post("/api/chat", json={"messages": [{"role": "tool", "content": "x"}]})

    assert image.status_code == 422
    assert tool.status_code == 422


def test_uploads_serves_and_removes_validated_image_source() -> None:
    """An uploaded source is addressable by ID and removal invalidates its link."""
    client = TestClient(create_app(settings=configured_settings()))
    source_id = str(uuid4())
    png = b"\x89PNG\r\n\x1a\n" + b"image data"

    upload = client.put(
        f"/api/attachments/{source_id}",
        content=png,
        headers={"content-type": "image/png", "x-file-name": "notes%20page.png"},
    )

    assert upload.status_code == 200
    assert upload.json() == {
        "id": source_id,
        "filename": "notes page.png",
        "mediaType": "image/png",
        "pageCount": 1,
    }
    served = client.get(f"/api/attachments/{source_id}?page=1")
    assert served.content == png
    assert served.headers["content-type"] == "image/png"
    assert client.get(f"/api/attachments/{source_id}?page=2").status_code == 404
    assert client.delete(f"/api/attachments/{source_id}").status_code == 204
    assert client.get(f"/api/attachments/{source_id}").status_code == 404


def test_upload_rejects_mismatched_mime_type_and_file_signature() -> None:
    """Client-provided content types do not bypass the file signature check."""
    client = TestClient(create_app(settings=configured_settings()))

    response = client.put(
        f"/api/attachments/{uuid4()}",
        content=b"not a png",
        headers={"content-type": "image/png", "x-file-name": "image.png"},
    )

    assert response.status_code == 422
    assert "contents do not match" in response.json()["detail"]


def test_chat_resolves_uploaded_image_and_validates_citation_pages() -> None:
    """Chat receives image bytes from the shared source ID; only valid citations link."""
    source_id = str(uuid4())
    model = RecordingFakeChatModel(responses=[f"I can see the image [[cite:{source_id}:1]]."])
    client = TestClient(create_app(settings=configured_settings(), model=model))
    png = b"\x89PNG\r\n\x1a\n" + b"image data"
    source = client.put(
        f"/api/attachments/{source_id}",
        content=png,
        headers={"content-type": "image/png", "x-file-name": "diagram.png"},
    )
    assert source.status_code == 200

    response = client.post(
        "/api/chat",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Explain this."},
                        {
                            "type": "file",
                            "data": source_id,
                            "mimeType": "image/png",
                            "filename": "diagram.png",
                        },
                    ],
                }
            ]
        },
    )

    assert response.status_code == 200
    stream_text = "".join(
        json.loads(line[2:]) for line in response.text.splitlines() if line.startswith("0:")
    )
    citation = f"[diagram.png, p. 1](/api/attachments/{source_id}?page=1#page=1)"
    assert f"I can see the image {citation}." in stream_text
    assert f"**Sources:** {citation}" in stream_text
    sent_content = model._seen_messages[0][-1].content
    assert any(part.get("type") == "image_url" for part in sent_content)
    assert any(source_id in part.get("text", "") for part in sent_content)

    uploaded_source = create_source(source_id, "diagram.png", "image/png", png)
    sources = {source_id: uploaded_source}
    assert _citation_link(source_id, 1, sources, {source_id}) == (
        f"[diagram.png, p. 1](/api/attachments/{source_id}?page=1#page=1)"
    )
    assert _citation_link(source_id, 2, sources, {source_id}) == "[source unavailable]"
    assert _citation_link(source_id, 1, sources, set()) == "[source unavailable]"


def test_pdf_source_keeps_page_boundaries() -> None:
    """Extracted PDF source records retain one text slot per original page."""
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_blank_page(width=100, height=100)
    buffer = io.BytesIO()
    writer.write(buffer)

    source = create_source(str(uuid4()), "lesson.pdf", "application/pdf", buffer.getvalue())

    assert source.page_count == 2
    assert source.pages == ("", "")


def test_chat_adds_relevant_page_link_when_model_omits_citation_marker() -> None:
    """The chat still cites a relevant attached page when the model ignores the marker format."""
    source_id = str(uuid4())
    source = UploadedSource(
        id=source_id,
        filename="Carbohydrates_Review.pdf",
        media_type="application/pdf",
        data=b"%PDF-test",
        pages=(
            "Glucose provides quick energy for cellular respiration.",
            "Cellulose is a structural polysaccharide in plant cell walls.",
        ),
    )
    app = create_app(
        settings=configured_settings(),
        model=FakeListChatModel(responses=["Cellulose supports plant cell walls."]),
    )
    app.state.sources[source_id] = source
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is cellulose used for?"},
                        {
                            "type": "file",
                            "data": source_id,
                            "mimeType": "application/pdf",
                            "filename": source.filename,
                        },
                    ],
                }
            ]
        },
    )
    stream_text = "".join(
        json.loads(line[2:]) for line in response.text.splitlines() if line.startswith("0:")
    )

    assert response.status_code == 200
    assert "**Sources:**" in stream_text
    assert (
        f"Cellulose supports plant cell walls. "
        f"[Carbohydrates_Review.pdf, p. 2](/api/attachments/{source_id}?page=2#page=2)"
        in stream_text
    )
    assert (
        f"[Carbohydrates_Review.pdf, p. 2](/api/attachments/{source_id}?page=2#page=2)"
        in stream_text
    )
    assert _fallback_citations(
        "Cellulose supports plant cell walls.", {source_id: source}, {source_id}
    ) == [f"[Carbohydrates_Review.pdf, p. 2](/api/attachments/{source_id}?page=2#page=2)"]


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user", "content": ""},
        {"role": "user", "content": "  "},
        {"role": "user"},
        {"role": "user", "content": [None]},
        {"role": "user", "content": [{"type": "text"}]},
    ],
)
def test_rejects_blank_or_malformed_user_messages(message: dict[str, object]) -> None:
    """Malformed or whitespace-only user input fails validation before reaching a model."""
    client = TestClient(
        create_app(
            settings=configured_settings(),
            model=FakeListChatModel(responses=["unused"]),
        )
    )

    response = client.post("/api/chat", json={"messages": [message]})

    assert response.status_code == 422


def test_graph_is_built_from_an_injected_langchain_model() -> None:
    """The provider-neutral graph can run against a simple in-memory LangChain model."""
    model = FakeListChatModel(responses=["A helpful answer"])
    graph = create_graph(model)

    response = asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="A question")]}))

    assert response["messages"][-1].content == "A helpful answer"
    assert len(SYSTEM_PROMPT) > 0


def test_model_adapter_uses_only_the_current_endpoint_configuration(monkeypatch) -> None:
    """Each adapter uses its URL, model, and optional key without provider-specific options."""
    calls: list[dict[str, object]] = []

    def fake_chat_openai(**kwargs: object) -> object:
        """Capture adapter arguments instead of constructing an HTTP client."""
        calls.append(kwargs)
        return object()

    monkeypatch.setattr("app.models.ChatOpenAI", fake_chat_openai)
    create_chat_model(
        configured_settings(
            llm_base_url="https://first.example/v1",
            llm_model="model-a",
            llm_api_key="first-key",
        )
    )
    create_chat_model(
        configured_settings(
            llm_base_url="https://second.example/v1",
            llm_model="model-b",
            llm_api_key="",
        )
    )

    assert calls[0]["base_url"] == "https://first.example/v1"
    assert calls[0]["model"] == "model-a"
    assert calls[0]["api_key"] == "first-key"
    assert calls[1]["base_url"] == "https://second.example/v1"
    assert calls[1]["model"] == "model-b"
    assert callable(calls[1]["api_key"])
    assert calls[1]["api_key"]() == ""
    assert calls[1]["use_responses_api"] is False
    assert calls[1]["stream_usage"] is False


def test_model_without_api_key_sends_no_authorization_header() -> None:
    """An endpoint that needs no key receives no placeholder Authorization header."""
    model = create_chat_model(configured_settings(llm_api_key=""))

    assert model.root_client.auth_headers == {}
    assert model.root_async_client.auth_headers == {}


def test_configured_graph_is_built_once_and_reused(monkeypatch) -> None:
    """The first configured chat builds the graph; later requests reuse it."""
    model = FakeListChatModel(responses=["First answer", "Second answer"])
    builds = 0
    original_create_graph = create_graph

    def count_graph_builds(selected_model):
        """Count graph construction while delegating to the real graph factory."""
        nonlocal builds
        builds += 1
        return original_create_graph(selected_model)

    monkeypatch.setattr("app.main.create_chat_model", lambda _settings: model)
    monkeypatch.setattr("app.main.create_graph", count_graph_builds)
    client = TestClient(create_app(settings=configured_settings()))

    first = client.post("/api/chat", json={"messages": [user_message("A")]})
    second = client.post("/api/chat", json={"messages": [user_message("B")]})

    assert first.status_code == second.status_code == 200
    assert builds == 1


def test_openai_compatible_stream_works_through_fastapi_and_langgraph() -> None:
    """A mocked Chat Completions SSE endpoint streams through the full Python bridge."""
    sent_requests: list[dict[str, object]] = []

    def event(delta: dict[str, str], finish_reason: str | None = None) -> str:
        """Encode one OpenAI-compatible streaming chunk as a server-sent event."""
        payload = {
            "id": "test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    events = [
        event({"role": "assistant"}),
        event({"content": "OpenAI-compatible "}),
        event({"content": "stream works."}),
        event({}, "stop"),
        "data: [DONE]\n\n",
    ]

    def respond(request: httpx.Request) -> httpx.Response:
        """Record the model payload and return the prepared streaming events."""
        sent_requests.append(json.loads(request.content))
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=StaticEventStream(events),
        )

    model_http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    model = ChatOpenAI(
        model="test-model",
        base_url="https://example.invalid/v1",
        api_key="test-key",
        use_responses_api=False,
        stream_usage=False,
        streaming=True,
        http_async_client=model_http_client,
    )
    client = TestClient(create_app(settings=configured_settings(), model=model))

    response = client.post(
        "/api/chat",
        json={
            "messages": [
                user_message("First turn"),
                {"role": "assistant", "content": [{"type": "text", "text": "Prior answer"}]},
                user_message("Second turn"),
                {"role": "assistant", "content": []},
            ]
        },
    )
    asyncio.run(model_http_client.aclose())

    deltas = [json.loads(line[2:]) for line in response.text.splitlines() if line.startswith("0:")]
    assert response.status_code == 200
    assert "".join(deltas) == "OpenAI-compatible stream works."
    assert len(deltas) > 1
    sent_messages = sent_requests[0]["messages"]
    assert [message["content"] for message in sent_messages] == [
        SYSTEM_PROMPT,
        "First turn",
        "Prior answer",
        "Second turn",
    ]


def test_model_failure_keeps_partial_text_and_reports_a_safe_actionable_error() -> None:
    """A failed stream retains text already sent and avoids exposing endpoint details."""
    model = FakeListChatModel(responses=["partial response"], error_on_chunk_number=7)
    client = TestClient(create_app(settings=configured_settings(), model=model))

    response = client.post("/api/chat", json={"messages": [user_message("Hi")]})

    assert response.status_code == 200
    assert '0:"p"' in response.text
    assert "model request failed" in response.text
    assert "example.invalid" not in response.text


def test_closing_a_stream_cancels_an_active_model_request() -> None:
    """Closing the HTTP stream propagates cancellation into the model's async generator."""
    model = CancellableFakeChatModel(responses=["a long response"], sleep=0.01)
    application = create_app(settings=configured_settings(), model=model)
    endpoint = next(
        route.endpoint
        for route in application.routes
        if getattr(route, "path", None) == "/api/chat"
    )

    async def receive_one_chunk_and_close() -> bytes:
        """Read one response event, then close it as if the browser pressed Stop."""
        response = await endpoint(ChatRequest(messages=[user_message("Stop this response")]))
        stream = response.body_iterator
        first_chunk = await anext(stream)
        await stream.aclose()
        return first_chunk

    first_chunk = asyncio.run(receive_one_chunk_and_close())

    assert "0:" in first_chunk
    assert model._was_cancelled
