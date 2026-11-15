"""Full path: streamed model tool call -> fixed renderer -> artifact card -> download."""

import asyncio
import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from test_chat import StaticEventStream, configured_settings, user_message

from app.artifacts import ArtifactStore
from app.graph import MAX_TOOL_CALL_RETRIES
from app.main import _messages_for_graph, _stream_graph, create_app

GUIDE = {
    "title": "Limit Proofs",
    "sections": [
        {"heading": "Core idea", "blocks": [{"type": "paragraph", "text": "SECRET-BODY-TEXT"}]}
    ],
}


def event(delta, finish_reason=None) -> str:
    chunk = {
        "id": "t",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return "data: " + json.dumps(chunk) + "\n\n"


def make_client(arguments: dict[str, object]):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            text = json.dumps(arguments)
            call = {"index": 0, "id": "doc-1", "type": "function"}
            events = [
                event(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                **call,
                                "function": {
                                    "name": "create_study_document",
                                    "arguments": text[:20],
                                },
                            }
                        ],
                    }
                ),
                event({"tool_calls": [{"index": 0, "function": {"arguments": text[20:]}}]}),
                event({}, "tool_calls"),
            ]
        else:
            events = [event({"content": "Your study guide is ready."}), event({}, "stop")]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=StaticEventStream([*events, "data: [DONE]\n\n"]),
        )

    model = ChatOpenAI(
        model="test-model",
        base_url="https://example.invalid/v1",
        api_key="k",
        use_responses_api=False,
        stream_usage=False,
        streaming=True,
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    return TestClient(create_app(settings=configured_settings(), model=model)), requests


def tool_events(text: str):
    return [
        (line[0], json.loads(line[2:]))
        for line in text.splitlines()
        if line.startswith(("b:", "c:", "a:"))
    ]


def test_tool_call_creates_a_downloadable_pdf_without_bytes_in_the_stream() -> None:
    client, requests = make_client({"format": "pdf", "guide": GUIDE})
    response = client.post("/api/chat", json={"messages": [user_message("Make this a PDF")]})

    names = {tool["function"]["name"] for tool in requests[0]["tools"]}
    assert names == {"load_skill", "create_study_document"}
    kinds = tool_events(response.text)
    assert [kind for kind, _ in kinds] == ["b", "c", "a"]
    assert json.loads(kinds[1][1]["argsTextDelta"]) == {"format": "pdf"}
    result = kinds[2][1]["result"]
    assert result["title"] == "Limit Proofs"
    assert result["outline"] == [{"heading": "Core idea", "blocks": 1}]
    # Neither the document text nor file bytes travel through the browser stream.
    assert "SECRET-BODY-TEXT" not in response.text and "%PDF" not in response.text
    # The model only sees a short confirmation.
    tool_message = requests[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert (
        "%PDF" not in tool_message["content"] and "SECRET-BODY-TEXT" not in tool_message["content"]
    )

    download = client.get(result["url"])
    assert download.status_code == 200
    assert download.content.startswith(b"%PDF")
    assert download.headers["content-disposition"].startswith("attachment;")
    inline = client.get(result["url"], params={"inline": "true"})
    assert inline.headers["content-disposition"].startswith("inline;")


def test_docx_is_never_served_inline() -> None:
    client, _ = make_client({"format": "docx", "guide": GUIDE})
    response = client.post("/api/chat", json={"messages": [user_message("Word please")]})
    url = tool_events(response.text)[2][1]["result"]["url"]

    served = client.get(url, params={"inline": "true"})
    assert served.headers["content-disposition"].startswith("attachment;")
    assert served.content.startswith(b"PK")


def test_invalid_guide_is_a_recoverable_error_and_creates_nothing() -> None:
    client, requests = make_client({"format": "pdf", "guide": {"title": "x", "sections": []}})
    response = client.post("/api/chat", json={"messages": [user_message("PDF")]})

    kinds = tool_events(response.text)
    assert kinds[2][1]["isError"] is True
    # Schema validation fails before the tool body runs, so the model gets LangGraph's error
    # text naming the bad field, and no artifact is stored.
    assert "sections" in requests[1]["messages"][-1]["content"]


def test_unknown_or_expired_artifact_is_404() -> None:
    client, _ = make_client({"format": "pdf", "guide": GUIDE})
    assert client.get("/api/artifacts/not-a-real-id").status_code == 404


def test_document_traces_are_not_replayed_to_the_model() -> None:
    messages = _messages_for_graph(
        [
            user_message("PDF please"),
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool-call",
                        "toolCallId": "d",
                        "toolName": "create_study_document",
                        "input": {},
                    }
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool-result",
                        "toolCallId": "d",
                        "toolName": "create_study_document",
                        "output": {},
                    }
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Ready."}]},
            user_message("Thanks"),
        ]
    )
    assert [m.content for m in messages] == ["PDF please", "Ready.", "Thanks"]


def stream_error(message: str, code: str) -> str:
    """A provider error delivered mid-stream, as Groq does for off-schema tool calls."""
    return "data: " + json.dumps({"error": {"message": message, "code": code}}) + "\n\n"


REJECTION = stream_error(
    "Tool call validation failed: parameters for tool create_study_document did not match "
    "schema: errors: [missing properties: 'format', `/guide`: expected object, but got string]",
    "tool_use_failed",
)


def tool_call_events(arguments: dict[str, object]) -> list[str]:
    text = json.dumps(arguments)
    call = {"index": 0, "id": "doc-1", "type": "function"}
    function = {"name": "create_study_document", "arguments": text[:20]}
    return [
        event({"role": "assistant", "tool_calls": [{**call, "function": function}]}),
        event({"tool_calls": [{"index": 0, "function": {"arguments": text[20:]}}]}),
        event({}, "tool_calls"),
    ]


def scripted_client(scripts: list[list[str]]):
    """Answer the nth model request with the nth list of events (the last one repeats)."""
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        events = scripts[min(len(requests), len(scripts)) - 1]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=StaticEventStream([*events, "data: [DONE]\n\n"]),
        )

    model = ChatOpenAI(
        model="test-model",
        base_url="https://example.invalid/v1",
        api_key="k",
        use_responses_api=False,
        stream_usage=False,
        streaming=True,
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    return TestClient(create_app(settings=configured_settings(), model=model)), requests


def test_provider_rejected_tool_call_is_retried_with_the_providers_feedback() -> None:
    client, requests = scripted_client(
        [
            [REJECTION],
            tool_call_events({"format": "pdf", "guide": GUIDE}),
            [event({"content": "Your study guide is ready."}), event({}, "stop")],
        ]
    )
    response = client.post("/api/chat", json={"messages": [user_message("Make this a PDF")]})

    assert len(requests) == 3
    feedback = requests[1]["messages"][-1]
    assert feedback["role"] == "user"
    assert "Your last tool call was rejected" in feedback["content"]
    assert "missing properties: 'format'" in feedback["content"]
    kinds = tool_events(response.text)
    assert [kind for kind, _ in kinds] == ["b", "c", "a"]
    assert kinds[2][1]["result"]["title"] == "Limit Proofs"
    assert "Your study guide is ready." in response.text
    assert "model request failed" not in response.text
    # The corrective note was for one attempt only; it is not part of the saved conversation.
    assert all(
        "Your last tool call was rejected" not in str(message.get("content"))
        for message in requests[2]["messages"]
    )


def test_repeated_provider_rejections_end_with_a_clear_message() -> None:
    client, requests = scripted_client([[REJECTION]])
    response = client.post("/api/chat", json={"messages": [user_message("Make this a PDF")]})

    assert len(requests) == MAX_TOOL_CALL_RETRIES + 1
    assert "could not produce a valid document" in response.text
    assert "model request failed" not in response.text


def test_unrelated_provider_errors_are_not_retried() -> None:
    client, requests = scripted_client(
        [[stream_error("Rate limit reached", "rate_limit_exceeded")]]
    )
    response = client.post("/api/chat", json={"messages": [user_message("Hello")]})

    assert len(requests) == 1
    assert "model request failed" in response.text


class FakeTool:
    def __init__(self) -> None:
        self.args, self.response = "", None

    def append_args_text(self, text: str) -> None:
        self.args += text

    def set_response(self, result, is_error=False) -> None:
        self.response = (result, is_error)


class FakeController:
    is_cancelled = False

    def __init__(self) -> None:
        self.tools: list[FakeTool] = []

    async def add_tool_call(self, name: str, call_id: str) -> FakeTool:
        tool = FakeTool()
        self.tools.append(tool)
        return tool

    def append_text(self, text: str) -> None:
        pass


class FailingGraph:
    """Announces tool calls, then dies before their results arrive."""

    async def astream(self, *_args, **_kwargs):
        calls = [
            {"name": "create_study_document", "args": {"format": "pdf", "guide": {}}, "id": "a"},
            {"name": "load_skill", "args": {"name": "study-document"}, "id": "b"},
        ]
        yield "updates", {"chat": {"messages": [AIMessage(content="", tool_calls=calls)]}}
        raise RuntimeError("provider went away")


def test_failed_run_settles_every_announced_tool_card() -> None:
    controller = FakeController()

    with pytest.raises(RuntimeError):
        asyncio.run(_stream_graph(FailingGraph(), [], controller))

    document, skill = controller.tools
    assert document.response == ({"error": "The document could not be created."}, True)
    assert skill.response == ({"loaded": False}, True)


def test_downloads_are_never_cached_or_sniffed() -> None:
    client, _ = make_client({"format": "pdf", "guide": GUIDE})
    response = client.post("/api/chat", json={"messages": [user_message("Make this a PDF")]})
    url = tool_events(response.text)[2][1]["result"]["url"]

    served = client.get(url)
    assert served.headers["cache-control"] == "private, no-store"
    assert served.headers["x-content-type-options"] == "nosniff"
    assert served.headers["content-disposition"] == 'attachment; filename="limit-proofs.pdf"'


def test_an_injected_store_serves_downloads() -> None:
    store = ArtifactStore()
    client = TestClient(create_app(settings=configured_settings(), artifacts=store))
    artifact = store.put("note.pdf", "application/pdf", b"%PDF-test")

    served = client.get(f"/api/artifacts/{artifact.id}")
    assert served.status_code == 200 and served.content == b"%PDF-test"


def test_default_in_memory_store_warns_on_vercel_only(monkeypatch, caplog) -> None:
    monkeypatch.setenv("VERCEL", "1")
    with caplog.at_level(logging.WARNING, logger="app.main"):
        create_app(settings=configured_settings())
    assert "kept in this instance's memory" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="app.main"):
        create_app(settings=configured_settings(), artifacts=ArtifactStore())
    assert "memory" not in caplog.text


def test_the_model_is_shown_a_flat_block_schema_without_unions() -> None:
    client, requests = make_client({"format": "pdf", "guide": GUIDE})
    client.post("/api/chat", json={"messages": [user_message("Make this a PDF")]})

    tool = next(t for t in requests[0]["tools"] if t["function"]["name"] == "create_study_document")
    schema = json.dumps(tool["function"]["parameters"])
    assert "oneOf" not in schema and "discriminator" not in schema


def test_stray_fields_on_a_block_do_not_fail_the_document() -> None:
    guide = {
        "title": "Limit Proofs",
        "sections": [
            {
                "heading": "Method",
                "blocks": [{"type": "bullets", "items": ["a", "b"], "text": "", "kind": "Tip"}],
            }
        ],
    }
    client, _ = make_client({"format": "pdf", "guide": guide})
    response = client.post("/api/chat", json={"messages": [user_message("Make this a PDF")]})

    result = tool_events(response.text)[2][1]["result"]
    assert result["outline"] == [{"heading": "Method", "blocks": 1}]
    assert client.get(result["url"]).status_code == 200


def test_a_block_missing_its_fields_gets_feedback_naming_the_path() -> None:
    guide = {"title": "T", "sections": [{"heading": "H", "blocks": [{"type": "key_terms"}]}]}
    client, requests = make_client({"format": "pdf", "guide": guide})
    response = client.post("/api/chat", json={"messages": [user_message("Make this a PDF")]})

    assert tool_events(response.text)[2][1]["isError"] is True
    feedback = requests[1]["messages"][-1]["content"]
    assert "sections.0.blocks.0.key_terms.terms" in feedback


def test_object_items_sent_by_the_model_become_plain_bullets() -> None:
    items = [{"text": "one"}, {"term": "ε", "definition": "tolerance"}]
    guide = {
        "title": "T",
        "sections": [{"heading": "H", "blocks": [{"type": "bullets", "items": items}]}],
    }
    client, _ = make_client({"format": "pdf", "guide": guide})
    response = client.post("/api/chat", json={"messages": [user_message("Make this a PDF")]})

    result = tool_events(response.text)[2][1]["result"]
    assert result["outline"] == [{"heading": "H", "blocks": 1}]
    assert client.get(result["url"]).status_code == 200
