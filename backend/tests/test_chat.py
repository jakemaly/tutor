"""End-to-end checks for the FastAPI, LangGraph, and assistant-ui stream bridge."""

# asyncio drives async graph calls; json decodes the line-based streaming protocol in assertions.
import asyncio
import json

# pytest and httpx test routes and emulate the model endpoint without a network.
import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel as BaseFakeChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr

from app.config import Settings
from app.graph import SYSTEM_PROMPT, create_graph
from app.main import ChatRequest, _messages_for_graph, create_app
from app.models import create_chat_model


class FakeListChatModel(BaseFakeChatModel):
    """Text-only fixture that accepts tool binding; tool calls use HTTP fixtures below."""

    def bind_tools(self, tools, **kwargs):
        return self


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


def test_skill_json_is_literal_context_not_a_prompt_template(monkeypatch) -> None:
    prompt = SYSTEM_PROMPT + '\nExample: {"step": "explain"}'
    monkeypatch.setattr("app.graph.SYSTEM_PROMPT", prompt)
    model = RecordingFakeChatModel(responses=["A hint"])
    graph = create_graph(model)
    asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="Help me")]}))
    assert model._seen_messages[0][0].content == prompt


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
    assert sent_messages[0]["role"] == "system"
    assert '<loaded_skill name="socratic-questioning">' in sent_messages[0]["content"]
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


@pytest.mark.parametrize("skill_name", ["worked-example", "../config.py"])
def test_streamed_tool_call_loads_skill_and_resumes_model(skill_name: str) -> None:
    """Exercise chunked tool arguments, real tool execution, and the follow-up API request."""
    from app.skills import discover_skills

    requests = []

    def event(delta, finish_reason=None):
        return (
            "data: "
            + json.dumps(
                {
                    "id": "test",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "test-model",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                }
            )
            + "\n\n"
        )

    def respond(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            arguments = json.dumps({"name": skill_name})
            events = [
                event(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "load-1",
                                "type": "function",
                                "function": {"name": "load_skill", "arguments": arguments[:8]},
                            }
                        ],
                    }
                ),
                event(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": arguments[8:]},
                            }
                        ]
                    }
                ),
                event({}, "tool_calls"),
            ]
        else:
            events = [event({"content": "The answer is x = 4."}), event({}, "stop")]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=StaticEventStream([*events, "data: [DONE]\n\n"]),
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    model = ChatOpenAI(
        model="test-model",
        base_url="https://example.invalid/v1",
        api_key="test-key",
        use_responses_api=False,
        stream_usage=False,
        streaming=True,
        http_async_client=http_client,
    )
    client = TestClient(create_app(settings=configured_settings(), model=model))
    response = client.post(
        "/api/chat",
        json={
            "messages": [
                user_message(
                    "Give me the complete worked solution to 2x + 3 = 11, including a check."
                )
            ]
        },
    )

    assert len(requests) == 2
    assert requests[0]["tools"][0]["function"]["name"] == "load_skill"
    body = discover_skills()["worked-example"].instructions
    assert body not in requests[0]["messages"][0]["content"]
    messages = requests[1]["messages"]
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "tool"]
    assert messages[-1]["tool_call_id"] == "load-1"
    if skill_name == "worked-example":
        assert body in messages[-1]["content"]
    else:
        assert "Unknown skill" in messages[-1]["content"]
        assert "Available skills" in messages[-1]["content"]
    deltas = [json.loads(line[2:]) for line in response.text.splitlines() if line.startswith("0:")]
    assert "".join(deltas) == "The answer is x = 4."
    assert body not in response.text

    events = [
        (line[0], json.loads(line[2:]))
        for line in response.text.splitlines()
        if line.startswith(("b:", "c:", "a:"))
    ]
    assert [kind for kind, _ in events] == ["b", "c", "a"]
    assert all(event["toolCallId"] == "load-1" for _, event in events)
    assert events[0][1]["toolName"] == "load_skill"
    assert json.loads(events[1][1]["argsTextDelta"]) == {"name": skill_name}
    assert events[2][1]["result"] == {"loaded": skill_name == "worked-example"}
    assert events[2][1].get("isError", False) == (skill_name != "worked-example")

    # The runtime serializes traces into history; keep the next turn text-only for the model.
    followup = client.post(
        "/api/chat",
        json={
            "messages": [
                user_message("Solve the equation"),
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool-call",
                            "toolCallId": "load-1",
                            "toolName": "load_skill",
                            "input": {"name": skill_name},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool-result",
                            "toolCallId": "load-1",
                            "toolName": "load_skill",
                            "output": {"type": "json", "value": {"loaded": True}},
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "The answer is x = 4."}],
                },
                user_message("Why?"),
            ]
        },
    )
    assert followup.status_code == 200
    assert [message["role"] for message in requests[-1]["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    asyncio.run(http_client.aclose())


@pytest.mark.parametrize(
    ("finish_reason", "total_tokens", "models_response", "expected_chats", "expected_lookups"),
    [
        # The size comes from the server, so the same usage continues or stops by its value.
        pytest.param("length", 2000, {"max_model_len": 4096}, 2, 1, id="continues"),
        pytest.param("length", 4096, {"max_model_len": 4096}, 1, 1, id="context-exhausted"),
        pytest.param("length", 4096, {"max_model_len": 8192}, 2, 1, id="larger-discovered"),
        pytest.param("stop", 2000, {"max_model_len": 4096}, 1, 0, id="natural-stop"),
        pytest.param("length", 2000, {}, 1, 1, id="no-metadata"),
        pytest.param(
            "length",
            2000,
            {"id": "other-model", "max_model_len": 4096},
            1,
            1,
            id="other-model-only",
        ),
        pytest.param("length", 2000, None, 1, 1, id="models-endpoint-fails"),
    ],
)
def test_length_limited_answer_continues_only_with_discovered_context(
    finish_reason, total_tokens, models_response, expected_chats, expected_lookups
) -> None:
    """Resume truncated answers only while the server-reported context has room."""
    chats = []
    lookups = []

    def event(delta, reason=None):
        payload = {
            "id": "continuation-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    def respond(request):
        if request.method == "GET" and request.url.path == "/v1/models":
            lookups.append(request)
            if models_response is None:
                return httpx.Response(404, json={"error": {"message": "not found"}})
            card = {"id": "test-model", "object": "model", **models_response}
            return httpx.Response(200, json={"object": "list", "data": [card]})

        chats.append(json.loads(request.content))
        first_request = len(chats) == 1
        events = [
            event({"role": "assistant"}),
            event({"content": "The answer is " if first_request else "ten."}),
            event({}, finish_reason if first_request else "stop"),
        ]
        usage = {
            "id": "continuation-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": [],
            "usage": {
                "prompt_tokens": total_tokens - 100,
                "completion_tokens": 100,
                "total_tokens": total_tokens,
            },
        }
        events.append(f"data: {json.dumps(usage)}\n\n")
        events.append("data: [DONE]\n\n")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=StaticEventStream(events),
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    model = ChatOpenAI(
        model="test-model",
        base_url="https://example.invalid/v1",
        api_key="test-key",
        use_responses_api=False,
        stream_usage=False,
        streaming=True,
        max_retries=0,
        http_async_client=http_client,
    )
    client = TestClient(create_app(settings=configured_settings(), model=model))

    try:
        response = client.post(
            "/api/chat",
            json={"messages": [user_message("What is five plus five?")]},
        )
    finally:
        asyncio.run(http_client.aclose())

    assert response.status_code == 200
    assert "model request failed" not in response.text
    assert len(chats) == expected_chats
    assert len(lookups) == expected_lookups
    # Discovery reuses the chat model's credentials.
    assert all(lookup.headers["authorization"] == "Bearer test-key" for lookup in lookups)

    text = "".join(
        json.loads(line[2:]) for line in response.text.splitlines() if line.startswith("0:")
    )
    # Both pieces stream into one browser response, without repeating the prefix.
    expected_text = "The answer is ten." if expected_chats == 2 else "The answer is "
    assert text == expected_text
    for chat in chats:
        assert "max_tokens" not in chat
        assert "max_completion_tokens" not in chat
    assert chats[0]["stream_options"] == {"include_usage": True}

    if expected_chats == 2:
        assert chats[1]["continue_final_message"] is True
        assert chats[1]["add_generation_prompt"] is False
        assert chats[1]["tool_choice"] == "none"
        assert chats[1]["messages"][-1] == {
            "role": "assistant",
            "content": "The answer is ",
        }
