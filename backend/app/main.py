"""FastAPI bridge between assistant-ui's data stream and LangGraph."""

# asyncio preserves cancellation so a browser Stop can reach the model request.
import asyncio
import json
import logging
import os
from typing import Any

# These libraries adapt the browser stream, define HTTP routes, and represent model messages.
from assistant_stream import RunController, create_run
from assistant_stream.serialization import DataStreamResponse
from fastapi import FastAPI, HTTPException, Response
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from pydantic import BaseModel, ConfigDict

# Local modules separate configuration, graph behavior, and endpoint-specific model setup.
from app.artifacts import ArtifactBackend, ArtifactStore
from app.config import Settings
from app.graph import create_graph, is_tool_call_rejection
from app.models import create_chat_model
from app.study_document import TOOL_NAME as STUDY_DOCUMENT_TOOL

logger = logging.getLogger(__name__)

# Display-only tool traces: shown in the browser, never sent back to the model as history.
DISPLAY_ONLY_TOOLS = {"load_skill", STUDY_DOCUMENT_TOOL}

MODEL_FAILURE_MESSAGE = (
    "The model request failed. Check LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, and the backend logs."
)
TOOL_REJECTION_MESSAGE = (
    "The model could not produce a valid document. Try again, or ask for something shorter."
)


class ChatRequest(BaseModel):
    """The text conversation sent by assistant-ui's data-stream runtime."""

    model_config = ConfigDict(extra="ignore")

    # Keep the incoming assistant-ui shape here; validate and convert each part below.
    messages: list[dict[str, Any]]


def _message_text(message: dict[str, Any]) -> str:
    """Read text from the assistant-ui / AI SDK generic message representation."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Tool traces are display metadata, not model history or executable client tools.
        if message.get("role") == "assistant":
            content = [
                part
                for part in content
                if not (
                    isinstance(part, dict)
                    and part.get("type") == "tool-call"
                    and part.get("toolName") in DISPLAY_ONLY_TOOLS
                )
            ]
        if any(
            not isinstance(part, dict)
            or part.get("type") != "text"
            or not isinstance(part.get("text"), str)
            for part in content
        ):
            raise HTTPException(status_code=422, detail="Only text messages are supported.")
        return "".join(part["text"] for part in content)
    raise HTTPException(status_code=422, detail="Only text messages are supported.")


def _messages_for_graph(messages: list[dict[str, Any]]) -> list[BaseMessage]:
    """Convert frontend messages to LangChain messages, dropping an empty run stub."""
    relevant = list(messages)
    if relevant and relevant[-1].get("role") == "assistant":
        # assistant-ui appends an empty assistant message before sending a new user turn.
        if not _message_text(relevant[-1]).strip():
            relevant.pop()

    result: list[BaseMessage] = []
    for message in relevant:
        role = message.get("role")
        content = message.get("content")
        if (
            role == "tool"
            and isinstance(content, list)
            and content
            and all(
                isinstance(part, dict)
                and part.get("type") == "tool-result"
                and part.get("toolName") in DISPLAY_ONLY_TOOLS
                for part in content
            )
        ):
            continue
        text = _message_text(message)
        if role == "user":
            if not text.strip():
                raise HTTPException(status_code=422, detail="User messages must contain text.")
            result.append(HumanMessage(content=text))
        elif role == "assistant" and text.strip():
            result.append(AIMessage(content=text))
        elif role != "assistant":
            raise HTTPException(
                status_code=422, detail="Only user and assistant messages are supported."
            )
    return result


def _text_from_chunk(chunk: AIMessageChunk) -> str:
    """Extract only text from a LangChain delta, ignoring non-text metadata parts."""
    content = chunk.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part["text"]
            for part in content
            if isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        )
    return ""


def _finish_tool(name: str, tool: Any, artifact: Any, failed: bool) -> None:
    """Send a display-only tool's final state: artifact metadata or load status, never bytes."""
    if name == STUDY_DOCUMENT_TOOL:
        # Artifact metadata only: id, URL, title, outline. Never file bytes.
        usable = not failed and isinstance(artifact, dict)
        result = artifact if usable else {"error": "The document could not be created."}
        tool.set_response(result, is_error=not usable)
    else:
        tool.set_response({"loaded": not failed}, is_error=failed)


async def _stream_graph(graph: Any, messages: list[BaseMessage], controller: RunController) -> None:
    """Forward answer text and tool status, keeping skill text and file bytes server-side."""
    calls: dict[str, tuple[str, Any]] = {}
    try:
        async for mode, payload in graph.astream(
            {"messages": messages}, stream_mode=["messages", "updates"]
        ):
            if controller.is_cancelled:
                break
            if mode == "messages":
                chunk, _metadata = payload
                if isinstance(chunk, AIMessageChunk):
                    text = _text_from_chunk(chunk)
                    if text:
                        controller.append_text(text)
            elif mode == "updates":
                for update in payload.values():
                    for message in update.get("messages", []):
                        if isinstance(message, AIMessage):
                            for call in message.tool_calls:
                                if call["name"] in DISPLAY_ONLY_TOOLS:
                                    tool = await controller.add_tool_call(call["name"], call["id"])
                                    # The document body is large and not needed by the browser.
                                    args = (
                                        {"format": call["args"].get("format")}
                                        if call["name"] == STUDY_DOCUMENT_TOOL
                                        else call["args"]
                                    )
                                    tool.append_args_text(json.dumps(args))
                                    calls[call["id"]] = (call["name"], tool)
                        elif isinstance(message, ToolMessage) and message.tool_call_id in calls:
                            name, tool = calls.pop(message.tool_call_id)
                            _finish_tool(name, tool, message.artifact, message.status == "error")
    finally:
        # A failed or cancelled run must not leave a tool card spinning forever.
        for name, tool in calls.values():
            try:
                _finish_tool(name, tool, None, failed=True)
            except Exception:
                logger.exception("Could not close tool call %s", name)


def create_app(
    settings: Settings | None = None,
    model: BaseChatModel | None = None,
    artifacts: ArtifactBackend | None = None,
) -> FastAPI:
    """Create an app, allowing tests and other integrations to inject a chat model."""
    app_settings = settings or Settings()
    if artifacts is None:
        artifacts = ArtifactStore()
        if os.environ.get("VERCEL"):
            logger.warning(
                "Generated documents are kept in this instance's memory, but this deployment "
                "may run several instances, so downloads can 404. Pass a shared store to "
                "create_app(artifacts=...); see docs/study-documents.md."
            )
    # Tests can inject a fake model; normal startup waits until the first configured chat request.
    graph = create_graph(model, artifacts) if model is not None else None

    app = FastAPI(title="Tutor API", version="0.1.0")
    app.state.settings = app_settings
    app.state.graph = graph
    app.state.artifacts = artifacts

    @app.get("/api/health")
    async def health() -> dict[str, str | bool]:
        """Report service readiness without returning model URLs, names, or credentials."""
        return {
            "status": "ok",
            "model_configured": app_settings.model_configured or model is not None,
        }

    @app.get("/api/artifacts/{artifact_id}")
    async def artifact(artifact_id: str, inline: bool = False) -> Response:
        """Serve a generated file by its unguessable id until it expires."""
        found = artifacts.get(artifact_id)
        if found is None:
            raise HTTPException(status_code=404, detail="This file has expired. Ask for it again.")
        # Only PDFs are safe to display in the page; Word files always download.
        disposition = "inline" if inline and found.media_type == "application/pdf" else "attachment"
        return Response(
            found.data,
            media_type=found.media_type,
            headers={
                "Content-Disposition": f'{disposition}; filename="{found.filename}"',
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.post("/api/chat")
    async def chat(request: ChatRequest) -> DataStreamResponse:
        """Validate supplied history, run the graph, and return assistant-ui stream events."""
        selected_graph = app.state.graph
        if selected_graph is None:
            if not app_settings.model_configured:
                raise HTTPException(
                    status_code=503,
                    detail="Configure LLM_BASE_URL and LLM_MODEL in .env, then restart the app.",
                )
            # Build lazily so health/UI still work without credentials, then reuse the graph.
            selected_graph = create_graph(create_chat_model(app_settings), artifacts)
            app.state.graph = selected_graph

        input_messages = _messages_for_graph(request.messages)
        if not input_messages:
            raise HTTPException(status_code=422, detail="Send a non-empty text message to start.")

        async def run(controller: RunController) -> None:
            """Forward graph text to the stream and translate failures into safe UI errors."""
            try:
                await _stream_graph(selected_graph, input_messages, controller)
            except asyncio.CancelledError:
                # Cancellation is control flow, not a model error to display to the user.
                raise
            except Exception as error:
                # Keep provider details in server logs; the browser receives an actionable message.
                logger.exception("Chat model request failed")
                controller.add_error(
                    TOOL_REJECTION_MESSAGE
                    if is_tool_call_rejection(error)
                    else MODEL_FAILURE_MESSAGE
                )

        return DataStreamResponse(create_run(run))

    return app


# Uvicorn imports this module-level ASGI app when started by the root dev command.
app = create_app()
