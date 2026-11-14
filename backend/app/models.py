"""Model adapter construction kept separate from the LangGraph workflow."""

import logging

# The OpenAI SDK's error type covers failed metadata requests from the reused client.
import openai

# BaseChatModel is the graph's provider-neutral interface; ChatOpenAI speaks the compatible API.
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config import Settings

logger = logging.getLogger(__name__)


def create_chat_model(settings: Settings) -> BaseChatModel:
    """Build a standard Chat Completions client for an OpenAI-compatible URL."""
    if not settings.model_configured:
        raise ValueError("Set LLM_BASE_URL and LLM_MODEL before starting a chat.")

    # SecretStr reveals the key only at the point where the HTTP client needs it.
    api_key = settings.llm_api_key.get_secret_value().strip() if settings.llm_api_key else ""
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        # An empty callable lets the OpenAI client omit auth without a placeholder token.
        api_key=api_key or (lambda: ""),
        # Use the widely supported Chat Completions endpoint; the graph requests usage per call.
        use_responses_api=False,
        stream_usage=False,
        # Streaming is what lets the browser render text before the full answer is complete.
        streaming=True,
    )


async def discover_context_window(model: BaseChatModel) -> int | None:
    """Read the served model's current context size from vLLM's /v1/models metadata.

    vLLM reports the resolved max_model_len, including one chosen by --max-model-len auto.
    Other endpoints may omit it, so None means "unknown"; callers must not guess a size.
    """
    if not isinstance(model, ChatOpenAI):
        return None
    try:
        # Reuse the chat model's client, so the base URL, credentials, and redirects match.
        page = await model.root_async_client.models.list()
    except openai.OpenAIError as error:
        logger.warning("Could not read model metadata for context size: %s", error)
        return None
    for card in page.data:
        if card.id == model.model_name:
            context_window = getattr(card, "max_model_len", None)
            if type(context_window) is int and context_window > 0:
                return context_window
            break
    logger.warning("The endpoint does not report max_model_len for %s", model.model_name)
    return None
