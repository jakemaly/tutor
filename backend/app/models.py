"""Model adapter construction kept separate from the LangGraph workflow."""

# BaseChatModel is the graph's provider-neutral interface; ChatOpenAI speaks the compatible API.
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config import Settings


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
        # Use the widely supported Chat Completions endpoint and avoid extra usage calls.
        use_responses_api=False,
        stream_usage=False,
        # Streaming is what lets the browser render text before the full answer is complete.
        streaming=True,
    )
