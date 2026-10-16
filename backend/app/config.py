"""Backend-only settings for an OpenAI-compatible chat endpoint."""

# Pydantic reads and validates environment settings; SecretStr masks credentials in reprs.
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Read model connection details from the environment or repository .env file."""

    # Keeping this file backend-only prevents Vite from exposing credentials to the browser.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_base_url: str | None = None  # Chat Completions API base URL.
    llm_model: str | None = None  # Provider's model or deployment name.
    llm_api_key: SecretStr | None = None  # Optional when the endpoint needs no auth.

    @property
    def model_configured(self) -> bool:
        """A URL and model are required before the backend can create a client."""
        return bool(self.llm_base_url and self.llm_model)
