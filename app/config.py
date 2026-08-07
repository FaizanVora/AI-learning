"""Application settings loaded from environment / .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Mirrors Django settings / Express dotenv: values come from the environment,
    not from hardcoded constants in source.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = "openrouter"
    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-4o-mini"
    app_name: str = "ai-learning"
    app_env: str = "development"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (safe to call per-request)."""
    return Settings()
