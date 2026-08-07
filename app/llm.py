"""OpenRouter client (OpenAI-compatible SDK)."""

from openai import OpenAI

from app.config import Settings, get_settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_openrouter_client(settings: Settings | None = None) -> OpenAI:
    """Build an OpenAI SDK client aimed at OpenRouter.

    Same pattern as pointing axios/fetch at a different baseURL in Express —
    the request shape stays OpenAI chat.completions; only the host and key change.
    """
    settings = settings or get_settings()
    if not settings.openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is empty. Set it in .env before calling the LLM."
        )
    return OpenAI(
        api_key=settings.openrouter_api_key,
        base_url=OPENROUTER_BASE_URL,
    )


def chat_completion(message: str, settings: Settings | None = None) -> tuple[str, str]:
    """Single-turn, non-streaming chat completion.

    Returns (assistant_text, model_id).
    """
    settings = settings or get_settings()
    client = get_openrouter_client(settings)
    completion = client.chat.completions.create(
        model=settings.openrouter_model,
        messages=[{"role": "user", "content": message}],
    )
    content = completion.choices[0].message.content or ""
    model = completion.model or settings.openrouter_model
    return content, model
