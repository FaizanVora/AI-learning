"""FastAPI application entrypoint."""

from fastapi import FastAPI

from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Minimal FastAPI service for learning OpenRouter LLM patterns.",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe — no LLM calls, just prove the process is up."""
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "env": settings.app_env,
    }


def main() -> None:
    """Run with: uv run uvicorn app.main:app --reload"""
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
