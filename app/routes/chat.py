"""Chat routes."""

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.llm import chat_completion
from app.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    """Send one user message to OpenRouter; return the full reply (no streaming yet)."""
    settings = get_settings()
    try:
        reply, model = chat_completion(body.message, settings)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        # Broad catch for now — step 6 will map rate limits / timeouts properly.
        raise HTTPException(
            status_code=502,
            detail=f"Upstream LLM error: {exc}",
        ) from exc

    return ChatResponse(reply=reply, model=model)
