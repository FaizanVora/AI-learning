"""Chat routes."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.llm import stream_chat
from app.schemas import ChatRequest

router = APIRouter(tags=["chat"])


@router.post("/chat")
def chat(body: ChatRequest) -> EventSourceResponse:
    """Stream the LLM reply as Server-Sent Events (may include a tool round-trip).

    Event types: status | tool | delta | done
    """
    settings = get_settings()

    def event_publisher():
        try:
            for item in stream_chat(body.message, settings):
                yield {
                    "event": item["event"],
                    "data": json.dumps(item["data"]),
                }
        except RuntimeError as exc:
            # Pre-stream failures can still become HTTP errors if nothing was sent;
            # once streaming starts, status is locked (handled properly in step 6).
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Upstream LLM error: {exc}",
            ) from exc

    return EventSourceResponse(event_publisher())
