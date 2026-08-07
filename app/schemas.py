"""Request/response schemas for the chat API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming user message for /chat."""

    message: str = Field(..., min_length=1, description="User query to send to the LLM")


class ChatResponse(BaseModel):
    """Non-streaming chat reply (plain text from the model)."""

    reply: str
    model: str
