"""OpenRouter client (OpenAI-compatible SDK)."""

from __future__ import annotations

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage

from app.config import Settings, get_settings
from app.tools import TOOLS, execute_tool

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


def _assistant_message_to_dict(message: ChatCompletionMessage) -> dict:
    """Serialize the assistant turn (including tool_calls) for the next request."""
    payload: dict = {
        "role": "assistant",
        "content": message.content,
    }
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]
    return payload


def chat_completion(
    message: str, settings: Settings | None = None
) -> tuple[str, str, list[str]]:
    """Chat with optional tool use (non-streaming).

    Flow:
      1) Send user message + tool definitions.
      2) If the model returns tool_calls, execute them locally.
      3) Append assistant + tool result messages, call the model again for the final answer.

    Returns (assistant_text, model_id, tools_used).
    """
    settings = settings or get_settings()
    client = get_openrouter_client(settings)
    messages: list[dict] = [{"role": "user", "content": message}]
    tools_used: list[str] = []

    completion = client.chat.completions.create(
        model=settings.openrouter_model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )
    assistant = completion.choices[0].message
    model = completion.model or settings.openrouter_model

    if not assistant.tool_calls:
        return assistant.content or "", model, tools_used

    # Turn 2: model asked for tools — we run them, then ask again.
    messages.append(_assistant_message_to_dict(assistant))
    for tool_call in assistant.tool_calls:
        name = tool_call.function.name
        tools_used.append(name)
        result = execute_tool(name, tool_call.function.arguments)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
        )

    follow_up = client.chat.completions.create(
        model=settings.openrouter_model,
        messages=messages,
        tools=TOOLS,
    )
    final = follow_up.choices[0].message
    model = follow_up.model or model
    return final.content or "", model, tools_used
