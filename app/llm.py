"""OpenRouter client (OpenAI-compatible SDK)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

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


def _assistant_dict_from_accumulated(
    content: str | None, tool_calls_by_index: dict[int, dict[str, str]]
) -> dict:
    """Build the assistant history message from streamed tool-call fragments."""
    payload: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls_by_index:
        payload["tool_calls"] = [
            {
                "id": tool_calls_by_index[i]["id"],
                "type": "function",
                "function": {
                    "name": tool_calls_by_index[i]["name"],
                    "arguments": tool_calls_by_index[i]["arguments"],
                },
            }
            for i in sorted(tool_calls_by_index)
        ]
    return payload


def _accumulate_tool_call_delta(
    bucket: dict[int, dict[str, str]], tool_call_delta: Any
) -> None:
    """Merge one streamed tool_call fragment into the accumulator by index.

    Streaming splits a single tool call across many chunks: early chunks carry
    id/name; later chunks only append to arguments. Index is the join key.
    """
    index = tool_call_delta.index
    entry = bucket.setdefault(
        index, {"id": "", "name": "", "arguments": ""}
    )
    if tool_call_delta.id:
        entry["id"] = tool_call_delta.id
    if tool_call_delta.function:
        if tool_call_delta.function.name:
            entry["name"] = tool_call_delta.function.name
        if tool_call_delta.function.arguments:
            entry["arguments"] += tool_call_delta.function.arguments


def chat_completion(
    message: str, settings: Settings | None = None
) -> tuple[str, str, list[str]]:
    """Chat with optional tool use (non-streaming). Kept for comparison/tests."""
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


def stream_chat(
    message: str, settings: Settings | None = None
) -> Iterator[dict[str, Any]]:
    """Stream chat over SSE-friendly events, including a tool-call round trip.

    Yields dicts: {"event": <name>, "data": <json-serializable>}.

    Events:
      - status: phase changes (calling_model, executing_tools, ...)
      - tool: a fully assembled tool call we are about to execute
      - delta: a text fragment of the final answer
      - done: model, tools_used, full reply text
    """
    settings = settings or get_settings()
    client = get_openrouter_client(settings)
    messages: list[dict] = [{"role": "user", "content": message}]
    tools_used: list[str] = []
    model = settings.openrouter_model

    yield {"event": "status", "data": {"phase": "calling_model"}}

    stream = client.chat.completions.create(
        model=settings.openrouter_model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        stream=True,
    )

    tool_calls_by_index: dict[int, dict[str, str]] = {}
    assistant_text_parts: list[str] = []
    finish_reason: str | None = None

    for chunk in stream:
        if chunk.model:
            model = chunk.model
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        if delta.content:
            # Rare on the tool-calling first turn; still forward if present.
            assistant_text_parts.append(delta.content)
            yield {"event": "delta", "data": {"text": delta.content}}
        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                _accumulate_tool_call_delta(tool_calls_by_index, tc_delta)
        if choice.finish_reason:
            finish_reason = choice.finish_reason

    if tool_calls_by_index and finish_reason in (None, "tool_calls", "stop"):
        # Assemble the same history the non-streaming path builds, then stream
        # the final natural-language answer.
        assistant_content = "".join(assistant_text_parts) or None
        messages.append(
            _assistant_dict_from_accumulated(assistant_content, tool_calls_by_index)
        )

        yield {"event": "status", "data": {"phase": "executing_tools"}}
        for index in sorted(tool_calls_by_index):
            call = tool_calls_by_index[index]
            name = call["name"]
            tools_used.append(name)
            yield {
                "event": "tool",
                "data": {
                    "name": name,
                    "arguments": call["arguments"],
                    "id": call["id"],
                },
            }
            result = execute_tool(name, call["arguments"])
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result,
                }
            )

        yield {"event": "status", "data": {"phase": "calling_model"}}
        follow_up = client.chat.completions.create(
            model=settings.openrouter_model,
            messages=messages,
            tools=TOOLS,
            stream=True,
        )
        reply_parts: list[str] = []
        for chunk in follow_up:
            if chunk.model:
                model = chunk.model
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                reply_parts.append(delta.content)
                yield {"event": "delta", "data": {"text": delta.content}}
        reply = "".join(reply_parts)
    else:
        reply = "".join(assistant_text_parts)

    yield {
        "event": "done",
        "data": {
            "reply": reply,
            "model": model,
            "tools_used": tools_used,
        },
    }
